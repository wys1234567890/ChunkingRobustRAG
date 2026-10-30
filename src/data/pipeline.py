"""Deterministic corpus construction with disk-backed staging and atomic publish."""

from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
import subprocess
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from .adapters import adapt_hotpotqa, adapt_nq
from .common import DataError, answer_key, canonical_json, contains_answer, digest, file_sha256, normalize_inline, rank, write_json, write_jsonl
from .config import DataSettings
from .sources import acquire_sources, iter_source
from .validation import DATA_FILES, validate_dataset, validate_records


PREPARER_VERSION = "data-preparation-v1"


@contextmanager
def _staging_directory(parent: Path, dataset: str):
    """Create a staging directory with inherited workspace ACLs on Windows.

    tempfile.TemporaryDirectory creates owner-only ACLs on Windows. Renaming
    that directory into the published tree would make the generated dataset
    unreadable to another workspace user, so create it with normal inherited
    permissions instead.
    """
    staging = parent / f".{dataset}-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o777)
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _code_identity(settings: DataSettings) -> dict:
    files = {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=settings.root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"preparer_version": PREPARER_VERSION, "source_sha256": digest(files), "git_revision": revision}


def allocate_strata(counts: dict[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if population < total:
        raise DataError(f"insufficient eligible queries: need {total}, found {population}")
    allocations = {key: counts[key] * total // population for key in sorted(counts)}
    remainder_order = sorted(counts, key=lambda key: (-(counts[key] * total % population), key))
    for key in remainder_order[:total - sum(allocations.values())]:
        allocations[key] += 1
    return allocations


def _summary(values: list[int]) -> dict:
    values = sorted(values)
    return {"min": values[0], "max": values[-1], "mean": sum(values) / len(values), "p50": values[(len(values) - 1) // 2], "p95": values[math.ceil(len(values) * .95) - 1]}


def _statistics(documents: list[dict], queries: list[dict], evidence: list[dict]) -> dict:
    evidence_counts = Counter(item["query_id"] for item in evidence)
    hard_negative_counts = [len(item.get("hard_negative_doc_ids", [])) for item in queries]
    related_counts = [len(item.get("related_doc_ids", [])) for item in queries]
    return {
        "document_chars": _summary([len(item["text"]) for item in documents]),
        "document_words": _summary([len(item["text"].split()) for item in documents]),
        "answer_words": _summary([max(len(answer.split()) for answer in item["ground_truth"]) for item in queries]),
        "evidence_per_query": _summary([evidence_counts[item["query_id"]] for item in queries]),
        "hard_negatives_per_query": _summary(hard_negative_counts),
        "related_documents_per_query": _summary(related_counts),
        "document_roles": dict(sorted(Counter(item.get("document_role", "background") for item in documents).items())),
        "question_types": dict(sorted(Counter(item["question_type"] for item in queries).items())),
    }


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_inline(text).casefold())


def _selection_tokens(document: dict) -> list[str]:
    # Candidate ranking does not need every token in a long Wikipedia page.
    # Keeping the title and a bounded prefix makes the full-dev-set scan
    # practical while the published document remains unchanged.
    return _tokens(f"{document['title']} {document['text'][:4096]}")


def _bm25_candidate_scores(query_tokens: list[str], corpus_tokens: list[list[str]]) -> dict[int, float]:
    if not query_tokens or not corpus_tokens:
        return {}
    k1, b = 1.5, 0.75
    n = len(corpus_tokens)
    average_length = sum(len(item) for item in corpus_tokens) / n
    query_terms = Counter(query_tokens)
    postings: dict[str, set[int]] = defaultdict(set)
    for index, tokens in enumerate(corpus_tokens):
        for term in set(tokens) & set(query_terms):
            postings[term].add(index)
    document_frequency = {term: len(postings[term]) for term in query_terms}
    candidate_indexes = set().union(*(postings[term] for term in query_terms if postings[term]))
    scores: dict[int, float] = {}
    for index in candidate_indexes:
        frequencies = Counter(corpus_tokens[index])
        length = len(corpus_tokens[index])
        score = 0.0
        for term, qtf in query_terms.items():
            df = document_frequency[term]
            tf = frequencies[term]
            if not tf:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1.0 - b + b * length / average_length)
            score += idf * (tf * (k1 + 1.0) / denominator)
        scores[index] = score
    return scores


def _near_duplicate(text: str, evidence_texts: list[str]) -> bool:
    candidate = " ".join(_tokens(text))
    if not candidate:
        return True
    for evidence in evidence_texts:
        reference = " ".join(_tokens(evidence))
        if candidate == reference:
            return True
        if min(len(candidate), len(reference)) >= 40 and SequenceMatcher(None, candidate, reference).ratio() >= 0.95:
            return True
    return False


class _StagingStore:
    """Keep full-page corpora off the Python heap during the full dev-set scan."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT, rank TEXT, payload TEXT);
            CREATE TABLE queries (id TEXT PRIMARY KEY, question_key TEXT, rank TEXT, stratum TEXT, payload TEXT);
        """)

    def add_document(self, document: dict, ordering: str) -> bool:
        old = self.connection.execute("SELECT payload FROM documents WHERE id=?", (document["doc_id"],)).fetchone()
        if old is not None:
            previous = json.loads(old[0])
            refs = {canonical_json(ref): ref for ref in previous["source_refs"] + document["source_refs"]}
            previous["source_refs"] = [refs[key] for key in sorted(refs)]
            self.connection.execute("UPDATE documents SET payload=? WHERE id=?", (canonical_json(previous), document["doc_id"]))
            return False
        self.connection.execute("INSERT INTO documents VALUES (?,?,?,?)", (document["doc_id"], document["title"], ordering, canonical_json(document)))
        return True

    def add_query(self, query: dict, evidence: list[dict], ordering: str) -> bool:
        payload = canonical_json({"query": query, "evidence": sorted(evidence, key=lambda item: item["evidence_id"])})
        existing = self.connection.execute("SELECT payload FROM queries WHERE id=?", (query["query_id"],)).fetchone()
        if existing is not None:
            if payload != existing[0]:
                raise DataError(f"conflicting records for source query ID {query['query_id']}")
            return False
        self.connection.execute("INSERT INTO queries VALUES (?,?,?,?,?)", (query["query_id"], answer_key(query["question"]), ordering, query["question_type"], payload))
        return True

    def deduplicate_questions(self) -> int:
        before = self.connection.execute("SELECT count(*) FROM queries").fetchone()[0]
        self.connection.execute("""DELETE FROM queries WHERE id IN (
            SELECT id FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY question_key ORDER BY rank,id) AS n FROM queries) WHERE n>1
        )""")
        return before - self.connection.execute("SELECT count(*) FROM queries").fetchone()[0]

    def choose_queries(self, settings: DataSettings) -> tuple[list[dict], dict]:
        counts = dict(self.connection.execute("SELECT stratum,count(*) FROM queries GROUP BY stratum"))
        allocations = allocate_strata(counts, settings.query_count)
        rows = []
        for stratum, count in sorted(allocations.items()):
            rows.extend(json.loads(row[0]) for row in self.connection.execute("SELECT payload FROM queries WHERE stratum=? ORDER BY rank,id LIMIT ?", (stratum, count)))
        return sorted(rows, key=lambda row: row["query"]["query_id"]), {"eligible_question_types": counts, "allocated_question_types": allocations}

    def choose_documents(self, mandatory: set[str], count: int) -> list[dict]:
        available = self.connection.execute("SELECT count(*) FROM documents").fetchone()[0]
        if available < count:
            raise DataError(f"insufficient unique documents: need {count}, found {available}")
        if len(mandatory) > count:
            raise DataError(f"target contexts exceed document_count: {len(mandatory)} > {count}")
        documents = {}
        for identifier in sorted(mandatory):
            row = self.connection.execute("SELECT payload FROM documents WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise DataError(f"missing target context: {identifier}")
            documents[identifier] = json.loads(row[0])
        for identifier, payload in self.connection.execute("SELECT id,payload FROM documents ORDER BY rank,id"):
            if len(documents) == count:
                break
            if identifier not in documents:
                documents[identifier] = json.loads(payload)
        return [documents[key] for key in sorted(documents)]

    def all_documents(self) -> list[dict]:
        return [json.loads(payload) for payload, in self.connection.execute("SELECT payload FROM documents ORDER BY id")]

    def choose_hard_negatives(
        self, queries: list[dict], count: int
    ) -> tuple[dict[str, list[str]], dict[str, object]]:
        """Select deterministic lexical near-neighbours from the full source pool.

        All selected-question context documents are excluded globally, so a
        document has one unambiguous role in the published corpus. A clean
        candidate may serve as a hard negative for more than one query.
        """
        if count == 0:
            return {query["query_id"]: [] for query in queries}, {
                "enabled": False, "requested_per_query": 0,
                "candidate_shortfalls": [], "selections": [],
            }
        pool = self.all_documents()
        globally_excluded = {doc_id for query in queries for doc_id in query["context_doc_ids"]}
        candidate_docs = [doc for doc in pool if doc["doc_id"] not in globally_excluded]
        corpus_tokens = [_selection_tokens(doc) for doc in candidate_docs]
        selected: dict[str, list[str]] = {}
        selections: list[dict] = []
        shortfalls: list[dict] = []
        for query in sorted(queries, key=lambda item: item["query_id"]):
            answer_fragments = query["ground_truth"]
            excluded = set(query["context_doc_ids"])
            evidence_texts = [
                doc["text"] for doc in pool if doc["doc_id"] in set(query["evidence_doc_ids"])
            ]
            query_tokens = _tokens(query["question"])
            score_map = _bm25_candidate_scores(query_tokens, corpus_tokens)
            ranked = sorted(
                ((score_map[index], candidate_docs[index]) for index in score_map),
                key=lambda item: (-item[0], item[1]["doc_id"]),
            )
            picked: list[str] = []
            for score, doc in ranked:
                if doc["doc_id"] in excluded:
                    continue
                if score <= 0:
                    continue
                if contains_answer(doc["text"], answer_fragments):
                    continue
                if _near_duplicate(doc["text"], evidence_texts):
                    continue
                if doc["doc_id"] in picked:
                    continue
                picked.append(doc["doc_id"])
                selections.append({
                    "query_id": query["query_id"], "doc_id": doc["doc_id"],
                    "score": round(float(score), 8),
                    "reason": "bm25_query_overlap_without_answer_or_evidence_duplicate",
                })
                if len(picked) == count:
                    break
            if len(picked) != count:
                shortfalls.append({
                    "query_id": query["query_id"],
                    "requested": count, "available": len(picked),
                })
            selected[query["query_id"]] = picked
        if shortfalls:
            details = ", ".join(f"{item['query_id']} ({item['available']}/{item['requested']})" for item in shortfalls)
            raise DataError(f"insufficient hard-negative candidates: {details}")
        return selected, {
            "enabled": True, "requested_per_query": count,
            "candidate_pool_documents": len(candidate_docs),
            "candidate_shortfalls": shortfalls,
            "selections": selections,
        }


def prepare_dataset(
    settings: DataSettings, dataset: str, *, offline: bool = False,
    progress: Callable[[str], None] = lambda message: None,
) -> dict:
    sources = acquire_sources(settings, dataset, offline=offline, progress=progress)
    code = _code_identity(settings)
    fingerprint = digest({"config": settings.contract(dataset), "sources": sources, "preparer_version": code["preparer_version"], "source_sha256": code["source_sha256"]})
    destination = settings.output_dir(dataset)
    if destination.exists():
        report = validate_dataset(destination)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("input_fingerprint") != fingerprint:
            raise DataError(f"{destination}: frozen inputs/configuration/code changed; choose a new data_version")
        return {**report, "reused": True, "output_dir": str(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _staging_directory(destination.parent, dataset) as staging:
        store = _StagingStore(staging / "staging.sqlite")
        counts = Counter()
        rejected = Counter()
        try:
            adapter = adapt_nq if dataset == "nq" else adapt_hotpotqa
            for source in sources:
                progress(f"{dataset}: processing {source['filename']}")
                for raw in iter_source(settings.resolve(source["path"]), dataset):
                    counts["raw_records"] += 1
                    adapted = adapter(raw, source["filename"], settings)
                    counts["empty_documents"] += adapted.empty_documents
                    counts["document_occurrences"] += len(adapted.documents)
                    for doc in adapted.documents:
                        unique = store.add_document(doc, rank(settings.seed, dataset, doc["doc_id"]))
                        counts["unique_documents" if unique else "merged_document_occurrences"] += 1
                    if adapted.query is not None:
                        unique = store.add_query(adapted.query, adapted.evidence, rank(settings.seed, dataset, adapted.query["source_query_id"]))
                        counts["eligible_before_question_dedup" if unique else "duplicate_query_ids"] += 1
                    else:
                        rejected[adapted.rejection or "unknown"] += 1
                    if counts["raw_records"] % 250 == 0:
                        store.connection.commit()
                        progress(f"{dataset}: scanned {counts['raw_records']} records")
            expected = settings.sources[dataset].get("expected_records")
            if expected is not None and counts["raw_records"] != expected:
                raise DataError(f"{dataset}: incomplete source split: expected {expected} records, found {counts['raw_records']}")
            counts["duplicate_questions"] = store.deduplicate_questions()
            selected, sampling = store.choose_queries(settings)
            queries = [row["query"] for row in selected]
            evidence = sorted((item for row in selected for item in row["evidence"]), key=lambda item: item["evidence_id"])
            hard_negative_ids, hard_negative_report = store.choose_hard_negatives(queries, settings.hard_negative_count)
            hard_negative_by_doc: dict[str, set[str]] = {}
            target_queries_by_doc: dict[str, set[str]] = {}
            for query in queries:
                query_id = query["query_id"]
                ids = hard_negative_ids[query_id]
                query["hard_negative_doc_ids"] = sorted(ids)
                query["related_doc_ids"] = sorted(set(query["evidence_doc_ids"]) | set(ids))
                for doc_id in query["evidence_doc_ids"]:
                    target_queries_by_doc.setdefault(doc_id, set()).add(query_id)
                for doc_id in ids:
                    hard_negative_by_doc.setdefault(doc_id, set()).add(query_id)
                    target_queries_by_doc.setdefault(doc_id, set()).add(query_id)
            mandatory = {identifier for query in queries for identifier in query["context_doc_ids"]}
            mandatory.update(hard_negative_by_doc)
            documents = store.choose_documents(mandatory, settings.document_count)
            evidence_doc_ids = {doc_id for query in queries for doc_id in query["evidence_doc_ids"]}
            for doc in documents:
                doc_id = doc["doc_id"]
                if doc_id in evidence_doc_ids:
                    doc["document_role"] = "evidence"
                elif doc_id in hard_negative_by_doc:
                    doc["document_role"] = "hard_negative"
                else:
                    doc["document_role"] = "background"
                doc["target_query_ids"] = sorted(target_queries_by_doc.get(doc_id, set()))
            counts["mandatory_context_documents"] = len(mandatory)
            counts["background_documents_added"] = len(documents) - len(mandatory)
            counts["title_conflict_groups"] = store.connection.execute("SELECT count(*) FROM (SELECT title FROM documents GROUP BY title HAVING count(*)>1)").fetchone()[0]
            conflicts = [{"title": title, "document_count": amount} for title, amount in store.connection.execute("SELECT title,count(*) FROM documents GROUP BY title HAVING count(*)>1 ORDER BY title")]
        finally:
            store.connection.close()
        report = validate_records(documents, queries, evidence)
        audit_queries = sorted(queries, key=lambda item: (rank(settings.seed, dataset + ":audit", item["source_query_id"]), item["query_id"]))[:settings.audit_count]
        audits = [{"query_id": query["query_id"], "query": query, "evidence": [item for item in evidence if item["query_id"] == query["query_id"]], "review_status": "pending"} for query in audit_queries]
        output = staging / "output"
        output.mkdir()
        for filename, records in (("documents.jsonl", documents), ("queries.jsonl", queries), ("evidence.jsonl", evidence), ("audit_samples.jsonl", audits)):
            write_jsonl(output / filename, records)
        manifest = {
            "dataset": dataset, "data_version": settings.data_version, "schema_version": settings.schema_version,
            "status": "validated", "human_review_status": "pending", "config": settings.contract(dataset),
            "input_fingerprint": fingerprint, "code": code, "sources": sources,
            "output_sha256": {filename: file_sha256(output / filename) for filename in DATA_FILES},
            "sampled_query_ids": sorted(query["query_id"] for query in queries),
            "counts": dict(sorted(counts.items())), "rejection_reasons": dict(sorted(rejected.items())),
            "sampling": sampling, "title_conflicts": conflicts, "statistics": _statistics(documents, queries, evidence),
            "hard_negative": hard_negative_report,
            "validation": report,
            "limitations": ["Short answers only; a controlled-size development-set corpus, not a full-Wikipedia benchmark.", "NQ documents are full pages; HotpotQA documents are supplied paragraphs.", "NQ requires two agreeing single-span annotations; HotpotQA excludes yes/no answers."],
        }
        write_json(output / "manifest.json", manifest)
        validate_dataset(output)
        output.rename(destination)
    return {**report, "dataset": dataset, "data_version": settings.data_version, "status": "validated", "human_review_status": "pending", "reused": False, "output_dir": str(destination)}
