"""Deterministic corpus construction with disk-backed staging and atomic publish."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

from .adapters import adapt_hotpotqa, adapt_nq
from .common import DataError, answer_key, canonical_json, digest, file_sha256, rank, write_json, write_jsonl
from .config import DataSettings
from .sources import acquire_sources, iter_source
from .validation import DATA_FILES, validate_dataset, validate_records


PREPARER_VERSION = "data-preparation-v1"


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
    return {
        "document_chars": _summary([len(item["text"]) for item in documents]),
        "document_words": _summary([len(item["text"].split()) for item in documents]),
        "answer_words": _summary([max(len(answer.split()) for answer in item["ground_truth"]) for item in queries]),
        "evidence_per_query": _summary([evidence_counts[item["query_id"]] for item in queries]),
        "question_types": dict(sorted(Counter(item["question_type"] for item in queries).items())),
    }


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
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
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
            mandatory = {identifier for query in queries for identifier in query["context_doc_ids"]}
            documents = store.choose_documents(mandatory, settings.document_count)
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
            "validation": report,
            "limitations": ["Short answers only; a controlled-size development-set corpus, not a full-Wikipedia benchmark.", "NQ documents are full pages; HotpotQA documents are supplied paragraphs.", "NQ requires two agreeing single-span annotations; HotpotQA excludes yes/no answers."],
        }
        write_json(output / "manifest.json", manifest)
        validate_dataset(output)
        output.rename(destination)
    return {**report, "dataset": dataset, "data_version": settings.data_version, "status": "validated", "human_review_status": "pending", "reused": False, "output_dir": str(destination)}
