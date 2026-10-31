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
from typing import Callable, Iterable

from .adapters import adapt_hotpotqa, adapt_nq
from .common import DataError, answer_key, canonical_json, contains_answer, digest, file_sha256, normalize_inline, rank, write_json, write_jsonl
from .config import DataSettings
from .sources import acquire_sources, iter_source
from .validation import DATA_FILES, validate_dataset, validate_records


PREPARER_VERSION = "data-preparation-v2-strict-hardneg"

_STOP_WORDS = set("a an and are as at be been by for from has have how i in is it of on or the to was were what when where which who with did does do list".split())
_GENERIC_ANCHORS = {"episode", "episodes", "film", "films", "song", "songs", "person", "people", "list", "series", "season", "seasons", "novel", "novels"}
_TYPE_PATTERNS = {
    "person": re.compile(r"\b(who|which actor|which actress|which author|which person|who played|who wrote)\b", re.I),
    "location": re.compile(r"\b(where|which city|which country|which state|what location)\b", re.I),
    "date": re.compile(r"\b(when|what year|which year|what date|what month)\b", re.I),
    "numeric": re.compile(r"\b(how many|how much|how long|how old|how often|what percentage|what number)\b", re.I),
    "work": re.compile(r"\b(which|what)\s+(song|film|movie|novel|book|magazine|album|series|episode|work)\b", re.I),
    "organization": re.compile(r"\b(which|what)\s+(company|organization|organisation|university|foundation|team|committee)\b", re.I),
}


def classify_question_type(question: str, evidence_text: str = "") -> str:
    for name, pattern in _TYPE_PATTERNS.items():
        if pattern.search(question):
            return name
    if re.search(r"\b(inc|ltd|company|organization|organisation|university|foundation|committee)\b", evidence_text, re.I):
        return "organization"
    if re.search(r"\b(city|country|state|river|lake|mountain|located)\b", evidence_text, re.I):
        return "location"
    if re.search(r"\b(born|died|founded|released|opened)\b", evidence_text, re.I):
        return "date"
    if re.search(r"\b(singer|actor|actress|author|player|politician|director|writer)\b", evidence_text, re.I):
        return "person"
    return "general"


def extract_anchor_terms(titles: list[str]) -> list[str]:
    anchors: set[str] = set()
    for title in titles:
        words = [word.casefold() for word in _TOKEN_RE.findall(title) if len(word) >= 3 and word.casefold() not in _STOP_WORDS | _GENERIC_ANCHORS]
        anchors.update(words)
        for width in (2, 3):
            anchors.update(" ".join(words[index:index + width]) for index in range(len(words) - width + 1))
    return sorted(anchors, key=lambda item: (-len(item.split()), item))


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


class _BM25Index:
    def __init__(self, corpus_tokens: Iterable[list[str]], vocabulary: set[str] | None = None):
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        total_length = 0
        for index, tokens in enumerate(corpus_tokens):
            self.lengths.append(len(tokens))
            total_length += len(tokens)
            frequencies = Counter(tokens)
            for term, frequency in frequencies.items():
                if vocabulary is None or term in vocabulary:
                    self.postings[term].append((index, frequency))
        self.document_count = len(self.lengths)
        self.average_length = total_length / self.document_count if self.document_count else 0.0

    def scores(self, query_tokens: list[str]) -> dict[int, float]:
        if not query_tokens or not self.lengths:
            return {}
        query_terms = Counter(query_tokens)
        n = self.document_count
        scores: dict[int, float] = defaultdict(float)
        k1, b = 1.5, 0.75
        for term in query_terms:
            postings = self.postings.get(term, [])
            df = len(postings)
            if not df:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for index, tf in postings:
                length = self.lengths[index]
                denom = tf + k1 * (1.0 - b + b * length / self.average_length) if self.average_length else tf + k1
                scores[index] += idf * (tf * (k1 + 1.0) / denom)
        return dict(scores)


def build_candidate_pool(documents: list[dict], excluded_doc_ids: set[str]) -> list[dict]:
    """Return a stable full-source candidate pool with excluded contexts removed."""
    return sorted(
        (document for document in documents if document["doc_id"] not in excluded_doc_ids),
        key=lambda document: document["doc_id"],
    )


def build_bm25_index(documents: list[dict], questions: list[str]) -> _BM25Index:
    """Index the bounded candidate text while retaining only query vocabulary postings."""
    vocabulary = {term for question in questions for term in _tokens(question)}
    return _BM25Index(
        (_tokens(f"{document['title']} {_truncate_tokens(document['text'])}") for document in documents),
        vocabulary,
    )


def rank_bm25_candidates(index: _BM25Index, question: str, pool: list[dict], limit: int, excluded_doc_ids: set[str] | None = None) -> list[dict]:
    """Rank positive BM25 matches using title plus a bounded document prefix."""
    scores = index.scores(_tokens(question))
    ranked = sorted(scores, key=lambda position: (-scores[position], pool[position]["doc_id"]))
    excluded_doc_ids = excluded_doc_ids or set()
    results = []
    for position, raw_rank in ((position, rank) for rank, position in enumerate(ranked, 1)):
        if pool[position]["doc_id"] in excluded_doc_ids:
            continue
        results.append({
            "candidate_index": position,
            "doc": pool[position],
            "bm25_rank": raw_rank,
            "bm25_score": float(scores[position]),
        })
        if len(results) == limit:
            break
    return results


def _truncate_tokens(text: str, limit: int = 1024) -> str:
    tokens = _TOKEN_RE.findall(text)
    return " ".join(tokens[:limit])


def _window_around_span(text: str, start: int, end: int, limit: int = 1024) -> str:
    matches = list(_TOKEN_RE.finditer(text))
    if len(matches) <= limit:
        return " ".join(match.group() for match in matches)
    first = next((index for index, match in enumerate(matches) if match.end() > start), 0)
    last = next((index for index, match in enumerate(matches) if match.start() >= end), first + 1)
    left = max(0, min(first - limit // 2, len(matches) - limit))
    right = min(len(matches), max(left + limit, last + limit // 2))
    if right - left > limit:
        right = left + limit
    return " ".join(match.group() for match in matches[left:right])


def build_evidence_representation(query: dict, evidence_items: list[dict], evidence_docs: list[dict]) -> str:
    titles = " ".join(doc["title"] for doc in evidence_docs)
    if query["dataset"] != "nq":
        return f"{titles} " + " ".join(item["text"] for item in evidence_items)
    docs = {doc["doc_id"]: doc for doc in evidence_docs}
    shorts = {item["source_ref"]["annotation_id"]: item for item in evidence_items if item["kind"] == "short_answer"}
    snippets = []
    for item in evidence_items:
        if item["kind"] != "long_answer":
            continue
        short = shorts.get(item["source_ref"]["annotation_id"])
        if short is not None:
            snippets.append(_window_around_span(docs[item["doc_id"]]["text"], short["start_char"], short["end_char"]))
    return f"{titles} " + " ".join(snippets)


def _candidate_type_match(question_type: str, title: str, text: str) -> bool:
    if question_type == "general":
        return True
    sample = f"{title} {text}"
    patterns = {
        "person": r"\b(born|died|actor|actress|author|singer|player|politician|director|writer|composer|athlete)\b",
        "location": r"\b(city|country|state|river|lake|mountain|island|located|capital|province)\b",
        "date": r"\b(\d{4}|january|february|march|april|may|june|july|august|september|october|november|december|founded|released|opened)\b",
        "numeric": r"\b\d+(?:\.\d+)?\b|\b(number|percent|percentage|times|million|thousand)\b",
        "work": r"\b(song|film|movie|novel|book|magazine|album|series|episode|television|radio)\b",
        "organization": r"\b(company|organization|organisation|university|foundation|team|committee|association|agency)\b",
    }
    return bool(re.search(patterns[question_type], sample, re.I))


def _candidate_has_anchor(title: str, text: str, anchors: list[str]) -> bool:
    if not anchors:
        return False
    candidate_tokens = set(_tokens(f"{title} {_truncate_tokens(text)}"))
    normalized = " ".join(_tokens(f"{title} {_truncate_tokens(text)}"))
    for anchor in anchors:
        anchor_tokens = anchor.split()
        if len(anchor_tokens) == 1 and anchor in candidate_tokens:
            return True
        if len(anchor_tokens) > 1 and anchor in normalized:
            return True
    return False


def _model_fingerprint(model_path: Path) -> str:
    if not model_path.is_dir():
        raise DataError(f"hard-negative E5 model directory does not exist: {model_path}")
    files = []
    for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
        files.append([str(path.relative_to(model_path)), file_sha256(path)])
    if not files:
        raise DataError(f"hard-negative E5 model directory is empty: {model_path}")
    return digest(files)


def _load_e5_model(settings: DataSettings):
    if not settings.hard_negative_e5_model_path:
        raise DataError("hard-negative E5 model path is required")
    model_path = settings.resolve(settings.hard_negative_e5_model_path)
    if not model_path.is_dir():
        raise DataError(f"hard-negative E5 model directory does not exist: {model_path}")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    except Exception as exc:
        raise DataError(f"cannot load local hard-negative E5 model {model_path}: {exc}") from exc
    return model


def _encode_e5(model, texts: list[str]) -> object:
    if not texts:
        return []
    values = model.encode(
        texts, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return values


class _E5EmbeddingCache:
    """Reuse passage/query vectors across questions during corpus-wide filtering."""

    def __init__(self, model):
        self.model = model
        self.vectors: dict[str, object] = {}

    def encode(self, texts: list[str]) -> list[object]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.vectors))
        if missing:
            vectors = _encode_e5(self.model, missing)
            self.vectors.update(zip(missing, vectors))
        return [self.vectors[text] for text in texts]


def encode_e5_candidates(model, question: str, evidence_text: str, candidates: list[dict], cache: _E5EmbeddingCache | None = None) -> list[dict]:
    """Attach query/evidence cosine scores to BM25-ranked candidates."""
    representations = [
        f"query: {question}",
        f"passage: {_truncate_tokens(evidence_text)}",
        *(f"passage: {item['doc']['title']} {_truncate_tokens(item['doc']['text'])}" for item in candidates),
    ]
    vectors = (cache or _E5EmbeddingCache(model)).encode(representations)
    query_vector, evidence_vector = vectors[:2]
    return [
        {
            **item,
            "query_e5_cosine": _dot(query_vector, vector),
            "evidence_e5_cosine": _dot(evidence_vector, vector),
        }
        for item, vector in zip(candidates, vectors[2:])
    ]


def select_hard_negatives(candidates: list[dict], count: int) -> list[dict]:
    """Select a deterministic top-scoring set of hard negatives."""
    ranked = sorted(candidates, key=lambda item: (-item["combined_score"], item["doc_id"]))
    return ranked[:count]


def validate_active_competition(selected: list[dict], active_top_k: int, minimum: int) -> bool:
    """Check that enough selected candidates appear in the BM25 active window."""
    return sum(item["bm25_rank"] <= active_top_k for item in selected) >= minimum


def rank_retrieval_competition(index: _BM25Index, question: str, pool: list[dict], evidence_doc_ids: set[str], hard_negative_doc_ids: set[str], active_top_k: int) -> dict:
    """Report final-corpus BM25 ranks for evidence and hard negatives."""
    scores = index.scores(_tokens(question))
    ranked = sorted(scores, key=lambda position: (-scores[position], pool[position]["doc_id"]))
    positions = {pool[position]["doc_id"]: rank for rank, position in enumerate(ranked, 1)}
    evidence_ranks = {doc_id: positions.get(doc_id) for doc_id in sorted(evidence_doc_ids)}
    hard_negative_ranks = {doc_id: positions.get(doc_id) for doc_id in sorted(hard_negative_doc_ids)}
    return {
        "evidence_bm25_ranks": evidence_ranks,
        "hard_negative_bm25_ranks": hard_negative_ranks,
        "active_hard_negative_count": sum(
            rank is not None and rank <= active_top_k
            for rank in hard_negative_ranks.values()
        ),
    }


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _near_duplicate(text: str, evidence_texts: list[str]) -> bool:
    candidate = normalize_inline(text)
    if not candidate:
        return True
    for evidence in evidence_texts:
        reference = normalize_inline(evidence)
        if candidate == reference:
            return True
        # SequenceMatcher is quadratic. Compare only bounded token windows
        # after the cheap full-text equality check so long NQ pages cannot
        # dominate candidate filtering.
        if min(len(candidate), len(reference)) >= 40:
            candidate_window = _truncate_tokens(candidate, 512)
            reference_window = _truncate_tokens(reference, 512)
            if SequenceMatcher(None, candidate_window, reference_window).ratio() >= 0.95:
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

    def all_documents(self, *, compact: bool = False) -> list[dict]:
        documents = []
        for payload, in self.connection.execute("SELECT payload FROM documents ORDER BY id"):
            doc = json.loads(payload)
            if compact:
                doc = {"doc_id": doc["doc_id"], "title": doc["title"], "text": _truncate_tokens(doc["text"])}
            documents.append(doc)
        return documents

    def get_document(self, doc_id: str) -> dict:
        row = self.connection.execute("SELECT payload FROM documents WHERE id=?", (doc_id,)).fetchone()
        if row is None:
            raise DataError(f"missing source document: {doc_id}")
        return json.loads(row[0])

    def all_queries(self) -> list[dict]:
        return [json.loads(payload) for payload, in self.connection.execute("SELECT payload FROM queries ORDER BY rank,id")]

    def choose_strict_queries(
        self, settings: DataSettings, model,
        progress: Callable[[str], None] = lambda message: None,
    ) -> tuple[list[dict], dict, dict]:
        all_rows = self.all_queries()
        dataset_name = all_rows[0]["query"]["dataset"] if all_rows else "dataset"
        total_queries = len(all_rows)
        progress(f"{dataset_name}: strict hard-negative filtering started ({total_queries} candidate queries)")
        all_documents = self.all_documents(compact=True)
        pool = build_candidate_pool(all_documents, set())
        index = build_bm25_index(pool, [row["query"]["question"] for row in all_rows])
        eligible: dict[str, list[dict]] = defaultdict(list)
        rejected = Counter()
        candidate_audit = []
        query_rejections = []
        e5_cache = _E5EmbeddingCache(model)
        def report_progress(index: int) -> None:
            if index % 10 == 0 or index == total_queries:
                progress(f"{dataset_name}: strict filtering {index}/{total_queries}; qualified={sum(map(len, eligible.values()))}; rejected={sum(rejected.values())}; cached_e5={len(e5_cache.vectors)}")

        for query_index, row in enumerate(all_rows, 1):
            query = row["query"]
            question_id = query["query_id"]
            evidence_ids = set(query["evidence_doc_ids"])
            context_ids = set(query["context_doc_ids"])
            evidence_docs = [self.get_document(doc_id) for doc_id in sorted(evidence_ids)]
            anchors = extract_anchor_terms([doc["title"] for doc in evidence_docs])
            if not anchors:
                rejected["low_anchor_overlap"] += 1
                query_rejections.append({"query_id": question_id, "reason": "low_anchor_overlap"})
                report_progress(query_index)
                continue
            evidence_items = row["evidence"]
            evidence_text = build_evidence_representation(query, evidence_items, evidence_docs)
            question_type = classify_question_type(query["question"], evidence_text)
            ranked = rank_bm25_candidates(index, query["question"], pool, settings.hard_negative_bm25_k)
            candidates = []
            base_audits: dict[str, dict] = {}
            reasons = Counter()
            for ranked_item in ranked:
                doc = self.get_document(ranked_item["doc"]["doc_id"])
                base_audit = {
                    "query_id": question_id, "doc_id": doc["doc_id"],
                    "bm25_rank": ranked_item["bm25_rank"],
                    "bm25_score": round(ranked_item["bm25_score"], 8),
                    "anchor_terms": anchors, "question_type_match": False,
                    "active_in_bm25_top20": ranked_item["bm25_rank"] <= settings.hard_negative_active_top_k,
                    "query_e5_cosine": None, "evidence_e5_cosine": None,
                    "combined_score": None, "selection_status": None,
                }
                if doc["doc_id"] in context_ids:
                    candidate_audit.append({**base_audit, "selection_status": "excluded_context", "reason": "context_document"})
                    continue
                if contains_answer(doc["text"], query["ground_truth"]):
                    reasons["answer_leakage"] += 1
                    candidate_audit.append({**base_audit, "selection_status": "rejected", "reason": "answer_leakage"})
                    continue
                if _near_duplicate(doc["text"], [item["text"] for item in evidence_docs]):
                    reasons["evidence_duplicate"] += 1
                    candidate_audit.append({**base_audit, "selection_status": "rejected", "reason": "evidence_duplicate"})
                    continue
                if not _candidate_has_anchor(doc["title"], doc["text"], anchors):
                    reasons["low_anchor_overlap"] += 1
                    candidate_audit.append({**base_audit, "selection_status": "rejected", "reason": "low_anchor_overlap"})
                    continue
                if not _candidate_type_match(question_type, doc["title"], _truncate_tokens(doc["text"])):
                    reasons["question_type_mismatch"] += 1
                    candidate_audit.append({**base_audit, "selection_status": "rejected", "reason": "question_type_mismatch"})
                    continue
                base_audit["question_type_match"] = True
                base_audits[doc["doc_id"]] = base_audit
                candidates.append({**ranked_item, "doc": doc})
            if len(candidates) < settings.hard_negative_count:
                reason = max(reasons, key=reasons.get) if reasons else "insufficient_candidates"
                rejected[reason] += 1
                query_rejections.append({
                    "query_id": question_id, "reason": reason,
                    "qualified_before_e5": len(candidates), "required": settings.hard_negative_count,
                })
                report_progress(query_index)
                continue
            encoded = encode_e5_candidates(model, query["question"], evidence_text, candidates, e5_cache)
            maximum_bm25 = max(item["bm25_score"] for item in candidates)
            qualified = []
            for item in encoded:
                doc = item["doc"]
                query_cosine = item["query_e5_cosine"]
                evidence_cosine = item["evidence_e5_cosine"]
                if query_cosine < settings.hard_negative_min_query_cosine or evidence_cosine < settings.hard_negative_min_evidence_cosine:
                    reasons["low_e5_similarity"] += 1
                    candidate_audit.append({
                        **base_audits[doc["doc_id"]],
                        "query_e5_cosine": round(query_cosine, 8), "evidence_e5_cosine": round(evidence_cosine, 8),
                        "selection_status": "rejected", "reason": "low_e5_similarity",
                    })
                    continue
                qualified.append({
                    "query_id": question_id, "doc_id": doc["doc_id"],
                    "bm25_rank": item["bm25_rank"], "bm25_score": round(item["bm25_score"], 8),
                    "query_e5_cosine": round(query_cosine, 8), "evidence_e5_cosine": round(evidence_cosine, 8),
                    "combined_score": round(0.4 * query_cosine + 0.4 * evidence_cosine + 0.2 * item["bm25_score"] / maximum_bm25, 8),
                    "anchor_terms": anchors, "question_type_match": True,
                    "question_type": question_type,
                    "active_in_bm25_top20": item["bm25_rank"] <= settings.hard_negative_active_top_k,
                    "selection_status": "selected",
                })
            for item in qualified:
                candidate_audit.append({
                    **item,
                    "selection_status": "qualified_not_selected",
                })
            chosen = select_hard_negatives(qualified, settings.hard_negative_count)
            if len(chosen) < settings.hard_negative_count:
                rejected["low_e5_similarity"] += 1
                query_rejections.append({
                    "query_id": question_id, "reason": "low_e5_similarity",
                    "qualified_after_e5": len(qualified), "required": settings.hard_negative_count,
                })
                report_progress(query_index)
                continue
            if not validate_active_competition(chosen, settings.hard_negative_active_top_k, settings.hard_negative_min_active):
                rejected["inactive_retrieval_competition"] += 1
                query_rejections.append({"query_id": question_id, "reason": "inactive_retrieval_competition"})
                report_progress(query_index)
                continue
            eligible[row["query"]["question_type"]].append({**row, "audit": qualified})
            report_progress(query_index)
        counts = {key: len(value) for key, value in eligible.items()}
        if sum(counts.values()) < settings.query_count:
            raise DataError(f"insufficient qualified queries: need {settings.query_count}, found {sum(counts.values())}; rejection reasons: {dict(rejected)}")
        allocations = allocate_strata(counts, settings.query_count)
        chosen_positions = {stratum: list(range(amount)) for stratum, amount in allocations.items()}
        next_position = {stratum: amount for stratum, amount in allocations.items()}
        while True:
            selected = [(stratum, position, eligible[stratum][position]) for stratum in sorted(chosen_positions) for position in chosen_positions[stratum]]
            selected_context_ids = {doc_id for _, _, row in selected for doc_id in row["query"]["context_doc_ids"]}
            selected_negatives = {}
            failed = None
            for stratum, position, row in selected:
                compatible = [item for item in row["audit"] if item["doc_id"] not in selected_context_ids]
                negatives = select_hard_negatives(compatible, settings.hard_negative_count)
                if len(negatives) != settings.hard_negative_count or not validate_active_competition(negatives, settings.hard_negative_active_top_k, settings.hard_negative_min_active):
                    failed = (stratum, position)
                    break
                selected_negatives[row["query"]["query_id"]] = negatives
            if failed is None:
                break
            stratum, position = failed
            rejected["document_role_conflict"] += 1
            if next_position[stratum] >= len(eligible[stratum]):
                failed_row = eligible[stratum][position]
                compatible_count = sum(item["doc_id"] not in selected_context_ids for item in failed_row["audit"])
                raise DataError(
                    f"insufficient role-compatible qualified queries in {stratum}: need {allocations[stratum]}; "
                    f"query {failed_row['query']['query_id']} has {compatible_count} compatible candidates; "
                    f"rejection reasons: {dict(rejected)}"
                )
            chosen_positions[stratum][chosen_positions[stratum].index(position)] = next_position[stratum]
            next_position[stratum] += 1
        selected = [row for _, _, row in selected]
        for row in selected:
            query = row["query"]
            negatives = selected_negatives[query["query_id"]]
            negative_ids = {item["doc_id"] for item in negatives}
            query["hard_negative_doc_ids"] = sorted(negative_ids)
            query["related_doc_ids"] = sorted(set(query["evidence_doc_ids"]) | negative_ids)
            query["hard_negative_quality"] = {
                "min_query_e5_cosine": min(item["query_e5_cosine"] for item in negatives),
                "min_evidence_e5_cosine": min(item["evidence_e5_cosine"] for item in negatives),
                "active_hard_negative_count": sum(item["active_in_bm25_top20"] for item in negatives),
                "anchor_terms": negatives[0]["anchor_terms"],
                "question_type": negatives[0]["question_type"],
            }
            for item in candidate_audit:
                if item.get("query_id") == query["query_id"] and item.get("doc_id") in negative_ids and item.get("selection_status") == "qualified_not_selected":
                    item["selection_status"] = "selected"
            row["audit"] = negatives
        selected.sort(key=lambda item: item["query"]["query_id"])
        selected_audits = [item for row in selected for item in row["audit"]]
        reuse = Counter(item["doc_id"] for item in selected_audits)
        sampling = {"eligible_question_types": counts, "allocated_question_types": allocations}
        report = {
            "enabled": True, "requested_per_query": settings.hard_negative_count,
            "candidate_pool_documents": len(pool), "qualified_queries": sum(counts.values()),
            "rejection_reasons": dict(sorted(rejected.items())), "selections": selected_audits,
            "candidate_audit": candidate_audit,
            "candidate_shortfalls": query_rejections,
            "per_query_hard_negative_count": {row["query"]["query_id"]: len(row["query"]["hard_negative_doc_ids"]) for row in selected},
            "unique_hard_negative_documents": len(reuse),
            "hard_negative_reuse_count": sum(amount - 1 for amount in reuse.values()),
        }
        return selected, sampling, report

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
    strict = settings.schema_version == "data-v2" and settings.hard_negative_count > 0
    model_sha256 = _model_fingerprint(settings.resolve(settings.hard_negative_e5_model_path)) if strict else None
    code = _code_identity(settings)
    sources = acquire_sources(settings, dataset, offline=offline, progress=progress)
    fingerprint = digest({"config": settings.contract(dataset), "sources": sources, "preparer_version": code["preparer_version"], "source_sha256": code["source_sha256"], "model_sha256": model_sha256})
    destination = settings.output_dir(dataset)
    if destination.exists():
        report = validate_dataset(destination)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("input_fingerprint") != fingerprint:
            raise DataError(f"{destination}: frozen inputs/configuration/code changed; choose a new data_version")
        return {**report, "reused": True, "output_dir": str(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = _load_e5_model(settings) if strict else None
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
            if strict:
                selected, sampling, hard_negative_report = store.choose_strict_queries(settings, model, progress=progress)
            else:
                selected, sampling = store.choose_queries(settings)
                hard_negative_report = None
            queries = [row["query"] for row in selected]
            evidence = sorted((item for row in selected for item in row["evidence"]), key=lambda item: item["evidence_id"])
            if strict:
                hard_negative_ids = {query["query_id"]: query["hard_negative_doc_ids"] for query in queries}
            else:
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
            if strict:
                # Recheck the exact published corpus after the 2,000-document
                # background fill. This prevents a candidate that was active in
                # the complete source pool from becoming inactive after sampling.
                final_pool = build_candidate_pool(documents, set())
                final_index = build_bm25_index(final_pool, [query["question"] for query in queries])
                final_competition = {}
                for query in queries:
                    competition = rank_retrieval_competition(
                        final_index, query["question"], final_pool,
                        set(query["evidence_doc_ids"]), set(query["hard_negative_doc_ids"]),
                        settings.hard_negative_active_top_k,
                    )
                    if competition["active_hard_negative_count"] < settings.hard_negative_min_active:
                        raise DataError(
                            f"inactive retrieval competition in final corpus for {query['query_id']}: "
                            f"{competition['active_hard_negative_count']} hard negatives in BM25 Top-{settings.hard_negative_active_top_k}"
                        )
                    final_competition[query["query_id"]] = competition
                hard_negative_report["final_corpus_competition"] = final_competition
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
            "hard_negative_e5_model": {"name": settings.hard_negative_e5_model, "sha256": model_sha256} if strict else None,
            "validation": report,
            "limitations": ["Short answers only; a controlled-size development-set corpus, not a full-Wikipedia benchmark.", "NQ documents are full pages; HotpotQA documents are supplied paragraphs.", "NQ requires two agreeing single-span annotations; HotpotQA excludes yes/no answers."],
        }
        write_json(output / "manifest.json", manifest)
        validate_dataset(output)
        output.rename(destination)
    return {**report, "dataset": dataset, "data_version": settings.data_version, "status": "validated", "human_review_status": "pending", "reused": False, "output_dir": str(destination)}
