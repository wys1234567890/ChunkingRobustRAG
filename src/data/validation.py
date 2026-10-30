"""Public JSONL loaders and strict cross-file verification for clean-v1 data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import DataError, answer_key, contains_answer, digest, file_sha256, read_jsonl


DATA_FILES = ("documents.jsonl", "queries.jsonl", "evidence.jsonl", "audit_samples.jsonl")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataError(message)


def _string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _versions(record: dict) -> None:
    _require(record.get("dataset", "nq") in ("nq", "hotpotqa"), "unknown dataset")
    _require(record.get("schema_version") == "data-v1", "invalid schema_version")
    _require(_string(record.get("data_version")), "missing data_version")


def _string_list(value: object, *, nonempty: bool = True) -> bool:
    return isinstance(value, list) and (bool(value) or not nonempty) and all(_string(item) for item in value) and len(value) == len(set(value))


def _check_document(doc: dict) -> None:
    _versions(doc)
    for key in ("doc_id", "dataset", "text", "text_sha256", "source_doc_id"):
        _require(_string(doc.get(key)), f"document missing {key}")
    _require(isinstance(doc.get("title"), str), "document title must be a string")
    _require(doc.get("source_doc_id") == doc["doc_id"], "clean source_doc_id must equal doc_id")
    _require(doc.get("document_role") in {"candidate", "evidence", "hard_negative", "background", "poison"}, "invalid document_role")
    _require(doc.get("is_poison") is False, "clean document is_poison must be false")
    _require(all(key in doc and doc[key] is None for key in ("corruption_type", "true_fact", "false_fact")), "clean document contains pollution metadata")
    _require(_string_list(doc.get("target_query_ids"), nonempty=False), "invalid target_query_ids")
    _require(doc.get("poison_version") == doc["data_version"], "clean poison_version mismatch")
    _require(isinstance(doc.get("source_refs"), list) and bool(doc["source_refs"]), "document requires source_refs")
    for ref in doc["source_refs"]:
        _require(isinstance(ref, dict) and _string(ref.get("raw_file")) and _string(ref.get("source_query_id")), "invalid document source reference")
    _require(doc["text_sha256"] == hashlib.sha256(doc["text"].encode("utf-8")).hexdigest(), "document text hash mismatch")
    _require(doc["doc_id"] == f"{doc['dataset']}-doc-{digest([doc['dataset'], doc['title'], doc['text']])}", "document identity mismatch")


def _check_query(query: dict) -> None:
    _versions(query)
    for key in ("query_id", "dataset", "question", "source_query_id"):
        _require(_string(query.get(key)), f"query missing {key}")
    _require(query["query_id"] == f"{query['dataset']}-{query['source_query_id']}", "query identity mismatch")
    for key in ("ground_truth", "evidence_doc_ids", "context_doc_ids"):
        _require(_string_list(query.get(key)), f"invalid query {key}")
    for key in ("hard_negative_doc_ids", "related_doc_ids"):
        _require(_string_list(query.get(key), nonempty=False), f"invalid query {key}")
    _require(set(query["evidence_doc_ids"]) <= set(query["related_doc_ids"]), "evidence missing from related documents")
    _require(set(query["hard_negative_doc_ids"]) <= set(query["related_doc_ids"]), "hard negative missing from related documents")
    _require(set(query["evidence_doc_ids"]) & set(query["hard_negative_doc_ids"]) == set(), "evidence also marked hard negative")


def _check_evidence(evidence: dict) -> None:
    _versions(evidence)
    for key in ("evidence_id", "query_id", "doc_id", "text"):
        _require(_string(evidence.get(key)), f"evidence missing {key}")
    _require(evidence.get("kind") in ("short_answer", "long_answer", "supporting_sentence"), "invalid evidence kind")
    start, end = evidence.get("start_char"), evidence.get("end_char")
    _require(type(start) is int and type(end) is int and 0 <= start < end, "invalid evidence offsets")
    ref = evidence.get("source_ref")
    _require(isinstance(ref, dict) and _string(ref.get("source_query_id")) and _string(ref.get("raw_file")), "invalid evidence source_ref")
    if evidence["kind"] == "supporting_sentence":
        _require(isinstance(ref.get("title"), str) and all(type(ref.get(key)) is int and ref[key] >= 0 for key in ("context_index", "sentence_id")), "invalid supporting sentence provenance")
    else:
        _require(_string(ref.get("annotation_id")), "missing NQ annotation_id")
        for unit in ("token", "byte"):
            a, b = ref.get(f"start_{unit}"), ref.get(f"end_{unit}")
            _require(type(a) is int and type(b) is int and 0 <= a < b, f"invalid NQ {unit} provenance")
        if evidence["kind"] == "short_answer":
            _require(_string(ref.get("original_answer_text")), "missing original NQ answer")
    identity = {key: value for key, value in evidence.items() if key != "evidence_id"}
    _require(evidence["evidence_id"] == "ev-" + digest(identity), "evidence identity mismatch")


def _load(path: str | Path, check) -> list[dict]:
    records = read_jsonl(path)
    for index, record in enumerate(records, 1):
        try:
            check(record)
        except DataError as exc:
            raise DataError(f"{path}:{index}: {exc}") from exc
    return records


def load_documents(path: str | Path) -> list[dict]:
    return _load(path, _check_document)


def load_queries(path: str | Path) -> list[dict]:
    return _load(path, _check_query)


def load_evidence(path: str | Path) -> list[dict]:
    return _load(path, _check_evidence)


def validate_records(documents: list[dict], queries: list[dict], evidence: list[dict]) -> dict:
    for records, field, check in ((documents, "doc_id", _check_document), (queries, "query_id", _check_query), (evidence, "evidence_id", _check_evidence)):
        for record in records:
            check(record)
        _require(len(records) == len({record[field] for record in records}), f"duplicate {field}")
    docs = {doc["doc_id"]: doc for doc in documents}
    questions = {query["query_id"]: query for query in queries}
    by_query: dict[str, list[dict]] = {key: [] for key in questions}
    for item in evidence:
        _require(item["doc_id"] in docs, "dangling evidence doc_id")
        _require(item["query_id"] in questions, "dangling evidence query_id")
        doc = docs[item["doc_id"]]
        query = questions[item["query_id"]]
        _require(item["source_ref"]["source_query_id"] == query["source_query_id"], "evidence source query mismatch")
        _require(any(all(item["source_ref"].get(key) == value for key, value in ref.items()) for ref in doc["source_refs"]), "evidence provenance does not belong to document")
        _require(item["end_char"] <= len(doc["text"]) and doc["text"][item["start_char"]:item["end_char"]] == item["text"], "evidence span readback mismatch")
        _require(all(item[key] == query[key] == doc[key] for key in ("data_version", "schema_version")), "cross-file version mismatch")
        _require(doc["dataset"] == query["dataset"], "cross-dataset evidence reference")
        by_query[item["query_id"]].append(item)
    for query in queries:
        items = by_query[query["query_id"]]
        _require(bool(items), "query has no evidence")
        _require(set(query["evidence_doc_ids"]) == {item["doc_id"] for item in items}, "query evidence_doc_ids mismatch")
        _require(set(query["evidence_doc_ids"]) <= set(query["context_doc_ids"]) <= docs.keys(), "dangling query context reference")
        _require(set(query["hard_negative_doc_ids"]) <= docs.keys(), "dangling hard negative reference")
        _require(set(query["related_doc_ids"]) <= docs.keys(), "dangling related document reference")
        _require(set(query["hard_negative_doc_ids"]) & set(query["context_doc_ids"]) == set(), "hard negative is a context document")
        for doc_id in query["hard_negative_doc_ids"]:
            _require(not contains_answer(docs[doc_id]["text"], query["ground_truth"]), "hard negative contains ground truth")
        _require(all(docs[doc_id]["dataset"] == query["dataset"] for doc_id in query["context_doc_ids"]), "cross-dataset query context")
        _require(all(any(ref["source_query_id"] == query["source_query_id"] for ref in docs[doc_id]["source_refs"]) for doc_id in query["context_doc_ids"]), "query context does not belong to source question")
        if query["dataset"] == "nq":
            shorts = [item for item in items if item["kind"] == "short_answer"]
            longs = {item["source_ref"]["annotation_id"]: item for item in items if item["kind"] == "long_answer"}
            _require(len(shorts) >= 2 and len({item["source_ref"]["annotation_id"] for item in shorts}) == len(shorts), "NQ requires two distinct agreeing annotations")
            _require(len(items) == 2 * len(shorts) == 2 * len(longs), "NQ short/long annotation mismatch")
            _require(len({answer_key(item["source_ref"]["original_answer_text"]) for item in shorts}) == 1, "NQ annotation answers conflict")
            for item in shorts:
                long = longs.get(item["source_ref"]["annotation_id"])
                _require(long is not None and long["start_char"] <= item["start_char"] < item["end_char"] <= long["end_char"], "NQ answer outside long evidence")
                _require(item["text"] in query["ground_truth"] and item["source_ref"]["original_answer_text"] in query["ground_truth"], "NQ ground_truth mismatch")
        else:
            _require(all(item["kind"] == "supporting_sentence" for item in items), "HotpotQA requires sentence evidence")
    _require(bool(documents) and bool(queries), "prepared dataset must not be empty")
    return {"document_count": len(documents), "query_count": len(queries), "evidence_count": len(evidence), "evidence_reference_completeness": 1.0, "span_readback_rate": 1.0}


def validate_dataset(directory: str | Path, *, expected_config: dict | None = None) -> dict:
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise DataError(f"cannot load dataset manifest: {exc}") from exc
    _require(isinstance(manifest, dict) and manifest.get("status") == "validated", "dataset lacks a successful validation manifest")
    checksums = manifest.get("output_sha256")
    _require(isinstance(checksums, dict) and set(checksums) == set(DATA_FILES), "manifest requires all output hashes")
    for filename in DATA_FILES:
        path = directory / filename
        _require(path.is_file(), f"missing output: {filename}")
        _require(file_sha256(path) == checksums[filename], f"output hash mismatch: {filename}")
    docs = load_documents(directory / "documents.jsonl")
    queries = load_queries(directory / "queries.jsonl")
    evidence = load_evidence(directory / "evidence.jsonl")
    report = validate_records(docs, queries, evidence)
    config = manifest.get("config")
    _require(isinstance(config, dict), "missing manifest configuration")
    if expected_config is not None:
        _require(config == expected_config, "frozen dataset configuration changed; choose a new data_version")
    _require(len(docs) == config.get("document_count") and len(queries) == config.get("query_count"), "dataset count mismatch")
    hard_negative_count = config.get("hard_negative_count", 0)
    _require(type(hard_negative_count) is int and hard_negative_count >= 0, "invalid hard_negative_count")
    _require(all(len(query["hard_negative_doc_ids"]) == hard_negative_count for query in queries), "hard negative count mismatch")
    _require(all(item["data_version"] == config.get("data_version") and item["schema_version"] == config.get("schema_version") for item in docs + queries + evidence), "manifest version mismatch")
    _require(all(item["dataset"] == manifest.get("dataset") for item in docs + queries), "manifest dataset mismatch")
    _require(manifest.get("sampled_query_ids") == sorted(item["query_id"] for item in queries), "manifest query selection mismatch")
    audits = read_jsonl(directory / "audit_samples.jsonl")
    _require(len(audits) == config.get("audit_count") and len({item.get("query_id") for item in audits}) == len(audits), "audit sample count mismatch")
    by_query = {query["query_id"]: query for query in queries}
    for audit in audits:
        query_id = audit.get("query_id")
        _require(query_id in by_query and audit.get("query") == by_query[query_id], "audit query mismatch")
        expected = sorted((item for item in evidence if item["query_id"] == query_id), key=lambda item: item["evidence_id"])
        _require(audit.get("evidence") == expected, "audit evidence mismatch")
        _require(audit.get("review_status") == "pending", "generated audits must not claim human approval")
    _require(manifest.get("validation") == report, "manifest validation report mismatch")
    hard_negative_ids = {doc_id for query in queries for doc_id in query["hard_negative_doc_ids"]}
    _require(all(doc["document_role"] != "hard_negative" or doc["doc_id"] in hard_negative_ids for doc in docs), "unreferenced hard negative document")
    _require(all(doc["doc_id"] not in hard_negative_ids or set(doc["target_query_ids"]) >= {query["query_id"] for query in queries if doc["doc_id"] in query["hard_negative_doc_ids"]} for doc in docs), "hard negative target query mismatch")
    _require(all(max(len(answer.split()) for answer in item["ground_truth"]) <= config.get("max_answer_words", 0) for item in queries), "answer exceeds configured word limit")
    _require(all(answer_key(answer) not in {"yes", "no"} for item in queries for answer in item["ground_truth"]), "yes/no answer in short-answer subset")
    return {"dataset": manifest["dataset"], "data_version": config["data_version"], "status": "validated", "human_review_status": manifest.get("human_review_status"), **report}
