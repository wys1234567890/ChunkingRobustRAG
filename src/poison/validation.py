"""Independent validation for published Stage 2 clean+poison datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.common import DataError, file_sha256, read_jsonl
from ..data.validation import _check_document, _check_evidence, _check_query
from .config import DATASETS, DENOMINATORS
from .identity import poison_doc_id, text_sha256
from .operators import DECLARATIVE_TEMPLATE, _contains_phrase, _phrase_pattern
from .ratios import actual_ratio, poison_count
from .safety import check_text

DATA_FILES = ("documents.jsonl", "queries.jsonl", "evidence.jsonl", "poison_records.jsonl", "audit_samples.jsonl")
AUDIT_SAMPLE_SIZE = 20
REVIEW_STATUSES = ("pending", "approved", "rejected")
CONSTRUCTION_MODES = ("answer_replacement", "declarative_append")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataError(message)


def _string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any, *, nonempty: bool = True) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(_string(item) for item in value)
        and len(value) == len(set(value))
    )


def _check_poison_document(doc: dict, clean_docs: dict[str, dict], queries: dict[str, dict]) -> None:
    for key in ("doc_id", "dataset", "title", "text", "source_doc_id", "text_sha256", "poison_version", "data_version"):
        _require(_string(doc.get(key)), f"poison document missing {key}")
    _require(doc["dataset"] in DATASETS, "poison document has an unknown dataset")
    _require(doc.get("schema_version") in ("data-v1", "data-v2"), "poison document has an invalid schema_version")
    _require(doc.get("is_poison") is True, "poison document is_poison must be true")
    _require(doc.get("document_role") == "poison", "poison document_role must be poison")
    _require(doc.get("corruption_type") == "entity_substitution", "unknown corruption_type")
    _require(doc.get("review_status") in REVIEW_STATUSES, "invalid poison review_status")
    _require(doc.get("construction_mode") in CONSTRUCTION_MODES, "invalid construction mode")
    _require(type(doc.get("variant_index")) is int and doc["variant_index"] >= 0, "invalid variant_index")
    _require(doc.get("source_doc_id") in clean_docs, "poison source_doc_id must point at a clean document")
    _require(doc["source_doc_id"] != doc["doc_id"], "poison doc_id must not equal its source_doc_id")
    _require(_string(doc.get("true_fact")) and _string(doc.get("false_fact")), "poison facts must be non-empty strings")
    _require(_string_list(doc.get("target_query_ids")) and len(doc["target_query_ids"]) == 1, "poison must serve exactly one query")
    _require(doc["target_query_ids"][0] in queries, "poison target query is missing")
    _require(doc["text_sha256"] == text_sha256(doc["text"]), "poison text hash mismatch")
    _require(
        doc["doc_id"]
        == poison_doc_id(
            doc["dataset"],
            doc["data_version"],
            doc["poison_version"],
            doc["target_query_ids"][0],
            doc["source_doc_id"],
            doc["corruption_type"],
            doc.get("corruption_operator_version", ""),
            doc["variant_index"],
            doc["true_fact"],
            doc["false_fact"],
            doc["text"],
        ),
        "poison document identity mismatch",
    )
    _require(isinstance(doc.get("source_refs"), list) and bool(doc["source_refs"]), "poison document requires source_refs")
    for ref in doc["source_refs"]:
        _require(
            isinstance(ref, dict) and _string(ref.get("raw_file")) and _string(ref.get("source_query_id")),
            "invalid poison source reference",
        )


def _check_record(
    record: dict,
    documents: dict[str, dict],
    queries: dict[str, dict],
    evidence_by_query: dict[str, list[dict]],
) -> None:
    for key in ("poison_doc_id", "dataset", "target_query_id", "source_doc_id", "corruption_type", "operator_version"):
        _require(_string(record.get(key)), f"poison record missing {key}")
    _require(record["poison_doc_id"] in documents, "poison record points at a missing document")
    doc = documents[record["poison_doc_id"]]
    _require(doc["is_poison"] is True, "poison record must point at a poison document")
    _require(record["dataset"] == doc["dataset"], "poison record dataset mismatch")
    _require(record["target_query_id"] in queries, "poison record target query is missing")
    _require(record["source_doc_id"] == doc["source_doc_id"], "poison record source mismatch")
    _require(record["source_doc_id"] in documents and documents[record["source_doc_id"]]["is_poison"] is False, "poison record source must be a clean document")
    _require(record["source_doc_id"] in queries[record["target_query_id"]]["evidence_doc_ids"], "poison source is not an evidence document")
    _require(record["corruption_type"] == doc["corruption_type"], "poison record corruption type mismatch")
    _require(record["operator_version"] == doc["corruption_operator_version"], "poison record operator mismatch")
    _require(record.get("variant_index") == doc["variant_index"], "poison record variant mismatch")
    _require(record.get("text_sha256") == doc["text_sha256"], "poison record hash mismatch")
    _require(record.get("review_status") in REVIEW_STATUSES, "invalid poison record review status")
    _require(record.get("contains_true_fact_after") is False, "poison record still contains the true fact")
    _require(record.get("explicit_instruction_hits") == [], "poison record has blocklist hits")
    _require(record.get("construction_mode") in CONSTRUCTION_MODES, "invalid poison record construction mode")
    _require(type(record.get("replacement_count")) is int and record["replacement_count"] >= 0, "invalid replacement_count")
    truth = record.get("true_fact")
    falsity = record.get("false_fact")
    _require(isinstance(truth, dict) and _string(truth.get("value")), "invalid record true_fact")
    _require(isinstance(falsity, dict) and _string(falsity.get("value")), "invalid record false_fact")
    _require(truth["value"] == doc["true_fact"] and falsity["value"] == doc["false_fact"], "poison record facts mismatch the document")
    _require(_string_list(truth.get("answer_keys")), "invalid true_fact answer_keys")
    _require(_string(falsity.get("selection_source")), "invalid false_fact selection_source")
    _require(_string_list(record.get("source_evidence_ids")), "invalid source_evidence_ids")
    known_ids = {item["evidence_id"] for item in evidence_by_query.get(record["target_query_id"], [])}
    _require(set(record["source_evidence_ids"]) <= known_ids, "poison record references foreign evidence")
    _require(
        all(value.casefold() != falsity["value"].casefold() for value in truth["answer_keys"]),
        "true and false facts must differ",
    )
    occurrences = len(_phrase_pattern(falsity["value"]).findall(doc["text"]))
    _require(occurrences >= 1, "false fact is missing from the poison text")
    _require(not _contains_phrase(doc["text"], truth["value"]), "true fact remains in the poison text")
    _require(not check_text(doc["text"]), "poison text hits the explicit-instruction blocklist")
    if doc["construction_mode"] == "answer_replacement":
        _require(record["replacement_count"] >= 1, "answer_replacement requires at least one replacement")
        _require(record["replacement_count"] <= occurrences, "replacement count exceeds false-fact occurrences")
    else:
        _require(record["replacement_count"] == 0, "declarative_append must not report replacements")
        expected = DECLARATIVE_TEMPLATE.format(false_fact=doc["false_fact"])
        _require(expected in doc["text"], "declarative_append sentence is missing")


def load_poison_documents(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def load_poison_records(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def validate_poison_dataset(
    directory: str | Path,
    *,
    expected_clean_manifest: dict | None = None,
) -> dict:
    from .builder import manifest_sha256

    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise DataError(f"cannot load poison manifest: {exc}") from exc
    _require(isinstance(manifest, dict) and manifest.get("status") == "validated", "poison dataset lacks a validated manifest")
    checksums = manifest.get("output_sha256")
    _require(isinstance(checksums, dict) and set(checksums) == set(DATA_FILES), "poison manifest requires all output hashes")
    for filename in DATA_FILES:
        path = directory / filename
        _require(path.is_file(), f"missing poison output: {filename}")
        _require(file_sha256(path) == checksums[filename], f"poison output hash mismatch: {filename}")

    dataset = manifest.get("dataset")
    _require(dataset in DATASETS, "invalid poison manifest dataset")
    ratio = manifest.get("poison_ratio_target")
    _require(type(ratio) in (int, float) and 0 <= float(ratio) < 1, "invalid poison_ratio_target")
    denominator = manifest.get("ratio_denominator")
    _require(denominator in DENOMINATORS, "invalid ratio_denominator")

    documents = read_jsonl(directory / "documents.jsonl")
    queries = read_jsonl(directory / "queries.jsonl")
    evidence = read_jsonl(directory / "evidence.jsonl")
    records = load_poison_records(directory / "poison_records.jsonl")
    audits = read_jsonl(directory / "audit_samples.jsonl")

    for field, rows in (("doc_id", documents), ("query_id", queries), ("evidence_id", evidence), ("poison_doc_id", records)):
        _require(all(_string(row.get(field)) for row in rows), f"invalid or missing {field}")
        _require(len(rows) == len({row[field] for row in rows}), f"duplicate {field}")

    docs_by_id = {doc["doc_id"]: doc for doc in documents}
    queries_by_id = {query["query_id"]: query for query in queries}
    evidence_by_query: dict[str, list[dict]] = {query["query_id"]: [] for query in queries}
    for item in evidence:
        evidence_by_query.setdefault(item["query_id"], []).append(item)

    clean_docs = {doc_id: doc for doc_id, doc in docs_by_id.items() if doc.get("is_poison") is False}
    poison_docs = {doc_id: doc for doc_id, doc in docs_by_id.items() if doc.get("is_poison") is True}
    _require(len(clean_docs) + len(poison_docs) == len(docs_by_id), "documents must be clearly clean or poison")
    for doc in clean_docs.values():
        _check_document(doc)
    for query in queries:
        _check_query(query)
    for item in evidence:
        _check_evidence(item)
    for doc in poison_docs.values():
        _check_poison_document(doc, clean_docs, queries_by_id)
    for record in records:
        _check_record(record, docs_by_id, queries_by_id, evidence_by_query)

    _require(set(poison_docs) == {record["poison_doc_id"] for record in records}, "poison documents and records must be one-to-one")

    clean_count = len(clean_docs)
    poison_count_actual = len(poison_docs)
    final_count = len(docs_by_id)
    _require(manifest.get("clean_document_count") == clean_count, "clean document count mismatch")
    _require(manifest.get("poison_document_count") == poison_count_actual, "poison document count mismatch")
    _require(manifest.get("final_document_count") == final_count, "final document count mismatch")
    _require(manifest.get("query_count") == len(queries_by_id), "query count mismatch")
    expected_poison = poison_count(clean_count, float(ratio), denominator)
    _require(expected_poison == poison_count_actual, f"ratio yields {expected_poison} poison documents, found {poison_count_actual}")
    _require(abs(manifest.get("poison_ratio_actual", -1) - actual_ratio(clean_count, poison_count_actual)) < 1e-12, "poison_ratio_actual mismatch")

    poison_ids_by_query = {}
    for query in queries:
        ids = query.get("poison_doc_ids")
        _require(_string_list(ids, nonempty=False), f"query {query['query_id']} has invalid poison_doc_ids")
        _require(all(doc_id in poison_docs for doc_id in ids), "query references a non-poison document")
        poison_ids_by_query[query["query_id"]] = sorted(ids)
    for poison in poison_docs.values():
        _require(
            poison_ids_by_query[poison["target_query_ids"][0]] == sorted(
                doc_id
                for doc_id, candidate in poison_docs.items()
                if candidate["target_query_ids"][0] == poison["target_query_ids"][0]
            ),
            "query poison_doc_ids do not match the reverse mapping",
        )

    covered = sorted(query_id for query_id, ids in poison_ids_by_query.items() if ids)
    uncovered = sorted(query_id for query_id, ids in poison_ids_by_query.items() if poison_count_actual and not ids)
    expected_coverage = None if not poison_count_actual else len(covered) / len(queries_by_id)
    _require(manifest.get("poison_question_coverage") == expected_coverage, "poison_question_coverage mismatch")
    if poison_count_actual:
        _require(expected_coverage == 1.0 and not uncovered, "poison dataset does not cover every query")
        counts = [len(ids) for ids in poison_ids_by_query.values()]
        _require(max(counts) - min(counts) <= 1, "poison documents are not evenly allocated across queries")
    else:
        _require(all(not ids for ids in poison_ids_by_query.values()), "0% ratio must not reference poison documents")
        _require(manifest.get("coverage_status") == "baseline_no_poison", "0% coverage status must be baseline_no_poison")
        _require(not records and not audits, "0% ratio must not contain records or audits")

    _require(manifest.get("covered_query_ids") == covered, "covered_query_ids mismatch")
    _require(manifest.get("uncovered_query_ids") == uncovered, "uncovered_query_ids mismatch")

    clean_texts = {doc["text"] for doc in clean_docs.values()}
    poison_texts = [doc["text"] for doc in poison_docs.values()]
    _require(len(poison_texts) == len(set(poison_texts)), "duplicate poison text")
    _require(all(text not in clean_texts for text in poison_texts), "poison text duplicates a clean document")

    expected_audits = min(AUDIT_SAMPLE_SIZE, poison_count_actual)
    _require(len(audits) == expected_audits, "audit sample count mismatch")
    _require(all(item.get("review_status") == "pending" for item in audits), "audit samples must start pending")
    for audit in audits:
        poison = poison_docs.get(audit.get("poison_document", {}).get("doc_id"))
        _require(poison is not None, "audit sample references a missing poison document")
        query = audit.get("query")
        _require(isinstance(query, dict) and query.get("query_id") in queries_by_id, "audit sample has an invalid query")
        _require(query == queries_by_id[query["query_id"]], "audit query mismatch")
        _require(audit["poison_document"]["text"] == poison["text"], "audit poison text mismatch")

    _require(manifest.get("human_review_status") in REVIEW_STATUSES, "invalid human_review_status")
    _require(manifest.get("human_review_status") != "approved", "automatic generation must not claim human approval")
    if expected_clean_manifest is not None:
        _require(
            manifest.get("clean_manifest_sha256") == manifest_sha256(expected_clean_manifest),
            "clean manifest changed since this poison version was built",
        )
        expected_counts = expected_clean_manifest.get("config", {})
        _require(expected_counts.get("document_count") == clean_count, "clean document count differs from the input manifest")
        _require(expected_counts.get("query_count") == len(queries_by_id), "clean query count differs from the input manifest")

    return {
        "dataset": dataset,
        "poison_version": manifest.get("poison_version"),
        "poison_ratio_target": ratio,
        "poison_ratio_actual": manifest.get("poison_ratio_actual"),
        "clean_document_count": clean_count,
        "poison_document_count": poison_count_actual,
        "final_document_count": final_count,
        "query_count": len(queries_by_id),
        "poison_question_coverage": expected_coverage,
        "uncovered_query_ids": uncovered,
        "blocklist_hit_count": 0,
        "duplicate_text_count": 0,
        "status": "validated",
    }
