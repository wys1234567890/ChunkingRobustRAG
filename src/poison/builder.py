"""Build clean+poison Stage 2 knowledge bases with atomic publication."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from ..data.common import (
    DataError,
    canonical_json,
    digest,
    file_sha256,
    normalize_inline,
    rank,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ..data.validation import load_documents, load_evidence, load_queries, validate_dataset
from .config import DATASETS, PoisonSettings
from .identity import poison_doc_id, text_sha256
from .operators import DECLARATIVE_TEMPLATE, CorruptionCandidate, EntitySubstitutionOperator
from .ratios import actual_ratio, allocate_fixed, allocate_variants, fixed_poison_count, poison_count
from .safety import BLOCKLIST_VERSION, check_text

DATA_FILES = ("documents.jsonl", "queries.jsonl", "evidence.jsonl", "poison_records.jsonl", "audit_samples.jsonl")
MANIFEST_FILE = "manifest.json"
AUDIT_SAMPLE_SIZE = 20
EVIDENCE_KIND_ORDER = {"short_answer": 0, "long_answer": 1, "supporting_sentence": 2}


def manifest_sha256(value: dict) -> str:
    """Hash a manifest the same way :func:`write_json` serializes it."""

    import hashlib

    return hashlib.sha256((canonical_json(value) + "\n").encode("utf-8")).hexdigest()


def _code_hash() -> str:
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    return digest([[path.name, file_sha256(path)] for path in files])


def _git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _source_ref(source_doc: dict, query: dict) -> dict:
    for ref in source_doc.get("source_refs", []):
        if ref.get("source_query_id") == query["source_query_id"]:
            return {"raw_file": ref["raw_file"], "source_query_id": ref["source_query_id"]}
    refs = source_doc.get("source_refs") or []
    if refs:
        return {"raw_file": refs[0]["raw_file"], "source_query_id": query["source_query_id"]}
    raise DataError(f"source document {source_doc.get('doc_id')!r} has no source_refs")


def _evidence_items(items: Iterable[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (EVIDENCE_KIND_ORDER.get(item.get("kind"), 9), item["evidence_id"]),
    )


def _decoy_pool(queries: Iterable[dict]) -> list[dict]:
    pool: list[dict] = []
    for query in queries:
        for value in query.get("ground_truth", []):
            pool.append({"value": normalize_inline(str(value)), "source_query_id": query["query_id"]})
    return pool


def _decoys_for(query: dict, pool: list[dict], seed: int, dataset: str) -> list[dict]:
    candidates = [item for item in pool if item["source_query_id"] != query["query_id"]]
    candidates.sort(
        key=lambda item: (
            rank(seed, dataset, f"{query['query_id']}|{item['source_query_id']}|{item['value']}"),
            item["value"],
        )
    )
    return candidates


def _build_poison_document(
    settings: PoisonSettings,
    dataset: str,
    schema_version: str,
    query: dict,
    source_doc: dict,
    candidate: CorruptionCandidate,
) -> tuple[dict, dict]:
    true_fact = candidate.true_fact
    false_fact = candidate.false_fact
    doc_id = poison_doc_id(
        dataset,
        settings.clean_data_version,
        settings.poison_version,
        query["query_id"],
        source_doc["doc_id"],
        settings.operator,
        settings.operator_version,
        candidate.variant_index,
        true_fact,
        false_fact,
        candidate.text,
    )
    poison = {
        "doc_id": doc_id,
        "dataset": dataset,
        "title": candidate.title,
        "text": candidate.text,
        "source_doc_id": source_doc["doc_id"],
        "source_refs": [_source_ref(source_doc, query)],
        "text_sha256": text_sha256(candidate.text),
        "document_role": "poison",
        "is_poison": True,
        "corruption_type": settings.operator,
        "true_fact": true_fact,
        "false_fact": false_fact,
        "target_query_ids": [query["query_id"]],
        "poison_version": settings.poison_version,
        "data_version": settings.clean_data_version,
        "schema_version": schema_version,
        "corruption_operator_version": settings.operator_version,
        "construction_mode": candidate.construction_mode,
        "variant_index": candidate.variant_index,
        "review_status": settings.review_status_on_generate,
    }
    record = {
        "poison_doc_id": doc_id,
        "dataset": dataset,
        "target_query_id": query["query_id"],
        "source_doc_id": source_doc["doc_id"],
        "source_evidence_ids": sorted(item["evidence_id"] for item in query["_evidence_items"]),
        "corruption_type": settings.operator,
        "operator_version": settings.operator_version,
        "variant_index": candidate.variant_index,
        "true_fact": {
            "value": true_fact,
            "answer_keys": [normalize_inline(str(value)).casefold() for value in query["ground_truth"]],
        },
        "false_fact": {"value": false_fact, "selection_source": candidate.selection_source},
        "replacement_count": candidate.replacement_count,
        "construction_mode": candidate.construction_mode,
        "contains_true_fact_after": False,
        "explicit_instruction_hits": [hit.to_dict() for hit in candidate.safety_hits],
        "text_sha256": poison["text_sha256"],
        "review_status": settings.review_status_on_generate,
        "review_notes": None,
    }
    return poison, record


def _audit_samples(
    records: list[dict],
    documents: dict[str, dict],
    queries: dict[str, dict],
    evidence_by_query: dict[str, list[dict]],
    *,
    settings: PoisonSettings,
    dataset: str,
    ratio: float,
    schema_version: str,
) -> list[dict]:
    ordered = sorted(
        records,
        key=lambda record: rank(settings.seed, dataset, f"audit|{record['poison_doc_id']}"),
    )[:AUDIT_SAMPLE_SIZE]
    samples = []
    for record in sorted(ordered, key=lambda record: record["poison_doc_id"]):
        source = documents[record["source_doc_id"]]
        poison = documents[record["poison_doc_id"]]
        samples.append(
            {
                "dataset": dataset,
                "poison_version": settings.poison_version,
                "schema_version": schema_version,
                "poison_ratio_target": ratio,
                "query": queries[record["target_query_id"]],
                "source_document": {
                    "doc_id": source["doc_id"],
                    "title": source["title"],
                    "text_sha256": source["text_sha256"],
                },
                "source_evidence": evidence_by_query[record["target_query_id"]],
                "poison_document": {
                    "doc_id": poison["doc_id"],
                    "title": poison["title"],
                    "text": poison["text"],
                },
                "true_fact": record["true_fact"],
                "false_fact": record["false_fact"],
                "operator": record["corruption_type"],
                "operator_version": record["operator_version"],
                "construction_mode": record["construction_mode"],
                "variant_index": record["variant_index"],
                "explicit_instruction_hits": record["explicit_instruction_hits"],
                "automatic_checks": {
                    "false_fact_present": True,
                    "true_fact_absent": True,
                    "blocklist_clean": True,
                    "hash_matches": True,
                },
                "review_status": settings.review_status_on_generate,
            }
        )
    return samples


def build_poison_dataset(
    clean_dir: str | Path,
    output_dir: str | Path,
    settings: PoisonSettings,
    *,
    ratio: float,
    dataset: str | None = None,
) -> dict:
    clean_dir = Path(clean_dir)
    output_dir = Path(output_dir)
    clean_report = validate_dataset(clean_dir)
    clean_manifest = json.loads((clean_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = dataset or clean_manifest.get("dataset") or clean_dir.name
    if dataset not in DATASETS:
        raise DataError(f"unknown dataset: {dataset!r}")
    schema_version = clean_manifest.get("config", {}).get("schema_version")
    if schema_version not in ("data-v1", "data-v2"):
        raise DataError(f"unsupported clean schema_version: {schema_version!r}")

    documents = load_documents(clean_dir / "documents.jsonl")
    queries = load_queries(clean_dir / "queries.jsonl")
    evidence = load_evidence(clean_dir / "evidence.jsonl")
    docs_by_id = {doc["doc_id"]: doc for doc in documents}
    queries_by_id = {query["query_id"]: query for query in queries}
    evidence_by_query: dict[str, list[dict]] = {query["query_id"]: [] for query in queries}
    for item in evidence:
        evidence_by_query[item["query_id"]].append(item)
    for query in queries:
        if not query["ground_truth"]:
            raise DataError(f"query {query['query_id']!r} has no ground truth")
        if not query["evidence_doc_ids"]:
            raise DataError(f"query {query['query_id']!r} has no evidence document")
        for doc_id in query["evidence_doc_ids"]:
            if doc_id not in docs_by_id:
                raise DataError(f"query {query['query_id']!r} references missing document {doc_id!r}")

    clean_count = len(documents)
    if settings.query_allocation == "fixed_per_query":
        poison_docs_per_query = settings.poison_budget(ratio)
        poison_total = fixed_poison_count(len(queries_by_id), poison_docs_per_query)
        budgets = allocate_fixed(sorted(queries_by_id), poison_docs_per_query)
    else:
        poison_docs_per_query = None
        poison_total = poison_count(clean_count, ratio, settings.ratio_denominator)
        budgets = (
            allocate_variants(sorted(queries_by_id), poison_total, seed=settings.seed, dataset=dataset)
            if poison_total
            else {query_id: 0 for query_id in queries_by_id}
        )
    poison_actual = actual_ratio(clean_count, poison_total)
    poison_clean_ratio = poison_total / clean_count if clean_count else 0.0
    if settings.query_allocation != "fixed_per_query" and poison_total and poison_total < len(queries):
        raise DataError(
            f"{dataset}: ratio {ratio} yields {poison_total} poison documents but "
            f"{len(queries)} queries require at least one each; raise the ratio or add documents"
        )
    operator = EntitySubstitutionOperator()
    if operator.name != settings.operator or operator.version != settings.operator_version:
        raise DataError("configured operator does not match the implemented operator")
    pool = _decoy_pool(queries)

    poison_documents: list[dict] = []
    poison_records: list[dict] = []
    counts = {
        "answer_replacement": 0,
        "declarative_append": 0,
        "blocklist_rejected": 0,
        "true_fact_remaining_rejected": 0,
        "duplicate_rejected": 0,
    }
    used_doc_ids: set[str] = set(docs_by_id)
    used_texts: set[str] = {doc["text"] for doc in documents}
    poison_ids_by_query: dict[str, list[str]] = {query_id: [] for query_id in queries_by_id}

    for query_id in sorted(queries_by_id):
        variant_count = budgets[query_id]
        if not variant_count:
            continue
        query = queries_by_id[query_id]
        source_doc = docs_by_id[sorted(query["evidence_doc_ids"])[0]]
        items = _evidence_items(evidence_by_query[query_id])
        if not items:
            raise DataError(f"query {query_id!r} has no evidence records")
        enriched = dict(query, _evidence_items=items)
        candidates = operator.build_candidates(
            enriched,
            source_doc,
            items,
            _decoys_for(query, pool, settings.seed, dataset),
            variant_count=variant_count,
            seed=settings.seed,
            max_source_chars=settings.max_source_chars,
            max_poison_chars=settings.max_poison_chars,
        )
        for candidate in candidates:
            if check_text(candidate.text):
                counts["blocklist_rejected"] += 1
                raise DataError(f"query {query_id!r} variant {candidate.variant_index} hits the blocklist")
            poison_doc, record = _build_poison_document(
                settings, dataset, schema_version, enriched, source_doc, candidate
            )
            if poison_doc["doc_id"] in used_doc_ids:
                counts["duplicate_rejected"] += 1
                raise DataError(f"duplicate poison doc_id for query {query_id!r}")
            if poison_doc["text"] in used_texts:
                counts["duplicate_rejected"] += 1
                raise DataError(f"duplicate poison text for query {query_id!r}")
            used_doc_ids.add(poison_doc["doc_id"])
            used_texts.add(poison_doc["text"])
            poison_documents.append(poison_doc)
            poison_records.append(record)
            poison_ids_by_query[query_id].append(poison_doc["doc_id"])
            counts[candidate.construction_mode] += 1

    poison_documents.sort(key=lambda doc: doc["doc_id"])
    poison_records.sort(key=lambda record: (record["target_query_id"], record["variant_index"], record["poison_doc_id"]))
    ordered_documents = sorted(documents, key=lambda doc: doc["doc_id"]) + poison_documents
    ordered_queries = []
    for query in sorted(queries, key=lambda item: item["query_id"]):
        ordered_queries.append(
            {
                **query,
                "poison_doc_ids": sorted(poison_ids_by_query[query["query_id"]]),
                "poison_docs_per_query": len(poison_ids_by_query[query["query_id"]]),
                "poison_question_coverage": 1.0 if poison_total else None,
                "poison_ratio_target": ratio,
                "poison_ratio_actual": poison_actual,
                "poison_ratio_clean": poison_clean_ratio,
                "poison_version": settings.poison_version,
            }
        )
    ordered_evidence = sorted(evidence, key=lambda item: item["evidence_id"])

    covered = sorted(query_id for query_id, ids in poison_ids_by_query.items() if ids)
    uncovered = sorted(query_id for query_id, ids in poison_ids_by_query.items() if poison_total and not ids)
    coverage = len(covered) / len(queries_by_id) if poison_total else None
    if poison_total and (coverage != 1.0 or uncovered):
        raise DataError(f"{dataset}: poison coverage is {coverage}, not 1.0")

    audit_documents = {doc["doc_id"]: doc for doc in ordered_documents}
    ordered_query_map = {query["query_id"]: query for query in ordered_queries}
    audits = _audit_samples(
        poison_records,
        audit_documents,
        ordered_query_map,
        evidence_by_query,
        settings=settings,
        dataset=dataset,
        ratio=ratio,
        schema_version=schema_version,
    )

    clean_manifest_hash = file_sha256(clean_dir / "manifest.json")
    fingerprint = digest(
        [
            clean_manifest_hash,
            settings.contract(),
            dataset,
            ratio,
            _code_hash(),
            _git_revision(settings.root),
        ]
    )

    manifest = {
        "status": "validated",
        "dataset": dataset,
        "clean_data_version": settings.clean_data_version,
        "clean_manifest_sha256": clean_manifest_hash,
        "poison_version": settings.poison_version,
        "schema_version": schema_version,
        "operator": settings.operator,
        "operator_version": settings.operator_version,
        "template_version": settings.template_version,
        "blocklist_version": BLOCKLIST_VERSION,
        "seed": settings.seed,
        "poison_ratio_target": ratio,
        "poison_ratio_actual": poison_actual,
        "poison_ratio_clean": poison_clean_ratio,
        "ratio_denominator": settings.ratio_denominator,
        "query_allocation": settings.query_allocation,
        "poison_docs_per_query": poison_docs_per_query,
        "clean_document_count": clean_count,
        "poison_document_count": poison_total,
        "final_document_count": clean_count + poison_total,
        "query_count": len(queries_by_id),
        "poison_question_coverage": coverage,
        "coverage_status": "verified" if poison_total else "baseline_no_poison",
        "coverage_definition": "queries with at least one poison document / all selected queries",
        "selected_query_coverage": 1.0,
        "covered_query_ids": covered,
        "uncovered_query_ids": uncovered,
        "audit_sample_count": len(audits),
        "input_fingerprint": fingerprint,
        "output_sha256": {},
        "counts": counts,
        "validation": {},
        "human_review_status": settings.review_status_on_generate,
    }

    staging = output_dir.with_name(f".staging-{output_dir.name}")
    if output_dir.exists():
        raise DataError(f"refusing to overwrite an existing poison dataset: {output_dir}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir(parents=True)
        write_jsonl(staging / "documents.jsonl", ordered_documents)
        write_jsonl(staging / "queries.jsonl", ordered_queries)
        write_jsonl(staging / "evidence.jsonl", ordered_evidence)
        write_jsonl(staging / "poison_records.jsonl", poison_records)
        write_jsonl(staging / "audit_samples.jsonl", audits)
        manifest["output_sha256"] = {name: file_sha256(staging / name) for name in DATA_FILES}
        write_json(staging / MANIFEST_FILE, manifest)

        from .validation import validate_poison_dataset

        report = validate_poison_dataset(staging, expected_clean_manifest=clean_manifest)
        manifest["validation"] = {
            key: value
            for key, value in report.items()
            if key in ("blocklist_hit_count", "duplicate_text_count", "status", "poison_question_coverage")
        }
        write_json(staging / MANIFEST_FILE, manifest)
        report = validate_poison_dataset(staging, expected_clean_manifest=clean_manifest)
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return report
