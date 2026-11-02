"""Offline answer scoring shared by no-RAG, clean RAG and poison RAG runs.

English normalization: lowercase, remove ASCII punctuation and a/an/the,
then collapse whitespace. Scores are fractions in [0, 1].
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Sequence

EVALUATION_VERSION = "answer-em-f1-v1"


def normalize_answer(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_answer(prediction: str, answers: Sequence[str]) -> dict[str, float]:
    if not isinstance(prediction, str):
        raise ValueError("model_output must be a string")
    if not isinstance(answers, (list, tuple)) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError("ground_truth must be a non-empty list of non-empty answer strings")
    predicted = normalize_answer(prediction)
    em = f1 = 0.0
    for answer in answers:
        gold = normalize_answer(answer)
        em = max(em, float(predicted == gold))
        p, g = predicted.split(), gold.split()
        if not p or not g:
            value = float(p == g)
        else:
            common = sum((Counter(p) & Counter(g)).values())
            value = 2 * common / (len(p) + len(g))
        f1 = max(f1, value)
    return {"em": em, "token_f1": f1}


def validate_answer_inputs(queries: Sequence[dict], records: Sequence[dict]) -> tuple[dict, dict]:
    """Validate the entire run before scoring or making any Judge requests."""
    if not queries:
        raise ValueError("query set is empty")
    gold: dict[str, dict] = {}
    for q in queries:
        qid = q.get("query_id")
        if not isinstance(qid, str) or not qid or qid in gold:
            raise ValueError(f"invalid or duplicate query_id in query set: {qid!r}")
        score_answer("", q.get("ground_truth"))
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            raise ValueError(f"invalid question: {qid}")
        gold[qid] = q
    indexed: dict[str, dict] = {}
    for r in records:
        qid = r.get("query_id")
        if not isinstance(qid, str) or qid not in gold:
            raise ValueError(f"unknown result query_id: {qid!r}")
        if qid in indexed:
            raise ValueError(f"duplicate result query_id: {qid!r}")
        if r.get("question") != gold[qid].get("question"):
            raise ValueError(f"question mismatch: {qid}")
        if r.get("api_status") not in {"success", "error"}:
            raise ValueError(f"invalid api_status: {qid}")
        if r["api_status"] == "success" and not isinstance(r.get("model_output"), str):
            raise ValueError(f"model_output must be a string: {qid}")
        indexed[qid] = r
    # A single result file must describe a single experimental condition.
    conditions = {tuple(str(r.get(k)) for k in (
        "dataset", "api_model", "prompt_version", "config_version", "retriever",
        "retriever_version", "chunk_type", "chunk_size", "overlap", "top_k",
        "poison_ratio", "poison_version",
    )) for r in records}
    if len(conditions) > 1:
        raise ValueError("mixed experimental conditions; evaluate each run separately")
    return gold, indexed


def evaluate_answers(queries: Sequence[dict], records: Sequence[dict]) -> tuple[dict, list[dict]]:
    """Use the query set as denominator; missing and failed requests score zero."""
    gold, indexed = validate_answer_inputs(queries, records)
    details: list[dict[str, Any]] = []
    for qid, q in gold.items():
        r = indexed.get(qid)
        status = r["api_status"] if r else "missing"
        if r and r.get("error_code") == "truncated_response":
            status = "error"
        scores = score_answer(r.get("model_output"), q["ground_truth"]) if status == "success" else {"em": 0.0, "token_f1": 0.0}
        details.append({"query_id": qid, "question": q["question"], "ground_truth": q["ground_truth"],
                        "model_output": r.get("model_output", "") if r else "", "api_status": status, **scores})
    total = len(details)
    success = sum(d["api_status"] == "success" for d in details)
    summary = {
        "evaluation_version": EVALUATION_VERSION, "total_queries": total,
        "result_records": len(records), "successful_queries": success,
        "error_queries": sum(d["api_status"] == "error" for d in details),
        "missing_queries": total - len(records), "api_success_rate": success / total,
        "em": sum(d["em"] for d in details) / total,
        "token_f1": sum(d["token_f1"] for d in details) / total,
        "successful_only_em": sum(d["em"] for d in details) / success if success else None,
        "successful_only_token_f1": sum(d["token_f1"] for d in details) / success if success else None,
        "denominator": "all supplied queries; API errors and missing results score zero",
    }
    return summary, details
