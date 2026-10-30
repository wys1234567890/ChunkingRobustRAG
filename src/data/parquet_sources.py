"""Restore the original annotation shape from publisher-hosted HF Parquet data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .common import DataError


def _rows(value: object, label: str) -> list[dict]:
    """HF Sequence can be a list of structs or a struct of parallel lists."""
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    if isinstance(value, dict) and value and all(isinstance(items, list) for items in value.values()):
        sizes = {len(items) for items in value.values()}
        if len(sizes) == 1:
            return [{key: items[index] for key, items in value.items()} for index in range(next(iter(sizes)))]
    raise DataError(f"invalid Hugging Face {label}: expected records or equally-sized columns")


def restore_nq(row: dict) -> dict:
    try:
        document = row["document"]
        tokens = []
        for token in _rows(document["tokens"], "NQ tokens"):
            tokens.append({
                "token": token["token"], "html_token": token["is_html"],
                "start_byte": token["start_byte"], "end_byte": token["end_byte"],
            })
        annotations = []
        for annotation in _rows(row["annotations"], "NQ annotations"):
            answer = annotation["yes_no_answer"]
            if type(answer) is int and answer in (-1, 0, 1):
                answer = {-1: "NONE", 0: "NO", 1: "YES"}[answer]
            if answer not in ("NONE", "NO", "YES"):
                raise DataError("invalid Hugging Face NQ yes/no label")
            annotations.append({
                "annotation_id": str(annotation["id"]),
                "long_answer": annotation["long_answer"],
                "short_answers": _rows(annotation["short_answers"], "NQ short answers"),
                "yes_no_answer": answer,
            })
        return {
            "example_id": str(row["id"]), "question_text": row["question"]["text"],
            "document_title": document["title"], "document_url": document["url"],
            "document_html": document["html"], "document_tokens": tokens,
            "annotations": annotations,
        }
    except (KeyError, TypeError) as exc:
        raise DataError(f"Hugging Face NQ is missing original evidence fields: {exc}") from exc


def restore_hotpotqa(row: dict) -> dict:
    try:
        return {
            "_id": str(row["id"]), "question": row["question"], "answer": row["answer"],
            "type": row["type"], "level": row.get("level"),
            "context": [[item["title"], item["sentences"]] for item in _rows(row["context"], "HotpotQA context")],
            "supporting_facts": [[item["title"], item["sent_id"]] for item in _rows(row["supporting_facts"], "HotpotQA supporting facts")],
        }
    except (KeyError, TypeError) as exc:
        raise DataError(f"Hugging Face HotpotQA is missing evidence fields: {exc}") from exc


def iter_parquet(path: Path, dataset: str) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DataError("Parquet sources require pyarrow; install requirements.lock") from exc
    restore = restore_nq if dataset == "nq" else restore_hotpotqa
    # Full NQ HTML pages can be large; never materialize an entire shard.
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                yield restore(row)
