"""Stable identifiers shared by the poison builder and validator."""

from __future__ import annotations

import hashlib

from ..data.common import digest


def poison_doc_id(
    dataset: str,
    clean_data_version: str,
    poison_version: str,
    target_query_id: str,
    source_doc_id: str,
    corruption_type: str,
    operator_version: str,
    variant_index: int,
    true_fact: str,
    false_fact: str,
    text: str,
) -> str:
    return f"{dataset}-poison-" + digest(
        [
            dataset,
            clean_data_version,
            poison_version,
            target_query_id,
            source_doc_id,
            corruption_type,
            operator_version,
            variant_index,
            true_fact,
            false_fact,
            text,
        ]
    )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
