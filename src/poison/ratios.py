"""Decimal poison-ratio arithmetic and stable query allocation."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

from ..data.common import DataError, rank


def poison_count(
    clean_document_count: int,
    ratio: float,
    denominator: str = "clean_documents",
) -> int:
    """Return the poison document count for one target ratio.

    ``round_half_up`` is implemented with :class:`decimal.Decimal` so that
    binary float boundaries never change the published corpus size.
    """

    if clean_document_count < 0:
        raise DataError("clean document count must be non-negative")
    value = Decimal(str(ratio))
    if value < 0 or value >= 1:
        raise DataError("ratio must satisfy 0 <= ratio < 1")
    documents = Decimal(clean_document_count)
    if denominator == "final_corpus_documents":
        raw = value * documents / (Decimal(1) - value)
    elif denominator == "clean_documents":
        raw = value * documents
    else:
        raise DataError(f"unknown ratio denominator: {denominator!r}")
    return int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fixed_poison_count(query_count: int, poison_docs_per_query: int) -> int:
    """Return the exact poison-document count for a fixed query budget."""

    if query_count < 0:
        raise DataError("query count must be non-negative")
    if type(poison_docs_per_query) is not int or poison_docs_per_query < 0:
        raise DataError("poison_docs_per_query must be a non-negative integer")
    return query_count * poison_docs_per_query


def allocate_fixed(query_ids: Iterable[str], poison_docs_per_query: int) -> dict[str, int]:
    """Assign the same number of poison documents to every query."""

    ordered = sorted(str(query_id) for query_id in query_ids)
    if type(poison_docs_per_query) is not int or poison_docs_per_query < 0:
        raise DataError("poison_docs_per_query must be a non-negative integer")
    return {query_id: poison_docs_per_query for query_id in ordered}


def actual_ratio(clean_document_count: int, poison_document_count: int) -> float:
    total = clean_document_count + poison_document_count
    if total == 0:
        return 0.0
    return poison_document_count / total


def distinct_ratio_labels(ratios: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in ratios)
    if len(values) != len(set(values)):
        raise DataError("ratios must not contain duplicates")
    return values


def allocate_variants(
    query_ids: Iterable[str],
    poison_total: int,
    *,
    seed: int,
    dataset: str,
) -> dict[str, int]:
    """Distribute poison documents evenly across queries by stable rank."""

    ordered = sorted((str(query_id) for query_id in query_ids), key=lambda qid: rank(seed, dataset, qid))
    if not ordered:
        raise DataError("cannot allocate poison documents without queries")
    if poison_total < len(ordered):
        raise DataError(
            f"{dataset}: {poison_total} poison documents cannot cover {len(ordered)} queries; "
            f"at least {len(ordered)} are required"
        )
    base, extra = divmod(poison_total, len(ordered))
    budgets = {query_id: base for query_id in ordered}
    for query_id in ordered[:extra]:
        budgets[query_id] += 1
    return budgets
