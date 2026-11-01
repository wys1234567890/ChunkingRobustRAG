"""Stage 2 implicit fact-corruption dataset construction."""

from __future__ import annotations

from .builder import build_poison_dataset
from .config import DATASETS, PoisonSettings, load_poison_settings
from .operators import CorruptionCandidate, EntitySubstitutionOperator
from .ratios import actual_ratio, allocate_fixed, allocate_variants, fixed_poison_count, poison_count
from .validation import load_poison_documents, load_poison_records, validate_poison_dataset

__all__ = [
    "DATASETS",
    "PoisonSettings",
    "load_poison_settings",
    "build_poison_dataset",
    "CorruptionCandidate",
    "EntitySubstitutionOperator",
    "poison_count",
    "fixed_poison_count",
    "allocate_fixed",
    "actual_ratio",
    "allocate_variants",
    "load_poison_documents",
    "load_poison_records",
    "validate_poison_dataset",
]
