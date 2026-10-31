"""Configuration for Stage 2 implicit fact-corruption datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ..data.common import DataError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("nq", "hotpotqa")
OPERATORS = ("entity_substitution",)
DENOMINATORS = ("final_corpus_documents", "clean_documents")
ALLOCATIONS = ("balanced_stable_hash",)


@dataclass(frozen=True)
class PoisonSettings:
    clean_data_version: str
    clean_processed_root: str
    poisoned_root: str
    poison_version: str
    operator: str
    operator_version: str
    template_version: str
    seed: int
    ratios: tuple[float, ...]
    ratio_denominator: str
    question_coverage_required: float
    query_allocation: str
    max_source_chars: int
    max_poison_chars: int
    blocklist_version: str
    review_status_on_generate: str
    root: Path = REPOSITORY_ROOT

    def resolve(self, path: str) -> Path:
        return (self.root / path).resolve()

    def clean_dir(self, dataset: str) -> Path:
        return self.resolve(self.clean_processed_root) / self.clean_data_version / dataset

    def output_dir(self, dataset: str, ratio: float) -> Path:
        return self.resolve(self.poisoned_root) / self.poison_version / dataset / f"ratio-{ratio:.2f}".replace(".", "p")

    def contract(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("root")
        result["ratios"] = [round(value, 8) for value in self.ratios]
        return result


def _safe_version(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataError(f"{field} must not be empty")
    if not all(char.isascii() and (char.isalnum() or char in "-_.") for char in value):
        raise DataError(f"{field} must be a safe ASCII version name")
    if value in {".", ".."}:
        raise DataError(f"{field} is invalid")
    return value


def load_poison_settings(
    path: str | Path | None = None,
    *,
    root: Path | None = None,
    ratios: list[float] | None = None,
) -> PoisonSettings:
    base = (root or REPOSITORY_ROOT).resolve()
    config_path = Path(path) if path is not None else base / "configs/corruption.yaml"
    try:
        values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DataError(f"cannot read corruption configuration: {exc}") from exc
    if not isinstance(values, dict):
        raise DataError("corruption configuration must be a mapping")
    if ratios is not None:
        values = dict(values, ratios=ratios)
    try:
        settings = PoisonSettings(**values, root=base)
    except TypeError as exc:
        raise DataError(f"invalid corruption configuration: {exc}") from exc

    for field in (
        "clean_data_version",
        "poison_version",
        "operator_version",
        "template_version",
        "blocklist_version",
    ):
        _safe_version(getattr(settings, field), field)
    for field in ("clean_processed_root", "poisoned_root"):
        if not isinstance(getattr(settings, field), str) or not getattr(settings, field).strip():
            raise DataError(f"{field} must not be empty")
    if settings.operator not in OPERATORS:
        raise DataError(f"unknown operator: {settings.operator!r}")
    if settings.ratio_denominator not in DENOMINATORS:
        raise DataError(f"unknown ratio_denominator: {settings.ratio_denominator!r}")
    if settings.query_allocation not in ALLOCATIONS:
        raise DataError(f"unknown query_allocation: {settings.query_allocation!r}")
    if type(settings.seed) is not int or settings.seed < 0:
        raise DataError("seed must be a non-negative integer")
    if settings.question_coverage_required != 1.0:
        raise DataError("question_coverage_required must be exactly 1.0")
    if settings.max_source_chars <= 0 or settings.max_poison_chars <= 0:
        raise DataError("max character limits must be positive")
    if settings.review_status_on_generate != "pending":
        raise DataError("newly generated samples must have review_status=pending")

    if not isinstance(settings.ratios, (list, tuple)) or not settings.ratios:
        raise DataError("ratios must be a non-empty list")
    normalized: list[float] = []
    for value in settings.ratios:
        if type(value) not in (int, float):
            raise DataError("each ratio must be a number")
        if not 0.0 <= float(value) < 1.0:
            raise DataError("each ratio must satisfy 0 <= ratio < 1")
        if float(value) in normalized:
            raise DataError("ratios must not contain duplicates")
        normalized.append(float(value))
    object.__setattr__(settings, "ratios", tuple(normalized))
    return settings
