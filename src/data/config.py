"""Independent, credential-free configuration for Stage 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .common import DataError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("nq", "hotpotqa")


@dataclass(frozen=True)
class DataSettings:
    seed: int
    data_version: str
    schema_version: str
    query_count: int
    document_count: int
    audit_count: int
    max_answer_words: int
    processed_root: str
    sources: dict[str, dict[str, Any]]
    # When enabled, each selected query receives this many lexical hard
    # negatives from the complete source-document pool.  Zero preserves the
    # original Stage 1 corpus construction behaviour.
    hard_negative_count: int = 0
    root: Path = REPOSITORY_ROOT

    def resolve(self, path: str) -> Path:
        return (self.root / path).resolve()

    def output_dir(self, dataset: str) -> Path:
        return self.resolve(self.processed_root) / self.data_version / dataset

    def contract(self, dataset: str) -> dict:
        result = asdict(self)
        result.pop("root")
        # Output location is not an input to the scientific data selection.
        result.pop("processed_root")
        result["sources"] = {dataset: self.sources[dataset]}
        return result


def load_data_settings(
    path: str | Path | None = None, *, root: Path | None = None
) -> DataSettings:
    base = (root or REPOSITORY_ROOT).resolve()
    config_path = Path(path) if path is not None else base / "configs/data.yaml"
    try:
        values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DataError(f"cannot read data configuration: {exc}") from exc
    if not isinstance(values, dict):
        raise DataError("data configuration must be a mapping")
    try:
        settings = DataSettings(**values, root=base)
    except TypeError as exc:
        raise DataError(f"invalid data configuration: {exc}") from exc
    for field in ("seed", "query_count", "document_count", "audit_count", "max_answer_words", "hard_negative_count"):
        value = getattr(settings, field)
        minimum = 0 if field in {"seed", "hard_negative_count"} else 1
        if type(value) is not int or value < minimum:
            raise DataError(f"{field} must be an integer >= {minimum}")
    if settings.audit_count > settings.query_count:
        raise DataError("audit_count must not exceed query_count")
    for field in ("data_version", "schema_version"):
        value = getattr(settings, field)
        if not isinstance(value, str) or not value or not all(c.isascii() and (c.isalnum() or c in "-_.") for c in value) or value in {".", ".."}:
            raise DataError(f"{field} must be a safe version name")
    if settings.schema_version != "data-v1":
        raise DataError("only schema_version=data-v1 is supported")
    if not isinstance(settings.processed_root, str) or not settings.processed_root.strip():
        raise DataError("processed_root must not be empty")
    if not isinstance(settings.sources, dict):
        raise DataError("sources must be a mapping")
    if not settings.sources:
        raise DataError("sources must not be empty")
    for dataset, source in settings.sources.items():
        if dataset not in DATASETS or not isinstance(source, dict):
            raise DataError(f"invalid dataset source: {dataset}")
        if not all(isinstance(source.get(key), str) and source[key] for key in ("version", "license")):
            raise DataError(f"{dataset}: version and license are required")
        expected = source.get("expected_records")
        if expected is not None and (type(expected) is not int or expected < 1):
            raise DataError(f"{dataset}: expected_records must be a positive integer")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise DataError(f"{dataset}: source files must not be empty")
        paths = []
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
                raise DataError(f"{dataset}: each source requires a local path")
            paths.append(settings.resolve(item["path"]))
            if "url" in item and (not isinstance(item["url"], str) or not item["url"].startswith(("https://", "http://"))):
                raise DataError(f"{dataset}: invalid source URL")
            checksum = item.get("sha256")
            if checksum is not None and (not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum)):
                raise DataError(f"{dataset}: invalid expected sha256")
        if len(paths) != len(set(paths)):
            raise DataError(f"{dataset}: duplicate source paths")
        if len({p.name for p in paths}) != len(paths):
            raise DataError(f"{dataset}: source filenames must be unique for provenance")
    return settings
