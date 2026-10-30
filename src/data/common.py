"""Deterministic serialization, text normalization, and source identifiers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


class DataError(ValueError):
    """An invalid input or an incomplete/inconsistent prepared dataset."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def normalize_inline(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def answer_key(text: str) -> str:
    # Do not remove punctuation: e.g. 1.5 and 15 are different facts.
    return normalize_inline(text).casefold()


def normalize_body(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [normalize_inline(line) for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def rank(seed: int, dataset: str, identifier: str) -> str:
    return digest([seed, dataset, identifier])


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise DataError(f"{path}:{line_number}: expected an object")
                records.append(record)
    except (OSError, UnicodeError, ValueError) as exc:
        raise DataError(f"cannot load JSONL {path}: {exc}") from exc
    return records
