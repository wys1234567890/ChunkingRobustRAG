"""Small JSON cache used to make API calls reproducible and restartable."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_cache_key(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt_version: str,
    config_version: str,
    request_options: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "messages": list(messages),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_version": prompt_version,
        "config_version": config_version,
    }
    if request_options:
        payload["request_options"] = dict(request_options)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class CacheStore:
    """Persist successful generation results without storing prompts or secrets."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("cache key must be a lowercase SHA-256 hex digest")
        return self.directory / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def save(self, key: str, value: Mapping[str, Any]) -> None:
        path = self._path(key)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def invalidate(self, key: str) -> None:
        """Remove a response that failed downstream validation."""
        self._path(key).unlink(missing_ok=True)

