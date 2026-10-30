"""Acquire official source files atomically and iterate their original records."""

from __future__ import annotations

import gzip
import http.client
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

from .common import DataError, file_sha256
from .config import DataSettings
from .parquet_sources import iter_parquet


def iter_source(path: Path, dataset: str, *, compressed: bool | None = None, parquet: bool = False) -> Iterator[dict]:
    try:
        if parquet or path.suffix == ".parquet":
            yield from iter_parquet(path, dataset)
        elif dataset == "nq":
            opener = gzip.open if (compressed if compressed is not None else path.suffix == ".gz") else open
            with opener(path, "rt", encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    value = json.loads(line)
                    if not isinstance(value, dict) or not {"example_id", "question_text", "document_tokens", "annotations"} <= value.keys():
                        raise DataError(f"{path.name}:{number}: not an original NQ record")
                    yield value
        else:
            with path.open(encoding="utf-8") as handle:
                values = json.load(handle)
            if not isinstance(values, list):
                raise DataError(f"{path.name}: HotpotQA must be a JSON array")
            for value in values:
                if not isinstance(value, dict) or not {"_id", "question", "answer", "context", "supporting_facts"} <= value.keys():
                    raise DataError(f"{path.name}: not an original labeled HotpotQA record")
                yield value
    except (OSError, EOFError, UnicodeError, ValueError) as exc:
        raise DataError(f"invalid source {path.name}: {exc}") from exc


def _verify_download(path: Path, dataset: str, compressed: bool, *, parquet: bool = False) -> None:
    count = sum(1 for _ in iter_source(path, dataset, compressed=compressed, parquet=parquet))
    if count == 0:
        raise DataError(f"empty source: {path.name}")


def download_source(
    url: str, path: Path, dataset: str, *, expected_sha256: str | None = None,
    progress: Callable[[str], None] = lambda message: None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each attempt has a unique partial file, never a reusable successful filename.
    for attempt in range(3):
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
        part = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as output:
                request = urllib.request.Request(url, headers={"User-Agent": "ChunkingRobustRAG/clean-v1"})
                progress(f"downloading {path.name} (attempt {attempt + 1})")
                with urllib.request.urlopen(request, timeout=60) as response:
                    total = 0
                    reported = 0
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
                        total += len(block)
                        if total - reported >= 32 * 1024 * 1024:
                            progress(f"{path.name}: {total // (1024 * 1024)} MiB")
                            reported = total
            if expected_sha256 and file_sha256(part) != expected_sha256:
                raise DataError(f"checksum mismatch for downloaded {path.name}")
            progress(f"checking download {path.name}")
            _verify_download(part, dataset, path.suffix == ".gz", parquet=path.suffix == ".parquet")
            part.replace(path)
            return
        except (OSError, urllib.error.URLError, http.client.HTTPException, DataError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 and exc.code not in (408, 429):
                hint = " Try --config configs/data.huggingface.yaml or provide local source files." if exc.code in (401, 403, 404, 410) else ""
                raise DataError(f"download refused for {path.name}: HTTP {exc.code} {exc.reason}; retrying the same URL will not help.{hint}") from exc
            if attempt == 2:
                raise DataError(f"download failed for {path.name}: {exc}") from exc
            progress(f"retrying {path.name}: {type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt)
        finally:
            part.unlink(missing_ok=True)


def acquire_sources(
    settings: DataSettings, dataset: str, *, offline: bool,
    progress: Callable[[str], None] = lambda message: None,
) -> list[dict]:
    if dataset not in settings.sources:
        raise DataError(f"no source configuration for {dataset}")
    result = []
    source = settings.sources[dataset]
    for item in sorted(source["files"], key=lambda value: value["path"]):
        path = settings.resolve(item["path"])
        if not path.is_file():
            if offline:
                raise DataError(f"offline mode: source file missing: {path}")
            if not item.get("url"):
                raise DataError(f"no download URL for missing local source: {path}")
            download_source(item["url"], path, dataset, expected_sha256=item.get("sha256"), progress=progress)
        checksum = file_sha256(path)
        if item.get("sha256") and checksum != item["sha256"]:
            raise DataError(f"checksum mismatch for local source: {path}")
        result.append({
            "path": item["path"], "filename": path.name, "url": item.get("url"),
            "version": source["version"], "license": source["license"],
            "size_bytes": path.stat().st_size, "sha256": checksum,
        })
    return result
