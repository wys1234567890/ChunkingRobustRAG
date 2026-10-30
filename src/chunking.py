"""Deterministic Stage 3 chunkers with character-preserving offsets.

All chunkers operate on one document at a time and return ``Chunk`` records whose
``text`` is exactly ``document.text[start:end]``. ``chunk_size`` is measured with
the configured tokenizer; ``overlap`` is a fraction of ``chunk_size``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .data.common import canonical_json, read_jsonl, write_jsonl

STRATEGIES = ("fixed", "semantic", "hierarchical")
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 0.0
DEFAULT_TOKENIZER = "hf:intfloat/multilingual-e5-small"
CHUNK_CONFIG_VERSION = "stage3-v1"

_WHITESPACE = re.compile(r"\S+")
_HEADING = re.compile(r"(?m)^(#{1,6})\s+\S.*$")
_PARAGRAPH = re.compile(r"(?m)(?:[^\n]+\n?)+")
_TERMINATORS = ".!?。！？"


class ChunkingError(ValueError):
    """Invalid chunking input or configuration."""


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer interface used for chunk-size accounting."""

    name: str

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Return left-closed, right-open character spans for tokens."""


class WhitespaceTokenizer:
    """Deterministic fallback tokenizer: one token per non-whitespace run."""

    name = "whitespace-v1"

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in _WHITESPACE.finditer(text)]


class HuggingFaceTokenizer:
    """Fast Hugging Face tokenizer with offset mapping (e.g. the local E5 tokenizer)."""

    def __init__(self, model_name: str, *, revision: str | None = None) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise ChunkingError("transformers is required for the huggingface tokenizer") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
        if not self._tokenizer.is_fast:
            raise ChunkingError(f"tokenizer {model_name!r} must provide offset mappings")
        self.name = f"hf:{model_name}" + (f"@{revision}" if revision else "")

    def spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        return [(int(start), int(end)) for start, end in encoded["offset_mapping"] if end > start]


def build_tokenizer(spec: str = DEFAULT_TOKENIZER, *, revision: str | None = None) -> Tokenizer:
    """Build a tokenizer from ``whitespace`` or ``hf:<model>``."""

    if spec in {"whitespace", "whitespace-v1"}:
        return WhitespaceTokenizer()
    if spec.startswith("hf:"):
        return HuggingFaceTokenizer(spec[3:], revision=revision)
    if "/" in spec:
        return HuggingFaceTokenizer(spec, revision=revision)
    raise ChunkingError(f"unknown tokenizer: {spec!r}")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    is_poison: bool
    strategy: str
    chunk_size: int
    overlap: float
    tokenizer: str
    token_count: int
    dataset: str = ""
    source_doc_id: str = ""
    poison_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "is_poison": self.is_poison,
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "tokenizer": self.tokenizer,
            "token_count": self.token_count,
            "dataset": self.dataset,
            "source_doc_id": self.source_doc_id,
            "poison_version": self.poison_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Chunk":
        poison = value.get("is_poison")
        if not isinstance(poison, bool):
            raise ChunkingError("chunk is_poison must be a boolean")
        try:
            chunk = cls(
                chunk_id=str(value["chunk_id"]),
                doc_id=str(value["doc_id"]),
                text=str(value["text"]),
                start=int(value["start"]),
                end=int(value["end"]),
                is_poison=poison,
                strategy=str(value.get("strategy", "unknown")),
                chunk_size=int(value.get("chunk_size", 0)),
                overlap=float(value.get("overlap", 0.0)),
                tokenizer=str(value.get("tokenizer", "unknown")),
                token_count=int(value.get("token_count", 0)),
                dataset=str(value.get("dataset", "")),
                source_doc_id=str(value.get("source_doc_id", value["doc_id"])),
                poison_version=value.get("poison_version"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ChunkingError(f"invalid chunk record: {exc}") from exc
        return chunk


def _value(document: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(document, Mapping):
        return document.get(key, default)
    return getattr(document, key, default)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _fit_span(
    text: str,
    start: int,
    end: int,
    chunk_size: int,
    tokenizer: Tokenizer,
) -> tuple[int, int]:
    """Shrink a sliced span until its own re-tokenization fits the budget."""

    while True:
        spans = tokenizer.spans(text[start:end])
        if len(spans) <= chunk_size:
            return start, end
        cut = start + spans[chunk_size - 1][1]
        new_start, new_end = _trim(text, start, cut)
        if new_end <= new_start or new_end >= end:
            return start, end
        start, end = new_start, new_end


def _validate_config(chunk_size: int, overlap: float) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ChunkingError("chunk_size must be a positive integer")
    if not isinstance(overlap, (int, float)) or isinstance(overlap, bool):
        raise ChunkingError("overlap must be numeric")
    if not 0 <= float(overlap) < 1:
        raise ChunkingError("overlap must satisfy 0 <= overlap < 1")


def _window(token_spans: Sequence[tuple[int, int]], chunk_size: int, overlap: float) -> list[tuple[int, int]]:
    if not token_spans:
        return []
    step = max(1, chunk_size - int(chunk_size * float(overlap)))
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(token_spans):
        end_index = min(index + chunk_size, len(token_spans))
        spans.append((token_spans[index][0], token_spans[end_index - 1][1]))
        if end_index >= len(token_spans):
            break
        index += step
    return spans


def _fixed_in_range(
    text: str,
    start: int,
    end: int,
    chunk_size: int,
    overlap: float,
    tokenizer: Tokenizer,
) -> list[tuple[int, int]]:
    local = tokenizer.spans(text[start:end])
    return [(start + a, start + b) for a, b in _window(local, chunk_size, overlap)]


def _fixed_spans(text: str, chunk_size: int, overlap: float, tokenizer: Tokenizer) -> list[tuple[int, int]]:
    return _fixed_in_range(text, 0, len(text), chunk_size, overlap, tokenizer)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split on sentence punctuation and line breaks without losing characters."""

    spans: list[tuple[int, int]] = []
    start: int | None = None
    length = len(text)
    for index, char in enumerate(text):
        if start is None:
            if not char.isspace():
                start = index
            else:
                continue
        if char in _TERMINATORS:
            cursor = index + 1
            while cursor < length and text[cursor] in _TERMINATORS:
                cursor += 1
            if cursor == length or text[cursor].isspace():
                a, b = _trim(text, start, cursor)
                if a < b:
                    spans.append((a, b))
                start = None
        elif char == "\n" and start is not None:
            a, b = _trim(text, start, index)
            if a < b:
                spans.append((a, b))
            start = None
    if start is not None:
        a, b = _trim(text, start, length)
        if a < b:
            spans.append((a, b))
    return spans


def _semantic_spans(text: str, chunk_size: int, overlap: float, tokenizer: Tokenizer) -> list[tuple[int, int]]:
    sentences = _sentence_spans(text)
    if not sentences:
        return []
    counts = [len(tokenizer.spans(text[a:b])) for a, b in sentences]
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(sentences):
        end_index = index
        total = 0
        while end_index < len(sentences):
            count = counts[end_index]
            if end_index > index and total + count > chunk_size:
                break
            total += count
            end_index += 1
            if count > chunk_size:
                break
        if end_index == index:
            end_index = index + 1
        if counts[index] > chunk_size and end_index == index + 1:
            spans.extend(
                _fixed_in_range(text, sentences[index][0], sentences[index][1], chunk_size, overlap, tokenizer)
            )
        else:
            spans.append((sentences[index][0], sentences[end_index - 1][1]))
        if end_index >= len(sentences):
            break
        if overlap <= 0:
            index = end_index
            continue
        target = chunk_size * float(overlap)
        cursor = end_index
        carried = 0
        while cursor > index + 1 and carried + counts[cursor - 1] <= target:
            cursor -= 1
            carried += counts[cursor]
        index = max(index + 1, cursor)
        if index >= end_index:
            index = end_index
    return spans


def _section_spans(text: str) -> list[tuple[int, int]]:
    headings = list(_HEADING.finditer(text))
    if headings:
        sections: list[tuple[int, int]] = []
        if text[: headings[0].start()].strip():
            sections.append((0, headings[0].start()))
        for position, heading in enumerate(headings):
            end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
            sections.append((heading.start(), end))
        return sections
    paragraphs = [match.span() for match in _PARAGRAPH.finditer(text) if match.group().strip()]
    return paragraphs or [(0, len(text))]


def _hierarchical_spans(text: str, chunk_size: int, overlap: float, tokenizer: Tokenizer) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start, end in _section_spans(text):
        spans.extend(_fixed_in_range(text, start, end, chunk_size, overlap, tokenizer))
    return spans


def _chunk_id(
    doc_id: str,
    strategy: str,
    chunk_size: int,
    overlap: float,
    tokenizer: str,
    start: int,
    end: int,
    text: str,
) -> str:
    identity = {
        "doc_id": doc_id,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": round(float(overlap), 6),
        "tokenizer": tokenizer,
        "start": start,
        "end": end,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return "chunk-" + hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:24]


def chunk_document(
    document: Mapping[str, Any] | Any,
    *,
    strategy: str = "fixed",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    tokenizer: Tokenizer | None = None,
    dataset: str | None = None,
) -> list[Chunk]:
    """Chunk one document, preserving its poison metadata and exact character spans."""

    if strategy not in STRATEGIES:
        raise ChunkingError(f"unknown strategy: {strategy!r}")
    _validate_config(chunk_size, overlap)

    doc_id = _value(document, "doc_id")
    text = _value(document, "text")
    is_poison = _value(document, "is_poison")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ChunkingError("document requires a non-empty doc_id")
    if not isinstance(text, str) or not text.strip():
        raise ChunkingError(f"document {doc_id!r} has empty text")
    if not isinstance(is_poison, bool):
        raise ChunkingError(f"document {doc_id!r} is_poison must be a boolean")

    tokenizer = tokenizer or build_tokenizer()
    if strategy == "fixed":
        raw_spans = _fixed_spans(text, chunk_size, overlap, tokenizer)
    elif strategy == "semantic":
        raw_spans = _semantic_spans(text, chunk_size, overlap, tokenizer)
    else:
        raw_spans = _hierarchical_spans(text, chunk_size, overlap, tokenizer)

    chunks: list[Chunk] = []
    seen: set[tuple[int, int]] = set()
    for raw_start, raw_end in raw_spans:
        start, end = _trim(text, raw_start, raw_end)
        start, end = _fit_span(text, start, end, chunk_size, tokenizer)
        if start >= end or (start, end) in seen:
            continue
        seen.add((start, end))
        chunk_text = text[start:end]
        count = len(tokenizer.spans(chunk_text)) or 1
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc_id, strategy, chunk_size, overlap, tokenizer.name, start, end, chunk_text),
                doc_id=doc_id,
                text=chunk_text,
                start=start,
                end=end,
                is_poison=is_poison,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=float(overlap),
                tokenizer=tokenizer.name,
                token_count=count,
                dataset=str(dataset if dataset is not None else _value(document, "dataset", "")),
                source_doc_id=str(_value(document, "source_doc_id", doc_id)),
                poison_version=_value(document, "poison_version"),
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Mapping[str, Any] | Any],
    *,
    strategy: str = "fixed",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Chunk a corpus in order, rejecting duplicate document IDs."""

    tokenizer = tokenizer or build_tokenizer()
    chunks: list[Chunk] = []
    seen_docs: set[str] = set()
    for document in documents:
        doc_id = _value(document, "doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ChunkingError("document requires a non-empty doc_id")
        if doc_id in seen_docs:
            raise ChunkingError(f"duplicate doc_id: {doc_id}")
        seen_docs.add(doc_id)
        chunks.extend(
            chunk_document(
                document,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                tokenizer=tokenizer,
            )
        )
    return chunks


def validate_chunks(chunks: Sequence[Chunk], documents: Iterable[Mapping[str, Any] | Any] | None = None) -> None:
    """Check uniqueness, poison flags, and character-span readback."""

    docs = {_value(document, "doc_id"): _value(document, "text") for document in documents} if documents else None
    ids: set[str] = set()
    spans: set[tuple[str, int, int]] = set()
    for chunk in chunks:
        if not chunk.text.strip():
            raise ChunkingError(f"empty chunk: {chunk.chunk_id}")
        if chunk.start < 0 or chunk.end <= chunk.start:
            raise ChunkingError(f"invalid span: {chunk.chunk_id}")
        if chunk.chunk_id in ids:
            raise ChunkingError(f"duplicate chunk_id: {chunk.chunk_id}")
        ids.add(chunk.chunk_id)
        key = (chunk.doc_id, chunk.start, chunk.end)
        if key in spans:
            raise ChunkingError(f"duplicate chunk span: {key}")
        spans.add(key)
        if docs is not None:
            if chunk.doc_id not in docs:
                raise ChunkingError(f"unknown doc_id: {chunk.doc_id}")
            text = docs[chunk.doc_id]
            if chunk.end > len(text) or text[chunk.start : chunk.end] != chunk.text:
                raise ChunkingError(f"span readback failed: {chunk.chunk_id}")


def save_chunks(chunks: Iterable[Chunk], path: str | Path) -> int:
    records = [chunk.to_dict() for chunk in chunks]
    write_jsonl(Path(path), records)
    return len(records)


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks = [Chunk.from_dict(record) for record in read_jsonl(path)]
    validate_chunks(chunks)
    return chunks
