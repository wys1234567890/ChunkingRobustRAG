"""Unified BM25 and local E5 retrieval over Stage 3 chunks.

Both retrievers expose ``search(query, top_k=5)`` and return ``RetrievalHit``
records carrying the source chunk, score, rank, retriever name, and version.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .chunking import Chunk, ChunkingError

TOP_K = 5
DEFAULT_E5_MODEL = "intfloat/multilingual-e5-small"
BM25_TOKENIZER = "unicode-words-v1"
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "
_WORD = re.compile(r"\w+")
_LOGGER = logging.getLogger(__name__)


def _resolve_e5_device(device: str | None) -> str:
    """Select an accelerator explicitly; older Sentence Transformers misses XPU."""
    import torch

    xpu = getattr(torch, "xpu", None)
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if xpu is not None and xpu.is_available():
            return "xpu"
        from sentence_transformers.util import get_device_name

        return get_device_name()
    if device.split(":", 1)[0] == "xpu":
        if xpu is None or not xpu.is_available():
            raise RetrievalError(
                "Intel XPU is unavailable. Install the Intel graphics driver and "
                "XPU-enabled PyTorch (see requirements-xpu.txt), or use --device cpu."
            )
        try:
            index = torch.device(device).index
        except (RuntimeError, ValueError) as exc:
            raise RetrievalError(f"Invalid XPU device: {device}") from exc
        if index is not None and index >= xpu.device_count():
            raise RetrievalError(f"XPU device index out of range: {device}")
    return device


class RetrievalError(ValueError):
    """Invalid retrieval input or unavailable retriever dependency."""


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int
    retriever: str
    retriever_version: str

    def to_dict(self) -> dict[str, Any]:
        value = self.chunk.to_dict()
        value.update(
            {
                "score": self.score,
                "rank": self.rank,
                "retriever": self.retriever,
                "retriever_version": self.retriever_version,
            }
        )
        return value


class Retriever(Protocol):
    name: str
    version: str

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievalHit]:
        """Return the highest-scoring chunks for a query."""


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - only in broken environments
        return "unknown"


def _coerce_chunk(value: Chunk | Mapping[str, Any]) -> Chunk:
    if isinstance(value, Chunk):
        return value
    if isinstance(value, Mapping):
        try:
            return Chunk.from_dict(value)
        except ChunkingError as exc:
            raise RetrievalError(str(exc)) from exc
    raise RetrievalError(f"unsupported chunk type: {type(value).__name__}")


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalError("top_k must be a positive integer")
    return top_k


class BM25Retriever:
    """Okapi BM25 over chunk text with deterministic tie-breaking."""

    name = "bm25"

    def __init__(
        self,
        chunks: Iterable[Chunk | Mapping[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        tokenizer: str = BM25_TOKENIZER,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise RetrievalError("rank-bm25 is required for the BM25 retriever") from exc

        self._chunks = tuple(_coerce_chunk(chunk) for chunk in chunks)
        if not self._chunks:
            raise RetrievalError("BM25 retriever requires at least one chunk")
        self._tokenizer = tokenizer
        corpus = []
        for index, chunk in enumerate(self._chunks):
            tokens = _words(chunk.text)
            corpus.append(tokens or [f"__empty_{index}__"])
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b, epsilon=epsilon)
        self.version = (
            f"rank-bm25=={_package_version('rank-bm25')};k1={k1};b={b};"
            f"epsilon={epsilon};tokenizer={tokenizer}"
        )

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievalHit]:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query must be a non-empty string")
        tokens = _words(query)
        if not tokens:
            raise RetrievalError("query contains no searchable words")
        limit = _validate_top_k(top_k)
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=float)
        order = sorted(range(len(self._chunks)), key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id))
        return [
            RetrievalHit(
                chunk=self._chunks[index],
                score=float(scores[index]),
                rank=rank,
                retriever=self.name,
                retriever_version=self.version,
            )
            for rank, index in enumerate(order[:limit], 1)
        ]


class E5Retriever:
    """Local E5 dense retrieval over chunk text using cosine similarity."""

    name = "e5"

    def __init__(
        self,
        chunks: Iterable[Chunk | Mapping[str, Any]],
        *,
        model_name: str = DEFAULT_E5_MODEL,
        model: Any | None = None,
        batch_size: int = 32,
        device: str | None = None,
        query_prefix: str = _E5_QUERY_PREFIX,
        passage_prefix: str = _E5_PASSAGE_PREFIX,
    ) -> None:
        self._chunks = tuple(_coerce_chunk(chunk) for chunk in chunks)
        if not self._chunks:
            raise RetrievalError("E5 retriever requires at least one chunk")
        self._model_name = model_name
        self._model = model
        self._batch_size = batch_size
        self._device = device
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._vectors: np.ndarray | None = None
        model_version = _package_version("sentence-transformers")
        self.version = f"sentence-transformers=={model_version};model={model_name};normalized=cosine"

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional heavy dependency
                raise RetrievalError("sentence-transformers is required for the E5 retriever") from exc
            self._device = _resolve_e5_device(self._device)
            _LOGGER.info("Loading E5 model %s on %s", self._model_name, self._device)
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                local_files_only=Path(self._model_name).is_dir(),
            )
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_model()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RetrievalError("embedding model returned an unexpected shape")
        return vectors

    def _ensure_vectors(self) -> np.ndarray:
        if self._vectors is None:
            _LOGGER.info("Encoding %d chunks (batch size %d)", len(self._chunks), self._batch_size)
            self._vectors = self._encode([self._passage_prefix + chunk.text for chunk in self._chunks])
            _LOGGER.info("Chunk encoding complete; vectors cached in memory")
        return self._vectors

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievalHit]:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query must be a non-empty string")
        limit = _validate_top_k(top_k)
        query_vector = self._encode([self._query_prefix + query])[0]
        vectors = self._ensure_vectors()
        if query_vector.shape[0] != vectors.shape[1]:
            raise RetrievalError("query and passage embedding dimensions differ")
        scores = vectors @ query_vector
        order = sorted(range(len(self._chunks)), key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id))
        return [
            RetrievalHit(
                chunk=self._chunks[index],
                score=float(scores[index]),
                rank=rank,
                retriever=self.name,
                retriever_version=self.version,
            )
            for rank, index in enumerate(order[:limit], 1)
        ]


def build_retriever(
    name: str,
    chunks: Iterable[Chunk | Mapping[str, Any]],
    *,
    e5_model: Any | None = None,
    e5_model_name: str = DEFAULT_E5_MODEL,
    device: str | None = None,
) -> BM25Retriever | E5Retriever:
    """Build a retriever by name: ``bm25``, ``e5``, or ``local-e5``."""

    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"bm25", "sparse"}:
        return BM25Retriever(chunks)
    if normalized in {"e5", "local-e5", "dense"}:
        return E5Retriever(chunks, model_name=e5_model_name, model=e5_model, device=device)
    raise RetrievalError(f"unknown retriever: {name!r}")
