from __future__ import annotations

import numpy as np
import pytest

from src.chunking import WhitespaceTokenizer, chunk_document, chunk_documents
from src.retrieval import BM25Retriever, E5Retriever, RetrievalError, build_retriever

DOCUMENTS = [
    {"doc_id": "d0", "dataset": "nq", "text": "alpha beta gamma.", "is_poison": False},
    {"doc_id": "d1", "dataset": "nq", "text": "beta gamma delta.", "is_poison": False},
    {"doc_id": "d2", "dataset": "nq", "text": "gamma delta epsilon.", "is_poison": False},
    {"doc_id": "d3", "dataset": "nq", "text": "delta epsilon zeta.", "is_poison": False},
    {"doc_id": "d4", "dataset": "nq", "text": "epsilon zeta eta.", "is_poison": False},
    {"doc_id": "d5", "dataset": "nq", "text": "zeta eta theta.", "is_poison": True},
]


@pytest.fixture
def chunks():
    return chunk_documents(
        DOCUMENTS,
        strategy="fixed",
        chunk_size=100,
        overlap=0.0,
        tokenizer=WhitespaceTokenizer(),
    )


class FakeE5:
    """Deterministic encoder for testing the E5 interface without downloading a model."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [
                    float("alpha" in lowered),
                    float("beta" in lowered),
                    float("gamma" in lowered),
                    float("zeta" in lowered),
                    0.01,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def test_bm25_returns_top_five_with_source_chunk_fields(chunks):
    hits = BM25Retriever(chunks).search("alpha")
    assert len(hits) == 5
    assert hits[0].chunk.doc_id == "d0"
    assert "alpha" in hits[0].chunk.text
    assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]
    assert all(5 >= hit.rank >= 1 for hit in hits)
    assert all(hit.retriever == "bm25" for hit in hits)
    assert hits[0].chunk.text == chunks[0].text or hits[0].chunk == chunks[0]
    assert "rank-bm25==" in hits[0].retriever_version


def test_bm25_top_k_is_capped_by_corpus_size(chunks):
    hits = BM25Retriever(chunks).search("gamma", top_k=50)
    assert len(hits) == len(chunks)


def test_bm25_rejects_blank_or_untokenizable_queries(chunks):
    retriever = BM25Retriever(chunks)
    with pytest.raises(RetrievalError, match="non-empty"):
        retriever.search("   ")
    with pytest.raises(RetrievalError, match="searchable"):
        retriever.search("!!!")


def test_e5_with_injected_model_returns_top_five(chunks):
    hits = E5Retriever(chunks, model=FakeE5()).search("alpha")
    assert len(hits) == 5
    assert hits[0].chunk.doc_id == "d0"
    assert all(hit.retriever == "e5" for hit in hits)
    assert "sentence-transformers==" in hits[0].retriever_version
    assert all(hit.chunk.chunk_id in {chunk.chunk_id for chunk in chunks} for hit in hits)


def test_build_retriever_supports_bm25_and_local_e5(chunks):
    assert isinstance(build_retriever("bm25", chunks), BM25Retriever)
    assert isinstance(build_retriever("local-e5", chunks, e5_model=FakeE5()), E5Retriever)


def test_build_retriever_rejects_unknown_name(chunks):
    with pytest.raises(RetrievalError, match="unknown retriever"):
        build_retriever("tfidf", chunks)


def test_retriever_accepts_chunk_dicts(chunks):
    records = [chunk.to_dict() for chunk in chunks]
    hits = BM25Retriever(records).search("beta")
    assert len(hits) == 5
    assert all(hit.chunk.chunk_id in {record["chunk_id"] for record in records} for hit in hits)


def test_empty_corpus_is_rejected():
    with pytest.raises(RetrievalError, match="at least one chunk"):
        BM25Retriever([])
    with pytest.raises(RetrievalError, match="at least one chunk"):
        E5Retriever([], model=FakeE5())


def test_hit_serializes_chunk_identity_and_score(chunks):
    hit = BM25Retriever(chunks).search("alpha", top_k=1)[0]
    value = hit.to_dict()
    assert value["chunk_id"] == hit.chunk.chunk_id
    assert value["doc_id"] == hit.chunk.doc_id
    assert value["is_poison"] == hit.chunk.is_poison
    assert value["retriever"] == "bm25"
    assert isinstance(value["score"], float)
