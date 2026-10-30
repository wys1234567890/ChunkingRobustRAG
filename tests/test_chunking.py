from __future__ import annotations

from dataclasses import replace

import pytest

from src.chunking import (
    ChunkingError,
    WhitespaceTokenizer,
    build_tokenizer,
    chunk_document,
    chunk_documents,
    validate_chunks,
)

TEXT = "Alpha beta gamma delta. Epsilon zeta eta.\n\n# Section Two\nTheta iota kappa. Lambda mu nu xi."


def _document(text: str = TEXT, **extra):
    return {"doc_id": "doc-1", "dataset": "nq", "text": text, "is_poison": False, **extra}


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()


@pytest.mark.parametrize("strategy", ["fixed", "semantic", "hierarchical"])
def test_all_strategies_preserve_exact_spans(strategy, tokenizer):
    chunks = chunk_document(_document(), strategy=strategy, chunk_size=5, overlap=0.0, tokenizer=tokenizer)
    validate_chunks(chunks, [_document()])
    assert chunks
    assert all(chunk.text == TEXT[chunk.start : chunk.end] for chunk in chunks)
    assert all(chunk.token_count <= 5 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert len({(chunk.start, chunk.end) for chunk in chunks}) == len(chunks)


def test_fixed_chunks_are_deterministic_and_cover_text(tokenizer):
    first = chunk_document(_document(), strategy="fixed", chunk_size=5, overlap=0.0, tokenizer=tokenizer)
    second = chunk_document(_document(), strategy="fixed", chunk_size=5, overlap=0.0, tokenizer=tokenizer)
    assert [chunk.to_dict() for chunk in first] == [chunk.to_dict() for chunk in second]
    assert first[0].start == 0
    assert first[-1].end == len(TEXT)
    assert all(left.end <= right.start for left, right in zip(first, first[1:]))


def test_overlap_reuses_material_without_duplicate_spans(tokenizer):
    without = chunk_document(_document(), strategy="fixed", chunk_size=6, overlap=0.0, tokenizer=tokenizer)
    with_overlap = chunk_document(_document(), strategy="fixed", chunk_size=6, overlap=0.5, tokenizer=tokenizer)
    assert len(with_overlap) > len(without)
    assert with_overlap[1].start < with_overlap[0].end
    assert len({(chunk.start, chunk.end) for chunk in with_overlap}) == len(with_overlap)
    assert all(chunk.token_count <= 6 for chunk in with_overlap)


class _BoundaryShiftTokenizer:
    """A tokenizer whose tokenization changes when a slice starts mid-token."""

    name = "boundary-shift-v1"

    def spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            if index % 2 == 1 and index + 1 < len(text):
                spans.append((index, index + 2))
                index += 2
            else:
                spans.append((index, index + 1))
                index += 1
        return spans


@pytest.mark.parametrize("strategy", ["fixed", "hierarchical"])
@pytest.mark.parametrize("chunk_size", [3, 4, 5])
def test_token_budget_holds_after_slice_retokenization(strategy, chunk_size):
    tokenizer = _BoundaryShiftTokenizer()
    text = "the quick brown fox jumps over the lazy dog"
    chunks = chunk_document(_document(text), strategy=strategy, chunk_size=chunk_size, overlap=0.0, tokenizer=tokenizer)
    assert chunks
    assert all(chunk.token_count <= chunk_size for chunk in chunks)
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)


def test_semantic_chunks_stop_at_sentence_boundaries(tokenizer):
    text = "Alpha beta gamma delta. Epsilon zeta eta. Theta iota kappa."
    chunks = chunk_document(_document(text), strategy="semantic", chunk_size=6, overlap=0.0, tokenizer=tokenizer)
    assert chunks
    assert all(chunk.text[-1] in ".!?。！？" for chunk in chunks)
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)


def test_semantic_oversized_sentence_falls_back_to_token_windows(tokenizer):
    text = ("word " * 20).strip() + "."
    chunks = chunk_document(_document(text), strategy="semantic", chunk_size=4, overlap=0.0, tokenizer=tokenizer)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 4 for chunk in chunks)
    validate_chunks(chunks, [_document(text)])


def test_hierarchical_chunks_do_not_cross_heading_boundaries(tokenizer):
    heading_start = TEXT.index("# Section Two")
    chunks = chunk_document(_document(), strategy="hierarchical", chunk_size=4, overlap=0.0, tokenizer=tokenizer)
    assert chunks
    for chunk in chunks:
        assert chunk.end <= heading_start or chunk.start >= heading_start
    assert any(chunk.text.startswith("# Section Two") for chunk in chunks)


def test_chunks_carry_poison_and_source_metadata(tokenizer):
    document = _document(
        is_poison=True,
        source_doc_id="source-1",
        poison_version="fact-v1",
        target_query_ids=["q-1"],
    )
    chunks = chunk_document(document, strategy="fixed", chunk_size=4, overlap=0.0, tokenizer=tokenizer)
    assert all(chunk.is_poison is True for chunk in chunks)
    assert all(chunk.source_doc_id == "source-1" for chunk in chunks)
    assert all(chunk.poison_version == "fact-v1" for chunk in chunks)
    assert all(chunk.dataset == "nq" for chunk in chunks)


def test_chunk_ids_depend_on_document_and_config(tokenizer):
    first = chunk_document(_document(), strategy="fixed", chunk_size=4, overlap=0.0, tokenizer=tokenizer)[0]
    other_doc = chunk_document(_document(doc_id="doc-2"), strategy="fixed", chunk_size=4, overlap=0.0, tokenizer=tokenizer)[0]
    other_config = chunk_document(_document(), strategy="fixed", chunk_size=6, overlap=0.0, tokenizer=tokenizer)[0]
    assert first.chunk_id != other_doc.chunk_id
    assert first.chunk_id != other_config.chunk_id


def test_chunk_documents_rejects_duplicate_doc_ids(tokenizer):
    with pytest.raises(ChunkingError, match="duplicate doc_id"):
        chunk_documents([_document(), _document()], tokenizer=tokenizer)


def test_multiple_documents_never_cross_concatenate(tokenizer):
    documents = [
        _document("First document sentence.", doc_id="doc-a"),
        _document("Second document sentence.", doc_id="doc-b"),
    ]
    chunks = chunk_documents(documents, strategy="semantic", chunk_size=3, overlap=0.0, tokenizer=tokenizer)
    texts = {document["doc_id"]: document["text"] for document in documents}
    assert {chunk.doc_id for chunk in chunks} == {"doc-a", "doc-b"}
    assert all(chunk.text == texts[chunk.doc_id][chunk.start : chunk.end] for chunk in chunks)


@pytest.mark.parametrize(
    "document,message",
    [
        ({"doc_id": "doc-1", "text": "   ", "is_poison": False}, "empty text"),
        ({"doc_id": "doc-1", "text": TEXT}, "is_poison"),
        ({"text": TEXT, "is_poison": False}, "doc_id"),
    ],
)
def test_invalid_documents_are_rejected(document, message, tokenizer):
    with pytest.raises(ChunkingError, match=message):
        chunk_document(document, tokenizer=tokenizer)


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 2])
def test_invalid_overlap_is_rejected(overlap, tokenizer):
    with pytest.raises(ChunkingError, match="overlap"):
        chunk_document(_document(), overlap=overlap, tokenizer=tokenizer)


def test_validate_chunks_detects_bad_readback(tokenizer):
    chunks = chunk_document(_document(), strategy="fixed", chunk_size=4, overlap=0.0, tokenizer=tokenizer)
    broken = replace(chunks[0], text="wrong")
    with pytest.raises(ChunkingError, match="readback"):
        validate_chunks([broken], [_document()])


def test_whitespace_tokenizer_is_the_explicit_offline_default():
    tokenizer = build_tokenizer("whitespace")
    assert tokenizer.name == "whitespace-v1"
    assert len(tokenizer.spans("a  b\nc")) == 3
