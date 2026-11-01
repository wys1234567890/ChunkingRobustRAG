from dataclasses import dataclass, field

import pytest

from src.api_client import ApiClient, GenerationResult
from src.chunking import Chunk
from src.config import load_settings
from src.rag import RagError, RagPipeline, build_context, build_messages
from src.retrieval import RetrievalHit


def _chunk(chunk_id: str, text: str, *, is_poison: bool = False) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=text,
        start=0,
        end=len(text),
        is_poison=is_poison,
        strategy="fixed",
        chunk_size=512,
        overlap=0.0,
        tokenizer="whitespace",
        token_count=len(text.split()),
    )


def _hit(chunk: Chunk, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk=chunk,
        score=1.0 - rank * 0.1,
        rank=rank,
        retriever="fake",
        retriever_version="fake-v1",
    )


class FakeRetriever:
    name = "fake"
    version = "fake-v1"

    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self._hits[:top_k]


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeUsage:
    prompt_tokens: int = 8
    completion_tokens: int = 5


@dataclass
class FakeResponse:
    choices: list
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, completions):
        self.chat = FakeChat(completions)


class HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"http {status_code}")
        self.status_code = status_code


def _settings(tmp_path):
    return load_settings(
        environ={
            "LLM_API_BASE": "https://example.test/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "CACHE_DIR": str(tmp_path / "cache"),
            "MAX_RETRIES": "0",
            "CONFIG_VERSION": "stage4-v1",
            "PROMPT_VERSION": "rag-v1",
        },
        dotenv_path=None,
    )


def _pipeline(tmp_path, hits, responses):
    from src.cache import CacheStore

    return RagPipeline(
        retriever=FakeRetriever(hits),
        api_client=ApiClient(
            _settings(tmp_path),
            client=FakeClient(FakeCompletions(responses)),
            cache=CacheStore(tmp_path / "cache"),
            sleep_fn=lambda _: None,
        ),
        top_k=5,
        chunk_type="fixed",
        chunk_size=512,
        overlap=0.0,
        poison_ratio=0.05,
        poison_version="fact-v1",
        dataset="nq",
    )


def test_build_context_numbers_chunks():
    hits = [_hit(_chunk("c1", "alpha"), 1), _hit(_chunk("c2", "beta"), 2)]
    ctx = build_context(hits)
    assert ctx == "[1] alpha\n\n[2] beta"


def test_build_context_respects_top_k():
    hits = [_hit(_chunk(f"c{i}", f"t{i}"), i) for i in range(1, 4)]
    assert build_context(hits, top_k=2) == "[1] t1\n\n[2] t2"


def test_build_context_rejects_non_positive_top_k():
    with pytest.raises(RagError):
        build_context([], top_k=0)


def test_build_messages_has_system_and_user():
    messages = build_messages("what is x?", "[1] context")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "what is x?" in messages[1]["content"]
    assert "[1] context" in messages[1]["content"]


def test_build_messages_rejects_empty_question():
    with pytest.raises(RagError):
        build_messages("   ", "ctx")


def test_run_records_retrieved_poison_true(tmp_path):
    hits = [
        _hit(_chunk("c1", "clean text"), 1),
        _hit(_chunk("c2", "poisoned text", is_poison=True), 2),
    ]
    pipeline = _pipeline(tmp_path, hits, [FakeResponse([FakeChoice(FakeMessage("answer"))])])
    result = pipeline.run("q1", "question?", ground_truth=["truth"], false_facts=["lie"])
    assert result.retrieved_poison is True
    assert result.poison_chunk_ids == ["c2"]
    assert result.retrieved_chunk_ids == ["c1", "c2"]
    assert result.api_status == "success"
    assert result.model_output == "answer"
    assert result.ground_truth == ["truth"]
    assert result.false_facts == ["lie"]
    assert result.chunk_type == "fixed"
    assert result.poison_ratio == 0.05
    assert result.poison_version == "fact-v1"
    assert result.config_version == "stage4-v1"
    assert result.prompt_version == "rag-v1"


def test_run_records_retrieved_poison_false(tmp_path):
    hits = [_hit(_chunk("c1", "clean text"), 1)]
    pipeline = _pipeline(tmp_path, hits, [FakeResponse([FakeChoice(FakeMessage("answer"))])])
    result = pipeline.run("q1", "question?")
    assert result.retrieved_poison is False
    assert result.poison_chunk_ids == []


def test_run_propagates_api_error(tmp_path):
    hits = [_hit(_chunk("c1", "clean text"), 1)]
    pipeline = _pipeline(tmp_path, hits, [HttpError(400)])
    result = pipeline.run("q1", "question?")
    assert result.api_status == "error"
    assert result.error_code == "http_400"
    assert result.model_output == ""


def test_run_rejects_empty_query_id(tmp_path):
    pipeline = _pipeline(tmp_path, [], [])
    with pytest.raises(RagError):
        pipeline.run("", "question?")


def test_run_cache_hit_skips_api_call(tmp_path):
    hits = [_hit(_chunk("c1", "clean text"), 1)]
    pipeline = _pipeline(tmp_path, hits, [FakeResponse([FakeChoice(FakeMessage("answer"))])])
    first = pipeline.run("q1", "question?")
    second = pipeline.run("q1", "question?")
    assert first.api_status == "success"
    assert second.cache_hit is True
    completions = pipeline.api._client.chat.completions
    assert completions.calls == 1
