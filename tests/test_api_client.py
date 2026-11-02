from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from src.api_client import ApiClient
from src.cache import CacheStore
from src.config import load_settings


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeUsage:
    prompt_tokens: int = 4
    completion_tokens: int = 3


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
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
            "MAX_RETRIES": "2",
        },
        dotenv_path=None,
    )


def test_success_then_cache_hit(tmp_path):
    completions = FakeCompletions([FakeResponse([FakeChoice(FakeMessage("ok"))])])
    client = ApiClient(
        _settings(tmp_path),
        client=FakeClient(completions),
        cache=CacheStore(tmp_path / "cache"),
        sleep_fn=lambda _: None,
    )
    messages = [{"role": "user", "content": "hello"}]
    first = client.generate(messages, request_id="one")
    second = client.generate(messages, request_id="two")
    assert first.status == "success"
    assert not first.cache_hit
    assert second.cache_hit
    assert completions.calls == 1
    assert second.text == "ok"


def test_retryable_error_is_retried(tmp_path):
    completions = FakeCompletions(
        [TimeoutError("temporary"), FakeResponse([FakeChoice(FakeMessage("ok"))])]
    )
    client = ApiClient(
        _settings(tmp_path),
        client=FakeClient(completions),
        cache=CacheStore(tmp_path / "cache"),
        sleep_fn=lambda _: None,
    )
    result = client.generate(
        [{"role": "user", "content": "hello"}],
        request_id="retry",
    )
    assert result.status == "success"
    assert result.retry_count == 1
    assert completions.calls == 2


def test_empty_response_is_error_and_not_cached(tmp_path):
    completions = FakeCompletions([FakeResponse([])] * 3)
    client = ApiClient(
        _settings(tmp_path),
        client=FakeClient(completions),
        cache=CacheStore(tmp_path / "cache"),
        sleep_fn=lambda _: None,
    )
    result = client.generate(
        [{"role": "user", "content": "hello"}],
        request_id="empty",
    )
    assert result.status == "error"
    assert result.error_code == "empty_response"
    assert result.retry_count == 2
    assert completions.calls == 3
    assert not list((tmp_path / "cache").glob("*.json"))


def test_empty_content_retries_then_success_is_cached(tmp_path):
    from dataclasses import replace

    completions = FakeCompletions([
        FakeResponse([FakeChoice(FakeMessage(None))]),
        FakeResponse([FakeChoice(FakeMessage("  "))]),
        TimeoutError("temporary"),
        FakeResponse([FakeChoice(FakeMessage("ok"))]),
    ])
    delays = []
    client = ApiClient(replace(_settings(tmp_path), max_retries=3),
                       client=FakeClient(completions), sleep_fn=delays.append)
    messages = [{"role": "user", "content": "hello"}]
    result = client.generate(messages, request_id="empty-then-ok")
    assert result.status == "success"
    assert result.retry_count == 3
    assert completions.calls == 4
    assert delays == [1, 2, 4]
    assert client.generate(messages, request_id="cached").cache_hit
    assert completions.calls == 4


def test_empty_response_with_zero_retries(tmp_path):
    from dataclasses import replace

    completions = FakeCompletions([FakeResponse([])])
    client = ApiClient(replace(_settings(tmp_path), max_retries=0),
                       client=FakeClient(completions), sleep_fn=lambda _: None)
    result = client.generate([{"role": "user", "content": "hello"}], request_id="zero")
    assert result.error_code == "empty_response"
    assert result.retry_count == 0
    assert completions.calls == 1


def test_client_error_is_not_retried(tmp_path):
    completions = FakeCompletions([HttpError(400)])
    client = ApiClient(
        _settings(tmp_path),
        client=FakeClient(completions),
        cache=CacheStore(tmp_path / "cache"),
        sleep_fn=lambda _: None,
    )
    result = client.generate(
        [{"role": "user", "content": "hello"}],
        request_id="bad-request",
    )
    assert result.status == "error"
    assert result.error_code == "http_400"
    assert result.retry_count == 0
    assert completions.calls == 1


def test_truncated_response_is_error_and_not_cached(tmp_path):
    completions = FakeCompletions(
        [FakeResponse([FakeChoice(FakeMessage("partial"), finish_reason="length")])]
    )
    client = ApiClient(
        _settings(tmp_path),
        client=FakeClient(completions),
        cache=CacheStore(tmp_path / "cache"),
        sleep_fn=lambda _: None,
    )
    result = client.generate(
        [{"role": "user", "content": "hello"}],
        request_id="truncated",
    )
    assert result.status == "error"
    assert result.error_code == "truncated_response"
    assert not list((tmp_path / "cache").glob("*.json"))


@pytest.mark.parametrize("finish,refusal,expected", [
    ("length", None, "truncated_response"),
    ("content_filter", None, "content_filtered"),
    ("stop", "declined", "refusal"),
    ("stop", None, "empty_response"),
])
def test_empty_response_preserves_termination_and_usage(tmp_path, finish, refusal, expected):
    from dataclasses import replace
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
            content=None, reasoning_content="private reasoning", refusal=refusal))],
        usage=SimpleNamespace(prompt_tokens=200, completion_tokens=1024,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=1024)))
    completions = FakeCompletions([response])
    client = ApiClient(replace(_settings(tmp_path), max_retries=0),
                       client=FakeClient(completions), sleep_fn=lambda _: None)
    result = client.generate([{"role": "user", "content": "question"}], request_id="diagnose")
    assert result.error_code == expected and result.status == "error"
    assert result.finish_reason == finish
    assert result.input_tokens == 200 and result.output_tokens == 1024
    assert result.reasoning_tokens == 1024 and result.has_reasoning_content
    assert result.text == ""  # Reasoning must never become the answer.
    assert not list((tmp_path / "cache").glob("*.json"))


def test_empty_truncation_is_not_retried_as_transient_empty(tmp_path):
    completions = FakeCompletions([FakeResponse([FakeChoice(FakeMessage(None), "length")])])
    client = ApiClient(_settings(tmp_path), client=FakeClient(completions), sleep_fn=lambda _: None)
    result = client.generate([{"role": "user", "content": "question"}], request_id="truncated-empty")
    assert result.error_code == "truncated_response"
    assert result.retry_count == 0 and completions.calls == 1

