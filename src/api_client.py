"""OpenAI-compatible API client with retries, cache, and metadata-only logging."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Literal, Sequence, TypedDict

from openai import OpenAI

from .cache import CacheStore, build_cache_key
from .config import ApiSettings, ConfigError

logger = logging.getLogger(__name__)


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    status: Literal["success", "error"]
    model: str
    request_id: str
    cache_hit: bool
    retry_count: int
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost: float | None
    error_code: str | None
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    has_reasoning_content: bool = False


class ApiClient:
    """Generate one response through the configured LLM or judge endpoint."""

    def __init__(
        self,
        settings: ApiSettings,
        *,
        role: Literal["llm", "judge"] = "llm",
        client: Any | None = None,
        cache: CacheStore | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        settings.validate()
        self.settings = settings
        self.role = role
        self.base_url, self.api_key, self.model = settings.for_role(role)
        self.cache = cache or CacheStore(settings.cache_dir)
        self._sleep = sleep_fn
        self._client = client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )

    def _request_options(self) -> dict[str, Any]:
        if self.role == "judge" and self.settings.judge_thinking is not None:
            return {"extra_body": {"thinking": {"type": self.settings.judge_thinking}}}
        return {}

    def make_cache_key(
        self,
        messages: Sequence[Message],
        *,
        prompt_version: str | None = None,
        config_version: str | None = None,
    ) -> str:
        return build_cache_key(
            messages,
            model=self.model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            prompt_version=prompt_version or self.settings.prompt_version,
            config_version=config_version or self.settings.config_version,
            request_options=self._request_options(),
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        request_id: str,
        cache_key: str | None = None,
    ) -> GenerationResult:
        if not messages:
            raise ConfigError("messages must not be empty")
        key = cache_key or self.make_cache_key(messages)
        cached = self.cache.load(key)
        if cached is not None:
            try:
                return replace(GenerationResult(**cached), cache_hit=True)
            except TypeError:
                logger.warning("ignoring incompatible cache entry", extra={"request_id": request_id})

        started = time.monotonic()
        retries = 0
        last_error: Exception | None = None
        while retries <= self.settings.max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    **self._request_options(),
                )
                result = self._response_result(response, request_id, retries, started)
                if result.error_code == "empty_response" and retries < self.settings.max_retries:
                    self._sleep(min(2**retries, 8))
                    retries += 1
                    continue
                if result.status == "success":
                    self.cache.save(key, asdict(result))
                self._log_result(result)
                return result
            except Exception as exc:  # SDK/provider exceptions vary by compatible service.
                last_error = exc
                if retries >= self.settings.max_retries or not self._retryable(exc):
                    break
                delay = min(2**retries, 8)
                retries += 1
                self._sleep(delay)

        result = GenerationResult(
            text="",
            status="error",
            model=self.model,
            request_id=request_id,
            cache_hit=False,
            retry_count=retries,
            latency_ms=self._elapsed_ms(started),
            input_tokens=None,
            output_tokens=None,
            estimated_cost=None,
            error_code=self._error_code(last_error),
        )
        self._log_result(result)
        return result

    def _response_result(
        self,
        response: Any,
        request_id: str,
        retry_count: int,
        started: float,
    ) -> GenerationResult:
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        text = getattr(message, "content", None) if message is not None else None
        text = text if isinstance(text, str) else ""
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        token_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(token_details, "reasoning_tokens", None)
        has_reasoning = bool(getattr(message, "reasoning_content", None))
        error_code = None
        # Inspect termination before text: a reasoning-only truncated response
        # must not be misclassified as a transient empty response.
        if finish_reason == "length":
            error_code = "truncated_response"
        elif finish_reason == "content_filter":
            error_code = "content_filtered"
        elif getattr(message, "refusal", None):
            error_code = "refusal"
        elif not text.strip():
            error_code = "empty_response"

        return GenerationResult(
            text=text,
            status="error" if error_code else "success",
            model=self.model,
            request_id=request_id,
            cache_hit=False,
            retry_count=retry_count,
            latency_ms=self._elapsed_ms(started),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=self._cost(input_tokens, output_tokens),
            error_code=error_code,
            finish_reason=finish_reason,
            reasoning_tokens=reasoning_tokens,
            has_reasoning_content=has_reasoning,
        )

    def _error_result(
        self,
        request_id: str,
        retry_count: int,
        started: float,
        error_code: str,
    ) -> GenerationResult:
        return GenerationResult(
            text="",
            status="error",
            model=self.model,
            request_id=request_id,
            cache_hit=False,
            retry_count=retry_count,
            latency_ms=self._elapsed_ms(started),
            input_tokens=None,
            output_tokens=None,
            estimated_cost=None,
            error_code=error_code,
        )

    def _cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        input_rate = self.settings.input_price_per_1k
        output_rate = self.settings.output_price_per_1k
        if input_rate is None or output_rate is None:
            return None
        if input_tokens is None or output_tokens is None:
            return None
        return (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 429}:
            return True
        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return True
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True
        return exc.__class__.__name__ in {
            "APITimeoutError",
            "APIConnectionError",
            "ConnectTimeout",
            "PoolTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "TimeoutException",
        }

    @staticmethod
    def _error_code(exc: Exception | None) -> str:
        if exc is None:
            return "unknown_error"
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return f"http_{status_code}"
        return exc.__class__.__name__.lower()

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.monotonic() - started) * 1000)

    @staticmethod
    def _log_result(result: GenerationResult) -> None:
        logger.info(
            "api generation completed",
            extra={
                "request_id": result.request_id,
                "model": result.model,
                "status": result.status,
                "cache_hit": result.cache_hit,
                "retry_count": result.retry_count,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "error_code": result.error_code,
                "finish_reason": result.finish_reason,
                "reasoning_tokens": result.reasoning_tokens,
                "has_reasoning_content": result.has_reasoning_content,
            },
        )

