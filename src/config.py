"""Configuration loading and validation for the OpenAI-compatible API layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Raised when configuration cannot be loaded or validated."""


@dataclass(frozen=True)
class ApiSettings:
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    judge_api_base: str
    judge_api_key: str
    judge_model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    max_retries: int = 3
    concurrency: int = 4
    cache_dir: str = "results/cache"
    config_version: str = "stage0-v1"
    prompt_version: str = "rag-v1"
    input_price_per_1k: float | None = None
    output_price_per_1k: float | None = None
    judge_thinking: str | None = None

    def validate(self, *, require_credentials: bool = False) -> None:
        """Validate values, optionally requiring credentials for live calls."""

        for name, value in (
            ("llm_api_base", self.llm_api_base),
            ("judge_api_base", self.judge_api_base),
        ):
            parsed = urlparse(value or "")
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ConfigError(f"{name} must be a valid HTTP(S) URL")

        if self.temperature < 0:
            raise ConfigError("temperature must be >= 0")
        if self.judge_thinking not in {None, "enabled", "disabled"}:
            raise ConfigError("judge_thinking must be enabled, disabled, or empty")
        if self.max_tokens <= 0:
            raise ConfigError("max_tokens must be > 0")
        if self.timeout_seconds <= 0:
            raise ConfigError("timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise ConfigError("max_retries must be >= 0")
        if self.concurrency < 1:
            raise ConfigError("concurrency must be >= 1")
        if not self.cache_dir.strip():
            raise ConfigError("cache_dir must not be empty")

        if require_credentials:
            missing = []
            for name, value in (
                ("LLM_API_KEY", self.llm_api_key),
                ("LLM_MODEL", self.llm_model),
                ("JUDGE_API_KEY", self.judge_api_key),
                ("JUDGE_MODEL", self.judge_model),
            ):
                if not value.strip():
                    missing.append(name)
            if missing:
                raise ConfigError("live API calls require: " + ", ".join(missing))

    def for_role(self, role: str) -> tuple[str, str, str]:
        """Return base URL, key, and model for ``llm`` or ``judge``."""

        if role == "llm":
            return self.llm_api_base, self.llm_api_key, self.llm_model
        if role == "judge":
            return self.judge_api_base, self.judge_api_key, self.judge_model
        raise ConfigError(f"unknown API role: {role}")


_DEFAULTS: dict[str, Any] = {
    "llm_api_base": "https://provider.example/v1",
    "llm_api_key": "",
    "llm_model": "example-model",
    "judge_api_base": None,
    "judge_api_key": None,
    "judge_model": None,
    "judge_thinking": None,
    "temperature": 0.0,
    "max_tokens": 1024,
    "timeout_seconds": 60.0,
    "max_retries": 3,
    "concurrency": 4,
    "cache_dir": "results/cache",
    "config_version": "stage0-v1",
    "prompt_version": "rag-v1",
    "input_price_per_1k": None,
    "output_price_per_1k": None,
}

_ENV_TO_FIELD = {
    "LLM_API_BASE": "llm_api_base",
    "LLM_API_KEY": "llm_api_key",
    "LLM_MODEL": "llm_model",
    "JUDGE_API_BASE": "judge_api_base",
    "JUDGE_API_KEY": "judge_api_key",
    "JUDGE_MODEL": "judge_model",
    "JUDGE_THINKING": "judge_thinking",
    "TEMPERATURE": "temperature",
    "MAX_TOKENS": "max_tokens",
    "TIMEOUT_SECONDS": "timeout_seconds",
    "MAX_RETRIES": "max_retries",
    "CONCURRENCY": "concurrency",
    "CACHE_DIR": "cache_dir",
    "CONFIG_VERSION": "config_version",
    "PROMPT_VERSION": "prompt_version",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"configuration file must contain a mapping: {path}")
    return value


def _yaml_values(data: Mapping[str, Any]) -> dict[str, Any]:
    api = data.get("api", {})
    runtime = data.get("runtime", {})
    metadata = data.get("metadata", {})
    pricing = data.get("pricing", {})
    if not all(isinstance(item, Mapping) for item in (api, runtime, metadata, pricing)):
        raise ConfigError("api, runtime, metadata, and pricing must be mappings")

    values: dict[str, Any] = {}
    for source in (api, runtime, metadata):
        values.update(source)
    values.update(
        {
            "input_price_per_1k": pricing.get("input_per_1k"),
            "output_price_per_1k": pricing.get("output_per_1k"),
        }
    )
    return {str(key): value for key, value in values.items() if value is not None}


def _coerce(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field == "judge_thinking":
        return str(value).strip().lower() or None
    if field in {"temperature", "timeout_seconds", "input_price_per_1k", "output_price_per_1k"}:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{field} must be numeric") from exc
    if field in {"max_tokens", "max_retries", "concurrency"}:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{field} must be an integer") from exc
    if field in {
        "llm_api_base",
        "llm_api_key",
        "llm_model",
        "judge_api_base",
        "judge_api_key",
        "judge_model",
        "cache_dir",
        "config_version",
        "prompt_version",
    }:
        return str(value)
    return value


def _candidate_config_path(config_path: str | Path | None) -> Path | None:
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"configuration file does not exist: {path}")
        return path
    for candidate in (Path("configs/api.yaml"), Path("configs/api.example.yaml")):
        if candidate.exists():
            return candidate
    return None


def load_settings(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = ".env",
) -> ApiSettings:
    """Load YAML, dotenv, and environment values with deterministic precedence."""

    values = dict(_DEFAULTS)
    selected_path = _candidate_config_path(config_path)
    if selected_path is not None:
        values.update(_yaml_values(_read_yaml(selected_path)))

    if dotenv_path is not None:
        dotenv = dotenv_values(dotenv_path) if Path(dotenv_path).exists() else {}
        for env_name, field in _ENV_TO_FIELD.items():
            if dotenv.get(env_name) is not None:
                values[field] = dotenv[env_name]

    actual_env = dict(os.environ if environ is None else environ)
    for env_name, field in _ENV_TO_FIELD.items():
        if env_name in actual_env:
            values[field] = actual_env[env_name]

    values["judge_api_base"] = values.get("judge_api_base") or values["llm_api_base"]
    values["judge_api_key"] = values.get("judge_api_key") or values["llm_api_key"]
    values["judge_model"] = values.get("judge_model") or values["llm_model"]

    converted = {field: _coerce(field, value) for field, value in values.items()}
    settings = ApiSettings(**converted)
    settings.validate()
    return settings

