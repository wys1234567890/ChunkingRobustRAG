from pathlib import Path

import pytest

from src.config import ConfigError, load_settings


def test_environment_overrides_yaml_and_judge_falls_back(tmp_path: Path):
    config = tmp_path / "api.yaml"
    config.write_text(
        """
api:
  llm_api_base: https://yaml.example/v1
  llm_model: yaml-model
runtime:
  max_retries: 1
""",
        encoding="utf-8",
    )
    settings = load_settings(
        config,
        environ={
            "LLM_API_BASE": "https://env.example/v1",
            "LLM_API_KEY": "secret",
            "LLM_MODEL": "env-model",
        },
        dotenv_path=None,
    )
    assert settings.llm_api_base == "https://env.example/v1"
    assert settings.llm_model == "env-model"
    assert settings.max_retries == 1
    assert settings.judge_api_base == settings.llm_api_base
    assert settings.judge_api_key == settings.llm_api_key
    assert settings.judge_model == settings.llm_model


def test_invalid_url_is_rejected(tmp_path: Path):
    config = tmp_path / "api.yaml"
    config.write_text("api:\n  llm_api_base: not-a-url\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="llm_api_base"):
        load_settings(config, dotenv_path=None)


def test_live_validation_requires_credentials():
    settings = load_settings(dotenv_path=None)
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        settings.validate(require_credentials=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("TEMPERATURE", "-1", "temperature"),
        ("MAX_TOKENS", "0", "max_tokens"),
        ("MAX_RETRIES", "-1", "max_retries"),
        ("CONCURRENCY", "0", "concurrency"),
    ],
)
def test_numeric_validation(field: str, value: str, message: str):
    with pytest.raises(ConfigError, match=message):
        load_settings(environ={field: value}, dotenv_path=None)

