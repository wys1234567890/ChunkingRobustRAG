"""Offline and live smoke tests for the Stage 0 API contract."""

from __future__ import annotations

import argparse
import json
import sys

from src.api_client import ApiClient
from src.config import ConfigError, load_settings


MESSAGES = [
    {"role": "system", "content": "Answer concisely using the supplied context."},
    {"role": "user", "content": "What is the purpose of this API smoke test?"},
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="validate without network access")
    mode.add_argument("--live", action="store_true", help="make one real API request")
    parser.add_argument("--config", help="YAML config path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
        settings.validate(require_credentials=args.live)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    client = ApiClient(settings)
    cache_key = client.make_cache_key(MESSAGES)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "config_version": settings.config_version,
                    "prompt_version": settings.prompt_version,
                    "model": settings.llm_model,
                    "cache_key": cache_key,
                    "network_called": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    first = client.generate(MESSAGES, request_id="stage0-smoke-1", cache_key=cache_key)
    second = client.generate(MESSAGES, request_id="stage0-smoke-2", cache_key=cache_key)
    print(
        json.dumps(
            {
                "mode": "live",
                "first": first.__dict__,
                "second": second.__dict__,
                "cache_verified": second.cache_hit,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if first.status == "success" and second.cache_hit else 1


if __name__ == "__main__":
    raise SystemExit(main())

