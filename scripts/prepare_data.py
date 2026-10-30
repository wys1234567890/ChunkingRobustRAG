"""Prepare reproducible, evidence-aligned NQ and HotpotQA development subsets."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from src.data import DataError, load_data_settings, prepare_dataset, validate_dataset
from src.data.config import DATASETS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--config", help="data YAML configuration; relative data paths use the repository root")
    parser.add_argument("--offline", action="store_true", help="never access the network")
    parser.add_argument("--validate-only", action="store_true", help="read and validate prepared files only")
    args = parser.parse_args(argv)
    try:
        settings = load_data_settings(args.config)
        datasets = DATASETS if args.dataset == "all" else (args.dataset,)
        results = []
        for dataset in datasets:
            if dataset not in settings.sources:
                raise DataError(f"no source configuration for {dataset}")
            if args.validate_only:
                result = validate_dataset(settings.output_dir(dataset), expected_config=settings.contract(dataset))
            else:
                result = prepare_dataset(settings, dataset, offline=args.offline, progress=lambda message: print(message, file=sys.stderr, flush=True))
            results.append(result)
        print(json.dumps({"datasets": results}, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, sqlite3.Error) as exc:
        print(f"data preparation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
