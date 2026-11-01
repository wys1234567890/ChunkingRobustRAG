"""Build and validate Stage 2 implicit fact-corruption datasets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

from src.data.common import DataError, read_jsonl
from src.poison.builder import build_poison_dataset
from src.poison.config import DATASETS, PoisonSettings, load_poison_settings
from src.poison.validation import validate_poison_dataset

_WORD = re.compile(r"\w+")


def _ratios(value: str) -> list[float]:
    try:
        ratios = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ratio list: {value}") from exc
    if not ratios:
        raise argparse.ArgumentTypeError("at least one ratio is required")
    return ratios


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/corruption.yaml")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--clean-dir", default=None, help="override the clean dataset directory")
    parser.add_argument("--ratio", type=_ratios, default=None, help="comma-separated ratios overriding the config")
    parser.add_argument("--offline", action="store_true", help="no network access (Stage 2 never downloads)")
    parser.add_argument("--validate-only", action="store_true", help="validate published ratios without rebuilding")
    parser.add_argument("--force", action="store_true", help="accepted for interface parity; never overwrites outputs")
    parser.add_argument("--synthetic", action="store_true", help="use the committed synthetic smoke dataset")
    parser.add_argument("--smoke-retrieval", action="store_true", help="report BM25 Top-50 poison recall per ratio")
    return parser


def _settings(args: argparse.Namespace) -> PoisonSettings:
    # Synthetic smoke runs may request an ad-hoc ratio; assign one poison
    # document per query for that level after loading the canonical config.
    settings = load_poison_settings(args.config, ratios=None if args.synthetic else args.ratio)
    if args.synthetic:
        synthetic_ratios = args.ratio or (0.0, 0.05)
        synthetic_budgets = dict(settings.poison_docs_per_query)
        for ratio in synthetic_ratios:
            synthetic_budgets.setdefault(f"{float(ratio):.8f}", 0 if float(ratio) == 0 else 1)
        settings = replace(
            settings,
            clean_data_version="synthetic-smoke-v2",
            clean_processed_root="data/processed",
            poisoned_root="data/synthetic/poisoned",
            poison_version=f"{settings.poison_version}-synthetic",
            ratios=synthetic_ratios,
            poison_docs_per_query=synthetic_budgets,
        )
    elif args.clean_dir:
        manifest = json.loads((Path(args.clean_dir) / "manifest.json").read_text(encoding="utf-8"))
        settings = replace(settings, clean_data_version=manifest["data_version"])
    return settings


def _dataset_names(value: str) -> tuple[str, ...]:
    return DATASETS if value == "all" else (value,)


def _clean_dir(settings: PoisonSettings, dataset: str, override: str | None) -> Path:
    return Path(override) if override else settings.clean_dir(dataset)


def _smoke_retrieval(directory: Path, datasets: list[tuple[str, Path]], limit: int = 20, depth: int = 50) -> dict:
    from rank_bm25 import BM25Okapi

    documents = read_jsonl(directory / "documents.jsonl")
    queries = read_jsonl(directory / "queries.jsonl")
    tokenized = [_WORD.findall(document["text"].casefold()) for document in documents]
    index = BM25Okapi(tokenized)
    poison_by_query: dict[str, list[str]] = {}
    for document in documents:
        if document.get("is_poison"):
            poison_by_query.setdefault(document["target_query_ids"][0], []).append(document["doc_id"])
    positions = {document["doc_id"]: position for position, document in enumerate(documents)}
    chosen = sorted(queries, key=lambda query: query["query_id"])[:limit]
    report = {"queries": len(chosen), "poison_in_top_50": 0, "details": []}
    for query in chosen:
        scores = index.get_scores(_WORD.findall(query["question"].casefold()))
        ranked = sorted(range(len(documents)), key=lambda index_: (-scores[index_], documents[index_]["doc_id"]))[:depth]
        best = None
        for doc_id in poison_by_query.get(query["query_id"], []):
            position = positions[doc_id]
            rank = ranked.index(position) + 1 if position in ranked else None
            if rank is not None and (best is None or rank < best):
                best = rank
        report["details"].append({"query_id": query["query_id"], "best_poison_rank": best})
        if best is not None and best <= 5:
            report["poison_in_top_50"] += 1
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = _settings(args)
        summaries = []
        for dataset in _dataset_names(args.dataset):
            clean_dir = _clean_dir(settings, dataset, args.clean_dir)
            clean_manifest = json.loads((clean_dir / "manifest.json").read_text(encoding="utf-8"))
            for ratio in settings.ratios:
                output_dir = settings.output_dir(dataset, ratio)
                if args.validate_only:
                    report = validate_poison_dataset(output_dir, expected_clean_manifest=clean_manifest)
                elif output_dir.exists():
                    report = validate_poison_dataset(output_dir, expected_clean_manifest=clean_manifest)
                    report = {**report, "reused": True}
                else:
                    report = build_poison_dataset(clean_dir, output_dir, settings, ratio=ratio, dataset=dataset)
                summary = dict(report)
                summary["output_dir"] = str(output_dir)
                if args.smoke_retrieval and not args.validate_only:
                    summary["smoke_retrieval"] = _smoke_retrieval(output_dir, [])
                summaries.append(summary)
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
