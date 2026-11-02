"""Score one Stage 4 run against an explicit query set, without API calls."""

import argparse
import json
from pathlib import Path

from src.data.common import read_jsonl, write_jsonl
from src.evaluation import evaluate_answers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="summary JSON")
    parser.add_argument("--details", type=Path, help="optional per-query JSONL")
    args = parser.parse_args(argv)
    paths = [args.results.resolve(), args.queries.resolve(), args.output.resolve()]
    if args.details:
        paths.append(args.details.resolve())
    if len(paths) != len(set(paths)):
        parser.error("input and output paths must be distinct")
    try:
        summary, details = evaluate_answers(read_jsonl(args.queries), read_jsonl(args.results))
    except ValueError as exc:
        parser.error(str(exc))
    summary["results_file"] = str(args.results)
    summary["queries_file"] = str(args.queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.details, details)
    print(f"queries={summary['total_queries']} success={summary['successful_queries']} "
          f"errors={summary['error_queries']} missing={summary['missing_queries']}")
    print(f"EM={summary['em']:.2%} token_F1={summary['token_f1']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
