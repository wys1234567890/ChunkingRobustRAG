"""Score one Stage 4 run with LLM Judge (default) or offline EM/token F1."""

import argparse
import json
from pathlib import Path

from src.data.common import file_sha256, read_jsonl, write_jsonl
from src.evaluation import evaluate_answers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="summary JSON")
    parser.add_argument("--details", type=Path, help="optional per-query JSONL")
    parser.add_argument("--mode", choices=("judge", "offline"), default="judge",
                        help="judge calls the configured API; offline needs no credentials")
    parser.add_argument("--config", help="API YAML configuration for Judge")
    args = parser.parse_args(argv)
    paths = [args.results.resolve(), args.queries.resolve(), args.output.resolve()]
    if args.details:
        paths.append(args.details.resolve())
    if args.config:
        paths.append(Path(args.config).resolve())
    if len(paths) != len(set(paths)):
        parser.error("input and output paths must be distinct")
    try:
        queries, records = read_jsonl(args.queries), read_jsonl(args.results)
        # Finish all input validation before constructing the API client.
        summary, details = evaluate_answers(queries, records)
        hashes = {"results_sha256": file_sha256(args.results), "queries_sha256": file_sha256(args.queries)}
        if args.mode == "judge":
            from src.config import load_settings
            from src.judge import create_judge_client, evaluate_answers_with_judge

            client = create_judge_client(load_settings(args.config))
            print(f"Judge model={client.model}; temperature=0; API calls enabled", flush=True)
            summary, details = evaluate_answers_with_judge(
                queries, records, client,
                progress=lambda done, total, d: print(
                    f"judge {done}/{total} {d['query_id']}: {d['judge_status']}", flush=True),
            )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    summary.update(hashes, mode=args.mode)
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
    if args.mode == "judge":
        accuracy = summary["judge_accuracy"]
        print(f"judge_accuracy={accuracy:.2%}" if accuracy is not None else
              f"judge_accuracy=null; incomplete: {summary['judge_error_queries']} Judge errors; rerun to retry")
        if not summary["evaluation_complete"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
