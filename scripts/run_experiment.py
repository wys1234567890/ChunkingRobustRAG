"""Stage 4: run a no-RAG baseline or RAG with one chunk config + retriever.

RAG reads a Stage 3 chunk store and a Stage 1/2 query set, retrieves Top-k chunks
per query, builds the RAG prompt, and calls the configured LLM API. Results are
appended to ``--output`` JSONL; existing ``query_id`` records are skipped so the
experiment can be resumed after interruption.
Use --no-rag to answer queries without a chunk store or retriever.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from src.api_client import ApiClient
from src.chunking import load_chunks
from src.config import ConfigError, load_settings
from src.data.common import canonical_json, read_jsonl
from src.rag import RagPipeline
from src.retrieval import DEFAULT_E5_MODEL, TOP_K, build_retriever


def _false_facts_by_query(path: Path | None) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    if path is None:
        return mapping
    for record in read_jsonl(path):
        qid = record.get("target_query_id")
        false_fact = record.get("false_fact")
        if isinstance(false_fact, dict):
            value = false_fact.get("value")
        else:
            value = false_fact
        if qid and value:
            mapping.setdefault(str(qid), []).append(str(value))
    return mapping


def _done_query_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(record.get("query_id", "")) for record in read_jsonl(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, help="chunk JSONL; required for RAG")
    parser.add_argument("--no-rag", action="store_true", help="answer from model knowledge without loading chunks or a retriever")
    parser.add_argument("--queries", required=True, type=Path, help="query JSONL with query_id and question")
    parser.add_argument("--poison-records", type=Path, default=None, help="poison_records.jsonl for false_fact lookup")
    parser.add_argument("--retriever", choices=("bm25", "e5"))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model-name", default=DEFAULT_E5_MODEL, help="E5 model name or local path")
    parser.add_argument("--device", default=None, help="E5 device: auto (default), cpu, cuda, xpu or xpu:0")
    parser.add_argument("--output", required=True, type=Path, help="output JSONL (append mode)")
    parser.add_argument("--limit", type=int, default=0, help="process only first N queries (0 = all)")
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--poison-ratio", type=float, default=None, help="metadata: target poison ratio")
    parser.add_argument("--dataset", default="", help="override dataset label (default: from chunks)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.no_rag:
        if args.chunks or args.retriever or args.poison_records or args.poison_ratio is not None:
            parser.error("--no-rag cannot be combined with chunks, retriever or poison options")
    elif not args.chunks or not args.retriever:
        parser.error("RAG requires --chunks and --retriever; use --no-rag for the baseline")
    if args.top_k <= 0:
        print("--top-k must be positive", file=sys.stderr)
        return 2
    if args.limit < 0:
        print("--limit must be non-negative", file=sys.stderr)
        return 2

    try:
        settings = load_settings(args.config)
        if args.no_rag:
            settings = replace(settings, prompt_version="no-rag-v1")
        settings.validate(require_credentials=True)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    chunks = [] if args.no_rag else load_chunks(args.chunks)
    if not args.no_rag and not chunks:
        print("chunk store is empty", file=sys.stderr)
        return 2
    queries = read_jsonl(args.queries)
    if not queries:
        print("query file is empty", file=sys.stderr)
        return 2
    if args.limit > 0:
        queries = queries[: args.limit]

    false_facts_map = _false_facts_by_query(args.poison_records)
    done = _done_query_ids(args.output)

    sample = chunks[0] if chunks else None
    dataset = args.dataset or (sample.dataset if sample else str(queries[0].get("dataset", "")))

    if args.output.exists():
        for record in read_jsonl(args.output):
            if record.get("retriever") != ("none" if args.no_rag else args.retriever):
                parser.error("output contains a different retrieval mode; choose a separate output file")

    retriever = None if args.no_rag else build_retriever(
        args.retriever,
        chunks,
        e5_model_name=args.model_name,
        device=args.device,
    )
    api = ApiClient(settings)
    pipeline = RagPipeline(
        retriever=retriever,
        api_client=api,
        top_k=args.top_k,
        chunk_type=sample.strategy if sample else "none",
        chunk_size=sample.chunk_size if sample else 0,
        overlap=sample.overlap if sample else 0.0,
        poison_ratio=args.poison_ratio,
        poison_version=sample.poison_version if sample else None,
        dataset=dataset,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    errors = 0
    with args.output.open("a", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            qid = str(query.get("query_id", ""))
            if not qid:
                print("skipping query without query_id", file=sys.stderr)
                continue
            if qid in done:
                continue
            question = query.get("question")
            if not isinstance(question, str) or not question.strip():
                print(f"skipping {qid}: missing question", file=sys.stderr)
                continue
            ground_truth = list(query.get("ground_truth") or [])
            false_facts = false_facts_map.get(qid, [])
            result = pipeline.run(
                qid,
                question,
                ground_truth=ground_truth,
                false_facts=false_facts,
            )
            handle.write(canonical_json(result.to_dict()) + "\n")
            handle.flush()
            if result.api_status == "success":
                success += 1
            else:
                errors += 1
            print(
                f"{qid}: {result.api_status} "
                f"(cache_hit={result.cache_hit}, poison={result.retrieved_poison}, "
                f"err={result.error_code})"
            )

    print(f"done: success={success}, errors={errors}, skipped={len(done)}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
