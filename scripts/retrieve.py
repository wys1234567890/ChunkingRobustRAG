"""Search a Stage 3 chunk store with BM25, local E5, or both."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.chunking import load_chunks
from src.data.common import read_jsonl, write_jsonl
from src.retrieval import DEFAULT_E5_MODEL, TOP_K, build_retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, type=Path, help="chunk JSONL from scripts/chunk_documents.py")
    parser.add_argument("--query", default=None, help="single query string")
    parser.add_argument("--queries", default=None, type=Path, help="JSONL with query_id and question fields")
    parser.add_argument("--retriever", choices=("bm25", "e5", "both"), default="bm25")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model-name", default=DEFAULT_E5_MODEL)
    parser.add_argument("--device", default=None, help="optional E5 device, e.g. cpu or cuda")
    parser.add_argument("--output", default=None, type=Path, help="optional output JSONL; stdout by default")
    args = parser.parse_args(argv)

    if bool(args.query) == bool(args.queries):
        parser.error("provide exactly one of --query or --queries")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    chunks = load_chunks(args.chunks)
    if args.queries:
        queries = read_jsonl(args.queries)
        if not queries:
            parser.error("query file is empty")
    else:
        queries = [{"query_id": "query-1", "question": args.query}]

    names = ("bm25", "e5") if args.retriever == "both" else (args.retriever,)
    retrievers = [
        build_retriever(name, chunks, e5_model_name=args.model_name, device=args.device)
        for name in names
    ]

    records = []
    for query in queries:
        query_id = query.get("query_id", "")
        question = query.get("question")
        if not isinstance(question, str) or not question.strip():
            raise SystemExit(f"query {query_id!r} has no question")
        for retriever in retrievers:
            hits = retriever.search(question, top_k=args.top_k)
            records.append(
                {
                    "query_id": query_id,
                    "query": question,
                    "retriever": retriever.name,
                    "retriever_version": retriever.version,
                    "top_k": args.top_k,
                    "hits": [hit.to_dict() for hit in hits],
                }
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output, records)
    else:
        for record in records:
            print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
