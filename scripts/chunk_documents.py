"""Build Stage 3 chunk stores from a clean or poisoned document JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from src.chunk_reuse import CleanChunkReuse
from src.chunking import (
    CHUNK_CONFIG_VERSION,
    ChunkingError,
    STRATEGIES,
    build_tokenizer,
    chunk_documents,
    save_chunks,
    validate_chunks,
)
from src.data.common import DataError, file_sha256, read_jsonl, write_json


def _parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk sizes must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("chunk sizes must be positive integers")
    return values


def _parse_float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("overlaps must be comma-separated numbers") from exc
    if not values or any(not 0 <= item < 1 for item in values):
        raise argparse.ArgumentTypeError("overlaps must satisfy 0 <= overlap < 1")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="documents.jsonl produced by Stage 1/2")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for chunk JSONL and manifest")
    parser.add_argument("--strategies", default=",".join(STRATEGIES), help="comma-separated: fixed,semantic,hierarchical")
    parser.add_argument("--chunk-sizes", default="512", help="comma-separated token budgets, e.g. 256,512,1024,2048")
    parser.add_argument("--overlaps", default="0.0", help="comma-separated fractions, e.g. 0.0,0.1,0.2")
    parser.add_argument("--tokenizer", default="hf:intfloat/multilingual-e5-small",
                        help="whitespace or hf:<model>; the tokenizer name is recorded on every chunk")
    parser.add_argument("--revision", default=None, help="optional Hugging Face tokenizer revision")
    parser.add_argument("--dataset", default=None, help="optional dataset filter (nq or hotpotqa)")
    parser.add_argument("--limit", type=int, default=None, help="optional document limit for smoke runs")
    parser.add_argument("--reuse-clean-chunks", type=Path, default=None,
                        help="completed clean chunk directory; only new poison documents are tokenized")
    args = parser.parse_args(argv)

    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        parser.error(f"unknown strategies: {', '.join(unknown)}")
    chunk_sizes = _parse_int_list(args.chunk_sizes)
    overlaps = _parse_float_list(args.overlaps)
    if args.reuse_clean_chunks and args.limit is not None:
        parser.error("--limit cannot be combined with --reuse-clean-chunks; use a matching clean smoke store instead")
    if args.reuse_clean_chunks and args.output_dir.resolve() == args.reuse_clean_chunks.resolve():
        parser.error("--output-dir must differ from --reuse-clean-chunks")

    documents = read_jsonl(args.input)
    if args.dataset:
        documents = [document for document in documents if document.get("dataset") == args.dataset]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        documents = documents[: args.limit]
    if not documents:
        parser.error("no documents selected")

    tokenizer = build_tokenizer(args.tokenizer, revision=args.revision)
    configurations = [(s, c, o) for s in strategies for c in chunk_sizes for o in overlaps]
    reuse = None
    if args.reuse_clean_chunks:
        print("Checking clean source, configurations and chunk hashes...", file=sys.stderr, flush=True)
        try:
            reuse = CleanChunkReuse(args.reuse_clean_chunks, documents, tokenizer=tokenizer.name,
                                   configurations=configurations)
        except (ChunkingError, DataError, OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"cannot reuse clean chunks: {exc}")
        print(f"Reuse {len(reuse.clean_documents)} clean documents; "
              f"chunk {len(reuse.new_documents)} poison documents.", file=sys.stderr, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for strategy in strategies:
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                started = time.perf_counter()
                filename = f"{strategy}-cs{chunk_size}-ov{round(overlap * 100):02d}.jsonl"
                print(f"[{len(runs) + 1}/{len(configurations)}] START {filename}", file=sys.stderr, flush=True)
                chunks = chunk_documents(
                    reuse.new_documents if reuse else documents,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    tokenizer=tokenizer,
                )
                new_count = len(chunks)
                reused_count = 0
                if reuse:
                    chunks, reused_count = reuse.merge(chunks, strategy=strategy, chunk_size=chunk_size,
                                                      overlap=overlap, tokenizer=tokenizer.name)
                validate_chunks(chunks, documents)
                path = args.output_dir / filename
                # Publish a complete JSONL, so an interruption leaves no partial target file.
                with tempfile.NamedTemporaryFile(dir=args.output_dir, prefix=filename, suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    count = save_chunks(chunks, temporary)
                    output_sha256 = file_sha256(temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                runs.append(
                    {
                        "strategy": strategy,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "tokenizer": tokenizer.name,
                        "file": filename,
                        "sha256": output_sha256,
                        "chunks": count,
                        "documents": len(documents),
                        "reused_chunks": reused_count,
                        "new_chunks": new_count,
                    }
                )
                print(f"[{len(runs)}/{len(configurations)}] DONE {filename}: "
                      f"{count} chunks (reused={reused_count}, new={new_count}), "
                      f"{time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)

    manifest = {
        "chunk_config_version": CHUNK_CONFIG_VERSION,
        "source": str(args.input),
        "source_sha256": file_sha256(args.input),
        "tokenizer": tokenizer.name,
        "selection": {"dataset": args.dataset, "limit": args.limit},
        "runs": runs,
    }
    if reuse:
        manifest["reuse_clean"] = {
            "manifest": str(reuse.manifest_path),
            "manifest_sha256": reuse.manifest_sha256,
            "documents": len(reuse.clean_documents),
            "new_documents": len(reuse.new_documents),
        }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
