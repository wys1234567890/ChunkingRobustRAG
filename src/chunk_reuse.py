"""Reuse verified clean chunks when adding poison documents to a corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .chunking import CHUNK_CONFIG_VERSION, Chunk, ChunkingError
from .data.common import file_sha256, read_jsonl


def document_index(documents: Iterable[dict]) -> dict[str, dict]:
    indexed = {}
    for document in documents:
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ChunkingError("document requires a non-empty doc_id")
        if doc_id in indexed:
            raise ChunkingError(f"duplicate doc_id: {doc_id}")
        if not isinstance(document.get("is_poison"), bool):
            raise ChunkingError(f"document {doc_id!r} requires boolean is_poison")
        if not isinstance(document.get("text"), str) or not document["text"].strip():
            raise ChunkingError(f"document {doc_id!r} has empty text")
        indexed[doc_id] = document
    return indexed


def _chunk_metadata(document: dict) -> tuple:
    """Document fields that affect chunk contents, identity, or metadata."""
    return (
        document["text"], document["is_poison"],
        str(document.get("dataset", "")),
        str(document.get("source_doc_id", document["doc_id"])),
        document.get("poison_version"),
    )


class CleanChunkReuse:
    """Preflight a completed clean store before producing any incremental files."""

    def __init__(
        self, directory: Path, documents: list[dict], *, tokenizer: str,
        configurations: list[tuple[str, int, float]],
    ) -> None:
        self.directory = directory.resolve()
        self.manifest_path = self.directory / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest_sha256 = file_sha256(self.manifest_path)
        if manifest.get("chunk_config_version") != CHUNK_CONFIG_VERSION:
            raise ChunkingError("clean chunk_config_version does not match")
        if manifest.get("tokenizer") != tokenizer:
            raise ChunkingError("clean tokenizer does not match (including revision)")

        source = Path(manifest["source"])
        if not source.is_absolute() and not source.is_file():
            # Older manifests store a path relative to the original working directory.
            for parent in self.directory.parents:
                candidate = parent / source
                if candidate.is_file():
                    source = candidate
                    break
        if file_sha256(source) != manifest.get("source_sha256"):
            raise ChunkingError("clean source SHA256 does not match its manifest")
        base_documents = read_jsonl(source)
        selection = manifest.get("selection", {})
        if selection.get("dataset"):
            base_documents = [d for d in base_documents if d.get("dataset") == selection["dataset"]]
        if selection.get("limit") is not None:
            base_documents = base_documents[: selection["limit"]]
        base = document_index(base_documents)
        if not base or any(d["is_poison"] for d in base.values()):
            raise ChunkingError("reuse source must contain only clean documents")
        current = document_index(documents)
        clean = {key: d for key, d in current.items() if not d["is_poison"]}
        if clean.keys() != base.keys():
            raise ChunkingError("clean document IDs differ from the reuse source")
        for doc_id, document in clean.items():
            if _chunk_metadata(document) != _chunk_metadata(base[doc_id]):
                raise ChunkingError(f"clean document changed: {doc_id}")
        self.clean_documents = clean
        self.new_documents = [d for d in documents if d["is_poison"]]
        self.documents = documents
        self.runs: dict[tuple[str, int, float], dict[str, Any]] = {}
        for strategy, size, overlap in configurations:
            matches = [
                run for run in manifest["runs"]
                if (run.get("strategy"), run.get("chunk_size"), run.get("overlap"))
                == (strategy, size, overlap)
            ]
            if len(matches) != 1:
                raise ChunkingError(f"missing or ambiguous clean configuration: {strategy}/{size}/{overlap}")
            run = matches[0]
            if run.get("tokenizer") != tokenizer or run.get("documents") != len(base):
                raise ChunkingError("clean run tokenizer or document count does not match")
            path = (self.directory / run["file"]).resolve()
            if path.parent != self.directory:
                raise ChunkingError("clean chunk file must be inside the reuse directory")
            if file_sha256(path) != run.get("sha256"):
                raise ChunkingError(f"clean chunk SHA256 does not match: {path.name}")
            self.runs[strategy, size, overlap] = dict(run, path=path)

    def merge(
        self, new_chunks: list[Chunk], *, strategy: str, chunk_size: int,
        overlap: float, tokenizer: str,
    ) -> tuple[list[Chunk], int]:
        """Keep input document order, exactly as a full chunk_documents call does."""
        run = self.runs[strategy, chunk_size, overlap]
        grouped: dict[str, list[Chunk]] = {d["doc_id"]: [] for d in self.documents}
        count = 0
        with run["path"].open(encoding="utf-8") as handle:
            for line in handle:
                chunk = Chunk.from_dict(json.loads(line))
                document = self.clean_documents.get(chunk.doc_id)
                if document is None:
                    raise ChunkingError(f"unexpected clean chunk document: {chunk.doc_id}")
                if (chunk.strategy, chunk.chunk_size, chunk.overlap, chunk.tokenizer) != (
                    strategy, chunk_size, overlap, tokenizer,
                ):
                    raise ChunkingError(f"clean chunk configuration does not match: {chunk.chunk_id}")
                if (chunk.text, chunk.is_poison, chunk.dataset, chunk.source_doc_id, chunk.poison_version) != (
                    document["text"][chunk.start:chunk.end], False,
                    str(document.get("dataset", "")),
                    str(document.get("source_doc_id", document["doc_id"])),
                    document.get("poison_version"),
                ):
                    raise ChunkingError(f"clean chunk metadata or text does not match: {chunk.chunk_id}")
                grouped[chunk.doc_id].append(chunk)
                count += 1
        if count != run.get("chunks"):
            raise ChunkingError("clean chunk count does not match its manifest")
        if any(not grouped[doc_id] for doc_id in self.clean_documents):
            raise ChunkingError("clean chunk file is missing documents")
        for chunk in new_chunks:
            grouped[chunk.doc_id].append(chunk)
        return [chunk for d in self.documents for chunk in grouped[d["doc_id"]]], count
