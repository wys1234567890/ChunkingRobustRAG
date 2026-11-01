from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import chunk_documents as cli
from src.chunking import CHUNK_CONFIG_VERSION
from src.data.common import file_sha256, read_jsonl, write_json, write_jsonl


def _document(doc_id, text, **extra):
    return {"doc_id": doc_id, "text": text, "dataset": "nq", "is_poison": False, **extra}


@pytest.fixture
def store(tmp_path):
    clean = [
        _document("b", "# First\nAlpha beta gamma. Delta epsilon zeta.\n\n## Child\nEta theta iota."),
        _document("a", "One two three four five six seven eight nine ten."),
    ]
    source = tmp_path / "clean.jsonl"
    write_jsonl(source, clean)
    directory = tmp_path / "clean-chunks"
    assert cli.main([
        "--input", str(source), "--output-dir", str(directory),
        "--tokenizer", "whitespace", "--chunk-sizes", "3,8", "--overlaps", "0,0.2",
    ]) == 0
    return source, directory, clean


def _run(tmp_path, documents, base, *, name="incremental", extra=()):
    source = tmp_path / f"{name}.jsonl"
    write_jsonl(source, documents)
    output = tmp_path / name
    args = [
        "--input", str(source), "--output-dir", str(output),
        "--tokenizer", "whitespace", "--chunk-sizes", "3,8", "--overlaps", "0,0.2",
    ]
    if base:
        args.extend(["--reuse-clean-chunks", str(base)])
    cli.main([*args, *extra])
    return output


@pytest.mark.parametrize("poison_count", [0, 1, 3])
def test_incremental_is_byte_identical_and_never_tokenizes_clean(tmp_path, store, monkeypatch, capsys, poison_count):
    _, base, clean = store
    poison = [
        _document(f"poison-{i}", f"Wrong fact {i}. Another sentence here.",
                  is_poison=True, source_doc_id="a", poison_version="test-v1")
        for i in range(poison_count)
    ]
    # Deliberately interleave poison and reorder clean documents.
    documents = poison[:1] + list(reversed(clean)) + poison[1:]
    full = _run(tmp_path, documents, None, name="full")
    original = cli.chunk_documents
    tokenized = []

    def only_poison(selected, **kwargs):
        selected = list(selected)
        assert all(d["is_poison"] for d in selected)
        tokenized.extend(d["doc_id"] for d in selected)
        return original(selected, **kwargs)

    monkeypatch.setattr(cli, "chunk_documents", only_poison)
    incremental = _run(tmp_path, documents, base)
    for path in full.glob("*.jsonl"):
        assert path.read_bytes() == (incremental / path.name).read_bytes()
    manifest = json.loads((incremental / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunk_config_version"] == CHUNK_CONFIG_VERSION
    assert manifest["reuse_clean"]["new_documents"] == poison_count
    assert manifest["reuse_clean"]["manifest_sha256"] == file_sha256(base / "manifest.json")
    assert len(tokenized) == poison_count * 12
    for run in manifest["runs"]:
        records = read_jsonl(incremental / run["file"])
        assert run["reused_chunks"] == sum(not r["is_poison"] for r in records)
        assert run["new_chunks"] == sum(r["is_poison"] for r in records)
    assert "DONE" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["text", "dataset", "source_doc_id", "poison_version", "missing", "duplicate"])
def test_changed_clean_documents_fail_before_writing(tmp_path, store, change):
    _, base, clean = store
    documents = [dict(d) for d in clean]
    if change == "missing":
        documents.pop()
    elif change == "duplicate":
        documents.append(dict(documents[0]))
    else:
        documents[0][change] = "changed"
    with pytest.raises(SystemExit, match="2"):
        _run(tmp_path, documents, base)
    assert not (tmp_path / "incremental").exists()


@pytest.mark.parametrize("change", ["tokenizer", "version", "source", "chunk_hash", "missing_run", "documents", "incomplete"])
def test_invalid_base_is_rejected_before_writing(tmp_path, store, change):
    source, base, clean = store
    path = base / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change == "tokenizer":
        manifest["tokenizer"] = "hf:other@revision"
    elif change == "version":
        manifest["chunk_config_version"] = "old-version"
    elif change == "source":
        source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif change == "chunk_hash":
        (base / manifest["runs"][0]["file"]).write_text("{}\n", encoding="utf-8")
    elif change == "missing_run":
        manifest["runs"].pop()
    elif change == "documents":
        manifest["runs"][0]["documents"] = 1
    elif change == "incomplete":
        path.unlink()
    if change != "incomplete":
        write_json(path, manifest)
    with pytest.raises(SystemExit, match="2"):
        _run(tmp_path, clean, base)
    assert not (tmp_path / "incremental").exists()


def test_supports_legacy_manifest_without_selection(tmp_path, store):
    _, base, clean = store
    path = base / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("selection")
    write_json(path, manifest)
    output = _run(tmp_path, clean, base)
    assert (output / "manifest.json").exists()


@pytest.mark.parametrize("change", ["configuration", "metadata", "missing_document", "count"])
def test_merge_validates_reused_records(tmp_path, store, change):
    _, base, clean = store
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest["runs"][0]
    path = base / run["file"]
    records = read_jsonl(path)
    if change == "configuration":
        records[0]["chunk_size"] = 999
    elif change == "metadata":
        records[0]["is_poison"] = True
    elif change == "missing_document":
        records = [r for r in records if r["doc_id"] != clean[0]["doc_id"]]
        run["chunks"] = len(records)
    else:
        run["chunks"] += 1
    write_jsonl(path, records)
    run["sha256"] = file_sha256(path)
    write_json(manifest_path, manifest)
    from src.chunking import ChunkingError

    with pytest.raises(ChunkingError):
        _run(tmp_path, clean, base)
    assert not list((tmp_path / "incremental").glob("*.jsonl"))


def test_dataset_selection_is_preserved(tmp_path):
    nq = _document("nq", "alpha beta gamma.")
    hp = _document("hp", "delta epsilon zeta.", dataset="hotpotqa")
    base = _run(tmp_path, [nq, hp], None, name="base", extra=["--dataset", "nq"])
    output = _run(tmp_path, [nq, hp], base, extra=["--dataset", "nq"])
    assert {r["doc_id"] for r in read_jsonl(output / "fixed-cs3-ov00.jsonl")} == {"nq"}


def test_reuse_directory_cannot_be_output(tmp_path, store):
    source, base, _ = store
    before = (base / "manifest.json").read_bytes()
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--input", str(source), "--output-dir", str(base), "--reuse-clean-chunks", str(base)])
    assert (base / "manifest.json").read_bytes() == before


def test_limit_is_not_silently_applied_before_poison(tmp_path, store):
    _, base, clean = store
    with pytest.raises(SystemExit, match="2"):
        _run(tmp_path, clean, base, extra=["--limit", "1"])


def test_atomic_output_preserves_previous_file_on_write_failure(tmp_path, store, monkeypatch):
    _, base, clean = store
    output = _run(tmp_path, clean, base)
    before = (output / "fixed-cs3-ov00.jsonl").read_bytes()

    def fail(chunks, path):
        Path(path).write_text("partial", encoding="utf-8")
        raise OSError("simulated write failure")

    monkeypatch.setattr(cli, "save_chunks", fail)
    with pytest.raises(OSError, match="simulated"):
        _run(tmp_path, clean, base)
    assert (output / "fixed-cs3-ov00.jsonl").read_bytes() == before
    assert not list(output.glob("*.tmp"))
