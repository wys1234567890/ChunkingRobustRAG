from __future__ import annotations

import copy
import gzip
import io
import json
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data import DataError, load_data_settings, load_documents, load_queries, prepare_dataset
from src.data.adapters import adapt_hotpotqa, adapt_nq
from src.data.parquet_sources import restore_hotpotqa, restore_nq
from src.data.sources import download_source, iter_source
from tests.data_factory import hotpot_record, nq_record
from tests.test_data import settings  # Shared small offline preparation configuration.


def columns(rows, names):
    return {name: [row[name] for row in rows] for name in names}


def hf_nq(raw, columnar=True):
    tokens = [{"token": t["token"], "is_html": t["html_token"], "start_byte": t["start_byte"], "end_byte": t["end_byte"]} for t in raw["document_tokens"]]
    annotations = []
    for a in raw["annotations"]:
        shorts = a["short_answers"]
        annotations.append({
            "id": str(a["annotation_id"]), "long_answer": a["long_answer"],
            "short_answers": columns(shorts, ("start_byte", "end_byte", "start_token", "end_token")) if columnar else shorts,
            "yes_no_answer": {"NONE": -1, "NO": 0, "YES": 1}[a["yes_no_answer"]],
        })
    return {
        "id": str(raw["example_id"]),
        "document": {"title": raw["document_title"], "url": raw["document_url"], "html": raw["document_html"], "tokens": columns(tokens, tokens[0].keys()) if columnar else tokens},
        "question": {"text": raw["question_text"], "tokens": raw["question_text"].split()},
        "annotations": columns(annotations, annotations[0].keys()) if columnar else annotations,
    }


def hf_hotpot(raw, columnar=True):
    context = [{"title": title, "sentences": sentences} for title, sentences in raw["context"]]
    supports = [{"title": title, "sent_id": index} for title, index in raw["supporting_facts"]]
    return {
        "id": raw["_id"], "question": raw["question"], "answer": raw["answer"],
        "type": raw["type"], "level": raw["level"],
        "context": columns(context, ("title", "sentences")) if columnar else context,
        "supporting_facts": columns(supports, ("title", "sent_id")) if columnar else supports,
    }


@pytest.mark.parametrize("columnar", [False, True])
def test_hf_conversion_preserves_original_alignment_and_annotations(settings, columnar):
    raw = nq_record(answer="東京")
    raw["annotations"][0]["short_answers"] = []
    restored = restore_nq(hf_nq(raw, columnar))
    assert restored["example_id"] == str(raw["example_id"])
    assert restored["document_html"] == raw["document_html"]
    assert restored["document_tokens"] == raw["document_tokens"]
    original = adapt_nq(raw, "same-source.parquet", settings)
    mirrored = adapt_nq(restored, "same-source.parquet", settings)
    assert (mirrored.documents, mirrored.query, mirrored.evidence) == (original.documents, original.query, original.evidence)
    raw_hotpot = hotpot_record()
    restored_hotpot = restore_hotpotqa(hf_hotpot(raw_hotpot, columnar))
    assert restored_hotpot == raw_hotpot
    assert adapt_hotpotqa(restored_hotpot, "same-source.parquet", settings) == adapt_hotpotqa(raw_hotpot, "same-source.parquet", settings)


@pytest.mark.parametrize("label,expected", [(-1, "NONE"), (0, "NO"), (1, "YES")])
def test_hf_yes_no_class_labels_are_not_mistaken_for_extractable_answers(label, expected):
    row = hf_nq(nq_record(), columnar=False)
    row["annotations"][0]["yes_no_answer"] = label
    assert restore_nq(row)["annotations"][0]["yes_no_answer"] == expected


def test_incomplete_or_misaligned_mirror_data_is_rejected():
    row = hf_nq(nq_record())
    del row["document"]["tokens"]["start_byte"]
    with pytest.raises(DataError, match="missing original evidence fields"):
        restore_nq(row)
    hotpot = hf_hotpot(hotpot_record())
    hotpot["supporting_facts"]["sent_id"].pop()
    with pytest.raises(DataError, match="equally-sized columns"):
        restore_hotpotqa(hotpot)


@pytest.mark.parametrize("dataset", ["nq", "hotpotqa"])
def test_parquet_pipeline_and_atomic_download_validation(settings, dataset):
    factory = (lambda i: hf_nq(nq_record(i))) if dataset == "nq" else (lambda i: hf_hotpot(hotpot_record(i, "bridge" if i <= 4 else "comparison")))
    rows = [factory(i) for i in range(1, 7)]
    path = settings.root / f"{dataset}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    restored = list(iter_source(path, dataset))
    assert len(restored) == 6
    sources = copy.deepcopy(settings.sources)
    sources[dataset]["files"] = [{"path": path.name}]
    sources[dataset]["expected_records"] = 6
    configured = replace(settings, sources=sources)
    with patch("urllib.request.urlopen", side_effect=AssertionError("offline mode used network")):
        result = prepare_dataset(configured, dataset, offline=True)
    assert (result["query_count"], result["document_count"], result["span_readback_rate"]) == (2, 6, 1.0)
    assert load_queries(configured.output_dir(dataset) / "queries.jsonl")
    assert load_documents(configured.output_dir(dataset) / "documents.jsonl")
    body = path.read_bytes()
    with patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
        destination = settings.root / f"download-{dataset}.parquet"
        download_source("https://example.test/file.parquet", destination, dataset)
    assert destination.read_bytes() == body and not list(settings.root.glob("*.part"))
    incomplete = replace(configured, processed_root="incomplete")
    incomplete.sources[dataset]["expected_records"] = 7
    with pytest.raises(DataError, match="incomplete source split"):
        prepare_dataset(incomplete, dataset, offline=True)
    assert not incomplete.output_dir(dataset).exists()


def test_403_fails_immediately_and_points_to_mirror_config(tmp_path):
    error = urllib.error.HTTPError("https://example.test/data", 403, "Forbidden", {}, None)
    with patch("urllib.request.urlopen", side_effect=error) as request, patch("src.data.sources.time.sleep") as sleep:
        with pytest.raises(DataError, match="configs/data.huggingface.yaml"):
            download_source(error.url, tmp_path / "nq.gz", "nq")
    assert request.call_count == 1 and sleep.call_count == 0
    assert not list(tmp_path.iterdir())


def test_429_is_retried_then_download_can_succeed(tmp_path):
    error = urllib.error.HTTPError("https://example.test/data", 429, "Too Many Requests", {}, None)
    body = gzip.compress((json.dumps(nq_record()) + "\n").encode())
    with patch("urllib.request.urlopen", side_effect=[error, io.BytesIO(body)]) as request, patch("src.data.sources.time.sleep"):
        download_source(error.url, tmp_path / "nq.gz", "nq")
    assert request.call_count == 2


def test_mirror_config_pins_publisher_revisions_and_complete_development_splits():
    path = Path(__file__).resolve().parents[1] / "configs/data.huggingface.yaml"
    settings = load_data_settings(path)
    assert settings.data_version == "clean-hf-v3-hardneg"
    assert len(settings.sources["nq"]["files"]) == 7
    assert len(settings.sources["hotpotqa"]["files"]) == 1
    for source in settings.sources.values():
        assert len(source["revision"]) == 40 and source["split"] == "validation"
        assert all(f"/resolve/{source['revision']}/" in item["url"] for item in source["files"])
    assert settings.sources["nq"]["expected_records"] == 7830
    assert settings.sources["hotpotqa"]["expected_records"] == 7405
