from __future__ import annotations

import copy
import gzip
import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.prepare_data import main
from src.data import DataError, load_data_settings, load_documents, load_evidence, load_queries, prepare_dataset, validate_dataset, validate_records
from src.data.adapters import _document, adapt_hotpotqa, adapt_nq, clean_hotpot_sentences, clean_nq_tokens
from src.data.common import contains_answer, file_sha256
from src.data.pipeline import _BM25Index, _StagingStore, _load_e5_model, allocate_strata, build_bm25_index, build_candidate_pool, rank_bm25_candidates, rank_retrieval_competition
from src.data.sources import download_source
from tests.data_factory import hotpot_record, nq_record


@pytest.fixture
def settings(tmp_path):
    fixtures = Path(__file__).parent / "fixtures" / "data"
    for name in ("nq.jsonl", "hotpotqa.json"):
        (tmp_path / name).write_bytes((fixtures / name).read_bytes())
    config = {
        "seed": 42, "data_version": "clean-v1", "schema_version": "data-v1",
        "query_count": 2, "document_count": 6, "audit_count": 2,
        "max_answer_words": 16, "processed_root": "processed",
        "sources": {
            "nq": {"version": "synthetic", "license": "test-only", "files": [{"path": "nq.jsonl"}]},
            "hotpotqa": {"version": "synthetic", "license": "test-only", "files": [{"path": "hotpotqa.json"}]},
        },
    }
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_data_settings(path, root=tmp_path)


def test_nq_unicode_entities_and_boundaries_are_mapped_during_cleaning():
    tokens = [
        {"token": "<p>", "html_token": True},
        {"token": "Cafe\u0301", "html_token": False},
        {"token": "&amp;", "html_token": False},
        {"token": "\u00a0東京  ", "html_token": False},
        {"token": "</p>", "html_token": True},
        {"token": "<table>", "html_token": True},
        {"token": "<tr>", "html_token": True},
        {"token": "One", "html_token": False},
        {"token": "</tr>", "html_token": True},
        {"token": "<tr>", "html_token": True},
        {"token": "Two", "html_token": False},
    ]
    text, offsets = clean_nq_tokens(tokens)
    assert text == "Café & 東京\n\nOne\nTwo"
    assert text[slice(*offsets[1])] == "Café"
    assert text[slice(*offsets[3])] == "東京"
    assert offsets[0] is None


def test_nq_repeated_answer_uses_annotation_not_first_text_match(settings):
    raw = nq_record()
    # Point every short annotation to the second occurrence, with its own long span.
    starts = [i for i, token in enumerate(raw["document_tokens"]) if token["token"] == "<p>"]
    start = starts[1]
    end = len(raw["document_tokens"])
    for annotation in raw["annotations"]:
        for span, a, b in ((annotation["long_answer"], start, end), (annotation["short_answers"][0], start + 1, start + 2)):
            span.update(start_token=a, end_token=b, start_byte=raw["document_tokens"][a]["start_byte"], end_byte=raw["document_tokens"][b - 1]["end_byte"])
    result = adapt_nq(raw, "nq.jsonl", settings)
    short = next(e for e in result.evidence if e["kind"] == "short_answer")
    assert short["start_char"] == result.documents[0]["text"].rindex("Paris")
    assert isinstance(result.query["source_query_id"], str)
    assert result.query["ground_truth"] == ["Paris"]
    validate_records(result.documents, [result.query], result.evidence)


@pytest.mark.parametrize("scenario,reason", [
    ("no_answer", "insufficient_answer_agreement"),
    ("multi", "multiple_short_spans"), ("yes", "yes_no_answer"),
    ("invalid", "invalid_answer_span"), ("single", "insufficient_answer_agreement"),
    ("duplicate", "duplicate_annotation_id"), ("conflict", "conflicting_answers"),
])
def test_nq_rejects_non_target_annotations(settings, scenario, reason):
    raw = nq_record()
    if scenario == "no_answer":
        for a in raw["annotations"]:
            a["short_answers"] = []
    elif scenario == "multi":
        raw["annotations"][0]["short_answers"] *= 2
    elif scenario == "yes":
        raw["annotations"][0]["yes_no_answer"] = "YES"
    elif scenario == "invalid":
        raw["annotations"][0]["short_answers"][0]["end_token"] = 1000
    elif scenario == "single":
        raw["annotations"] = raw["annotations"][:1]
    elif scenario == "duplicate":
        raw["annotations"][1]["annotation_id"] = raw["annotations"][0]["annotation_id"]
    else:
        span = raw["annotations"][0]["short_answers"][0]
        span["start_token"] += 1
        span["end_token"] += 1
        token = raw["document_tokens"][span["start_token"]]
        span.update(start_byte=token["start_byte"], end_byte=token["end_byte"])
    result = adapt_nq(raw, "nq.jsonl", settings)
    assert result.query is None and result.rejection == reason
    assert result.documents  # Excluded queries still contribute background documents.


def test_hotpot_empty_sentence_keeps_original_sentence_number(settings):
    text, offsets = clean_hotpot_sentences(["Café\r\n東京", "  ", "Next."])
    assert len(offsets) == 3 and offsets[1][0] == offsets[1][1]
    assert text[slice(*offsets[2])] == "Next."
    result = adapt_hotpotqa(hotpot_record(), "hotpotqa.json", settings)
    intermediate = next(e for e in result.evidence if e["source_ref"]["title"] == "Bridge")
    assert intermediate["source_ref"]["sentence_id"] == 2
    assert "City 1" not in intermediate["text"]
    validate_records(result.documents, [result.query], result.evidence)


@pytest.mark.parametrize("scenario,reason", [
    ("missing", "missing_support_title"), ("bounds", "support_sentence_out_of_bounds"),
    ("empty", "empty_support_sentence"), ("ambiguous", "ambiguous_support_title"),
    ("yes", "yes_no_answer"), ("long", "long_answer_text"),
])
def test_hotpot_rejects_invalid_evidence_and_answers(settings, scenario, reason):
    raw = hotpot_record()
    if scenario == "missing":
        raw["supporting_facts"][0][0] = "Unknown"
    elif scenario == "bounds":
        raw["supporting_facts"][0][1] = -1
    elif scenario == "empty":
        raw["supporting_facts"][0][1] = 1
    elif scenario == "ambiguous":
        raw["context"].append(copy.deepcopy(raw["context"][0]))
    elif scenario == "yes":
        raw["answer"] = "YES"
    else:
        raw["answer"] = "word " * 17
    result = adapt_hotpotqa(raw, "hotpotqa.json", settings)
    assert result.query is None and result.rejection == reason


def test_largest_remainder_allocations():
    assert allocate_strata({"bridge": 8, "comparison": 3}, 5) == {"bridge": 4, "comparison": 1}
    assert allocate_strata({"bridge": 1, "comparison": 1}, 1) == {"bridge": 1, "comparison": 0}


@pytest.mark.parametrize("text,answers,expected", [
    ("A Yorkshire museum.", ["York"], False),
    ("A museum in YORK.", ["York"], True),
    ("New York has a museum.", ["New York"], True),
    ("New Yorkshire has a museum.", ["New York"], False),
])
def test_answer_leakage_matches_complete_token_phrases(text, answers, expected):
    assert contains_answer(text, answers) is expected


def test_hard_negative_validation_uses_same_answer_matcher(settings):
    record = adapt_nq(nq_record(answer="York"), "nq.jsonl", settings)
    negative = _document("nq", "Museum", "A Yorkshire museum.",
                         {"raw_file": "nq.jsonl", "source_query_id": "other"}, settings)
    record.query["hard_negative_doc_ids"] = [negative["doc_id"]]
    record.query["related_doc_ids"] = sorted(record.query["evidence_doc_ids"] + [negative["doc_id"]])
    validate_records(record.documents + [negative], [record.query], record.evidence)


def test_hard_negative_selection_is_deterministic_and_excludes_answer(settings, tmp_path):
    store = _StagingStore(tmp_path / "staging.sqlite")
    docs = [
        _document("nq", "Evidence", "Acme Institute was founded by Ada Lovelace.", {"raw_file": "x", "source_query_id": "q"}, settings),
        _document("nq", "Museum", "Acme Institute opened a museum in London.", {"raw_file": "x", "source_query_id": "m"}, settings),
        _document("nq", "Funding", "Acme Institute received funding from the council.", {"raw_file": "x", "source_query_id": "f"}, settings),
        _document("nq", "AnswerLeak", "Acme Institute was founded by Ada Lovelace in 2001.", {"raw_file": "x", "source_query_id": "l"}, settings),
    ]
    for index, doc in enumerate(docs):
        store.add_document(doc, f"{index:064d}")
    query = {
        "query_id": "nq-q", "question": "Who founded the Acme Institute?",
        "ground_truth": ["Ada Lovelace"], "context_doc_ids": [docs[0]["doc_id"]],
        "evidence_doc_ids": [docs[0]["doc_id"]],
    }
    selected, report = store.choose_hard_negatives([query], 2)
    assert len(selected["nq-q"]) == 2
    assert docs[2]["doc_id"] in selected["nq-q"] or docs[1]["doc_id"] in selected["nq-q"]
    assert docs[3]["doc_id"] not in selected["nq-q"]
    assert len(report["selections"]) == 2
    assert all(item["query_id"] == "nq-q" and item["score"] > 0 for item in report["selections"])
    selected_again, _ = store.choose_hard_negatives([query], 2)
    assert selected_again == selected
    store.connection.close()


def test_strict_hard_negatives_require_anchor_type_e5_and_active_competition(settings, tmp_path, monkeypatch):
    store = _StagingStore(tmp_path / "strict.sqlite")
    record = adapt_nq(nq_record(answer="Ada"), "nq.jsonl", settings)
    evidence = record.documents[0]
    store.add_document(evidence, "0" * 64)
    candidates = []
    for index in range(4):
        doc = _document(
            "nq", f"Page 1 City Profile {index}",
            f"Page 1 city profile {index}. This city page describes a location founded in 197{index}.",
            {"raw_file": "nq.jsonl", "source_query_id": f"negative-{index}"}, settings,
        )
        candidates.append(doc)
        store.add_document(doc, f"{index + 1:064d}")
    store.add_query(record.query, record.evidence, "a" * 64)
    configured = replace(
        settings, query_count=1, hard_negative_count=4,
        hard_negative_bm25_k=50, hard_negative_active_top_k=20,
        hard_negative_min_active=2, hard_negative_min_query_cosine=0.55,
        hard_negative_min_evidence_cosine=0.60,
    )
    vectors = [[1.0, 0.0] for _ in range(6)]
    monkeypatch.setattr("src.data.pipeline._encode_e5", lambda model, texts: vectors[:len(texts)])
    selected, sampling, report = store.choose_strict_queries(configured, object())
    assert len(selected) == 1
    assert len(selected[0]["query"]["hard_negative_doc_ids"]) == 4
    assert selected[0]["query"]["hard_negative_quality"]["active_hard_negative_count"] == 4
    assert all(item["query_e5_cosine"] == 1.0 and item["evidence_e5_cosine"] == 1.0 for item in report["selections"])
    assert report["unique_hard_negative_documents"] == 4
    store.connection.close()


def test_strict_hard_negative_selection_fails_when_no_queries_qualify(settings, tmp_path, monkeypatch):
    store = _StagingStore(tmp_path / "strict-low-score.sqlite")
    record = adapt_nq(nq_record(answer="Ada"), "nq.jsonl", settings)
    store.add_document(record.documents[0], "0" * 64)
    candidate = _document(
        "nq", "Page 1 City Profile", "Page 1 city profile describes a location.",
        {"raw_file": "nq.jsonl", "source_query_id": "negative"}, settings,
    )
    store.add_document(candidate, "1" * 64)
    store.add_query(record.query, record.evidence, "a" * 64)
    configured = replace(settings, query_count=1, hard_negative_count=1)
    monkeypatch.setattr("src.data.pipeline._encode_e5", lambda model, texts: [[1.0, 0.0], [1.0, 0.0], *([[0.0, 1.0]] * (len(texts) - 2))])
    with pytest.raises(DataError, match="insufficient qualified queries"):
        store.choose_strict_queries(configured, object())
    store.connection.close()


def test_strict_candidate_pool_and_bm25_order_are_stable(settings):
    documents = [
        _document("nq", "Zeta", "Acme city profile", {"raw_file": "x", "source_query_id": "z"}, settings),
        _document("nq", "Alpha", "Acme city profile", {"raw_file": "x", "source_query_id": "a"}, settings),
    ]
    first = build_candidate_pool(documents, set())
    second = build_candidate_pool(list(reversed(documents)), set())
    assert [item["doc_id"] for item in first] == [item["doc_id"] for item in second]
    first_ranked = rank_bm25_candidates(_BM25Index(["acme city profile".split()] * 2), "Acme city", first, 2)
    second_ranked = rank_bm25_candidates(_BM25Index(["acme city profile".split()] * 2), "Acme city", second, 2)
    assert [item["doc"]["doc_id"] for item in first_ranked] == [item["doc"]["doc_id"] for item in second_ranked]


def test_strict_e5_model_path_is_required_and_local(settings):
    configured = replace(
        settings, schema_version="data-v2", hard_negative_count=1,
        hard_negative_e5_model="intfloat/e5-base-v2", hard_negative_e5_model_path="missing-model",
    )
    with pytest.raises(DataError, match="does not exist"):
        _load_e5_model(configured)


@pytest.mark.parametrize("dataset", ["nq", "hotpotqa"])
def test_strict_full_size_selection_keeps_100_queries_and_2000_documents(settings, tmp_path, monkeypatch, dataset):
    store = _StagingStore(tmp_path / f"{dataset}-full-size.sqlite")
    for index in range(100):
        if dataset == "nq":
            raw = nq_record(index + 1, answer=f"Answer {index}", title=f"Topic {index}")
            raw["question_text"] = f"Which city is linked to topic {index}?"
            adapted = adapt_nq(raw, "nq.jsonl", settings)
        else:
            raw = hotpot_record(index + 1, "bridge" if index % 2 == 0 else "comparison")
            raw["answer"] = f"Destination {index}"
            adapted = adapt_hotpotqa(raw, "hotpotqa.json", settings)
        store.add_query(adapted.query, adapted.evidence, f"{index:064d}")
        for document in adapted.documents:
            store.add_document(document, f"{index:064d}")
        anchor = f"Topic {index}" if dataset == "nq" else f"City {index + 1}"
        person = index if dataset == "nq" else index + 1
        for candidate_index in range(4):
            negative = _document(
                dataset, f"{anchor} Location Profile {candidate_index}",
                f"Person {person} is linked to {anchor}, a city location profile with regional detail {candidate_index}.",
                {"raw_file": "synthetic.json", "source_query_id": f"negative-{index}-{candidate_index}"}, settings,
            )
            store.add_document(negative, f"{1000 + index * 4 + candidate_index:064d}")
    for index in range(1600):
        background = _document(
            dataset, f"Background {index}", f"Unrelated archival background document {index}.",
            {"raw_file": "synthetic.json", "source_query_id": f"background-{index}"}, settings,
        )
        store.add_document(background, f"{5000 + index:064d}")
    configured = replace(
        settings, query_count=100, document_count=2000, hard_negative_count=4,
        hard_negative_bm25_k=50, hard_negative_active_top_k=20, hard_negative_min_active=2,
    )
    monkeypatch.setattr(
        "src.data.pipeline._encode_e5",
        lambda model, texts: [[1.0, 0.0] for _ in texts],
    )
    selected, _, report = store.choose_strict_queries(configured, object())
    assert len(selected) == 100
    assert all(len(row["query"]["hard_negative_doc_ids"]) == 4 for row in selected)
    query_records = [row["query"] for row in selected]
    mandatory = {doc_id for query in query_records for doc_id in query["context_doc_ids"] + query["hard_negative_doc_ids"]}
    documents = store.choose_documents(mandatory, 2000)
    assert len(documents) == 2000
    final_pool = build_candidate_pool(documents, set())
    index = build_bm25_index(final_pool, [query["question"] for query in query_records])
    for query in query_records:
        result = rank_retrieval_competition(
            index, query["question"], final_pool, set(query["evidence_doc_ids"]),
            set(query["hard_negative_doc_ids"]), 20,
        )
        assert result["active_hard_negative_count"] >= 2
    assert report["unique_hard_negative_documents"] == 400
    store.connection.close()


@pytest.mark.parametrize("dataset", ["nq", "hotpotqa"])
def test_end_to_end_loaders_counts_reuse_and_order_independence(settings, dataset):
    with patch("urllib.request.urlopen", side_effect=AssertionError("network used in offline mode")):
        first = prepare_dataset(settings, dataset, offline=True)
        again = prepare_dataset(settings, dataset, offline=True)
    assert first["status"] == "validated" and again["reused"] is True
    output = settings.output_dir(dataset)
    documents = load_documents(output / "documents.jsonl")
    queries = load_queries(output / "queries.jsonl")
    evidence = load_evidence(output / "evidence.jsonl")
    assert len(documents) == 6 and len(queries) == 2
    assert validate_dataset(output)["span_readback_rate"] == 1
    mandatory = {doc for q in queries for doc in q["context_doc_ids"]}
    assert mandatory <= {doc["doc_id"] for doc in documents}
    if dataset == "hotpotqa":
        assert {q["question_type"] for q in queries} == {"bridge", "comparison"}
        bridge = next(d for d in documents if d["title"] == "Bridge")
        assert len(bridge["source_refs"]) == 6
        assert len({d["doc_id"] for d in documents if d["title"] == "City"}) >= 1
    # Shuffle source record order and rebuild in another output location.
    source = settings.resolve(settings.sources[dataset]["files"][0]["path"])
    if dataset == "nq":
        source.write_text("\n".join(reversed(source.read_text(encoding="utf-8").splitlines())) + "\n", encoding="utf-8")
    else:
        rows = list(reversed(json.loads(source.read_text(encoding="utf-8"))))
        source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    other = replace(settings, processed_root="other")
    prepare_dataset(other, dataset, offline=True)
    for filename in ("documents.jsonl", "queries.jsonl", "evidence.jsonl", "audit_samples.jsonl"):
        assert file_sha256(output / filename) == file_sha256(other.output_dir(dataset) / filename)
    with pytest.raises(DataError, match="frozen inputs"):
        prepare_dataset(settings, dataset, offline=True)


def test_frozen_config_and_corrupt_outputs_are_not_overwritten(settings):
    prepare_dataset(settings, "nq", offline=True)
    with pytest.raises(DataError, match="new data_version"):
        prepare_dataset(replace(settings, seed=43), "nq", offline=True)
    path = settings.output_dir("nq") / "documents.jsonl"
    path.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(DataError, match="output hash mismatch"):
        prepare_dataset(settings, "nq", offline=True)
    assert path.read_text(encoding="utf-8") == "corrupted\n"


def test_insufficient_candidates_and_documents_leave_no_completed_or_partial_dataset(settings):
    for changed, reason in ((replace(settings, query_count=100), "eligible queries"), (replace(settings, document_count=1000), "unique documents"), (replace(settings, document_count=1), "contexts exceed")):
        with pytest.raises(DataError, match=reason):
            prepare_dataset(changed, "hotpotqa", offline=True)
        assert not settings.output_dir("hotpotqa").exists()
        assert not list(settings.output_dir("hotpotqa").parent.glob(".hotpotqa-*"))


def test_question_text_dedup_is_deterministic(settings):
    path = settings.resolve("nq.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[1]["question_text"] = "  " + records[0]["question_text"].upper() + "   "
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    prepare_dataset(settings, "nq", offline=True)
    manifest = json.loads((settings.output_dir("nq") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["duplicate_questions"] == 1


def test_independent_validation_rejects_dangling_references(settings):
    record = adapt_hotpotqa(hotpot_record(), "hotpotqa.json", settings)
    record.query["context_doc_ids"].append("missing-doc")
    with pytest.raises(DataError, match="context reference"):
        validate_records(record.documents, [record.query], record.evidence)


def test_corrupt_source_and_offline_missing_source(settings):
    path = settings.resolve("nq.jsonl")
    path.write_text("<html>error</html>", encoding="utf-8")
    with pytest.raises(DataError, match="invalid source"):
        prepare_dataset(settings, "nq", offline=True)
    assert not settings.output_dir("nq").exists()
    path.unlink()
    with patch("urllib.request.urlopen", side_effect=AssertionError("must stay offline")):
        with pytest.raises(DataError, match="offline mode"):
            prepare_dataset(settings, "nq", offline=True)


def test_invalid_download_is_never_published(tmp_path):
    destination = tmp_path / "nq.jsonl.gz"
    truncated = gzip.compress((json.dumps(nq_record()) + "\n").encode())[:-8]
    with patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(truncated)), patch("src.data.sources.time.sleep"):
        with pytest.raises(DataError, match="download failed"):
            download_source("https://example.test/nq.gz", destination, "nq")
    assert not destination.exists() and not list(tmp_path.glob("*.part"))


def test_valid_download_and_checksum_verification(tmp_path):
    destination = tmp_path / "nq.jsonl.gz"
    body = gzip.compress((json.dumps(nq_record()) + "\n").encode())
    with patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
        download_source("https://example.test/nq.gz", destination, "nq")
    assert destination.read_bytes() == body
    with patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(body)), patch("src.data.sources.time.sleep"):
        with pytest.raises(DataError, match="checksum mismatch"):
            download_source("https://example.test/nq.gz", tmp_path / "bad.gz", "nq", expected_sha256="0" * 64)


def test_cli_validate_only_does_not_read_sources_or_network(settings, monkeypatch, capsys):
    monkeypatch.setattr("scripts.prepare_data.load_data_settings", lambda path: settings)
    prepare_dataset(settings, "nq", offline=True)
    settings.resolve("nq.jsonl").unlink()
    with patch("urllib.request.urlopen", side_effect=AssertionError("network used")):
        assert main(["--dataset", "nq", "--validate-only"]) == 0
    assert '"span_readback_rate": 1.0' in capsys.readouterr().out
    changed = replace(settings, seed=43)
    monkeypatch.setattr("scripts.prepare_data.load_data_settings", lambda path: changed)
    assert main(["--dataset", "nq", "--validate-only"]) == 1


def test_default_configuration_matches_frozen_plan():
    settings = load_data_settings()
    assert (settings.seed, settings.query_count, settings.document_count, settings.audit_count) == (42, 100, 2000, 20)
    assert len(settings.sources["nq"]["files"]) == 5


def test_nq_byte_annotations_must_agree_with_tokens(settings):
    raw = nq_record(answer="東京")
    result = adapt_nq(raw, "nq.jsonl", settings)
    assert result.query["ground_truth"] == ["東京"]
    validate_records(result.documents, [result.query], result.evidence)
    for annotation in raw["annotations"]:
        annotation["short_answers"][0]["end_byte"] += 1
    assert adapt_nq(raw, "nq.jsonl", settings).rejection == "invalid_answer_span"


def test_empty_documents_are_counted_and_excluded(settings):
    raw = hotpot_record()
    raw["context"].append(["Empty", ["  ", "\r\n"]])
    result = adapt_hotpotqa(raw, "hotpotqa.json", settings)
    assert result.empty_documents == 1 and len(result.documents) == 3
    raw_nq = nq_record()
    raw_nq["document_tokens"] = []
    result_nq = adapt_nq(raw_nq, "nq.jsonl", settings)
    assert result_nq.empty_documents == 1 and result_nq.documents == []


def test_same_title_variants_never_rebind_support(settings):
    a = adapt_hotpotqa(hotpot_record(1), "hotpotqa.json", settings)
    b = adapt_hotpotqa(hotpot_record(2), "hotpotqa.json", settings)
    city_a = next(doc for doc in a.documents if doc["title"] == "City")
    city_b = next(doc for doc in b.documents if doc["title"] == "City")
    assert city_a["doc_id"] != city_b["doc_id"]
    support_b = next(e for e in b.evidence if e["source_ref"]["title"] == "City")
    assert support_b["doc_id"] == city_b["doc_id"] and "Person 2" in support_b["text"]


def test_local_expected_checksum_is_enforced(settings):
    sources = copy.deepcopy(settings.sources)
    sources["nq"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(DataError, match="checksum mismatch"):
        prepare_dataset(replace(settings, sources=sources), "nq", offline=True)
    assert not settings.output_dir("nq").exists()


@pytest.mark.parametrize("field,value", [
    ("seed", -1), ("query_count", True), ("audit_count", 1000),
    ("data_version", "../escape"), ("schema_version", "unknown"), ("sources", {}),
])
def test_invalid_configuration_is_rejected(settings, field, value):
    path = settings.root / "data.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config[field] = value
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DataError):
        load_data_settings(path, root=settings.root)


@pytest.mark.parametrize("dataset", ["nq", "hotpotqa"])
def test_default_scale_contract_using_synthetic_records(settings, dataset):
    """Exercise the 100/2,000/20 contract without substituting benchmark data."""
    configured = replace(settings, query_count=100, document_count=2000, audit_count=20)
    path = settings.resolve(settings.sources[dataset]["files"][0]["path"])
    if dataset == "nq":
        with path.open("w", encoding="utf-8") as handle:
            for index in range(1, 2001):
                handle.write(json.dumps(nq_record(index)) + "\n")
    else:
        rows = [hotpot_record(index, "bridge" if index <= 800 else "comparison") for index in range(1, 1001)]
        path.write_text(json.dumps(rows), encoding="utf-8")
    report = prepare_dataset(configured, dataset, offline=True)
    assert (report["document_count"], report["query_count"], report["span_readback_rate"]) == (2000, 100, 1.0)
    manifest = json.loads((configured.output_dir(dataset) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["audit_count"] == 20
    if dataset == "hotpotqa":
        assert manifest["sampling"]["allocated_question_types"] == {"bridge": 80, "comparison": 20}
