"""Convert original dataset annotations without searching for answer substrings."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from typing import Any

from .common import DataError, answer_key, digest, normalize_body, normalize_inline
from .config import DataSettings


@dataclass
class AdaptedRecord:
    documents: list[dict]
    query: dict | None
    evidence: list[dict]
    rejection: str | None = None
    empty_documents: int = 0


def _document(dataset: str, title: str, text: str, source: dict, settings: DataSettings) -> dict:
    title = normalize_inline(title)
    identifier = f"{dataset}-doc-{digest([dataset, title, text])}"
    return {
        "doc_id": identifier, "dataset": dataset, "title": title, "text": text,
        "source_doc_id": identifier, "source_refs": [source],
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "document_role": "candidate",
        "is_poison": False, "corruption_type": None, "true_fact": None,
        "false_fact": None, "target_query_ids": [], "poison_version": settings.data_version,
        "data_version": settings.data_version, "schema_version": settings.schema_version,
    }


def _query(dataset: str, identifier: str, question: str, answers: list[str], docs: list[str], evidence: list[dict], settings: DataSettings, **extra: Any) -> dict:
    return {
        "query_id": f"{dataset}-{identifier}", "dataset": dataset,
        "source_query_id": identifier, "question": normalize_inline(question),
        "ground_truth": sorted(set(answers)), "context_doc_ids": sorted(set(docs)),
        "evidence_doc_ids": sorted({item["doc_id"] for item in evidence}),
        "hard_negative_doc_ids": [],
        "related_doc_ids": sorted(set(docs) | {item["doc_id"] for item in evidence}),
        "data_version": settings.data_version, "schema_version": settings.schema_version,
        **extra,
    }


def _evidence(query_id: str, doc_id: str, kind: str, text: str, start: int, end: int, source: dict, settings: DataSettings) -> dict:
    value = {
        "query_id": query_id, "doc_id": doc_id, "kind": kind,
        "start_char": start, "end_char": end, "text": text[start:end],
        "source_ref": source, "data_version": settings.data_version,
        "schema_version": settings.schema_version,
    }
    return {"evidence_id": "ev-" + digest(value), **value}


def clean_nq_tokens(tokens: list[dict]) -> tuple[str, list[tuple[int, int] | None]]:
    """Construct the final text and offsets together; HTML tokens map to None."""
    pieces: list[str] = []
    offsets: list[tuple[int, int] | None] = []
    cursor = 0
    separator = " "
    for item in tokens:
        if not isinstance(item, dict) or not isinstance(item.get("token"), str) or type(item.get("html_token")) is not bool:
            raise DataError("NQ document_tokens contains an invalid token")
        token = item["token"]
        if item["html_token"]:
            tag = re.match(r"</?([a-zA-Z0-9]+)", token)
            name = tag.group(1).lower() if tag else ""
            boundary = "\n\n" if name in {"p", "div", "section", "article", "table", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"} else "\n" if name in {"li", "tr", "br"} else " "
            if len(boundary) > len(separator) or (boundary == "\n" and separator == " "):
                separator = boundary
            offsets.append(None)
            continue
        normalized = normalize_inline(html.unescape(token))
        if not normalized:
            offsets.append(None)
            continue
        if pieces:
            pieces.append(separator)
            cursor += len(separator)
        start = cursor
        pieces.append(normalized)
        cursor += len(normalized)
        offsets.append((start, cursor))
        separator = " "
    return "".join(pieces), offsets


def _nq_span(span: dict, offsets: list, html_bytes: bytes, tokens: list[dict]) -> tuple[int, int, str]:
    start, end = span.get("start_token"), span.get("end_token")
    byte_start, byte_end = span.get("start_byte"), span.get("end_byte")
    if not all(type(value) is int for value in (start, end, byte_start, byte_end)):
        raise DataError("NQ span has non-integer offsets")
    if not 0 <= start < end <= len(offsets) or not 0 <= byte_start < byte_end <= len(html_bytes):
        raise DataError("NQ span is outside the source document")
    if byte_start != tokens[start].get("start_byte") or byte_end != tokens[end - 1].get("end_byte"):
        raise DataError("NQ byte and token annotations disagree")
    visible = [item for item in offsets[start:end] if item is not None]
    if not visible:
        raise DataError("NQ span contains no visible text")
    fragment = html_bytes[byte_start:byte_end].decode("utf-8")
    original = normalize_inline(html.unescape(re.sub(r"<[^>]*>", "", fragment)))
    return visible[0][0], visible[-1][1], original


def adapt_nq(raw: dict, filename: str, settings: DataSettings) -> AdaptedRecord:
    identifier = str(raw["example_id"])
    if not isinstance(raw.get("document_html"), str) or not isinstance(raw.get("document_tokens"), list):
        raise DataError(f"NQ {identifier}: original HTML and tokens are required")
    text, offsets = clean_nq_tokens(raw["document_tokens"])
    source = {"raw_file": filename, "source_query_id": identifier, "url": raw.get("document_url"), "title": raw.get("document_title", "")}
    if not text:
        return AdaptedRecord([], None, [], "empty_document", 1)
    doc = _document("nq", raw.get("document_title") or raw.get("document_url") or "", text, source, settings)

    def reject(reason: str) -> AdaptedRecord:
        return AdaptedRecord([doc], None, [], reason)

    if not isinstance(raw.get("question_text"), str) or not normalize_inline(raw["question_text"]):
        return reject("empty_question")
    annotations = raw.get("annotations")
    if not isinstance(annotations, list):
        raise DataError(f"NQ {identifier}: annotations must be a list")
    accepted = []
    keys = set()
    seen_annotations = set()
    html_bytes = raw["document_html"].encode("utf-8")
    for annotation in annotations:
        if not isinstance(annotation, dict):
            return reject("invalid_annotation")
        if annotation.get("yes_no_answer", "NONE") != "NONE":
            return reject("yes_no_answer")
        shorts = annotation.get("short_answers", [])
        if not isinstance(shorts, list):
            return reject("invalid_annotation")
        if len(shorts) > 1:
            return reject("multiple_short_spans")
        if not shorts:
            continue
        if "annotation_id" not in annotation:
            return reject("missing_annotation_id")
        annotation_id = str(annotation["annotation_id"])
        if annotation_id in seen_annotations:
            return reject("duplicate_annotation_id")
        seen_annotations.add(annotation_id)
        short = shorts[0]
        long = annotation.get("long_answer", {})
        try:
            short_start, short_end, original = _nq_span(short, offsets, html_bytes, raw["document_tokens"])
            long_start, long_end, _ = _nq_span(long, offsets, html_bytes, raw["document_tokens"])
            if not long["start_token"] <= short["start_token"] < short["end_token"] <= long["end_token"]:
                return reject("short_outside_long_answer")
        except (DataError, KeyError, TypeError, AttributeError, UnicodeError):
            return reject("invalid_answer_span")
        clean_answer = text[short_start:short_end]
        if not original or not clean_answer:
            return reject("empty_answer")
        if max(len(original.split()), len(clean_answer.split())) > settings.max_answer_words:
            return reject("long_answer_text")
        keys.add(answer_key(original))
        accepted.append((annotation_id, short, long, short_start, short_end, long_start, long_end, original, clean_answer))
    if len(keys) > 1:
        return reject("conflicting_answers")
    if len(accepted) < 2:
        return reject("insufficient_answer_agreement")
    evidence = []
    answers = []
    for annotation_id, short, long, short_start, short_end, long_start, long_end, original, clean_answer in accepted:
        answers.extend((original, clean_answer))
        for kind, span, start, end in (("short_answer", short, short_start, short_end), ("long_answer", long, long_start, long_end)):
            reference = {
                **source, "annotation_id": annotation_id,
                **{key: span[key] for key in ("start_token", "end_token", "start_byte", "end_byte")},
            }
            if kind == "short_answer":
                reference["original_answer_text"] = original
            evidence.append(_evidence(f"nq-{identifier}", doc["doc_id"], kind, text, start, end, reference, settings))
    query = _query("nq", identifier, raw["question_text"], answers, [doc["doc_id"]], evidence, settings, question_type="single_span", answer_annotation_count=len(accepted))
    return AdaptedRecord([doc], query, evidence)


def clean_hotpot_sentences(sentences: list[str]) -> tuple[str, list[tuple[int, int]]]:
    pieces = []
    offsets = []
    cursor = 0
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise DataError("HotpotQA sentences must be strings")
        clean = normalize_body(sentence)
        if clean:
            if pieces:
                pieces.append(" ")
                cursor += 1
            start = cursor
            pieces.append(clean)
            cursor += len(clean)
            offsets.append((start, cursor))
        else:
            offsets.append((cursor, cursor))
    return "".join(pieces), offsets


def adapt_hotpotqa(raw: dict, filename: str, settings: DataSettings) -> AdaptedRecord:
    identifier = str(raw["_id"])
    if not isinstance(raw.get("context"), list):
        raise DataError(f"HotpotQA {identifier}: context must be a list")
    documents = []
    contexts: dict[str, list[tuple[int, dict | None, list]]] = {}
    empty_count = 0
    for position, entry in enumerate(raw["context"]):
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str) or not isinstance(entry[1], list):
            raise DataError(f"HotpotQA {identifier}: invalid context entry")
        title, sentences = entry
        text, offsets = clean_hotpot_sentences(sentences)
        doc = None
        if text:
            source = {"raw_file": filename, "source_query_id": identifier, "title": title, "context_index": position}
            doc = _document("hotpotqa", title, text, source, settings)
            documents.append(doc)
        else:
            empty_count += 1
        contexts.setdefault(title, []).append((position, doc, offsets))

    def reject(reason: str) -> AdaptedRecord:
        return AdaptedRecord(documents, None, [], reason, empty_count)

    if not isinstance(raw.get("question"), str) or not normalize_inline(raw["question"]):
        return reject("empty_question")
    answer = raw.get("answer")
    if not isinstance(answer, str) or not normalize_inline(answer):
        return reject("empty_answer")
    if answer_key(answer) in {"yes", "no"}:
        return reject("yes_no_answer")
    if len(answer.split()) > settings.max_answer_words:
        return reject("long_answer_text")
    if raw.get("type") not in {"bridge", "comparison"}:
        return reject("unknown_question_type")
    supports = raw.get("supporting_facts")
    if not isinstance(supports, list) or not supports:
        return reject("missing_supporting_facts")
    evidence = []
    seen = set()
    for entry in supports:
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str) or type(entry[1]) is not int:
            return reject("invalid_supporting_fact")
        title, sentence_id = entry
        matches = contexts.get(title, [])
        if not matches:
            return reject("missing_support_title")
        if len(matches) != 1:
            return reject("ambiguous_support_title")
        position, doc, offsets = matches[0]
        if sentence_id < 0 or sentence_id >= len(offsets):
            return reject("support_sentence_out_of_bounds")
        start, end = offsets[sentence_id]
        if doc is None or start == end:
            return reject("empty_support_sentence")
        if (title, sentence_id) in seen:
            continue
        seen.add((title, sentence_id))
        reference = {"raw_file": filename, "source_query_id": identifier, "title": title, "context_index": position, "sentence_id": sentence_id}
        evidence.append(_evidence(f"hotpotqa-{identifier}", doc["doc_id"], "supporting_sentence", doc["text"], start, end, reference, settings))
    query = _query("hotpotqa", identifier, raw["question"], [answer], [doc["doc_id"] for doc in documents], evidence, settings, question_type=raw["type"], level=raw.get("level"))
    return AdaptedRecord(documents, query, evidence, empty_documents=empty_count)
