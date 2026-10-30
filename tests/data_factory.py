"""Synthetic original-format examples; these are not benchmark observations."""

from __future__ import annotations

import re


def nq_record(identifier: int = 1, answer: str = "Paris", *, title: str | None = None) -> dict:
    document_html = f"<h1>Page {identifier}</h1><p>{answer} is a city .</p><p>{answer} appears again .</p>"
    tokens = []
    for match in re.finditer(r"<[^>]+>|[^\s<>]+", document_html):
        tokens.append({
            "token": match.group(), "html_token": match.group().startswith("<"),
            "start_byte": len(document_html[:match.start()].encode("utf-8")),
            "end_byte": len(document_html[:match.end()].encode("utf-8")),
        })
    start = next(index for index, token in enumerate(tokens) if token["token"] == "<p>")
    end = next(index + 1 for index, token in enumerate(tokens) if token["token"] == "</p>")
    long_span = {"start_token": start, "end_token": end, "start_byte": tokens[start]["start_byte"], "end_byte": tokens[end - 1]["end_byte"], "candidate_index": 0}
    short_span = {"start_token": start + 1, "end_token": start + 2, "start_byte": tokens[start + 1]["start_byte"], "end_byte": tokens[start + 1]["end_byte"]}
    return {
        "example_id": 9007199254740992 + identifier,
        "question_text": f"Which city is described in page {identifier}?",
        "document_title": title or f"Page {identifier}",
        "document_url": f"https://en.wikipedia.org/wiki/Page_{identifier}?oldid={identifier}",
        "document_html": document_html, "document_tokens": tokens,
        "long_answer_candidates": [long_span],
        "annotations": [{"annotation_id": identifier * 10 + i, "long_answer": dict(long_span), "short_answers": [dict(short_span)], "yes_no_answer": "NONE"} for i in range(5)],
    }


def hotpot_record(identifier: int = 1, question_type: str = "bridge") -> dict:
    return {
        "_id": f"synthetic-{identifier}", "question": f"Which city is linked to person {identifier}?",
        "answer": f"City {identifier}", "type": question_type, "level": "hard",
        "context": [
            ["Bridge", ["A shared intermediate fact.", "", "The relation requires another document."]],
            ["City", [f"Person {identifier} is linked to City {identifier}."]],
            [f"Noise {identifier}", [f"An irrelevant paragraph numbered {identifier}."]],
        ],
        "supporting_facts": [["Bridge", 2], ["City", 0]],
    }
