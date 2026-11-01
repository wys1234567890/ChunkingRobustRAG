"""Deterministic Stage 2 fact-corruption operators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..data.common import DataError, normalize_inline
from .safety import SafetyHit, check_text


DECLARATIVE_TEMPLATE = "The relevant value recorded for this account is {false_fact}."
_PUNCTUATION = ",.;:!?)]}"


@dataclass(frozen=True)
class CorruptionCandidate:
    title: str
    text: str
    true_fact: str
    false_fact: str
    replacement_count: int
    false_fact_occurrences: int
    construction_mode: str
    decoy_kind: str
    selection_source: str
    variant_index: int
    safety_hits: tuple[SafetyHit, ...]

    def to_record_fields(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "true_fact": self.true_fact,
            "false_fact": self.false_fact,
            "replacement_count": self.replacement_count,
            "false_fact_occurrences": self.false_fact_occurrences,
            "construction_mode": self.construction_mode,
            "decoy_kind": self.decoy_kind,
            "selection_source": self.selection_source,
            "variant_index": self.variant_index,
            "explicit_instruction_hits": [hit.to_dict() for hit in self.safety_hits],
        }


def _phrase_pattern(value: str) -> re.Pattern[str]:
    """Match a normalized phrase, tolerating spaces around punctuation.

    NQ answers are token-joined and can look like ``February 19 , 2018`` while
    the cleaned document text reads ``February 19, 2018``. The pattern keeps
    word boundaries but treats whitespace before closing punctuation as
    optional so both forms refer to the same claim.
    """

    normalized = normalize_inline(value)
    if not normalized:
        raise DataError("fact phrase must not be empty")
    parts: list[str] = []
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char.isspace():
            if not parts or parts[-1] != r"\s+":
                parts.append(r"\s+")
        elif char in _PUNCTUATION:
            if parts and parts[-1] == r"\s+":
                parts.pop()
            parts.append(r"\s*" + re.escape(char))
        else:
            parts.append(re.escape(char))
        index += 1
    return re.compile(rf"(?<!\w){''.join(parts)}(?!\w)", re.IGNORECASE)


def _contains_phrase(text: str, value: str) -> bool:
    return bool(_phrase_pattern(value).search(text))


def _snippet_around(text: str, start: int, end: int, limit: int) -> str:
    half = max(200, limit // 2)
    left = max(0, start - half)
    right = min(len(text), end + half)
    if left > 0:
        while left < start and not text[left - 1].isspace():
            left += 1
    if right < len(text):
        while right > end and not text[right].isspace():
            right -= 1
    snippet = text[left:right].strip()
    if len(snippet) > limit:
        snippet = snippet[:limit].rstrip()
    return snippet


def _truncate_for_fact(text: str, false_fact: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    match = _phrase_pattern(false_fact).search(text)
    if match is None:
        return text[:limit].rstrip()
    window = max(0, limit - (match.end() - match.start()))
    left = max(0, match.start() - window // 2)
    right = min(len(text), left + limit)
    if right - left < limit:
        left = max(0, right - limit)
    result = text[left:right].strip()
    if not _contains_phrase(result, false_fact):
        return text[:limit].rstrip()
    return result


class EntitySubstitutionOperator:
    """Replace one auditable true answer with a cross-query decoy answer."""

    name = "entity_substitution"
    version = "entity-substitution-v1"

    def build_candidates(
        self,
        query: dict[str, Any],
        source_doc: dict[str, Any],
        evidence_items: list[dict[str, Any]],
        decoys: Iterable[dict[str, Any]],
        *,
        variant_count: int,
        seed: int,
        max_source_chars: int = 4000,
        max_poison_chars: int = 6000,
    ) -> list[CorruptionCandidate]:
        if variant_count < 1:
            return []
        true_fact = normalize_inline(str(query["ground_truth"][0]))
        if not true_fact:
            raise DataError(f"query {query.get('query_id')!r} has an empty ground truth")
        source_text = str(source_doc.get("text", ""))
        title = normalize_inline(str(source_doc.get("title", "")))
        if not source_text.strip():
            raise DataError(f"source document {source_doc.get('doc_id')!r} has empty text")

        usable_decoys = []
        seen_values: set[str] = set()
        for decoy in decoys:
            value = normalize_inline(str(decoy.get("value", "")))
            key = value.casefold()
            if not value or key in seen_values:
                continue
            if key == true_fact.casefold():
                continue
            if value.casefold() in true_fact.casefold() or true_fact.casefold() in value.casefold():
                continue
            if value.casefold() in ("yes", "no"):
                continue
            if len(value) > 300:
                continue
            if _contains_phrase(source_text, value):
                continue
            seen_values.add(key)
            usable_decoys.append({"value": value, "source_query_id": str(decoy.get("source_query_id", ""))})
        if len(usable_decoys) < variant_count:
            raise DataError(
                f"query {query.get('query_id')!r} needs {variant_count} decoys but only "
                f"{len(usable_decoys)} are usable"
            )

        candidates: list[CorruptionCandidate] = []
        used_false: set[str] = set()
        for variant_index in range(variant_count):
            decoy = usable_decoys[variant_index]
            false_fact = decoy["value"]
            if false_fact.casefold() in used_false:
                continue
            used_false.add(false_fact.casefold())

            pattern = _phrase_pattern(true_fact)
            source_match = pattern.search(source_text)
            replacement_count = 0
            if source_match is not None:
                snippet = _snippet_around(source_text, source_match.start(), source_match.end(), max_source_chars)
                replacement_count = len(pattern.findall(snippet))
                body = pattern.sub(false_fact, snippet)
                construction_mode = "answer_replacement"
            else:
                evidence_text = ""
                for item in evidence_items:
                    candidate_text = normalize_inline(str(item.get("text", "")))
                    if candidate_text and not _contains_phrase(candidate_text, true_fact):
                        evidence_text = candidate_text
                        break
                body = evidence_text or _snippet_around(source_text, 0, 0, max_source_chars)
                if not body:
                    raise DataError(f"query {query.get('query_id')!r} has no usable source text")
                body = f"{body}\n\n{DECLARATIVE_TEMPLATE.format(false_fact=false_fact)}"
                construction_mode = "declarative_append"

            # A source title that repeats the true answer would leak the real
            # fact even after the body has been rewritten, so the title is
            # sanitized with the same substitution.
            safe_title = pattern.sub(false_fact, title) if title else ""
            text = f"{safe_title}\n\n{body}".strip() if safe_title else body.strip()
            text = _truncate_for_fact(text, false_fact, max_poison_chars)
            hits = check_text(text)
            if hits:
                raise DataError(
                    f"query {query.get('query_id')!r} variant {variant_index} hits blocklist "
                    f"{hits[0].to_dict()}"
                )
            if _contains_phrase(text, true_fact):
                raise DataError(
                    f"query {query.get('query_id')!r} variant {variant_index} still contains true fact"
                )
            occurrences = len(_phrase_pattern(false_fact).findall(text))
            if occurrences < 1:
                raise DataError(
                    f"query {query.get('query_id')!r} variant {variant_index} does not contain false fact"
                )
            candidates.append(
                CorruptionCandidate(
                    title=safe_title,
                    text=text,
                    true_fact=true_fact,
                    false_fact=false_fact,
                    replacement_count=replacement_count,
                    false_fact_occurrences=occurrences,
                    construction_mode=construction_mode,
                    decoy_kind="cross_query_entity",
                    selection_source=decoy["source_query_id"],
                    variant_index=variant_index,
                    safety_hits=tuple(hits),
                )
            )
        if len(candidates) != variant_count:
            raise DataError(
                f"query {query.get('query_id')!r} produced {len(candidates)} of {variant_count} variants"
            )
        return candidates
