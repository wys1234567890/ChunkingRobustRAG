"""Versioned static checks for explicit prompt-injection language."""

from __future__ import annotations

import re
from dataclasses import dataclass

BLOCKLIST_VERSION = "explicit-instruction-v1"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_previous", re.compile(r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\b", re.IGNORECASE)),
    ("disregard_previous", re.compile(r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\b", re.IGNORECASE)),
    ("follow_instructions", re.compile(r"\bfollow\s+(?:these|the)\s+instructions?\b", re.IGNORECASE)),
    ("system_message", re.compile(r"\bsystem\s+message\b", re.IGNORECASE)),
    ("developer_message", re.compile(r"\bdeveloper\s+message\b", re.IGNORECASE)),
    ("user_message", re.compile(r"\buser\s+message\b", re.IGNORECASE)),
    ("chatgpt_role", re.compile(r"\byou\s+are\s+(?:chatgpt|an?\s+(?:ai|assistant|language\s+model))\b", re.IGNORECASE)),
    ("role_play", re.compile(r"\brole[-\s]?play\b", re.IGNORECASE)),
    ("act_as_role", re.compile(r"\bact\s+as\s+(?:a|an|the)\s+(?:chatgpt|ai|assistant|system|model|expert)\b", re.IGNORECASE)),
    ("answer_only", re.compile(r"\b(?:answer|output)\s+only\b", re.IGNORECASE)),
    ("always_answer", re.compile(r"\balways\s+answer\b", re.IGNORECASE)),
    ("do_not_answer", re.compile(r"\bdo\s+not\s+answer\b", re.IGNORECASE)),
    ("override_context", re.compile(r"\boverride\s+(?:the\s+)?(?:context|instructions?)\b", re.IGNORECASE)),
    ("prompt_injection", re.compile(r"\bprompt\s+injection\b", re.IGNORECASE)),
    ("context_above", re.compile(r"\bcontext\s+above\b", re.IGNORECASE)),
    ("previous_context", re.compile(r"\bprevious\s+context\b", re.IGNORECASE)),
    ("zh_ignore", re.compile(r"(?:忽略|无视)(?:之前|以上|前面|所有)?(?:的)?(?:上下文|指令|提示)")),
    ("zh_follow", re.compile(r"遵循(?:以下|下面|这些)?指令")),
    ("zh_system", re.compile(r"(?:系统|开发者|用户)消息")),
    ("zh_role", re.compile(r"你是一个?(?:AI|人工智能|助手)")),
    ("zh_answer_only", re.compile(r"只(?:回答|输出)")),
    ("zh_override", re.compile(r"覆盖(?:上下文|指令)")),
    ("zh_previous_context", re.compile(r"之前的上下文")),
    ("zh_role_play", re.compile(r"角色扮演")),
)


@dataclass(frozen=True)
class SafetyHit:
    rule_id: str
    start: int
    end: int
    phrase: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "start": self.start,
            "end": self.end,
            "phrase": self.phrase,
        }


def check_text(text: str) -> list[SafetyHit]:
    if not isinstance(text, str):
        return [SafetyHit("invalid_text", 0, 0, "")]
    hits: list[SafetyHit] = []
    for rule_id, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            hits.append(SafetyHit(rule_id, match.start(), match.end(), match.group(0)))
    return sorted(hits, key=lambda hit: (hit.start, hit.end, hit.rule_id))


def assert_safe(text: str) -> None:
    hits = check_text(text)
    if hits:
        raise ValueError(f"explicit instruction blocklist hit: {hits[0].to_dict()}")
