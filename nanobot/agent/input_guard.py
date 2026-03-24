"""Lightweight local input compression and clarification heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

InputGuardMode = Literal["off", "light", "guide", "strict"]

_CODE_FENCE_RE = re.compile(r"(```[\s\S]*?```)")
_SPACES_RE = re.compile(r"[ \t]{2,}")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_PUNCTUATION_RE = re.compile(r"([!?！？。．…,.，])\1{2,}")
_LINE_BREAK_RE = re.compile(r"\r\n?")

_VAGUE_PATTERNS = (
    re.compile(r"^(这个|那个|它|这块|这边|这里)(呢|啊|呀|咋办|怎么弄|怎么搞)?[!?？。 ]*$"),
    re.compile(r"^(帮我)?(看下|看看|分析下|优化下|改下|处理下)(这个|那个|它)?[!?？。 ]*$"),
    re.compile(r"^(咋办|怎么办|怎么搞|怎么弄)[!?？。 ]*$"),
    re.compile(r"^(this|that|it)(\s+(please|pls))?[!?., ]*$", re.IGNORECASE),
    re.compile(r"^(help me|look at this|optimize this|fix this|what about this)[!?., ]*$", re.IGNORECASE),
)

_STRUCTURED_HINTS = (
    "goal", "scope", "context", "output", "constraint", "constraints",
    "目标", "范围", "上下文", "输入", "输出", "约束", "限制",
)


@dataclass(frozen=True)
class InputGuardResult:
    """Normalized input and optional local clarification response."""

    content: str
    clarification_message: str | None = None


def preprocess_input(text: str, *, mode: InputGuardMode = "guide") -> InputGuardResult:
    """Apply local compression and optionally short-circuit with a clarification prompt."""
    if mode == "off":
        return InputGuardResult(content=text)

    compact = _compress_text(text, dedupe=(mode == "strict"))
    if mode in {"guide", "strict"} and _needs_clarification(compact, strict=(mode == "strict")):
        return InputGuardResult(content=compact, clarification_message=_clarification_message(compact))
    return InputGuardResult(content=compact)


def _compress_text(text: str, *, dedupe: bool) -> str:
    """Compress obviously redundant text without touching fenced code blocks."""
    normalized = _LINE_BREAK_RE.sub("\n", text)
    parts = _CODE_FENCE_RE.split(normalized)
    for idx, part in enumerate(parts):
        if idx % 2 == 0:
            parts[idx] = _compress_plaintext(part, dedupe=dedupe)
    return "".join(parts)


def _compress_plaintext(text: str, *, dedupe: bool) -> str:
    """Compress redundant whitespace and repeated prose lines."""
    leading_newlines = len(text) - len(text.lstrip("\n"))
    trailing_newlines = len(text) - len(text.rstrip("\n"))
    core = text.strip("\n")
    if not core:
        if leading_newlines or trailing_newlines:
            return "\n" * min(max(leading_newlines, trailing_newlines), 2)
        return ""

    lines = core.split("\n")
    out: list[str] = []
    previous = None
    repeat_count = 0

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            if out and out[-1] != "":
                out.append("")
            continue

        line = _SPACES_RE.sub(" ", line.strip())
        line = _PUNCTUATION_RE.sub(lambda m: m.group(1) * 2, line)

        if dedupe and line == previous:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            previous = line
            repeat_count = 0

        out.append(line)

    compact = _BLANK_LINES_RE.sub("\n\n", "\n".join(out).strip())
    prefix = "\n" * min(leading_newlines, 2)
    suffix = "\n" * min(trailing_newlines, 2)
    return f"{prefix}{compact}{suffix}"


def _needs_clarification(text: str, *, strict: bool) -> bool:
    """Detect obviously vague requests that should be clarified locally."""
    candidate = text.strip()
    if not candidate or candidate.startswith("/"):
        return False
    if "```" in candidate:
        return False

    lowered = candidate.lower()
    if any(hint in lowered for hint in _STRUCTURED_HINTS) or ":" in candidate or "|" in candidate:
        return False

    for pattern in _VAGUE_PATTERNS:
        if pattern.fullmatch(candidate):
            return True

    if strict:
        words = [part for part in re.split(r"\s+", lowered) if part]
        compact_chars = re.sub(r"\s+", "", candidate)
        if len(words) <= 3 or len(compact_chars) <= 10:
            return True

    return False


def _clarification_message(text: str) -> str:
    """Build a local clarification prompt without calling the model."""
    preview = text if len(text) <= 80 else f"{text[:77]}..."
    return (
        "I have not sent your last message to the model yet because it looks too vague to answer reliably.\n\n"
        f"Last message: {preview}\n\n"
        "Please resend it in one compact format so I can continue while saving tokens:\n"
        "1. Goal | Input/Context | Expected output\n"
        "2. Question | Constraints | What you've tried\n"
        "3. Decision to make | Options A/B/C | Success criteria\n\n"
        'Tip: avoid messages that only say "this / that / optimize it / 看下这个".'
    )
