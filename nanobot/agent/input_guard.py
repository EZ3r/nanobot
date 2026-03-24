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
_MAX_PUNCT_REPEAT = 2
_MAX_IDENTICAL_LINES = 2
# Match either 2+ contiguous CJK characters or identifier-like Latin tokens.
_DETAIL_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")
_STEP_MARKER_RE = re.compile(r"(?:^|[；;。]\s*|\s+)(?:\d+[\.:：]|[①②③④⑤⑥⑦⑧⑨⑩])\s*")

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
    if mode in {"guide", "strict"}:
        compact = _semantic_compact_text(compact)
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
        line = _PUNCTUATION_RE.sub(lambda m: m.group(1) * _MAX_PUNCT_REPEAT, line)

        if dedupe and line == previous:
            repeat_count += 1
            if repeat_count >= _MAX_IDENTICAL_LINES:
                continue
        else:
            previous = line
            repeat_count = 0

        out.append(line)

    compact = _BLANK_LINES_RE.sub("\n\n", "\n".join(out).strip())
    prefix = "\n" * min(leading_newlines, 2)
    suffix = "\n" * min(trailing_newlines, 2)
    return f"{prefix}{compact}{suffix}"


def _semantic_compact_text(text: str) -> str:
    """Rewrite a few common verbose requests into a more compact local form."""
    parts = _CODE_FENCE_RE.split(text)
    for idx, part in enumerate(parts):
        if idx % 2 == 0:
            parts[idx] = _semantic_compact_plaintext(part)
    return "".join(parts)


def _semantic_compact_plaintext(text: str) -> str:
    """Apply lightweight intent-preserving rewrites for common verbose prose."""
    stripped = text.strip()
    if not stripped or "\n" in stripped:
        return text

    for rewriter in (_compact_weather_request, _compact_move_request):
        rewritten = rewriter(stripped)
        if rewritten and rewritten != stripped:
            return rewritten
    return text


def _compact_weather_request(text: str) -> str | None:
    """Compress weather requests with filler phrases into a direct query."""
    if "天气" not in text or not any(day in text for day in ("今天", "明天", "后天")):
        return None

    location = None
    for pattern in (
        r"(?:哦对|对了)?我在(?P<location>[^，,。；;!?？]+)",
        r"(?:查询|看看|查下|查一下)(?P<location>[^，,。；;!?？]+?)(?:今天|明天|后天)的天气",
    ):
        if match := re.search(pattern, text):
            location = _strip_filler(match.group("location"))
            break

    days = [day for day in ("今天", "明天", "后天") if day in text]
    if not days:
        return None

    day_part = "和".join(days)
    location_part = f"{location}" if location else ""
    return f"查询{location_part}{day_part}的天气"


def _compact_move_request(text: str) -> str | None:
    """Compress explicit cut/paste folder steps into a direct move request."""
    if "文件夹" not in text or not any(keyword in text for keyword in ("剪切", "粘贴", "移动")):
        return None
    if not _STEP_MARKER_RE.search(text):
        return None

    folders = re.findall(r"打开\s*([A-Za-z0-9_\-/\u4e00-\u9fff]+)\s*文件夹", text)
    if len(folders) < 2:
        return None

    files = None
    for pattern in (
        r"选中其中的\s*([A-Za-z0-9_\-./,\u4e00-\u9fff、 ]+?)\s*文件",
        r"选中\s*([A-Za-z0-9_\-./,\u4e00-\u9fff、 ]+?)\s*文件",
    ):
        if match := re.search(pattern, text):
            files = match.group(1)
            break
    if not files:
        return None

    files_without_spaces = re.sub(r"\s+", "", files)
    files_with_ascii_separators = files_without_spaces.replace("、", ",")
    normalized_files = files_with_ascii_separators.strip(",")
    if not normalized_files:
        return None
    return f"把{folders[0]}文件夹中的{normalized_files}文件移动到{folders[1]}文件夹"


def _strip_filler(text: str) -> str:
    """Trim filler particles around extracted slots."""
    cleaned = re.sub(r"^(一下|一下下|帮我|请帮我|请|麻烦你)", "", text)
    cleaned = re.sub(r"(那边|这里|这边)$", "", cleaned)
    return cleaned.strip(" ，,。；;:：")


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
        detail_tokens = _DETAIL_TOKEN_RE.findall(candidate)
        detail_chars = sum(len(token) for token in detail_tokens)
        compact_text = re.sub(r"\s+", "", candidate)
        if len(compact_text) <= 10:
            return True
        if len(words) <= 3 and len(detail_tokens) <= 1 and detail_chars <= 12:
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
