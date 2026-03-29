"""
Split model markdown into ordered segments (OmniDocBench md_tex_filter–style, lightweight).

Extracts structural blocks first (fenced code, HTML tables, pipe tables, display math),
then splits remaining prose by blank lines. Text blocks get light markdown stripping
(headings, list markers) so alignment compares words more like plain text.

Does not duplicate OmniDocBench’s full LaTeX pipeline (pylatexenc, md→html tables);
it targets typical Docling / VLM markdown.
"""

from __future__ import annotations

import re
from typing import Callable, Literal

PieceKind = Literal["text", "code", "html_table", "md_table", "math"]

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
_DISPLAY_MATH = re.compile(
    r"\$\$(.*?)\$\$|\\\[(.*?)\\\]",
    re.DOTALL,
)


def _is_md_table_line(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("<!--"):
        return False
    return "|" in s and s.count("|") >= 2


def _is_md_table_sep(line: str) -> bool:
    s = line.strip()
    if "|" not in s:
        return False
    core = re.sub(r"\s+", "", s)
    return bool(re.match(r"^[\|\s:\-]+$", core)) and "-" in core


def split_md_pipe_tables(s: str) -> list[tuple[PieceKind, str]]:
    """Split out GitHub-style pipe tables (consecutive | lines, optional --- sep)."""
    lines = s.splitlines(keepends=True)
    out: list[tuple[PieceKind, str]] = []
    i = 0
    buf_start = 0

    def flush_text(end_line: int) -> None:
        chunk = "".join(lines[buf_start:end_line])
        if chunk:
            out.append(("text", chunk))

    while i < len(lines):
        if not _is_md_table_line(lines[i]):
            i += 1
            continue
        j = i
        while j < len(lines) and (_is_md_table_line(lines[j]) or _is_md_table_sep(lines[j])):
            j += 1
        if j - i >= 2:
            flush_text(i)
            table_text = "".join(lines[i:j])
            out.append(("md_table", table_text))
            buf_start = j
            i = j
        else:
            i += 1

    flush_text(len(lines))
    return out if out else [("text", s)]


def split_html_tables(s: str) -> list[tuple[PieceKind, str]]:
    out: list[tuple[PieceKind, str]] = []
    last = 0
    for m in _HTML_TABLE.finditer(s):
        if m.start() > last:
            out.append(("text", s[last : m.start()]))
        out.append(("html_table", m.group(0)))
        last = m.end()
    if last < len(s):
        out.append(("text", s[last:]))
    return out if out else [("text", s)]


def split_code_fences(s: str) -> list[tuple[PieceKind, str]]:
    out: list[tuple[PieceKind, str]] = []
    last = 0
    for m in _FENCE.finditer(s):
        if m.start() > last:
            out.append(("text", s[last : m.start()]))
        out.append(("code", m.group(2)))
        last = m.end()
    if last < len(s):
        out.append(("text", s[last:]))
    return out if out else [("text", s)]


def split_display_math(s: str) -> list[tuple[PieceKind, str]]:
    out: list[tuple[PieceKind, str]] = []
    last = 0
    for m in _DISPLAY_MATH.finditer(s):
        if m.start() > last:
            out.append(("text", s[last : m.start()]))
        body = (m.group(1) or m.group(2) or "").strip()
        out.append(("math", body if body else m.group(0).strip()))
        last = m.end()
    if last < len(s):
        out.append(("text", s[last:]))
    return out if out else [("text", s)]


def light_strip_md_paragraph(p: str) -> str:
    """Strip common markdown decoration from a prose block (per-line)."""
    if not p.strip():
        return ""
    lines_out: list[str] = []
    for line in p.split("\n"):
        t = line.strip()
        t = re.sub(r"^#{1,6}\s+", "", t)
        t = re.sub(r"^[-*+]\s+", "", t)
        t = re.sub(r"^\d+\.\s+", "", t)
        if t:
            lines_out.append(t)
    joined = "\n".join(lines_out).strip()
    joined = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", joined)
    joined = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", joined)
    joined = re.sub(r"`([^`]+)`", r"\1", joined)
    joined = re.sub(r"\*\*([^*]+)\*\*", r"\1", joined)
    joined = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", joined)
    return joined.strip()


def _text_to_paragraph_segments(text: str) -> list[str]:
    """Blank-line paragraphs; fallback to single newlines if almost no blanks."""
    if not text or not text.strip():
        return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", t)
    parts = [light_strip_md_paragraph(p) for p in parts if p.strip()]
    if len(parts) <= 1 and "\n" in t:
        parts = [light_strip_md_p for p in t.split("\n") if (light_strip_md_p := light_strip_md_paragraph(p))]
    return [p for p in parts if p]


def prediction_segments_from_markdown(text: str) -> list[str]:
    """
    Ordered segments for full-page markdown prediction (structure-aware).

    Order of extraction passes on each ``text`` fragment: code fences → HTML tables
    → pipe tables → display math; then remaining ``text`` is split into paragraphs.
    """
    if not text or not text.strip():
        return []

    pieces: list[tuple[PieceKind, str]] = [("text", text)]

    def apply_splitter(
        splitter: Callable[[str], list[tuple[PieceKind, str]]],
        only_text: bool = True,
    ) -> None:
        nonlocal pieces
        new_pieces: list[tuple[PieceKind, str]] = []
        for kind, chunk in pieces:
            if only_text and kind != "text":
                new_pieces.append((kind, chunk))
                continue
            if not chunk:
                continue
            new_pieces.extend(splitter(chunk))
        pieces = new_pieces

    apply_splitter(split_code_fences)
    apply_splitter(split_html_tables)
    apply_splitter(split_md_pipe_tables)
    apply_splitter(split_display_math)

    segments: list[str] = []
    for kind, chunk in pieces:
        if kind == "text":
            segments.extend(_text_to_paragraph_segments(chunk))
        else:
            c = chunk.strip()
            if c:
                segments.append(c)

    return segments


def prediction_segments(
    text: str,
    *,
    use_markdown_structure: bool = True,
) -> list[str]:
    """Dispatch: structure-aware markdown vs legacy blank-line-only split."""
    if use_markdown_structure:
        return prediction_segments_from_markdown(text)
    from .normalize import segment_prediction

    return segment_prediction(text)
