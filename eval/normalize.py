"""
Text normalization inspired by OmniDocBench (fair comparison across model styles).

Applied to both ground-truth snippets and prediction paragraphs before similarity /
edit-distance. Raw strings are preserved on the side.

OmniDocBench parity (non–LaTeX / non–HTML): ``remove_markdown_fences``, circled-Unicode
``replace_textcircle``, ``fullwidth_to_halfwidth``, ``replace_repeated_chars`` (four
underscores / four spaces). We intentionally omit full LaTeX parsing and HTML table
parsing; use ``omnidocbench_clean_string`` only when you need Omni’s aggressive
``clean_string`` (word chars + CJK only).

When ``strip_markdown_tokens`` is True (default), ``strip_markdown_markup`` removes common
Markdown/HTML presentation (fences, emphasis, headings, lists, tables, links, etc.) on both
GT and predictions so CER/WER are less dominated by formatting.
"""

from __future__ import annotations

import re
import unicodedata

CIRCLED_UNICODE_MAP: dict[str, str] = {chr(0x2460 + i): str(i + 1) for i in range(20)}
CIRCLED_UNICODE_MAP.update({chr(0x24D0 + i): chr(ord("a") + i) for i in range(26)})
CIRCLED_UNICODE_MAP.update({chr(0x24B6 + i): chr(ord("A") + i) for i in range(26)})


def replace_textcircle_unicode(text: str) -> str:
    """Map circled Unicode digits/letters to plain ASCII."""
    return "".join(CIRCLED_UNICODE_MAP.get(c, c) for c in text)


def remove_markdown_fences(content: str) -> str:
    """Strip leading ```markdown/html/latex and trailing ``` fences."""
    content = re.sub(r"^```markdown\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^```html\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^```latex\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"```\n?$", "", content, flags=re.MULTILINE)
    return content


def replace_repeated_chars_omni(input_str: str) -> str:
    """Cap long underscore / space runs to exactly four."""
    input_str = re.sub(r"_{4,}", "____", input_str)
    input_str = re.sub(r" {4,}", "    ", input_str)
    return input_str


def fullwidth_to_halfwidth(s: str) -> str:
    """Map fullwidth space and U+FF01–U+FF5E to ASCII."""
    out: list[str] = []
    for c in s:
        code = ord(c)
        if c == "\u3000":
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(c)
    return "".join(out)


def clean_string_omni(input_string: str) -> str:
    """
    OmniDocBench clean_string: drop escapes/newlines/tabs then keep only ``\\w`` and CJK.
    Very aggressive (removes spaces between English words).
    """
    input_string = replace_textcircle_unicode(input_string)
    input_string = (
        input_string.replace("\\t", "")
        .replace("\\n", "")
        .replace("\t", "")
        .replace("\n", "")
        .replace("/t", "")
        .replace("/n", "")
    )
    return re.sub(r"[^\w\u4e00-\u9fff]", "", input_string)


def collapse_repeated_chars(text: str, max_repeat: int = 3) -> str:
    """Cap runs of the same character (models sometimes emit long underscores/dashes)."""
    if max_repeat < 1:
        return text

    def repl(m: re.Match[str]) -> str:
        ch = m.group(1)
        return ch * min(len(m.group(0)), max_repeat)

    return re.sub(r"(.)\1{2,}", repl, text)


def strip_image_markdown(text: str) -> str:
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)


def strip_vlm_output_artifacts(text: str) -> str:
    """
    Remove VLM/chat-template boilerplate before splitting and normalization.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"<\|[^|]+\|>", " ", t)
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r"(?i)\bText Recognition:\s*", "", t)
    t = re.sub(r"^\s*---+?\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*\*{3,}\s*$", "", t, flags=re.MULTILINE)
    lines = t.splitlines()
    if len(lines) >= 2:
        fl = lines[0].strip()
        ll = lines[-1].strip()
        if fl.startswith("```") and ll == "```":
            t = "\n".join(lines[1:-1])
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```[^\n`]*\n", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "\n", text, flags=re.MULTILINE)
    return text


def strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)


def _strip_markdown_emphasis(t: str) -> str:
    for _ in range(10):
        n = t
        n = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", n)
        n = re.sub(r"___([^_]+)___", r"\1", n)
        n = re.sub(r"\*\*([^*]+)\*\*", r"\1", n)
        n = re.sub(r"__([^_]+)__", r"\1", n)
        n = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", n)
        n = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", n)
        n = re.sub(r"~~([^~]+)~~", r"\1", n)
        if n == t:
            break
        t = n
    return t


def _strip_pipe_table_lines(text: str) -> str:
    out_lines: list[str] = []
    for raw in text.splitlines():
        if "|" not in raw:
            out_lines.append(raw)
            continue
        ls = raw.strip()
        if re.match(r"^[\s|:\-–—]+$", ls):
            continue
        parts = [p.strip() for p in raw.split("|")]
        parts = [p for p in parts if p]
        if not parts:
            continue
        if all(re.fullmatch(r"[\-:]+", p) for p in parts):
            continue
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines)


def _strip_latex_markup(text: str) -> str:
    """
    Strip common LaTeX markup so OCR scoring compares textual content.
    """
    t = text
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", t)
    t = t.replace("$", " ")

    wrappers = (
        "text",
        "textrm",
        "mathrm",
        "mathbf",
        "mathit",
        "mathsf",
        "mathtt",
        "operatorname",
        "mbox",
    )
    for _ in range(6):
        prev = t
        for w in wrappers:
            t = re.sub(rf"\\{w}\s*\{{([^{{}}]*)\}}", r"\1", t)
        if t == prev:
            break

    t = re.sub(r"_\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\^\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"_(\w)", r" \1 ", t)
    t = re.sub(r"\^(\w)", r" \1 ", t)

    t = t.replace(r"\&", "&").replace(r"\%", "%").replace(r"\#", "#")
    t = t.replace(r"\_", "_").replace(r"\{", "{").replace(r"\}", "}")
    t = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", t)
    return t


def strip_markdown_markup(text: str) -> str:
    """
    Strip Markdown/HTML/LaTeX decoration so scoring compares words, not markup.
    """
    if not text:
        return ""
    t = text

    t = re.sub(r"```[^\n`]*\n[\s\S]*?```", " ", t)
    t = re.sub(r"```[\s\S]*?```", " ", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)

    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\(([^\s)]+)(?:\s+[\"'][^\"']*[\"'])?\)", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", t)
    t = re.sub(r"\[\^[^\]]+\]", " ", t)
    t = re.sub(r"(?m)^\[[^\]]+\]:\s*\S+(?:\s+[\"'][^\"']*[\"'])?\s*$", "", t)

    t = _strip_markdown_emphasis(t)
    t = _strip_pipe_table_lines(t)

    t = re.sub(r"(?m)^#{1,6}\s+", "", t)
    t = re.sub(r"(?m)^[=-]{2,}\s*$", "", t)
    t = re.sub(r"(?m)^>\s?", "", t)
    t = re.sub(r"(?m)^\s*(?:[-*_]\s*){3,}\s*$", "", t)
    t = re.sub(r"(?m)^\s*(?:[-*+]\s+(?:\[[ xX]\]\s+)?)", "", t)
    t = re.sub(r"(?m)^\s*\d+[.)]\s+", "", t)

    t = re.sub(r"<(https?://[^>\s]+)>", r" \1 ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = _strip_latex_markup(t)
    t = re.sub(r"\[\[([^\]]+)\]\]", r"\1", t)
    t = re.sub(r"[`*#]{2,}", " ", t)
    return t


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(
    text: str,
    *,
    unicode_form: str = "NFKC",
    strip_md_images: bool = True,
    strip_fences: bool = True,
    strip_html_comm: bool = True,
    collapse_repeats: bool = True,
    max_char_repeat: int = 3,
    lowercase: bool = False,
    omnidocbench_preprocess: bool = True,
    omnidocbench_clean_string: bool = False,
    strip_markdown_tokens: bool = True,
) -> str:
    if not text:
        return ""
    t = unicodedata.normalize(unicode_form, text)
    if omnidocbench_preprocess:
        t = remove_markdown_fences(t)
        t = replace_textcircle_unicode(t)
        t = fullwidth_to_halfwidth(t)
        t = replace_repeated_chars_omni(t)
    if strip_md_images:
        t = strip_image_markdown(t)
    if strip_fences:
        t = strip_code_fences(t)
    if strip_html_comm:
        t = strip_html_comments(t)
    if strip_markdown_tokens:
        t = strip_markdown_markup(t)
    if collapse_repeats and not omnidocbench_preprocess:
        t = collapse_repeated_chars(t, max_repeat=max_char_repeat)
    if lowercase:
        t = t.lower()
    if omnidocbench_clean_string:
        return clean_string_omni(t)
    return normalize_whitespace(t)


def segment_prediction(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", t)
    parts = [normalize_whitespace(p) for p in parts if p.strip()]
    if len(parts) <= 1 and "\n" in t:
        parts = [normalize_whitespace(p) for p in t.split("\n") if p.strip()]
    return parts
