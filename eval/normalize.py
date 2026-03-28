"""
Text normalization inspired by OmniDocBench (fair comparison across model styles).

Applied to both ground-truth snippets and prediction paragraphs before similarity /
edit-distance. Raw strings are preserved on the side for display.
"""

from __future__ import annotations

import re
import unicodedata


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


def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```[^\n`]*\n", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "\n", text, flags=re.MULTILINE)
    return text


def strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)


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
) -> str:
    """
    Single pipeline for scoring. Toggle pieces via kwargs if you need stricter/loser tests.
    """
    if not text:
        return ""
    t = unicodedata.normalize(unicode_form, text)
    if strip_md_images:
        t = strip_image_markdown(t)
    if strip_fences:
        t = strip_code_fences(t)
    if strip_html_comm:
        t = strip_html_comments(t)
    if collapse_repeats:
        t = collapse_repeated_chars(t, max_repeat=max_char_repeat)
    if lowercase:
        t = t.lower()
    return normalize_whitespace(t)


def segment_prediction(text: str) -> list[str]:
    """
    Split model output into blocks (OmniDocBench-style paragraph units).

    Prefer blank-line paragraphs; fall back to single newlines if there are no blanks.
    """
    if not text or not text.strip():
        return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", t)
    parts = [normalize_whitespace(p) for p in parts if p.strip()]
    if len(parts) <= 1 and "\n" in t:
        parts = [normalize_whitespace(p) for p in t.split("\n") if p.strip()]
    return parts
