"""
Load full-page prediction text from a file path (markdown/plain or JSON).

JSON is common for native exports; structure varies by model. Use ``json_key`` for an
explicit path, or rely on ``auto`` heuristics below.

PaddleOCR ``paddleocr_predict.json`` may embed huge ``input_img`` arrays; use
``text_from_paddle_predict_json`` (tail scan for ``rec_texts``) instead of loading
the whole file. Docling ``docling_document.json`` follows the DoclingDocument schema;
``text_from_docling_document_json`` walks ``#/body`` in reading order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Tried in order when root is a dict and no json_key (string values only).
_JSON_STRING_KEYS = (
    "full_text",
    "full_page_text",
    "text",
    "markdown",
    "md",
    "content",
    "prediction",
    "output",
    "transcription",
    "ocr_text",
    "result",
    "document",
)


def _join_ocr_line_list(items: list[Any]) -> str | None:
    """Paddle / generic list of dicts with text fields."""
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for k in ("rec_text", "text", "transcription", "content", "txt"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                lines.append(v.strip())
                break
    if not lines:
        return None
    return "\n".join(lines)


def _guess_json_text(data: Any) -> str | None:
    if isinstance(data, str) and data.strip():
        return data
    if isinstance(data, list):
        joined = _join_ocr_line_list(data)
        if joined:
            return joined
        if len(data) == 1:
            return _guess_json_text(data[0])
        return None
    if not isinstance(data, dict):
        return None
    for k in _JSON_STRING_KEYS:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, (dict, list)):
            inner = _guess_json_text(v)
            if inner:
                return inner
    return None


def _paddle_find_rec_texts(obj: Any, depth: int = 0) -> list[str] | None:
    """Return first ``rec_texts`` list of strings in a small Paddle JSON tree."""
    if depth > 24:
        return None
    if isinstance(obj, dict):
        rt = obj.get("rec_texts")
        if isinstance(rt, list) and rt and all(isinstance(x, str) for x in rt):
            return rt
        for v in obj.values():
            r = _paddle_find_rec_texts(v, depth + 1)
            if r is not None:
                return r
    if isinstance(obj, list):
        for item in obj:
            r = _paddle_find_rec_texts(item, depth + 1)
            if r is not None:
                return r
    return None


def text_from_paddle_predict_json(path: Path, *, encoding: str = "utf-8", errors: str = "replace") -> str:
    """
    Extract line text from PaddleOCR prediction JSON.

    Large exports repeat the page image under ``doc_preprocessor_res``; we only need the
    ``rec_texts`` array, which is read from the **end** of the file when the file is large.
    """
    path = Path(path)
    size = path.stat().st_size
    max_small = 50 * 1024 * 1024
    if size <= max_small:
        raw = path.read_text(encoding=encoding, errors=errors)
        data = json.loads(raw)
        lines = _paddle_find_rec_texts(data)
        if lines is None:
            raise ValueError(f"No rec_texts list found in {path}")
        return "\n".join(lines)

    tail_n = min(55 * 1024 * 1024, size)
    with path.open("rb") as f:
        f.seek(size - tail_n)
        blob = f.read()
    needle = b'"rec_texts"'
    idx = blob.find(needle)
    if idx < 0:
        raise ValueError(f'"rec_texts" not found in tail of {path} (file may not be Paddle OCR JSON)')
    text = blob[idx:].decode(encoding, errors=errors)
    j = text.find('"rec_texts"')
    if j < 0:
        raise ValueError(f"Could not decode rec_texts region in {path}")
    bracket = text.find("[", j)
    if bracket < 0:
        raise ValueError(f"No array after rec_texts in {path}")
    dec = json.JSONDecoder()
    arr, _end = dec.raw_decode(text, bracket)
    if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
        raise ValueError(f"rec_texts is not a list of strings in {path}")
    return "\n".join(arr)


def _docling_resolve_ref(root: dict[str, Any], ref: str) -> Any:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise ValueError(f"bad ref {ref!r}")
    cur: Any = root
    for seg in ref[2:].split("/"):
        if seg.isdigit():
            cur = cur[int(seg)]
        else:
            cur = cur[seg]
    return cur


def _docling_collect_lines(root: dict[str, Any], children: list[Any] | None) -> list[str]:
    lines: list[str] = []
    for ch in children or []:
        if not isinstance(ch, dict) or "$ref" not in ch:
            continue
        try:
            node = _docling_resolve_ref(root, ch["$ref"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if not isinstance(node, dict):
            continue
        lab = node.get("label")
        if lab in ("text", "section_header", "page_footer", "caption", "list_item", "formula"):
            tx = node.get("text")
            if isinstance(tx, str) and tx.strip():
                lines.append(tx.strip())
        if node.get("children"):
            lines.extend(_docling_collect_lines(root, node.get("children")))
    return lines


def text_from_docling_document_json(path: Path, *, encoding: str = "utf-8", errors: str = "replace") -> str:
    """Join DoclingDocument body text nodes in reading order (same order as body tree)."""
    path = Path(path)
    raw = path.read_text(encoding=encoding, errors=errors)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Docling JSON root must be an object: {path}")
    body = data.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"Docling JSON missing body: {path}")
    lines = _docling_collect_lines(data, body.get("children"))
    return "\n".join(lines)


def _navigate_json(obj: Any, key_path: str) -> Any:
    """
    Dot-separated path; numeric segments index into lists.

    Examples: ``text``, ``result.markdown``, ``pages.0.body``, ``outputs.0.text``
    """
    cur: Any = obj
    for part in key_path.split("."):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(f"cannot use {part!r} on {type(cur).__name__} in {key_path!r}")
    return cur


def load_prediction_text(
    path: Path,
    *,
    pred_format: str = "auto",
    json_key: str | None = None,
    encoding: str = "utf-8",
    errors: str = "replace",
    native_json_schema: bool = True,
) -> str:
    """
    Read full-page prediction string from ``path``.

    :param pred_format: ``auto`` (from suffix + sniff), ``text`` (raw file), ``json``.
    :param json_key: Dot path to a string field, e.g. ``result.markdown``. If omitted
        under JSON, common top-level keys and OCR line lists are tried.
    :param native_json_schema: If True (default), ``paddleocr_predict.json`` and
        ``docling_document.json`` use dedicated extractors (see module docstring).
        Set False to force generic JSON heuristics when ``json_key`` is None.
    """
    path = Path(path)
    suf = path.suffix.lower()
    fmt = pred_format.lower().strip()
    if fmt == "auto":
        fmt = "json" if suf == ".json" else "text"

    if fmt == "text":
        return path.read_bytes().decode(encoding, errors=errors)

    name_l = path.name.lower()
    if json_key is None and native_json_schema:
        if name_l == "paddleocr_predict.json":
            return text_from_paddle_predict_json(path, encoding=encoding, errors=errors)
        if name_l == "docling_document.json":
            return text_from_docling_document_json(path, encoding=encoding, errors=errors)

    if name_l == "paddleocr_predict.json" and path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError(
            "paddleocr_predict.json is very large (often contains embedded images). "
            "Use native extraction (omit --pred-json-key), or point at paddleocr.txt, "
            "or set native_json_schema=True without json_key."
        )

    text = path.read_text(encoding=encoding, errors=errors)
    data = json.loads(text)

    if json_key:
        cur = _navigate_json(data, json_key)
        if isinstance(cur, str):
            return cur
        if isinstance(cur, (dict, list)):
            guessed = _guess_json_text(cur)
            if guessed:
                return guessed
        raise ValueError(
            f"JSON key {json_key!r} did not resolve to a string (got {type(cur).__name__})"
        )

    out = _guess_json_text(data)
    if out is None:
        raise ValueError(
            "Could not infer prediction text from JSON; pass --pred-json-key KEY "
            f"(dot path to a string field). Top-level keys: "
            f"{list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
        )
    return out
