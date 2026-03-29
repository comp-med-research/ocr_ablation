"""
Load full-page prediction text from a file path (markdown/plain or JSON).

JSON is common for native exports; structure varies by model. Use ``json_key`` for an
explicit path, or rely on ``auto`` heuristics below.
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
) -> str:
    """
    Read full-page prediction string from ``path``.

    :param pred_format: ``auto`` (from suffix + sniff), ``text`` (raw file), ``json``.
    :param json_key: Dot path to a string field, e.g. ``result.markdown``. If omitted
        under JSON, common top-level keys and OCR line lists are tried.
    """
    path = Path(path)
    suf = path.suffix.lower()
    fmt = pred_format.lower().strip()
    if fmt == "auto":
        fmt = "json" if suf == ".json" else "text"

    raw_bytes = path.read_bytes()
    if fmt == "text":
        return raw_bytes.decode(encoding, errors=errors)

    text = raw_bytes.decode(encoding, errors=errors)
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
