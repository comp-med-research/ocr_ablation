"""
Build a minimal Docling-shaped document dict from PaddleOCR ``predict`` JSON.

Paddle exports ``paddleocr_predict.json`` as a list of result dicts with ``rec_texts``,
``rec_boxes`` (xyxy in **top-left** pixel coordinates), and often huge ``input_img`` /
``output_img`` arrays. Full ``json.loads`` on such files is impractical; use
:func:`extract_docling_like_from_paddle_predict` which locates ``rec_*`` fields via mmap
and parses only those slices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _parse_json_value_after_key(mm: Any, key: bytes, *, max_chunk: int = 80 * 1024 * 1024) -> Any | None:
    """Find ``"key"`` in mmap and decode the JSON value that follows (object or array)."""
    idx = mm.find(key)
    if idx < 0:
        return None
    hi = min(len(mm), idx + max_chunk)
    s = mm[idx:hi].decode("utf-8", errors="replace")
    colon = s.find(":")
    if colon < 0:
        return None
    i = colon + 1
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    if i >= len(s):
        return None
    dec = json.JSONDecoder()
    try:
        val, _end = dec.raw_decode(s, i)
        return val
    except json.JSONDecodeError:
        return None


def _infer_page_size_from_boxes(rec_boxes: list[Any]) -> tuple[float, float] | None:
    if not rec_boxes:
        return None
    mx = 0.0
    my = 0.0
    for b in rec_boxes:
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        try:
            x2 = float(b[2])
            y2 = float(b[3])
        except (TypeError, ValueError):
            continue
        mx = max(mx, x2)
        my = max(my, y2)
    if mx <= 0 or my <= 0:
        return None
    return (mx, my)


def paddle_item_to_docling_document(item: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert one Paddle result dict to a minimal ``DoclingDocument``-like structure
    (``pages`` + ``texts`` with ``prov[].bbox``) understood by :func:`iter_docling_text_spans`.
    """
    rec_texts = item.get("rec_texts")
    rec_boxes = item.get("rec_boxes")
    if not isinstance(rec_texts, list) or not isinstance(rec_boxes, list):
        return None
    n = min(len(rec_texts), len(rec_boxes))
    if n == 0:
        return None
    size = _infer_page_size_from_boxes(rec_boxes[:n])
    if size is None:
        return None
    page_w, page_h = size
    texts: list[dict[str, Any]] = []
    for i in range(n):
        txt = str(rec_texts[i] or "").strip()
        b = rec_boxes[i]
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        try:
            x1, y1, x2, y2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except (TypeError, ValueError):
            continue
        if not txt:
            continue
        # Paddle: top-left xyxy pixels. Docling prov.bbox: l,t,r,b in pixel space, BOTTOMLEFT origin.
        texts.append(
            {
                "text": txt,
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": x1,
                            "t": page_h - y1,
                            "r": x2,
                            "b": page_h - y2,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
            }
        )
    return {
        "pages": {"1": {"size": {"width": float(page_w), "height": float(page_h)}}},
        "texts": texts,
    }


def _extract_from_mmap(path: Path) -> dict[str, Any] | None:
    import mmap

    with path.open("rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            rt = _parse_json_value_after_key(mm, b'"rec_texts"')
            rb = _parse_json_value_after_key(mm, b'"rec_boxes"')
            if not isinstance(rt, list) or not isinstance(rb, list):
                return None
            item = {"rec_texts": rt, "rec_boxes": rb}
            return paddle_item_to_docling_document(item)


def _extract_from_full_parse(path: Path) -> dict[str, Any] | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw)
    if isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            return None
        item = data[0]
    elif isinstance(data, dict):
        item = data
    else:
        return None
    return paddle_item_to_docling_document(item)


def extract_docling_like_from_paddle_predict(path: Path | str) -> dict[str, Any] | None:
    """
    Return a Docling-like document dict, or ``None`` if this is not Paddle predict JSON.

    Uses mmap + partial parse for large files; full parse only when the file is small.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        sz = p.stat().st_size
    except OSError:
        return None
    # Heuristic: huge files are almost always predict JSON with embedded images — never full parse.
    _SMALL = 100 * 1024 * 1024
    if sz > _SMALL:
        return _extract_from_mmap(p)
    try:
        return _extract_from_full_parse(p)
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None


def write_docling_sidecar_from_paddle_predict(
    predict_path: Path | str,
    out_path: Path | str,
    *,
    indent: int = 2,
) -> bool:
    """Write ``docling_document.json`` next to (or at) ``out_path`` from a predict file."""
    doc = extract_docling_like_from_paddle_predict(predict_path)
    if doc is None:
        return False
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(doc, ensure_ascii=False, indent=indent), encoding="utf-8")
    return True


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Convert paddleocr_predict.json to Docling-shaped JSON (text + boxes only).")
    ap.add_argument("predict_json", type=Path, help="Path to paddleocr_predict.json")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: same dir as input, docling_document.json)",
    )
    args = ap.parse_args()
    out = args.output
    if out is None:
        out = args.predict_json.parent / "docling_document.json"
    ok = write_docling_sidecar_from_paddle_predict(args.predict_json, out)
    if not ok:
        raise SystemExit("Could not extract rec_texts/rec_boxes from input (wrong format?).")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
