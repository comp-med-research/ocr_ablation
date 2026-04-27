"""
Align ``gt_manifest.json`` regions to PaddleOCR ``paddleocr_predict.json`` using boxes.

Uses GT ``bbox_pct`` (Label Studio %) and ``rec_boxes`` / ``rec_texts`` (pixel coords,
top-left origin). Large JSON files embed images; we read ``rec_texts`` and ``rec_boxes``
from the file tail only (same strategy as ``pred_loader.text_from_paddle_predict_json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Any

from .docling_layout import _indices_overlapping_gt
from .edit_distance import normalized_edit_distance
from .layout_geometry import BoxNorm, box_from_bbox_pct, rotated_polygon_from_bbox_pct
from .matching import RegionMatch, TaskTextEvalResult, TextEvalConfig, _aggregate_metrics
from .pred_loader import _paddle_find_rec_texts


@dataclass
class PaddleTextSpan:
    """One recognition line with axis-aligned box in normalized page coordinates."""

    text: str
    box: BoxNorm
    page_no: int = 1


def _paddle_read_texts_and_boxes(path: Path) -> tuple[list[str], list[list[float]]]:
    """Load ``rec_texts`` and ``rec_boxes`` (same length) from small or large Paddle JSON."""
    path = Path(path)
    size = path.stat().st_size
    max_small = 50 * 1024 * 1024
    if size <= max_small:
        data = json.loads(path.read_text(encoding="utf-8"))
        texts = _paddle_find_rec_texts(data)
        boxes = _paddle_find_rec_boxes(data)
        if texts is None or boxes is None:
            raise ValueError(f"Could not find rec_texts and rec_boxes in {path}")
        if len(texts) != len(boxes):
            raise ValueError(
                f"rec_texts length {len(texts)} != rec_boxes length {len(boxes)} in {path}"
            )
        return texts, boxes

    tail_n = min(60 * 1024 * 1024, size)
    with path.open("rb") as f:
        f.seek(size - tail_n)
        blob = f.read().decode("utf-8", errors="replace")
    dec = JSONDecoder()

    t_idx = blob.find('"rec_texts"')
    if t_idx < 0:
        raise ValueError(f'"rec_texts" not found in tail of {path}')
    t_bracket = blob.find("[", t_idx)
    texts, t_end = dec.raw_decode(blob, t_bracket)
    if not isinstance(texts, list) or not texts or not all(isinstance(x, str) for x in texts):
        raise ValueError(f"rec_texts is not a list of strings in {path}")

    b_idx = blob.find('"rec_boxes"', t_end)
    if b_idx < 0:
        raise ValueError(f'"rec_boxes" not found after rec_texts in tail of {path}')
    b_bracket = blob.find("[", b_idx)
    boxes, _b_end = dec.raw_decode(blob, b_bracket)
    if not isinstance(boxes, list) or not boxes:
        raise ValueError(f"rec_boxes is not a list in {path}")
    flat: list[list[float]] = []
    for row in boxes:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"Bad rec_boxes row in {path}: {row!r}")
        flat.append([float(x) for x in row])
    if len(texts) != len(flat):
        raise ValueError(
            f"rec_texts length {len(texts)} != rec_boxes length {len(flat)} in {path} (tail parse)"
        )
    return texts, flat


def _paddle_find_rec_boxes(obj: Any, depth: int = 0) -> list[list[float]] | None:
    if depth > 24:
        return None
    if isinstance(obj, dict):
        rb = obj.get("rec_boxes")
        if isinstance(rb, list) and rb and isinstance(rb[0], list) and len(rb[0]) == 4:
            return [[float(x) for x in row] for row in rb]
        for v in obj.values():
            r = _paddle_find_rec_boxes(v, depth + 1)
            if r is not None:
                return r
    if isinstance(obj, list):
        for item in obj:
            r = _paddle_find_rec_boxes(item, depth + 1)
            if r is not None:
                return r
    return None


def iter_paddle_predict_spans(json_path: Path | str) -> list[PaddleTextSpan]:
    """
    Build one span per ``(rec_texts[i], rec_boxes[i])`` in normalized top-left page coords.

    Page size is inferred from the maximum right/bottom edge among all boxes.
    """
    path = Path(json_path)
    texts, boxes = _paddle_read_texts_and_boxes(path)
    page_w = max((b[2] for b in boxes), default=1.0)
    page_h = max((b[3] for b in boxes), default=1.0)
    page_w = max(page_w, 1.0)
    page_h = max(page_h, 1.0)
    out: list[PaddleTextSpan] = []
    for txt, bx in zip(texts, boxes):
        x0, y0, x1, y1 = bx
        box = BoxNorm(x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h)
        t = (txt or "").strip()
        if not t:
            continue
        out.append(PaddleTextSpan(text=t, box=box))
    return out


def match_gt_to_paddle_predict_json_path(
    regions: list[dict[str, Any]],
    json_path: Path | str,
    *,
    config: TextEvalConfig | None = None,
    min_iou: float = 0.05,
) -> TaskTextEvalResult:
    """
    For each GT region with ``bbox_pct``, merge Paddle ``rec_texts`` lines whose ``rec_boxes``
    overlap GT (IoU ≥ ``min_iou`` or line box center inside GT), reading-order sort, NED/CER path.
    """
    cfg = config or TextEvalConfig()
    path = Path(json_path)
    spans = iter_paddle_predict_spans(path)
    pred_strings = [s.text for s in spans]

    result = TaskTextEvalResult(
        task_id=None,
        pred_segments=pred_strings,
        match_mode=cfg.match_mode,
        alignment="layout-paddle-json",
    )
    result.notes.append("pairing=IoU/center overlap on rec_boxes vs GT bbox_pct")
    result.notes.append(f"paddle_spans={len(spans)} min_iou={min_iou} path={path.name}")

    gts_raw = [r.get("transcription_gt") or "" for r in regions]
    ids = [str(r.get("region_id", "")) for r in regions]
    n = len(gts_raw)
    if n == 0:
        result.notes.append("no_gt_regions")
        return result

    gts_norm = [cfg.normalize(t) for t in gts_raw]

    for i, r in enumerate(regions):
        bp = r.get("bbox_pct")
        gt_box = box_from_bbox_pct(bp) if isinstance(bp, dict) else None
        gn = gts_norm[i]

        if gt_box is None:
            pn = cfg.normalize("")
            result.regions_matched.append(
                RegionMatch(
                    gt_index=i,
                    region_id=ids[i],
                    gt_raw=gts_raw[i],
                    pred_raw_merged="",
                    gt_norm=gn,
                    pred_norm=pn,
                    ned=1.0,
                    similarity=0.0,
                    pred_segment_start=0,
                    pred_segment_end=0,
                )
            )
            continue

        gt_poly = rotated_polygon_from_bbox_pct(bp) if isinstance(bp, dict) else None
        idxs = _indices_overlapping_gt(
            gt_box,
            gt_poly,
            spans,  # type: ignore[arg-type]  # duck-typed like DoclingTextSpan (.box)
            min_iou=min_iou,
            center_in_gt=True,
        )
        pred_raw_merged = " ".join(spans[j].text for j in idxs)
        pn = cfg.normalize(pred_raw_merged)
        ned = float(
            normalized_edit_distance(cfg.normalize(gts_raw[i]), cfg.normalize(pred_raw_merged))
        )

        start = min(idxs) if idxs else 0
        end = (max(idxs) + 1) if idxs else 0

        result.regions_matched.append(
            RegionMatch(
                gt_index=i,
                region_id=ids[i],
                gt_raw=gts_raw[i],
                pred_raw_merged=pred_raw_merged,
                gt_norm=gn,
                pred_norm=pn,
                ned=ned,
                similarity=1.0 - ned,
                pred_segment_start=start,
                pred_segment_end=end,
            )
        )

    _aggregate_metrics(result)
    return result
