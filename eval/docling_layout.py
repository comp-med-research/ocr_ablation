"""
Align ``gt_manifest.json`` regions to Docling ``DoclingDocument`` JSON using bounding boxes.

Expects GT ``bbox_pct`` (Label Studio %) and Docling ``texts[].prov[].bbox`` (BOTTOMLEFT, pixel or norm).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .edit_distance import normalized_edit_distance
from .layout_geometry import (
    BoxNorm,
    box_from_bbox_pct,
    docling_bbox_to_box_norm,
    iou,
    iou_polygon_vs_box,
    point_in_polygon,
    reading_order_sort_key,
    rotated_polygon_from_bbox_pct,
)
from .matching import RegionMatch, TaskTextEvalResult, TextEvalConfig, _aggregate_metrics


@dataclass
class DoclingTextSpan:
    text: str
    box: BoxNorm
    page_no: int


def load_docling_json(path: Path | str) -> Any:
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


def infer_docling_doc_index_from_pred_path(path: Path | str) -> int | None:
    """
    If ``path`` contains ``page_NNNN`` (e.g. native/docling/page_0003/docling.json), return ``NNNN`` as int.

    That index usually matches the position of that page inside an omnibus Docling JSON array
    (page_0000 → 0, page_0003 → 3). This fixes using ``inner_id - 1`` when task 1933 maps to page_0003.
    """
    m = re.search(r"page_(\d+)", Path(path).as_posix(), re.I)
    if not m:
        return None
    return int(m.group(1))


def resolve_docling_list_index(
    json_path: Path | str,
    *,
    doc_index: int | None,
    inner_id: int | None,
) -> tuple[int, list[str]]:
    """
    Choose which element of a Docling JSON **array** to use.

    Priority: explicit ``doc_index`` → infer from ``page_NNNN`` in path → ``inner_id - 1`` → ``0``.
    """
    notes: list[str] = []
    path = Path(json_path)
    if doc_index is not None:
        return int(doc_index), notes
    inferred = infer_docling_doc_index_from_pred_path(path)
    if inferred is not None:
        notes.append(
            f"doc_index={inferred} inferred from page_NNNN in pred path (not inner_id); "
            f"for omnibus JSON this should match page_{{inferred:04d}}"
        )
        return inferred, notes
    if inner_id is not None:
        fallback = int(inner_id) - 1
        notes.append(
            f"doc_index={fallback} from inner_id-1 (no page_NNNN in path — wrong for omnibus JSON "
            f"unless export order equals LS inner_id); pass --docling-doc-index explicitly"
        )
        return fallback, notes
    notes.append("doc_index=0 (default; no page in path, no inner_id)")
    return 0, notes


def pick_docling_document(
    data: Any,
    *,
    list_index: int,
) -> tuple[dict[str, Any], list[str]]:
    """If root is a list, pick ``data[list_index]`` (clamped). If a dict, return it."""
    notes: list[str] = []
    if isinstance(data, list):
        if not data:
            raise ValueError("Docling JSON list is empty")
        desired = int(list_index)
        idx = max(0, min(desired, len(data) - 1))
        if desired != idx:
            notes.append(
                f"warn: requested Docling doc index {desired} but list length is {len(data)}; using {idx}"
            )
        doc = data[idx]
        if not isinstance(doc, dict):
            raise ValueError(f"Docling document at index {idx} is not an object")
        return doc, notes
    if isinstance(data, dict):
        return data, notes
    raise ValueError(f"Unexpected Docling JSON root type: {type(data).__name__}")


def _page_size_for(doc: dict[str, Any], page_no: int) -> tuple[float, float]:
    pages = doc.get("pages") or {}
    key = str(page_no)
    info = pages.get(key)
    if info is None and pages:
        info = next(iter(pages.values()))
    if not info:
        raise ValueError("Docling document has no pages.size; cannot convert bboxes")
    size = info.get("size") or {}
    w = float(size.get("width", 0))
    h = float(size.get("height", 0))
    if w <= 0 or h <= 0:
        raise ValueError("Docling page size.width/height missing or invalid")
    return w, h


def iter_docling_text_spans(doc: dict[str, Any], page_no: int = 1) -> list[DoclingTextSpan]:
    pw, ph = _page_size_for(doc, page_no)
    out: list[DoclingTextSpan] = []
    for item in doc.get("texts") or []:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or item.get("orig") or "").strip()
        if not text:
            continue
        for prov in item.get("prov") or []:
            if not isinstance(prov, dict):
                continue
            pno = prov.get("page_no")
            if pno is not None and int(pno) != int(page_no):
                continue
            bb = prov.get("bbox")
            if not isinstance(bb, dict):
                continue
            box = docling_bbox_to_box_norm(bb, pw, ph)
            if box is None or box.area <= 0:
                continue
            out.append(DoclingTextSpan(text=text, box=box, page_no=int(pno or page_no)))
            break
    return out


def _indices_overlapping_gt(
    gt: BoxNorm,
    gt_poly: list[tuple[float, float]] | None,
    spans: list[DoclingTextSpan],
    *,
    min_iou: float,
    center_in_gt: bool = True,
) -> list[int]:
    idxs: list[int] = []
    for j, sp in enumerate(spans):
        iv = iou_polygon_vs_box(gt_poly, sp.box) if gt_poly else iou(gt, sp.box)
        if iv >= min_iou:
            idxs.append(j)
            continue
        if center_in_gt:
            cx, cy = sp.box.center()
            if gt_poly:
                inside = point_in_polygon(cx, cy, gt_poly)
            else:
                inside = gt.x0 <= cx <= gt.x1 and gt.y0 <= cy <= gt.y1
            if inside:
                idxs.append(j)
    idxs.sort(key=lambda k: reading_order_sort_key(spans[k].box))
    return idxs


def match_gt_to_docling_layout(
    regions: list[dict[str, Any]],
    doc: dict[str, Any],
    *,
    config: TextEvalConfig | None = None,
    page_no: int = 1,
    min_iou: float = 0.05,
    doc_pick_notes: list[str] | None = None,
) -> TaskTextEvalResult:
    """
    For each GT region with ``bbox_pct``, merge Docling text spans whose box overlaps GT
    (IoU ≥ ``min_iou`` or span center inside GT), sort by top-left reading order, score NED.
    """
    cfg = config or TextEvalConfig()
    spans = iter_docling_text_spans(doc, page_no=page_no)
    pred_strings = [s.text for s in spans]

    result = TaskTextEvalResult(
        task_id=None,
        pred_segments=pred_strings,
        match_mode=cfg.match_mode,
        alignment="layout-docling",
    )
    result.notes.append("pairing=IoU/center overlap (match_mode applies only to normalize, not assignment)")
    for w in doc_pick_notes or ():
        result.notes.append(w)
    result.notes.append(f"docling_spans={len(spans)} page_no={page_no} min_iou={min_iou}")

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
            spans,
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


def match_docling_to_docling_layout(
    gt_doc: dict[str, Any],
    pred_doc: dict[str, Any],
    *,
    config: TextEvalConfig | None = None,
    page_no: int = 1,
    min_iou: float = 0.05,
    doc_pick_notes: list[str] | None = None,
) -> TaskTextEvalResult:
    """
    Pair **600 DPI Docling JSON** (``gt_doc``) against **300 DPI Docling JSON** (``pred_doc``).

    Each GT text span on ``page_no`` becomes a synthetic region (``bbox_pct`` + ``transcription_gt``);
    prediction merges overlapping spans on ``pred_doc`` (same rule as ``match_gt_to_docling_layout``).
    """
    gt_spans = iter_docling_text_spans(gt_doc, page_no=page_no)
    regions: list[dict[str, Any]] = []
    for i, sp in enumerate(gt_spans):
        x0, y0, x1, y1 = sp.box.x0, sp.box.y0, sp.box.x1, sp.box.y1
        w = (x1 - x0) * 100.0
        h = (y1 - y0) * 100.0
        x = x0 * 100.0
        y = y0 * 100.0
        regions.append(
            {
                "transcription_gt": sp.text,
                "region_id": str(i),
                "bbox_pct": {"x": x, "y": y, "width": w, "height": h},
            }
        )
    notes = list(doc_pick_notes or [])
    notes.append("dpi_compare: GT regions = Docling text spans (600 DPI export)")
    return match_gt_to_docling_layout(
        regions,
        pred_doc,
        config=config,
        page_no=page_no,
        min_iou=min_iou,
        doc_pick_notes=notes,
    )


def match_gt_to_docling_json_path(
    regions: list[dict[str, Any]],
    json_path: Path | str,
    *,
    config: TextEvalConfig | None = None,
    page_no: int = 1,
    doc_index: int | None = None,
    inner_id: int | None = None,
    min_iou: float = 0.05,
) -> TaskTextEvalResult:
    path = Path(json_path)
    data = load_docling_json(path)
    if isinstance(data, list):
        list_index, resolve_notes = resolve_docling_list_index(
            path,
            doc_index=doc_index,
            inner_id=inner_id,
        )
    else:
        list_index = 0
        resolve_notes = []
    doc, pick_notes = pick_docling_document(data, list_index=list_index)
    all_notes = [*resolve_notes, *pick_notes]
    return match_gt_to_docling_layout(
        regions,
        doc,
        config=config,
        page_no=page_no,
        min_iou=min_iou,
        doc_pick_notes=all_notes,
    )
