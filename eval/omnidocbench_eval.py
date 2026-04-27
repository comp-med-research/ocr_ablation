"""
OmniDocBench-inspired evaluation on Label Studio manifest ground truth.

OmniDocBench flattens structured layout GT into a document-like string and scores against
a full-page prediction. We approximate that by:

1. **Reading order** — sort ``regions`` by ``bbox_pct`` (top-to-bottom, left-to-right) using
   the same row-bucketing idea as layout IoU code (``reading_order_sort_key``).
2. **Region alignment** — run the usual ``match_gt_to_prediction`` on the **sorted** region
   list (same quick/simple/full matchers and ``prediction_segments`` on the model output).
3. **Document-level metrics** — concatenate normalized region texts with blank lines
   (paragraph breaks), normalize the full prediction, then NED / CER / WER on that **single
   pair** (pooled page-level string edit, analogous to scoring one synthetic GT MD vs one pred).

This does **not** import OmniDocBench JSON schema; it reuses your LS ``transcription_gt`` boxes.
"""

from __future__ import annotations

from typing import Any

from .edit_distance import character_error_rate, normalized_edit_distance, word_error_rate
from .layout_geometry import BoxNorm, box_from_bbox_pct, reading_order_sort_key
from .matching import TaskTextEvalResult, TextEvalConfig, match_gt_to_prediction
from .normalize import strip_vlm_output_artifacts


def sort_regions_reading_order(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort Label Studio regions by on-page reading order (bbox center row, then x)."""
    scored: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for r in regions:
        bp = r.get("bbox_pct")
        box = box_from_bbox_pct(bp) if isinstance(bp, dict) else None
        if box is None or box.area <= 0:
            key = reading_order_sort_key(BoxNorm(0.0, 0.0, 1e-6, 1e-6))
        else:
            key = reading_order_sort_key(box)
        scored.append((key, r))
    scored.sort(key=lambda x: (x[0][0], x[0][1]))
    return [r for _, r in scored]


def match_gt_to_prediction_omnidocbench_style(
    regions: list[dict[str, Any]],
    prediction_full_text: str,
    *,
    config: TextEvalConfig | None = None,
) -> TaskTextEvalResult:
    """
    Region matching on **reading-order** GT plus **document-level** NED/CER/WER on
    concatenated GT vs full-page prediction (both fully normalized).
    """
    cfg = config or TextEvalConfig()
    ordered = sort_regions_reading_order(list(regions))
    er = match_gt_to_prediction(ordered, prediction_full_text, config=cfg)
    er.notes.insert(
        0,
        "eval_style=omnidocbench: GT regions sorted by bbox; document_* = full-page string metrics.",
    )

    pred_stripped = strip_vlm_output_artifacts(prediction_full_text or "")
    gt_parts = [cfg.normalize((r.get("transcription_gt") or "").strip()) for r in ordered]
    gt_parts = [p for p in gt_parts if p]
    gt_doc = "\n\n".join(gt_parts)
    pred_doc = cfg.normalize(pred_stripped)
    er.document_gt_norm = gt_doc
    er.document_pred_norm = pred_doc
    er.document_ned = float(normalized_edit_distance(gt_doc, pred_doc))
    er.document_cer = float(character_error_rate(gt_doc, pred_doc))
    er.document_wer = float(word_error_rate(gt_doc, pred_doc))
    return er
