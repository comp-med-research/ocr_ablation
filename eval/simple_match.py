"""
OmniDocBench ``match_gt2pred_simple`` (Hungarian one-to-one), text-only.

No table cell splitting, no category typing — same string lists as ``quick_match``.
"""

from __future__ import annotations

from typing import Any, Callable

from scipy.optimize import linear_sum_assignment

from .edit_distance import compute_edit_distance_matrix_new, normalized_edit_distance


def simple_match_gt_pred_lines(
    gt_raw: list[str],
    gt_norm: list[str],
    pred_raw: list[str],
    pred_norm: list[str],
    normalize: Callable[[str], str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    One-to-one minimum-cost matching on normalized edit distance (OmniDocBench ``simple_match``).

    Returns (per_gt_rows, notes) where per_gt_rows has length n_gt, ordered by gt index.
    """
    notes: list[str] = []
    n, m = len(gt_raw), len(pred_raw)
    if n == 0:
        return [], ["no_gt_regions"]
    if m == 0:
        return (
            [
                {
                    "gt_index": i,
                    "pred_raw": "",
                    "pred_norm": "",
                    "ned": normalized_edit_distance(gt_norm[i], ""),
                    "pred_segment_start": 0,
                    "pred_segment_end": 0,
                    "pred_indices": [],
                }
                for i in range(n)
            ],
            ["empty_prediction"],
        )

    if n == 1 and m == 1:
        ned = normalized_edit_distance(gt_norm[0], pred_norm[0])
        return (
            [
                {
                    "gt_index": 0,
                    "pred_raw": pred_raw[0],
                    "pred_norm": pred_norm[0],
                    "ned": ned,
                    "pred_segment_start": 0,
                    "pred_segment_end": 1,
                    "pred_indices": [0],
                }
            ],
            [],
        )

    cost = compute_edit_distance_matrix_new(gt_norm, pred_norm)
    row_ind, col_ind = linear_sum_assignment(cost)
    col_matched = set(int(c) for c in col_ind)

    col_for_gt: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        col_for_gt[int(r)] = int(c)

    unmatched_preds = [j for j in range(m) if j not in col_matched]
    if unmatched_preds:
        notes.append(f"{len(unmatched_preds)} unmatched pred segment(s) (simple_match is one-to-one)")

    out: list[dict[str, Any]] = []
    for i in range(n):
        if i in col_for_gt:
            pj = col_for_gt[i]
            pr = pred_raw[pj]
            pn = pred_norm[pj]
            gn = normalize(gt_raw[i])
            ned = normalized_edit_distance(gn, normalize(pr))
            out.append(
                {
                    "gt_index": i,
                    "pred_raw": pr,
                    "pred_norm": pn,
                    "ned": ned,
                    "pred_segment_start": pj,
                    "pred_segment_end": pj + 1,
                    "pred_indices": [pj],
                }
            )
        else:
            gn = normalize(gt_raw[i])
            out.append(
                {
                    "gt_index": i,
                    "pred_raw": "",
                    "pred_norm": "",
                    "ned": normalized_edit_distance(gn, ""),
                    "pred_segment_start": 0,
                    "pred_segment_end": 0,
                    "pred_indices": [],
                }
            )
    return out, notes
