"""Shared Levenshtein + normalized edit distance (OmniDocBench-compatible costs)."""

from __future__ import annotations

import numpy as np


def _levenshtein_python(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    la, lb = len(a), len(b)
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def levenshtein_distance(a: str, b: str) -> int:
    """Prefer C implementation (OmniDocBench-style); fall back to pure Python."""
    try:
        from Levenshtein import distance as _c_dist  # type: ignore[import-untyped]

        return int(_c_dist(a, b))
    except Exception:
        return _levenshtein_python(a, b)


def normalized_edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    d = levenshtein_distance(a, b)
    return d / max(len(a), len(b), 1)


def compute_edit_distance_matrix_new(gt_lines: list[str], matched_lines: list[str]) -> np.ndarray:
    """Same as OmniDocBench ``utils.match.compute_edit_distance_matrix_new``."""
    distance_matrix = np.zeros((len(gt_lines), len(matched_lines)))
    for i, gt_line in enumerate(gt_lines):
        for j, matched_line in enumerate(matched_lines):
            if len(gt_line) == 0 and len(matched_line) == 0:
                distance_matrix[i][j] = 0.0
            else:
                distance_matrix[i][j] = levenshtein_distance(gt_line, matched_line) / max(
                    len(matched_line), len(gt_line), 1
                )
    return distance_matrix
