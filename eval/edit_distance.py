"""Shared Levenshtein + normalized edit distance (OmniDocBench-compatible costs)."""

from __future__ import annotations

from typing import Any


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


def character_error_rate(ref: str, hyp: str) -> float:
    """CER = Levenshtein(ref, hyp) / len(ref). Empty reference: 0 if hyp empty, else 1.0."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / len(ref)


def word_edit_distance_tokens(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance (substitution / insert / delete tokens)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
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


def word_error_rate(ref: str, hyp: str) -> float:
    """WER = word_edit_distance(ref.split(), hyp.split()) / len(ref.split())."""
    rw = ref.split()
    hw = hyp.split()
    if not rw:
        return 0.0 if not hw else 1.0
    return word_edit_distance_tokens(rw, hw) / len(rw)


def aggregate_cer_wer(pairs: list[tuple[str, str]]) -> tuple[float, float, float, float]:
    """
    Aggregate (mean_cer, micro_cer, mean_wer, micro_wer) over aligned (ref, hyp) pairs.

    Micro CER = sum edit distances / sum(len(ref)); micro WER = sum word edit distances /
    sum of reference word counts (0 if no ref characters / words).
    """
    if not pairs:
        return (0.0, 0.0, 0.0, 0.0)
    mean_cer = sum(character_error_rate(a, b) for a, b in pairs) / len(pairs)
    mean_wer = sum(word_error_rate(a, b) for a, b in pairs) / len(pairs)
    num_c = sum(levenshtein_distance(a, b) for a, b in pairs)
    den_c = sum(len(a) for a, _ in pairs)
    micro_cer = num_c / den_c if den_c else 0.0
    num_w = sum(word_edit_distance_tokens(a.split(), b.split()) for a, b in pairs)
    den_w = sum(len(a.split()) for a, _ in pairs)
    micro_wer = num_w / den_w if den_w else 0.0
    return (mean_cer, micro_cer, mean_wer, micro_wer)


def compute_edit_distance_matrix_new(gt_lines: list[str], matched_lines: list[str]) -> Any:
    """Same as OmniDocBench ``utils.match.compute_edit_distance_matrix_new``."""
    import numpy as np

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
