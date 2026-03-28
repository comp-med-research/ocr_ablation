"""Ordered matching: GT regions vs merged prediction segments (OmniDocBench-style DP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .normalize import normalize_text, segment_prediction


def levenshtein_distance(a: str, b: str) -> int:
    """Classic O(len(a)*len(b)) dynamic programming."""
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
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[lb]


def normalized_edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    d = levenshtein_distance(a, b)
    return d / max(len(a), len(b), 1)


def similarity_from_ned(a: str, b: str) -> float:
    return 1.0 - normalized_edit_distance(a, b)


@dataclass
class TextEvalConfig:
    """Normalization toggles for scoring (both sides use the same pipeline)."""

    unicode_form: str = "NFKC"
    strip_md_images: bool = True
    strip_fences: bool = True
    strip_html_comm: bool = True
    collapse_repeats: bool = True
    max_char_repeat: int = 3
    lowercase: bool = False

    def normalize(self, s: str) -> str:
        return normalize_text(
            s,
            unicode_form=self.unicode_form,
            strip_md_images=self.strip_md_images,
            strip_fences=self.strip_fences,
            strip_html_comm=self.strip_html_comm,
            collapse_repeats=self.collapse_repeats,
            max_char_repeat=self.max_char_repeat,
            lowercase=self.lowercase,
        )


@dataclass
class RegionMatch:
    gt_index: int
    region_id: str
    gt_raw: str
    pred_raw_merged: str
    gt_norm: str
    pred_norm: str
    ned: float
    similarity: float
    pred_segment_start: int
    pred_segment_end: int  # exclusive index into pred_segments


@dataclass
class TaskTextEvalResult:
    task_id: Any
    regions_matched: list[RegionMatch] = field(default_factory=list)
    mean_ned: float = 0.0
    mean_similarity: float = 0.0
    micro_ned: float = 0.0  # pooled chars
    pred_segments: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _merge_pred_segments(parts: list[str], start: int, end: int) -> str:
    if start >= end:
        return ""
    return "\n\n".join(parts[start:end])


def match_gt_to_prediction(
    regions: list[dict[str, Any]],
    prediction_full_text: str,
    *,
    config: TextEvalConfig | None = None,
    sim_fn: Callable[[str, str], float] | None = None,
) -> TaskTextEvalResult:
    """
    Match each GT region (in list order) to a contiguous merge of prediction paragraphs.

    Dynamic program: ``dp[i][j]`` = best total similarity after matching first ``i`` GT
    regions using prediction segments ``0 .. j-1`` (all of those segments consumed),
    with segment ``j-1`` ending the block for GT ``i-1``.

    Final score: ``max_j dp[n][j]`` so trailing prediction paragraphs may be unused.
    """
    cfg = config or TextEvalConfig()
    sim = sim_fn or (lambda g, p: similarity_from_ned(g, p))

    gts_raw = [r.get("transcription_gt") or "" for r in regions]
    ids = [str(r.get("region_id", "")) for r in regions]
    n = len(gts_raw)
    pred_segments = segment_prediction(prediction_full_text)
    m = len(pred_segments)

    result = TaskTextEvalResult(task_id=None, pred_segments=pred_segments)

    if n == 0:
        result.notes.append("no_gt_regions")
        return result

    if m == 0:
        for i in range(n):
            gn = cfg.normalize(gts_raw[i])
            ned = normalized_edit_distance(gn, "")
            result.regions_matched.append(
                RegionMatch(
                    gt_index=i,
                    region_id=ids[i],
                    gt_raw=gts_raw[i],
                    pred_raw_merged="",
                    gt_norm=gn,
                    pred_norm="",
                    ned=ned,
                    similarity=1.0 - ned,
                    pred_segment_start=0,
                    pred_segment_end=0,
                )
            )
        _aggregate_metrics(result)
        result.notes.append("empty_prediction")
        return result

    gts_norm = [cfg.normalize(t) for t in gts_raw]

    # dp[i][j]: first i GT, consumed pred[0:j]
    neg = -1e18
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    back_k = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_val = neg
            best_k = 0
            for k in range(j):
                chunk = _merge_pred_segments(pred_segments, k, j)
                pn = cfg.normalize(chunk)
                add = sim(gts_norm[i - 1], pn)
                val = dp[i - 1][k] + add
                if val > best_val:
                    best_val = val
                    best_k = k
            dp[i][j] = best_val
            back_k[i][j] = best_k

    best_j = 1
    best_val = neg
    for j in range(1, m + 1):
        if dp[n][j] > best_val:
            best_val = dp[n][j]
            best_j = j

    # Backtrace
    matches_rev: list[tuple[int, int, int]] = []
    j = best_j
    for i in range(n, 0, -1):
        k = back_k[i][j]
        matches_rev.append((i - 1, k, j))
        j = k
    matches_rev.reverse()

    for gt_i, k, j in matches_rev:
        chunk_raw = _merge_pred_segments(pred_segments, k, j)
        gn = gts_norm[gt_i]
        pn = cfg.normalize(chunk_raw)
        ned = normalized_edit_distance(gn, pn)
        result.regions_matched.append(
            RegionMatch(
                gt_index=gt_i,
                region_id=ids[gt_i],
                gt_raw=gts_raw[gt_i],
                pred_raw_merged=chunk_raw,
                gt_norm=gn,
                pred_norm=pn,
                ned=ned,
                similarity=1.0 - ned,
                pred_segment_start=k,
                pred_segment_end=j,
            )
        )

    _aggregate_metrics(result)
    return result


def _aggregate_metrics(result: TaskTextEvalResult) -> None:
    rm = result.regions_matched
    if not rm:
        return
    result.mean_ned = sum(x.ned for x in rm) / len(rm)
    result.mean_similarity = sum(x.similarity for x in rm) / len(rm)
    num = den = 0.0
    for x in rm:
        w = float(max(len(x.gt_norm), len(x.pred_norm), 1))
        num += x.ned * w
        den += w
    result.micro_ned = num / den if den else 0.0
