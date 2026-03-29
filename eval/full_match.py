"""
OmniDocBench ``match_full`` / ``FuzzyMatch`` (substring DP + multi-stage combine), text-only.

Ported from https://github.com/opendatalab/OmniDocBench/blob/main/utils/match_full.py
using ``eval.edit_distance.levenshtein_distance`` instead of ``Levenshtein``.

``_combine_match`` uses the mean of edit distances (upstream mistakenly used ``np.mean`` on the wrong object). ``match_gt2pred_full``: ``seen_gt_s`` updates are applied per-entry (upstream indentation was wrong).
"""

from __future__ import annotations

from typing import Any, Callable

from .edit_distance import normalized_edit_distance


class FuzzyMatch:
    def __init__(self, gts: list[str], preds: list[str]) -> None:
        self._gs = gts
        self._preds = preds

    def match(
        self,
    ) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]], dict[int, list[int]]]:
        equal_match_pair: dict[int, int] = {}
        gs_used_s: set[int] = set()
        preds_used_s: set[int] = set()
        for i in range(len(self._gs)):
            for j in range(len(self._preds)):
                if self._gs[i] == self._preds[j]:
                    equal_match_pair[i] = j
                    gs_used_s.add(i)
                    preds_used_s.add(j)

        combine_gs_match_preds_ret_h = self._combine_match(self._gs, self._preds, gs_used_s, preds_used_s)

        for pred_idx in combine_gs_match_preds_ret_h.keys():
            preds_used_s.add(pred_idx)
            for gt_idx, _ in combine_gs_match_preds_ret_h[pred_idx]:
                gs_used_s.add(gt_idx)

        combine_preds_match_gs_ret_h = self._combine_match(self._preds, self._gs, preds_used_s, gs_used_s)

        for gt_idx in combine_preds_match_gs_ret_h.keys():
            gs_used_s.add(gt_idx)
            for pred_idx, _ in combine_preds_match_gs_ret_h[gt_idx]:
                preds_used_s.add(pred_idx)

        gs_free_arr = [i for i in range(len(self._gs)) if i not in gs_used_s]
        pred_free_arr = [i for i in range(len(self._preds)) if i not in preds_used_s]

        match_gs_free_pred_combine_ret_h = self._free_match_1(
            gs_free_arr, self._gs, combine_gs_match_preds_ret_h, self._preds
        )
        match_pred_free_gs_combine_ret_h = self._free_match_1(
            pred_free_arr, self._preds, combine_preds_match_gs_ret_h, self._gs
        )

        # Keys must be actual (gt_idx, pred_idx) — upstream used loop indices by mistake.
        pred_free_gt_free_h: dict[tuple[int, int], tuple[int, int]] = {}
        for i in range(len(pred_free_arr)):
            for j in range(len(gs_free_arr)):
                pi, gj = pred_free_arr[i], gs_free_arr[j]
                edit_dis, pos = self._dp(self._preds[pi], self._gs[gj])
                pred_free_gt_free_h[(gj, pi)] = (edit_dis, pos)

        gt_free_pred_free_h: dict[tuple[int, int], tuple[int, int]] = {}
        for i in range(len(gs_free_arr)):
            for j in range(len(pred_free_arr)):
                gi, pj = gs_free_arr[i], pred_free_arr[j]
                edit_dis, pos = self._dp(self._gs[gi], self._preds[pj])
                gt_free_pred_free_h[(pj, gi)] = (edit_dis, pos)

        class MatchPair:
            __slots__ = ("pred_idx", "gt_idx")

            def __init__(self, pred_idx: int, gt_idx: int) -> None:
                self.pred_idx = pred_idx
                self.gt_idx = gt_idx

        edit_dis_q: list[tuple[tuple[int, int], MatchPair]] = []

        for free_gt_idx in gs_free_arr:
            for pred_idx in range(len(self._preds)):
                if (pred_idx, free_gt_idx) in match_gs_free_pred_combine_ret_h:
                    edit_dis_q.append(
                        (match_gs_free_pred_combine_ret_h[(pred_idx, free_gt_idx)], MatchPair(pred_idx, free_gt_idx))
                    )
                if (pred_idx, free_gt_idx) in gt_free_pred_free_h:
                    edit_dis_q.append((gt_free_pred_free_h[(pred_idx, free_gt_idx)], MatchPair(pred_idx, free_gt_idx)))

        for free_pred_idx in pred_free_arr:
            for gt_idx in range(len(self._gs)):
                if (gt_idx, free_pred_idx) in match_pred_free_gs_combine_ret_h:
                    edit_dis_q.append(
                        (
                            match_pred_free_gs_combine_ret_h[(gt_idx, free_pred_idx)],
                            MatchPair(free_pred_idx, gt_idx),
                        )
                    )
                if (gt_idx, free_pred_idx) in pred_free_gt_free_h:
                    edit_dis_q.append((pred_free_gt_free_h[(gt_idx, free_pred_idx)], MatchPair(free_pred_idx, gt_idx)))

        matched_gt_pred_h: dict[tuple[int, int, int], bool] = {}

        for gt_idx in equal_match_pair:
            pred_idx = equal_match_pair[gt_idx]
            matched_gt_pred_h[(gt_idx, pred_idx, -1)] = True

        for gt_idx in combine_preds_match_gs_ret_h.keys():
            for pred_idx, pos in combine_preds_match_gs_ret_h[gt_idx]:
                matched_gt_pred_h[(gt_idx, pred_idx, pos)] = True

        for pred_idx in combine_gs_match_preds_ret_h.keys():
            for gt_idx, pos in combine_gs_match_preds_ret_h[pred_idx]:
                matched_gt_pred_h[(gt_idx, pred_idx, pos)] = True

        edit_dis_q.sort(key=lambda x: x[0][0])

        used_pred_idx_s: set[int] = set()
        used_gt_idx_s: set[int] = set()

        for match_info, p in edit_dis_q:
            edit_dis, pos = match_info
            pred_idx, gt_idx = p.pred_idx, p.gt_idx
            la, lb = len(self._preds[pred_idx]), len(self._gs[gt_idx])
            md = min(la, lb)
            denom = md if md > 0 else max(la, lb, 1)
            if edit_dis * 1.0 / denom >= 0.5:
                continue
            if pred_idx in pred_free_arr and pred_idx in used_pred_idx_s:
                continue
            if gt_idx in gs_free_arr and gt_idx in used_gt_idx_s:
                continue

            matched_gt_pred_h[(gt_idx, pred_idx, pos)] = True

            if pred_idx in pred_free_arr:
                used_pred_idx_s.add(pred_idx)
            if gt_idx in gs_free_arr:
                used_gt_idx_s.add(gt_idx)

        group_by_gt: dict[int, list[tuple[int, int]]] = {}
        group_by_pred: dict[int, list[tuple[int, int]]] = {}
        for gt_idx, pred_idx, pos in matched_gt_pred_h.keys():
            group_by_gt.setdefault(gt_idx, []).append((pred_idx, pos))
            group_by_pred.setdefault(pred_idx, []).append((gt_idx, pos))

        one: set[tuple[int, int, int]] = set()
        for gt_idx in group_by_gt.keys():
            if len(group_by_gt[gt_idx]) == 1:
                one.add((gt_idx, *group_by_gt[gt_idx][0]))
        for pred_idx in group_by_pred.keys():
            if len(group_by_pred[pred_idx]) == 1:
                gt_idx, pos = group_by_pred[pred_idx][0]
                one.add((gt_idx, pred_idx, pos))

        gt_one_pred: dict[int, list[int]] = {}
        for gt_idx, pred_idx, pos in one:
            if gt_idx in group_by_gt and pred_idx in group_by_pred:
                if len(group_by_gt[gt_idx]) == 1 and len(group_by_pred[pred_idx]) == 1:
                    gt_one_pred[gt_idx] = [pred_idx, pos]
                elif len(group_by_gt[gt_idx]) == 1:
                    group_by_gt.pop(gt_idx, None)
                else:
                    group_by_pred.pop(pred_idx, None)

        return group_by_gt, group_by_pred, gt_one_pred

    def _free_match_1(
        self,
        free_source_idx: list[int],
        source_arr: list[str],
        combined_target_h: dict[int, list[tuple[int, int]]],
        combined_target_arr: list[str],
    ) -> dict[tuple[int, int], tuple[int, int]]:
        ret: dict[tuple[int, int], tuple[int, int]] = {}

        def _do_match(matched_target_idx: int, target_str_segment: str) -> None:
            for free_idx in free_source_idx:
                edit_dis, pos = self._dp(source_arr[free_idx], target_str_segment)
                key = (matched_target_idx, free_idx)
                if key not in ret or ret[key][0] > edit_dis:
                    ret[key] = (edit_dis, pos)

        for matched_target_idx in combined_target_h.keys():
            matched_source_idx_pos = sorted(combined_target_h[matched_target_idx], key=lambda x: x[1])
            if len(matched_source_idx_pos) == 0:
                continue

            for i, v in enumerate(matched_source_idx_pos):
                matched_source_idx, pos = v
                if i == 0:
                    hole_len = pos + 1 - len(source_arr[matched_source_idx])
                else:
                    hole_len = pos - len(source_arr[matched_source_idx]) - matched_source_idx_pos[i - 1][1]

                if hole_len <= 0:
                    continue
                target_str_segment = combined_target_arr[matched_target_idx][pos + 1 - hole_len : pos + 1]
                _do_match(matched_target_idx, target_str_segment)

            tail = combined_target_arr[matched_target_idx][matched_source_idx_pos[-1][1] + 1 :]
            if len(tail) > 0:
                _do_match(matched_target_idx, tail)

        return ret

    def slide_window_dp(self, line: str, window: str) -> list[list[float]]:
        n, m = len(line), len(window)
        dp = [[float("inf")] * m for _ in range(n)]
        for i in range(n):
            dp[i][0] = 1
            if line[i] == window[0]:
                dp[i][0] = 0

        for j in range(1, m):
            for i in range(1, n):
                dp[i][j] = dp[i - 1][j - 1]
                if line[i] != window[j]:
                    dp[i][j] += 1
                dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1, dp[i - 1][j] + 1)
        return dp

    def _dp(self, window: str, line: str) -> tuple[int, int]:
        if not window:
            return 0, 0
        if not line:
            return len(window), 0
        dp = self.slide_window_dp(line, window)
        ret = float("inf")
        pos = 0
        last = len(window) - 1
        for i in range(len(line)):
            if ret > dp[i][last]:
                ret = dp[i][last]
                pos = i
        return int(ret), pos

    def _combine_match(
        self,
        window_arr: list[str],
        line_arr: list[str],
        window_used_s: set[int],
        line_used_s: set[int],
    ) -> dict[int, list[tuple[int, int]]]:
        match_edit_dis_ratio = 0.10
        abs_diff_len = 20
        abs_diff_char_count = 5
        sigma_multiple = 2
        edit_dis_h: dict[tuple[int, int], tuple[int, int]] = {}

        for i in range(len(window_arr)):
            if i in window_used_s:
                continue
            for j in range(len(line_arr)):
                if j in line_used_s:
                    continue
                edit_dis_h[(i, j)] = self._dp(window_arr[i], line_arr[j])

        matched_pair_h_gt: dict[int, list[tuple[int, int]]] = {}
        matched_gt_idx_s: set[int] = set()
        for i in range(len(window_arr)):
            if i in window_used_s:
                continue
            edit_dis, pos = float("inf"), -1
            min_j_idx = -1

            for j in range(len(line_arr)):
                if j in line_used_s:
                    continue
                if edit_dis > edit_dis_h[(i, j)][0]:
                    edit_dis, pos = edit_dis_h[(i, j)]
                    min_j_idx = j

            if pos == -1 or min_j_idx < 0:
                continue
            matched_pair_h_gt.setdefault(min_j_idx, [])
            wlen = len(window_arr[i])
            if edit_dis < wlen * match_edit_dis_ratio or (abs_diff_len >= wlen and abs_diff_char_count >= edit_dis):
                matched_pair_h_gt[min_j_idx].append((i, pos))
                matched_gt_idx_s.add(i)

        for i in range(len(window_arr)):
            if i in window_used_s or i in matched_gt_idx_s:
                continue

            edit_dis_pair: list[tuple[int, int, int]] = []
            for j in range(len(line_arr)):
                if j in line_used_s:
                    continue
                edit_dis_pair.append((j, *edit_dis_h[i, j]))

            if len(edit_dis_pair) == 0:
                continue
            if len(edit_dis_pair) == 1:
                best_j_idx, edit_dis, pos = edit_dis_pair[0]
                matched_gt_idx_s.add(i)
                matched_pair_h_gt.setdefault(best_j_idx, []).append((i, pos))
                continue

            vals = [float(edit_dis) for _, edit_dis, _ in edit_dis_pair]
            mean = sum(vals) / len(vals)
            std_var = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0

            beyond_sigma = sorted(
                [(j, edit_dis, pos) for j, edit_dis, pos in edit_dis_pair if mean - sigma_multiple * std_var >= edit_dis],
                key=lambda x: x[1],
            )
            if len(beyond_sigma) > 0:
                matched_pair_h_gt.setdefault(beyond_sigma[0][0], []).append((i, beyond_sigma[0][2]))

        return {k: v for k, v in matched_pair_h_gt.items() if len(v) > 0}


def match_gt_pred(
    gts: list[str], predications: list[str]
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]], dict[int, list[int]]]:
    if any(len(v) == 0 for v in predications):
        raise ValueError("OmniDocBench full match: remove empty strings from predications list")
    if len(predications) == 0 or len(gts) == 0:
        return {}, {}, {}
    return FuzzyMatch(gts, predications).match()


def match_gt2pred_full(gts: list[str], predications: list[str]) -> list[dict[str, Any]]:
    group_by_gt, group_by_pred, gt_one_pred = match_gt_pred(gts, predications)
    seen_gt_s: set[int] = set()

    ret: list[dict[str, Any]] = []
    for gt_idx in group_by_gt.keys():
        matched_preds = [p[0] for p in sorted(group_by_gt[gt_idx], key=lambda x: x[1])]
        ret.append(
            {
                "gt_idx": [gt_idx],
                "gt": gts[gt_idx],
                "pred_idx": matched_preds,
                "pred": "".join(predications[pr_idx] for pr_idx in matched_preds),
            }
        )
        seen_gt_s.add(gt_idx)

    for pred_idx in group_by_pred:
        matched_gts = [p[0] for p in sorted(group_by_pred[pred_idx], key=lambda x: x[1])]
        ret.append(
            {
                "gt_idx": matched_gts,
                "gt": "".join(gts[gt_idx] for gt_idx in matched_gts),
                "pred_idx": [pred_idx],
                "pred": predications[pred_idx],
            }
        )
        for gt_idx in matched_gts:
            seen_gt_s.add(gt_idx)

    for gt_idx in gt_one_pred.keys():
        pred_idx = gt_one_pred[gt_idx][0]
        ret.append(
            {
                "gt_idx": [gt_idx],
                "gt": gts[gt_idx],
                "pred_idx": [pred_idx],
                "pred": predications[pred_idx],
            }
        )
        seen_gt_s.add(gt_idx)

    for i in range(len(gts)):
        if i not in seen_gt_s:
            ret.append({"gt_idx": [i], "gt": gts[i], "pred_idx": [], "pred": ""})
    return ret


def _per_gt_rows_from_fuzzy_groups(
    n_gt: int,
    gt_raw: list[str],
    pred_raw: list[str],
    group_by_gt: dict[int, list[tuple[int, int]]],
    group_by_pred: dict[int, list[tuple[int, int]]],
    gt_one_pred: dict[int, list[int]],
    compact_to_orig: dict[int, int],
    normalize: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Build one output row per GT from ``FuzzyMatch`` structures (compact pred indices)."""

    def to_orig(compact_list: list[int]) -> list[int]:
        return [compact_to_orig[c] for c in compact_list if c in compact_to_orig]

    assigned: dict[int, tuple[list[int], str]] = {}

    for gi, pair in gt_one_pred.items():
        cpi = int(pair[0])
        if cpi not in compact_to_orig:
            continue
        oi = compact_to_orig[cpi]
        assigned[gi] = ([oi], pred_raw[oi])

    for gi, plist in group_by_gt.items():
        if gi in assigned:
            continue
        c_preds = [int(p[0]) for p in sorted(plist, key=lambda x: x[1])]
        o_preds = to_orig(c_preds)
        if not o_preds:
            continue
        pred_s = "".join(pred_raw[j] for j in o_preds)
        assigned[gi] = (o_preds, pred_s)

    for cpi, gtlist in group_by_pred.items():
        if cpi not in compact_to_orig:
            continue
        oi = compact_to_orig[cpi]
        pred_s = pred_raw[oi]
        for _gt_idx, _pos in sorted(gtlist, key=lambda x: x[1]):
            gi = int(_gt_idx)
            if gi not in assigned:
                assigned[gi] = ([oi], pred_s)

    out: list[dict[str, Any]] = []
    for i in range(n_gt):
        if i in assigned:
            o_preds, pred_s = assigned[i]
            start = min(o_preds) if o_preds else 0
            end = (max(o_preds) + 1) if o_preds else 0
            gn = normalize(gt_raw[i])
            pn = normalize(pred_s)
            out.append(
                {
                    "gt_index": i,
                    "pred_raw": pred_s,
                    "pred_norm": pn,
                    "ned": normalized_edit_distance(gn, pn),
                    "pred_segment_start": start,
                    "pred_segment_end": end,
                    "pred_indices": o_preds,
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
    return out


def full_match_gt_pred_lines(
    gt_raw: list[str],
    gt_norm: list[str],
    pred_raw: list[str],
    pred_norm: list[str],
    normalize: Callable[[str], str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Run OmniDocBench ``match_gt2pred_full`` on normalized strings; map back to per-GT rows.

    Empty pred segments are dropped for ``FuzzyMatch`` (upstream forbids empty preds); indices refer
    to the original ``pred_*`` lists.
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

    keep_j: list[int] = [j for j in range(m) if pred_norm[j]]
    if len(keep_j) < m:
        notes.append(f"dropped {m - len(keep_j)} empty pred segment(s) for full_match (upstream requires non-empty)")

    if not keep_j:
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
            notes + ["all_pred_segments_empty"],
        )

    compact_norm = [pred_norm[j] for j in keep_j]
    compact_to_orig = {ci: keep_j[ci] for ci in range(len(keep_j))}

    group_by_gt, group_by_pred, gt_one_pred = match_gt_pred(gt_norm, compact_norm)
    rows = _per_gt_rows_from_fuzzy_groups(
        n,
        gt_raw,
        pred_raw,
        group_by_gt,
        group_by_pred,
        gt_one_pred,
        compact_to_orig,
        normalize,
    )
    return rows, notes
