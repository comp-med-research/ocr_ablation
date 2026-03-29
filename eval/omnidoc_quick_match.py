"""
OmniDocBench ``match_gt2pred_quick`` core (text-only, no table/formula split).

Ported from https://github.com/opendatalab/OmniDocBench/blob/main/utils/match_quick.py
with Levenshtein replaced by ``eval.edit_distance`` (no python-Levenshtein).

Ignores the upstream ``ignore`` category branch (headers/footers/…): all GT lines
use the main matching path, matching our Label Studio text regions.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .edit_distance import compute_edit_distance_matrix_new, levenshtein_distance, normalized_edit_distance


def merge_duplicates_add_unmatched(
    converted_results: list[dict[str, Any]],
    norm_gt_lines: list[str],
    norm_pred_lines: list[str],
    gt_lines: list[str],
    pred_lines: list[str],
    all_gt_indices: set[int],
    all_pred_indices: set[int],
) -> list[dict[str, Any]]:
    merged_results: list[dict[str, Any]] = []
    processed_pred: set[tuple[Any, ...]] = set()
    processed_gt: set[int] = set()

    for entry in converted_results:
        pred_idx = tuple(entry["pred_idx"]) if isinstance(entry["pred_idx"], list) else (entry["pred_idx"],)
        if pred_idx not in processed_pred and pred_idx != ("",):
            merged_entry = {
                "gt_idx": [entry["gt_idx"]],
                "gt": entry["gt"],
                "pred_idx": entry["pred_idx"],
                "pred": entry["pred"],
                "edit": entry["edit"],
            }
            for other_entry in converted_results:
                other_pred_idx = (
                    tuple(other_entry["pred_idx"])
                    if isinstance(other_entry["pred_idx"], list)
                    else (other_entry["pred_idx"],)
                )
                if other_pred_idx == pred_idx and other_entry is not entry:
                    merged_entry["gt_idx"].append(other_entry["gt_idx"])
                    merged_entry["gt"] += other_entry["gt"]
                    processed_gt.add(other_entry["gt_idx"])
            merged_results.append(merged_entry)
            processed_pred.add(pred_idx)
            processed_gt.add(entry["gt_idx"])

    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx not in processed_gt:
            merged_results.append(
                {
                    "gt_idx": [gt_idx],
                    "gt": gt_lines[gt_idx],
                    "pred_idx": [""],
                    "pred": "",
                    "edit": 1,
                }
            )
    return merged_results


def merge_lists_with_sublists(main_list: list[Any], sub_lists: list[list[Any]]) -> list[Any]:
    main_list_final = list(copy.deepcopy(main_list))
    for sub_list in sub_lists:
        pop_idx = main_list_final.index(sub_list[0])
        for _ in sub_list:
            main_list_final.pop(pop_idx)
        main_list_final.insert(pop_idx, sub_list)
    return main_list_final


def sub_pred_fuzzy_matching(gt: str, pred: str) -> float | bool:
    min_d = float("inf")
    gt_len = len(gt)
    pred_len = len(pred)
    if gt_len >= pred_len and pred_len > 0:
        for i in range(gt_len - pred_len + 1):
            sub = gt[i : i + pred_len]
            dist = levenshtein_distance(sub, pred) / pred_len
            if dist < min_d:
                min_d = dist
        return min_d
    return False


def sub_gt_fuzzy_matching(pred: str, gt: str) -> tuple[float, Any, int, str]:
    min_d = float("inf")
    pos: Any = ""
    matched_sub = ""
    gt_len = len(gt)
    pred_len = len(pred)
    if pred_len >= gt_len and gt_len > 0:
        for i in range(pred_len - gt_len + 1):
            sub = pred[i : i + gt_len]
            dist = levenshtein_distance(sub, gt) / gt_len
            if dist < min_d:
                min_d = dist
                pos = i
                matched_sub = sub
        return min_d, pos, gt_len, matched_sub
    return 1, "", gt_len, ""


def get_final_subset(subset_certain: list[Any], subset_certain_cost: list[float]) -> list[Any]:
    if not subset_certain or not subset_certain_cost:
        return []

    subset_turple = sorted(
        [(a, b) for a, b in zip(subset_certain, subset_certain_cost)], key=lambda x: x[0][0]
    )

    group_list: defaultdict[int, list[Any]] = defaultdict(list)
    group_idx = 0
    group_list[group_idx].append(subset_turple[0])

    for item in subset_turple[1:]:
        overlap_flag = False
        for subset in group_list[group_idx]:
            for idx in item[0]:
                if idx in subset[0]:
                    overlap_flag = True
                    break
            if overlap_flag:
                break
        if overlap_flag:
            group_list[group_idx].append(item)
        else:
            group_idx += 1
            group_list[group_idx].append(item)

    final_subset: list[Any] = []
    for _, group in group_list.items():
        if len(group) == 1:
            final_subset.append(group[0][0])
        else:
            path_dict: defaultdict[int, list[Any]] = defaultdict(list)
            path_idx = 0
            path_dict[path_idx].append(group[0])

            for subset in group[1:]:
                new_path = True
                for path_idx_s, path_items in path_dict.items():
                    is_dup = False
                    is_same = False
                    for path_item in path_items:
                        if path_item[0] == subset[0]:
                            is_dup = True
                            is_same = True
                            if path_item[1] > subset[1]:
                                path_dict[path_idx_s].pop(path_dict[path_idx_s].index(path_item))
                                path_dict[path_idx_s].append(subset)
                        else:
                            for num_1 in path_item[0]:
                                for num_2 in subset[0]:
                                    if num_1 == num_2:
                                        is_dup = True
                    if not is_dup:
                        path_dict[path_idx_s].append(subset)
                        new_path = False
                    if is_same:
                        new_path = False
                if new_path:
                    path_idx = len(path_dict.keys())
                    path_dict[path_idx].append(subset)

            saved_cost = float("inf")
            saved_subset: list[Any] = []
            for _, path in path_dict.items():
                avg_cost = sum(i[1] for i in path) / len(path)
                if avg_cost < saved_cost:
                    saved_subset = [i[0] for i in path]
                    saved_cost = avg_cost

            final_subset.extend(saved_subset)

    return final_subset


def judge_pred_merge(gt_list: list[str], pred_list: list[str], threshold: float = 0.6) -> tuple[bool, bool]:
    if len(pred_list) == 1:
        return False, False

    cur_pred = " ".join(pred_list[:-1])
    merged_pred = " ".join(pred_list)

    cur_dist = levenshtein_distance(gt_list[0], cur_pred) / max(len(gt_list[0]), len(cur_pred), 1)
    merged_dist = levenshtein_distance(gt_list[0], merged_pred) / max(
        len(gt_list[0]), len(merged_pred), 1
    )

    if merged_dist > cur_dist:
        return False, False

    cur_fuzzy_dists = [sub_pred_fuzzy_matching(gt_list[0], cur_pred) for cur_pred in pred_list[:-1]]
    if any(dist is False or dist > threshold for dist in cur_fuzzy_dists):
        return False, False

    add_fuzzy_dist = sub_pred_fuzzy_matching(gt_list[0], pred_list[-1])
    if add_fuzzy_dist is False:
        return False, False

    merged_pred_flag = add_fuzzy_dist < threshold
    continue_flag = len(merged_pred) <= len(gt_list[0])

    return merged_pred_flag, continue_flag


def deal_with_truncated(
    cost_matrix: np.ndarray,
    norm_gt_lines: list[str],
    norm_pred_lines: list[str],
) -> tuple[np.ndarray, list[str], list[Any]]:
    matched_first = np.argwhere(cost_matrix < 0.25)
    masked_gt_idx = [i[0] for i in matched_first]
    unmasked_gt_idx = [i for i in range(cost_matrix.shape[0]) if i not in masked_gt_idx]
    masked_pred_idx = [i[1] for i in matched_first]
    unmasked_pred_idx = [i for i in range(cost_matrix.shape[1]) if i not in masked_pred_idx]

    merges_gt_dict: dict[int, Any] = {}

    for gt_idx in unmasked_gt_idx:
        check_merge_subset: list[list[int]] = []
        merged_dist: list[float] = []

        for pred_idx in unmasked_pred_idx:
            step = 1
            merged_pred = [norm_pred_lines[pred_idx]]

            while True:
                if pred_idx + step in masked_pred_idx or pred_idx + step >= len(norm_pred_lines):
                    break
                merged_pred.append(norm_pred_lines[pred_idx + step])
                merged_pred_flag, continue_flag = judge_pred_merge([norm_gt_lines[gt_idx]], merged_pred)
                if not merged_pred_flag:
                    break
                step += 1
                if not continue_flag:
                    break

            check_merge_subset.append(list(range(pred_idx, pred_idx + step)))
            matched_line = " ".join(norm_pred_lines[i] for i in range(pred_idx, pred_idx + step))
            dist = levenshtein_distance(norm_gt_lines[gt_idx], matched_line) / max(
                len(matched_line), len(norm_gt_lines[gt_idx]), 1
            )
            merged_dist.append(dist)

        if not merged_dist:
            subset_certain: list[int] = []
            min_cost_idx: Any = ""
            min_cost = float("inf")
        else:
            min_cost = min(merged_dist)
            min_cost_idx = merged_dist.index(min_cost)
            subset_certain = check_merge_subset[min_cost_idx]

        merges_gt_dict[gt_idx] = {
            "merge_subset": check_merge_subset,
            "merged_cost": merged_dist,
            "min_cost_idx": min_cost_idx,
            "subset_certain": subset_certain,
            "min_cost": min_cost,
        }

    subset_certain = [
        merges_gt_dict[gt_idx]["subset_certain"]
        for gt_idx in unmasked_gt_idx
        if merges_gt_dict[gt_idx]["subset_certain"]
    ]
    subset_certain_cost = [
        merges_gt_dict[gt_idx]["min_cost"]
        for gt_idx in unmasked_gt_idx
        if merges_gt_dict[gt_idx]["subset_certain"]
    ]

    subset_certain_final = get_final_subset(subset_certain, subset_certain_cost)

    if not subset_certain_final:
        return cost_matrix, norm_pred_lines, list(range(len(norm_pred_lines)))

    final_pred_idx_list = merge_lists_with_sublists(list(range(len(norm_pred_lines))), subset_certain_final)
    final_norm_pred_lines = [
        " ".join(norm_pred_lines[idx_list[0] : idx_list[-1] + 1])
        if isinstance(idx_list, list)
        else norm_pred_lines[idx_list]
        for idx_list in final_pred_idx_list
    ]

    new_cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, final_norm_pred_lines)

    return new_cost_matrix, final_norm_pred_lines, final_pred_idx_list


def cal_final_match(
    cost_matrix: np.ndarray,
    norm_gt_lines: list[str],
    norm_pred_lines: list[str],
) -> tuple[list[Any], np.ndarray, list[float]]:
    new_cost_matrix, final_norm_pred_lines, final_pred_idx_list = deal_with_truncated(
        cost_matrix, norm_gt_lines, norm_pred_lines
    )

    row_ind, col_ind = linear_sum_assignment(new_cost_matrix)

    cost_list = [new_cost_matrix[r][c] for r, c in zip(row_ind, col_ind)]
    matched_col_idx = [final_pred_idx_list[i] for i in col_ind]

    return matched_col_idx, row_ind, cost_list


def initialize_indices(norm_gt_lines: list[str], norm_pred_lines: list[str]) -> tuple[dict[int, int], dict[int, int]]:
    gt_lens_dict = {idx: len(gt_line) for idx, gt_line in enumerate(norm_gt_lines)}
    pred_lens_dict = {idx: len(pred_line) for idx, pred_line in enumerate(norm_pred_lines)}
    return gt_lens_dict, pred_lens_dict


def process_matches(
    matched_col_idx: list[Any],
    row_ind: np.ndarray,
    cost_list: list[float],
    norm_gt_lines: list[str],
    norm_pred_lines: list[str],
    pred_lines: list[str],
) -> tuple[dict[int, Any], list[int], list[int]]:
    matches: dict[int, Any] = {}
    unmatched_gt_indices: list[int] = []
    unmatched_pred_indices: list[int] = []

    for i in range(len(norm_gt_lines)):
        if i in row_ind:
            idx = list(row_ind).index(i)
            pred_idx = matched_col_idx[idx]

            if pred_idx is None or (isinstance(pred_idx, list) and None in pred_idx):
                unmatched_pred_indices.append(pred_idx)
                continue

            if isinstance(pred_idx, list):
                pred_line = " | ".join(norm_pred_lines[pred_idx[0] : pred_idx[-1] + 1])
                ori_pred_line = " | ".join(pred_lines[pred_idx[0] : pred_idx[-1] + 1])
                matched_pred_indices_range = list(range(pred_idx[0], pred_idx[-1] + 1))
            else:
                pred_line = norm_pred_lines[pred_idx]
                ori_pred_line = pred_lines[pred_idx]
                matched_pred_indices_range = [pred_idx]

            edit = cost_list[idx]

            if edit > 0.7:
                unmatched_pred_indices.extend(matched_pred_indices_range)
                unmatched_gt_indices.append(i)
            else:
                matches[i] = {
                    "pred_indices": matched_pred_indices_range,
                    "edit_distance": edit,
                }
                for matched_pred_idx in matched_pred_indices_range:
                    if matched_pred_idx in unmatched_pred_indices:
                        unmatched_pred_indices.remove(matched_pred_idx)
        else:
            unmatched_gt_indices.append(i)

    return matches, unmatched_gt_indices, unmatched_pred_indices


def fuzzy_match_unmatched_items(
    unmatched_gt_indices: list[int],
    norm_gt_lines: list[str],
    norm_pred_lines: list[str],
) -> dict[int, list[int]]:
    matching_dict: dict[int, list[int]] = {}

    for pred_idx, pred_content in enumerate(norm_pred_lines):
        if isinstance(pred_idx, list):
            continue

        matching_indices: list[int] = []

        for unmatched_gt_idx in unmatched_gt_indices:
            gt_content = norm_gt_lines[unmatched_gt_idx]
            cur_fuzzy_dist_unmatch, _cur_pos, _gt_lens, _matched_field = sub_gt_fuzzy_matching(
                pred_content, gt_content
            )
            if cur_fuzzy_dist_unmatch < 0.4:
                matching_indices.append(unmatched_gt_idx)

        if matching_indices:
            matching_dict[pred_idx] = matching_indices

    return matching_dict


def merge_matches(matches: dict[int, Any], matching_dict: dict[int, list[int]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    final_matches: dict[tuple[Any, ...], dict[str, Any]] = {}
    processed_gt_indices: set[int] = set()

    for gt_idx, match_info in matches.items():
        pred_indices = match_info["pred_indices"]
        edit_distance = match_info["edit_distance"]

        pred_key = tuple(sorted(pred_indices))

        if pred_key in final_matches:
            if gt_idx not in processed_gt_indices:
                final_matches[pred_key]["gt_indices"].append(gt_idx)
                processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                "gt_indices": [gt_idx],
                "edit_distance": edit_distance,
            }
            processed_gt_indices.add(gt_idx)

    for pred_idx, gt_indices in matching_dict.items():
        pred_key = (pred_idx,) if not isinstance(pred_idx, (list, tuple)) else tuple(sorted(pred_idx))

        if pred_key in final_matches:
            for gt_idx in gt_indices:
                if gt_idx not in processed_gt_indices:
                    final_matches[pred_key]["gt_indices"].append(gt_idx)
                    processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                "gt_indices": [gt_idx for gt_idx in gt_indices if gt_idx not in processed_gt_indices],
                "edit_distance": None,
            }
            processed_gt_indices.update(final_matches[pred_key]["gt_indices"])

    return final_matches


def recalculate_edit_distances(
    final_matches: dict[tuple[Any, ...], dict[str, Any]],
    gt_lens_dict: dict[int, int],
    norm_gt_lines: list[str],
    norm_pred_lines: list[str],
) -> None:
    _ = gt_lens_dict
    for _pred_key, info in final_matches.items():
        pred_key = _pred_key
        gt_indices = sorted(set(info["gt_indices"]))

        if not gt_indices:
            info["edit_distance"] = 1
            continue

        if len(gt_indices) > 1:
            merged_gt_content = "".join(norm_gt_lines[gt_idx] for gt_idx in gt_indices)
            pred_content = norm_pred_lines[pred_key[0]] if isinstance(pred_key[0], int) else ""

            try:
                edit_distance = levenshtein_distance(merged_gt_content, pred_content)
                normalized = edit_distance / max(len(merged_gt_content), len(pred_content), 1)
            except ZeroDivisionError:
                normalized = 1

            info["edit_distance"] = normalized
        else:
            gt_idx = gt_indices[0]
            pred_content = " ".join(
                norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int)
            )

            try:
                edit_distance = levenshtein_distance(norm_gt_lines[gt_idx], pred_content)
                normalized = edit_distance / max(len(norm_gt_lines[gt_idx]), len(pred_content), 1)
            except ZeroDivisionError:
                normalized = 1

            info["edit_distance"] = normalized
            info["pred_content"] = pred_content


def convert_final_matches(
    final_matches: dict[tuple[Any, ...], dict[str, Any]],
    norm_gt_lines: list[str],
    norm_pred_lines: list[str],
) -> list[dict[str, Any]]:
    converted_results: list[dict[str, Any]] = []

    all_gt_indices = set(range(len(norm_gt_lines)))
    all_pred_indices = set(range(len(norm_pred_lines)))

    for pred_key, info in final_matches.items():
        pred_content = " ".join(
            norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int)
        )

        for gt_idx in sorted(set(info["gt_indices"])):
            result_entry = {
                "gt_idx": int(gt_idx),
                "gt": norm_gt_lines[gt_idx],
                "pred_idx": list(pred_key),
                "pred": pred_content,
                "edit": info["edit_distance"],
            }
            converted_results.append(result_entry)

    matched_gt_indices = set().union(*[set(info["gt_indices"]) for info in final_matches.values()])
    unmatched_gt_indices = all_gt_indices - matched_gt_indices
    matched_pred_indices = {idx for pred_key in final_matches for idx in pred_key if isinstance(idx, int)}
    unmatched_pred_indices = all_pred_indices - matched_pred_indices

    if unmatched_pred_indices:
        if unmatched_gt_indices:
            distance_matrix = [
                [
                    levenshtein_distance(norm_gt_lines[gt_idx], norm_pred_lines[pred_idx])
                    / max(len(norm_gt_lines[gt_idx]), len(norm_pred_lines[pred_idx]), 1)
                    for pred_idx in unmatched_pred_indices
                ]
                for gt_idx in unmatched_gt_indices
            ]

            row_ind, col_ind = linear_sum_assignment(distance_matrix)
            um_list = sorted(unmatched_gt_indices)
            up_list = sorted(unmatched_pred_indices)

            for i, j in zip(row_ind, col_ind):
                gt_idx = um_list[i]
                pred_idx = up_list[j]
                result_entry = {
                    "gt_idx": int(gt_idx),
                    "gt": norm_gt_lines[gt_idx],
                    "pred_idx": [pred_idx],
                    "pred": norm_pred_lines[pred_idx],
                    "edit": 1,
                }
                converted_results.append(result_entry)

            matched_gt_indices.update(um_list[k] for k in row_ind)
        else:
            result_entry = {
                "gt_idx": "",
                "gt": "",
                "pred_idx": list(unmatched_pred_indices),
                "pred": " ".join(norm_pred_lines[pred_idx] for pred_idx in unmatched_pred_indices),
                "edit": 1,
            }
            converted_results.append(result_entry)
    else:
        for gt_idx in unmatched_gt_indices:
            result_entry = {
                "gt_idx": int(gt_idx),
                "gt": norm_gt_lines[gt_idx],
                "pred_idx": "",
                "pred": "",
                "edit": 1,
            }
            converted_results.append(result_entry)

    return converted_results


def run_quick_match_no_ignore(
    no_ignores_gt_lines: list[str],
    no_ignores_ori_gt_lines: list[str],
    no_ignores_pred_lines: list[str],
    no_ignores_ori_pred_lines: list[str],
) -> list[dict[str, Any]]:
    """OmniDocBench main path when no GT rows are in the ``ignore`` category set."""
    all_gt_indices = set(range(len(no_ignores_gt_lines)))
    all_pred_indices = set(range(len(no_ignores_pred_lines)))

    cost_matrix = compute_edit_distance_matrix_new(no_ignores_gt_lines, no_ignores_pred_lines)
    matched_col_idx, row_ind, cost_list = cal_final_match(
        cost_matrix, no_ignores_gt_lines, no_ignores_pred_lines
    )
    gt_lens_dict, _pred_lens_dict = initialize_indices(no_ignores_gt_lines, no_ignores_pred_lines)
    matches, unmatched_gt_indices, unmatched_pred_indices = process_matches(
        matched_col_idx,
        row_ind,
        cost_list,
        no_ignores_gt_lines,
        no_ignores_pred_lines,
        no_ignores_ori_pred_lines,
    )
    matching_dict = fuzzy_match_unmatched_items(
        unmatched_gt_indices, no_ignores_gt_lines, no_ignores_pred_lines
    )
    final_matches = merge_matches(matches, matching_dict)
    recalculate_edit_distances(final_matches, gt_lens_dict, no_ignores_gt_lines, no_ignores_pred_lines)
    converted_results = convert_final_matches(final_matches, no_ignores_gt_lines, no_ignores_pred_lines)
    merged_results = merge_duplicates_add_unmatched(
        converted_results,
        no_ignores_gt_lines,
        no_ignores_pred_lines,
        no_ignores_ori_gt_lines,
        no_ignores_ori_pred_lines,
        all_gt_indices,
        all_pred_indices,
    )
    for entry in merged_results:
        if isinstance(entry["gt_idx"], list) and entry["gt_idx"] != [""]:
            entry["gt"] = "".join(no_ignores_ori_gt_lines[g] for g in entry["gt_idx"])
        if isinstance(entry["pred_idx"], list) and entry["pred_idx"] != [""]:
            parts = []
            for pi in entry["pred_idx"]:
                if isinstance(pi, int):
                    parts.append(no_ignores_ori_pred_lines[pi])
            entry["pred"] = "".join(parts) if parts else entry.get("pred", "")

    return merged_results


def quick_match_gt_pred_lines(
    gt_raw: list[str],
    gt_norm: list[str],
    pred_raw: list[str],
    pred_norm: list[str],
    normalize: Callable[[str], str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Run OmniDocBench quick_match on parallel raw/normalized line lists.

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

    merged = run_quick_match_no_ignore(gt_norm, gt_raw, pred_norm, pred_raw)

    rows_by_gt: dict[int, dict[str, Any]] = {}
    pred_only = 0
    for entry in merged:
        gidx = entry.get("gt_idx")
        pidx = entry.get("pred_idx")
        pred_raw_m = entry.get("pred") or ""

        if gidx == [""] or gidx == "":
            pred_only += 1
            continue
        if not isinstance(gidx, list):
            gidx = [gidx]

        pred_indices = [int(x) for x in pidx if x != "" and isinstance(x, int)]
        start = min(pred_indices) if pred_indices else 0
        end = (max(pred_indices) + 1) if pred_indices else 0
        pn = normalize(pred_raw_m)

        for g in gidx:
            if not isinstance(g, int) or g < 0 or g >= n:
                continue
            gn = normalize(gt_raw[g])
            ned = normalized_edit_distance(gn, pn)
            rows_by_gt[g] = {
                "gt_index": g,
                "pred_raw": pred_raw_m,
                "pred_norm": pn,
                "ned": ned,
                "pred_segment_start": start,
                "pred_segment_end": end,
                "pred_indices": pred_indices,
            }

    if pred_only:
        notes.append(f"{pred_only} prediction-only slot(s) from quick_match merge (no GT row)")

    out = []
    for i in range(n):
        if i in rows_by_gt:
            out.append(rows_by_gt[i])
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
