"""
Join ``gt_manifest.json`` region ``choices`` to ``task_*_alignment.json`` matches and
aggregate NED by Label Studio dimensions (``text_type``, ``region_type``, etc.).

Mean NED: unweighted average over regions in the stratum.
Micro NED within stratum: ``sum_i NED_i * w_i / sum_i w_i`` with
``w_i = max(len(gt_norm), len(pred_norm), 1)`` (same weighting as ``eval.matching._aggregate_metrics``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _norm_task_id(t: Any) -> str:
    return str(t).strip()


def _choice_scalar(choices: dict[str, Any], key: str) -> str:
    raw = choices.get(key)
    if raw is None:
        return "__missing__"
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip():
                return str(x).strip()
        return "__missing__"
    return str(raw).strip() or "__missing__"


def _micro_weights(gt_norm: str, pred_norm: str) -> float:
    return float(max(len(gt_norm or ""), len(pred_norm or ""), 1))


def discover_choice_keys(tasks: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for t in tasks:
        for r in t.get("regions") or []:
            ch = r.get("choices") or {}
            if isinstance(ch, dict):
                keys.update(ch.keys())
    return sorted(keys)


def build_task_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """task_id str -> task dict (with regions)."""
    out: dict[str, dict[str, Any]] = {}
    for t in manifest.get("tasks") or []:
        tid = t.get("task_id")
        if tid is None:
            continue
        out[_norm_task_id(tid)] = t
    return out


def load_alignment_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tid = data.get("task_id")
        if tid is None:
            continue
        matches = data.get("matches") or []
        for m in matches:
            if not isinstance(m, dict):
                continue
            rows.append(
                {
                    "task_id": _norm_task_id(tid),
                    "gt_index": int(m.get("gt_index", -1)),
                    "region_id": str(m.get("region_id", "")),
                    "ned": float(m.get("ned", 0.0)),
                    "gt_norm": str(m.get("gt_norm", "") or ""),
                    "pred_norm": str(m.get("pred_norm", "") or ""),
                    "alignment": str(data.get("alignment", "")),
                    "match_mode": str(data.get("match_mode", "")),
                    "source_file": str(p),
                }
            )
    return rows


def join_manifest(
    match_rows: list[dict[str, Any]],
    task_index: dict[str, dict[str, Any]],
    *,
    choice_keys: list[str],
    task_data_keys: list[str],
) -> tuple[list[dict[str, Any]], int]:
    """Return long rows with choice columns + count of skipped (index mismatch)."""
    long: list[dict[str, Any]] = []
    skipped = 0
    for row in match_rows:
        tid = row["task_id"]
        gi = row["gt_index"]
        task = task_index.get(tid)
        if task is None:
            skipped += 1
            base = {**row, "_join": "no_task"}
            for k in choice_keys:
                base[k] = "__no_task__"
            for dk in task_data_keys:
                base[f"task_data.{dk}"] = "__no_task__"
            long.append(base)
            continue
        regions = task.get("regions") or []
        if gi < 0 or gi >= len(regions):
            skipped += 1
            base = {**row, "_join": "bad_gt_index"}
            for k in choice_keys:
                base[k] = "__bad_index__"
            for dk in task_data_keys:
                base[f"task_data.{dk}"] = "__bad_index__"
            long.append(base)
            continue
        reg = regions[gi]
        choices = reg.get("choices") or {}
        if not isinstance(choices, dict):
            choices = {}
        base = {**row, "_join": "ok"}
        for k in choice_keys:
            base[k] = _choice_scalar(choices, k)
        data = task.get("data") or {}
        if isinstance(data, dict):
            for dk in task_data_keys:
                v = data.get(dk)
                base[f"task_data.{dk}"] = (
                    "" if v is None else str(v).strip() or "__missing__"
                )
        else:
            for dk in task_data_keys:
                base[f"task_data.{dk}"] = "__missing__"
        long.append(base)
    return long, skipped


def aggregate_stratum(neds: list[float], weights: list[float]) -> dict[str, float]:
    if not neds:
        return {"n": 0, "mean_ned": 0.0, "micro_ned": 0.0}
    n = len(neds)
    mean = sum(neds) / n
    sw = sum(weights)
    micro = sum(ned * w for ned, w in zip(neds, weights)) / sw if sw > 0 else 0.0
    return {"n": n, "mean_ned": mean, "micro_ned": micro}


def stratify_by_column(
    long_rows: list[dict[str, Any]],
    column: str,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in long_rows:
        key = str(r.get(column, "__missing__"))
        w = _micro_weights(r["gt_norm"], r["pred_norm"])
        buckets[key].append((r["ned"], w))
    out: dict[str, dict[str, float]] = {}
    for k, pairs in sorted(buckets.items(), key=lambda x: (-len(x[1]), x[0])):
        neds = [p[0] for p in pairs]
        ws = [p[1] for p in pairs]
        out[k] = aggregate_stratum(neds, ws)
    return out


def overall_metrics(long_rows: list[dict[str, Any]]) -> dict[str, float]:
    neds = [r["ned"] for r in long_rows]
    ws = [_micro_weights(r["gt_norm"], r["pred_norm"]) for r in long_rows]
    agg = aggregate_stratum(neds, ws)
    return {"n_regions": float(agg["n"]), "mean_ned": agg["mean_ned"], "micro_ned": agg["micro_ned"]}


def run_stratified(
    manifest_path: Path,
    alignment_paths: list[Path],
    *,
    choice_keys: list[str] | None,
    task_data_keys: list[str],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks") or []
    task_index = build_task_index(manifest)
    keys = choice_keys if choice_keys is not None else discover_choice_keys(tasks)
    match_rows = load_alignment_paths(alignment_paths)
    long_rows, skipped = join_manifest(match_rows, task_index, choice_keys=keys, task_data_keys=task_data_keys)

    summary = overall_metrics(long_rows)
    by_choice: dict[str, Any] = {}
    for col in keys:
        by_choice[col] = stratify_by_column(long_rows, col)
    by_task_data: dict[str, Any] = {}
    for dk in task_data_keys:
        col = f"task_data.{dk}"
        by_task_data[dk] = stratify_by_column(long_rows, col)

    return {
        "manifest": str(manifest_path),
        "n_alignment_files": len({r["source_file"] for r in match_rows}),
        "n_match_rows": len(match_rows),
        "n_long_rows": len(long_rows),
        "skipped_or_unmatched_join": skipped,
        "choice_keys": keys,
        "task_data_keys": task_data_keys,
        "summary": summary,
        "by_choice": by_choice,
        "by_task_data": by_task_data,
        "long_rows": long_rows,
    }


def _write_long_csv(path: Path, long_rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in long_rows:
            w.writerow(r)


def format_report(result: dict[str, Any], *, use_tabulate: bool) -> str:
    """Same text as printed to stdout (summary + markdown-style tables)."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None  # type: ignore[misc, assignment]

    lines: list[str] = []
    s = result["summary"]
    lines.append(
        f"Regions: {int(s['n_regions'])}  mean NED: {s['mean_ned']:.4f}  micro NED: {s['micro_ned']:.4f}"
    )
    lines.append(f"Join issues (no task / bad index): {result['skipped_or_unmatched_join']}")

    def append_table(title: str, strata: dict[str, dict[str, float]]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        rows = [
            [k, int(v["n"]), f"{v['mean_ned']:.4f}", f"{v['micro_ned']:.4f}"]
            for k, v in strata.items()
        ]
        if use_tabulate and tabulate:
            lines.append(tabulate(rows, headers=["value", "n", "mean_ned", "micro_ned"], tablefmt="github"))
        else:
            for row in rows:
                lines.append(f"  {row[0]!r}\tn={row[1]}\tmean={row[2]}\tmicro={row[3]}")

    for col, strata in result["by_choice"].items():
        append_table(f"choices.{col}", strata)
    for dk, strata in result["by_task_data"].items():
        append_table(f"task.data.{dk}", strata)

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stratified NED by manifest region choices + optional task.data fields.")
    p.add_argument("--manifest", type=Path, required=True, help="Path to gt_manifest.json")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--reports-dir",
        type=Path,
        help="Directory containing task_*_alignment.json (non-recursive unless --recursive)",
    )
    g.add_argument(
        "--alignments",
        type=Path,
        nargs="+",
        help="Explicit alignment JSON paths",
    )
    p.add_argument(
        "--glob",
        default="task_*_alignment.json",
        help="Glob under --reports-dir (default: task_*_alignment.json)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursive glob under --reports-dir",
    )
    p.add_argument(
        "--choice-keys",
        nargs="*",
        default=None,
        help="Subset of LS choice names to stratify (default: all keys seen in manifest)",
    )
    p.add_argument(
        "--task-data-keys",
        nargs="*",
        default=[],
        help="Extra columns from each task's manifest ``data`` (e.g. scanner_id)",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Write full report JSON (no long_rows if --compact-json)")
    p.add_argument("--out-csv", type=Path, default=None, help="Write one row per region-match")
    p.add_argument(
        "--out-txt",
        type=Path,
        default=None,
        help="Write summary and stratified tables (same as stdout) to a .txt file",
    )
    p.add_argument(
        "--compact-json",
        action="store_true",
        help="Omit long_rows from --out-json (smaller file)",
    )
    p.add_argument("--no-tabulate", action="store_true", help="Plain-text tables to stdout")
    args = p.parse_args(argv)

    if args.reports_dir:
        rd = args.reports_dir
        pattern = args.glob
        if args.recursive:
            paths = sorted(rd.rglob(pattern))
        else:
            paths = sorted(rd.glob(pattern))
    else:
        paths = list(args.alignments or [])

    paths = [x for x in paths if x.is_file()]
    if not paths:
        print("No alignment JSON files found.", file=sys.stderr)
        return 1

    result = run_stratified(
        args.manifest,
        paths,
        choice_keys=list(args.choice_keys) if args.choice_keys else None,
        task_data_keys=list(args.task_data_keys),
    )
    long_rows = result.pop("long_rows")

    use_tab = not args.no_tabulate
    report_text = format_report(result, use_tabulate=use_tab)
    print(report_text, end="")

    if args.out_txt:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text(report_text, encoding="utf-8")
        print(f"Wrote tables: {args.out_txt}")

    if args.out_csv:
        if long_rows:
            keys = list(long_rows[0].keys())
            _write_long_csv(args.out_csv, long_rows, keys)
            print(f"\nWrote long table: {args.out_csv}")

    if args.out_json:
        payload = dict(result)
        if not args.compact_json:
            payload["long_rows"] = long_rows
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
