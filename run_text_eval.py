#!/usr/bin/env python3
"""
Fair **full-page** OCR evaluation against Label Studio region transcriptions.

**Inference assumption:** Each prediction file is the model’s output for the **entire
page** (the same page as the LS task image). Models are **not** evaluated on
per-box crops; GT boxes only define **what** to score after alignment.

**Pipeline:**
1. Build or load a GT manifest (regions with transcriptions from LS).
2. Match **one full-page prediction** per task to those regions using shared
   normalization + ordered DP over prediction **paragraphs** (merge adjacent
   segments so paragraph splits do not unfairly dominate scores).
3. Score **NED** (normalized edit distance) on normalized text; write HTML + JSON
   so you can audit alignments.

**Native formats:** This entrypoint consumes a single **text** stream per task
(typically plain text or markdown pasted into one file). Models that also emit
JSON/MD on disk can keep those for future table/structure metrics; for NED,
point ``--pred`` / ``--pred-map`` at the full-page text (or MD-as-text) you want
compared.

Examples::

    python run_text_eval.py \\
      --ls-export project-12-….json \\
      --write-manifest gt_manifest.json

    python run_text_eval.py --manifest gt_manifest.json \\
      --pred-map preds.json --out eval_reports

``preds.json`` maps LS ``task_id`` → path to that task’s **full-page** output.

Region order for GT (until you add reading-order annotations): default is
rectangle first-seen in the export; use ``--region-order geometric`` for y-then-x.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.manifest import build_manifest_from_ls_export, load_manifest, save_manifest
from eval.matching import TextEvalConfig, match_gt_to_prediction
from eval.visualize import write_eval_report


def _load_pred_map(path: Path) -> dict[str, str]:
    m = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in m.items()}


def _default_pred_path(pred_dir: Path, task_id: int | str, suffix: str) -> Path:
    return pred_dir / f"{task_id}{suffix}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Full-page OCR vs Label Studio region GT: NED + alignment HTML. "
            "Each prediction file must be one whole page, not per-crop."
        ),
    )
    p.add_argument("--ls-export", type=Path, help="Label Studio JSON export (builds manifest)")
    p.add_argument("--manifest", type=Path, help="Use existing gt_manifest.json instead of --ls-export")
    p.add_argument(
        "--region-order",
        choices=("rectangle_first_seen", "geometric"),
        default="rectangle_first_seen",
        help="How to order GT regions (reading-order annotation can replace this later)",
    )
    p.add_argument("--pred", type=Path, help="Single prediction text file (one task via --task-id)")
    p.add_argument("--pred-dir", type=Path, help="Directory of per-task prediction .txt files")
    p.add_argument("--pred-suffix", type=str, default="_output.txt", help="Filename = task_id + suffix")
    p.add_argument("--pred-map", type=Path, help="JSON map task_id -> file path")
    p.add_argument("--task-id", type=str, help="Restrict to one task (required with --pred)")
    p.add_argument("--out", type=Path, default=Path("eval_reports"), help="Output directory")
    p.add_argument("--model-name", type=str, default="", help="Label in HTML/JSON")
    p.add_argument("--write-manifest", type=Path, help="Write built manifest to this path and exit eval")
    # Normalization toggles
    p.add_argument("--lowercase", action="store_true", help="Lowercase before NED (usually off)")
    p.add_argument("--no-strip-md-images", action="store_true")
    p.add_argument("--no-strip-fences", action="store_true")
    args = p.parse_args()

    if not args.ls_export and args.manifest is None:
        p.error("Provide --ls-export and/or --manifest")

    manifest = None
    if args.ls_export:
        manifest = build_manifest_from_ls_export(
            args.ls_export,
            region_order=args.region_order,
        )
        if args.write_manifest:
            save_manifest(manifest, args.write_manifest)
            print(f"Wrote manifest ({len(manifest['tasks'])} tasks) -> {args.write_manifest}")
    if args.manifest is not None and manifest is None:
        manifest = load_manifest(args.manifest)

    if manifest is None:
        p.error("No manifest loaded")

    has_eval = bool(args.pred or args.pred_dir or args.pred_map)
    if args.write_manifest and not has_eval:
        return 0

    cfg = TextEvalConfig(
        lowercase=args.lowercase,
        strip_md_images=not args.no_strip_md_images,
        strip_fences=not args.no_strip_fences,
    )

    pred_map: dict[str, str] | None = None
    if args.pred_map:
        pred_map = _load_pred_map(args.pred_map)

    tasks = manifest.get("tasks") or []
    args.out.mkdir(parents=True, exist_ok=True)

    if args.pred:
        if not args.task_id:
            p.error("--pred requires --task-id")
        pred_text = args.pred.read_text(encoding="utf-8", errors="replace")
        task = next((t for t in tasks if str(t.get("task_id")) == str(args.task_id)), None)
        if not task:
            print(f"Task id {args.task_id} not in manifest", file=sys.stderr)
            return 1
        er = match_gt_to_prediction(task["regions"], pred_text, config=cfg)
        er.task_id = task.get("task_id")
        out_html = args.out / f"task_{args.task_id}_alignment.html"
        write_eval_report(
            out_html,
            task_id=task.get("task_id"),
            eval_result=er,
            manifest_task=task,
            model_name=args.model_name or "single_pred",
        )
        print(f"Wrote {out_html} and {out_html.with_suffix('.json')}")
        return 0

    if not args.pred_dir and not pred_map:
        p.error("Provide --pred, or --pred-dir / --pred-map for batch mode")

    done = 0
    for task in tasks:
        tid = task.get("task_id")
        if args.task_id and str(tid) != str(args.task_id):
            continue
        if pred_map is not None:
            pth = pred_map.get(str(tid))
            if not pth:
                continue
            pred_path = Path(pth)
        else:
            assert args.pred_dir is not None
            pred_path = _default_pred_path(args.pred_dir, tid, args.pred_suffix)
        if not pred_path.is_file():
            print(f"skip task {tid}: missing {pred_path}", file=sys.stderr)
            continue
        pred_text = pred_path.read_text(encoding="utf-8", errors="replace")
        # Strip common page-break markers from ablation runner
        pred_text = pred_text.replace("\n\n--- Page Break ---\n\n", "\n\n")
        er = match_gt_to_prediction(task["regions"], pred_text, config=cfg)
        er.task_id = tid
        safe = str(tid).replace("/", "_")
        out_html = args.out / f"task_{safe}_alignment.html"
        write_eval_report(
            out_html,
            task_id=tid,
            eval_result=er,
            manifest_task=task,
            model_name=args.model_name,
        )
        done += 1
        print(f"task {tid} -> {out_html.name} (mean NED {er.mean_ned:.4f})")

    if done == 0:
        print("No reports written (check task ids and prediction paths).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
