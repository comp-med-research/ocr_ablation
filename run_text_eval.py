#!/usr/bin/env python3
"""
Fair **full-page** OCR evaluation against Label Studio region transcriptions.

**Inference assumption:** Each prediction file is the model’s output for the **entire
page** (the same page as the LS task image). Models are **not** evaluated on
per-box crops; GT boxes only define **what** to score after alignment.

**Pipeline:**
1. Build or load a GT manifest (regions with transcriptions from LS).
2. Split each prediction into segments: by default **markdown-aware** extraction
   (fenced code, HTML/pipe tables, display math, then prose with light MD stripping;
   see ``eval/md_segment.py``). Use ``--legacy-paragraph-segments`` for blank-line-only
   splits. Then OmniDocBench-style **quick_match** (ported from ``utils/match_quick.py``).
   Install ``Levenshtein`` for speed (see ``requirements.txt``).
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
Docling natives live under ``results/model_outputs/out_docling/native/docling/page_*/``
(relative to the pred-map directory, i.e. next to ``ocr_models/``).
``results/evaluations/eval_reports_*`` are **eval outputs** (HTML/JSON), not model markdown.

Region order for GT (until you add reading-order annotations): default is
rectangle first-seen in the export; use ``--region-order geometric`` for y-then-x.
"""

from __future__ import annotations

import argparse
import json
import re
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


# Pred-map paths are usually relative to the package root (same directory as
# ``ocr_models/``, where ``pred_map_docling.json`` typically lives).
_DOC_NATIVE_FALLBACKS = (
    "results/model_outputs/out_docling/native/docling",
    "results/out_docling/native/docling",
)


def _resolve_prediction_path(
    raw: str,
    *,
    pred_map_path: Path,
    pred_native_docling: Path | None,
    pred_filename: str = "docling.md",
) -> Path:
    """
    Resolve a path from ``pred_map``:

    1. As given (absolute), if it exists.
    2. If relative: ``<pred_map_dir>/<path>``, then ``cwd/<path>``.
    3. If the map says ``results/out_docling/...`` but only ``model_outputs`` exists,
       try the same path under ``results/model_outputs/out_docling/…`` (and reverse).
    4. If still missing, locate ``page_XXXX`` in the string and try each
       ``_DOC_NATIVE_FALLBACKS`` under pred-map parent and cwd, then
       ``pred_native_docling`` if set.
    """
    raw = (raw or "").strip()
    norm = raw.replace("\\", "/")
    p = Path(raw).expanduser()
    pkg_root = pred_map_path.parent

    def _exists(c: Path) -> bool:
        return c.is_file()

    if _exists(p):
        return p

    try_paths: list[Path] = []
    if not p.is_absolute():
        try_paths.append(pkg_root / p)
        try_paths.append(Path.cwd() / p)
        # Alternate tree: only one of model_outputs vs legacy may exist.
        rel = Path(p)
        parts = list(rel.parts)
        if parts[:2] == ("results", "out_docling"):
            alt = Path("results") / "model_outputs" / Path(*parts[1:])
            try_paths.append(pkg_root / alt)
            try_paths.append(Path.cwd() / alt)
        if len(parts) >= 3 and parts[:3] == ("results", "model_outputs", "out_docling"):
            alt2 = Path("results") / Path(*parts[2:])
            try_paths.append(pkg_root / alt2)
            try_paths.append(Path.cwd() / alt2)
    else:
        try_paths.append(p)

    for c in try_paths:
        if _exists(c):
            return c

    m = re.search(r"page_(\d+)", norm, re.I)
    if m:
        idx = int(m.group(1))
        page = f"page_{idx:04d}"
        fname = Path(norm).name if "/" in norm or "\\" in norm else pred_filename
        if not fname or fname == norm:
            fname = pred_filename
        search_roots: list[Path] = []
        if pred_native_docling is not None:
            search_roots.append(pred_native_docling)
        for sub in _DOC_NATIVE_FALLBACKS:
            search_roots.append(pkg_root / sub)
            search_roots.append(Path.cwd() / sub)
        for root in search_roots:
            if root.is_dir():
                cand = root / page / fname
                if _exists(cand):
                    return cand

    return try_paths[0] if try_paths else p


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
    p.add_argument(
        "--pred-native-docling",
        type=Path,
        help=(
            "Directory containing page_XXXX folders (e.g. results/.../native/docling). "
            "If a pred-map path is missing, retry page_XXXX/docling.md under here."
        ),
    )
    p.add_argument("--task-id", type=str, help="Restrict to one task (required with --pred)")
    p.add_argument("--out", type=Path, default=Path("eval_reports"), help="Output directory")
    p.add_argument("--model-name", type=str, default="", help="Label in HTML/JSON")
    p.add_argument("--write-manifest", type=Path, help="Write built manifest to this path and exit eval")
    # Normalization toggles
    p.add_argument("--lowercase", action="store_true", help="Lowercase before NED (usually off)")
    p.add_argument("--no-strip-md-images", action="store_true")
    p.add_argument("--no-strip-fences", action="store_true")
    p.add_argument(
        "--legacy-paragraph-segments",
        action="store_true",
        help="Split prediction on blank lines only (ignore markdown structure extraction)",
    )
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
        use_markdown_structure=not args.legacy_paragraph_segments,
    )

    pred_map: dict[str, str] | None = None
    pred_map_path: Path | None = None
    if args.pred_map:
        pred_map_path = args.pred_map.resolve()
        pred_map = _load_pred_map(pred_map_path)

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
    skipped_missing = 0
    for task in tasks:
        tid = task.get("task_id")
        if args.task_id and str(tid) != str(args.task_id):
            continue
        if pred_map is not None:
            pth = pred_map.get(str(tid))
            if not pth:
                continue
            assert pred_map_path is not None
            pred_path = _resolve_prediction_path(
                pth,
                pred_map_path=pred_map_path,
                pred_native_docling=args.pred_native_docling,
            )
        else:
            assert args.pred_dir is not None
            pred_path = _default_pred_path(args.pred_dir, tid, args.pred_suffix)
        if not pred_path.is_file():
            skipped_missing += 1
            hint = ""
            if pred_map is not None and args.pred_native_docling is None:
                hint = (
                    "  (hint: run Docling → results/model_outputs/out_docling/native/docling/page_*/docling.md "
                    "or pass --pred-native-docling PATH_TO/native/docling)"
                )
            elif pred_map is None:
                hint = "  (hint: check --pred-dir and --pred-suffix)"
            print(f"skip task {tid}: missing {pred_path}{hint}", file=sys.stderr)
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
        print(
            "No reports written (check task ids and prediction paths).",
            file=sys.stderr,
        )
        if skipped_missing and pred_map is not None:
            print(
                "Pred-map entries point at files that are not on disk — run the OCR pipeline "
                "(outputs under results/model_outputs/out_docling/…), fix paths relative to the "
                "pred-map file (same folder as ocr_models/), or use --pred-native-docling.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
