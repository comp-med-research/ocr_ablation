"""HTML visualization for text matching (character-level diff + alignment table)."""

from __future__ import annotations

import html
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .matching import TaskTextEvalResult


def _span_diff_colored(a: str, b: str) -> tuple[str, str]:
    """Produce two HTML fragments with <span class=ins|del|eq> for chars."""
    sm = SequenceMatcher(None, a, b, autojunk=False)
    out_a: list[str] = []
    out_b: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        sa = html.escape(a[i1:i2])
        sb = html.escape(b[j1:j2])
        if tag == "equal":
            out_a.append(f'<span class="eq">{sa}</span>')
            out_b.append(f'<span class="eq">{sb}</span>')
        elif tag == "delete":
            out_a.append(f'<span class="del">{sa}</span>')
        elif tag == "insert":
            out_b.append(f'<span class="ins">{sb}</span>')
        else:
            out_a.append(f'<span class="del">{sa}</span>')
            out_b.append(f'<span class="ins">{sb}</span>')
    return "".join(out_a), "".join(out_b)


def render_task_html(
    task_id: Any,
    eval_result: TaskTextEvalResult,
    *,
    title: str = "Text transcription alignment",
) -> str:
    rows = []
    for m in eval_result.regions_matched:
        da, db = _span_diff_colored(m.gt_norm, m.pred_norm)
        seg_info = f"[pred §{m.pred_segment_start}:{m.pred_segment_end})"
        rows.append(
            f"""
            <tr class="region-row">
              <td class="meta">
                <div class="rid">#{m.gt_index + 1}</div>
                <div class="nid">{html.escape(m.region_id)}</div>
                <div class="seg">{html.escape(seg_info)}</div>
                <div class="score">NED {m.ned:.3f}</div>
              </td>
              <td class="raw">
                <div class="lbl">GT (raw)</div>
                <pre>{html.escape(m.gt_raw)}</pre>
                <div class="lbl">Pred (matched span, raw)</div>
                <pre>{html.escape(m.pred_raw_merged)}</pre>
              </td>
              <td class="norm">
                <div class="lbl">Normalized — GT</div>
                <div class="diff">{da}</div>
                <div class="lbl">Normalized — Pred</div>
                <div class="diff">{db}</div>
              </td>
            </tr>
            """
        )

    summary = f"""
    <div class="summary">
      <p><strong>Task</strong> {html.escape(str(task_id))}</p>
      <p>Match mode: <strong>{html.escape(str(eval_result.match_mode))}</strong> ·
      Regions: {len(eval_result.regions_matched)} · Mean NED: {eval_result.mean_ned:.4f} ·
      Micro NED: {eval_result.micro_ned:.4f} · Mean similarity: {eval_result.mean_similarity:.4f}</p>
      <p>Prediction split into <strong>{len(eval_result.pred_segments)}</strong> paragraph segment(s).</p>
      {('<p class="note">' + html.escape("; ".join(eval_result.notes)) + "</p>") if eval_result.notes else ""}
    </div>
    """

    pred_list = "".join(
        f'<li><pre>{html.escape(s[:500])}{"…" if len(s) > 500 else ""}</pre></li>'
        for s in eval_result.pred_segments
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #fafafa; color: #111; }}
    h1 {{ font-size: 1.25rem; }}
    .summary {{ background: #fff; border: 1px solid #ddd; padding: 1rem; margin-bottom: 1rem; border-radius: 8px; }}
    .note {{ color: #666; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #eee; vertical-align: top; padding: 0.75rem; }}
    .meta {{ width: 140px; background: #f6f8fa; }}
    .rid {{ font-weight: 700; font-size: 1.1rem; }}
    .nid {{ font-size: 0.7rem; color: #555; word-break: break-all; }}
    .seg {{ font-size: 0.75rem; color: #0366d6; margin-top: 0.35rem; }}
    .score {{ margin-top: 0.5rem; font-weight: 600; }}
    .lbl {{ font-size: 0.7rem; text-transform: uppercase; color: #666; margin: 0.5rem 0 0.2rem; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 0.85rem; background: #f6f8fa; padding: 0.5rem; border-radius: 4px; }}
    .diff {{ font-family: ui-monospace, monospace; font-size: 0.85rem; line-height: 1.5; padding: 0.5rem; background: #f6f8fa; border-radius: 4px; }}
    .eq {{ background: transparent; }}
    .ins {{ background: #d4fcbc; }}
    .del {{ background: #f8cbcb; text-decoration: line-through; }}
    .segments {{ background: #fff; border: 1px solid #ddd; padding: 1rem; margin-top: 1rem; border-radius: 8px; }}
    .segments ol {{ margin: 0.5rem 0 0 1.2rem; padding: 0; }}
    .segments li pre {{ background: #fff; border: 1px solid #eee; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {summary}
  <table>
    <thead>
      <tr><th>Region</th><th>Raw text</th><th>Normalized diff (scoring view)</th></tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <div class="segments">
    <h2>Prediction segments (paragraph split)</h2>
    <ol>{pred_list}</ol>
  </div>
</body>
</html>
"""


def write_eval_report(
    out_path: Path | str,
    *,
    task_id: Any,
    eval_result: TaskTextEvalResult,
    manifest_task: dict[str, Any] | None = None,
    model_name: str = "",
) -> None:
    """Write HTML + JSON sidecar with machine-readable alignment."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Alignment — task {task_id}" + (f" — {model_name}" if model_name else "")
    html_doc = render_task_html(task_id, eval_result, title=title)
    out_path.write_text(html_doc, encoding="utf-8")

    json_path = out_path.with_suffix(".json")
    payload: dict[str, Any] = {
        "task_id": task_id,
        "model_name": model_name,
        "match_mode": eval_result.match_mode,
        "mean_ned": eval_result.mean_ned,
        "micro_ned": eval_result.micro_ned,
        "mean_similarity": eval_result.mean_similarity,
        "pred_segment_count": len(eval_result.pred_segments),
        "notes": eval_result.notes,
        "matches": [
            {
                "gt_index": m.gt_index,
                "region_id": m.region_id,
                "gt_raw": m.gt_raw,
                "pred_raw_merged": m.pred_raw_merged,
                "gt_norm": m.gt_norm,
                "pred_norm": m.pred_norm,
                "ned": m.ned,
                "similarity": m.similarity,
                "pred_segment_start": m.pred_segment_start,
                "pred_segment_end": m.pred_segment_end,
            }
            for m in eval_result.regions_matched
        ],
        "pred_segments": eval_result.pred_segments,
    }
    if manifest_task:
        payload["manifest_task"] = {
            "task_id": manifest_task.get("task_id"),
            "inner_id": manifest_task.get("inner_id"),
            "region_count": len(manifest_task.get("regions") or []),
        }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
