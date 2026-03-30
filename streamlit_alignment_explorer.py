#!/usr/bin/env python3
"""
Step-by-step alignment explorer: markdown (text+quick_match) vs Docling JSON (layout).

Run from repo root::

    cd ocr_ablation
    pip install -r requirements-demo.txt
    streamlit run streamlit_alignment_explorer.py

If you see **inotify watch limit reached**, either run from this directory (uses
``.streamlit/config.toml`` with ``fileWatcherType = "none"``) or::

    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none streamlit run streamlit_alignment_explorer.py

Defaults use ``gt_manifest.json``, Docling paths from ``pred_map_docling*.json``, and page
rasters from ``test_cases15`` (Label Studio ``uuid-NNNNN.jpg`` → ``NNNNN.jpg``).
"""

from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from eval.docling_layout import (
    iter_docling_text_spans,
    load_docling_json,
    match_gt_to_docling_json_path,
    pick_docling_document,
    resolve_docling_list_index,
)
from eval.edit_distance import levenshtein_distance, normalized_edit_distance
from eval.layout_geometry import BoxNorm, box_from_bbox_pct, iou
from eval.manifest import load_manifest
from eval.matching import TextEvalConfig, match_gt_to_prediction
from eval.md_segment import prediction_segments
from eval.quick_match import compute_edit_distance_matrix_new, quick_match_gt_pred_lines


def _load_json_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def _resolve(p: str, base: Path) -> Path:
    q = Path(p).expanduser()
    if q.is_file():
        return q
    c = base / p
    if c.is_file():
        return c
    d = Path.cwd() / p
    if d.is_file():
        return d
    return c


def _resolve_manifest_page_image(manifest_image: str, base: Path, images_dir: Path | None) -> Path | None:
    """
    Map Label Studio ``data.image`` (e.g. ``/data/upload/12/be33ebd5-00000725.jpg``) to a local file.

    Tries: absolute path, ``base``-relative, cwd, then ``images_dir / basename`` and
    ``images_dir / <after first '-' in basename>`` (e.g. ``00000725.jpg`` in ``test_cases15``).
    """
    p = str(manifest_image).strip()
    if not p:
        return None
    trials: list[Path] = []
    q = Path(p).expanduser()
    trials.append(q)
    trials.append(base / p.lstrip("/"))
    trials.append(Path.cwd() / p)
    if images_dir is not None:
        idir = images_dir.expanduser()
        if idir.is_dir():
            name = Path(p).name
            trials.append(idir / name)
            if "-" in name:
                trials.append(idir / name.split("-", 1)[1])
    for t in trials:
        if t.is_file():
            return t
    return None


def _ned_breakdown(gt_norm: str, pred_norm: str) -> tuple[int, int, float]:
    d = levenshtein_distance(gt_norm, pred_norm)
    denom = max(len(gt_norm), len(pred_norm), 1)
    ned = d / denom
    return d, denom, ned


def _diff_html(a: str, b: str) -> str:
    sm = SequenceMatcher(None, a, b, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        sa = a[i1:i2].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        sb = b[j1:j2].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if tag == "equal":
            parts.append(f'<span style="background:#e6ffe6">{sa}</span>')
        elif tag == "delete":
            parts.append(f'<span style="background:#ffd6d6;text-decoration:line-through">{sa}</span>')
        elif tag == "insert":
            parts.append(f'<span style="background:#d6e8ff">{sb}</span>')
        else:
            parts.append(f'<span style="background:#ffd6d6;text-decoration:line-through">{sa}</span>')
            parts.append(f'<span style="background:#d6e8ff">{sb}</span>')
    return "".join(parts)


def _load_page_image_array(path: Path) -> np.ndarray | None:
    """RGB uint8 array (H, W, 3), or None if missing/unreadable."""
    if not path.is_file():
        return None
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        return np.asarray(im)
    except Exception:
        try:
            arr = plt.imread(str(path))
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.dtype == np.floating else arr.astype(np.uint8)
        except Exception:
            return None


def _fig_boxes_norm(
    gt_boxes: list[tuple[BoxNorm, str, int]],
    pred_boxes: list[tuple[BoxNorm, str, int]],
    highlight_gt: int | None,
    matched_pred_idx: set[int],
    *,
    background_image_path: Path | None = None,
) -> plt.Figure:
    img = _load_page_image_array(background_image_path) if background_image_path else None
    if img is not None:
        ih, iw = int(img.shape[0]), int(img.shape[1])
        fig_w, fig_h = 8.0, max(3.0, 8.0 * ih / max(iw, 1))
    else:
        fig_w, fig_h = 8.0, 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    if img is not None:
        ax.imshow(img, extent=(0, 1, 1, 0), origin="upper", zorder=0)
        ax.set_aspect(iw / max(ih, 1))
    else:
        ax.set_aspect("equal")
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized, top=0)")
    title = "Boxes: green = GT regions, blue = Docling spans (bold = matched to selected GT)"
    if img is not None:
        title = "Page image + " + title
    ax.set_title(title)
    for box, _label, idx in gt_boxes:
        ec = "#228B22" if highlight_gt is None or idx == highlight_gt else "#90EE90"
        lw = 2.5 if highlight_gt is not None and idx == highlight_gt else 1.0
        w, h = box.x1 - box.x0, box.y1 - box.y0
        ax.add_patch(
            mpatches.Rectangle(
                (box.x0, box.y0),
                w,
                h,
                fill=False,
                edgecolor=ec,
                linewidth=lw,
                zorder=2,
            )
        )
    for box, _t, idx in pred_boxes:
        ec = "#1E90FF" if idx in matched_pred_idx else "#87CEEB"
        lw = 2.0 if idx in matched_pred_idx else 0.8
        w, h = box.x1 - box.x0, box.y1 - box.y0
        ax.add_patch(
            mpatches.Rectangle(
                (box.x0, box.y0),
                w,
                h,
                fill=False,
                edgecolor=ec,
                linewidth=lw,
                linestyle="-" if idx in matched_pred_idx else ":",
                zorder=2,
            )
        )
    ax.legend(
        handles=[
            mpatches.Patch(edgecolor="#228B22", facecolor="none", label="GT"),
            mpatches.Patch(edgecolor="#1E90FF", facecolor="none", label="Docling span"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Alignment explorer", layout="wide")
    st.title("Alignment explorer — markdown vs Docling layout")
    st.caption(
        "Walks through real **gt_manifest.json** regions vs predictions: "
        "**text** path (md segments + quick_match) and **layout** path (bbox overlap + NED)."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        manifest_path = st.text_input("Manifest path", value=str(ROOT / "gt_manifest.json"))
    with col2:
        base_dir = st.text_input("Resolve relative paths from", value=str(ROOT))
    with col3:
        page_images_dir_in = st.text_input(
            "Page images directory",
            value=str(ROOT / "test_cases15"),
            help="Manifest paths like …/uuid-00000725.jpg are matched to 00000725.jpg here.",
        )

    base = Path(base_dir)
    page_images_dir = Path(page_images_dir_in).expanduser() if page_images_dir_in.strip() else None
    mp = Path(manifest_path)
    if not mp.is_file():
        st.error(f"Manifest not found: {mp}")
        st.stop()

    manifest = load_manifest(mp)
    tasks = manifest.get("tasks") or []
    task_ids = [str(t.get("task_id")) for t in tasks if t.get("task_id") is not None]
    tid_s = st.selectbox("Task ID", task_ids, index=task_ids.index("1933") if "1933" in task_ids else 0)
    task = next(t for t in tasks if str(t.get("task_id")) == tid_s)
    regions = task.get("regions") or []
    gts_raw = [r.get("transcription_gt") or "" for r in regions]

    map_md = _load_json_map(ROOT / "pred_map_docling.json")
    map_js = _load_json_map(ROOT / "pred_map_docling_layout.json")
    default_md = map_md.get(tid_s, "")
    default_js = map_js.get(tid_s, default_md.replace(".md", "_document.json").replace("docling.md", "docling_document.json"))

    st.subheader("Prediction files (this task)")
    c1, c2 = st.columns(2)
    with c1:
        md_path_in = st.text_input("Markdown / full-page text (.md)", value=default_md)
    with c2:
        json_path_in = st.text_input("Docling layout JSON", value=default_js)

    md_path = _resolve(md_path_in, base)
    js_path = _resolve(json_path_in, base)

    cfg = TextEvalConfig()
    inner_id = int(task.get("inner_id") or 1)

    tab_md, tab_lo = st.tabs(["1 · Markdown / text alignment", "2 · JSON layout alignment"])

    # ----- Markdown tab -----
    with tab_md:
        st.markdown("### Step A — Load full-page prediction")
        if not md_path.is_file():
            st.warning(f"Missing file: {md_path}")
        else:
            pred_text = md_path.read_text(encoding="utf-8", errors="replace")
            st.metric("Characters", len(pred_text))
            with st.expander("Raw file preview (first 4000 chars)"):
                st.code(pred_text[:4000], language="markdown")

            st.markdown("### Step B — Split into segments (`md_segment.prediction_segments`)")
            segs = prediction_segments(pred_text, use_markdown_structure=cfg.use_markdown_structure)
            st.metric("Segment count", len(segs))
            df_seg = {
                "idx": list(range(len(segs))),
                "chars": [len(s) for s in segs],
                "preview": [s[:120].replace("\n", " ↵ ") + ("…" if len(s) > 120 else "") for s in segs],
            }
            st.dataframe(df_seg, width="stretch", height=min(400, 40 + 28 * len(segs)))

            st.markdown("### Step C — Ground-truth regions (from manifest)")
            st.dataframe(
                {
                    "gt_index": list(range(len(regions))),
                    "region_id": [str(r.get("region_id", "")) for r in regions],
                    "transcription_gt": gts_raw,
                },
                width="stretch",
                height=min(400, 40 + 28 * len(regions)),
            )

            st.markdown("### Step D — Match each GT region to prediction segments (`quick_match`)")
            st.markdown(
                """
**What this step is for.** You have one ordered list of **GT regions** (Step C) and one ordered list of **markdown segments** (Step B). They are usually *not* aligned one-to-one: one region might need several segments concatenated, or boundaries differ. Step D runs the **OmniDocBench-style quick matcher** (`eval/quick_match.py`) to decide, for *each* GT row, *which* segment indices to use and how to **merge** their text before scoring.

**The heatmap below (when shown)** is the **raw** pairwise **NED** grid: cell *(i, j)* = normalized edit distance between normalized GT *i* and normalized segment *j*. With ``viridis_r`` (0 = best match), **lighter / yellow** ≈ **NED near 0** (more similar); **darker / purple** ≈ **NED near 1** (less similar). This matrix is only the *ingredient* cost data — the matcher then applies **global assignment** (Hungarian / `linear_sum_assignment` on a **transformed** matrix that allows merging or splitting prediction lines), **fuzzy** handling for leftovers, and **merging** of adjacent segments when that improves the match. Details: `run_quick_match_no_ignore` → `cal_final_match`, `process_matches`, `fuzzy_match_unmatched_items`, etc.

**Step E** shows the **outcome**: for every GT index, the chosen **segment index range** `[start:end)`, the **merged raw prediction** string, and **NED** between that GT and the merged pred.
                """.strip()
            )
            gts_norm = [cfg.normalize(t) for t in gts_raw]
            preds_norm = [cfg.normalize(t) for t in segs]
            if len(segs) and len(regions):
                cm = compute_edit_distance_matrix_new(gts_norm, preds_norm)
                st.caption(
                    "Heatmap: **C[i,j]** = NED(norm_gt_i, norm_seg_j). The table in Step E comes from **`quick_match_gt_pred_lines`**, not from taking argmin per row of this grid."
                )
                fig_cm, ax_cm = plt.subplots(figsize=(max(6, len(segs) * 0.35), max(4, len(regions) * 0.35)))
                im = ax_cm.imshow(cm, cmap="viridis_r", aspect="auto", vmin=0, vmax=1)
                ax_cm.set_xlabel("pred segment j")
                ax_cm.set_ylabel("GT region i")
                ax_cm.set_xticks(range(len(segs)))
                ax_cm.set_yticks(range(len(regions)))
                plt.colorbar(im, ax=ax_cm, label="NED")
                fig_cm.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            rows, qnotes = quick_match_gt_pred_lines(
                gts_raw, gts_norm, segs, preds_norm, cfg.normalize
            )
            for n in qnotes:
                st.info(n)
            assign = {int(r["gt_index"]): r for r in rows}
            st.markdown("### Step E — Result of quick_match: merged pred per GT region")
            st.dataframe(
                {
                    "gt_index": [assign[i]["gt_index"] for i in sorted(assign.keys())],
                    "pred_segment_range": [
                        f'[{assign[i]["pred_segment_start"]}:{assign[i]["pred_segment_end"]})'
                        for i in sorted(assign.keys())
                    ],
                    "pred_raw_merged": [assign[i]["pred_raw"][:200] for i in sorted(assign.keys())],
                    "NED": [round(float(assign[i]["ned"]), 4) for i in sorted(assign.keys())],
                },
                width="stretch",
                height=min(500, 40 + 28 * len(assign)),
            )

            er_md = match_gt_to_prediction(regions, pred_text, config=cfg)
            st.metric("Mean NED (full pipeline)", f"{er_md.mean_ned:.4f}")
            st.metric("Micro NED", f"{er_md.micro_ned:.4f}")

            st.markdown("### Step F — NED for one region (normalized strings)")
            ri = st.selectbox("Pick GT index (markdown path)", list(range(len(regions))), key="md_r")
            m = er_md.regions_matched[ri]
            gn, pn = m.gt_norm, m.pred_norm
            d, denom, ned = _ned_breakdown(gn, pn)
            st.latex(
                r"\mathrm{NED} = \frac{d_{\mathrm{Lev}}(\mathrm{gt}, \mathrm{pred})}{\max(|\mathrm{gt}|, |\mathrm{pred}|, 1)}"
                + rf" = \frac{{{d}}}{{{denom}}} = {ned:.4f}"
            )
            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown("**GT (normalized)**")
                st.code(gn or "∅")
            with c_b:
                st.markdown("**Pred (normalized)**")
                st.code(pn or "∅")
            st.markdown("**Character diff** (green=match, red=del, blue=ins)")
            st.markdown(
                f'<div style="font-family:ui-monospace,monospace;font-size:14px;line-height:1.6">{_diff_html(gn, pn)}</div>',
                unsafe_allow_html=True,
            )

    # ----- Layout tab -----
    with tab_lo:
        task_data = task.get("data") or {}
        img_key = task_data.get("image")
        page_img_path = (
            _resolve_manifest_page_image(str(img_key), base, page_images_dir)
            if img_key
            else None
        )

        st.markdown("### Step A — Load Docling JSON → text spans + boxes")
        if not js_path.is_file():
            st.warning(f"Missing file: {js_path}")
        else:
            data = load_docling_json(js_path)
            page_no = st.number_input("Docling page_no", min_value=1, value=1, step=1)
            min_iou = st.slider("layout min IoU", 0.0, 0.5, 0.05, 0.01)

            if isinstance(data, list):
                li, res_notes = resolve_docling_list_index(js_path, doc_index=None, inner_id=inner_id)
                doc, pick_extra = pick_docling_document(data, list_index=li)
                for x in res_notes + pick_extra:
                    st.caption(x)
            else:
                doc = data

            pw, ph = 0.0, 0.0
            try:
                pages = doc.get("pages") or {}
                pinfo = pages.get(str(page_no)) or next(iter(pages.values()), {})
                sz = pinfo.get("size") or {}
                pw, ph = float(sz.get("width", 0)), float(sz.get("height", 0))
            except (StopIteration, TypeError, ValueError):
                pass
            st.metric("Docling page size (px)", f"{pw:.0f} × {ph:.0f}")
            spans = iter_docling_text_spans(doc, page_no=int(page_no))
            st.metric("Text spans (with bbox)", len(spans))

            span_rows = []
            for j, sp in enumerate(spans):
                span_rows.append(
                    {
                        "j": j,
                        "text": sp.text[:80] + ("…" if len(sp.text) > 80 else ""),
                        "x0": round(sp.box.x0, 4),
                        "y0": round(sp.box.y0, 4),
                        "x1": round(sp.box.x1, 4),
                        "y1": round(sp.box.y1, 4),
                    }
                )
            st.dataframe(span_rows, width="stretch", height=min(400, 40 + 24 * len(span_rows)))

            st.markdown("### Step B — GT boxes (manifest `bbox_pct` → normalized top-left)")
            if img_key:
                if page_img_path is not None and page_img_path.is_file():
                    st.caption(
                        f"Box overlay (Step C) draws on the manifest page image: `{page_img_path}` "
                        "(same coordinate space as `bbox_pct`)."
                    )
                else:
                    st.caption(
                        f"Manifest has `data.image` = `{img_key}` but no matching file was found "
                        f"(check **Page images directory**: `{page_images_dir or '—'}`; "
                        f"also tries basename and the part after the first `-` in the filename). "
                        "GT/Docling rectangles still plot on a blank canvas."
                    )
            else:
                st.caption(
                    "No `data.image` on this task — GT vs Docling boxes are drawn on a blank normalized square."
                )
            gt_boxes: list[tuple[BoxNorm, str, int]] = []
            for i, r in enumerate(regions):
                bp = r.get("bbox_pct")
                b = box_from_bbox_pct(bp) if isinstance(bp, dict) else None
                if b:
                    gt_boxes.append((b, str(r.get("region_id", "")), i))
            st.metric("GT regions with bbox", len(gt_boxes))

            st.markdown("### Step C — Overlap rule per GT")
            st.caption("Assign span j if IoU(gt, span_j) ≥ threshold OR span center lies inside GT.")

            ri2 = st.selectbox("Pick GT index (layout path)", list(range(len(regions))), key="lo_r")
            r = regions[ri2]
            bp = r.get("bbox_pct")
            gt_box = box_from_bbox_pct(bp) if isinstance(bp, dict) else None
            if gt_box is None:
                st.warning("No bbox_pct for this region.")
            else:
                overlap_rows = []
                matched: set[int] = set()
                for j, sp in enumerate(spans):
                    iv = iou(gt_box, sp.box)
                    cx, cy = sp.box.center()
                    cin = gt_box.x0 <= cx <= gt_box.x1 and gt_box.y0 <= cy <= gt_box.y1
                    hit = iv >= min_iou or cin
                    if hit:
                        matched.add(j)
                    overlap_rows.append(
                        {
                            "j": j,
                            "IoU": round(iv, 4),
                            "center_in_GT": cin,
                            "assigned": hit,
                            "text": sp.text[:60],
                        }
                    )
                st.dataframe(
                    [x for x in overlap_rows if x["assigned"]],
                    width="stretch",
                    height=min(300, 40 + 24 * len(matched)),
                )

                idxs = [j for j, x in enumerate(overlap_rows) if x["assigned"]]
                idxs.sort(key=lambda k: (spans[k].box.y0, spans[k].box.x0))
                merged = " ".join(spans[k].text for k in idxs)
                gt_raw = gts_raw[ri2] if ri2 < len(gts_raw) else ""
                gn = cfg.normalize(gt_raw)
                pn = cfg.normalize(merged)
                d, denom, ned = _ned_breakdown(gn, pn)
                st.markdown("**Merged pred (reading order)**")
                st.code(merged or "∅")
                st.latex(
                    rf"\mathrm{{NED}}=\frac{{{d}}}{{{denom}}}={ned:.4f}"
                )

                pred_box_list = [(sp.box, sp.text, j) for j, sp in enumerate(spans)]
                fig = _fig_boxes_norm(
                    gt_boxes,
                    pred_box_list,
                    highlight_gt=ri2,
                    matched_pred_idx=matched,
                    background_image_path=page_img_path if img_key else None,
                )
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("### Step D — Full layout metrics (`match_gt_to_docling_json_path`)")
            try:
                er_lo = match_gt_to_docling_json_path(
                    regions,
                    js_path,
                    config=cfg,
                    page_no=int(page_no),
                    doc_index=None,
                    inner_id=inner_id,
                    min_iou=float(min_iou),
                )
                st.metric("Mean NED (layout pipeline)", f"{er_lo.mean_ned:.4f}")
                st.metric("Micro NED", f"{er_lo.micro_ned:.4f}")
                for n in er_lo.notes:
                    st.caption(n)
            except Exception as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
