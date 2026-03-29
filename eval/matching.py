"""GT regions vs prediction segments: OmniDocBench-style matchers (ported, text-only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .md_segment import prediction_segments
from .normalize import normalize_text
from .full_match import full_match_gt_pred_lines
from .quick_match import quick_match_gt_pred_lines
from .simple_match import simple_match_gt_pred_lines

MatchMode = Literal["quick", "simple", "full"]


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
    #: If True, split prediction with ``md_segment.prediction_segments_from_markdown``
    #: (code/tables/math first, then prose). If False, use blank-line ``segment_prediction`` only.
    use_markdown_structure: bool = True
    #: ``quick`` — ``match_quick`` (truncated merge + Hungarian + fuzzy). ``simple`` — one-to-one
    #: Hungarian on NED. ``full`` — ``FuzzyMatch`` substring combine (empty pred segments dropped).
    match_mode: MatchMode = "quick"

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
    pred_segment_end: int  # exclusive index into pred_segments; start==end means no segment


@dataclass
class TaskTextEvalResult:
    task_id: Any
    regions_matched: list[RegionMatch] = field(default_factory=list)
    mean_ned: float = 0.0
    mean_similarity: float = 0.0
    micro_ned: float = 0.0  # pooled chars
    pred_segments: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    match_mode: MatchMode = "quick"
    #: ``text`` (md_segment + match_mode) or ``layout-docling`` (bbox overlap on Docling JSON).
    alignment: str = "text"


def match_gt_to_prediction(
    regions: list[dict[str, Any]],
    prediction_full_text: str,
    *,
    config: TextEvalConfig | None = None,
) -> TaskTextEvalResult:
    """
    Align each GT region using OmniDocBench-style matching (text-only), selected by
    ``TextEvalConfig.match_mode``:

    - ``quick`` — ``match_quick`` core (truncated-merge + Hungarian + fuzzy + merge).
    - ``simple`` — ``match_gt2pred_simple`` (one-to-one Hungarian on NED).
    - ``full`` — ``FuzzyMatch`` / ``match_full`` substring combine (drops empty pred segments).

    Prediction is split with ``md_segment.prediction_segments`` (markdown structure by default)
    or legacy blank-line paragraphs if ``use_markdown_structure`` is False.
    """
    cfg = config or TextEvalConfig()

    gts_raw = [r.get("transcription_gt") or "" for r in regions]
    ids = [str(r.get("region_id", "")) for r in regions]
    n = len(gts_raw)
    pred_segments = prediction_segments(
        prediction_full_text,
        use_markdown_structure=cfg.use_markdown_structure,
    )
    m = len(pred_segments)

    result = TaskTextEvalResult(
        task_id=None,
        pred_segments=pred_segments,
        match_mode=cfg.match_mode,
        alignment="text",
    )

    if n == 0:
        result.notes.append("no_gt_regions")
        return result

    gts_norm = [cfg.normalize(t) for t in gts_raw]
    preds_norm = [cfg.normalize(t) for t in pred_segments]

    if cfg.match_mode == "quick":
        rows, qnotes = quick_match_gt_pred_lines(
            gts_raw,
            gts_norm,
            pred_segments,
            preds_norm,
            cfg.normalize,
        )
    elif cfg.match_mode == "simple":
        rows, qnotes = simple_match_gt_pred_lines(
            gts_raw,
            gts_norm,
            pred_segments,
            preds_norm,
            cfg.normalize,
        )
    elif cfg.match_mode == "full":
        rows, qnotes = full_match_gt_pred_lines(
            gts_raw,
            gts_norm,
            pred_segments,
            preds_norm,
            cfg.normalize,
        )
    else:
        raise ValueError(f"unknown match_mode: {cfg.match_mode!r}")
    result.notes.extend(qnotes)

    for r in rows:
        i = int(r["gt_index"])
        gn = gts_norm[i]
        pn = r["pred_norm"]
        ned = float(r["ned"])
        result.regions_matched.append(
            RegionMatch(
                gt_index=i,
                region_id=ids[i],
                gt_raw=gts_raw[i],
                pred_raw_merged=r["pred_raw"],
                gt_norm=gn,
                pred_norm=pn,
                ned=ned,
                similarity=1.0 - ned,
                pred_segment_start=int(r["pred_segment_start"]),
                pred_segment_end=int(r["pred_segment_end"]),
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
