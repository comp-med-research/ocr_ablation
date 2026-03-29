"""Fair text transcription evaluation (GT manifest + OmniDocBench-style matchers + HTML viz)."""

from .manifest import build_manifest_from_ls_export, save_manifest, load_manifest
from .matching import MatchMode, match_gt_to_prediction, TextEvalConfig
from .normalize import normalize_text

__all__ = [
    "build_manifest_from_ls_export",
    "save_manifest",
    "load_manifest",
    "match_gt_to_prediction",
    "MatchMode",
    "TextEvalConfig",
    "normalize_text",
]
