#!/usr/bin/env python3
"""CLI entry: stratified NED by manifest choices (delegates to ``stratified_ned``)."""

from __future__ import annotations

import sys
from pathlib import Path

# Project root (parent of ``eval/``) so ``import eval.…`` works when run as
# ``python eval/stratify_ned.py`` from ``ocr_ablation``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.stratified_ned import main

if __name__ == "__main__":
    raise SystemExit(main())
