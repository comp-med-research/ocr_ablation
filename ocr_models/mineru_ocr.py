"""MinerU (OpenDataLab) — layout-aware Markdown/JSON via ``mineru`` CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .base import BaseOCR


class MinerUOCR(BaseOCR):
    """
    Wraps the ``mineru`` CLI (``pip install mineru``).

    Environment (optional):

    - ``MINERU_METHOD``: ``auto`` | ``txt`` | ``ocr`` (default ``ocr`` for scans/images).
    - ``MINERU_BACKEND``: e.g. ``pipeline``, ``hybrid-auto-engine`` (omit for CLI default).
    - ``MINERU_LANG``: e.g. ``en``, ``ch`` (default ``en``).
    - ``MINERU_API_URL``: if set, passed as ``--api-url`` (remote ``mineru-api``).
    """

    name = "MinerU"

    def is_markdown_primary(self) -> bool:
        return True

    def __init__(self, timeout_seconds: int = 900):
        super().__init__()
        self.timeout_seconds = timeout_seconds

    def load_model(self) -> None:
        if not shutil.which("mineru"):
            raise RuntimeError(
                "MinerU CLI not found. Install with: pip install mineru "
                "and ensure `mineru` is on PATH. See https://opendatalab.github.io/MinerU/"
            )
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        method = (os.environ.get("MINERU_METHOD") or "ocr").strip().lower()
        backend = (os.environ.get("MINERU_BACKEND") or "").strip()
        lang = (os.environ.get("MINERU_LANG") or "en").strip()
        api_url = (os.environ.get("MINERU_API_URL") or "").strip()

        work = Path(tempfile.mkdtemp(prefix="mineru_ocr_"))
        try:
            in_path = work / "page.png"
            image.save(in_path, "PNG")
            out_root = work / "out"
            out_root.mkdir(parents=True)

            cmd = [
                "mineru",
                "-p",
                str(in_path),
                "-o",
                str(out_root),
                "-m",
                method,
            ]
            if backend:
                cmd.extend(["-b", backend])
            if lang:
                cmd.extend(["-l", lang])
            if api_url:
                cmd.extend(["--api-url", api_url])

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(work),
            )
            err_blob = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"mineru failed (code {proc.returncode}): {err_blob[-2500:]}"
                )

            md_files = sorted(out_root.rglob("*.md"))
            text = ""
            for m in md_files:
                t = m.read_text(encoding="utf-8", errors="replace").strip()
                if len(t) > len(text):
                    text = t

            out_files = [f for f in out_root.rglob("*") if f.is_file()]
            if not md_files and not out_files:
                raise RuntimeError(
                    "mineru wrote no output files (MinerU sometimes exits 0 on import errors). "
                    "Install full deps from https://opendatalab.github.io/MinerU/ — e.g. "
                    "`pip install doclayout-yolo ultralytics`. Log tail:\n"
                    f"{err_blob[-4000:]}"
                )
            if not md_files and out_files and (
                "ModuleNotFoundError" in err_blob or "No module named" in err_blob
            ):
                raise RuntimeError(
                    "mineru missing Python dependencies (no .md produced). Log tail:\n"
                    f"{err_blob[-4000:]}"
                )

            nd = self.native_page_dir()
            if nd is not None:
                mu_root = nd / "mineru"
                mu_root.mkdir(parents=True, exist_ok=True)
                for f in out_root.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(out_root)
                        dest = mu_root / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dest)
                json_files = [str(p.relative_to(out_root)) for p in out_root.rglob("*.json")]
                (nd / "mineru_meta.json").write_text(
                    json.dumps(
                        {
                            "method": method,
                            "backend": backend or None,
                            "lang": lang,
                            "cli_returncode": proc.returncode,
                            "markdown_files_found": len(md_files),
                            "json_files": json_files[:200],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            return text
        finally:
            shutil.rmtree(work, ignore_errors=True)
