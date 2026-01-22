"""Marker OCR implementation - PDF/image to Markdown conversion."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .base import BaseOCR


class MarkerOCR(BaseOCR):
    """Marker - Fast, accurate PDF/image to Markdown conversion (Datalab)."""

    name = "Marker"

    def __init__(self):
        """
        Initialize Marker OCR.
        
        Marker converts PDFs and images to Markdown with table formatting,
        equation LaTeX, and code blocks. Supports GPU, CPU, or MPS.
        """
        super().__init__()
        self._temp_dir = None

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return False  # Marker may use GPU if available; we don't detect here

    def load_model(self) -> None:
        """Verify Marker is available (CLI-based)."""
        try:
            subprocess.run(
                ["marker_single", "--help"],
                capture_output=True,
                check=False,
                timeout=5,
            )
        except FileNotFoundError:
            pass
        try:
            subprocess.run(
                ["marker", "--help"],
                capture_output=True,
                check=False,
                timeout=5,
            )
        except FileNotFoundError:
            pass
        self._temp_dir = tempfile.mkdtemp(prefix="marker_ocr_")
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run Marker on an image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        path = Path(self._temp_dir) / "input.png"
        image.save(path, "PNG")
        out_dir = Path(self._temp_dir) / "out"
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)

        attempts = [
            ["marker_single", str(path), "--output_dir", str(out_dir)],
            ["marker_single", str(path), str(out_dir)],
            ["marker", str(path), "--output_dir", str(out_dir)],
            ["marker", str(path), str(out_dir)],
        ]
        for argv in attempts:
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(self._temp_dir),
                )
                if proc.returncode != 0:
                    continue
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

            md_files = list(out_dir.glob("**/*.md"))
            for m in md_files:
                text = m.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return text
            return ""

        raise RuntimeError(
            "Marker CLI not found. Install with: pip install marker-pdf. "
            "Ensure 'marker_single' or 'marker' is on PATH."
        )

    def __del__(self):
        """Clean up temp directory on deletion."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
