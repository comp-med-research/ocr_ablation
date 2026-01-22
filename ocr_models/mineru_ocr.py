"""MinerU OCR implementation - Magic-PDF document understanding."""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import pymupdf as fitz
from PIL import Image

from .base import BaseOCR


class MinerUOCR(BaseOCR):
    """MinerU - Magic-PDF document understanding (OpenDataLab)."""

    name = "MinerU"
    supports_native_pdf = True

    def __init__(self, lang: str = "en"):
        """
        Initialize MinerU OCR.

        MinerU converts PDFs to Markdown with structure preservation,
        table/formula extraction, and multilingual OCR.
        """
        super().__init__()
        self.lang = lang
        self._temp_dir = None

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return False  # MinerU may use GPU if available; we don't detect here

    def load_model(self) -> None:
        """Verify magic-pdf CLI is available."""
        try:
            subprocess.run(
                ["magic-pdf", "--help"],
                capture_output=True,
                check=False,
                timeout=10,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "magic-pdf not found. Install with: pip install magic-pdf "
                "or pip install magic-pdf[full-cpu]"
            ) from None
        self._temp_dir = tempfile.mkdtemp(prefix="mineru_ocr_")
        self._is_loaded = True

    def run_ocr_on_pdf(
        self, pdf_path: Path, pages: Optional[list[int]] = None
    ) -> tuple[list[str], list[float]]:
        """
        Run MinerU on a PDF via magic-pdf CLI.

        Processes the full PDF; pagination is not supported by magic-pdf.
        Returns ( [full_markdown], [total_time] ) for compatibility.
        """
        if not self._is_loaded:
            self.load_model()

        out_dir = Path(self._temp_dir) / "out"
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)

        cmd = [
            "magic-pdf",
            "-p", str(pdf_path),
            "-o", str(out_dir),
            "-m", "auto",
            "-l", self.lang,
        ]
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(self._temp_dir),
        )
        elapsed = time.perf_counter() - start

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() or "Unknown error"
            raise RuntimeError(f"magic-pdf failed: {err}")

        md_files = list(out_dir.glob("**/*.md"))
        text = ""
        for m in md_files:
            t = m.read_text(encoding="utf-8", errors="replace").strip()
            if t and len(t) > len(text):
                text = t
        if not text and md_files:
            text = md_files[0].read_text(encoding="utf-8", errors="replace").strip()

        return [text], [elapsed]

    def _run_ocr(self, image: Image.Image) -> str:
        """Run MinerU on an image (fallback)."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        path = Path(self._temp_dir) / "input.png"
        image.save(path, "PNG")
        pdf_path = Path(self._temp_dir) / "input.pdf"
        doc = fitz.open()
        img_doc = fitz.open(str(path))
        pdf_bytes = img_doc.convert_to_pdf()
        img_doc.close()
        img_pdf = fitz.open("pdf", pdf_bytes)
        doc.insert_pdf(img_pdf)
        doc.save(str(pdf_path))
        doc.close()
        img_pdf.close()
        path.unlink(missing_ok=True)

        texts, _ = self.run_ocr_on_pdf(pdf_path)
        return texts[0] if texts else ""

    def __del__(self):
        """Clean up temp directory on deletion."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
