"""Docling OCR implementation - Document conversion and text extraction."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from .base import BaseOCR


class DoclingOCR(BaseOCR):
    """Docling - Document conversion library for gen AI (DS4SD)."""

    name = "Docling"

    def __init__(self):
        """
        Initialize Docling OCR.
        
        Docling converts PDFs and images to Markdown with layout detection,
        table recognition, and formula extraction. Runs locally.
        """
        super().__init__()
        self._converter = None
        self._temp_dir = None
        self._page_document_dicts: list[dict] = []

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return False  # Docling uses CPU by default; GPU via backends if configured

    def load_model(self) -> None:
        """Load Docling document converter."""
        from docling.document_converter import DocumentConverter

        self._converter = DocumentConverter()
        self._temp_dir = tempfile.mkdtemp(prefix="docling_ocr_")
        self._page_document_dicts = []
        self._is_loaded = True

    def is_markdown_primary(self) -> bool:
        return True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run Docling on an image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        path = Path(self._temp_dir) / "input.png"
        image.save(path, "PNG")

        result = self._converter.convert(str(path))
        doc = getattr(result, "document", None)
        md = doc.export_to_markdown() if doc else ""
        md = md.strip() if md else ""

        if doc is not None and hasattr(doc, "export_to_dict"):
            try:
                self._page_document_dicts.append(doc.export_to_dict())
            except Exception:
                self._page_document_dicts.append({"_error": "export_to_dict failed"})

        nd = self.native_page_dir()
        if nd is not None and doc is not None:
            (nd / "docling.md").write_text(md, encoding="utf-8")
            try:
                if hasattr(doc, "export_to_dict"):
                    blob = doc.export_to_dict()
                    (nd / "docling_document.json").write_text(
                        json.dumps(blob, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
            except Exception:
                pass

        if path.exists():
            path.unlink(missing_ok=True)

        return md

    def get_json_string(self, indent: int = 2) -> str:
        """Serialised document structure per processed page (for combined ``*_output.json``)."""
        if not self._page_document_dicts:
            return "[]"
        return json.dumps(self._page_document_dicts, indent=indent, ensure_ascii=False, default=str)

    def __del__(self):
        """Clean up temp directory on deletion."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
