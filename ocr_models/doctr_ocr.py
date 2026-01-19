"""DocTR OCR implementation."""

import json
import time
from pathlib import Path
from PIL import Image
import numpy as np

from .base import BaseOCR


class DocTROCR(BaseOCR):
    """DocTR - Document Text Recognition library by Mindee."""

    name = "DocTR"
    supports_native_pdf = True  # Flag indicating native PDF support

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        pretrained: bool = True
    ):
        """
        Initialize DocTR.
        
        Args:
            det_arch: Detection architecture
            reco_arch: Recognition architecture
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.pretrained = pretrained
        self._json_exports = []  # Accumulate JSON exports for all pages

    def load_model(self) -> None:
        """Load DocTR model."""
        from doctr.models import ocr_predictor

        self._model = ocr_predictor(
            det_arch=self.det_arch,
            reco_arch=self.reco_arch,
            pretrained=self.pretrained
        )
        self._is_loaded = True

    def run_ocr_on_pdf(self, pdf_path: Path, pages: list[int] = None) -> tuple[list[str], list[float]]:
        """
        Run DocTR directly on a PDF using native PDF processing.
        
        Args:
            pdf_path: Path to the PDF file
            pages: Optional list of page indices to process (0-indexed). None = all pages.
            
        Returns:
            Tuple of (list of text per page, list of processing times per page)
        """
        from doctr.io import DocumentFile
        from tqdm import tqdm

        if not self._is_loaded:
            self.load_model()

        # Clear previous exports
        self._json_exports = []

        # Load PDF natively with DocTR
        with tqdm(total=1, desc="Loading PDF", leave=False) as pbar:
            doc = DocumentFile.from_pdf(str(pdf_path))
            pbar.update(1)

        # Filter pages if specified
        if pages is not None:
            doc = [doc[i] for i in pages if i < len(doc)]

        # Process pages one at a time to get accurate per-page timings
        all_texts = []
        page_times = []
        
        for i, page_doc in enumerate(tqdm(doc, desc="Processing pages")):
            # Process single page
            start_time = time.perf_counter()
            result = self._model([page_doc])
            elapsed = time.perf_counter() - start_time
            page_times.append(elapsed)
            
            # Store JSON export
            self._json_exports.append(result.export())
            
            # Extract text from this page
            for page in result.pages:
                lines = []
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join(word.value for word in line.words)
                        lines.append(line_text)
                all_texts.append("\n".join(lines))

        return all_texts, page_times

    def _run_ocr(self, image: Image.Image) -> str:
        """Run DocTR on an image (fallback for image-based processing)."""
        # Convert PIL to numpy array
        img_array = np.array(image)

        # DocTR expects a list of numpy arrays or a DocumentFile
        result = self._model([img_array])

        # Store JSON export (nested dict with Page, Block, Line, Word structure)
        self._json_exports.append(result.export())

        # Extract text from result
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join(word.value for word in line.words)
                    lines.append(line_text)

        return "\n".join(lines)

    def get_json_export(self) -> list:
        """Get all JSON exports from OCR runs.
        
        Returns:
            List of nested dicts with document structure (Page, Block, Line, Word, Artefact)
        """
        return self._json_exports

    def get_json_string(self, indent: int = 2) -> str:
        """Get all JSON exports as a formatted string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation of all processed pages
        """
        if not self._json_exports:
            return "[]"
        return json.dumps(self._json_exports, indent=indent, ensure_ascii=False)
    
    def clear_json_exports(self) -> None:
        """Clear accumulated JSON exports."""
        self._json_exports = []

