"""Base class for OCR models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from PIL import Image
import time


class BaseOCR(ABC):
    """Abstract base class for all OCR models."""

    name: str = "BaseOCR"

    def __init__(self):
        self._model = None
        self._is_loaded = False
        self._native_root: Path | None = None
        self._page_index: int = 0

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return False  # Default to False, subclasses can override

    def configure_native_exports(self, output_dir: Path, model_slug: str) -> None:
        """
        Enable per-page native artifacts under ``output_dir/native/<model_slug>/page_XXXX/``.

        Called by the ablation runner unless ``--no-native-exports`` is set.
        """
        self._native_root = (output_dir / "native" / model_slug).resolve()
        self._native_root.mkdir(parents=True, exist_ok=True)

    def set_page_index(self, index: int) -> None:
        """0-based page index for the current ``run_ocr`` call (set by runner)."""
        self._page_index = int(index)

    def native_page_dir(self) -> Path | None:
        """Directory for the current page, or ``None`` if native exports are disabled."""
        if self._native_root is None:
            return None
        p = self._native_root / f"page_{self._page_index:04d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_native_text(self, text: str, filename: str = "transcription.md") -> None:
        """Write plain text / markdown into the current native page folder."""
        d = self.native_page_dir()
        if d is not None:
            (d / filename).write_text(text or "", encoding="utf-8")

    def is_markdown_primary(self) -> bool:
        """
        If True, the runner writes a combined ``*_output.md`` (Markdown) instead of default ``*.txt``.

        Plain OCR models keep ``*_output.txt`` unless the user disables it.
        """
        return False

    def combined_markdown_extension(self) -> str:
        """
        File extension for the combined export when ``is_markdown_primary()`` (including the dot).
        Default ``.md``; override e.g. for Nougat (``.mmd``).
        """
        return ".md"

    @abstractmethod
    def load_model(self) -> None:
        """Load the OCR model into memory."""
        pass

    @abstractmethod
    def _run_ocr(self, image: Image.Image) -> str:
        """Run OCR on a single image. Must be implemented by subclasses."""
        pass

    def run_ocr(self, image: Image.Image) -> tuple[str, float]:
        """
        Run OCR on a single image and return text with timing.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Tuple of (extracted_text, time_taken_seconds)
        """
        if not self._is_loaded:
            self.load_model()

        start_time = time.perf_counter()
        text = self._run_ocr(image)
        elapsed = time.perf_counter() - start_time

        return text, elapsed

    def run_ocr_on_images(self, images: list[Image.Image]) -> tuple[list[str], float]:
        """
        Run OCR on multiple images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Tuple of (list_of_texts, total_time_seconds)
        """
        if not self._is_loaded:
            self.load_model()

        texts = []
        total_time = 0.0

        for img in images:
            text, elapsed = self.run_ocr(img)
            texts.append(text)
            total_time += elapsed

        return texts, total_time

    def __repr__(self) -> str:
        return f"{self.name}(loaded={self._is_loaded})"

