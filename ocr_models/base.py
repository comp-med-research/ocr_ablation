"""Base class for OCR models."""

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

