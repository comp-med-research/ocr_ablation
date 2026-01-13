"""Kraken OCR implementation."""

from PIL import Image

from .base import BaseOCR


class KrakenOCR(BaseOCR):
    """Kraken OCR engine wrapper - specialized for historical documents."""

    name = "Kraken"

    def __init__(self, model_path: str = None):
        """
        Initialize Kraken OCR.
        
        Args:
            model_path: Path to custom Kraken model. If None, uses default.
        """
        super().__init__()
        self.model_path = model_path
        self._recognizer = None

    def load_model(self) -> None:
        """Load Kraken OCR model."""
        from kraken import blla
        from kraken.lib import models

        # Load the default model or custom model
        if self.model_path:
            self._recognizer = models.load_any(self.model_path)
        else:
            # Use default English model
            self._recognizer = models.load_any("en_best.mlmodel")

        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run Kraken OCR on an image."""
        from kraken import blla, rpred
        from kraken.lib import models

        # Convert to grayscale if needed
        if image.mode != "L":
            image = image.convert("L")

        # Perform baseline segmentation
        baseline_seg = blla.segment(image)

        # Run recognition
        pred_it = rpred.rpred(self._recognizer, image, baseline_seg)

        lines = []
        for record in pred_it:
            lines.append(record.prediction)

        return "\n".join(lines)

