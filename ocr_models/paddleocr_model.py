"""PaddleOCR implementation."""

from PIL import Image
import numpy as np

from .base import BaseOCR


class PaddleOCRModel(BaseOCR):
    """PaddleOCR engine wrapper."""

    name = "PaddleOCR"

    def __init__(self, lang: str = "en", use_gpu: bool = True):
        """
        Initialize PaddleOCR.
        
        Args:
            lang: Language code ('en', 'ch', 'fr', etc.)
            uses_gpu: Whether to use GPU acceleration
        """
        super().__init__()
        self.lang = lang
        self.use_gpu = use_gpu

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self.use_gpu

    def load_model(self) -> None:
        """Load PaddleOCR model."""
        from paddleocr import PaddleOCR

        self._model = PaddleOCR(
            lang=self.lang,
            device ="gpu",
            # show_log=False
        )
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run PaddleOCR on an image."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run OCR
        result = self._model.predict(img_array)

        # Extract text from results
        if result is None or len(result) == 0:
            return ""
        
        lines = []
        for res in result:
            text = res["rec_texts"]
            for t in text:
                lines.append(t)

        return "\n".join(lines)

