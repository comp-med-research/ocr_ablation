"""PaddleOCR implementation."""

from PIL import Image
import numpy as np

from .base import BaseOCR


class PaddleOCRModel(BaseOCR):
    """PaddleOCR engine wrapper."""

    name = "PaddleOCR"

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        """
        Initialize PaddleOCR.
        
        Args:
            lang: Language code ('en', 'ch', 'fr', etc.)
            use_gpu: Whether to use GPU acceleration
        """
        super().__init__()
        self.lang = lang
        self.use_gpu = use_gpu

    def load_model(self) -> None:
        """Load PaddleOCR model."""
        from paddleocr import PaddleOCR

        self._model = PaddleOCR(
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False
        )
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run PaddleOCR on an image."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run OCR
        result = self._model.ocr(img_array, cls=True)

        # Extract text from results
        if result is None or len(result) == 0:
            return ""

        lines = []
        for page_result in result:
            if page_result is None:
                continue
            for line in page_result:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], tuple) else line[1]
                    lines.append(text)

        return "\n".join(lines)

