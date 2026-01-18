"""PaddleOCR-VL implementation - PaddleOCR's Vision-Language model."""

from PIL import Image
import numpy as np

from .base import BaseOCR


class PaddleOCRVL(BaseOCR):
    """PaddleOCR-VL - PaddlePaddle's Vision-Language OCR model."""

    name = "PaddleOCR-VL"

    def __init__(self, use_gpu: bool = True):
        """
        Initialize PaddleOCR-VL.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        super().__init__()
        self.use_gpu = use_gpu
        self._pipeline = None

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self.use_gpu

    def load_model(self) -> None:
        """Load PaddleOCR-VL model."""
        from paddleocr import PaddleOCRVL as PaddleVL

        self._pipeline = PaddleVL(
            device="gpu" if self.use_gpu else "cpu"
        )
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run PaddleOCR-VL on an image."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run VL-based OCR with a text extraction prompt
        prompt = "Extract all text from this image, preserving the layout."
        
        result = self._pipeline.predict(
            image=img_array,
            prompt=prompt
        )

        if result is None:
            return ""

        # Handle different result formats
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result.get("text", result.get("result", str(result)))
        elif isinstance(result, list):
            return "\n".join(str(r) for r in result)
        else:
            return str(result)

