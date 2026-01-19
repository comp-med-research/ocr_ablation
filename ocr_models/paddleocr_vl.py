"""PaddleOCR-VL implementation - PaddleOCR's Vision-Language model."""

from pathlib import Path
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
        self._output_dir = None  # Will be set by runner
        self._page_counter = 0

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self.use_gpu

    def set_output_dir(self, output_dir: Path) -> None:
        """Set the output directory for JSON/markdown files."""
        self._output_dir = Path(output_dir)
        self._page_counter = 0

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
        
        # Parse image to text
        result = self._pipeline.predict(img_array)

        if result is None:
            return ""

        # Determine output path
        if self._output_dir is not None:
            save_path = str(self._output_dir / f"paddleocr-vl_page_{self._page_counter}")
        else:
            save_path = "output"
        
        # Extract text and save results
        all_text = []
        for res in result:
            # Save JSON and markdown to the output directory
            res.save_to_json(save_path=save_path)
            res.save_to_markdown(save_path=save_path)
            
            # Extract text from result
            if hasattr(res, 'text'):
                all_text.append(res.text)
            elif hasattr(res, 'rec_text'):
                all_text.append(res.rec_text)
            else:
                # Try to get string representation
                all_text.append(str(res))
        
        self._page_counter += 1
        
        return "\n".join(all_text) if all_text else ""

