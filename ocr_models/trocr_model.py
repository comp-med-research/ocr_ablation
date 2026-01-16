"""TrOCR implementation - Microsoft's Transformer-based OCR."""

from PIL import Image
import torch

from .base import BaseOCR


class TrOCRModel(BaseOCR):
    """TrOCR - Transformer-based OCR by Microsoft."""

    name = "TrOCR"

    def __init__(
        self,
        model_tag: str = "microsoft/trocr-large-printed",
        device: str = None
    ):
        """
        Initialize TrOCR.
        
        Args:
            model_tag: HuggingFace model tag. Options:
                - microsoft/trocr-base-printed
                - microsoft/trocr-large-printed
                - microsoft/trocr-base-handwritten
                - microsoft/trocr-large-handwritten
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        super().__init__()
        self.model_tag = model_tag
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self.device == "cuda"

    def load_model(self) -> None:
        """Load TrOCR model."""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._processor = TrOCRProcessor.from_pretrained(self.model_tag)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_tag)
        self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """
        Run TrOCR on an image.
        
        Note: TrOCR is designed for single-line text. For full documents,
        we need to segment the image into lines first.
        """
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # For full documents, segment into lines using a simple approach
        lines_texts = self._process_document(image)
        return "\n".join(lines_texts)

    def _process_document(self, image: Image.Image) -> list[str]:
        """
        Process a full document image by detecting and recognizing text lines.
        
        For simplicity, this uses a sliding window approach.
        For production, you'd want proper line detection.
        """
        width, height = image.size
        
        # If image is small enough, process as single line
        if height < 100:
            return [self._recognize_line(image)]

        # Otherwise, segment into horizontal strips
        line_height = 50
        overlap = 10
        lines = []
        
        y = 0
        while y < height:
            y_end = min(y + line_height, height)
            line_img = image.crop((0, y, width, y_end))
            
            # Only process if there's content (not blank)
            if self._has_content(line_img):
                text = self._recognize_line(line_img)
                if text.strip():
                    lines.append(text)
            
            y += line_height - overlap

        return lines

    def _recognize_line(self, image: Image.Image) -> str:
        """Recognize a single line of text."""
        pixel_values = self._processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)

        text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text

    def _has_content(self, image: Image.Image) -> bool:
        """Check if an image has text content (not blank)."""
        import numpy as np
        arr = np.array(image.convert("L"))
        # Check if there's enough variance (not uniform)
        return arr.std() > 10

