"""Nougat OCR implementation - Layout-aware academic document OCR."""

from PIL import Image
import torch

from .base import BaseOCR


class NougatOCR(BaseOCR):
    """Nougat OCR - Meta's academic document understanding model."""

    name = "Nougat"

    def __init__(self, model_tag: str = "facebook/nougat-base", device: str = None):
        """
        Initialize Nougat OCR.
        
        Args:
            model_tag: HuggingFace model tag
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        super().__init__()
        self.model_tag = model_tag
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None

    def load_model(self) -> None:
        """Load Nougat model."""
        from transformers import NougatProcessor, VisionEncoderDecoderModel

        self._processor = NougatProcessor.from_pretrained(self.model_tag)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_tag)
        self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run Nougat OCR on an image."""
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image
        pixel_values = self._processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate output
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=3584,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # Decode
        sequence = self._processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )[0]

        # Post-process (Nougat uses Markdown-like format)
        sequence = self._processor.post_process_generation(
            sequence,
            fix_markdown=True
        )

        return sequence

