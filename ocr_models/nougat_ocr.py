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

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self.device == "cuda"

    def is_markdown_primary(self) -> bool:
        return True

    def combined_markdown_extension(self) -> str:
        return ".mmd"

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

        # Use ``image_processor`` directly: ``NougatProcessor.__call__`` can pass ``None`` into
        # strict kwargs (e.g. ``do_crop_margin``) on recent ``transformers`` (5.4+), raising
        # ``StrictDataclassFieldValidationError``.
        feats = self._processor.image_processor(image, return_tensors="pt")
        pixel_values = feats.pixel_values.to(self.device)

        # Generate output (``return_dict_in_generate`` → decode ``sequences``, not the full output object)
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=3584,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
        sequence = self._processor.batch_decode(ids, skip_special_tokens=True)[0]

        # Post-process (Nougat uses Markdown-like format)
        sequence = self._processor.post_process_generation(
            sequence,
            fix_markdown=True
        )

        # Nougat’s post-processed text is conventionally stored as ``.mmd`` (markdown + math).
        self.write_native_text(sequence, "nougat.mmd")
        return sequence

