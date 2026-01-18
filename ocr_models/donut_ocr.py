"""Donut OCR implementation - Document Understanding Transformer."""

from PIL import Image
import torch
from .base import BaseOCR


class DonutOCR(BaseOCR):
    """Donut OCR - Naver's Document Understanding Transformer."""

    name = "Donut"

    def __init__(
        self,
        model_tag: str = "naver-clova-ix/donut-base-finetuned-cord-v2",
        device: str = None
    ):
        """
        Initialize Donut OCR.
        
        Args:
            model_tag: HuggingFace model tag. Options:
                - "naver-clova-ix/donut-base-finetuned-cord-v2" (receipt parsing)
                - "naver-clova-ix/donut-base-finetuned-docvqa" (document QA)
                - "naver-clova-ix/donut-base-finetuned-rvlcdip" (document classification)
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
        """Load Donut model."""
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self._processor = DonutProcessor.from_pretrained(self.model_tag)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_tag)
        self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run Donut OCR on an image."""
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Prepare decoder input - use task prompt for text extraction
        task_prompt = "<s_cord-v2>" if "cord" in self.model_tag else "<s>"
        decoder_input_ids = self._processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Process image
        pixel_values = self._processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate output
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self._model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # Decode
        sequence = self._processor.batch_decode(outputs.sequences)[0]
        
        # Remove special tokens and clean up
        sequence = sequence.replace(self._processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self._processor.tokenizer.pad_token, "")
        
        # Try to parse as JSON-like structure and extract text
        sequence = self._extract_text_from_output(sequence)

        return sequence

    def _extract_text_from_output(self, sequence: str) -> str:
        """Extract readable text from Donut's structured output."""
        import re
        
        # Remove XML-like tags but keep content
        # Donut outputs things like <s_menu><s_nm>Coffee</s_nm><s_price>$3.00</s_price></s_menu>
        
        # First, try to extract all text between tags
        text_parts = []
        
        # Pattern to match content between tags
        pattern = r'>([^<]+)<'
        matches = re.findall(pattern, sequence)
        
        for match in matches:
            cleaned = match.strip()
            if cleaned:
                text_parts.append(cleaned)
        
        if text_parts:
            return "\n".join(text_parts)
        
        # Fallback: just remove all tags
        clean = re.sub(r'<[^>]+>', ' ', sequence)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        return clean

