"""
DeepSeek OCR implementation - DeepSeek's vision-language model for OCR.

PRIVACY NOTE: This implementation runs entirely locally. No data is sent to
external servers. The model weights are downloaded once from HuggingFace Hub
and cached locally. All inference happens on your local machine.
"""

from PIL import Image
import torch

from .base import BaseOCR


class DeepSeekOCR(BaseOCR):
    """
    DeepSeek OCR - Using DeepSeek-VL2 vision-language model for OCR.
    
    All processing is done locally. No images or text are sent to external servers.
    """

    name = "DeepSeekOCR"

    def __init__(
        self,
        model_tag: str = "deepseek-ai/deepseek-vl2-tiny",
        device: str = None,
        local_files_only: bool = False,
    ):
        """
        Initialize DeepSeek OCR.
        
        Args:
            model_tag: HuggingFace model tag or local path. Options:
                - "deepseek-ai/deepseek-vl2-tiny" (fastest, ~3B params)
                - "deepseek-ai/deepseek-vl2-small" (balanced, ~16B params)
                - "deepseek-ai/deepseek-vl2" (best quality, ~27B params)
                - Or a local path to downloaded model weights
            device: Device to run on ('cuda', 'cpu', or None for auto)
            local_files_only: If True, only use cached/local model files (no network).
                              Set to True for air-gapped environments.
        """
        super().__init__()
        self.model_tag = model_tag
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        self._processor = None
        self._tokenizer = None

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self.device == "cuda"

    def load_model(self) -> None:
        """
        Load DeepSeek-VL2 model locally.
        
        Model weights are loaded from local cache or downloaded once from
        HuggingFace Hub if not cached. All inference runs locally.
        """
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_tag,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_tag,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        
        if self.device == "cpu":
            self._model.to(self.device)
        
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run DeepSeek OCR on an image."""
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Build the conversation format for DeepSeek-VL2
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\nExtract all text from this image. Return only the extracted text, preserving the original layout as much as possible.",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Prepare inputs using the processor
        inputs = self._processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        # Decode the output
        # Get only the generated tokens (exclude input)
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        output_text = self._processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return output_text.strip()

