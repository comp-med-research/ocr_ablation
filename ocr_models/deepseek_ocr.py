"""
DeepSeek OCR implementation - DeepSeek's vision-language model for OCR.

PRIVACY NOTE: This implementation runs entirely locally. No data is sent to
external servers. The model weights are downloaded once from HuggingFace Hub
and cached locally. All inference happens on your local machine.
"""

import tempfile
import os
from pathlib import Path
from PIL import Image
import torch

from .base import BaseOCR


class DeepSeekOCR(BaseOCR):
    """
    DeepSeek OCR - Using DeepSeek-OCR vision-language model for OCR.
    
    All processing is done locally. No images or text are sent to external servers.
    """

    name = "DeepSeekOCR"

    def __init__(
        self,
        model_tag: str = "deepseek-ai/DeepSeek-OCR",
        device: str = None,
        local_files_only: bool = False,
        prompt_mode: str = "markdown",
    ):
        """
        Initialize DeepSeek OCR.
        
        Args:
            model_tag: deepseek-ai/DeepSeek-OCR or a local path to downloaded model weights
            device: Device to run on ('cuda', 'cpu', or None for auto)
            local_files_only: If True, only use cached/local model files (no network).
            prompt_mode: "markdown" for structured output, "free" for plain OCR
        """
        super().__init__()
        self.model_tag = model_tag
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        self.prompt_mode = prompt_mode
        self._tokenizer = None
        self._temp_dir = None

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self.device == "cuda"

    def load_model(self) -> None:
        """
        Load DeepSeek-OCR model locally.
        
        Model weights are loaded from local cache or downloaded once from
        HuggingFace Hub if not cached. All inference runs locally.
        """
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_tag,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Build model kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
            "use_safetensors": True,
            "attn_implementation": "eager",
            "torch_dtype": torch.bfloat16 if self.device=="cuda" else None,
            "device_map": "cuda" if self.device=="cuda" else None
            
        }
        
        # if self.use_flash_attention:
        #     model_kwargs["_attn_implementation"] = "eager"
        
        self._model = AutoModel.from_pretrained(
            self.model_tag,
            **model_kwargs
        )
        
        # Move to device with appropriate dtype
        if self.device == "cuda":
            self._model = self._model.eval().cuda().to(torch.bfloat16)
        else:
            self._model = self._model.eval()
        
        # Create temp directory for intermediate files
        self._temp_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")
        
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run DeepSeek OCR on an image."""
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save image to temp file (DeepSeek-OCR expects a file path)
        # temp_image_path = os.path.join(self._temp_dir, "temp_input.jpg")
        # image.save(temp_image_path, "JPEG", quality=95)

        # Select prompt based on mode
        if self.prompt_mode == "markdown":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        else:
            prompt = "<image>\nFree OCR. "

        # Run inference
        output_text = self._model.infer(
            self._tokenizer,
            prompt=prompt,
            image_file=image,
            # output_path=self._temp_dir,
            base_size=1024,
            image_size=640,
            crop_mode=False,
            save_results=False,
            test_compress=False,
        )

        # Clean up temp image
        # if os.path.exists(temp_image_path):
        #     os.remove(temp_image_path)

        if output_text is None:
            return ""
        
        return output_text.strip() if isinstance(output_text, str) else str(output_text)

    def __del__(self):
        """Clean up temp directory on deletion."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)

