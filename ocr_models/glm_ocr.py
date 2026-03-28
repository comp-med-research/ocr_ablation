"""GLM-OCR (Z.ai) — multimodal document OCR via Hugging Face Transformers."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import torch

from .base import BaseOCR


class GlmOcrModel(BaseOCR):
    """
    GLM-OCR: ``zai-org/GLM-OCR`` (see Hugging Face model card).

    Requires a recent ``transformers`` with ``GlmOcrForConditionalGeneration``.
    """

    name = "GLM-OCR"

    def __init__(self, model_id: str = "zai-org/GLM-OCR", max_new_tokens: int = 4096):
        super().__init__()
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._processor = None

    @property
    def uses_gpu(self) -> bool:
        return torch.cuda.is_available()

    def is_markdown_primary(self) -> bool:
        return True

    def load_model(self) -> None:
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        if torch.cuda.is_available():
            self._model = GlmOcrForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self._model = GlmOcrForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self._model.to("cpu")
        self._model.eval()
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        text = self._processor.decode(out[0], skip_special_tokens=True).strip()

        self.write_native_text(text, "glm_ocr.md")
        nd = self.native_page_dir()
        if nd is not None:
            (nd / "glm_ocr_meta.json").write_text(
                json.dumps({"model_id": self.model_id, "max_new_tokens": self.max_new_tokens}),
                encoding="utf-8",
            )

        return text
