"""RolmOCR implementation - Reducto's OCR model."""

from PIL import Image
import torch
import base64
from io import BytesIO

from .base import BaseOCR


class RolmOCR(BaseOCR):
    """RolmOCR - Reducto's vision-language model for OCR."""

    name = "RolmOCR"

    def __init__(
        self,
        model_tag: str = "reducto/RolmOCR",
        device: str = None,
        use_vllm: bool = False
    ):
        """
        Initialize RolmOCR.
        
        Args:
            model_tag: HuggingFace model tag
            device: Device to run on ('cuda', 'cpu', or None for auto)
            use_vllm: Whether to use vLLM for faster inference (requires GPU)
        """
        super().__init__()
        self.model_tag = model_tag
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vllm = use_vllm and torch.cuda.is_available()
        self._processor = None

    def load_model(self) -> None:
        """Load RolmOCR model."""
        if self.use_vllm:
            self._load_vllm_model()
        else:
            self._load_transformers_model()
        self._is_loaded = True

    def _load_vllm_model(self) -> None:
        """Load model using vLLM for optimized inference."""
        from vllm import LLM

        self._model = LLM(
            model=self.model_tag,
            trust_remote_code=True,
            max_model_len=8192
        )

    def _load_transformers_model(self) -> None:
        """Load model using transformers."""
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(
            self.model_tag,
            trust_remote_code=True
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_tag,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        if self.device == "cpu":
            self._model.to(self.device)

    def _run_ocr(self, image: Image.Image) -> str:
        """Run RolmOCR on an image."""
        if self.use_vllm:
            return self._run_vllm_inference(image)
        else:
            return self._run_transformers_inference(image)

    def _run_vllm_inference(self, image: Image.Image) -> str:
        """Run inference using vLLM."""
        from vllm import SamplingParams

        # Encode image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        prompt = self._build_prompt(img_base64)

        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=4096
        )

        outputs = self._model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def _run_transformers_inference(self, image: Image.Image) -> str:
        """Run inference using transformers."""
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Build the message format for Qwen2-VL style models
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from this image. Return only the extracted text, preserving the original layout as much as possible."}
                ]
            }
        ]

        # Process with the processor
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False
            )

        # Decode output
        output_text = self._processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return output_text.strip()

    def _build_prompt(self, img_base64: str) -> str:
        """Build prompt for vLLM inference."""
        return f"""<|im_start|>user
<image>
Extract all text from this image. Return only the extracted text, preserving the original layout as much as possible.
<|im_end|>
<|im_start|>assistant
"""

