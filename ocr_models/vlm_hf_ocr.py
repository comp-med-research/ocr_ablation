"""Large HF vision-language models used as document OCR (chat + generate)."""

from __future__ import annotations

import json
import os

from PIL import Image
import torch

from .base import BaseOCR


def _move_batch_to_device(batch, device: torch.device):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    return batch


def _strip_token_type_ids(batch):
    if hasattr(batch, "pop"):
        try:
            batch.pop("token_type_ids", None)
        except Exception:
            pass
    elif isinstance(batch, dict) and "token_type_ids" in batch:
        batch = dict(batch)
        batch.pop("token_type_ids", None)
    return batch


# Qwen3.5 chat template wraps chain-of-thought in ``<think>`` … ``</think>`` (see tokenizer_config.json).
_QWEN35_THINK_END = "</think>"


def _strip_qwen35_thinking_block(text: str) -> str:
    """Drop a leading ``<think>``…``</think>`` block if present in decoded text."""
    text = text.strip()
    if _QWEN35_THINK_END in text:
        return text.split(_QWEN35_THINK_END)[-1].strip()
    return text


def _transformers_has_qwen35() -> bool:
    """True if this ``transformers`` build registers ``model_type`` ``qwen3_5``."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        return "qwen3_5" in CONFIG_MAPPING_NAMES
    except Exception:
        try:
            from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401

            return True
        except ImportError:
            return False


def _load_qwen35_model(model_id: str, torch_dtype, device_map: str | None):
    """
    Load ``Qwen3.5-9B`` (``Qwen3_5ForConditionalGeneration``).

    The Hub config pins ``transformers`` ~4.57+; older wheels do not define ``qwen3_5``:
    https://huggingface.co/Qwen/Qwen3.5-9B
    """
    import transformers as _tf

    if not _transformers_has_qwen35():
        raise RuntimeError(
            f"Your installed transformers ({_tf.__version__}) does not support Qwen3.5 "
            "(model_type `qwen3_5` is missing). Install a newer build, then retry:\n\n"
            "  pip install -U \"transformers>=4.57.0\"\n\n"
            "If PyPI is still too old for this checkpoint, install from source:\n\n"
            "  pip install git+https://github.com/huggingface/transformers.git\n\n"
            "Optional pin file in this repo: requirements-qwen35.txt\n"
            "Model card: https://huggingface.co/Qwen/Qwen3.5-9B"
        )

    base_kw: dict = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if device_map is not None:
        base_kw["device_map"] = device_map

    from transformers import Qwen3_5ForConditionalGeneration

    for attn in ("sdpa", "eager"):
        try:
            return Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id, attn_implementation=attn, **base_kw
            )
        except (TypeError, ValueError, RuntimeError, OSError):
            continue
    return Qwen3_5ForConditionalGeneration.from_pretrained(model_id, **base_kw)


class Qwen35VLOCR(BaseOCR):
    """
    Qwen3.5 unified vision-language model (OCR via ``generate``).

    Default weights: ``Qwen/Qwen3.5-9B`` (`Qwen3.5-9B model card
    <https://huggingface.co/Qwen/Qwen3.5-9B>`_).

    Override the repo id with env ``QWEN35_MODEL_ID`` (or legacy ``QWEN35_VL_MODEL_ID``).
    Requires ``transformers`` new enough to provide ``Qwen3_5ForConditionalGeneration`` (see card).
    """

    name = "Qwen3.5-9B"

    def __init__(self, max_new_tokens: int = 4096):
        super().__init__()
        self.model_id = (
            (os.environ.get("QWEN35_MODEL_ID") or "").strip()
            or (os.environ.get("QWEN35_VL_MODEL_ID") or "").strip()
            or "Qwen/Qwen3.5-9B"
        )
        self.max_new_tokens = max_new_tokens
        self._processor = None

    @property
    def uses_gpu(self) -> bool:
        return torch.cuda.is_available()

    def is_markdown_primary(self) -> bool:
        return True

    def load_model(self) -> None:
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            self._model = _load_qwen35_model(self.model_id, dtype, "auto")
        else:
            self._model = _load_qwen35_model(self.model_id, torch.float32, None)
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
                    {
                        "type": "text",
                        "text": (
                            "Convert this document page to Markdown. Preserve headings, lists, "
                            "and tables. Output only the extracted content."
                        ),
                    },
                ],
            }
        ]

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs = _strip_token_type_ids(inputs)
        device = next(self._model.parameters()).device
        inputs = _move_batch_to_device(inputs, device)

        in_tensor = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        in_len = in_tensor.shape[1]
        trimmed = generated_ids[:, in_len:]
        text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        text = _strip_qwen35_thinking_block(text)

        self.write_native_text(text, "qwen35.md")
        nd = self.native_page_dir()
        if nd is not None:
            (nd / "qwen35_meta.json").write_text(
                json.dumps(
                    {
                        "model_id": self.model_id,
                        "max_new_tokens": self.max_new_tokens,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                ),
                encoding="utf-8",
            )
        return text


class Llama32VisionOCR(BaseOCR):
    """
    Llama 3.2 Vision (11B) instruct — multimodal OCR via ``MllamaForConditionalGeneration``.

    Default: ``meta-llama/Llama-3.2-11B-Vision-Instruct`` (gated; needs Hugging Face access + token).
    Override with env ``LLAMA32_VL_MODEL_ID``.
    Flash Attention 2 is disabled upstream for this architecture; we use ``attn_implementation='eager'``.
    """

    name = "Llama-3.2-11B-Vision-Instruct"

    def __init__(self, max_new_tokens: int = 4096):
        super().__init__()
        self.model_id = (
            os.environ.get("LLAMA32_VL_MODEL_ID")
            or "meta-llama/Llama-3.2-11B-Vision-Instruct"
        ).strip()
        self.max_new_tokens = max_new_tokens
        self._processor = None

    @property
    def uses_gpu(self) -> bool:
        return torch.cuda.is_available()

    def is_markdown_primary(self) -> bool:
        return True

    def load_model(self) -> None:
        try:
            from transformers import AutoProcessor, MllamaForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                "Llama 3.2 Vision needs `MllamaForConditionalGeneration` "
                "(transformers>=4.45 or so). "
                f"Import error: {e}"
            ) from e

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        common = {
            "attn_implementation": "eager",
        }
        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            self._model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto",
                **common,
            )
        else:
            self._model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                **common,
            )
            self._model.to("cpu")
        self._model.eval()
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": (
                                "Convert this document page to Markdown. Preserve headings, lists, "
                                "and tables. Output only the extracted content."
                            ),
                        },
                    ],
                }
            ]
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = _move_batch_to_device(inputs, device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        in_len = inputs["input_ids"].shape[1]
        text = self._processor.decode(
            out[0][in_len:], skip_special_tokens=True
        ).strip()

        self.write_native_text(text, "llama32_vision.md")
        nd = self.native_page_dir()
        if nd is not None:
            (nd / "llama32_vision_meta.json").write_text(
                json.dumps(
                    {"model_id": self.model_id, "max_new_tokens": self.max_new_tokens}
                ),
                encoding="utf-8",
            )
        return text
