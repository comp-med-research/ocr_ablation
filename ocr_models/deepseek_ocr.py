"""
DeepSeek-OCR-2 — vision-language document OCR (Hugging Face ``infer`` API).

See: https://github.com/deepseek-ai/DeepSeek-OCR-2

Runs locally; weights load from the Hugging Face Hub unless ``local_files_only`` is set.

Environment:

- ``DEEPSEEK_DEVICE``: ``cpu`` | ``cuda`` | ``auto`` — use ``cpu`` if CUDA kernels are missing for your GPU.
- ``DEEPSEEK_ATTN_ORDER``: ``flash_first`` to try flash attention before eager (default is eager-first).
- ``DEEPSEEK_TRY_FLASH_WITHOUT_PACKAGE=1``: still attempt ``flash_attention_2`` when ``flash_attn`` is not
  installed (default is to skip that implementation so the reported error is not misleading).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image
import torch

from .base import BaseOCR


def _deepseek_device_from_env(explicit: str | None) -> str:
    """``DEEPSEEK_DEVICE``: ``cpu`` | ``cuda`` | ``auto`` (default: auto)."""
    if explicit is not None:
        return explicit
    d = (os.environ.get("DEEPSEEK_DEVICE") or "auto").strip().lower()
    if d == "cpu":
        return "cpu"
    if d == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cuda_smoke_test() -> None:
    """Fail fast with a clear message if this PyTorch build has no kernels for the GPU (e.g. sm_120 + cu124 wheel)."""
    try:
        for dt in (torch.bfloat16, torch.float32):
            try:
                a = torch.zeros((8, 8), device="cuda", dtype=dt)
                torch.matmul(a, a)
                torch.cuda.synchronize()
                del a
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("CUDA matmul smoke test failed for bfloat16 and float32")
    except RuntimeError as e:
        msg = str(e).lower()
        if "no kernel image" in msg or "kernel image" in msg:
            cap = torch.cuda.get_device_capability(0)
            raise RuntimeError(
                "PyTorch cannot run CUDA kernels on this GPU (often: Blackwell sm_120 / compute "
                f"capability {cap} with a wheel built only through sm_90).\n\n"
                "Fix: install a PyTorch build that includes your architecture — see "
                "https://pytorch.org/get-started/locally/ (often **nightly** with cu128+ for Blackwell).\n\n"
                "Workaround: run on CPU (slow): export DEEPSEEK_DEVICE=cpu"
            ) from e
        raise


def _flash_attn_import_available() -> bool:
    """``flash_attn`` wheel present (Transformers' FA2 path requires it)."""
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except Exception:
        return False


def _prepare_cuda_sdp_backends() -> None:
    """Make ``sdpa`` fall back to math kernels instead of Flash SDP (no ``flash_attn``)."""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except (AttributeError, RuntimeError):
        pass


class DeepSeekOCR(BaseOCR):
    """DeepSeek-OCR-2 (replaces the previous DeepSeek-OCR v1 wrapper)."""

    name = "DeepSeek-OCR-2"

    def __init__(
        self,
        model_tag: str = "deepseek-ai/DeepSeek-OCR-2",
        device: str | None = None,
        local_files_only: bool = False,
        prompt_mode: str = "markdown",
        base_size: int = 1024,
        image_size: int = 768,
        crop_mode: bool = True,
    ):
        super().__init__()
        self.model_tag = model_tag
        self.device = _deepseek_device_from_env(device)
        self.local_files_only = local_files_only
        self.prompt_mode = (prompt_mode or "markdown").lower()
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self._tokenizer = None
        self._temp_dir: str | None = None

    @property
    def uses_gpu(self) -> bool:
        return self.device == "cuda"

    def is_markdown_primary(self) -> bool:
        return self.prompt_mode == "markdown"

    def load_model(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_tag,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        common = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
            "use_safetensors": True,
        }

        # Prefer eager/sdpa first: hub code + flash_attention_2 often pulls
        # ``LlamaFlashAttention2``, which newer transformers releases removed/renamed.
        attn_order_env = (os.environ.get("DEEPSEEK_ATTN_ORDER") or "").strip().lower()
        if attn_order_env in ("flash_first", "flash"):
            cuda_impls = ["flash_attention_2", "sdpa", "eager"]
        else:
            cuda_impls = ["eager", "sdpa", "flash_attention_2"]

        try_flash_without_pkg = (
            (os.environ.get("DEEPSEEK_TRY_FLASH_WITHOUT_PACKAGE") or "").strip() == "1"
        )
        if not _flash_attn_import_available() and not try_flash_without_pkg:
            cuda_impls = [x for x in cuda_impls if x != "flash_attention_2"]

        last_err: Exception | None = None
        if self.device == "cuda":
            common["torch_dtype"] = torch.bfloat16
            _prepare_cuda_sdp_backends()
            attempt_errors: list[str] = []
            for impl in cuda_impls:
                try:
                    self._model = AutoModel.from_pretrained(
                        self.model_tag,
                        attn_implementation=impl,
                        **common,
                    )
                    self._model = self._model.eval().cuda()
                    _cuda_smoke_test()
                    break
                except Exception as e:
                    last_err = e
                    attempt_errors.append(f"{impl}: {e}")
                    self._model = None
            if self._model is None:
                joined = "\n---\n".join(attempt_errors)
                hint = (
                    "If you see LlamaFlashAttention2 errors, keep default order (eager first) or "
                    "pin transformers to the OCR-2 repo pin: transformers==4.46.3.\n"
                )
                if not _flash_attn_import_available() and not try_flash_without_pkg:
                    hint += (
                        "``flash_attn`` is not installed — skipped ``flash_attention_2``. "
                        "Optional: pip install flash-attn (may not build on very new GPUs).\n"
                    )
                raise RuntimeError(
                    "Failed to load DeepSeek-OCR-2 on CUDA (tried "
                    f"{', '.join(cuda_impls)}).\n{hint}"
                    f"Per-attempt errors:\n---\n{joined}"
                ) from last_err
        else:
            self._model = AutoModel.from_pretrained(
                self.model_tag,
                attn_implementation="eager",
                torch_dtype=torch.float32,
                **common,
            )
            self._model = self._model.eval()

        self._temp_dir = tempfile.mkdtemp(prefix="deepseek_ocr2_")
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        assert self._temp_dir is not None
        img_path = Path(self._temp_dir) / f"in_{self._page_index:04d}.png"
        image.save(img_path, "PNG")
        infer_out = Path(self._temp_dir) / f"out_{self._page_index:04d}"
        infer_out.mkdir(parents=True, exist_ok=True)

        if self.prompt_mode == "markdown":
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        else:
            prompt = "<image>\nFree OCR. "

        try:
            output_text = self._model.infer(
                self._tokenizer,
                prompt=prompt,
                image_file=str(img_path),
                output_path=str(infer_out),
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                save_results=False,
            )
        except RuntimeError as e:
            if "no kernel image" in str(e).lower():
                raise RuntimeError(
                    "CUDA kernel error during DeepSeek-OCR-2 inference (same cause as load-time: "
                    "PyTorch wheel likely has no SASS for this GPU, e.g. Blackwell with an older CUDA wheel). "
                    "Upgrade PyTorch per https://pytorch.org/get-started/locally/ or use DEEPSEEK_DEVICE=cpu."
                ) from e
            raise

        text = ""
        if output_text is not None:
            text = output_text.strip() if isinstance(output_text, str) else str(output_text)
        if not text:
            for m in sorted(infer_out.rglob("*.md")):
                t = m.read_text(encoding="utf-8", errors="replace").strip()
                if t:
                    text = t
                    break

        ext = "md" if self.prompt_mode == "markdown" else "txt"
        self.write_native_text(text, f"deepseek_ocr2.{ext}")
        return text

    def __del__(self):
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
