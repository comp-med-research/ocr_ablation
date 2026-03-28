"""Chandra OCR 2 (Datalab) — ``datalab-to/chandra-ocr-2`` via in-process Hugging Face."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
import torch

from .base import BaseOCR

_DEFAULT_MODEL = "datalab-to/chandra-ocr-2"


class ChandraOCR(BaseOCR):
    """
    `Chandra 2 <https://huggingface.co/datalab-to/chandra-ocr-2>`_ using the official
    ``chandra.model.hf.generate_hf`` path (``pip install chandra-ocr[hf]``).

    Env:

    - ``CHANDRA_MODEL_ID``: Hub repo (default ``datalab-to/chandra-ocr-2``).
    - ``CHANDRA_PROMPT_TYPE``: e.g. ``ocr_layout`` (default).
    - ``CHANDRA_MAX_OUTPUT_TOKENS``: passed to ``generate_hf`` (default ``12384``).
    - ``CHANDRA_BACKEND``: ``hf`` (default) or ``cli`` to use the ``chandra`` subprocess instead.
    - If ``cli``: ``CHANDRA_METHOD`` is ``hf`` or ``vllm`` as in the Chandra CLI.
    """

    name = "Chandra OCR 2"

    def __init__(self, timeout_seconds: int = 900):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.model_id = (
            os.environ.get("CHANDRA_MODEL_ID") or _DEFAULT_MODEL
        ).strip()
        pt = (os.environ.get("CHANDRA_PROMPT_TYPE") or "ocr_layout").strip()
        self.prompt_type = pt or "ocr_layout"
        self.backend = (os.environ.get("CHANDRA_BACKEND") or "hf").strip().lower()
        try:
            self.max_output_tokens = int(
                os.environ.get("CHANDRA_MAX_OUTPUT_TOKENS") or "12384"
            )
        except ValueError:
            self.max_output_tokens = 12384

    @property
    def uses_gpu(self) -> bool:
        if self.backend != "hf":
            return False
        if self._is_loaded and self._model is not None:
            try:
                return next(self._model.parameters()).is_cuda
            except (StopIteration, RuntimeError):
                return False
        return torch.cuda.is_available()

    def is_markdown_primary(self) -> bool:
        return True

    def load_model(self) -> None:
        if self.backend == "cli":
            if not shutil.which("chandra"):
                raise RuntimeError(
                    "Chandra CLI not found. Install with: pip install 'chandra-ocr[hf]' "
                    "and ensure `chandra` is on PATH, or use default CHANDRA_BACKEND=hf."
                )
            self._model = None
            self._is_loaded = True
            return

        try:
            from chandra.model.hf import generate_hf  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Chandra 2 HF path requires: pip install 'chandra-ocr[hf]' "
                f"(import error: {e})"
            ) from e

        from transformers import AutoModelForImageTextToText, AutoProcessor

        common = {"trust_remote_code": True}
        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            model = None
            for attn in ("sdpa", "eager"):
                try:
                    model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        torch_dtype=dtype,
                        device_map="auto",
                        attn_implementation=attn,
                        **common,
                    )
                    break
                except (TypeError, ValueError, OSError, RuntimeError):
                    continue
            if model is None:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    **common,
                )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                **common,
            )
            model.to("cpu")

        model.eval()
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
        model.processor = processor

        if not hasattr(model, "device"):
            try:
                model.device = next(model.parameters()).device
            except StopIteration:
                model.device = torch.device("cpu")

        self._model = model
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.backend == "cli":
            return self._run_ocr_cli(image)

        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem
        from chandra.output import parse_markdown

        batch = [
            BatchInputItem(image=image, prompt_type=self.prompt_type or None)
        ]
        results = generate_hf(
            batch,
            self._model,
            max_output_tokens=self.max_output_tokens,
        )
        raw = results[0].raw if results else ""
        text = parse_markdown(raw) if raw else ""

        self.write_native_text(text, "chandra2.md")
        nd = self.native_page_dir()
        if nd is not None:
            (nd / "chandra2_raw.html").write_text(raw or "", encoding="utf-8")
            (nd / "chandra2_meta.json").write_text(
                json.dumps(
                    {
                        "model_id": self.model_id,
                        "prompt_type": self.prompt_type,
                        "backend": "hf",
                        "max_output_tokens": self.max_output_tokens,
                        "token_count": results[0].token_count if results else None,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return text

    def _run_ocr_cli(self, image: Image.Image) -> str:
        method = os.environ.get("CHANDRA_METHOD", "hf").strip().lower() or "hf"
        work = Path(tempfile.mkdtemp(prefix="chandra_ocr_cli_"))
        try:
            in_path = work / "page.png"
            image.save(in_path, "PNG")
            out_root = work / "out"
            out_root.mkdir()

            proc = subprocess.run(
                [
                    "chandra",
                    str(in_path),
                    str(out_root),
                    "--method",
                    method,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(work),
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()[:2000]
                raise RuntimeError(f"chandra failed (code {proc.returncode}): {err}")

            md_files = sorted(out_root.rglob("*.md"))
            text = ""
            for m in md_files:
                t = m.read_text(encoding="utf-8", errors="replace").strip()
                if t:
                    text = t
                    break

            nd = self.native_page_dir()
            if nd is not None:
                ch_root = nd / "chandra_cli"
                ch_root.mkdir(parents=True, exist_ok=True)
                if out_root.exists():
                    for f in out_root.rglob("*"):
                        if f.is_file():
                            rel = f.relative_to(out_root)
                            dest = ch_root / rel
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(f, dest)
                (nd / "chandra2_meta.json").write_text(
                    json.dumps(
                        {
                            "backend": "cli",
                            "method": method,
                            "cli_returncode": proc.returncode,
                            "markdown_files_found": len(md_files),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            self.write_native_text(text, "chandra2.md")
            return text
        finally:
            shutil.rmtree(work, ignore_errors=True)
