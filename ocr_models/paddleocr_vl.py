"""PaddleOCR-VL implementation - PaddleOCR's Vision-Language model."""

import os
from pathlib import Path

from PIL import Image
import numpy as np

from . import paddle_utils  # noqa: F401
from .base import BaseOCR
from .paddle_utils import (
    paddleocr_enable_mkldnn,
    paddleocr_vl_use_queues,
    resolve_paddle_device,
)


class PaddleOCRVL(BaseOCR):
    """PaddleOCR-VL / PaddleOCR-VL-1.5 pipeline (PaddleOCR 3.x + PaddleX)."""

    name = "PaddleOCR-VL"

    def is_markdown_primary(self) -> bool:
        return True

    def __init__(self, use_gpu: bool = True, pipeline_version: str | None = None):
        """
        Initialize PaddleOCR-VL.

        Args:
            use_gpu: Whether to use GPU when CUDA is available
            pipeline_version: ``v1`` or ``v1.5`` (default: env ``PADDLEOCR_VL_PIPELINE_VERSION`` or ``v1.5``).

        Inference uses synchronous ``predict(..., use_queues=False)`` by default; set
        ``PADDLEOCR_VL_USE_QUEUES=1`` to match PaddleX’s threaded pipeline (may fail on CPU).
        """
        super().__init__()
        self.use_gpu = use_gpu
        pv = (
            (pipeline_version or os.environ.get("PADDLEOCR_VL_PIPELINE_VERSION") or "v1.5")
            .strip()
            .lower()
        )
        if pv in ("1.5", "v1.5"):
            self.pipeline_version = "v1.5"
        elif pv in ("1", "v1"):
            self.pipeline_version = "v1"
        else:
            self.pipeline_version = "v1.5"
        self._pipeline = None
        self._output_dir = None  # Will be set by runner
        self._runtime_gpu: bool = False

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self._runtime_gpu if self._is_loaded else self.use_gpu

    def set_output_dir(self, output_dir: Path) -> None:
        """Set the output directory for JSON/markdown files."""
        self._output_dir = Path(output_dir)

    def load_model(self) -> None:
        """Load PaddleOCR-VL model."""
        from paddleocr import PaddleOCRVL as PaddleVL

        device, self._runtime_gpu = resolve_paddle_device(self.use_gpu)

        self._pipeline = PaddleVL(
            pipeline_version=self.pipeline_version,
            device=device,
            vl_rec_backend="native",
            enable_mkldnn=paddleocr_enable_mkldnn(),
        )
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run PaddleOCR-VL on an image."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Avoid default ``use_queues: True`` (PaddleX YAML): threaded VLM hides real errors behind
        # ``RuntimeError: Exception from the 'vlm' worker:``. Sync path is slower but reliable.
        result = self._pipeline.predict(img_array, use_queues=paddleocr_vl_use_queues())

        if result is None:
            return ""

        nd = self.native_page_dir()
        if nd is not None:
            save_path = str(nd / "paddleocr-vl")
        elif self._output_dir is not None:
            save_path = str(self._output_dir / f"paddleocr-vl_page_{self._page_index}")
        else:
            save_path = "output"
        
        for res in result:
            res.save_to_json(save_path=save_path)
            res.save_to_markdown(save_path=save_path)

        all_text: list[str] = []
        save_p = Path(save_path)
        if save_p.is_dir():
            for md_file in sorted(save_p.rglob("*.md")):
                try:
                    t = md_file.read_text(encoding="utf-8").strip()
                    if t:
                        all_text.append(t)
                except Exception:
                    pass

        if not all_text:
            for res in result:
                if hasattr(res, "text") and res.text:
                    all_text.append(str(res.text))
                elif hasattr(res, "rec_text") and res.rec_text:
                    all_text.append(str(res.rec_text))
                else:
                    all_text.append(str(res))

        return "\n\n".join(all_text) if all_text else ""

