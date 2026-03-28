"""PaddleOCR implementation."""

import json

from PIL import Image
import numpy as np

from . import paddle_utils  # noqa: F401 — env flags before paddle
from .base import BaseOCR
from .paddle_utils import (
    ocr_version_from_env,
    paddleocr_enable_mkldnn,
    paddleocr_verbose,
    resolve_paddle_device,
)


class PaddleOCRModel(BaseOCR):
    """PaddleOCR engine wrapper (PaddleOCR 3.x; default recognition stack **PP-OCRv5**)."""

    name = "PaddleOCR"

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        ocr_version: str | None = None,
    ):
        """
        Initialize PaddleOCR.

        Args:
            lang: Language code ('en', 'ch', 'fr', etc.)
            use_gpu: Whether to use GPU when CUDA is available
            ocr_version: e.g. ``PP-OCRv5``, ``PP-OCRv4``. Default: env ``PADDLEOCR_OCR_VERSION`` or ``PP-OCRv5``.
        """
        super().__init__()
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_version = ocr_version if ocr_version is not None else ocr_version_from_env()
        self._runtime_gpu: bool = False

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self._runtime_gpu if self._is_loaded else self.use_gpu

    def load_model(self) -> None:
        """Load PaddleOCR model."""
        from paddleocr import PaddleOCR

        device, self._runtime_gpu = resolve_paddle_device(self.use_gpu)

        self._model = PaddleOCR(
            lang=self.lang,
            device=device,
            ocr_version=self.ocr_version,
            enable_mkldnn=paddleocr_enable_mkldnn(),
        )
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run PaddleOCR on an image."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run OCR
        result = self._model.predict(img_array)

        # Extract text from results
        if result is None or len(result) == 0:
            self.write_native_text("", "paddleocr.txt")
            nd = self.native_page_dir()
            if nd is not None:
                (nd / "paddleocr_predict.json").write_text("[]", encoding="utf-8")
            return ""

        nd = self.native_page_dir()
        verbose_dir: str | None = None
        if nd is not None and paddleocr_verbose():
            vdir = nd / "paddleocr_verbose"
            vdir.mkdir(parents=True, exist_ok=True)
            verbose_dir = str(vdir)

        lines = []
        serial = []
        for res in result:
            if paddleocr_verbose():
                pr = getattr(res, "print", None)
                if callable(pr):
                    try:
                        pr()
                    except Exception:
                        pass
                if verbose_dir is not None:
                    for attr in ("save_to_img", "save_to_json"):
                        fn = getattr(res, attr, None)
                        if callable(fn):
                            try:
                                fn(verbose_dir)
                            except Exception:
                                pass

            text = _rec_texts_from_result(res)
            for t in text:
                lines.append(t if isinstance(t, str) else str(t))
            try:
                serial.append(_serialize_result_for_json(res))
            except Exception:
                serial.append({"repr": repr(res)})

        out = "\n".join(lines)
        self.write_native_text(out, "paddleocr.txt")
        if nd is not None:
            (nd / "paddleocr_predict.json").write_text(
                json.dumps(serial, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return out


def _rec_texts_from_result(res) -> list:
    text = None
    if isinstance(res, dict):
        text = res.get("rec_texts")
    else:
        text = getattr(res, "rec_texts", None)
        if text is None:
            j = _result_as_dict(res)
            if isinstance(j, dict):
                text = j.get("rec_texts")
    if text is None:
        text = []
    if not isinstance(text, (list, tuple)):
        text = [text] if text else []
    return list(text)


def _result_as_dict(res):
    if isinstance(res, dict):
        return res
    j = getattr(res, "json", None)
    if callable(j):
        try:
            return j()
        except Exception:
            return None
    if j is not None and isinstance(j, dict):
        return j
    return None


def _serialize_result_for_json(res):
    if isinstance(res, dict):
        return {k: _jsonify(v) for k, v in res.items()}
    d = _result_as_dict(res)
    if isinstance(d, dict):
        return {k: _jsonify(v) for k, v in d.items()}
    return _jsonify(res)


def _jsonify(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)

