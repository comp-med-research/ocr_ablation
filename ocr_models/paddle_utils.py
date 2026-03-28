"""Shared PaddlePaddle / PaddleOCR 3.x environment and device helpers."""

from __future__ import annotations

import os

# Import this module before ``import paddle`` to reduce oneDNN / PIR issues on some Linux builds.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")


def resolve_paddle_device(use_gpu: bool) -> tuple[str, bool]:
    """
    Return ``(device, actually_gpu)`` for PaddleOCR 3.x / PaddleX pipelines.

    ``device`` is ``\"gpu\"`` or ``\"cpu\"`` as expected by ``device=`` kwargs on pipelines.
    """
    import paddle

    if hasattr(paddle, "set_flags"):
        try:
            paddle.set_flags({"FLAGS_use_mkldnn": False})
        except (TypeError, ValueError, KeyError):
            pass

    try:
        has_cuda = paddle.device.cuda.device_count() > 0
    except Exception:
        has_cuda = False

    on_gpu = bool(use_gpu and has_cuda)
    return ("gpu" if on_gpu else "cpu"), on_gpu


def ocr_version_from_env(default: str = "PP-OCRv5") -> str:
    """``PADDLEOCR_OCR_VERSION`` for ``PaddleOCR`` / ``PPStructureV3`` (e.g. ``PP-OCRv5``, ``PP-OCRv4``)."""
    v = (os.environ.get("PADDLEOCR_OCR_VERSION") or default).strip()
    return v


def paddleocr_enable_mkldnn() -> bool:
    """
    Whether PaddleOCR 3.x / PaddleX pipelines should use Intel oneDNN on CPU.

    Default **False**: PaddlePaddle 3.3+ can crash on CPU with
    ``ConvertPirAttribute2RuntimeAttribute`` in the PIR→oneDNN path; setting
    ``FLAGS_use_mkldnn=0`` does **not** fix it because PaddleX picks ``run_mode=mkldnn``
    independently. Pass ``enable_mkldnn=False`` into pipeline constructors instead.
    See https://github.com/PaddlePaddle/Paddle/issues/77340

    Opt-in (e.g. older ``paddlepaddle==3.2.2`` or GPU-only): ``PADDLEOCR_ENABLE_MKLDNN=1``.
    """
    v = (os.environ.get("PADDLEOCR_ENABLE_MKLDNN") or "").strip().lower()
    return v in ("1", "true", "yes")


def paddleocr_vl_use_queues() -> bool:
    """
    PaddleOCR-VL-1.5 defaults to ``use_queues: True`` in PaddleX, which runs the VLM in a
    background thread; failures often appear as ``Exception from the 'vlm' worker:`` with no
    message. Default **False** (synchronous ``predict``) for reliable CPU/GPU runs.
    Opt-in: ``PADDLEOCR_VL_USE_QUEUES=1``.
    """
    v = (os.environ.get("PADDLEOCR_VL_USE_QUEUES") or "").strip().lower()
    return v in ("1", "true", "yes")


def paddleocr_verbose() -> bool:
    """
    If True, each ``predict`` result is passed to ``print()``, ``save_to_img()``, and
    ``save_to_json()`` when those methods exist (PaddleOCR 3.x), mirroring the upstream
    examples. Set ``PADDLEOCR_VERBOSE=1``.
    """
    v = (os.environ.get("PADDLEOCR_VERBOSE") or "").strip().lower()
    return v in ("1", "true", "yes")
