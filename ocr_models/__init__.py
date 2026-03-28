# OCR Models Package
from .base import BaseOCR
from .tesseract_ocr import TesseractOCR
from .paddleocr_model import PaddleOCRModel
from .nougat_ocr import NougatOCR
from .doctr_ocr import DocTROCR
from .rolm_ocr import RolmOCR
from .deepseek_ocr import DeepSeekOCR
from .donut_ocr import DonutOCR
from .pp_structure_ocr import PPStructureOCR
from .paddleocr_vl import PaddleOCRVL
from .docling_ocr import DoclingOCR
from .marker_ocr import MarkerOCR
from .glm_ocr import GlmOcrModel
from .chandra_ocr import ChandraOCR
from .mineru_ocr import MinerUOCR
from .vlm_hf_ocr import Llama32VisionOCR, Qwen35VLOCR

__all__ = [
    "BaseOCR",
    "TesseractOCR",
    "PaddleOCRModel",
    "NougatOCR",
    "DocTROCR",
    "RolmOCR",
    "DeepSeekOCR",
    "DonutOCR",
    "PPStructureOCR",
    "PaddleOCRVL",
    "DoclingOCR",
    "MarkerOCR",
    "GlmOcrModel",
    "ChandraOCR",
    "MinerUOCR",
    "Qwen35VLOCR",
    "Llama32VisionOCR",
]

