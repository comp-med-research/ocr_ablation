# OCR Models Package
from .base import BaseOCR
from .tesseract_ocr import TesseractOCR
from .paddleocr_model import PaddleOCRModel
from .nougat_ocr import NougatOCR
from .doctr_ocr import DocTROCR
from .trocr_model import TrOCRModel
from .rolm_ocr import RolmOCR
from .deepseek_ocr import DeepSeekOCR
from .donut_ocr import DonutOCR
from .pp_structure_ocr import PPStructureOCR
from .paddleocr_vl import PaddleOCRVL
from .docling_ocr import DoclingOCR
from .marker_ocr import MarkerOCR
from .mineru_ocr import MinerUOCR

__all__ = [
    "BaseOCR",
    "TesseractOCR",
    "PaddleOCRModel",
    "NougatOCR",
    "DocTROCR",
    "TrOCRModel",
    "RolmOCR",
    "DeepSeekOCR",
    "DonutOCR",
    "PPStructureOCR",
    "PaddleOCRVL",
    "DoclingOCR",
    "MarkerOCR",
    "MinerUOCR",
]

