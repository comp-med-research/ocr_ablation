"""Tesseract OCR implementation."""

from PIL import Image
import pytesseract

from .base import BaseOCR


class TesseractOCR(BaseOCR):
    """Tesseract OCR engine wrapper."""

    name = "Tesseract"

    def __init__(self, lang: str = "eng", config: str = ""):
        """
        Initialize Tesseract OCR.
        
        Args:
            lang: Language code (e.g., 'eng', 'fra', 'deu')
            config: Additional Tesseract config options
        """
        super().__init__()
        self.lang = lang
        self.config = config

    def load_model(self) -> None:
        """Tesseract doesn't require explicit model loading."""
        # Verify tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            self._is_loaded = True
        except Exception as e:
            raise RuntimeError(
                "Tesseract is not installed. Install with: "
                "brew install tesseract (macOS) or apt install tesseract-ocr (Linux)"
            ) from e

    def _run_ocr(self, image: Image.Image) -> str:
        """Run Tesseract OCR on an image."""
        text = pytesseract.image_to_string(
            image,
            lang=self.lang,
            config=self.config
        )
        text = text.strip()
        nd = self.native_page_dir()
        if nd is not None:
            (nd / "tesseract.txt").write_text(text, encoding="utf-8")
            try:
                tsv = pytesseract.image_to_data(
                    image, lang=self.lang, config=self.config, output_type=pytesseract.Output.STRING
                )
                (nd / "tesseract.tsv").write_text(tsv, encoding="utf-8")
            except Exception:
                pass
            try:
                hocr = pytesseract.image_to_pdf_or_hocr(
                    image, extension="hocr", lang=self.lang, config=self.config
                )
                if isinstance(hocr, (bytes, bytearray)):
                    (nd / "tesseract.hocr").write_bytes(hocr)
                else:
                    (nd / "tesseract.hocr").write_text(str(hocr), encoding="utf-8")
            except Exception:
                pass
        return text

