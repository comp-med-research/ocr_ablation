"""PP-StructureV3 implementation - PaddlePaddle's document structure analysis."""

from pathlib import Path

from PIL import Image
import numpy as np

from . import paddle_utils  # noqa: F401
from .base import BaseOCR
from .paddle_utils import ocr_version_from_env, paddleocr_enable_mkldnn, resolve_paddle_device


class PPStructureOCR(BaseOCR):
    """PP-StructureV3 — layout + OCR (PaddleOCR 3.x; default text stack **PP-OCRv5**)."""

    name = "PP-StructureV3"

    def is_markdown_primary(self) -> bool:
        return True

    def __init__(
        self,
        use_gpu: bool = True,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        ocr_version: str | None = None,
    ):
        """
        Initialize PP-StructureV3.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            use_doc_orientation_classify: Whether to classify document orientation
            use_doc_unwarping: Whether to unwarp curved documents
            ocr_version: e.g. ``PP-OCRv5``. Default: env ``PADDLEOCR_OCR_VERSION`` or ``PP-OCRv5``.
        """
        super().__init__()
        self.use_gpu = use_gpu
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.ocr_version = ocr_version if ocr_version is not None else ocr_version_from_env()
        self._output_dir = None
        self._runtime_gpu: bool = False

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self._runtime_gpu if self._is_loaded else self.use_gpu

    def set_output_dir(self, output_dir: Path) -> None:
        """Set the output directory for JSON/markdown files."""
        self._output_dir = Path(output_dir)

    def load_model(self) -> None:
        """Load PP-StructureV3 model."""
        from paddleocr import PPStructureV3

        device, self._runtime_gpu = resolve_paddle_device(self.use_gpu)

        self._model = PPStructureV3(
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping,
            ocr_version=self.ocr_version,
            device=device,
            enable_mkldnn=paddleocr_enable_mkldnn(),
        )
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run PP-StructureV3 on an image."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run structure analysis using predict method
        result = self._model.predict(img_array)

        if result is None:
            return ""

        # Prefer per-page native folder; else legacy flat output dir
        nd = self.native_page_dir()
        if nd is not None:
            save_path = str(nd / "ppstructure")
        elif self._output_dir is not None:
            save_path = str(self._output_dir / f"ppstructure_page_{self._page_index}")
        else:
            save_path = "output"

        # Save results (JSON and markdown)
        for res in result:
            res.save_to_json(save_path=save_path)
            res.save_to_markdown(save_path=save_path)

        # Collect markdown written under ``save_path`` (recursive; paths are directories)
        all_text = []
        save_p = Path(save_path)
        if save_p.is_dir():
            md_files = sorted(save_p.rglob("*.md"))
            for md_file in md_files:
                try:
                    content = md_file.read_text(encoding="utf-8")
                    if content.strip():
                        all_text.append(content)
                except Exception:
                    pass

        # Fallback: try to extract text directly from result
        if not all_text:
            for res in result:
                if hasattr(res, 'text'):
                    all_text.append(res.text)
                elif hasattr(res, 'get_text'):
                    all_text.append(res.get_text())
                else:
                    all_text.append(str(res))

        return "\n".join(all_text) if all_text else ""

