"""PP-StructureV3 implementation - PaddlePaddle's document structure analysis."""

import glob
from pathlib import Path
from PIL import Image
import numpy as np

from .base import BaseOCR


class PPStructureOCR(BaseOCR):
    """PP-StructureV3 - PaddlePaddle's document structure analysis and OCR."""

    name = "PP-StructureV3"

    def __init__(
        self,
        use_gpu: bool = True,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False
    ):
        """
        Initialize PP-StructureV3.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            use_doc_orientation_classify: Whether to classify document orientation
            use_doc_unwarping: Whether to unwarp curved documents
        """
        super().__init__()
        self.use_gpu = use_gpu
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self._output_dir = None
        self._page_counter = 0

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self.use_gpu

    def set_output_dir(self, output_dir: Path) -> None:
        """Set the output directory for JSON/markdown files."""
        self._output_dir = Path(output_dir)
        self._page_counter = 0

    def load_model(self) -> None:
        """Load PP-StructureV3 model."""
        from paddleocr import PPStructureV3

        self._model = PPStructureV3(
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping
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

        # Determine output path
        if self._output_dir is not None:
            save_path = str(self._output_dir / f"ppstructure_page_{self._page_counter}")
        else:
            save_path = "output"

        # Save results (JSON and markdown)
        for res in result:
            res.save_to_json(save_path=save_path)
            res.save_to_markdown(save_path=save_path)

        # Read the markdown file to get the actual text content
        all_text = []
        md_files = glob.glob(f"{save_path}*.md")
        if md_files:
            for md_file in sorted(md_files):
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
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

        self._page_counter += 1

        return "\n".join(all_text) if all_text else ""

