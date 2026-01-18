"""PP-StructureV3 implementation - PaddlePaddle's document structure analysis."""

from PIL import Image
import numpy as np

from .base import BaseOCR


class PPStructureOCR(BaseOCR):
    """PP-StructureV3 - PaddlePaddle's document structure analysis and OCR."""

    name = "PP-StructureV3"

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        recovery: bool = True,
        table: bool = True,
        layout: bool = True
    ):
        """
        Initialize PP-StructureV3.
        
        Args:
            lang: Language code ('en', 'ch', etc.)
            use_gpu: Whether to use GPU acceleration
            recovery: Whether to enable document recovery (convert to docx-like structure)
            table: Whether to enable table recognition
            layout: Whether to enable layout analysis
        """
        super().__init__()
        self.lang = lang
        self.use_gpu = use_gpu
        self.recovery = recovery
        self.table = table
        self.layout = layout

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return self.use_gpu

    def load_model(self) -> None:
        """Load PP-StructureV3 model."""
        from paddleocr import PPStructure

        self._model = PPStructure(
            lang=self.lang,
            recovery=self.recovery,
            table=self.table,
            layout=self.layout,
            show_log=False
        )
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run PP-StructureV3 on an image."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run structure analysis
        result = self._model(img_array)

        if result is None or len(result) == 0:
            return ""

        # Extract text from structured results
        return self._extract_text_from_structure(result)

    def _extract_text_from_structure(self, result: list) -> str:
        """Extract text from PP-Structure results maintaining document order."""
        text_blocks = []

        for item in result:
            item_type = item.get("type", "")
            
            if item_type == "table":
                # Handle table content
                table_text = self._extract_table_text(item)
                if table_text:
                    text_blocks.append(table_text)
            elif item_type == "figure":
                # Skip figures or add placeholder
                pass
            else:
                # Handle text regions
                res = item.get("res", [])
                if isinstance(res, list):
                    for line in res:
                        if isinstance(line, dict):
                            text = line.get("text", "")
                            if text:
                                text_blocks.append(text)
                        elif isinstance(line, (list, tuple)) and len(line) >= 2:
                            # Format: [bbox, (text, confidence)]
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)):
                                text_blocks.append(str(text_info[0]))
                            else:
                                text_blocks.append(str(text_info))

        return "\n".join(text_blocks)

    def _extract_table_text(self, table_item: dict) -> str:
        """Extract text from a table structure."""
        res = table_item.get("res", {})
        
        # Try to get HTML table and convert to text
        html = res.get("html", "")
        if html:
            return self._html_table_to_text(html)
        
        # Fallback: extract cell text directly
        cell_texts = []
        cells = res.get("cells", [])
        for cell in cells:
            text = cell.get("text", "")
            if text:
                cell_texts.append(text)
        
        return " | ".join(cell_texts) if cell_texts else ""

    def _html_table_to_text(self, html: str) -> str:
        """Convert HTML table to plain text representation."""
        import re
        
        # Simple HTML table to text conversion
        # Remove tags but preserve structure
        text = html
        
        # Replace row endings
        text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
        # Replace cell separators
        text = re.sub(r'</td>|</th>', ' | ', text, flags=re.IGNORECASE)
        # Remove all other tags
        text = re.sub(r'<[^>]+>', '', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\|\s*\|', '|', text)
        text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*\|\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()

