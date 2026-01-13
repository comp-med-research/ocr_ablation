"""DocTR OCR implementation."""

from PIL import Image
import numpy as np

from .base import BaseOCR


class DocTROCR(BaseOCR):
    """DocTR - Document Text Recognition library by Mindee."""

    name = "DocTR"

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        pretrained: bool = True
    ):
        """
        Initialize DocTR.
        
        Args:
            det_arch: Detection architecture
            reco_arch: Recognition architecture
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.pretrained = pretrained

    def load_model(self) -> None:
        """Load DocTR model."""
        from doctr.models import ocr_predictor

        self._model = ocr_predictor(
            det_arch=self.det_arch,
            reco_arch=self.reco_arch,
            pretrained=self.pretrained
        )
        self._is_loaded = True

    def _run_ocr(self, image: Image.Image) -> str:
        """Run DocTR on an image."""
        from doctr.io import DocumentFile

        # Convert PIL to numpy array
        img_array = np.array(image)

        # DocTR expects a list of numpy arrays or a DocumentFile
        result = self._model([img_array])

        # Extract text from result
        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join(word.value for word in line.words)
                    lines.append(line_text)

        return "\n".join(lines)

