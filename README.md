# OCR Ablation Study

A framework for comparing multiple OCR engines on PDF documents.

## Supported OCR Models

| Model | Description |
|-------|-------------|
| **Tesseract** | Classic open-source OCR engine |
| **PaddleOCR** | Baidu's multilingual OCR toolkit |
| **Kraken** | Specialized for historical documents |
| **Nougat** | Meta's layout-aware academic document OCR |
| **DocTR** | Mindee's document text recognition |
| **TrOCR** | Microsoft's transformer-based OCR |
| **RolmOCR** | Reducto's vision-language OCR model |

## Installation

### 1. Install System Dependencies

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr poppler-utils
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Kraken Models (optional)

```bash
kraken get default
```

## Usage

### Run All Models

```bash
python run_ocr_ablation.py your_document.pdf
```

### Run Specific Models

```bash
python run_ocr_ablation.py document.pdf --models tesseract paddleocr doctr
```

### Process Specific Pages

```bash
python run_ocr_ablation.py document.pdf --pages 0 1 2
```

### Custom Output Directory

```bash
python run_ocr_ablation.py document.pdf -o ./results
```

### List Available Models

```bash
python run_ocr_ablation.py --list-models
```

## Output

Results are saved to the output directory:

```
ocr_results_<timestamp>/
├── results.json           # Summary with timing metrics
├── sample_page_0.png      # Sample rendered page
├── tesseract_output.txt   # Tesseract OCR output
├── paddleocr_output.txt   # PaddleOCR output
├── doctr_output.txt       # DocTR output
└── ...
```

## Example Results

```
+------------+--------+-------------+-----------+---------------+
| Model      | Status | Total Time  | Time/Page | Output Chars  |
+============+========+=============+===========+===============+
| Tesseract  | ✓      | 2.34s       | 0.78s     | 12,456        |
| PaddleOCR  | ✓      | 3.21s       | 1.07s     | 12,789        |
| DocTR      | ✓      | 4.56s       | 1.52s     | 12,234        |
| Nougat     | ✓      | 15.23s      | 5.08s     | 13,102        |
+------------+--------+-------------+-----------+---------------+
```

## Notes

- **GPU Acceleration**: Nougat, TrOCR, and RolmOCR benefit significantly from GPU
- **Memory**: Some models (especially Nougat, RolmOCR) require significant RAM/VRAM
- **First Run**: Models download weights on first use (may take time)

## License

MIT

