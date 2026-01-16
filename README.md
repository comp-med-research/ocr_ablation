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
| **DeepSeek** | DeepSeek-VL2 vision-language model for OCR |

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
python run_ocr_ablation.py document.pdf --models tesseract paddleocr deepseek
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
+------------+--------+-----+-------------+-----------------+---------------+--------+
| Model      | Status | GPU | Total Time  | Time/Page (μ±σ) | IQR [p25,p75] | Chars  |
+============+========+=====+=============+=================+===============+========+
| Tesseract  | ✓      | No  | 2.34s       | 0.78s ± 0.12s   | [0.68, 0.85]  | 12,456 |
| PaddleOCR  | ✓      | No  | 3.21s       | 1.07s ± 0.18s   | [0.92, 1.15]  | 12,789 |
| DocTR      | ✓      | No  | 4.56s       | 1.52s ± 0.24s   | [1.35, 1.68]  | 12,234 |
| Nougat     | ✓      | Yes | 8.12s       | 2.71s ± 0.45s   | [2.42, 2.98]  | 13,102 |
| DeepSeek   | ✓      | Yes | 6.45s       | 2.15s ± 0.32s   | [1.92, 2.38]  | 13,456 |
+------------+--------+-----+-------------+-----------------+---------------+--------+
```

- **Time/Page (μ±σ)**: Mean time per page ± standard deviation
- **IQR [p25,p75]**: Interquartile range (25th and 75th percentiles)

The `results.json` file includes full timing statistics:

```json
{
  "model": "DeepSeek",
  "status": "success",
  "uses_gpu": true,
  "pages_processed": 3,
  "total_time_seconds": 6.45,
  "time_per_page": {
    "mean": 2.15,
    "std": 0.32,
    "min": 1.78,
    "max": 2.52,
    "p25": 1.92,
    "p75": 2.38
  },
  "page_times": [1.78, 2.15, 2.52],
  "total_chars": 13456,
  "output_file": "ocr_results_20260116/deepseek_output.txt"
}
```

## Notes

- **GPU Acceleration**: Nougat, TrOCR, RolmOCR, and DeepSeek benefit significantly from GPU
- **Memory**: Some models (especially Nougat, RolmOCR, DeepSeek) require significant RAM/VRAM
- **First Run**: Models download weights on first use (may take time)
- **Privacy**: All inference runs locally. No images or text are sent to external servers.

## License

MIT

