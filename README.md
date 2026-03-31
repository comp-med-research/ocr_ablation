# OCR Ablation Study

A framework for comparing multiple OCR engines on PDF documents.

## Supported OCR Models

| Model | Description |
|-------|-------------|
| **Tesseract** | Classic open-source OCR engine |
| **PaddleOCR** | Baidu's multilingual OCR toolkit |
| **Nougat** | Meta's layout-aware academic document OCR |
| **DocTR** | Mindee's document text recognition |
| **RolmOCR** | Reducto's vision-language OCR model |
| **DeepSeek** | DeepSeek-VL2 vision-language model for OCR |
| **Donut** | Naver's Document Understanding Transformer |
| **PP-StructureV3** | PaddlePaddle's document structure analysis |
| **PaddleOCR-VL** | PaddlePaddle's vision-language OCR model |
| **Docling** | Document conversion library for gen AI (DS4SD) |
| **Marker** | Fast PDF/image to Markdown (Datalab) |
| **qwen35-vl-9b** / **qwen35-9b** | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (`QWEN35_MODEL_ID` / legacy `QWEN35_VL_MODEL_ID` to override); install `requirements-qwen35.txt` or `pip install -U "transformers>=4.57.0"` (see model card if PyPI lags) |
| **llama32-vl-11b** | Meta Llama 3.2 11B Vision Instruct (gated Hub repo; needs HF token) |

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

### 2. Python virtual environment (recommended)

From the `ocr_ablation` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional Streamlit demo: `pip install -r requirements-demo.txt`

Optional **vLLM** (RolmOCR fast GPU path only): see `requirements-vllm.txt`. It is not in the main requirements file because some vLLM builds pin a Torch version your index may not ship.

The `.venv/` directory is gitignored.

**Dedicated environments (fewer dependency clashes):**

| Venv (example) | Requirements file | Typical models |
|----------------|-------------------|----------------|
| `.venv-marker` | `requirements-venv-marker.txt` | **marker** — isolated from MinerU’s `Pillow>=11` and from a crowded HF `.venv` |
| `.venv-ocr-stack` | `requirements-venv-ocr-stack.txt` | **mineru**, **paddleocr**, **deepseek**, **ppstructure**, **paddleocr-vl** |
| `.venv-deepseek` | `requirements-venv-deepseek.txt` | **deepseek** only (strict `transformers` pin) |

**ppstructure** and **paddleocr-vl** need PaddleX’s OCR optional dependencies (`paddlex[ocr]`); those requirement files include it so `pip install -r …` pulls them in.

### 3. Install Python Dependencies (without venv)

If you prefer a global or other environment:

```bash
pip install -r requirements.txt
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

## Benchmarking methodology (full-page OCR vs ground truth)

This project is set up to benchmark **whole-page** recognition, not crop-based OCR.

- **Inference:** Each model sees the **entire page** (one full rendered PDF page or one LS task image). We **do not** crop Label Studio rectangles and send crops through the models; crops exist only in the **ground truth** for scoring and analysis.
- **Ground truth:** A Label Studio export encodes **regions** (boxes, transcriptions, labels). Those regions define **evaluation units** (e.g. per-region NED), not model inputs.
- **Model outputs:** Prefer saving **native** artifacts when a library supports them (Markdown + JSON for layout pipelines, TSV/hOCR/ALTO for Tesseract, DocTR structured `export()`, etc.), in addition to any flat text used for quick comparisons. Formats can differ by model; evaluation code should consume them as needed (tables and reading order later; **text/NED** today via `run_text_eval.py`).
- **Alignment:** Evaluation takes **one full-page prediction** per task and **matches** it to GT regions with shared normalization and **ordered merging** of prediction paragraphs (OmniDocBench-style fairness for segmentation). See `python run_text_eval.py --help` and reports under `eval_reports/`.
- **Stratified NED:** After writing `task_*_alignment.json`, join manifest `choices` (e.g. `text_type`, `region_type`) and aggregate mean/micro NED per stratum: `python eval/stratify_ned.py --manifest gt_manifest.json --reports-dir results/evaluations/eval_reports_docling_layout --out-csv stratified_long.csv --out-json stratified_summary.json --out-txt stratified_tables.txt`. Use `--task-data-keys` for extra task-level fields from manifest `data`. See `python eval/stratify_ned.py --help`.

**Label Studio task id vs PDF runs:** `run_ocr_ablation.py` on a multi-page PDF writes one concatenated `{model}_output.txt` for the whole run. For LS you need **one prediction file per page/task** (each file = full-page model output for that image). Use a `--pred-map` JSON (`task_id` → path) or per-task files under `--pred-dir`; that is about **dataset alignment**, not cropping.

## Relation to [OmniDocBench](https://github.com/opendatalab/OmniDocBench)

[OmniDocBench](https://github.com/opendatalab/OmniDocBench) (CVPR 2025) is the public reference for **fair document parsing evaluation**: rich page JSON (`layout_dets`, polys, `ignore`, reading `order`, table HTML/LaTeX, formulas), **end-to-end** scoring on **full-page Markdown** predictions (one `.md` per page, filename aligned to the image), and config-driven runs via `pdf_validation.py`. It implements **normalized edit distance**, **TEDS** (tables), **CDM** (formulas), BLEU/METEOR, layout COCO-style detection, etc., plus matching modes such as `no_split`, `simple_match`, and **`quick_match`** (adjacency merge/split; v1.5 adds **hybrid** text/formula matching).

**How this repo lines up**

| OmniDocBench | Here (`eval/` + `run_text_eval.py`) |
|--------------|-------------------------------------|
| GT: `OmniDocBench.json` with categories, `ignore`, reading order | GT: Label Studio export → `gt_manifest.json` (regions + transcriptions) |
| Pred: directory of **page-level `.md`** files | Pred: **one text/Markdown file per LS task** (full page), same scoring idea |
| Extraction order (tables, formulas, code, then text) + normalization | Lighter **shared normalization** in `eval/normalize.py` (extensible) |
| `quick_match` / hybrid matching | **Ordered DP** merging adjacent **prediction paragraphs** to each GT region (NED-focused) |
| Tables, formulas, reading order in config | **Planned**; text/NED first |

**What is worth reusing from upstream**

- **Methodology:** Their preprocessing and **ignore** logic for headers/footers/captions (see paper/repo) when you add comparable tags in LS.
- **Metrics:** When you evaluate **tables**, pull in **TEDS** and HTML/LaTeX handling (they use LaTeXML for LaTeX tables; see their README).
- **Running their benchmark:** Clone the repo, install [`requirements.txt`](https://github.com/opendatalab/OmniDocBench/blob/main/requirements.txt), use their **`configs/`** and **`python pdf_validation.py --config ...`**; optional Docker image `sunyuefeng/omnidocbench-env:v1.5` per their docs.
- **Aligning with their file layout:** If you standardize outputs as **`<image_stem>.md` per page**, you stay close to their **prediction** convention and can compare notes with leaderboard-style workflows (`tools/generate_result_tables.ipynb`).

We do **not** bundle OmniDocBench; keep it as a **dependency or side-by-side checkout** if you use their dataset or full metric stack. Our code is intentionally smaller and **LS-ground-truth-centric** until you export to their JSON or adopt their pages.

### Streamlit demo (text matching + NED)

Interactive examples for **normalization**, **paragraph splitting**, **DP alignment**, and **per-region diffs**:

```bash
pip install -r requirements-demo.txt
cd ocr_ablation
streamlit run streamlit_demo/app.py
```

Built-in scenarios need no data files; you can also upload a Label Studio export and paste full-page prediction text for one `task_id`.

**Alignment explorer** (markdown vs Docling JSON layout, real `gt_manifest` + pred paths, cost matrix, NED math, bbox figure):

```bash
cd ocr_ablation
streamlit run streamlit_alignment_explorer.py
```

The **Page images directory** field defaults to ``test_cases15``; manifest paths like ``…/uuid-00000725.jpg`` are resolved to ``test_cases15/00000725.jpg``.

**Text match modes** (markdown tab, Step D): **simple** — one-to-one Hungarian on NED, unmatched segments ignored. **quick** — OmniDocBench ``match_quick`` (transformed cost matrix + merge/split + fuzzy; multiple segments per GT possible). **full** — OmniDocBench ``match_full`` / FuzzyMatch (substring combine; empty segments dropped). They can yield different NEDs per task because NED depends on the induced alignment, not because one mode is always “more accurate.”

If Linux logs **inotify watch limit reached**, the repo includes ``.streamlit/config.toml`` that disables the file watcher (refresh the browser after edits), or run: ``STREAMLIT_SERVER_FILE_WATCHER_TYPE=none streamlit run …``. Optionally raise system limits: ``sudo sysctl fs.inotify.max_user_watches=524288``.

## Example Results

```
+------------+--------+-----+-------------+-----------------+--------+---------+--------+
| Model      | Status | GPU | Total Time  | Time/Page (μ±σ) | RAM MB | VRAM MB | Chars  |
+============+========+=====+=============+=================+========+=========+========+
| Tesseract  | ✓      | No  | 2.34s       | 0.78s ± 0.12s   | 245    | -       | 12,456 |
| PaddleOCR  | ✓      | No  | 3.21s       | 1.07s ± 0.18s   | 892    | -       | 12,789 |
| DocTR      | ✓      | No  | 4.56s       | 1.52s ± 0.24s   | 1,204  | -       | 12,234 |
| Nougat     | ✓      | Yes | 8.12s       | 2.71s ± 0.45s   | 2,456  | 4,521   | 13,102 |
| DeepSeek   | ✓      | Yes | 6.45s       | 2.15s ± 0.32s   | 3,102  | 6,234   | 13,456 |
+------------+--------+-----+-------------+-----------------+--------+---------+--------+
```

- **Time/Page (μ±σ)**: Mean time per page ± standard deviation
- **RAM MB**: Peak RAM memory usage during processing
- **VRAM MB**: Peak GPU memory usage (if applicable)

The `results.json` file includes full timing and memory statistics:

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
  "memory": {
    "peak_ram_mb": 3102.4,
    "peak_gpu_mb": 6234.1
  },
  "total_chars": 13456,
  "output_file": "ocr_results_20260116/deepseek_output.txt"
}
```

## Privacy & data locality

**All models run fully on your machine.** No documents, images, or text are sent to any external server. Weights are downloaded once (e.g. from HuggingFace) and cached locally; all OCR inference runs locally.

## Notes

- **GPU Acceleration**: Nougat, RolmOCR, and DeepSeek benefit significantly from GPU
- **Memory**: Some models (especially Nougat, RolmOCR, DeepSeek) require significant RAM/VRAM
- **First Run**: Models download weights on first use (may take time)
- **Docling, Marker**: Install via `requirements.txt`. Marker uses CLI tools (`marker_single` or `marker`).

## License

MIT

