#!/usr/bin/env python3
"""
OCR Ablation Study Runner

Compare multiple OCR engines on PDF documents.
"""

import argparse
import json
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm
from tabulate import tabulate

# Optional: memory tracking
try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False

# Optional: GPU memory tracking
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ocr_models import (
    TesseractOCR,
    PaddleOCRModel,
    NougatOCR,
    DocTROCR,
    RolmOCR,
    DeepSeekOCR,
    DonutOCR,
    PPStructureOCR,
    PaddleOCRVL,
    DoclingOCR,
    MarkerOCR,
    GlmOcrModel,
    ChandraOCR,
    MinerUOCR,
    Qwen35VLOCR,
    Llama32VisionOCR,
)


def get_gpu_memory_mb() -> Optional[float]:
    """Get current GPU memory usage in MB. Returns None if not available."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None
    try:
        # Get memory for all GPUs, return max
        max_mem = 0
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.max_memory_allocated(i) / (1024 * 1024)  # Convert to MB
            max_mem = max(max_mem, mem)
        return max_mem
    except Exception:
        return None


def reset_gpu_memory_stats() -> None:
    """Reset GPU memory tracking statistics."""
    if HAS_TORCH and torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        except Exception:
            pass


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    import resource
    # Get memory in KB, convert to MB
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024  # Linux reports in KB


# Registry of available OCR models
OCR_MODELS = {
    "tesseract": TesseractOCR,
    "paddleocr": PaddleOCRModel,
    "paddleocr-v5": PaddleOCRModel,  # alias; default stack is PP-OCRv5 (see PADDLEOCR_OCR_VERSION)
    "nougat": NougatOCR,
    "doctr": DocTROCR,
    "rolmocr": RolmOCR,
    "deepseek": DeepSeekOCR,
    "donut": DonutOCR,
    "ppstructure": PPStructureOCR,
    "paddleocr-vl": PaddleOCRVL,
    "docling": DoclingOCR,
    "marker": MarkerOCR,
    "glm-ocr": GlmOcrModel,
    "chandra2": ChandraOCR,
    "mineru": MinerUOCR,
    "qwen35-vl-9b": Qwen35VLOCR,
    "qwen35-9b": Qwen35VLOCR,
    "llama32-vl-11b": Llama32VisionOCR,
}

# Per-page JSON saved next to output_dir by PP-Structure / PaddleOCR-VL (prefix + page index).
JSON_SIDECAR_GLOBS: dict[str, str] = {
    "ppstructure": "ppstructure_page_*.json",
    "paddleocr-vl": "paddleocr-vl_page_*.json",
}

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"})

# When using ``--images-dir`` without ``-m``, run this subset (override with ``-m``).
DEFAULT_IMAGES_DIR_MODELS = [
    "tesseract",
    "doctr",
    "docling",
    "donut",
]


def merge_json_sidecars(output_dir: Path, glob_pattern: str) -> str | None:
    """Load every matching JSON file and return a single JSON array string, or None if none found."""
    paths = sorted(output_dir.glob(glob_pattern), key=lambda p: p.name)
    if not paths:
        return None
    items: list = []
    for p in paths:
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            items.append({"_file": p.name, "_error": "invalid json"})
    return json.dumps(items, ensure_ascii=False, indent=2)


def write_combined_exports(
    model,
    model_name: str,
    output_dir: Path,
    full_text: str,
    *,
    write_combined_txt: bool,
    write_combined_md: bool,
    also_txt_for_markdown: bool,
) -> dict:
    """
    Write ``*_output.txt`` and/or ``*_output.md`` (or ``.mmd`` for Nougat) plus ``*_output.json``.

    Markdown-primary models default to a combined ``*_output.md`` (or model-specific extension
    such as ``.mmd``) only — no duplicate ``.txt`` unless ``also_txt_for_markdown`` is True.
    """
    slug = model_name.lower()
    md_primary = model.is_markdown_primary()
    out_paths: list[str] = []
    primary: str | None = None

    if md_primary:
        if write_combined_md:
            ext = model.combined_markdown_extension()
            p = output_dir / f"{slug}_output{ext}"
            p.write_text(full_text, encoding="utf-8")
            out_paths.append(str(p))
            primary = str(p)
        if also_txt_for_markdown and write_combined_txt:
            p = output_dir / f"{slug}_output.txt"
            p.write_text(full_text, encoding="utf-8")
            out_paths.append(str(p))
            if primary is None:
                primary = str(p)
    else:
        if write_combined_txt:
            p = output_dir / f"{slug}_output.txt"
            p.write_text(full_text, encoding="utf-8")
            out_paths.append(str(p))
            primary = str(p)

    json_path: str | None = None
    if hasattr(model, "get_json_string"):
        js = model.get_json_string()
        if js is not None:
            jp = output_dir / f"{slug}_output.json"
            jp.write_text(js, encoding="utf-8")
            json_path = str(jp)
    if json_path is None and slug in JSON_SIDECAR_GLOBS:
        merged = merge_json_sidecars(output_dir, JSON_SIDECAR_GLOBS[slug])
        if merged:
            jp = output_dir / f"{slug}_output.json"
            jp.write_text(merged, encoding="utf-8")
            json_path = str(jp)

    return {
        "output_files": out_paths,
        "output_file": primary,
        "output_json": json_path,
    }


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    """
    Convert PDF pages to PIL Images using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering
        
    Returns:
        List of PIL Images, one per page
    """
    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("PDF mode requires PyMuPDF: pip install PyMuPDF") from e

    images = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Create a matrix for the desired resolution
        zoom = dpi / 72  # 72 is the default PDF DPI
        matrix = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=matrix)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    doc.close()
    return images


def load_images_from_dir(dir_path: Path) -> tuple[list[Image.Image], list[str]]:
    """
    Load all images under ``dir_path``, sorted by filename.

    Returns:
        (images, basenames) with matching indices.
    """
    paths = sorted(
        p
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"No images in {dir_path} (allowed extensions: {sorted(IMAGE_EXTENSIONS)})"
        )
    images: list[Image.Image] = []
    names: list[str] = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
        names.append(p.name)
    return images, names


def run_ablation_on_images(
    image_dir: Path,
    output_dir: Path,
    models: Optional[list[str]] = None,
    native_exports: bool = True,
    write_combined_txt: bool = True,
    write_combined_md: bool = True,
    also_txt_for_markdown: bool = False,
) -> dict:
    """
    Run OCR on every image in a directory (sorted by name). No PDF / PyMuPDF required.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = list(DEFAULT_IMAGES_DIR_MODELS)
    else:
        for m in models:
            if m.lower() not in OCR_MODELS:
                print(f"Warning: Unknown model '{m}', skipping")
        models = [m for m in models if m.lower() in OCR_MODELS]

    images, source_names = load_images_from_dir(image_dir)
    page_indices = list(range(len(images)))

    print("OCR Ablation Study (image folder)")
    print("===================================")
    print(f"Image directory: {image_dir.resolve()}")
    print(f"Files: {len(images)}")
    print(f"Output: {output_dir}")
    print(f"Models: {', '.join(models)}")

    if images:
        safe0 = source_names[0].replace(".", "_")
        sample_path = output_dir / f"sample_{safe0}.png"
        images[0].save(sample_path)
        print(f"Sample saved: {sample_path}")

    results = {
        "metadata": {
            "image_dir": str(image_dir.resolve()),
            "source_files": source_names,
            "timestamp": datetime.now().isoformat(),
            "total_pages": len(images),
            "processed_pages": len(images),
            "dpi": None,
            "native_exports": native_exports,
            "write_combined_txt": write_combined_txt,
            "write_combined_md": write_combined_md,
            "also_txt_for_markdown": also_txt_for_markdown,
        },
        "results": [],
    }

    for model_name in models:
        result = run_single_model(
            model_name,
            images,
            output_dir,
            pdf_path=None,
            pages=None,
            page_indices=page_indices,
            native_exports=native_exports,
            write_combined_txt=write_combined_txt,
            write_combined_md=write_combined_md,
            also_txt_for_markdown=also_txt_for_markdown,
        )
        results["results"].append(result)

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results, output_dir)
    return results


def run_single_model(
    model_name: str,
    images: list[Image.Image],
    output_dir: Path,
    pdf_path: Path = None,
    pages: list[int] = None,
    page_indices: list[int] | None = None,
    native_exports: bool = True,
    write_combined_txt: bool = True,
    write_combined_md: bool = True,
    also_txt_for_markdown: bool = False,
) -> dict:
    """
    Run a single OCR model on all images.
    
    Args:
        model_name: Name of the model to run
        images: List of PIL images (used if model doesn't support native PDF)
        output_dir: Directory for output files
        pdf_path: Path to original PDF (for models with native PDF support)
        pages: Page indices to process for native PDF engines (0-indexed; same as CLI ``--pages``).
        page_indices: Maps each entry in ``images`` to its source PDF page index (for native export paths).
        native_exports: When True, write per-page artifacts under ``output_dir/native/<model>/``.
        write_combined_txt: For plain-text models, write ``*_output.txt``.
        write_combined_md: For Markdown-primary models, write combined text (``.md`` or model-specific).
        also_txt_for_markdown: If True, also write ``*_output.txt`` when the model is Markdown-primary.
    
    Returns:
        Dictionary with results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Running: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Track memory before
    ram_before = get_process_memory_mb()
    reset_gpu_memory_stats()
    
    try:
        # Instantiate model
        model_class = OCR_MODELS[model_name.lower()]
        model = model_class()

        slug = model_name.lower()
        if native_exports:
            model.configure_native_exports(output_dir, slug)
        
        # Set output directory for models that support it
        if hasattr(model, 'set_output_dir'):
            model.set_output_dir(output_dir)
        
        # Load model
        print(f"Loading {model.name} model...")
        model.load_model()
        
        # Check if model supports native PDF processing
        supports_native_pdf = getattr(model, 'supports_native_pdf', False)
        
        if supports_native_pdf and pdf_path is not None:
            # Use native PDF processing (more efficient)
            print(f"  Using native PDF processing...")
            all_texts, page_times = model.run_ocr_on_pdf(pdf_path, pages)
        else:
            # Fall back to image-by-image processing
            all_texts = []
            page_times = []
            
            for local_i, img in enumerate(tqdm(images, desc=f"Processing pages")):
                pidx = page_indices[local_i] if page_indices is not None else local_i
                model.set_page_index(pidx)
                text, elapsed = model.run_ocr(img)
                all_texts.append(text)
                page_times.append(elapsed)
        
        # Combine all text
        full_text = "\n\n--- Page Break ---\n\n".join(all_texts)

        export_info = write_combined_exports(
            model,
            model_name,
            output_dir,
            full_text,
            write_combined_txt=write_combined_txt,
            write_combined_md=write_combined_md,
            also_txt_for_markdown=also_txt_for_markdown,
        )
        output_file = export_info.get("output_file")
        
        # Measure memory after processing
        ram_after = get_process_memory_mb()
        peak_ram_mb = ram_after - ram_before  # Approximate increase
        peak_gpu_mb = get_gpu_memory_mb()
        
        # Calculate timing statistics
        total_time = sum(page_times)
        mean_time = statistics.mean(page_times)
        
        # For std/percentiles, need at least 2 data points
        if len(page_times) >= 2:
            std_time = statistics.stdev(page_times)
            sorted_times = sorted(page_times)
            n = len(sorted_times)
            p25 = sorted_times[int(n * 0.25)] if n >= 4 else sorted_times[0]
            p75 = sorted_times[int(n * 0.75)] if n >= 4 else sorted_times[-1]
        else:
            std_time = 0.0
            p25 = page_times[0] if page_times else 0.0
            p75 = page_times[0] if page_times else 0.0
        
        # GPU: use model's claim, or infer from measured VRAM (avoids "No" when VRAM was used)
        uses_gpu = model.uses_gpu or (peak_gpu_mb is not None and peak_gpu_mb > 0)

        result = {
            "model": model.name,
            "status": "success",
            "uses_gpu": uses_gpu,
            "pages_processed": len(all_texts),
            "total_time_seconds": round(total_time, 3),
            "time_per_page": {
                "mean": round(mean_time, 3),
                "std": round(std_time, 3),
                "min": round(min(page_times), 3),
                "max": round(max(page_times), 3),
                "p25": round(p25, 3),
                "p75": round(p75, 3),
            },
            "page_times": [round(t, 3) for t in page_times],
            "memory": {
                "peak_ram_mb": round(peak_ram_mb, 1),
                "peak_gpu_mb": round(peak_gpu_mb, 1) if peak_gpu_mb is not None else None,
            },
            "total_chars": len(full_text),
            "output_file": output_file,
            "output_files": export_info["output_files"],
            "output_json": export_info.get("output_json"),
        }
        if native_exports:
            result["native_exports_dir"] = str((output_dir / "native" / slug).resolve())
        
        gpu_status = "GPU" if uses_gpu else "CPU"
        mem_str = f"RAM: {peak_ram_mb:.0f}MB"
        if peak_gpu_mb is not None:
            mem_str += f", VRAM: {peak_gpu_mb:.0f}MB"
        print(f"✓ Completed in {total_time:.2f}s (mean: {mean_time:.2f}s ± {std_time:.2f}s per page) [{gpu_status}]")
        print(f"  Memory: {mem_str}")
        for p in export_info["output_files"]:
            print(f"  Combined output: {p}")
        if not export_info["output_files"]:
            print("  No combined text/md file written (disabled or empty).")
        if export_info.get("output_json"):
            print(f"  JSON output: {export_info['output_json']}")
        if native_exports:
            print(f"  Native exports: {result.get('native_exports_dir', '')}")
        
    except Exception as e:
        result = {
            "model": model_name,
            "status": "error",
            "uses_gpu": False,
            "error": str(e),
        }
        print(f"✗ Error: {e}")
    
    return result


def run_ablation(
    pdf_path: Path,
    output_dir: Path,
    models: Optional[list[str]] = None,
    pages: Optional[list[int]] = None,
    dpi: int = 300,
    native_exports: bool = True,
    write_combined_txt: bool = True,
    write_combined_md: bool = True,
    also_txt_for_markdown: bool = False,
) -> dict:
    """
    Run OCR ablation study on a PDF.
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Directory for outputs
        models: List of model names to run (None = all)
        pages: List of page numbers to process (None = all, 0-indexed)
        dpi: Resolution for PDF rendering
        native_exports: Enable per-model native artifacts under ``output_dir/native/``.
        write_combined_txt / write_combined_md / also_txt_for_markdown: See ``run_single_model``.
        
    Returns:
        Dictionary with all results
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which models to run
    if models is None:
        models = list(OCR_MODELS.keys())
    else:
        # Validate model names
        for m in models:
            if m.lower() not in OCR_MODELS:
                print(f"Warning: Unknown model '{m}', skipping")
        models = [m for m in models if m.lower() in OCR_MODELS]
    
    print(f"OCR Ablation Study")
    print(f"==================")
    print(f"PDF: {pdf_path}")
    print(f"Output: {output_dir}")
    print(f"Models: {', '.join(models)}")
    print(f"DPI: {dpi}")
    
    # Convert PDF to images
    print(f"\nConverting PDF to images...")
    all_images = pdf_to_images(pdf_path, dpi=dpi)
    print(f"  Total pages: {len(all_images)}")
    
    # Select specific pages if requested (keep aligned page_indices for native exports)
    if pages is not None:
        page_indices = [i for i in pages if 0 <= i < len(all_images)]
        images = [all_images[i] for i in page_indices]
        print(f"  Selected pages: {pages} (using {page_indices})")
    else:
        page_indices = None
        images = all_images
    
    # Save sample image for reference
    if images:
        sample_path = output_dir / "sample_page_0.png"
        images[0].save(sample_path)
        print(f"  Sample saved: {sample_path}")
    
    # Run each model
    results = {
        "metadata": {
            "pdf_path": str(pdf_path),
            "timestamp": datetime.now().isoformat(),
            "total_pages": len(all_images),
            "processed_pages": len(images),
            "dpi": dpi,
            "native_exports": native_exports,
            "write_combined_txt": write_combined_txt,
            "write_combined_md": write_combined_md,
            "also_txt_for_markdown": also_txt_for_markdown,
        },
        "results": []
    }
    
    for model_name in models:
        result = run_single_model(
            model_name,
            images,
            output_dir,
            pdf_path=pdf_path,
            pages=pages,
            page_indices=page_indices,
            native_exports=native_exports,
            write_combined_txt=write_combined_txt,
            write_combined_md=write_combined_md,
            also_txt_for_markdown=also_txt_for_markdown,
        )
        results["results"].append(result)
    
    # Save results summary
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print and save summary table
    print_summary(results, output_dir)
    
    return results


def print_summary(results: dict, output_dir: Path = None) -> None:
    """Print a summary table of results and optionally save to file."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    table_data = []
    for r in results["results"]:
        gpu_str = "Yes" if r.get("uses_gpu", False) else "No"
        if r["status"] == "success":
            tpp = r["time_per_page"]
            mem = r.get("memory", {})
            # Show mean ± std format
            time_str = f"{tpp['mean']:.2f}s ± {tpp['std']:.2f}s"
            # Memory string
            ram_mb = mem.get("peak_ram_mb", 0)
            gpu_mb = mem.get("peak_gpu_mb")
            mem_str = f"{ram_mb:.0f}"
            gpu_mem_str = f"{gpu_mb:.0f}" if gpu_mb is not None else "-"
            table_data.append([
                r["model"],
                "✓",
                gpu_str,
                f"{r['total_time_seconds']:.2f}s",
                time_str,
                mem_str,
                gpu_mem_str,
                f"{r['total_chars']:,}",
            ])
        else:
            table_data.append([
                r["model"],
                "✗",
                gpu_str,
                "-",
                "-",
                "-",
                "-",
                f"Error: {r.get('error', 'Unknown')[:20]}",
            ])
    
    headers = ["Model", "Status", "GPU", "Total Time", "Time/Page (μ±σ)", "RAM MB", "VRAM MB", "Chars"]
    
    # Print to console
    summary_table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(summary_table)
    
    # Save to file if output_dir provided
    if output_dir is not None:
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            meta = results["metadata"]
            f.write("OCR ABLATION STUDY - SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            if meta.get("pdf_path"):
                f.write(f"PDF: {meta['pdf_path']}\n")
            elif meta.get("image_dir"):
                f.write(f"Image directory: {meta['image_dir']}\n")
            f.write(f"Timestamp: {meta['timestamp']}\n")
            f.write(f"Pages processed: {meta['processed_pages']} / {meta['total_pages']}\n")
            dpi = meta.get("dpi")
            f.write(f"DPI: {dpi if dpi is not None else 'N/A'}\n")
            f.write("\n")
            f.write(summary_table)
            f.write("\n")
        print(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR ablation study on a PDF document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models
  python run_ocr_ablation.py document.pdf
  
  # Run specific models
  python run_ocr_ablation.py document.pdf --models tesseract paddleocr doctr
  
  # Process only first 3 pages
  python run_ocr_ablation.py document.pdf --pages 0 1 2
  
  # Custom output directory
  python run_ocr_ablation.py document.pdf -o ./my_results

  # DocTR: write *_output.json only (no combined *.txt):
  python run_ocr_ablation.py doc.pdf --no-combined-txt -m doctr

  # Markdown-primary models also emit *_output.txt alongside *.md:
  python run_ocr_ablation.py doc.pdf --also-combined-txt -m docling

  # All images in a folder (default five models unless you pass -m):
  python run_ocr_ablation.py --images-dir ./test_cases15 -o ./out_images
        """
    )
    
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        default=None,
        help="Path to input PDF (omit when using --images-dir)",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Run on every image in this directory (sorted by name) instead of a PDF. "
            f"Default models if -m omitted: {', '.join(DEFAULT_IMAGES_DIR_MODELS)}"
        ),
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: ./ocr_results_<timestamp>)"
    )
    
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        choices=list(OCR_MODELS.keys()),
        default=None,
        help="OCR models to run (default: all)"
    )
    
    parser.add_argument(
        "-p", "--pages",
        nargs="+",
        type=int,
        default=None,
        help="Page numbers to process (0-indexed, default: all)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF rendering (default: 300)"
    )

    parser.add_argument(
        "--no-native-exports",
        action="store_true",
        help="Disable per-model native outputs under OUTPUT/native/<model>/page_XXXX/",
    )

    parser.add_argument(
        "--no-combined-txt",
        action="store_true",
        help="Do not write *_output.txt for plain-text models (tesseract, DocTR, …).",
    )
    parser.add_argument(
        "--no-combined-md",
        action="store_true",
        help="Do not write combined markdown output (.md / .mmd) for Markdown-primary models.",
    )
    parser.add_argument(
        "--also-combined-txt",
        action="store_true",
        help="Also write *_output.txt when the model's primary combined file is Markdown.",
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available OCR models and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available OCR Models:")
        for name, cls in OCR_MODELS.items():
            print(f"  - {name}: {cls.name}")
        sys.exit(0)

    if args.images_dir is not None:
        if args.pdf is not None:
            print("Error: provide either a PDF path or --images-dir, not both.")
            sys.exit(1)
        if args.pages is not None:
            print("Error: --pages applies to PDF mode only.")
            sys.exit(1)
        if not args.images_dir.is_dir():
            print(f"Error: not a directory: {args.images_dir}")
            sys.exit(1)
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = Path(f"./ocr_image_results_{timestamp}")
        models = args.models if args.models is not None else list(DEFAULT_IMAGES_DIR_MODELS)
        run_ablation_on_images(
            image_dir=args.images_dir,
            output_dir=args.output,
            models=models,
            native_exports=not args.no_native_exports,
            write_combined_txt=not args.no_combined_txt,
            write_combined_md=not args.no_combined_md,
            also_txt_for_markdown=args.also_combined_txt,
        )
        return

    if args.pdf is None:
        print("Error: provide a PDF path, or use --images-dir for a folder of images.")
        sys.exit(1)
    
    # Validate PDF path
    if not args.pdf.exists():
        print(f"Error: PDF file not found: {args.pdf}")
        sys.exit(1)
    
    # Set output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"./ocr_results_{timestamp}")
    
    # Run ablation
    run_ablation(
        pdf_path=args.pdf,
        output_dir=args.output,
        models=args.models,
        pages=args.pages,
        dpi=args.dpi,
        native_exports=not args.no_native_exports,
        write_combined_txt=not args.no_combined_txt,
        write_combined_md=not args.no_combined_md,
        also_txt_for_markdown=args.also_combined_txt,
    )


if __name__ == "__main__":
    main()

