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

import pymupdf as fitz # PyMuPDF
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
    TrOCRModel,
    RolmOCR,
    DeepSeekOCR,
    DonutOCR,
    PPStructureOCR,
    PaddleOCRVL,
    DoclingOCR,
    MarkerOCR,
    MinerUOCR,
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
    "nougat": NougatOCR,
    "doctr": DocTROCR,
    "trocr": TrOCRModel,
    "rolmocr": RolmOCR,
    "deepseek": DeepSeekOCR,
    "donut": DonutOCR,
    "ppstructure": PPStructureOCR,
    "paddleocr-vl": PaddleOCRVL,
    "docling": DoclingOCR,
    "marker": MarkerOCR,
    "mineru": MinerUOCR,
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


def run_single_model(
    model_name: str,
    images: list[Image.Image],
    output_dir: Path,
    pdf_path: Path = None,
    pages: list[int] = None
) -> dict:
    """
    Run a single OCR model on all images.
    
    Args:
        model_name: Name of the model to run
        images: List of PIL images (used if model doesn't support native PDF)
        output_dir: Directory for output files
        pdf_path: Path to original PDF (for models with native PDF support)
        pages: List of page indices to process (for native PDF processing)
    
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
            
            for i, img in enumerate(tqdm(images, desc=f"Processing pages")):
                text, elapsed = model.run_ocr(img)
                all_texts.append(text)
                page_times.append(elapsed)
        
        # Combine all text
        full_text = "\n\n--- Page Break ---\n\n".join(all_texts)
        
        # Save output
        output_file = output_dir / f"{model_name}_output.txt"
        output_file.write_text(full_text, encoding="utf-8")
        
        # Save JSON output for models that support it (e.g., DocTR)
        if hasattr(model, 'get_json_string'):
            json_file = output_dir / f"{model_name}_output.json"
            json_file.write_text(model.get_json_string(), encoding="utf-8")
        
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
        
        result = {
            "model": model.name,
            "status": "success",
            "uses_gpu": model.uses_gpu,
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
            "output_file": str(output_file),
        }
        
        gpu_status = "GPU" if model.uses_gpu else "CPU"
        mem_str = f"RAM: {peak_ram_mb:.0f}MB"
        if peak_gpu_mb is not None:
            mem_str += f", VRAM: {peak_gpu_mb:.0f}MB"
        print(f"✓ Completed in {total_time:.2f}s (mean: {mean_time:.2f}s ± {std_time:.2f}s per page) [{gpu_status}]")
        print(f"  Memory: {mem_str}")
        print(f"  Output saved to: {output_file}")
        
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
    dpi: int = 300
) -> dict:
    """
    Run OCR ablation study on a PDF.
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Directory for outputs
        models: List of model names to run (None = all)
        pages: List of page numbers to process (None = all, 0-indexed)
        dpi: Resolution for PDF rendering
        
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
    
    # Select specific pages if requested
    if pages is not None:
        images = [all_images[i] for i in pages if i < len(all_images)]
        print(f"  Selected pages: {pages}")
    else:
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
        },
        "results": []
    }
    
    for model_name in models:
        result = run_single_model(model_name, images, output_dir, pdf_path=pdf_path, pages=pages)
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
            f.write("OCR ABLATION STUDY - SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"PDF: {results['metadata']['pdf_path']}\n")
            f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
            f.write(f"Pages processed: {results['metadata']['processed_pages']} / {results['metadata']['total_pages']}\n")
            f.write(f"DPI: {results['metadata']['dpi']}\n")
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
        """
    )
    
    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to input PDF file"
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
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()

