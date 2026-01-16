#!/usr/bin/env python3
"""
OCR Ablation Study Runner

Compare multiple OCR engines on PDF documents.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
from tabulate import tabulate

from ocr_models import (
    TesseractOCR,
    PaddleOCRModel,
    KrakenOCR,
    NougatOCR,
    DocTROCR,
    TrOCRModel,
    RolmOCR,
    DeepSeekOCR,
)


# Registry of available OCR models
OCR_MODELS = {
    "tesseract": TesseractOCR,
    "paddleocr": PaddleOCRModel,
    "kraken": KrakenOCR,
    "nougat": NougatOCR,
    "doctr": DocTROCR,
    "trocr": TrOCRModel,
    "rolmocr": RolmOCR,
    "deepseek": DeepSeekOCR,
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
    output_dir: Path
) -> dict:
    """
    Run a single OCR model on all images.
    
    Returns:
        Dictionary with results and metrics
    """
    print(f"\n{'='*60}")
    print(f"Running: {model_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Instantiate model
        model_class = OCR_MODELS[model_name.lower()]
        model = model_class()
        
        # Load model
        print(f"Loading {model.name} model...")
        model.load_model()
        
        # Run OCR on each page
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
            "pages_processed": len(images),
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
            "total_chars": len(full_text),
            "output_file": str(output_file),
        }
        
        gpu_status = "GPU" if model.uses_gpu else "CPU"
        print(f"✓ Completed in {total_time:.2f}s (mean: {mean_time:.2f}s ± {std_time:.2f}s per page) [{gpu_status}]")
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
        result = run_single_model(model_name, images, output_dir)
        results["results"].append(result)
    
    # Save results summary
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print_summary(results)
    
    return results


def print_summary(results: dict) -> None:
    """Print a summary table of results."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    table_data = []
    for r in results["results"]:
        gpu_str = "Yes" if r.get("uses_gpu", False) else "No"
        if r["status"] == "success":
            tpp = r["time_per_page"]
            # Show mean ± std format
            time_str = f"{tpp['mean']:.2f}s ± {tpp['std']:.2f}s"
            table_data.append([
                r["model"],
                "✓",
                gpu_str,
                f"{r['total_time_seconds']:.2f}s",
                time_str,
                f"[{tpp['p25']:.2f}, {tpp['p75']:.2f}]",
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
                f"Error: {r.get('error', 'Unknown')[:25]}",
            ])
    
    headers = ["Model", "Status", "GPU", "Total Time", "Time/Page (μ±σ)", "IQR [p25,p75]", "Chars"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


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

