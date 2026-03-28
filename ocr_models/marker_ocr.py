"""Marker OCR implementation - PDF/image to Markdown conversion."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from .base import BaseOCR

# ``marker_single`` supports ``--output_format markdown|json|html|chunks``. Markdown always runs
# first (ablation text + ``input.md``). By default we also run HTML, tree JSON, and chunks JSON.
# Disable extras for speed: ``MARKER_EXTRA_OUTPUT_FORMATS=`` (empty).
_DEFAULT_MARKER_EXTRA_FORMATS = "html,json,chunks"


def _resolve_marker_executable(name: str) -> str | None:
    """Resolve ``marker_single`` / ``marker`` on PATH or next to ``sys.executable`` (venv bin)."""
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).resolve().parent / name
    return str(candidate) if candidate.is_file() else None


class MarkerOCR(BaseOCR):
    """Marker - Fast, accurate PDF/image to Markdown conversion (Datalab)."""

    name = "Marker"

    def __init__(self):
        """
        Initialize Marker OCR.

        Marker converts PDFs and images to Markdown (primary text for the runner). Native
        exports default to ``input.md``, ``input.html``, ``input.tree.json`` (hierarchical
        JSON), and ``input.chunks.json``, unless ``MARKER_EXTRA_OUTPUT_FORMATS`` is empty or
        lists a subset (comma-separated: ``html``, ``json``, ``chunks``).
        """
        super().__init__()
        self._temp_dir = None

    @property
    def uses_gpu(self) -> bool:
        """Return whether this model is using GPU acceleration."""
        return False  # Marker may use GPU if available; we don't detect here

    def is_markdown_primary(self) -> bool:
        return True

    def load_model(self) -> None:
        """Verify Marker is available (CLI-based)."""
        self._marker_exes: list[str] = []
        for name in ("marker_single", "marker"):
            exe = _resolve_marker_executable(name)
            if exe and exe not in self._marker_exes:
                self._marker_exes.append(exe)
        if not self._marker_exes:
            raise RuntimeError(
                "Marker CLI not found. Install with: pip install marker-pdf\n"
                f"Expected `marker_single` or `marker` on PATH or next to this Python "
                f"({sys.executable}). Activate the venv that has marker-pdf, or run:\n"
                f"  {sys.executable} -m pip install marker-pdf"
            )
        # Smoke-check at least one entrypoint responds (import errors surface here).
        ok = False
        for exe in self._marker_exes:
            try:
                r = subprocess.run(
                    [exe, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if r.returncode == 0:
                    ok = True
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        if not ok:
            err_tail = ""
            if self._marker_exes:
                try:
                    r = subprocess.run(
                        [self._marker_exes[0], "--help"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    err_tail = (r.stderr or r.stdout or "").strip()[:800]
                except (subprocess.TimeoutExpired, OSError):
                    pass
            raise RuntimeError(
                "Marker is installed but its CLI failed (--help non-zero or crashed). "
                f"First executable tried: {self._marker_exes[0]}\n"
                + (f"Output:\n{err_tail}\n" if err_tail else "")
            )
        self._temp_dir = tempfile.mkdtemp(prefix="marker_ocr_")
        self._is_loaded = True

    def _marker_extra_formats(self) -> list[str]:
        raw = (os.environ.get("MARKER_EXTRA_OUTPUT_FORMATS") or _DEFAULT_MARKER_EXTRA_FORMATS).strip()
        if not raw:
            return []
        out: list[str] = []
        for part in raw.split(","):
            p = part.strip().lower()
            if p == "markdown":
                continue  # always produced first; listed only for symmetry in env
            if p in ("html", "chunks", "json") and p not in out:
                out.append(p)
        return out

    def _invoke_marker(
        self,
        exe: str,
        input_path: Path,
        out_dir: Path,
        output_format: str,
    ) -> bool:
        """Run Marker CLI once; return True on success (exit 0)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        arg_variants = [
            [
                str(input_path),
                "--output_dir",
                str(out_dir),
                "--output_format",
                output_format,
            ],
            [str(input_path), str(out_dir), "--output_format", output_format],
        ]
        for argv_tail in arg_variants:
            argv = [exe, *argv_tail]
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(self._temp_dir),
                )
                if proc.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return False

    def _run_ocr(self, image: Image.Image) -> str:
        """Run Marker on an image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        path = Path(self._temp_dir) / "input.png"
        image.save(path, "PNG")
        doc_stem = path.stem

        out_md = Path(self._temp_dir) / "out_md"
        shutil.rmtree(out_md, ignore_errors=True)
        out_md.mkdir(parents=True)

        text_out = ""
        for exe in self._marker_exes:
            if not self._invoke_marker(exe, path, out_md, "markdown"):
                continue

            md_files = list(out_md.glob("**/*.md"))
            for m in md_files:
                text = m.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    text_out = text
                    break

            nd = self.native_page_dir()
            if nd is not None:
                marker_root = nd / "marker"
                marker_root.mkdir(parents=True, exist_ok=True)
                for f in out_md.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(out_md)
                        dest = marker_root / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dest)

                extras = self._marker_extra_formats()
                html_out = Path(self._temp_dir) / "out_html"
                chunks_out = Path(self._temp_dir) / "out_chunks"
                json_out = Path(self._temp_dir) / "out_json"
                if "html" in extras:
                    shutil.rmtree(html_out, ignore_errors=True)
                    if self._invoke_marker(exe, path, html_out, "html"):
                        hf = html_out / doc_stem / f"{doc_stem}.html"
                        if hf.is_file():
                            dest = marker_root / doc_stem / f"{doc_stem}.html"
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(hf, dest)
                if "chunks" in extras:
                    shutil.rmtree(chunks_out, ignore_errors=True)
                    if self._invoke_marker(exe, path, chunks_out, "chunks"):
                        jf = chunks_out / doc_stem / f"{doc_stem}.json"
                        if jf.is_file():
                            dest = marker_root / doc_stem / f"{doc_stem}.chunks.json"
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(jf, dest)
                if "json" in extras:
                    shutil.rmtree(json_out, ignore_errors=True)
                    if self._invoke_marker(exe, path, json_out, "json"):
                        jf = json_out / doc_stem / f"{doc_stem}.json"
                        if jf.is_file():
                            dest = marker_root / doc_stem / f"{doc_stem}.tree.json"
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(jf, dest)

            return text_out if text_out else ""

        raise RuntimeError(
            "Marker CLI did not produce Markdown for this page. "
            f"Tried: {', '.join(self._marker_exes)} with --output_dir and positional output dir."
        )

    def __del__(self):
        """Clean up temp directory on deletion."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
