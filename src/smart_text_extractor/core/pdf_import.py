"""PDF -> per-page raster images, so multi-page PDFs feed the same OCR
pipeline as scanned/uploaded images (SourceType.UPLOAD_PDF, §5.1).

Uses PyMuPDF, the dependency already resolved for PDF work in §10 — its
role here (rendering pages to images) is unrelated to its later role in
Phase 4 (assembling the searchable PDF export).
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

DEFAULT_RENDER_DPI = 300  # matches the scan-resolution convention (§7.3)


def render_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = DEFAULT_RENDER_DPI) -> list[Path]:
    """Renders every page of pdf_path to a PNG in output_dir.

    Returns the resulting image paths in page order.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72  # PDF points are 72 per inch — PyMuPDF renders at 72 DPI by default
    matrix = pymupdf.Matrix(zoom, zoom)

    image_paths: list[Path] = []
    with pymupdf.open(str(pdf_path)) as document:
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix)
            out_path = output_dir / f"{pdf_path.stem}_page{page_index + 1:03d}.png"
            pixmap.save(str(out_path))
            image_paths.append(out_path)
    return image_paths
