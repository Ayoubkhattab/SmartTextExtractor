"""PDF -> per-page raster images, so multi-page PDFs feed the same OCR
pipeline as scanned/uploaded images (SourceType.UPLOAD_PDF, §5.1).

Uses PyMuPDF, the dependency already resolved for PDF work in §10 — its
role here (rendering pages to images) is unrelated to its later role in
Phase 4 (assembling the searchable PDF export).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from smart_text_extractor.core.models import OcrResult
from smart_text_extractor.ocr.native_pdf_text import extract_native_text_result

DEFAULT_RENDER_DPI = 300  # matches the scan-resolution convention (§7.3)


def _render_page_to_png(page: pymupdf.Page, out_path: Path, matrix: pymupdf.Matrix) -> None:
    pixmap = page.get_pixmap(matrix=matrix)
    pixmap.save(str(out_path))


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
            out_path = output_dir / f"{pdf_path.stem}_page{page_index + 1:03d}.png"
            _render_page_to_png(page, out_path, matrix)
            image_paths.append(out_path)
    return image_paths


@dataclass
class PdfPageImport:
    image_path: Path
    native_result: OcrResult | None  # None => this page has no usable text layer, still needs OCR


def import_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int = DEFAULT_RENDER_DPI) -> list[PdfPageImport]:
    """Like render_pdf_to_images, but also attempts native text-layer
    extraction (ocr/native_pdf_text.py) per page: a page with a real,
    substantial embedded text layer skips the OCR pipeline entirely —
    perfectly accurate, since there is no recognition step to make an
    error in. A page without one (a genuinely scanned page) still gets
    rendered to an image here exactly as before, for the OCR pipeline to
    pick up unchanged.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)

    results: list[PdfPageImport] = []
    with pymupdf.open(str(pdf_path)) as document:
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            out_path = output_dir / f"{pdf_path.stem}_page{page_index + 1:03d}.png"
            _render_page_to_png(page, out_path, matrix)
            native_result = extract_native_text_result(page, render_dpi=dpi)
            results.append(PdfPageImport(image_path=out_path, native_result=native_result))
    return results
