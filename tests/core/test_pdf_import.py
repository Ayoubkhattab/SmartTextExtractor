"""Uses a real PDF generated with PyMuPDF itself — not a fixture file
checked into the repo, so the test has no external binary dependency."""
from __future__ import annotations

from pathlib import Path

import pymupdf

from smart_text_extractor.core.pdf_import import render_pdf_to_images


def _make_test_pdf(path: Path, n_pages: int) -> None:
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page(width=400, height=300)
        page.insert_text((50, 50), f"Page {i + 1} content", fontsize=20)
    doc.save(str(path))
    doc.close()


def test_render_pdf_to_images_produces_one_image_per_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_test_pdf(pdf_path, n_pages=3)

    image_paths = render_pdf_to_images(pdf_path, tmp_path / "out")

    assert len(image_paths) == 3
    for path in image_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_render_pdf_to_images_preserves_page_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_test_pdf(pdf_path, n_pages=3)

    image_paths = render_pdf_to_images(pdf_path, tmp_path / "out")

    assert [p.name for p in image_paths] == [
        "doc_page001.png",
        "doc_page002.png",
        "doc_page003.png",
    ]


def test_render_pdf_to_images_respects_dpi_via_image_size(tmp_path: Path) -> None:
    from PIL import Image

    pdf_path = tmp_path / "doc.pdf"
    _make_test_pdf(pdf_path, n_pages=1)

    low_dpi_paths = render_pdf_to_images(pdf_path, tmp_path / "low", dpi=72)
    high_dpi_paths = render_pdf_to_images(pdf_path, tmp_path / "high", dpi=300)

    low_size = Image.open(low_dpi_paths[0]).size
    high_size = Image.open(high_dpi_paths[0]).size
    assert high_size[0] > low_size[0]
    assert high_size[1] > low_size[1]
