"""Searchable-PDF export tests (US-10). These build a real PDF and read it
back with PyMuPDF — the point of a searchable PDF is what a reader can get
out of it, so nothing here asserts on internal state.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from smart_text_extractor.core.models import BoundingBox, Rect
from smart_text_extractor.export.pdf_export import SearchablePage, export_searchable_pdf, find_arabic_capable_font

requires_arabic_font = pytest.mark.skipif(
    find_arabic_capable_font() is None, reason="no Arabic-capable font installed on this machine"
)


@pytest.fixture()
def page_image(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    Image.new("RGB", (1200, 600), "white").save(path)
    return path


def _box(text: str, x: int = 100, y: int = 100, width: int = 200, height: int = 50) -> BoundingBox:
    return BoundingBox(text=text, rect=Rect(x=x, y=y, width=width, height=height), confidence=95.0)


def test_page_size_comes_from_the_images_own_dpi(tmp_path: Path, page_image: Path) -> None:
    """§7.3/§11 risk #3: word boxes are pixels at the page's DPI, so the DPI
    is the only thing that maps them onto the PDF's points. A 1200x600
    image at 300 DPI is 4x2 inches = 288x144 points."""
    output = tmp_path / "out.pdf"

    export_searchable_pdf([SearchablePage(image_path=page_image, word_boxes=[], dpi=300)], output)

    with pymupdf.open(output) as document:
        page = document.load_page(0)
        assert round(page.rect.width) == 288
        assert round(page.rect.height) == 144


def test_latin_text_is_searchable_and_positioned_over_its_word(tmp_path: Path, page_image: Path) -> None:
    output = tmp_path / "out.pdf"
    boxes = [_box("Dashboard", x=400, y=100, width=300, height=50)]

    export_searchable_pdf([SearchablePage(image_path=page_image, word_boxes=boxes, dpi=300)], output)

    with pymupdf.open(output) as document:
        page = document.load_page(0)
        hits = page.search_for("Dashboard")
        assert len(hits) == 1
        # 400px at 300 DPI = 96 points from the left edge
        assert hits[0].x0 == pytest.approx(96, abs=2)


@requires_arabic_font
def test_arabic_text_round_trips_through_the_text_layer(tmp_path: Path, page_image: Path) -> None:
    """See _for_insertion: MuPDF stores Arabic as presentation forms, so
    the guarantee this export actually makes is that extracted text
    NFKC-normalises back to the original word, in reading order."""
    output = tmp_path / "out.pdf"
    boxes = [_box("المستندات")]

    export_searchable_pdf([SearchablePage(image_path=page_image, word_boxes=boxes, dpi=300)], output)

    with pymupdf.open(output) as document:
        extracted = document.load_page(0).get_text().strip()

    assert unicodedata.normalize("NFKC", extracted) == "المستندات"


def test_every_page_is_written(tmp_path: Path, page_image: Path) -> None:
    output = tmp_path / "out.pdf"
    pages = [SearchablePage(image_path=page_image, word_boxes=[_box("One")], dpi=300) for _ in range(3)]

    export_searchable_pdf(pages, output)

    with pymupdf.open(output) as document:
        assert len(document) == 3


def test_blank_words_are_skipped(tmp_path: Path, page_image: Path) -> None:
    output = tmp_path / "out.pdf"
    boxes = [_box("   "), _box("Real", x=500)]

    export_searchable_pdf([SearchablePage(image_path=page_image, word_boxes=boxes, dpi=300)], output)

    with pymupdf.open(output) as document:
        assert document.load_page(0).get_text().strip() == "Real"


def test_no_temp_file_is_left_behind(tmp_path: Path, page_image: Path) -> None:
    """Atomic write (§8.2): the reader must never find a half-written PDF."""
    output = tmp_path / "out.pdf"

    export_searchable_pdf([SearchablePage(image_path=page_image, word_boxes=[_box("Hi")], dpi=300)], output)

    assert output.exists()
    assert not list(tmp_path.glob("*.tmp"))
