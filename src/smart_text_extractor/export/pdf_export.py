"""Searchable-PDF export (§14 Phase 4, US-10): each page's image with an
invisible text layer laid over it, so the result looks exactly like the
original scan but can be selected, copied and searched.

The text is positioned from OcrResult.word_boxes, whose coordinates are
pixel offsets on the page image as captured at Page.dpi (§7.3) — which is
why the DPI has to be carried per page rather than assumed: it is the only
thing that converts those pixels into the PDF's points (72 per inch), and
guessing it would shift every word off its glyph (§11 risk #3).

Written atomically (temp file + rename) for the same reason
core/persistence.py is: an export interrupted halfway must not leave a
truncated PDF sitting where the user expects a finished one (§8.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from smart_text_extractor.core.models import BoundingBox

DEFAULT_DPI = 300
"""Used only when a page carries no DPI of its own. Matches
core/pdf_import.DEFAULT_RENDER_DPI and the scan-resolution convention
(§7.3), so it is the right assumption for this project's own pages rather
than an arbitrary constant — but a page that knows its DPI always wins."""

# Render mode 3 = "invisible" in PDF's text-rendering-mode operator: the
# glyphs are laid down and are fully selectable/searchable, but nothing is
# painted, so the page still looks exactly like its image.
_INVISIBLE_TEXT = 3

_MIN_FONT_SIZE = 1.0

_ARABIC_RANGE = range(0x0600, 0x0700)


def _is_arabic(text: str) -> bool:
    return any(ord(character) in _ARABIC_RANGE for character in text)


def _for_insertion(text: str) -> str:
    """Arabic is written reversed so that it comes back out in the right order.

    KNOWN LIMITATION, measured rather than assumed. MuPDF shapes Arabic
    when text is inserted and writes the resulting glyphs left to right,
    so the text layer ends up holding Unicode PRESENTATION forms (U+FE95
    …) in visual order. Two consequences:

      - Reversing the word before insertion makes extraction return it in
        reading order. Verified end to end: the extracted text then
        NFKC-normalises back to exactly the original word.
      - The stored characters are still presentation forms, so a reader
        that searches without normalising (PyMuPDF's own search_for among
        them) will not match a query typed in ordinary Arabic. Readers
        that normalise do. Latin text is unaffected and fully searchable.

    Every insertion path PyMuPDF offers was tried — insert_text,
    TextWriter, insert_htmlbox, and one character at a time — and all four
    produce presentation forms; getting ordinary code points into the text
    layer would mean writing the content stream and its ToUnicode CMap
    directly.
    """
    return text[::-1] if _is_arabic(text) else text

_FALLBACK_FONTNAME = "helv"
"""PyMuPDF's built-in fonts (the Base-14 set plus CJK) contain no Arabic
glyphs at all — confirmed: inserting Arabic with one produces a PDF whose
Arabic is simply absent from the text layer while English survives. So an
Arabic-capable font file has to be embedded, and this is only the
last-resort fallback for a machine where none can be found: the export
still succeeds and its Latin text is still searchable."""

_EMBEDDED_FONTNAME = "steta"

_ARABIC_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
)


def find_arabic_capable_font() -> Path | None:
    """First installed font known to carry Arabic glyphs, or None.

    Phase 5 packaging should bundle one instead of relying on the host
    (§10) — same shape as ocr/locate.py's Tesseract discovery, and for the
    same reason: this only has to hold while running from source.
    """
    for candidate in _ARABIC_FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


@dataclass
class SearchablePage:
    """One page to write: the image to show, and the words to hide behind it."""

    image_path: Path
    word_boxes: list[BoundingBox]
    dpi: int | None = None


def export_searchable_pdf(pages: list[SearchablePage], output_path: Path) -> None:
    output_path = Path(output_path)
    document = pymupdf.open()
    arabic_font = find_arabic_capable_font()

    try:
        for page in pages:
            _add_page(document, page, arabic_font)

        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        document.save(str(temp_path))
        temp_path.replace(output_path)
    finally:
        document.close()


def _add_page(document: pymupdf.Document, page: SearchablePage, arabic_font: Path | None) -> None:
    pixels_to_points = 72 / (page.dpi or DEFAULT_DPI)

    image_rect = pymupdf.Pixmap(str(page.image_path))
    width_points = image_rect.width * pixels_to_points
    height_points = image_rect.height * pixels_to_points
    image_rect = None  # release the pixmap; only its size was needed

    pdf_page = document.new_page(width=width_points, height=height_points)
    pdf_page.insert_image(pymupdf.Rect(0, 0, width_points, height_points), filename=str(page.image_path))

    fontname = _FALLBACK_FONTNAME
    if arabic_font is not None:
        pdf_page.insert_font(fontname=_EMBEDDED_FONTNAME, fontfile=str(arabic_font))
        fontname = _EMBEDDED_FONTNAME

    for box in page.word_boxes:
        if not box.text.strip():
            continue
        rect = pymupdf.Rect(
            box.rect.x * pixels_to_points,
            box.rect.y * pixels_to_points,
            (box.rect.x + box.rect.width) * pixels_to_points,
            (box.rect.y + box.rect.height) * pixels_to_points,
        )
        # Sized to the word's own box so a search hit highlights the right
        # area of the image, and so long words don't spill past their glyphs.
        font_size = max(rect.height * 0.8, _MIN_FONT_SIZE)
        pdf_page.insert_text(
            pymupdf.Point(rect.x0, rect.y1),
            _for_insertion(box.text),
            fontsize=font_size,
            fontname=fontname,
            render_mode=_INVISIBLE_TEXT,
        )
