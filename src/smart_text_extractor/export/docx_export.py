"""Word (.docx) export (§7.3, US-09) via python-docx.

Consumes the same DocumentUnit structure the Markdown export uses
(smart_text_extractor.ocr.reorder.classify_document_units, via
OcrResult.document_units) so heading/table classification (§7.1.1) lives
in exactly one place, not reimplemented per export format.
"""
from __future__ import annotations

from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from smart_text_extractor.core.models import DocumentUnit

# A page's content is either its still-structured OCR result (most pages)
# or a plain string — a page the user has edited only has edited_text, a
# plain string with no positional data left to classify into headings/
# tables (see MainWindow._on_export_markdown's identical distinction).
PageContent = list[DocumentUnit] | str


def _set_rtl(paragraph: Paragraph) -> None:
    """Marks a paragraph right-to-left — both visually (right alignment)
    and structurally (the real w:bidi OOXML flag, verified via a
    round-trip: python-docx reads it back correctly after save/reopen).
    Plain right-alignment alone is a cosmetic-only fix; w:bidi is what
    actually tells Word this paragraph's text direction and cursor
    behavior are RTL, which matters for a tool whose primary content is
    Arabic.
    """
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    paragraph_properties = paragraph._p.get_or_add_pPr()
    paragraph_properties.append(paragraph_properties.makeelement(qn("w:bidi"), {}))


def _add_table(document: docx.document.Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    column_count = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    for row_index, row in enumerate(rows):
        for column_index in range(column_count):
            cell = table.cell(row_index, column_index)
            cell.text = row[column_index] if column_index < len(row) else ""
            for paragraph in cell.paragraphs:
                _set_rtl(paragraph)


def _add_units(document: docx.document.Document, units: list[DocumentUnit]) -> None:
    for unit in units:
        if unit.kind == "heading":
            _set_rtl(document.add_heading(unit.text, level=2))
        elif unit.kind == "table":
            _add_table(document, unit.rows)
        else:
            _set_rtl(document.add_paragraph(unit.text))


def _add_plain_text(document: docx.document.Document, text: str) -> None:
    """Fallback for a page the user has edited (see PageContent) — one
    Word paragraph per line, since a single Word paragraph does not treat
    an embedded "\\n" as a line break the way plain text does."""
    for line in text.split("\n"):
        _set_rtl(document.add_paragraph(line))


def build_docx(pages: list[PageContent]) -> docx.document.Document:
    """pages: one entry per exported page, in export order. A page
    boundary becomes a real Word page break, not just blank space — Word
    has that primitive and Markdown doesn't."""
    document = docx.Document()
    for page_index, content in enumerate(pages):
        if page_index > 0:
            document.add_page_break()
        if isinstance(content, str):
            _add_plain_text(document, content)
        else:
            _add_units(document, content)
    return document


def export_docx(pages: list[PageContent], path: Path) -> None:
    build_docx(pages).save(str(path))
