from __future__ import annotations

from pathlib import Path

import docx

from smart_text_extractor.core.models import DocumentUnit, TextSegment
from smart_text_extractor.export.docx_export import build_docx, export_docx


def _has_page_break(paragraph) -> bool:
    return 'type="page"' in paragraph._p.xml


def _seg(text: str, confidence: float = 90.0) -> list[TextSegment]:
    return [TextSegment(text, confidence)]


def test_paragraph_unit_becomes_an_rtl_right_aligned_paragraph() -> None:
    document = build_docx([[DocumentUnit(kind="paragraph", segments=_seg("مرحباً بكم"))]])

    paragraphs = [p for p in document.paragraphs if p.text]
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "مرحباً بكم"
    assert 'w:bidi' in paragraphs[0]._p.xml


def test_heading_unit_becomes_a_heading_2_style_paragraph() -> None:
    document = build_docx([[DocumentUnit(kind="heading", segments=_seg("عنوان القسم"))]])

    headings = [p for p in document.paragraphs if p.style.name == "Heading 2"]
    assert len(headings) == 1
    assert headings[0].text == "عنوان القسم"


def test_table_unit_becomes_a_real_docx_table_with_matching_cells() -> None:
    rows = [[_seg("A"), _seg("B"), _seg("C")], [_seg("1"), _seg("2")]]  # ragged, like a real messy table
    unit = DocumentUnit(kind="table", rows=rows)

    document = build_docx([[unit]])

    assert len(document.tables) == 1
    table = document.tables[0]
    assert len(table.rows) == 2
    assert len(table.columns) == 3  # padded to the widest row
    # w:bidiVisual (real-world regression, docs/phases/phase-2-ocr-pipeline.md):
    # without it, a real Word render put cell 0 in the leftmost visual
    # column regardless of paragraph direction — verified by rendering
    # through actual Word to PDF, with and without this flag.
    assert "bidiVisual" in table._tbl.tblPr.xml
    assert [cell.text for cell in table.rows[0].cells] == ["A", "B", "C"]
    assert [cell.text for cell in table.rows[1].cells] == ["1", "2", ""]  # short row padded with an empty cell


def test_multiple_pages_get_a_real_page_break_between_them() -> None:
    document = build_docx(
        [
            [DocumentUnit(kind="paragraph", segments=_seg("page one"))],
            [DocumentUnit(kind="paragraph", segments=_seg("page two"))],
        ]
    )

    assert any(_has_page_break(p) for p in document.paragraphs)


def test_single_page_has_no_page_break() -> None:
    document = build_docx([[DocumentUnit(kind="paragraph", segments=_seg("only page"))]])

    assert not any(_has_page_break(p) for p in document.paragraphs)


def test_edited_page_falls_back_to_one_paragraph_per_line() -> None:
    """A page the user has edited only has plain edited_text left (no
    positional data for headings/tables — see MainWindow._on_export_word),
    so PageContent accepts a plain string for that page instead of
    list[DocumentUnit]."""
    document = build_docx(["line one\nline two"])

    texts = [p.text for p in document.paragraphs]
    assert texts == ["line one", "line two"]


def test_export_docx_writes_a_file_that_reopens_with_the_same_content(tmp_path: Path) -> None:
    path = tmp_path / "out.docx"

    export_docx([[DocumentUnit(kind="paragraph", segments=_seg("نص للتحقق"))]], path)

    assert path.exists()
    reopened = docx.Document(str(path))
    assert any(p.text == "نص للتحقق" for p in reopened.paragraphs)
