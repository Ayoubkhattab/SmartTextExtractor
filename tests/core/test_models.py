from __future__ import annotations

from pathlib import Path

from smart_text_extractor.core.models import Document, SourceType


def test_add_page_assigns_sequential_order_index(tmp_path: Path) -> None:
    doc = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)

    first = doc.add_page(tmp_path / "p1.bmp")
    second = doc.add_page(tmp_path / "p2.bmp")

    assert first.order_index == 0
    assert second.order_index == 1
    assert doc.pages == [first, second]


def test_each_document_and_page_gets_a_unique_id(tmp_path: Path) -> None:
    doc_a = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)
    doc_b = Document(source_type=SourceType.SCAN, temp_dir_path=tmp_path)
    page = doc_a.add_page(tmp_path / "p1.bmp")

    assert doc_a.id != doc_b.id
    assert page.id
