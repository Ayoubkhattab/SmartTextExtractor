from __future__ import annotations

from tests.ocr.conftest import make_text_image, requires_arabic_font, requires_tesseract


@requires_tesseract
@requires_arabic_font
def test_engine_extracts_arabic_and_english_text_correctly(ocr_engine) -> None:
    image = make_text_image(
        [
            ("مرحباً بكم في مستخرج النص الذكي", True),
            ("Smart Text Extractor 2026", False),
        ]
    )

    result = ocr_engine.run(image)

    assert "مرحبا" in result.raw_text  # tanween on the final alef is sometimes dropped by OCR
    assert "بكم" in result.raw_text
    assert "مستخرج" in result.raw_text
    assert "Smart Text Extractor 2026" in result.raw_text
    assert result.confidence_score > 0


@requires_tesseract
@requires_arabic_font
def test_engine_populates_word_boxes_with_real_coordinates(ocr_engine) -> None:
    image = make_text_image([("Hello World", False)])

    result = ocr_engine.run(image)

    assert len(result.word_boxes) == 2
    texts = {box.text for box in result.word_boxes}
    assert texts == {"Hello", "World"}
    for box in result.word_boxes:
        assert box.rect.width > 0
        assert box.rect.height > 0


@requires_tesseract
@requires_arabic_font
def test_engine_on_blank_page_returns_empty_result(ocr_engine) -> None:
    import numpy as np

    blank = np.full((200, 400, 3), 255, dtype="uint8")
    result = ocr_engine.run(blank)

    assert result.raw_text.strip() == ""
    assert result.word_boxes == []
    assert result.confidence_score == 0.0
