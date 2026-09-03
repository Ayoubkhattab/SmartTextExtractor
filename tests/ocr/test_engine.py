from __future__ import annotations

from tests.ocr.conftest import make_degraded_arabic_image, make_text_image, requires_arabic_font, requires_tesseract


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

    # Loose on purpose: a CI run misread "2026" as "6" on one platform's
    # font rendering, and local bisection found Arabic word-level fidelity
    # from this synthetic-image rendering path is itself non-monotonic in
    # font size across otherwise-identical runs (see make_text_image's
    # docstring) — real-scan accuracy is measured against real scans
    # (§11 risk #2), not synthetic Pillow-rendered text. What this test
    # actually needs to prove is that the pipeline extracts *some* correct
    # Arabic and *some* correct English from a mixed-script image, not
    # perfect character fidelity on either.
    arabic_words_found = sum(
        word in result.raw_text for word in ("مرحبا", "بكم", "في", "مستخرج", "النص", "الذكي")
    )
    assert arabic_words_found >= 4, f"expected most Arabic words recognized, got: {result.raw_text!r}"
    assert "Smart Text Extractor" in result.raw_text
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
def test_engine_applies_preprocessing_to_a_degraded_image_real_world_regression(ocr_engine) -> None:
    """Regression test for a real bug found from live user feedback ("Arabic
    extraction is weak"): OcrEngine.run() sent the raw image straight to
    Tesseract, silently skipping the deskew/contrast/denoise pipeline that
    had been built and tested but never wired in. Manually confirmed: on
    this exact degraded image, the un-preprocessed path returned only 3
    garbage characters ('ا ل ل') instead of the sentence — see
    docs/phases/phase-2-ocr-pipeline.md for the full before/after."""
    text = "مرحباً بكم في مستخرج النص الذكي وهذا اختبار لجودة الاستخراج"
    image = make_degraded_arabic_image(text)

    result = ocr_engine.run(image)

    expected_words = ("مرحبا", "بكم", "في", "مستخرج", "النص", "الذكي", "اختبار", "الاستخراج")
    found = sum(word in result.raw_text for word in expected_words)
    assert found >= 6, f"expected most words recognized from the degraded image, got: {result.raw_text!r}"


@requires_tesseract
@requires_arabic_font
def test_engine_on_blank_page_returns_empty_result(ocr_engine) -> None:
    import numpy as np

    blank = np.full((200, 400, 3), 255, dtype="uint8")
    result = ocr_engine.run(blank)

    assert result.raw_text.strip() == ""
    assert result.word_boxes == []
    assert result.confidence_score == 0.0
