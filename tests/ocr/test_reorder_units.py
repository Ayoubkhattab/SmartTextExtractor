"""Pure-logic unit tests for reorder.py — no real Tesseract needed, unlike
test_engine.py/test_reorder.py which exercise the real binary."""
from __future__ import annotations

from smart_text_extractor.core.models import BoundingBox, Rect, TextSegment
from smart_text_extractor.ocr.reorder import (
    Line,
    _block_is_tabular,
    _is_majority_arabic,
    _is_mostly_latin,
    _line_cells,
    _line_segments_with_cell_separators,
    _line_text_with_cell_separators,
    _rows_to_markdown_table,
    _split_line_into_column_runs,
    assemble_markdown,
    assemble_text,
    assemble_text_segments,
    correct_known_arabic_misreads,
    group_into_lines,
    merge_dual_language_passes,
    order_lines_reading_order,
    words_from_tsv,
)


def _fake_tsv(rows: list[tuple[str, float, int, int, int, int, int, int]]) -> dict:
    """rows: (text, conf, left, top, width, height, block, line)"""
    return {
        "text": [r[0] for r in rows],
        "conf": [r[1] for r in rows],
        "left": [r[2] for r in rows],
        "top": [r[3] for r in rows],
        "width": [r[4] for r in rows],
        "height": [r[5] for r in rows],
        "block_num": [r[6] for r in rows],
        "par_num": [0 for _ in rows],
        "line_num": [r[7] for r in rows],
    }


def test_words_from_tsv_drops_empty_and_negative_confidence_rows() -> None:
    data = _fake_tsv(
        [
            ("", 95.0, 0, 0, 10, 10, 1, 1),  # empty text — a block/line summary row
            ("Hello", -1.0, 0, 0, 10, 10, 1, 1),  # negative conf — also a summary row
            ("World", 88.5, 20, 0, 30, 10, 1, 1),
        ]
    )
    result = words_from_tsv(data)
    assert len(result) == 1
    box, block_num, par_num, line_num = result[0]
    assert box.text == "World"
    assert box.confidence == 88.5
    assert (block_num, par_num, line_num) == (1, 0, 1)


def test_words_from_tsv_strips_stray_bidi_control_marks() -> None:
    """Regression test: real Tesseract output on every mixed-script page
    tested this session embedded LRM/RLM marks directly in word text
    ('Jods‏', '٠‏', 'Smart‎') — zero-width bidi
    artifacts with no visible glyph, cluttering raw_text."""
    data = _fake_tsv(
        [
            ("Jods‏", 66.0, 0, 0, 40, 10, 1, 1),
            ("٠‏", 91.0, 50, 0, 5, 10, 1, 1),  # Arabic-Indic zero, digit + RLM
            ("‏", 80.0, 60, 0, 2, 10, 1, 1),  # a lone stray mark, nothing else — must be dropped entirely
        ]
    )
    result = words_from_tsv(data)
    texts = [box.text for box, *_ in result]
    assert texts == ["Jods", "٠"]


def test_group_into_lines_preserves_first_seen_order() -> None:
    box_a = (BoundingBox("a", Rect(0, 0, 10, 10), 90.0), 1, 0, 1)
    box_b = (BoundingBox("b", Rect(20, 0, 10, 10), 90.0), 1, 0, 1)
    box_c = (BoundingBox("c", Rect(0, 50, 10, 10), 90.0), 1, 0, 2)

    lines = group_into_lines([box_a, box_b, box_c])

    assert len(lines) == 2
    assert [w.text for w in lines[0].words] == ["a", "b"]
    assert [w.text for w in lines[1].words] == ["c"]


def test_is_majority_arabic() -> None:
    assert _is_majority_arabic("مرحبا بكم") is True
    assert _is_majority_arabic("Hello World") is False
    assert _is_majority_arabic("123 456") is False  # no letters at all
    assert _is_majority_arabic("") is False


def test_split_line_into_column_runs_splits_on_a_wide_gap() -> None:
    line = Line(
        words=[
            BoundingBox("First", Rect(30, 0, 60, 20), 90.0),
            BoundingBox("Second", Rect(100, 0, 60, 20), 90.0),
            BoundingBox("far", Rect(600, 0, 40, 20), 90.0),
        ],
        block_num=1,
        par_num=0,
        line_num=1,
    )
    runs = _split_line_into_column_runs(line)
    assert len(runs) == 2
    assert [w.text for w in runs[0]] == ["First", "Second"]
    assert [w.text for w in runs[1]] == ["far"]


def test_split_line_into_column_runs_no_split_when_words_are_close() -> None:
    line = Line(
        words=[
            BoundingBox("First", Rect(30, 0, 60, 20), 90.0),
            BoundingBox("Second", Rect(100, 0, 60, 20), 90.0),
        ],
        block_num=1,
        par_num=0,
        line_num=1,
    )
    runs = _split_line_into_column_runs(line)
    assert len(runs) == 1
    assert [w.text for w in runs[0]] == ["First", "Second"]


def test_is_mostly_latin() -> None:
    assert _is_mostly_latin("Hello") is True
    assert _is_mostly_latin("مرحبا") is False
    assert _is_mostly_latin("") is False
    assert _is_mostly_latin("123") is False  # digits aren't letters


def _tagged(text: str, confidence: float, rect: Rect) -> tuple[BoundingBox, int, int, int]:
    """A words_from_tsv()-shaped entry, for tests that build fixtures
    directly instead of going through a fake TSV dict."""
    return (BoundingBox(text=text, rect=rect, confidence=confidence), 1, 0, 1)


class TestMergeDualLanguagePasses:
    """Regression tests for a real bug (docs/phases/phase-2-ocr-pipeline.md):
    running lang="ara+eng" sometimes misclassifies isolated Arabic words as
    Latin garbage. All fixtures use the exact real words/confidences
    captured from the document that exposed this."""

    def test_replaces_low_confidence_latin_word_with_higher_confidence_arabic_match(self) -> None:
        rect = Rect(100, 50, 30, 20)
        primary = [_tagged("Fro", 18.0, rect)]
        arabic_only = [_tagged("صريح", 27.0, rect)]

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "صريح"

    def test_keeps_high_confidence_english_word_over_a_lower_confidence_arabic_match(self) -> None:
        """The exact real regression: "Plan" (conf 96) has an ara-only
        "alternative" ("مقا", conf 52) that IS mostly-Arabic-script but is
        actually garbage from forcing Arabic classification onto English
        glyphs — lower confidence than the correct primary reading."""
        rect = Rect(200, 50, 40, 20)
        primary = [_tagged("Plan", 96.0, rect)]
        arabic_only = [_tagged("مقا", 52.0, rect)]

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "Plan"

    def test_keeps_english_word_when_arabic_pass_produces_non_arabic_garbage(self) -> None:
        """"Software" -> ara-only pass produces "5011100121" (digit soup,
        not Arabic script) — must never replace real English regardless
        of confidence, since it isn't a valid alternative at all."""
        rect = Rect(0, 0, 60, 20)
        primary = [_tagged("Software", 96.0, rect)]
        arabic_only = [_tagged("5011100121", 99.0, rect)]  # even if "confident", it's not Arabic

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "Software"

    def test_leaves_arabic_words_untouched(self) -> None:
        rect = Rect(0, 0, 50, 20)
        primary = [_tagged("مرحبا", 90.0, rect)]
        arabic_only = [_tagged("مرحبا", 10.0, rect)]  # irrelevant — primary was never Latin

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "مرحبا"
        assert merged[0][0].confidence == 90.0

    def test_no_overlapping_arabic_word_leaves_primary_untouched(self) -> None:
        primary = [_tagged("Fro", 18.0, Rect(100, 50, 30, 20))]
        arabic_only = [_tagged("شيء", 99.0, Rect(500, 500, 30, 20))]  # far away, no overlap

        merged = merge_dual_language_passes(primary, arabic_only)

        assert merged[0][0].text == "Fro"


class TestCorrectKnownArabicMisreads:
    """Regression test for a real, confirmed-systematic misread
    (docs/phases/phase-2-ocr-pipeline.md): "مهندس" (engineer) read as
    "ميندس" on every single one of 7 occurrences on a real page (a
    large role/skills table, likely small table-cell font size straining
    the ه/ي distinction). Unlike the rejected general fuzzy-match
    dictionary idea, "ميندس" isn't a real Arabic word with a legitimate
    alternate meaning, so there's no ambiguity to weigh — this is a
    narrow, hand-verified substitution, not a heuristic guess.
    """

    def test_known_misread_is_corrected(self) -> None:
        words = [_tagged("ميندس", 39.0, Rect(0, 0, 60, 20))]

        corrected = correct_known_arabic_misreads(words)

        assert corrected[0][0].text == "مهندس"
        assert corrected[0][0].confidence == 39.0  # confidence carries over unchanged

    def test_unrelated_words_are_left_untouched(self) -> None:
        words = [_tagged("مهندس", 92.0, Rect(0, 0, 60, 20)), _tagged("مرحبا", 90.0, Rect(70, 0, 60, 20))]

        corrected = correct_known_arabic_misreads(words)

        assert [box.text for box, *_ in corrected] == ["مهندس", "مرحبا"]


class TestOrderLinesReadingOrderPerBlock:
    """Regression test for a real bug (docs/phases/phase-2-ocr-pipeline.md):
    clustering columns globally across the whole page scrambled reading
    order whenever a full-width paragraph line's x-range overlapped
    several narrower columns below it (confirmed real on a page with a
    3-box side-by-side flow diagram below a full-width paragraph) — they
    all got lumped into "one column" together. Column clustering must be
    scoped per Tesseract block, ordering blocks top-to-bottom.
    """

    def test_narrow_multi_column_row_below_a_full_width_paragraph_orders_correctly(self) -> None:
        # block 1: one full-width paragraph line spanning nearly the page.
        paragraph_words = [(BoundingBox("فقرة", Rect(10, 0, 900, 30), 90.0), 1, 0, 1)]
        # block 2: a single Tesseract "line" that actually spans 3
        # side-by-side boxes (the same-row-merge bug
        # _split_line_into_column_runs handles) — words given in the
        # original, already-correct order Tesseract produces: right box
        # first, then middle, then left.
        box_row_words = [
            (BoundingBox("يمين", Rect(700, 100, 100, 30), 90.0), 2, 0, 1),
            (BoundingBox("وسط", Rect(400, 100, 100, 30), 90.0), 2, 0, 1),
            (BoundingBox("يسار", Rect(50, 100, 100, 30), 90.0), 2, 0, 1),
        ]
        lines = group_into_lines(paragraph_words + box_row_words)

        ordered = order_lines_reading_order(lines)

        texts = [w.text for line in ordered for w in line.words]
        assert texts == ["فقرة", "يمين", "وسط", "يسار"]


class TestLineTextWithCellSeparators:
    """Regression tests for a real complaint (docs/phases/phase-2-ocr-pipeline.md):
    table rows read as an unreadable wall of words because every cell was
    joined with a plain space, indistinguishable from words within a
    cell. Fixtures for the "real row" case use the exact word
    left/width values captured from Tesseract TSV for one real table row
    on page 3 of هيكلية القسم والمكاتب.pdf ("تخطيط الدورة | بداية كل دورة
    ٠ ساعتان | المكاتب الخمسة + رئيس القسم | سجل الدورة مع تقديرات وبنود
    مستوفية الجاهزية"), given here in Tesseract's own (already-correct,
    right-to-left) word order.
    """

    def test_real_table_row_gets_separators_at_the_three_real_cell_boundaries(self) -> None:
        # (text, left, width) — y/height are irrelevant to this function, held constant.
        real_row = [
            ("تخطيط", 2135, 104),
            ("الدورة", 2039, 84),
            ("بداية", 1799, 67),
            ("كل", 1759, 28),
            ("دورة", 1687, 58),
            ("٠", 1667, 5),
            ("ساعتان", 1554, 99),
            ("المكاتب", 1348, 105),
            ("الخمسة", 1226, 109),
            ("+", 1197, 17),
            ("رئيس", 1111, 73),
            ("القسم", 1012, 85),
            ("سجل", 842, 76),
            ("الدورة", 751, 78),
            ("مع", 701, 38),
            ("تقديرات", 586, 106),
            ("وبنود", 504, 70),
            ("مستوفية", 371, 122),
            ("الجاهزية", 243, 116),
        ]
        line = Line(
            words=[BoundingBox(text, Rect(left, 1325, width, 37), 90.0) for text, left, width in real_row],
            block_num=10,
            par_num=1,
            line_num=1,
        )

        text = _line_text_with_cell_separators(line)

        assert text == (
            "تخطيط الدورة | بداية كل دورة ٠ ساعتان | المكاتب الخمسة + رئيس القسم"
            " | سجل الدورة مع تقديرات وبنود مستوفية الجاهزية"
        )

    def test_no_separator_when_all_gaps_are_uniform_prose_spacing(self) -> None:
        """Same page, a genuine prose line (block 2, no table): real gaps
        clustered at 10-18px with no outliers — must stay plain-spaced."""
        prose_row = [
            ("بنمط", 2096, 80),
            ("مزدوج:", 1969, 112),
            ("بنودها", 1850, 102),
            ("التطويرية", 1685, 147),
            ("تدخل", 1579, 94),
            ("الدورة.", 1458, 105),
        ]
        line = Line(
            words=[BoundingBox(text, Rect(left, 578, width, 45), 90.0) for text, left, width in prose_row],
            block_num=2,
            par_num=1,
            line_num=3,
        )

        text = _line_text_with_cell_separators(line)

        assert "|" not in text
        assert text == "بنمط مزدوج: بنودها التطويرية تدخل الدورة."

    def test_single_word_line_returns_its_own_text_unchanged(self) -> None:
        line = Line(words=[BoundingBox("وحيدة", Rect(0, 0, 50, 20), 90.0)], block_num=1, par_num=0, line_num=1)
        assert _line_text_with_cell_separators(line) == "وحيدة"


class TestAssembleTextParagraphSpacing:
    """Regression test for a real complaint: extracted text read as one
    undifferentiated wall of text. Different Tesseract blocks (paragraphs,
    table regions, diagram boxes) must be visually separated by a blank
    line in the assembled output, while lines within one block stay
    single-spaced.
    """

    def test_different_blocks_get_a_blank_line_between_them_same_block_does_not(self) -> None:
        lines = [
            Line(words=[BoundingBox("a", Rect(0, 0, 10, 10), 90.0)], block_num=1, par_num=0, line_num=1),
            Line(words=[BoundingBox("b", Rect(0, 20, 10, 10), 90.0)], block_num=1, par_num=0, line_num=2),
            Line(words=[BoundingBox("c", Rect(0, 50, 10, 10), 90.0)], block_num=2, par_num=0, line_num=1),
        ]

        text = assemble_text(lines)

        assert text == "a\nb\n\nc"

    def test_empty_lines_list_returns_empty_string(self) -> None:
        assert assemble_text([]) == ""


class TestSegments:
    """Regression tests for the segment-based representation added so the
    UI can highlight low-confidence words (§7.1.1): word segments carry
    confidence, every separator (space, " | ", "\\n", "\\n\\n") carries
    None, and concatenating every segment's text must always reproduce
    exactly what the plain-string functions return — the two must never
    be able to drift apart, since the UI renders one and edits the other.
    """

    def test_line_segments_mark_words_with_confidence_and_separators_with_none(self) -> None:
        line = Line(
            words=[
                BoundingBox("First", Rect(0, 0, 60, 20), 91.0),
                BoundingBox("Second", Rect(70, 0, 60, 20), 42.0),
            ],
            block_num=1,
            par_num=0,
            line_num=1,
        )

        segments = _line_segments_with_cell_separators(line)

        assert segments == [
            TextSegment("First", 91.0),
            TextSegment(" ", None),
            TextSegment("Second", 42.0),
        ]

    def test_line_segments_concatenation_matches_the_string_function(self) -> None:
        real_row = [
            ("تخطيط", 2135, 104),
            ("الدورة", 2039, 84),
            ("بداية", 1799, 67),  # real cell boundary here (173px gap)
        ]
        line = Line(
            words=[BoundingBox(text, Rect(left, 1325, width, 37), 90.0) for text, left, width in real_row],
            block_num=10,
            par_num=1,
            line_num=1,
        )

        segments = _line_segments_with_cell_separators(line)

        assert "".join(s.text for s in segments) == _line_text_with_cell_separators(line)
        assert segments[1] == TextSegment(" ", None)  # within the "تخطيط الدورة" cell
        assert segments[3] == TextSegment(" | ", None)  # the real cell boundary

    def test_assemble_text_segments_concatenation_matches_assemble_text(self) -> None:
        lines = [
            Line(words=[BoundingBox("a", Rect(0, 0, 10, 10), 90.0)], block_num=1, par_num=0, line_num=1),
            Line(words=[BoundingBox("b", Rect(0, 20, 10, 10), 30.0)], block_num=1, par_num=0, line_num=2),
            Line(words=[BoundingBox("c", Rect(0, 50, 10, 10), 90.0)], block_num=2, par_num=0, line_num=1),
        ]

        segments = assemble_text_segments(lines)

        assert "".join(s.text for s in segments) == assemble_text(lines) == "a\nb\n\nc"
        assert segments == [
            TextSegment("a", 90.0),
            TextSegment("\n", None),
            TextSegment("b", 30.0),
            TextSegment("\n\n", None),
            TextSegment("c", 90.0),
        ]

    def test_assemble_text_segments_on_empty_lines_returns_empty_list(self) -> None:
        assert assemble_text_segments([]) == []


def _one_word_line(text: str, height: int, block_num: int) -> Line:
    return Line(words=[BoundingBox(text, Rect(0, 0, 60, height), 90.0)], block_num=block_num, par_num=0, line_num=1)


class TestAssembleMarkdown:
    """Tests for the Markdown export (§7.1.1 extension) — real tables
    become Markdown tables, a real heading becomes '## ...'."""

    def test_real_world_regression_heading_line_flagged_but_table_header_and_title_are_not(self) -> None:
        """The exact 20 real word heights measured across every line of
        page 3, هيكلية القسم والمكاتب.pdf (docs/phases/phase-2-ocr-pipeline.md)
        — median 38px. The real section heading (74px, ratio 1.95) must
        be flagged; the real table header row (60px, ratio 1.58) and the
        page's own title (39px — apparently bolded, not enlarged, in the
        source) must not be, per _HEADING_HEIGHT_RATIO's calibration.
        """
        real_heights = [39, 74, 44, 46, 45, 29, 46, 31, 46, 60, 36, 38, 35, 38, 37, 35, 30, 43, 37, 37]
        lines = [_one_word_line(f"word{i}", h, block_num=i) for i, h in enumerate(real_heights)]
        lines[0] = _one_word_line("مقترح", 39, block_num=0)  # the real title
        lines[1] = _one_word_line("الإطار", 74, block_num=1)  # the real heading
        lines[9] = _one_word_line("التواتر", 60, block_num=9)  # the real table header row
        lines[16] = _one_word_line("Definition", 30, block_num=16)  # "Definition of Done" box label

        markdown = assemble_markdown(lines)

        assert "## الإطار" in markdown
        assert "مقترح" in markdown and "## مقترح" not in markdown
        assert "التواتر" in markdown and "## التواتر" not in markdown
        assert "Definition" in markdown and "## Definition" not in markdown

    def test_line_cells_splits_on_a_real_cell_boundary(self) -> None:
        # 3 real words from the real table row (docs/phases/phase-2-ocr-pipeline.md):
        # "الدورة"|"بداية كل" — needs a 3rd word so the heuristic has a
        # small reference gap (12px, بداية↔كل) to judge the 173px real
        # boundary gap (الدورة↔بداية) against — a 2-word line's one gap
        # can never be an outlier relative to itself, by construction.
        rect_start = Rect(2039, 1325, 84, 37)  # "الدورة"
        rect_mid = Rect(1799, 1325, 67, 37)  # "بداية"
        rect_end = Rect(1759, 1325, 28, 37)  # "كل"
        line = Line(
            words=[
                BoundingBox("الدورة", rect_start, 90.0),
                BoundingBox("بداية", rect_mid, 90.0),
                BoundingBox("كل", rect_end, 90.0),
            ],
            block_num=1,
            par_num=0,
            line_num=1,
        )
        assert _line_cells(line) == ["الدورة", "بداية كل"]

    def test_line_cells_returns_single_cell_when_no_boundary_detected(self) -> None:
        line = Line(
            words=[BoundingBox("First", Rect(0, 0, 60, 20), 90.0), BoundingBox("Second", Rect(70, 0, 60, 20), 90.0)],
            block_num=1,
            par_num=0,
            line_num=1,
        )
        assert _line_cells(line) == ["First Second"]

    @staticmethod
    def _two_cell_row(right_text: str, mid_text: str, left_text: str, block_num: int, line_num: int) -> Line:
        """A 2-cell row: `mid`+`left` sit close together (30px gap) forming
        one (2-word) cell, `right` sits far away (350px gap) forming the
        other, alone. Verified arithmetic: min_gap=30,
        threshold=max(30*4,40)=120, so the 350px gap splits (350>120) and
        the 30px one doesn't. 3 words are required, not 2 — a 2-word
        line's one gap can never be an outlier relative to itself — see
        test_line_cells_splits_on_a_real_cell_boundary.
        """
        return Line(
            words=[
                BoundingBox(right_text, Rect(500, 0, 60, 20), 90.0),
                BoundingBox(mid_text, Rect(90, 0, 60, 20), 90.0),
                BoundingBox(left_text, Rect(0, 0, 60, 20), 90.0),
            ],
            block_num=block_num,
            par_num=0,
            line_num=line_num,
        )

    def test_two_cell_row_helper_actually_splits_into_two_cells(self) -> None:
        row = self._two_cell_row("R", "M", "L", block_num=1, line_num=1)
        assert _line_cells(row) == ["R", "M L"]  # cell 2 is naturally 2 words, like a real table cell often is

    def test_block_is_tabular_requires_a_majority_not_every_row(self) -> None:
        def plain_row(n: int) -> Line:
            return Line(words=[BoundingBox(f"plain{n}", Rect(0, 0, 60, 20), 90.0)], block_num=1, par_num=0, line_num=n)

        row1 = self._two_cell_row("a1", "b1", "c1", block_num=1, line_num=1)
        row2 = self._two_cell_row("a2", "b2", "c2", block_num=1, line_num=2)

        assert _block_is_tabular([row1, row2, plain_row(3)]) is True  # 2/3 rows split
        assert _block_is_tabular([row1, plain_row(2), plain_row(3)]) is False  # only 1/3 rows split
        assert _block_is_tabular([row1]) is False  # a single row is never "a table"

    def test_rows_to_markdown_table_pads_ragged_rows(self) -> None:
        def cell(text: str) -> list[TextSegment]:
            return [TextSegment(text, 90.0)]

        rows = [[cell("A"), cell("B"), cell("C")], [cell("1"), cell("2")]]

        table = _rows_to_markdown_table(rows)

        lines = table.splitlines()
        assert lines[0] == "| A | B | C |"
        assert lines[1] == "| --- | --- | --- |"
        assert lines[2] == "| 1 | 2 |  |"  # padded to 3 columns with an empty cell

    def test_assemble_markdown_renders_a_multi_row_table_and_a_plain_paragraph(self) -> None:
        # height=20 to match _two_cell_row's word heights below — keeps
        # this test focused on table-vs-paragraph rendering, not heading
        # detection (a mismatched height here would pull the page median
        # down and wrongly flag this line as a heading by comparison).
        paragraph_line = _one_word_line("فقرة", 20, block_num=1)
        table_rows = [
            self._two_cell_row("خلية1ب", "M1", "خلية1أ", block_num=2, line_num=1),
            self._two_cell_row("خلية2ب", "M2", "خلية2أ", block_num=2, line_num=2),
        ]

        markdown = assemble_markdown([paragraph_line, *table_rows])

        assert markdown.startswith("فقرة\n\n")
        table_part = markdown.split("\n\n", 1)[1]
        assert table_part.splitlines()[1] == "| --- | --- |"

    def test_isolated_tabular_line_merges_as_header_for_the_table_block_after_it(self) -> None:
        """Illustrates the header-merge mechanism with a clean synthetic
        gap (honesty note: on the real page this was calibrated against,
        the real table header row did NOT itself have a detectable cell
        gap, so this exact path doesn't fire there — see
        assemble_markdown's docstring. This proves the mechanism works
        when a header row DOES have one, which real tables with a
        differently-styled header bar plausibly could.)
        """
        header_row = self._two_cell_row("Col2Header", "Col1", "Header", block_num=1, line_num=1)
        data_row_1 = self._two_cell_row("R1C2", "R1", "C1", block_num=2, line_num=1)
        data_row_2 = self._two_cell_row("R2C2", "R2", "C1", block_num=2, line_num=2)

        markdown = assemble_markdown([header_row, data_row_1, data_row_2])

        lines = markdown.splitlines()
        assert lines[0] == "| Col2Header | Col1 Header |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| R1C2 | R1 C1 |"
        assert lines[3] == "| R2C2 | R2 C1 |"

    def test_assemble_markdown_on_empty_lines_returns_empty_string(self) -> None:
        assert assemble_markdown([]) == ""
