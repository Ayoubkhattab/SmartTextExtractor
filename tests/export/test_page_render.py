

class TestColumnEdges:
    def test_measured_proportions_are_used(self) -> None:
        from smart_text_extractor.export.page_render import _column_edges

        edges = _column_edges([0.25, 0.5, 0.25], 3, 0.0, 400.0)

        assert edges == [0.0, 100.0, 300.0, 400.0]

    def test_a_mismatched_count_falls_back_to_an_even_split(self) -> None:
        """The model promises an even split when it could not measure the
        source's own geometry."""
        from smart_text_extractor.export.page_render import _column_edges

        assert _column_edges([], 2, 0.0, 100.0) == [0.0, 50.0, 100.0]
        assert _column_edges([0.5, 0.5], 4, 0.0, 100.0) == [0.0, 25.0, 50.0, 75.0, 100.0]

    def test_the_last_edge_lands_exactly_on_the_boundary(self) -> None:
        """Accumulated rounding must not leave the final column short of
        the table's right edge."""
        from smart_text_extractor.export.page_render import _column_edges

        edges = _column_edges([1 / 3, 1 / 3, 1 / 3], 3, 10.0, 310.0)

        assert edges[-1] == 310.0
