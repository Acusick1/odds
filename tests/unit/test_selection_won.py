import pytest
from odds_core.odds_math import selection_won


class TestSelectionWon:
    @pytest.mark.parametrize(
        ("selection", "line", "home", "away", "won"),
        [
            ("over", 2.5, 2, 1, True),
            ("over", 2.5, 1, 1, False),
            ("under", 2.5, 1, 1, True),
            ("under", 2.5, 2, 1, False),
            ("over", 0.5, 0, 0, False),
            ("under", 0.5, 0, 0, True),
        ],
    )
    def test_over_under_compare_total_with_line(
        self, selection: str, line: float, home: int, away: int, won: bool
    ) -> None:
        assert selection_won(selection, line, home, away) is won

    @pytest.mark.parametrize("total", range(8))
    def test_half_line_has_exactly_one_winner(self, total: int) -> None:
        assert selection_won("over", 2.5, total, 0) != selection_won("under", 2.5, total, 0)

    @pytest.mark.parametrize(
        ("selection", "home", "away", "won"),
        [
            ("home", 2, 1, True),
            ("home", 1, 1, False),
            ("draw", 1, 1, True),
            ("away", 0, 1, True),
            ("away", 2, 1, False),
        ],
    )
    def test_one_x_two_is_the_result(self, selection: str, home: int, away: int, won: bool) -> None:
        assert selection_won(selection, None, home, away) is won

    @pytest.mark.parametrize("line", [None, 2.0, 3.0])
    def test_over_under_without_a_half_line_raises(self, line: float | None) -> None:
        with pytest.raises(ValueError, match="half line"):
            selection_won("over", line, 2, 1)

    def test_one_x_two_with_a_line_raises(self) -> None:
        with pytest.raises(ValueError, match="takes no line"):
            selection_won("home", 2.5, 2, 1)
