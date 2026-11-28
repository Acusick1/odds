"""Tests for the long UK retail outlier rule."""

from __future__ import annotations

from decimal import Decimal

from odds_core.models import Price
from odds_core.vocabulary import Market, PriceOutcome, Side
from odds_lambda.storage.retail_outliers import long_retail_outliers

HOME, DRAW, AWAY = PriceOutcome.HOME, PriceOutcome.DRAW, PriceOutcome.AWAY


def back(book: str, outcome: PriceOutcome, price: str, line: str | None = None) -> Price:
    return Price(
        bookmaker=book,
        outcome=outcome,
        side=Side.BACK,
        price=Decimal(price),
        line=None if line is None else Decimal(line),
    )


# Exchange fair home probability 1/3.1 / (1/3.1 + 1/3.5 + 1/2.5) ~ 0.32, overround ~ 1.1%.
EXCHANGE = [
    back("betfair_exchange", HOME, "3.1"),
    back("betfair_exchange", DRAW, "3.5"),
    back("betfair_exchange", AWAY, "2.5"),
]


class TestLongRetailOutliers:
    def test_flags_a_price_twice_the_next_longest_and_above_half_ev(self) -> None:
        prices = [*EXCHANGE, back("williamhill", HOME, "20"), back("bet365", HOME, "3.0")]

        assert long_retail_outliers(Market.ONE_X_TWO, prices) == [3]

    def test_a_price_under_twice_the_next_longest_is_kept(self) -> None:
        # EV about +157%, ratio 1.9.
        prices = [*EXCHANGE, back("williamhill", HOME, "8"), back("bet365", HOME, "4.2")]

        assert long_retail_outliers(Market.ONE_X_TWO, prices) == []

    def test_a_price_at_most_half_ev_is_kept(self) -> None:
        # Ratio 2.25, EV about +45%.
        prices = [*EXCHANGE, back("williamhill", HOME, "4.5"), back("bet365", HOME, "2.0")]

        assert long_retail_outliers(Market.ONE_X_TWO, prices) == []

    def test_no_exchange_price_flags_nothing(self) -> None:
        prices = [back("williamhill", HOME, "20"), back("bet365", HOME, "3.0")]

        assert long_retail_outliers(Market.ONE_X_TWO, prices) == []

    def test_an_exchange_missing_an_outcome_flags_nothing(self) -> None:
        prices = [*EXCHANGE[:2], back("williamhill", HOME, "20"), back("bet365", HOME, "3.0")]

        assert long_retail_outliers(Market.ONE_X_TWO, prices) == []

    def test_an_exchange_over_the_overround_cap_flags_nothing(self) -> None:
        wide = [
            back("betfair_exchange", HOME, "2.8"),
            back("betfair_exchange", DRAW, "3.2"),
            back("betfair_exchange", AWAY, "2.3"),
        ]
        prices = [*wide, back("williamhill", HOME, "20"), back("bet365", HOME, "3.0")]

        assert long_retail_outliers(Market.ONE_X_TWO, prices) == []

    def test_a_lone_retail_price_has_no_next_longest(self) -> None:
        prices = [*EXCHANGE, back("williamhill", HOME, "20")]

        assert long_retail_outliers(Market.ONE_X_TWO, prices) == []

    def test_books_outside_uk_retail_are_neither_flagged_nor_compared(self) -> None:
        prices = [*EXCHANGE, back("pinnacle", HOME, "20"), back("bet365", HOME, "3.0")]

        assert long_retail_outliers(Market.ONE_X_TWO, prices) == []

    def test_lay_prices_are_ignored(self) -> None:
        lay = Price(bookmaker="williamhill", outcome=HOME, side=Side.LAY, price=Decimal("20"))

        assert (
            long_retail_outliers(Market.ONE_X_TWO, [*EXCHANGE, lay, back("bet365", HOME, "3")])
            == []
        )

    def test_totals_compare_within_a_line(self) -> None:
        prices = [
            back("betfair_exchange", PriceOutcome.OVER, "2.0", "2.5"),
            back("betfair_exchange", PriceOutcome.UNDER, "2.0", "2.5"),
            back("williamhill", PriceOutcome.OVER, "6", "2.5"),
            back("bet365", PriceOutcome.OVER, "1.9", "2.5"),
            # Another line's price is not the next-longest on 2.5.
            back("bet365", PriceOutcome.OVER, "5.5", "1.5"),
        ]

        assert long_retail_outliers(Market.TOTALS, prices) == [2]

    def test_spreads_are_not_guarded(self) -> None:
        prices = [
            back("betfair_exchange", HOME, "2.0", "-5.5"),
            back("betfair_exchange", AWAY, "2.0", "5.5"),
            back("williamhill", HOME, "6", "-5.5"),
            back("bet365", HOME, "1.9", "-5.5"),
        ]

        assert long_retail_outliers(Market.SPREADS, prices) == []
