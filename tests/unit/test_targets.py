"""Unit tests for training-label target functions."""

from datetime import UTC, datetime

import pytest
from odds_analytics.targets import calculate_regression_target
from odds_core.odds_math import american_to_decimal, devig_probabilities
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis

TIMESTAMP = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)


def _quote(
    bookmaker: str,
    american_price: int,
    *,
    market: Market = Market.H2H,
    outcome: PriceOutcome = PriceOutcome.HOME,
    line: float | None = None,
) -> Quote:
    """A back quote of the home (or given) outcome, priced from American odds."""
    return Quote(
        snapshot_id=1,
        event_id=1,
        snapshot_time=TIMESTAMP,
        created_at=TIMESTAMP,
        source=Source.ODDS_API,
        market=market,
        time_basis=TimeBasis.OBSERVED,
        in_play=None,
        market_status=None,
        total_matched=None,
        market_start_time=None,
        bookmaker=bookmaker,
        outcome=outcome,
        line=line,
        side=Side.BACK,
        price=american_to_decimal(american_price),
        size=None,
        matched=None,
        changed_at=None,
    )


class TestCalculateRegressionTarget:
    """Tests for calculate_regression_target helper function."""

    def test_h2h_probability_delta(self) -> None:
        """Test h2h market uses implied probability delta."""
        opening = [_quote("pinnacle", -120)]  # ~54.5% implied prob
        closing = [_quote("pinnacle", -150)]  # ~60% implied prob

        result = calculate_regression_target(opening, closing, "h2h")

        assert result is not None
        # Probability should increase (line moved in favor of the outcome)
        assert result > 0
        # Approximately 60% - 54.5% = 5.5%
        assert 0.04 < result < 0.07

    def test_spreads_point_delta(self) -> None:
        """Test spreads market uses point delta."""
        opening = [_quote("pinnacle", -110, market=Market.SPREADS, line=-2.5)]
        closing = [_quote("pinnacle", -110, market=Market.SPREADS, line=-3.5)]

        result = calculate_regression_target(opening, closing, "spreads")

        assert result is not None
        # Point delta: -3.5 - (-2.5) = -1.0
        assert result == -1.0

    def test_totals_point_delta(self) -> None:
        """Test totals market uses point delta."""
        opening = [
            _quote("pinnacle", -110, market=Market.TOTALS, outcome=PriceOutcome.OVER, line=215.5)
        ]
        closing = [
            _quote("pinnacle", -110, market=Market.TOTALS, outcome=PriceOutcome.OVER, line=218.5)
        ]

        result = calculate_regression_target(opening, closing, "totals")

        assert result is not None
        # Point delta: 218.5 - 215.5 = 3.0
        assert result == 3.0

    def test_missing_opening_quotes(self) -> None:
        """Test handling of missing opening quotes."""
        assert calculate_regression_target(None, [_quote("pinnacle", -150)], "h2h") is None

    def test_missing_closing_quotes(self) -> None:
        """Test handling of missing closing quotes."""
        assert calculate_regression_target([_quote("pinnacle", -120)], None, "h2h") is None

    def test_spreads_without_lines(self) -> None:
        """A point market whose quotes carry no line has no delta."""
        opening = [_quote("pinnacle", -110, market=Market.SPREADS)]
        closing = [_quote("pinnacle", -110, market=Market.SPREADS, line=-3.5)]

        assert calculate_regression_target(opening, closing, "spreads") is None

    def test_unknown_market(self) -> None:
        """Test handling of unknown market type."""
        opening = [_quote("pinnacle", -120)]
        closing = [_quote("pinnacle", -150)]

        assert calculate_regression_target(opening, closing, "unknown") is None

    def test_multiple_bookmakers_average(self) -> None:
        """Test averaging across multiple bookmakers."""
        opening = [
            _quote("pinnacle", -110, market=Market.SPREADS, line=-2.5),
            _quote("fanduel", -110, market=Market.SPREADS, line=-3.5),
        ]
        closing = [
            _quote("pinnacle", -110, market=Market.SPREADS, line=-4.5),
            _quote("fanduel", -110, market=Market.SPREADS, line=-5.5),
        ]

        result = calculate_regression_target(opening, closing, "spreads")

        assert result is not None
        # Opening avg: (-2.5 + -3.5) / 2 = -3.0
        # Closing avg: (-4.5 + -5.5) / 2 = -5.0
        # Delta: -5.0 - (-3.0) = -2.0
        assert result == -2.0


class TestDevigProbabilities:
    """Tests for devig_probabilities utility."""

    def test_symmetric_overround(self):
        """Equal vig on both sides normalizes to 0.5/0.5."""
        home, away = devig_probabilities(0.524, 0.524)
        assert home == pytest.approx(0.5)
        assert away == pytest.approx(0.5)

    def test_no_vig(self):
        """Probabilities already summing to 1 are unchanged."""
        home, away = devig_probabilities(0.6, 0.4)
        assert home == pytest.approx(0.6)
        assert away == pytest.approx(0.4)

    def test_zero_total(self):
        """Zero total returns 0.5/0.5 fallback."""
        home, away = devig_probabilities(0.0, 0.0)
        assert home == 0.5
        assert away == 0.5

    def test_typical_nba_vig(self):
        """Typical -110/-110 line: both ~0.524, devig to 0.5."""
        # -110 American = 1/1.909 ≈ 0.5238
        home_raw = 110 / 210  # 0.5238
        away_raw = 110 / 210
        home, away = devig_probabilities(home_raw, away_raw)
        assert home == pytest.approx(0.5, abs=0.001)
        assert away == pytest.approx(0.5, abs=0.001)
        assert home + away == pytest.approx(1.0)

    def test_asymmetric_line(self):
        """Asymmetric line preserves ratio after devigging."""
        # -150/+130: home=150/250=0.60, away=100/230≈0.4348
        home_raw = 0.60
        away_raw = 0.4348
        home, away = devig_probabilities(home_raw, away_raw)
        assert home + away == pytest.approx(1.0)
        assert home > away
        # Ratio should be preserved: home/away ≈ 0.60/0.4348
        assert home / away == pytest.approx(home_raw / away_raw, rel=1e-4)

    def test_three_way_sums_to_one(self):
        """3-way devigging (soccer h2h) returns 3 values summing to 1."""
        result = devig_probabilities(0.45, 0.30, 0.40)
        assert len(result) == 3
        assert sum(result) == pytest.approx(1.0)

    def test_three_way_preserves_ratios(self):
        """3-way devigging preserves pairwise ratios."""
        home_raw, draw_raw, away_raw = 0.45, 0.30, 0.40
        home, draw, away = devig_probabilities(home_raw, draw_raw, away_raw)
        assert home / draw == pytest.approx(home_raw / draw_raw, rel=1e-4)
        assert home / away == pytest.approx(home_raw / away_raw, rel=1e-4)

    def test_three_way_zero_total(self):
        """3-way with zero total returns uniform fallback."""
        result = devig_probabilities(0.0, 0.0, 0.0)
        assert len(result) == 3
        assert all(p == pytest.approx(1 / 3) for p in result)

    def test_empty_input(self):
        """No arguments returns empty tuple."""
        assert devig_probabilities() == ()

    def test_single_outcome(self):
        """Single outcome normalizes to 1.0."""
        result = devig_probabilities(0.7)
        assert result == (pytest.approx(1.0),)
