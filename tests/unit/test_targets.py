"""Unit tests for training-label target functions."""

from datetime import UTC, datetime

import pytest
from odds_analytics.targets import calculate_regression_target
from odds_core.models import Odds
from odds_core.odds_math import american_to_decimal, devig_probabilities


class TestCalculateRegressionTarget:
    """Tests for calculate_regression_target helper function."""

    def test_h2h_probability_delta(self):
        """Test h2h market uses implied probability delta."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-120),  # ~54.5% implied prob
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-150),  # ~60% implied prob
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = calculate_regression_target(opening_odds, closing_odds, "h2h")

        assert result is not None
        # Probability should increase (line moved in favor of Lakers)
        assert result > 0
        # Approximately 60% - 54.5% = 5.5%
        assert 0.04 < result < 0.07

    def test_spreads_point_delta(self):
        """Test spreads market uses point delta."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="spreads",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-110),
                point=-2.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="spreads",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-110),
                point=-3.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = calculate_regression_target(opening_odds, closing_odds, "spreads")

        assert result is not None
        # Point delta: -3.5 - (-2.5) = -1.0
        assert result == -1.0

    def test_totals_point_delta(self):
        """Test totals market uses point delta."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="totals",
                outcome_name="Over",
                price_decimal=american_to_decimal(-110),
                point=215.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="totals",
                outcome_name="Over",
                price_decimal=american_to_decimal(-110),
                point=218.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = calculate_regression_target(opening_odds, closing_odds, "totals")

        assert result is not None
        # Point delta: 218.5 - 215.5 = 3.0
        assert result == 3.0

    def test_missing_opening_odds(self):
        """Test handling of missing opening odds."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-150),
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = calculate_regression_target(None, closing_odds, "h2h")
        assert result is None

    def test_missing_closing_odds(self):
        """Test handling of missing closing odds."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-120),
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = calculate_regression_target(opening_odds, None, "h2h")
        assert result is None

    def test_unknown_market(self):
        """Test handling of unknown market type."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="unknown",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-120),
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="unknown",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-150),
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = calculate_regression_target(opening_odds, closing_odds, "unknown")
        assert result is None

    def test_multiple_bookmakers_average(self):
        """Test averaging across multiple bookmakers."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="spreads",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-110),
                point=-2.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
            Odds(
                event_id="test",
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="spreads",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-110),
                point=-3.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="spreads",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-110),
                point=-4.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
            Odds(
                event_id="test",
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="spreads",
                outcome_name="Lakers",
                price_decimal=american_to_decimal(-110),
                point=-5.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = calculate_regression_target(opening_odds, closing_odds, "spreads")

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
