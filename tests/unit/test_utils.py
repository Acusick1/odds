"""Unit tests for analytics utility functions."""

import pytest
from odds_analytics.utils import (
    calculate_market_hold,
    per_book_market_holds,
)
from odds_core.odds_math import (
    american_to_decimal,
    calculate_implied_probability,
    calculate_profit_from_odds,
    decimal_to_american,
    outcome_price_decimal,
)


class TestOddsConversion:
    """Test odds conversion functions."""

    def test_american_to_decimal_positive(self):
        """Test converting positive American odds to decimal."""
        assert american_to_decimal(+150) == pytest.approx(2.5, rel=0.01)
        assert american_to_decimal(+200) == pytest.approx(3.0, rel=0.01)
        assert american_to_decimal(+100) == pytest.approx(2.0, rel=0.01)

    def test_american_to_decimal_negative(self):
        """Test converting negative American odds to decimal."""
        assert american_to_decimal(-110) == pytest.approx(1.909, rel=0.01)
        assert american_to_decimal(-200) == pytest.approx(1.5, rel=0.01)
        assert american_to_decimal(-150) == pytest.approx(1.667, rel=0.01)

    def test_decimal_to_american_over_2(self):
        assert decimal_to_american(2.5) == 150
        assert decimal_to_american(3.0) == 200

    def test_decimal_to_american_rounds_float_error(self):
        assert decimal_to_american(2.02) == 102
        assert decimal_to_american(2.01) == 101

    def test_decimal_to_american_under_2_rounds_to_nearest(self):
        assert decimal_to_american(1.91) == -110
        assert decimal_to_american(1.909) == -110
        assert decimal_to_american(1.5) == -200

    def test_decimal_to_american_rejects_non_decimal_odds(self):
        with pytest.raises(ValueError):
            decimal_to_american(1.0)

    def test_roundtrip_conversion(self):
        for american_odds in range(-1000, -100):
            assert decimal_to_american(american_to_decimal(american_odds)) == american_odds
        for american_odds in range(100, 1001):
            assert decimal_to_american(american_to_decimal(american_odds)) == american_odds


class TestOutcomePriceDecimal:
    def test_pipeline_outcome_uses_decimal_key(self):
        assert outcome_price_decimal({"name": "A", "price_decimal": 1.91}) == 1.91

    def test_odds_api_outcome_converts_american_price(self):
        assert outcome_price_decimal({"name": "A", "price": -110}) == pytest.approx(
            1.90909, abs=1e-5
        )
        assert outcome_price_decimal({"name": "A", "price": 150}) == 2.5

    def test_decimal_key_wins_over_american_price(self):
        assert outcome_price_decimal({"price": -110, "price_decimal": 1.91}) == 1.91

    def test_missing_price_raises(self):
        with pytest.raises(KeyError):
            outcome_price_decimal({"name": "A"})


class TestProbabilityCalculations:
    """Test probability calculation functions."""

    def test_implied_probability_favorites(self):
        """Test implied probability for favorites (negative odds)."""
        assert calculate_implied_probability(-110) == pytest.approx(0.524, rel=0.01)
        assert calculate_implied_probability(-200) == pytest.approx(0.667, rel=0.01)

    def test_implied_probability_underdogs(self):
        """Test implied probability for underdogs (positive odds)."""
        assert calculate_implied_probability(+150) == pytest.approx(0.4, rel=0.01)
        assert calculate_implied_probability(+200) == pytest.approx(0.333, rel=0.01)


class TestProfitCalculations:
    """Test profit calculation functions."""

    def test_profit_win_favorite(self):
        """Test profit calculation for winning favorite bet."""
        profit = calculate_profit_from_odds(stake=100, american_odds=-110, won=True)
        assert profit == pytest.approx(90.91, rel=0.01)

    def test_profit_win_underdog(self):
        """Test profit calculation for winning underdog bet."""
        profit = calculate_profit_from_odds(stake=100, american_odds=+150, won=True)
        assert profit == pytest.approx(150.0, rel=0.01)

    def test_profit_loss(self):
        """Test profit calculation for losing bet."""
        profit = calculate_profit_from_odds(stake=100, american_odds=-110, won=False)
        assert profit == -100.0

        profit = calculate_profit_from_odds(stake=100, american_odds=+150, won=False)
        assert profit == -100.0


class TestMarketHold:
    """Test market hold calculation."""

    def test_market_hold_standard_two_way(self):
        """Test hold calculation for standard two-way market."""
        # -110 on both sides is typical (4.8% hold)
        hold = calculate_market_hold([american_to_decimal(-110), american_to_decimal(-110)])
        assert hold == pytest.approx(0.048, rel=0.01)

    def test_market_hold_high_vig(self):
        """Test hold calculation for high-vig market."""
        # -120 on both sides (higher hold)
        hold = calculate_market_hold([american_to_decimal(-120), american_to_decimal(-120)])
        assert hold > 0.048

    def test_market_hold_arbitrage(self):
        """Test that arbitrage shows negative hold."""
        # +200 and -150 create arbitrage (negative hold)
        hold = calculate_market_hold([american_to_decimal(+200), american_to_decimal(-150)])
        assert hold < 0

    def test_market_hold_fair_odds(self):
        """Test that fair odds have zero hold."""
        # +100 on both sides (even money, no hold)
        hold = calculate_market_hold([american_to_decimal(+100), american_to_decimal(+100)])
        assert hold == pytest.approx(0.0, abs=0.001)

    def test_market_hold_three_way(self):
        """Test hold calculation for three-way market."""
        # Three outcomes with typical bookmaker hold (e.g., soccer match)
        # Using more realistic odds that sum to > 100%
        hold = calculate_market_hold(
            [american_to_decimal(+120), american_to_decimal(+200), american_to_decimal(+350)]
        )
        # Should have positive hold
        assert hold > 0


class TestPerBookMarketHolds:
    def test_complete_books_included(self):
        prices = [
            ("bet365", "home", american_to_decimal(-110)),
            ("bet365", "away", american_to_decimal(-110)),
            ("betfred", "home", american_to_decimal(-105)),
            ("betfred", "away", american_to_decimal(-115)),
        ]
        holds = per_book_market_holds(prices, required_outcomes={"home", "away"})
        assert set(holds) == {"bet365", "betfred"}
        assert holds["bet365"] == pytest.approx(0.048, rel=0.01)

    def test_book_missing_outcome_skipped(self):
        prices = [
            ("bet365", "home", american_to_decimal(-110)),
            ("bet365", "draw", american_to_decimal(+250)),
            ("bet365", "away", american_to_decimal(+280)),
            ("partial", "home", american_to_decimal(-110)),
            ("partial", "draw", american_to_decimal(+250)),
        ]
        holds = per_book_market_holds(prices, required_outcomes={"home", "draw", "away"})
        assert "partial" not in holds
        assert "bet365" in holds

    def test_empty_input(self):
        assert per_book_market_holds([], required_outcomes={"home", "away"}) == {}
