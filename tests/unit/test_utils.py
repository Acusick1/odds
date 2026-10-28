"""Unit tests for analytics utility functions."""

import pytest
from odds_analytics.utils import (
    calculate_ev,
    calculate_kelly_stake,
    calculate_market_hold,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    detect_arbitrage,
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

    def test_calculate_ev_positive(self):
        """Test EV calculation for positive EV bets."""
        # 55% win probability at -110 odds is +EV
        ev = calculate_ev(bet_probability=0.55, american_odds=-110, stake=100)
        assert ev > 0

    def test_calculate_ev_negative(self):
        """Test EV calculation for negative EV bets."""
        # 45% win probability at -110 odds is -EV
        ev = calculate_ev(bet_probability=0.45, american_odds=-110, stake=100)
        assert ev < 0

    def test_calculate_ev_break_even(self):
        """Test EV calculation at break-even."""
        # At implied probability, EV should be close to 0
        implied_prob = calculate_implied_probability(-110)
        ev = calculate_ev(bet_probability=implied_prob, american_odds=-110, stake=100)
        assert abs(ev) < 1.0  # Within $1


class TestKellyCriterion:
    """Test Kelly stake calculation."""

    def test_kelly_positive_ev(self):
        """Test Kelly stake for positive EV bet."""
        stake = calculate_kelly_stake(
            bet_probability=0.55,
            american_odds=-110,
            bankroll=10000,
            kelly_fraction=1.0,  # Full Kelly
            max_stake_percentage=1.0,  # No cap
        )
        assert stake > 0
        assert stake < 10000

    def test_kelly_negative_ev(self):
        """Test Kelly returns 0 for negative EV bets."""
        stake = calculate_kelly_stake(
            bet_probability=0.40,
            american_odds=-110,
            bankroll=10000,
        )
        assert stake == 0.0

    def test_fractional_kelly(self):
        """Test fractional Kelly reduces stake."""
        full_kelly = calculate_kelly_stake(
            bet_probability=0.60,
            american_odds=-110,
            bankroll=10000,
            kelly_fraction=1.0,
            max_stake_percentage=1.0,
        )

        quarter_kelly = calculate_kelly_stake(
            bet_probability=0.60,
            american_odds=-110,
            bankroll=10000,
            kelly_fraction=0.25,
            max_stake_percentage=1.0,
        )

        assert quarter_kelly < full_kelly
        assert quarter_kelly == pytest.approx(full_kelly * 0.25, rel=0.1)

    def test_kelly_max_stake_cap(self):
        """Test max stake percentage cap."""
        stake = calculate_kelly_stake(
            bet_probability=0.70,  # Very high edge
            american_odds=-110,
            bankroll=10000,
            kelly_fraction=1.0,
            max_stake_percentage=0.05,  # Cap at 5%
        )
        assert stake <= 500  # 5% of 10000


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


class TestRiskMetrics:
    """Test risk metric calculations."""

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio for positive returns."""
        returns = [10, 20, -5, 15, 5]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0

    def test_sharpe_ratio_negative(self):
        """Test Sharpe ratio for negative returns."""
        returns = [-10, -20, 5, -15, -5]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0

    def test_sharpe_ratio_no_variance(self):
        """Test Sharpe ratio with no variance returns 0."""
        returns = [10, 10, 10, 10]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio with insufficient data."""
        returns = [10]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sortino_ratio(self):
        """Test Sortino ratio only penalizes downside."""
        # Sortino should be higher than Sharpe for same returns
        # because it only penalizes downside volatility
        returns = [20, 30, -10, 25, -5]
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        assert sortino >= sharpe

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        equity_curve = [10000, 10500, 10200, 9800, 10100, 10600]
        max_dd, max_dd_pct = calculate_max_drawdown(equity_curve)

        # Max drawdown from 10500 to 9800 = -700
        assert max_dd == pytest.approx(-700, rel=0.01)
        assert max_dd_pct == pytest.approx(-6.67, rel=0.1)

    def test_max_drawdown_no_drawdown(self):
        """Test drawdown with only increasing equity."""
        equity_curve = [10000, 10500, 11000, 11500]
        max_dd, max_dd_pct = calculate_max_drawdown(equity_curve)
        assert max_dd == 0.0
        assert max_dd_pct == 0.0

    def test_profit_factor(self):
        """Test profit factor calculation."""
        pf = calculate_profit_factor(winning_profit=1500, losing_loss=1000)
        assert pf == 1.5

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        pf = calculate_profit_factor(winning_profit=1000, losing_loss=0)
        assert pf == float("inf")

    def test_profit_factor_no_wins(self):
        """Test profit factor with no wins."""
        pf = calculate_profit_factor(winning_profit=0, losing_loss=1000)
        assert pf == 0.0


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


class TestArbitrageDetection:
    """Test arbitrage detection."""

    def test_detect_arbitrage_exists(self):
        """Test detecting a real arbitrage opportunity."""
        # If we can bet both sides and implied probabilities sum < 1.0
        # Using odds where implied probs sum to < 1.0 (rare but possible)
        # +200 (33.3%) and -150 (60%) = 93.3% < 100%, so arbitrage exists
        has_arb, profit_pct, stakes = detect_arbitrage(
            [("pinnacle", american_to_decimal(+200)), ("fanduel", american_to_decimal(-150))]
        )
        assert has_arb
        assert profit_pct > 0
        assert sum(stakes.values()) == pytest.approx(100.0, rel=0.1)

    def test_detect_arbitrage_none(self):
        """Test when no arbitrage exists."""
        # Normal market with hold
        has_arb, profit_pct, stakes = detect_arbitrage(
            [("pinnacle", american_to_decimal(-110)), ("fanduel", american_to_decimal(-110))]
        )
        assert not has_arb
        assert profit_pct == 0.0
        assert stakes == {}

    def test_arbitrage_stake_distribution(self):
        """Test that arbitrage stakes sum to 100%."""
        has_arb, profit_pct, stakes = detect_arbitrage(
            [("book1", american_to_decimal(+200)), ("book2", american_to_decimal(-180))]
        )
        if has_arb:
            assert sum(stakes.values()) == pytest.approx(100.0, rel=0.1)

    def test_arbitrage_profit_calculation(self):
        """Test arbitrage profit calculation is accurate."""
        # Example: Team A at +150 (implied 40%) and Team B at -180 (implied 64.3%)
        # Total implied prob = 104.3%, so NO arbitrage (typical market with vig)
        has_arb, profit_pct, stakes = detect_arbitrage(
            [("book1", american_to_decimal(+150)), ("book2", american_to_decimal(-180))]
        )
        assert not has_arb  # Sum > 1.0, no arbitrage

        # Example: Team A at +200 (implied 33.3%) and Team B at -150 (implied 60%)
        # Total implied prob = 93.3%, so arbitrage exists with ~6.7% profit
        has_arb, profit_pct, stakes = detect_arbitrage(
            [("book1", american_to_decimal(+200)), ("book2", american_to_decimal(-150))]
        )
        assert has_arb
        assert profit_pct == pytest.approx(6.7, rel=0.1)

    def test_arbitrage_stake_proportions(self):
        """Test that stake proportions are correctly calculated."""
        # +200 (33.3%) and -150 (60%)
        # Optimal stakes should be proportional to implied probabilities
        has_arb, profit_pct, stakes = detect_arbitrage(
            [("pinnacle", american_to_decimal(+200)), ("fanduel", american_to_decimal(-150))]
        )

        assert has_arb
        # Stake on pinnacle (+200) should be ~35.7% (33.3 / 93.3 * 100)
        # Stake on fanduel (-150) should be ~64.3% (60 / 93.3 * 100)
        assert stakes["pinnacle"] == pytest.approx(35.7, rel=0.1)
        assert stakes["fanduel"] == pytest.approx(64.3, rel=0.1)

    def test_arbitrage_multiple_outcomes(self):
        """Test arbitrage detection with more than 2 outcomes (e.g., 3-way market)."""
        # Three-way market (soccer draw, team A, team B)
        # If odds allow betting all three outcomes with profit
        has_arb, profit_pct, stakes = detect_arbitrage(
            [
                ("book1", american_to_decimal(+300)),
                ("book2", american_to_decimal(+350)),
                ("book3", american_to_decimal(+400)),
            ]
        )

        # Implied probs: 25%, 22.2%, 20% = 67.2% < 100%, so arbitrage!
        assert has_arb
        assert profit_pct > 0
        assert sum(stakes.values()) == pytest.approx(100.0, rel=0.1)
        assert len(stakes) == 3
