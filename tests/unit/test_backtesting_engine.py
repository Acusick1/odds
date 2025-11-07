"""Unit tests for BacktestEngine."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from analytics.backtesting.config import BacktestConfig, BetConstraintsConfig, BetSizingConfig
from analytics.backtesting.models import BacktestEvent, BetOpportunity, BetRecord
from analytics.backtesting.services import BacktestEngine, BettingStrategy
from core.models import EventStatus


# Test fixtures
@pytest.fixture
def sample_backtest_event():
    """Create a sample BacktestEvent for testing."""
    return BacktestEvent(
        id="test_event_1",
        commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
        home_team="Lakers",
        away_team="Warriors",
        home_score=110,
        away_score=105,
        status=EventStatus.FINAL,
    )


@pytest.fixture
def sample_backtest_event_tie():
    """Create a BacktestEvent with a tie score."""
    return BacktestEvent(
        id="test_event_tie",
        commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
        home_team="Lakers",
        away_team="Warriors",
        home_score=105,
        away_score=105,
        status=EventStatus.FINAL,
    )


@pytest.fixture
def backtest_config():
    """Create a sample BacktestConfig for testing."""
    return BacktestConfig(
        initial_bankroll=10000.0,
        start_date=datetime(2024, 10, 1, tzinfo=UTC),
        end_date=datetime(2024, 10, 31, tzinfo=UTC),
        decision_hours_before_game=1.0,
        sizing=BetSizingConfig(
            method="flat",
            flat_stake_amount=100.0,
        ),
        constraints=BetConstraintsConfig(
            min_bet_size=10.0,
            max_bet_size=500.0,
        ),
    )


@pytest.fixture
def mock_strategy():
    """Create a mock betting strategy."""

    class MockStrategy(BettingStrategy):
        def __init__(self):
            super().__init__("MockStrategy", test_param="test_value")

        async def evaluate_opportunity(
            self,
            event: BacktestEvent,
            odds_snapshot: list,
            config: BacktestConfig,
        ) -> list[BetOpportunity]:
            # Return a simple moneyline bet opportunity
            return [
                BetOpportunity(
                    event_id=event.id,
                    market="h2h",
                    outcome=event.home_team,
                    bookmaker="fanduel",
                    odds=-110,
                    line=None,
                    confidence=0.55,
                    rationale="Test bet",
                )
            ]

    return MockStrategy()


@pytest.fixture
def mock_reader():
    """Create a mock OddsReader."""
    reader = AsyncMock()
    reader.get_events_by_date_range = AsyncMock(return_value=[])
    reader.get_odds_at_time = AsyncMock(return_value=[])
    return reader


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return structlog.get_logger()


@pytest.fixture
def backtest_engine(mock_strategy, backtest_config, mock_reader, mock_logger):
    """Create a BacktestEngine instance for testing."""
    return BacktestEngine(
        strategy=mock_strategy,
        config=backtest_config,
        reader=mock_reader,
        logger_instance=mock_logger,
    )


class TestEvaluateMoneylineOutcome:
    """Test _evaluate_moneyline_outcome method."""

    def test_home_team_wins(self, backtest_engine, sample_backtest_event):
        """Test moneyline evaluation when home team wins."""
        # Lakers won 110-105
        result = backtest_engine._evaluate_moneyline_outcome("Lakers", sample_backtest_event)
        assert result is True

    def test_away_team_wins(self, backtest_engine, sample_backtest_event):
        """Test moneyline evaluation when away team wins."""
        # Warriors lost 105-110
        result = backtest_engine._evaluate_moneyline_outcome("Warriors", sample_backtest_event)
        assert result is False

    def test_tie_returns_none(self, backtest_engine, sample_backtest_event_tie):
        """Test that a tie returns None (push)."""
        result = backtest_engine._evaluate_moneyline_outcome("Lakers", sample_backtest_event_tie)
        assert result is None


class TestEvaluateSpreadOutcome:
    """Test _evaluate_spread_outcome method."""

    def test_home_team_covers_spread(self, backtest_engine, sample_backtest_event):
        """Test home team covering the spread."""
        # Lakers won by 5, with -3.5 spread they cover
        result = backtest_engine._evaluate_spread_outcome("Lakers", -3.5, sample_backtest_event)
        assert result is True

    def test_home_team_fails_spread(self, backtest_engine, sample_backtest_event):
        """Test home team failing to cover the spread."""
        # Lakers won by 5, with -7.5 spread they don't cover
        result = backtest_engine._evaluate_spread_outcome("Lakers", -7.5, sample_backtest_event)
        assert result is False

    def test_away_team_covers_spread(self, backtest_engine, sample_backtest_event):
        """Test away team covering the spread."""
        # Warriors lost by 5, with +7.5 spread they cover
        # Line is from home perspective, so Lakers -7.5 means Warriors +7.5
        result = backtest_engine._evaluate_spread_outcome("Warriors", -7.5, sample_backtest_event)
        assert result is True

    def test_away_team_fails_spread(self, backtest_engine, sample_backtest_event):
        """Test away team failing to cover the spread."""
        # Warriors lost by 5, with +3.5 spread they don't cover
        # Line is from home perspective, so Lakers -3.5 means Warriors +3.5
        result = backtest_engine._evaluate_spread_outcome("Warriors", -3.5, sample_backtest_event)
        assert result is False

    def test_exact_spread_push(self, backtest_engine, sample_backtest_event):
        """Test exact spread results in push."""
        # Lakers won by 5, with -5.0 spread it's a push
        result = backtest_engine._evaluate_spread_outcome("Lakers", -5.0, sample_backtest_event)
        assert result is None

    def test_none_line_returns_none(self, backtest_engine, sample_backtest_event):
        """Test that None line returns None."""
        result = backtest_engine._evaluate_spread_outcome("Lakers", None, sample_backtest_event)
        assert result is None


class TestEvaluateTotalOutcome:
    """Test _evaluate_total_outcome method."""

    def test_over_wins(self, backtest_engine, sample_backtest_event):
        """Test over bet wins."""
        # Total score is 215 (110 + 105)
        result = backtest_engine._evaluate_total_outcome("Over", 210.5, sample_backtest_event)
        assert result is True

    def test_over_loses(self, backtest_engine, sample_backtest_event):
        """Test over bet loses."""
        # Total score is 215
        result = backtest_engine._evaluate_total_outcome("Over", 220.5, sample_backtest_event)
        assert result is False

    def test_under_wins(self, backtest_engine, sample_backtest_event):
        """Test under bet wins."""
        # Total score is 215
        result = backtest_engine._evaluate_total_outcome("Under", 220.5, sample_backtest_event)
        assert result is True

    def test_under_loses(self, backtest_engine, sample_backtest_event):
        """Test under bet loses."""
        # Total score is 215
        result = backtest_engine._evaluate_total_outcome("Under", 210.5, sample_backtest_event)
        assert result is False

    def test_exact_total_push(self, backtest_engine, sample_backtest_event):
        """Test exact total results in push."""
        # Total score is 215
        result = backtest_engine._evaluate_total_outcome("Over", 215.0, sample_backtest_event)
        assert result is None

        result = backtest_engine._evaluate_total_outcome("Under", 215.0, sample_backtest_event)
        assert result is None

    def test_none_line_returns_none(self, backtest_engine, sample_backtest_event):
        """Test that None line returns None."""
        result = backtest_engine._evaluate_total_outcome("Over", None, sample_backtest_event)
        assert result is None

    def test_case_insensitive_over_under(self, backtest_engine, sample_backtest_event):
        """Test that over/under matching is case insensitive."""
        result_over = backtest_engine._evaluate_total_outcome("over", 210.5, sample_backtest_event)
        result_under = backtest_engine._evaluate_total_outcome("UNDER", 220.5, sample_backtest_event)

        assert result_over is True
        assert result_under is True


class TestEvaluateBetResult:
    """Test _evaluate_bet_result_for_opportunity method."""

    def test_moneyline_win(self, backtest_engine, sample_backtest_event):
        """Test moneyline bet that wins."""
        opportunity = BetOpportunity(
            event_id="test_event_1",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )

        result, profit = backtest_engine._evaluate_bet_result_for_opportunity(
            opportunity, sample_backtest_event, 100.0
        )

        assert result == "win"
        assert profit == pytest.approx(90.91, rel=0.01)

    def test_moneyline_loss(self, backtest_engine, sample_backtest_event):
        """Test moneyline bet that loses."""
        opportunity = BetOpportunity(
            event_id="test_event_1",
            market="h2h",
            outcome="Warriors",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.45,
            rationale="Test",
        )

        result, profit = backtest_engine._evaluate_bet_result_for_opportunity(
            opportunity, sample_backtest_event, 100.0
        )

        assert result == "loss"
        assert profit == pytest.approx(-100.0)

    def test_moneyline_push(self, backtest_engine, sample_backtest_event_tie):
        """Test moneyline bet that pushes (tie)."""
        opportunity = BetOpportunity(
            event_id="test_event_tie",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )

        result, profit = backtest_engine._evaluate_bet_result_for_opportunity(
            opportunity, sample_backtest_event_tie, 100.0
        )

        assert result == "push"
        assert profit == 0.0

    def test_spread_win(self, backtest_engine, sample_backtest_event):
        """Test spread bet that wins."""
        opportunity = BetOpportunity(
            event_id="test_event_1",
            market="spreads",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=-3.5,
            confidence=0.55,
            rationale="Test",
        )

        result, profit = backtest_engine._evaluate_bet_result_for_opportunity(
            opportunity, sample_backtest_event, 100.0
        )

        assert result == "win"
        assert profit == pytest.approx(90.91, rel=0.01)

    def test_spread_push(self, backtest_engine, sample_backtest_event):
        """Test spread bet that pushes."""
        opportunity = BetOpportunity(
            event_id="test_event_1",
            market="spreads",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=-5.0,  # Exact spread
            confidence=0.55,
            rationale="Test",
        )

        result, profit = backtest_engine._evaluate_bet_result_for_opportunity(
            opportunity, sample_backtest_event, 100.0
        )

        assert result == "push"
        assert profit == 0.0

    def test_total_over_win(self, backtest_engine, sample_backtest_event):
        """Test over bet that wins."""
        opportunity = BetOpportunity(
            event_id="test_event_1",
            market="totals",
            outcome="Over",
            bookmaker="fanduel",
            odds=-110,
            line=210.5,
            confidence=0.55,
            rationale="Test",
        )

        result, profit = backtest_engine._evaluate_bet_result_for_opportunity(
            opportunity, sample_backtest_event, 100.0
        )

        assert result == "win"
        assert profit == pytest.approx(90.91, rel=0.01)

    def test_total_push(self, backtest_engine, sample_backtest_event):
        """Test total bet that pushes."""
        opportunity = BetOpportunity(
            event_id="test_event_1",
            market="totals",
            outcome="Over",
            bookmaker="fanduel",
            odds=-110,
            line=215.0,  # Exact total
            confidence=0.55,
            rationale="Test",
        )

        result, profit = backtest_engine._evaluate_bet_result_for_opportunity(
            opportunity, sample_backtest_event, 100.0
        )

        assert result == "push"
        assert profit == 0.0

    def test_unknown_market(self, backtest_engine, sample_backtest_event):
        """Test handling of unknown market type."""
        opportunity = BetOpportunity(
            event_id="test_event_1",
            market="unknown_market",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )

        result, profit = backtest_engine._evaluate_bet_result_for_opportunity(
            opportunity, sample_backtest_event, 100.0
        )

        assert result == "unknown"
        assert profit == 0.0


class TestCalculateStake:
    """Test _calculate_stake method."""

    def test_flat_stake(self, backtest_config, mock_strategy, mock_reader, mock_logger):
        """Test flat stake betting."""
        config = backtest_config
        config.sizing.method = "flat"
        config.sizing.flat_stake_amount = 150.0

        engine = BacktestEngine(mock_strategy, config, mock_reader, logger_instance=mock_logger)

        opportunity = BetOpportunity(
            event_id="test",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )

        stake = engine._calculate_stake(opportunity, bankroll=10000.0)
        assert stake == 150.0

    def test_percentage_stake(self, backtest_config, mock_strategy, mock_reader, mock_logger):
        """Test percentage stake betting."""
        config = backtest_config
        config.sizing.method = "percentage"
        config.sizing.percentage_stake = 0.02  # 2%

        engine = BacktestEngine(mock_strategy, config, mock_reader, logger_instance=mock_logger)

        opportunity = BetOpportunity(
            event_id="test",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )

        stake = engine._calculate_stake(opportunity, bankroll=10000.0)
        assert stake == pytest.approx(200.0)  # 2% of 10000

    def test_kelly_stake(self, backtest_config, mock_strategy, mock_reader, mock_logger):
        """Test fractional Kelly stake betting."""
        config = backtest_config
        config.sizing.method = "fractional_kelly"
        config.sizing.kelly_fraction = 0.25  # Quarter Kelly

        engine = BacktestEngine(mock_strategy, config, mock_reader, logger_instance=mock_logger)

        opportunity = BetOpportunity(
            event_id="test",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )

        stake = engine._calculate_stake(opportunity, bankroll=10000.0)
        # Kelly should return some positive stake for positive EV bet
        assert stake > 0
        assert stake < 500.0  # Should be capped by max_bet_size

    def test_kelly_negative_ev_returns_zero(
        self, backtest_config, mock_strategy, mock_reader, mock_logger
    ):
        """Test that Kelly returns 0 for negative EV bet."""
        config = backtest_config
        config.sizing.method = "fractional_kelly"

        engine = BacktestEngine(mock_strategy, config, mock_reader, logger_instance=mock_logger)

        # Very low confidence on negative odds = negative EV
        opportunity = BetOpportunity(
            event_id="test",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-500,
            line=None,
            confidence=0.30,  # Way below implied probability
            rationale="Test",
        )

        stake = engine._calculate_stake(opportunity, bankroll=10000.0)
        assert stake == 10.0  # Returns min_bet_size when Kelly is 0

    def test_max_bet_size_enforcement(
        self, backtest_config, mock_strategy, mock_reader, mock_logger
    ):
        """Test that max_bet_size is enforced."""
        config = backtest_config
        config.sizing.method = "flat"
        config.sizing.flat_stake_amount = 1000.0
        config.constraints.max_bet_size = 500.0

        engine = BacktestEngine(mock_strategy, config, mock_reader, logger_instance=mock_logger)

        opportunity = BetOpportunity(
            event_id="test",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )

        stake = engine._calculate_stake(opportunity, bankroll=10000.0)
        assert stake == 500.0  # Capped at max

    def test_min_bet_size_enforcement(
        self, backtest_config, mock_strategy, mock_reader, mock_logger
    ):
        """Test that min_bet_size is enforced."""
        config = backtest_config
        config.sizing.method = "flat"
        config.sizing.flat_stake_amount = 5.0
        config.constraints.min_bet_size = 10.0

        engine = BacktestEngine(mock_strategy, config, mock_reader, logger_instance=mock_logger)

        opportunity = BetOpportunity(
            event_id="test",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )

        stake = engine._calculate_stake(opportunity, bankroll=10000.0)
        assert stake == 10.0  # Raised to min

    def test_unknown_bet_sizing_method_raises_error(
        self, backtest_config, mock_strategy, mock_reader, mock_logger
    ):
        """Test that unknown bet sizing method raises ValueError."""
        config = backtest_config
        config.sizing.method = "unknown_method"

        engine = BacktestEngine(mock_strategy, config, mock_reader, logger_instance=mock_logger)

        opportunity = BetOpportunity(
            event_id="test",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )

        with pytest.raises(ValueError, match="Unknown bet sizing method"):
            engine._calculate_stake(opportunity, bankroll=10000.0)


class TestBuildEquityCurve:
    """Test _build_equity_curve method."""

    def test_empty_bets(self, backtest_engine):
        """Test equity curve with no bets."""
        equity_curve = backtest_engine._build_equity_curve([])
        assert len(equity_curve) == 0

    def test_single_bet(self, backtest_engine):
        """Test equity curve with single bet."""
        bet = BetRecord(
            bet_id=1,
            event_id="test1",
            event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Warriors",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
            stake=100.0,
            bankroll_before=10000.0,
            result="win",
            profit=90.91,
            bankroll_after=10090.91,
        )

        equity_curve = backtest_engine._build_equity_curve([bet])

        assert len(equity_curve) == 1
        assert equity_curve[0].bankroll == pytest.approx(10090.91)
        assert equity_curve[0].cumulative_profit == pytest.approx(90.91)
        assert equity_curve[0].bets_to_date == 1

    def test_multiple_bets_same_day(self, backtest_engine):
        """Test equity curve with multiple bets on same day."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 10, 15, 21, 0, tzinfo=UTC),
                home_team="Celtics",
                away_team="Heat",
                market="h2h",
                outcome="Celtics",
                bookmaker="fanduel",
                odds=+150,
                line=None,
                decision_time=datetime(2024, 10, 15, 20, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10090.91,
                result="loss",
                profit=-100.0,
            ),
        ]

        equity_curve = backtest_engine._build_equity_curve(bets)

        # Should have one point for the day
        assert len(equity_curve) == 1
        assert equity_curve[0].cumulative_profit == pytest.approx(-9.09)  # 90.91 - 100
        assert equity_curve[0].bets_to_date == 2

    def test_multiple_bets_multiple_days(self, backtest_engine):
        """Test equity curve with bets across multiple days."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 10, 16, 19, 0, tzinfo=UTC),
                home_team="Celtics",
                away_team="Heat",
                market="h2h",
                outcome="Celtics",
                bookmaker="fanduel",
                odds=+150,
                line=None,
                decision_time=datetime(2024, 10, 16, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10090.91,
                result="win",
                profit=150.0,
            ),
        ]

        equity_curve = backtest_engine._build_equity_curve(bets)

        assert len(equity_curve) == 2
        assert equity_curve[0].cumulative_profit == pytest.approx(90.91)
        assert equity_curve[0].bets_to_date == 1
        assert equity_curve[1].cumulative_profit == pytest.approx(240.91)  # 90.91 + 150
        assert equity_curve[1].bets_to_date == 2


class TestCalculateStreaks:
    """Test _calculate_streaks method."""

    def test_empty_bets(self, backtest_engine):
        """Test streaks with no bets."""
        win_streak, loss_streak = backtest_engine._calculate_streaks([])
        assert win_streak == 0
        assert loss_streak == 0

    def test_all_wins(self, backtest_engine):
        """Test streaks with all wins."""
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            )
            for i in range(1, 6)
        ]

        win_streak, loss_streak = backtest_engine._calculate_streaks(bets)
        assert win_streak == 5
        assert loss_streak == 0

    def test_all_losses(self, backtest_engine):
        """Test streaks with all losses."""
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="loss",
                profit=-100.0,
            )
            for i in range(1, 6)
        ]

        win_streak, loss_streak = backtest_engine._calculate_streaks(bets)
        assert win_streak == 0
        assert loss_streak == 5

    def test_mixed_results(self, backtest_engine):
        """Test streaks with mixed results."""
        results = ["win", "win", "win", "loss", "win", "loss", "loss", "loss", "loss", "win"]
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result=result,
                profit=90.91 if result == "win" else -100.0,
            )
            for i, result in enumerate(results, start=1)
        ]

        win_streak, loss_streak = backtest_engine._calculate_streaks(bets)
        assert win_streak == 3  # First 3 wins
        assert loss_streak == 4  # Loss streak in middle

    def test_pushes_dont_break_streaks(self, backtest_engine):
        """Test that pushes don't affect win/loss streaks."""
        results = ["win", "win", "push", "loss", "loss"]
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result=result,
                profit=90.91 if result == "win" else (-100.0 if result == "loss" else 0.0),
            )
            for i, result in enumerate(results, start=1)
        ]

        win_streak, loss_streak = backtest_engine._calculate_streaks(bets)
        assert win_streak == 2
        assert loss_streak == 2


class TestCalculateDailyReturns:
    """Test _calculate_daily_returns method."""

    def test_empty_bets(self, backtest_engine):
        """Test daily returns with no bets."""
        returns = backtest_engine._calculate_daily_returns([])
        assert len(returns) == 0

    def test_single_day_single_bet(self, backtest_engine):
        """Test daily returns with one bet."""
        bet = BetRecord(
            bet_id=1,
            event_id="test1",
            event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Warriors",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
            stake=100.0,
            bankroll_before=10000.0,
            result="win",
            profit=90.91,
        )

        returns = backtest_engine._calculate_daily_returns([bet])
        assert len(returns) == 1
        assert returns[0] == pytest.approx(90.91)

    def test_multiple_bets_same_day(self, backtest_engine):
        """Test daily returns aggregated by day."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 10, 15, 21, 0, tzinfo=UTC),
                home_team="Celtics",
                away_team="Heat",
                market="h2h",
                outcome="Celtics",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 20, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10090.91,
                result="loss",
                profit=-100.0,
            ),
        ]

        returns = backtest_engine._calculate_daily_returns(bets)
        assert len(returns) == 1
        assert returns[0] == pytest.approx(-9.09)  # 90.91 - 100

    def test_multiple_days(self, backtest_engine):
        """Test daily returns across multiple days."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 10, 16, 19, 0, tzinfo=UTC),
                home_team="Celtics",
                away_team="Heat",
                market="h2h",
                outcome="Celtics",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 16, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10090.91,
                result="loss",
                profit=-100.0,
            ),
        ]

        returns = backtest_engine._calculate_daily_returns(bets)
        assert len(returns) == 2
        assert pytest.approx(90.91) in returns
        assert pytest.approx(-100.0) in returns


class TestCalculateMarketBreakdown:
    """Test _calculate_market_breakdown method."""

    def test_single_market(self, backtest_engine):
        """Test market breakdown with single market."""
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win" if i % 2 == 0 else "loss",
                profit=90.91 if i % 2 == 0 else -100.0,
            )
            for i in range(1, 6)
        ]

        breakdown = backtest_engine._calculate_market_breakdown(bets)

        assert "h2h" in breakdown
        stats = breakdown["h2h"]
        assert stats.bets == 5
        assert stats.profit == pytest.approx(2 * 90.91 - 3 * 100.0)

    def test_multiple_markets(self, backtest_engine):
        """Test market breakdown with multiple markets."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="spreads",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=-5.5,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10090.91,
                result="loss",
                profit=-100.0,
            ),
        ]

        breakdown = backtest_engine._calculate_market_breakdown(bets)

        assert "h2h" in breakdown
        assert "spreads" in breakdown
        assert breakdown["h2h"].bets == 1
        assert breakdown["spreads"].bets == 1


class TestCalculateBookmakerBreakdown:
    """Test _calculate_bookmaker_breakdown method."""

    def test_single_bookmaker(self, backtest_engine):
        """Test bookmaker breakdown with single bookmaker."""
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            )
            for i in range(1, 4)
        ]

        breakdown = backtest_engine._calculate_bookmaker_breakdown(bets)

        assert "fanduel" in breakdown
        stats = breakdown["fanduel"]
        assert stats.bets == 3
        assert stats.profit == pytest.approx(3 * 90.91)

    def test_multiple_bookmakers(self, backtest_engine):
        """Test bookmaker breakdown with multiple bookmakers."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="draftkings",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10090.91,
                result="loss",
                profit=-100.0,
            ),
        ]

        breakdown = backtest_engine._calculate_bookmaker_breakdown(bets)

        assert "fanduel" in breakdown
        assert "draftkings" in breakdown
        assert breakdown["fanduel"].bets == 1
        assert breakdown["draftkings"].bets == 1


class TestCalculateMonthlyPerformance:
    """Test _calculate_monthly_performance method."""

    def test_empty_bets(self, backtest_engine):
        """Test monthly performance with no bets."""
        monthly = backtest_engine._calculate_monthly_performance([])
        assert len(monthly) == 0

    def test_single_month(self, backtest_engine):
        """Test monthly performance with bets in single month."""
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            )
            for i in range(1, 4)
        ]

        monthly = backtest_engine._calculate_monthly_performance(bets)

        assert len(monthly) == 1
        assert monthly[0].month == "2024-10"
        assert monthly[0].bets == 3
        assert monthly[0].profit == pytest.approx(3 * 90.91)

    def test_multiple_months(self, backtest_engine):
        """Test monthly performance across multiple months."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 11, 15, 19, 0, tzinfo=UTC),
                home_team="Celtics",
                away_team="Heat",
                market="h2h",
                outcome="Celtics",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 11, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10090.91,
                result="win",
                profit=90.91,
            ),
        ]

        monthly = backtest_engine._calculate_monthly_performance(bets)

        assert len(monthly) == 2
        assert monthly[0].month == "2024-10"
        assert monthly[1].month == "2024-11"

    def test_cumulative_bankroll_tracking(self, backtest_engine):
        """Test that monthly performance tracks cumulative bankroll correctly."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=100.0,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 11, 15, 19, 0, tzinfo=UTC),
                home_team="Celtics",
                away_team="Heat",
                market="h2h",
                outcome="Celtics",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 11, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10100.0,
                result="win",
                profit=50.0,
            ),
        ]

        monthly = backtest_engine._calculate_monthly_performance(bets)

        assert monthly[0].start_bankroll == pytest.approx(10000.0)
        assert monthly[0].end_bankroll == pytest.approx(10100.0)
        assert monthly[1].start_bankroll == pytest.approx(10100.0)
        assert monthly[1].end_bankroll == pytest.approx(10150.0)


class TestCalculateMetrics:
    """Test _calculate_metrics method."""

    def test_empty_bets_returns_empty_result(self, backtest_engine):
        """Test that empty bets returns empty result."""
        result = backtest_engine._calculate_metrics([], [], execution_time=1.0)

        assert result.total_bets == 0
        assert result.final_bankroll == 10000.0  # Initial bankroll
        assert result.total_profit == 0.0
        assert result.roi == 0.0

    def test_single_winning_bet(self, backtest_engine, sample_backtest_event):
        """Test metrics with single winning bet."""
        bet = BetRecord(
            bet_id=1,
            event_id="test1",
            event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Warriors",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
            stake=100.0,
            bankroll_before=10000.0,
            result="win",
            profit=90.91,
            bankroll_after=10090.91,
        )

        result = backtest_engine._calculate_metrics(
            [sample_backtest_event], [bet], execution_time=1.0
        )

        assert result.total_bets == 1
        assert result.winning_bets == 1
        assert result.losing_bets == 0
        assert result.win_rate == 100.0
        assert result.total_profit == pytest.approx(90.91)
        assert result.roi == pytest.approx(0.9091)  # 90.91 / 10000 * 100

    def test_all_wins(self, backtest_engine, sample_backtest_event):
        """Test metrics with all winning bets."""
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0 + (i - 1) * 90.91,
                result="win",
                profit=90.91,
                bankroll_after=10000.0 + i * 90.91,
            )
            for i in range(1, 6)
        ]

        result = backtest_engine._calculate_metrics(
            [sample_backtest_event] * 5, bets, execution_time=1.0
        )

        assert result.total_bets == 5
        assert result.winning_bets == 5
        assert result.losing_bets == 0
        assert result.win_rate == 100.0
        assert result.total_profit == pytest.approx(5 * 90.91)

    def test_all_losses(self, backtest_engine, sample_backtest_event):
        """Test metrics with all losing bets."""
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Warriors",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0 - (i - 1) * 100.0,
                result="loss",
                profit=-100.0,
                bankroll_after=10000.0 - i * 100.0,
            )
            for i in range(1, 6)
        ]

        result = backtest_engine._calculate_metrics(
            [sample_backtest_event] * 5, bets, execution_time=1.0
        )

        assert result.total_bets == 5
        assert result.winning_bets == 0
        assert result.losing_bets == 5
        assert result.win_rate == 0.0
        assert result.total_profit == pytest.approx(-500.0)
        assert result.roi < 0

    def test_all_pushes(self, backtest_engine, sample_backtest_event_tie):
        """Test metrics with all pushes."""
        bets = [
            BetRecord(
                bet_id=i,
                event_id=f"test{i}",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="push",
                profit=0.0,
                bankroll_after=10000.0,
            )
            for i in range(1, 6)
        ]

        result = backtest_engine._calculate_metrics(
            [sample_backtest_event_tie] * 5, bets, execution_time=1.0
        )

        assert result.total_bets == 5
        assert result.winning_bets == 0
        assert result.losing_bets == 0
        assert result.push_bets == 5
        assert result.win_rate == 0.0  # Pushes don't count toward win rate
        assert result.total_profit == 0.0
        assert result.roi == 0.0

    def test_mixed_results(self, backtest_engine, sample_backtest_event):
        """Test metrics with mixed win/loss/push results."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
                bankroll_after=10090.91,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 10, 16, 19, 0, tzinfo=UTC),
                home_team="Celtics",
                away_team="Heat",
                market="h2h",
                outcome="Heat",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 16, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10090.91,
                result="loss",
                profit=-100.0,
                bankroll_after=9990.91,
            ),
            BetRecord(
                bet_id=3,
                event_id="test3",
                event_date=datetime(2024, 10, 17, 19, 0, tzinfo=UTC),
                home_team="Nets",
                away_team="Knicks",
                market="spreads",
                outcome="Nets",
                bookmaker="fanduel",
                odds=-110,
                line=-5.0,
                decision_time=datetime(2024, 10, 17, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=9990.91,
                result="push",
                profit=0.0,
                bankroll_after=9990.91,
            ),
        ]

        result = backtest_engine._calculate_metrics(
            [sample_backtest_event] * 3, bets, execution_time=1.0
        )

        assert result.total_bets == 3
        assert result.winning_bets == 1
        assert result.losing_bets == 1
        assert result.push_bets == 1
        assert result.win_rate == pytest.approx(50.0)  # 1 win / (1 win + 1 loss)
        assert result.total_profit == pytest.approx(-9.09)  # 90.91 - 100

    def test_metrics_include_breakdowns(self, backtest_engine, sample_backtest_event):
        """Test that metrics include market/bookmaker/monthly breakdowns."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
                bankroll_after=10090.91,
            ),
            BetRecord(
                bet_id=2,
                event_id="test2",
                event_date=datetime(2024, 10, 16, 19, 0, tzinfo=UTC),
                home_team="Celtics",
                away_team="Heat",
                market="spreads",
                outcome="Celtics",
                bookmaker="draftkings",
                odds=-110,
                line=-5.5,
                decision_time=datetime(2024, 10, 16, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10090.91,
                result="win",
                profit=90.91,
                bankroll_after=10181.82,
            ),
        ]

        result = backtest_engine._calculate_metrics(
            [sample_backtest_event] * 2, bets, execution_time=1.0
        )

        # Check market breakdown
        assert "h2h" in result.market_breakdown
        assert "spreads" in result.market_breakdown
        assert result.market_breakdown["h2h"].bets == 1
        assert result.market_breakdown["spreads"].bets == 1

        # Check bookmaker breakdown
        assert "fanduel" in result.bookmaker_breakdown
        assert "draftkings" in result.bookmaker_breakdown
        assert result.bookmaker_breakdown["fanduel"].bets == 1
        assert result.bookmaker_breakdown["draftkings"].bets == 1

        # Check monthly performance
        assert len(result.monthly_performance) == 1
        assert result.monthly_performance[0].month == "2024-10"
        assert result.monthly_performance[0].bets == 2

    def test_equity_curve_generated(self, backtest_engine, sample_backtest_event):
        """Test that equity curve is generated."""
        bets = [
            BetRecord(
                bet_id=1,
                event_id="test1",
                event_date=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Warriors",
                market="h2h",
                outcome="Lakers",
                bookmaker="fanduel",
                odds=-110,
                line=None,
                decision_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                stake=100.0,
                bankroll_before=10000.0,
                result="win",
                profit=90.91,
                bankroll_after=10090.91,
            )
        ]

        result = backtest_engine._calculate_metrics(
            [sample_backtest_event], bets, execution_time=1.0
        )

        assert len(result.equity_curve) == 1
        assert result.equity_curve[0].bankroll == pytest.approx(10090.91)


class TestCreateEmptyResult:
    """Test _create_empty_result method."""

    def test_empty_result_has_zero_values(self, backtest_engine):
        """Test that empty result has zero values."""
        result = backtest_engine._create_empty_result(execution_time=1.0)

        assert result.total_bets == 0
        assert result.winning_bets == 0
        assert result.losing_bets == 0
        assert result.push_bets == 0
        assert result.total_profit == 0.0
        assert result.roi == 0.0
        assert result.final_bankroll == 10000.0  # Unchanged
        assert len(result.bets) == 0
        assert len(result.equity_curve) == 0
        assert "No bets placed" in result.data_quality_issues
