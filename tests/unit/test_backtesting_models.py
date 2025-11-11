"""Unit tests for backtesting data models."""

import json
import tempfile
from datetime import datetime

import pytest
from odds_analytics.backtesting import (
    BacktestResult,
    BetRecord,
    BetStatistics,
    EquityPoint,
    MonthlyStats,
    PerformanceBreakdown,
    PerformanceStats,
    RiskMetrics,
)


class TestBetRecord:
    """Test BetRecord model."""

    def test_bet_record_creation(self):
        """Test creating a bet record."""
        bet = BetRecord(
            bet_id=1,
            event_id="test123",
            event_date=datetime(2024, 10, 15, 19, 0),
            home_team="Lakers",
            away_team="Warriors",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            decision_time=datetime(2024, 10, 15, 18, 0),
            stake=100.0,
            bankroll_before=10000.0,
        )

        assert bet.bet_id == 1
        assert bet.outcome == "Lakers"
        assert bet.our_odds == -110  # Auto-set from odds

    def test_bet_record_to_dict(self):
        """Test converting bet record to dictionary."""
        bet = BetRecord(
            bet_id=1,
            event_id="test123",
            event_date=datetime(2024, 10, 15, 19, 0),
            home_team="Lakers",
            away_team="Warriors",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            decision_time=datetime(2024, 10, 15, 18, 0),
            stake=100.0,
            bankroll_before=10000.0,
            result="win",
            profit=90.91,
        )

        data = bet.to_dict()
        assert data["bet_id"] == 1
        assert data["outcome"] == "Lakers"
        assert data["result"] == "win"
        assert isinstance(data["event_date"], str)  # ISO format

    def test_bet_record_from_dict(self):
        """Test reconstructing bet record from dictionary."""
        original = BetRecord(
            bet_id=1,
            event_id="test123",
            event_date=datetime(2024, 10, 15, 19, 0),
            home_team="Lakers",
            away_team="Warriors",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            decision_time=datetime(2024, 10, 15, 18, 0),
            stake=100.0,
            bankroll_before=10000.0,
        )

        data = original.to_dict()
        reconstructed = BetRecord.from_dict(data)

        assert reconstructed.bet_id == original.bet_id
        assert reconstructed.outcome == original.outcome
        assert reconstructed.event_date == original.event_date


class TestEquityPoint:
    """Test EquityPoint model."""

    def test_equity_point_creation(self):
        """Test creating an equity point."""
        point = EquityPoint(
            date=datetime(2024, 10, 15),
            bankroll=10500.0,
            cumulative_profit=500.0,
            bets_to_date=10,
        )

        assert point.bankroll == 10500.0
        assert point.cumulative_profit == 500.0

    def test_equity_point_serialization(self):
        """Test equity point to/from dict."""
        original = EquityPoint(
            date=datetime(2024, 10, 15),
            bankroll=10500.0,
            cumulative_profit=500.0,
            bets_to_date=10,
        )

        data = original.to_dict()
        reconstructed = EquityPoint.from_dict(data)

        assert reconstructed.date == original.date
        assert reconstructed.bankroll == original.bankroll


class TestStatsModels:
    """Test statistics models."""

    def test_performance_stats(self):
        """Test PerformanceStats model (used for markets and bookmakers)."""
        stats = PerformanceStats(
            bets=50,
            profit=250.50,
            roi=5.5,
            win_rate=52.0,
            total_wagered=4550.0,
        )

        data = stats.to_dict()
        reconstructed = PerformanceStats.from_dict(data)

        assert reconstructed.bets == 50
        assert reconstructed.profit == pytest.approx(250.50, rel=0.01)

    def test_bet_statistics(self):
        """Test BetStatistics model."""
        stats = BetStatistics(
            total_bets=100,
            winning_bets=55,
            losing_bets=43,
            push_bets=2,
            win_rate=56.1,
            total_wagered=10000.0,
            average_stake=100.0,
            average_odds=-105,
            median_odds=-110,
        )

        data = stats.to_dict()
        reconstructed = BetStatistics.from_dict(data)

        assert reconstructed.total_bets == 100
        assert reconstructed.winning_bets == 55

    def test_risk_metrics(self):
        """Test RiskMetrics model."""
        metrics = RiskMetrics(
            max_drawdown=-500.0,
            max_drawdown_percentage=-5.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=0.8,
            profit_factor=1.3,
            longest_winning_streak=7,
            longest_losing_streak=4,
            average_win=120.0,
            average_loss=-95.0,
            largest_win=450.0,
            largest_loss=-200.0,
        )

        data = metrics.to_dict()
        reconstructed = RiskMetrics.from_dict(data)

        assert reconstructed.sharpe_ratio == pytest.approx(1.5)
        assert reconstructed.longest_winning_streak == 7

    def test_monthly_stats(self):
        """Test MonthlyStats model."""
        stats = MonthlyStats(
            month="2024-10",
            bets=40,
            profit=200.0,
            roi=4.5,
            win_rate=53.0,
            start_bankroll=10000.0,
            end_bankroll=10200.0,
        )

        data = stats.to_dict()
        reconstructed = MonthlyStats.from_dict(data)

        assert reconstructed.month == "2024-10"
        assert reconstructed.profit == pytest.approx(200.0)


class TestBacktestResult:
    """Test BacktestResult model."""

    def create_sample_result(self):
        """Create a sample backtest result for testing."""
        bet1 = BetRecord(
            bet_id=1,
            event_id="test1",
            event_date=datetime(2024, 10, 15, 19, 0),
            home_team="Lakers",
            away_team="Warriors",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            decision_time=datetime(2024, 10, 15, 18, 0),
            stake=100.0,
            bankroll_before=10000.0,
            result="win",
            profit=90.91,
            bankroll_after=10090.91,
        )

        equity = EquityPoint(
            date=datetime(2024, 10, 15),
            bankroll=10090.91,
            cumulative_profit=90.91,
            bets_to_date=1,
        )

        # Create composed metric groups
        bet_stats = BetStatistics(
            total_bets=1,
            winning_bets=1,
            losing_bets=0,
            push_bets=0,
            win_rate=100.0,
            total_wagered=100.0,
            average_stake=100.0,
            average_odds=-110,
            median_odds=-110,
        )

        risk_metrics = RiskMetrics(
            max_drawdown=0.0,
            max_drawdown_percentage=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            profit_factor=1.0,
            longest_winning_streak=1,
            longest_losing_streak=0,
            average_win=90.91,
            average_loss=0.0,
            largest_win=90.91,
            largest_loss=0.0,
        )

        breakdowns = PerformanceBreakdown(
            market_breakdown={"h2h": PerformanceStats(1, 90.91, 90.91, 100.0, 100.0)},
            bookmaker_breakdown={"fanduel": PerformanceStats(1, 90.91, 90.91, 100.0, 100.0)},
            monthly_performance=[
                MonthlyStats("2024-10", 1, 90.91, 90.91, 100.0, 10000.0, 10090.91)
            ],
        )

        return BacktestResult(
            strategy_name="TestStrategy",
            strategy_params={"test": "value"},
            start_date=datetime(2024, 10, 1),
            end_date=datetime(2024, 10, 31),
            total_days=30,
            initial_bankroll=10000.0,
            final_bankroll=10090.91,
            total_profit=90.91,
            roi=0.91,
            bet_stats=bet_stats,
            risk_metrics=risk_metrics,
            breakdowns=breakdowns,
            equity_curve=[equity],
            bets=[bet1],
            total_events=10,
            events_with_complete_data=10,
            data_quality_issues=[],
            run_timestamp=datetime(2024, 10, 31, 12, 0),
            execution_time_seconds=5.5,
        )

    def test_backtest_result_creation(self):
        """Test creating a backtest result."""
        result = self.create_sample_result()

        assert result.strategy_name == "TestStrategy"
        assert result.total_bets == 1
        assert result.win_rate == 100.0

    def test_backtest_result_to_dict(self):
        """Test converting result to dictionary."""
        result = self.create_sample_result()
        data = result.to_dict()

        assert "metadata" in data
        assert "summary" in data
        assert "equity_curve" in data
        assert "bets" in data

        assert data["metadata"]["strategy_name"] == "TestStrategy"
        assert data["summary"]["bet_statistics"]["total_bets"] == 1

    def test_backtest_result_to_json(self):
        """Test JSON export."""
        result = self.create_sample_result()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            result.to_json(f.name)

            # Read and verify
            with open(f.name) as read_f:
                data = json.load(read_f)
                assert data["metadata"]["strategy_name"] == "TestStrategy"

    def test_backtest_result_from_json(self):
        """Test JSON reconstruction."""
        original = self.create_sample_result()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            original.to_json(f.name)

            # Reconstruct
            reconstructed = BacktestResult.from_json(f.name)

            assert reconstructed.strategy_name == original.strategy_name
            assert reconstructed.total_bets == original.total_bets
            assert reconstructed.final_bankroll == pytest.approx(original.final_bankroll)
            assert len(reconstructed.bets) == len(original.bets)
            assert reconstructed.bets[0].bet_id == original.bets[0].bet_id

    def test_backtest_result_to_csv(self):
        """Test CSV export."""
        result = self.create_sample_result()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            result.to_csv(f.name)

            # Read and verify
            with open(f.name) as read_f:
                lines = read_f.readlines()
                assert len(lines) == 2  # Header + 1 bet
                assert "bet_id" in lines[0]
                assert "Lakers" in lines[1]

    def test_backtest_result_summary_text(self):
        """Test summary text generation."""
        result = self.create_sample_result()
        summary = result.to_summary_text()

        assert "TestStrategy" in summary
        assert "Total Profit" in summary
        assert "$90.91" in summary
