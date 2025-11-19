"""Unit tests for HTML report generation."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from odds_analytics.backtesting.models import (
    BacktestResult,
    BetRecord,
    BetStatistics,
    EquityPoint,
    MonthlyStats,
    PerformanceBreakdown,
    PerformanceStats,
    RiskMetrics,
)
from odds_analytics.reporting import HTMLReportGenerator
from odds_analytics.reporting.charts import (
    create_bookmaker_breakdown_chart,
    create_drawdown_chart,
    create_equity_curve_chart,
    create_market_breakdown_chart,
    create_monthly_performance_chart,
    create_profit_distribution_chart,
)
from odds_analytics.reporting.tables import (
    create_bet_summary_table,
    create_bookmaker_breakdown_table,
    create_market_breakdown_table,
    create_monthly_performance_table,
    create_risk_metrics_table,
)
from odds_core.models import EventStatus


@pytest.fixture
def sample_backtest_result() -> BacktestResult:
    """Create a sample BacktestResult for testing."""
    # Create sample bet records
    bets = (
        BetRecord(
            bet_id=1,
            event_id="test_event_1",
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
            strategy_confidence=0.65,
            result="win",
            profit=90.91,
            bankroll_after=10090.91,
            home_score=105,
            away_score=98,
        ),
        BetRecord(
            bet_id=2,
            event_id="test_event_2",
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
            strategy_confidence=0.70,
            result="loss",
            profit=-100.0,
            bankroll_after=9990.91,
            home_score=100,
            away_score=102,
        ),
        BetRecord(
            bet_id=3,
            event_id="test_event_3",
            event_date=datetime(2024, 10, 17, 19, 0, tzinfo=UTC),
            home_team="Nuggets",
            away_team="Suns",
            market="totals",
            outcome="Over",
            bookmaker="pinnacle",
            odds=105,
            line=220.5,
            decision_time=datetime(2024, 10, 17, 18, 0, tzinfo=UTC),
            stake=100.0,
            bankroll_before=9990.91,
            strategy_confidence=0.75,
            result="win",
            profit=105.0,
            bankroll_after=10095.91,
            home_score=115,
            away_score=110,
        ),
    )

    # Create equity curve
    equity_curve = (
        EquityPoint(
            date=datetime(2024, 10, 15, tzinfo=UTC),
            bankroll=10090.91,
            cumulative_profit=90.91,
            bets_to_date=1,
        ),
        EquityPoint(
            date=datetime(2024, 10, 16, tzinfo=UTC),
            bankroll=9990.91,
            cumulative_profit=-9.09,
            bets_to_date=2,
        ),
        EquityPoint(
            date=datetime(2024, 10, 17, tzinfo=UTC),
            bankroll=10095.91,
            cumulative_profit=95.91,
            bets_to_date=3,
        ),
    )

    # Create statistics
    bet_stats = BetStatistics(
        total_bets=3,
        winning_bets=2,
        losing_bets=1,
        push_bets=0,
        win_rate=66.67,
        total_wagered=300.0,
        average_stake=100.0,
        average_odds=-5.0,
        median_odds=-110.0,
    )

    risk_metrics = RiskMetrics(
        max_drawdown=-109.09,
        max_drawdown_percentage=-1.09,
        sharpe_ratio=1.25,
        sortino_ratio=1.45,
        calmar_ratio=0.88,
        profit_factor=1.96,
        longest_winning_streak=2,
        longest_losing_streak=1,
        average_win=97.96,
        average_loss=-100.0,
        largest_win=105.0,
        largest_loss=-100.0,
    )

    # Create market breakdown
    market_breakdown = {
        "h2h": PerformanceStats(
            bets=1, profit=90.91, roi=90.91, win_rate=100.0, total_wagered=100.0
        ),
        "spreads": PerformanceStats(
            bets=1, profit=-100.0, roi=-100.0, win_rate=0.0, total_wagered=100.0
        ),
        "totals": PerformanceStats(
            bets=1, profit=105.0, roi=105.0, win_rate=100.0, total_wagered=100.0
        ),
    }

    # Create bookmaker breakdown
    bookmaker_breakdown = {
        "fanduel": PerformanceStats(
            bets=1, profit=90.91, roi=90.91, win_rate=100.0, total_wagered=100.0
        ),
        "draftkings": PerformanceStats(
            bets=1, profit=-100.0, roi=-100.0, win_rate=0.0, total_wagered=100.0
        ),
        "pinnacle": PerformanceStats(
            bets=1, profit=105.0, roi=105.0, win_rate=100.0, total_wagered=100.0
        ),
    }

    # Create monthly performance
    monthly_performance = (
        MonthlyStats(
            month="2024-10",
            bets=3,
            profit=95.91,
            roi=31.97,
            win_rate=66.67,
            start_bankroll=10000.0,
            end_bankroll=10095.91,
        ),
    )

    breakdowns = PerformanceBreakdown(
        market_breakdown=market_breakdown,
        bookmaker_breakdown=bookmaker_breakdown,
        monthly_performance=monthly_performance,
    )

    # Create BacktestResult
    return BacktestResult(
        strategy_name="BasicEVStrategy",
        strategy_params={"sharp_book": "pinnacle", "min_ev_threshold": 0.03},
        start_date=datetime(2024, 10, 1, tzinfo=UTC),
        end_date=datetime(2024, 10, 31, tzinfo=UTC),
        total_days=30,
        initial_bankroll=10000.0,
        final_bankroll=10095.91,
        total_profit=95.91,
        roi=0.96,
        bet_stats=bet_stats,
        risk_metrics=risk_metrics,
        breakdowns=breakdowns,
        equity_curve=equity_curve,
        bets=bets,
        total_events=10,
        events_with_complete_data=10,
        data_quality_issues=(),
        run_timestamp=datetime(2024, 11, 1, tzinfo=UTC),
        execution_time_seconds=45.2,
    )


class TestChartGeneration:
    """Test chart generation functions."""

    def test_equity_curve_chart(self, sample_backtest_result):
        """Test equity curve chart generation."""
        html = create_equity_curve_chart(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "plotly" in html.lower()
        assert "equity-curve-chart" in html

    def test_monthly_performance_chart(self, sample_backtest_result):
        """Test monthly performance chart generation."""
        html = create_monthly_performance_chart(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "plotly" in html.lower()
        assert "monthly-performance-chart" in html

    def test_market_breakdown_chart(self, sample_backtest_result):
        """Test market breakdown chart generation."""
        html = create_market_breakdown_chart(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "plotly" in html.lower()
        assert "market-breakdown-chart" in html

    def test_bookmaker_breakdown_chart(self, sample_backtest_result):
        """Test bookmaker breakdown chart generation."""
        html = create_bookmaker_breakdown_chart(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "plotly" in html.lower()
        assert "bookmaker-breakdown-chart" in html

    def test_drawdown_chart(self, sample_backtest_result):
        """Test drawdown chart generation."""
        html = create_drawdown_chart(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "plotly" in html.lower()
        assert "drawdown-chart" in html

    def test_profit_distribution_chart(self, sample_backtest_result):
        """Test profit distribution chart generation."""
        html = create_profit_distribution_chart(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "plotly" in html.lower()
        assert "profit-distribution-chart" in html

    def test_profit_distribution_chart_no_data(self):
        """Test profit distribution chart with no profit data."""
        # Create result with no bets
        result = BacktestResult(
            strategy_name="Test",
            strategy_params={},
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
            total_days=30,
            initial_bankroll=10000.0,
            final_bankroll=10000.0,
            total_profit=0.0,
            roi=0.0,
            bet_stats=BetStatistics(
                total_bets=0,
                winning_bets=0,
                losing_bets=0,
                push_bets=0,
                win_rate=0.0,
                total_wagered=0.0,
                average_stake=0.0,
                average_odds=0.0,
                median_odds=0.0,
            ),
            risk_metrics=RiskMetrics(
                max_drawdown=0.0,
                max_drawdown_percentage=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                profit_factor=0.0,
                longest_winning_streak=0,
                longest_losing_streak=0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
            ),
            breakdowns=PerformanceBreakdown(
                market_breakdown={},
                bookmaker_breakdown={},
                monthly_performance=(),
            ),
            equity_curve=(),
            bets=(),
            total_events=0,
            events_with_complete_data=0,
            data_quality_issues=(),
            run_timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            execution_time_seconds=1.0,
        )

        html = create_profit_distribution_chart(result)
        assert isinstance(html, str)
        assert "No profit data available" in html


class TestTableGeneration:
    """Test table generation functions."""

    def test_risk_metrics_table(self, sample_backtest_result):
        """Test risk metrics table generation."""
        html = create_risk_metrics_table(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "Sharpe Ratio" in html
        assert "Max Drawdown" in html

    def test_market_breakdown_table(self, sample_backtest_result):
        """Test market breakdown table generation."""
        html = create_market_breakdown_table(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "Moneyline" in html or "h2h" in html.lower()

    def test_bookmaker_breakdown_table(self, sample_backtest_result):
        """Test bookmaker breakdown table generation."""
        html = create_bookmaker_breakdown_table(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "fanduel" in html.lower() or "Fanduel" in html

    def test_monthly_performance_table(self, sample_backtest_result):
        """Test monthly performance table generation."""
        html = create_monthly_performance_table(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "2024-10" in html

    def test_bet_summary_table(self, sample_backtest_result):
        """Test bet summary table generation."""
        html = create_bet_summary_table(sample_backtest_result)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "Total Bets" in html
        assert "Win Rate" in html


class TestHTMLReportGenerator:
    """Test HTML report generator."""

    def test_generator_initialization(self, sample_backtest_result):
        """Test generator can be initialized."""
        generator = HTMLReportGenerator(sample_backtest_result)
        assert generator.result == sample_backtest_result

    def test_generate_html_report(self, sample_backtest_result):
        """Test full HTML report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"

            generator = HTMLReportGenerator(sample_backtest_result)
            generator.generate(str(output_path))

            # Check file was created
            assert output_path.exists()

            # Read and verify content
            content = output_path.read_text(encoding="utf-8")
            assert len(content) > 0
            assert "<!DOCTYPE html>" in content
            assert "BasicEVStrategy" in content
            assert "Bootstrap" in content or "bootstrap" in content
            assert "plotly" in content.lower()

    def test_generate_creates_directory(self, sample_backtest_result):
        """Test that generate() creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "test_report.html"

            generator = HTMLReportGenerator(sample_backtest_result)
            generator.generate(str(output_path))

            assert output_path.exists()
            assert output_path.is_file()

    def test_generate_validates_bets(self):
        """Test that generate() validates result has bets."""
        result = BacktestResult(
            strategy_name="Test",
            strategy_params={},
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
            total_days=30,
            initial_bankroll=10000.0,
            final_bankroll=10000.0,
            total_profit=0.0,
            roi=0.0,
            bet_stats=BetStatistics(
                total_bets=0,
                winning_bets=0,
                losing_bets=0,
                push_bets=0,
                win_rate=0.0,
                total_wagered=0.0,
                average_stake=0.0,
                average_odds=0.0,
                median_odds=0.0,
            ),
            risk_metrics=RiskMetrics(
                max_drawdown=0.0,
                max_drawdown_percentage=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                profit_factor=0.0,
                longest_winning_streak=0,
                longest_losing_streak=0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
            ),
            breakdowns=PerformanceBreakdown(
                market_breakdown={},
                bookmaker_breakdown={},
                monthly_performance=(),
            ),
            equity_curve=(),
            bets=(),  # Empty bets
            total_events=0,
            events_with_complete_data=0,
            data_quality_issues=(),
            run_timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            execution_time_seconds=1.0,
        )

        generator = HTMLReportGenerator(result)

        with pytest.raises(ValueError, match="must contain at least one bet"):
            with tempfile.TemporaryDirectory() as tmpdir:
                generator.generate(str(Path(tmpdir) / "test.html"))

    def test_generate_validates_equity_curve(self, sample_backtest_result):
        """Test that generate() validates result has equity curve."""
        # Create result with bets but no equity curve
        result = BacktestResult(
            strategy_name=sample_backtest_result.strategy_name,
            strategy_params=sample_backtest_result.strategy_params,
            start_date=sample_backtest_result.start_date,
            end_date=sample_backtest_result.end_date,
            total_days=sample_backtest_result.total_days,
            initial_bankroll=sample_backtest_result.initial_bankroll,
            final_bankroll=sample_backtest_result.final_bankroll,
            total_profit=sample_backtest_result.total_profit,
            roi=sample_backtest_result.roi,
            bet_stats=sample_backtest_result.bet_stats,
            risk_metrics=sample_backtest_result.risk_metrics,
            breakdowns=sample_backtest_result.breakdowns,
            equity_curve=(),  # Empty equity curve
            bets=sample_backtest_result.bets,
            total_events=sample_backtest_result.total_events,
            events_with_complete_data=sample_backtest_result.events_with_complete_data,
            data_quality_issues=sample_backtest_result.data_quality_issues,
            run_timestamp=sample_backtest_result.run_timestamp,
            execution_time_seconds=sample_backtest_result.execution_time_seconds,
        )

        generator = HTMLReportGenerator(result)

        with pytest.raises(ValueError, match="must contain equity curve data"):
            with tempfile.TemporaryDirectory() as tmpdir:
                generator.generate(str(Path(tmpdir) / "test.html"))

    def test_html_contains_all_sections(self, sample_backtest_result):
        """Test that generated HTML contains all expected sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"

            generator = HTMLReportGenerator(sample_backtest_result)
            generator.generate(str(output_path))

            content = output_path.read_text(encoding="utf-8")

            # Check for key sections
            assert "Strategy Information" in content
            assert "Performance Visualizations" in content
            assert "Detailed Breakdowns" in content

            # Check for charts
            assert "Equity Curve" in content
            assert "Drawdown Analysis" in content
            assert "Monthly Performance" in content

            # Check for tables
            assert "Bet Statistics" in content
            assert "Risk & Performance Metrics" in content
