"""Backtesting framework for betting strategies."""

import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog
from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn, TextColumn

from analytics.utils import (
    calculate_kelly_stake,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_profit_from_odds,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from core.models import Event, EventStatus, Odds
from storage.readers import OddsReader

logger = structlog.get_logger()


class BacktestEvent(BaseModel):
    """
    Event validated for backtesting - guaranteed to have final scores.

    This model ensures type safety by requiring non-optional scores,
    preventing the use of incomplete events in backtesting logic.
    """

    id: str
    commence_time: datetime
    home_team: str
    away_team: str
    home_score: int  # Required - not optional!
    away_score: int  # Required - not optional!
    status: EventStatus

    @classmethod
    def from_db_event(cls, event: Event) -> "BacktestEvent | None":
        """
        Convert database Event to BacktestEvent if scores are present.

        Args:
            event: Database Event model

        Returns:
            BacktestEvent if scores exist, None otherwise
        """
        if event.home_score is None or event.away_score is None:
            return None

        return cls(
            id=event.id,
            commence_time=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            home_score=event.home_score,
            away_score=event.away_score,
            status=event.status,
        )


@dataclass
class BetRecord:
    """Complete record of a single bet placed during backtest."""

    # Identification
    bet_id: int
    event_id: str

    # Event details
    event_date: datetime
    home_team: str
    away_team: str

    # Bet details
    market: str  # h2h, spreads, totals
    outcome: str  # Team name or Over/Under
    bookmaker: str
    odds: int  # American odds
    line: float | None  # Spread/total line (None for h2h)

    # Timing
    decision_time: datetime  # When bet was placed (in backtest)

    # Stake and bankroll
    stake: float
    bankroll_before: float

    # Strategy context
    strategy_confidence: float | None = None  # Strategy's confidence/signal strength

    # Result (populated after game)
    result: str | None = None  # "win", "loss", "push"
    profit: float | None = None  # Profit/loss amount
    bankroll_after: float | None = None

    # Actual game result
    home_score: int | None = None
    away_score: int | None = None

    # Odds movement tracking
    opening_odds: int | None = None  # Opening line
    closing_odds: int | None = None  # Closing line
    our_odds: int = field(init=False)  # Odds we got (same as 'odds')

    # Additional context
    bet_rationale: str | None = None  # Why strategy made this bet

    def __post_init__(self):
        """Set our_odds to match odds field."""
        self.our_odds = self.odds

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bet_id": self.bet_id,
            "event_id": self.event_id,
            "event_date": self.event_date.isoformat(),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "market": self.market,
            "outcome": self.outcome,
            "bookmaker": self.bookmaker,
            "odds": self.odds,
            "line": self.line,
            "decision_time": self.decision_time.isoformat(),
            "stake": self.stake,
            "bankroll_before": self.bankroll_before,
            "strategy_confidence": self.strategy_confidence,
            "result": self.result,
            "profit": self.profit,
            "bankroll_after": self.bankroll_after,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "opening_odds": self.opening_odds,
            "closing_odds": self.closing_odds,
            "our_odds": self.our_odds,
            "bet_rationale": self.bet_rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BetRecord":
        """Reconstruct from dictionary."""
        # Convert ISO strings back to datetime
        data = data.copy()
        data["event_date"] = datetime.fromisoformat(data["event_date"])
        data["decision_time"] = datetime.fromisoformat(data["decision_time"])

        # Remove our_odds since it's auto-set in __post_init__
        data.pop("our_odds", None)

        return cls(**data)


@dataclass
class EquityPoint:
    """Single point on equity curve (daily snapshot)."""

    date: datetime
    bankroll: float
    cumulative_profit: float
    bets_to_date: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "date": self.date.isoformat(),
            "bankroll": self.bankroll,
            "cumulative_profit": self.cumulative_profit,
            "bets_to_date": self.bets_to_date,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EquityPoint":
        """Reconstruct from dictionary."""
        data = data.copy()
        data["date"] = datetime.fromisoformat(data["date"])
        return cls(**data)


@dataclass
class PerformanceStats:
    """Performance statistics for any grouping (market, bookmaker, time period, etc.)."""

    bets: int
    profit: float
    roi: float
    win_rate: float
    total_wagered: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bets": self.bets,
            "profit": round(self.profit, 2),
            "roi": round(self.roi, 2),
            "win_rate": round(self.win_rate, 2),
            "total_wagered": round(self.total_wagered, 2),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerformanceStats":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class MonthlyStats:
    """Performance statistics for a specific month."""

    month: str  # YYYY-MM format
    bets: int
    profit: float
    roi: float
    win_rate: float
    start_bankroll: float
    end_bankroll: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "month": self.month,
            "bets": self.bets,
            "profit": round(self.profit, 2),
            "roi": round(self.roi, 2),
            "win_rate": round(self.win_rate, 2),
            "start_bankroll": round(self.start_bankroll, 2),
            "end_bankroll": round(self.end_bankroll, 2),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MonthlyStats":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class RiskMetrics:
    """Risk and volatility metrics for backtest performance."""

    max_drawdown: float
    max_drawdown_percentage: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    longest_winning_streak: int
    longest_losing_streak: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_percentage": round(self.max_drawdown_percentage, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "profit_factor": round(self.profit_factor, 2),
            "longest_winning_streak": self.longest_winning_streak,
            "longest_losing_streak": self.longest_losing_streak,
            "average_win": round(self.average_win, 2),
            "average_loss": round(self.average_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RiskMetrics":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class BetStatistics:
    """Statistics about bets placed during backtest."""

    total_bets: int
    winning_bets: int
    losing_bets: int
    push_bets: int
    win_rate: float
    total_wagered: float
    average_stake: float
    average_odds: float
    median_odds: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_bets": self.total_bets,
            "winning_bets": self.winning_bets,
            "losing_bets": self.losing_bets,
            "push_bets": self.push_bets,
            "win_rate": round(self.win_rate, 2),
            "total_wagered": round(self.total_wagered, 2),
            "average_stake": round(self.average_stake, 2),
            "average_odds": round(self.average_odds, 2),
            "median_odds": round(self.median_odds, 2),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BetStatistics":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class PerformanceBreakdown:
    """Performance breakdowns by different dimensions."""

    market_breakdown: dict[str, PerformanceStats]
    bookmaker_breakdown: dict[str, PerformanceStats]
    monthly_performance: list[MonthlyStats]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "market_breakdown": {k: v.to_dict() for k, v in self.market_breakdown.items()},
            "bookmaker_breakdown": {k: v.to_dict() for k, v in self.bookmaker_breakdown.items()},
            "monthly_performance": [m.to_dict() for m in self.monthly_performance],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerformanceBreakdown":
        """Reconstruct from dictionary."""
        return cls(
            market_breakdown={
                k: PerformanceStats.from_dict(v) for k, v in data["market_breakdown"].items()
            },
            bookmaker_breakdown={
                k: PerformanceStats.from_dict(v) for k, v in data["bookmaker_breakdown"].items()
            },
            monthly_performance=[MonthlyStats.from_dict(m) for m in data["monthly_performance"]],
        )


@dataclass
class BacktestResult:
    """Complete backtesting results with all metrics and data using composed structure."""

    # Metadata
    strategy_name: str
    strategy_params: dict
    start_date: datetime
    end_date: datetime
    total_days: int
    initial_bankroll: float

    # Summary metrics
    final_bankroll: float
    total_profit: float
    roi: float

    # Composed metric groups
    bet_stats: BetStatistics
    risk_metrics: RiskMetrics
    breakdowns: PerformanceBreakdown

    # Time series data
    equity_curve: list[EquityPoint]
    bets: list[BetRecord]

    # Data quality
    total_events: int
    events_with_complete_data: int
    data_quality_issues: list[str]

    # Execution metadata
    run_timestamp: datetime
    execution_time_seconds: float

    # Convenience properties for backward compatibility
    @property
    def total_bets(self) -> int:
        """Total number of bets placed."""
        return self.bet_stats.total_bets

    @property
    def winning_bets(self) -> int:
        """Number of winning bets."""
        return self.bet_stats.winning_bets

    @property
    def losing_bets(self) -> int:
        """Number of losing bets."""
        return self.bet_stats.losing_bets

    @property
    def push_bets(self) -> int:
        """Number of push bets."""
        return self.bet_stats.push_bets

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        return self.bet_stats.win_rate

    @property
    def total_wagered(self) -> float:
        """Total amount wagered."""
        return self.bet_stats.total_wagered

    @property
    def average_stake(self) -> float:
        """Average stake per bet."""
        return self.bet_stats.average_stake

    @property
    def average_odds(self) -> float:
        """Average odds across all bets."""
        return self.bet_stats.average_odds

    @property
    def median_odds(self) -> float:
        """Median odds across all bets."""
        return self.bet_stats.median_odds

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown amount."""
        return self.risk_metrics.max_drawdown

    @property
    def max_drawdown_percentage(self) -> float:
        """Maximum drawdown percentage."""
        return self.risk_metrics.max_drawdown_percentage

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio."""
        return self.risk_metrics.sharpe_ratio

    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio."""
        return self.risk_metrics.sortino_ratio

    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio."""
        return self.risk_metrics.calmar_ratio

    @property
    def profit_factor(self) -> float:
        """Profit factor."""
        return self.risk_metrics.profit_factor

    @property
    def longest_winning_streak(self) -> int:
        """Longest winning streak."""
        return self.risk_metrics.longest_winning_streak

    @property
    def longest_losing_streak(self) -> int:
        """Longest losing streak."""
        return self.risk_metrics.longest_losing_streak

    @property
    def average_win(self) -> float:
        """Average winning bet amount."""
        return self.risk_metrics.average_win

    @property
    def average_loss(self) -> float:
        """Average losing bet amount."""
        return self.risk_metrics.average_loss

    @property
    def largest_win(self) -> float:
        """Largest single win."""
        return self.risk_metrics.largest_win

    @property
    def largest_loss(self) -> float:
        """Largest single loss."""
        return self.risk_metrics.largest_loss

    @property
    def market_breakdown(self) -> dict[str, PerformanceStats]:
        """Performance breakdown by market."""
        return self.breakdowns.market_breakdown

    @property
    def bookmaker_breakdown(self) -> dict[str, PerformanceStats]:
        """Performance breakdown by bookmaker."""
        return self.breakdowns.bookmaker_breakdown

    @property
    def monthly_performance(self) -> list[MonthlyStats]:
        """Performance breakdown by month."""
        return self.breakdowns.monthly_performance

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "strategy_name": self.strategy_name,
                "strategy_params": self.strategy_params,
                "backtest_period": {
                    "start_date": self.start_date.isoformat(),
                    "end_date": self.end_date.isoformat(),
                    "total_days": self.total_days,
                },
                "config": {
                    "initial_bankroll": self.initial_bankroll,
                },
                "data_quality": {
                    "total_events": self.total_events,
                    "events_with_complete_data": self.events_with_complete_data,
                    "data_quality_issues": self.data_quality_issues,
                },
                "run_timestamp": self.run_timestamp.isoformat(),
                "execution_time_seconds": round(self.execution_time_seconds, 2),
            },
            "summary": {
                "initial_bankroll": round(self.initial_bankroll, 2),
                "final_bankroll": round(self.final_bankroll, 2),
                "total_profit": round(self.total_profit, 2),
                "total_profit_percentage": round(
                    (self.total_profit / self.initial_bankroll) * 100, 2
                ),
                "roi": round(self.roi, 2),
                "bet_statistics": self.bet_stats.to_dict(),
                "risk_metrics": self.risk_metrics.to_dict(),
                "market_breakdown": self.breakdowns.to_dict()["market_breakdown"],
                "bookmaker_breakdown": self.breakdowns.to_dict()["bookmaker_breakdown"],
                "monthly_performance": self.breakdowns.to_dict()["monthly_performance"],
            },
            "equity_curve": [e.to_dict() for e in self.equity_curve],
            "bets": [b.to_dict() for b in self.bets],
        }

    def to_json(self, filepath: str | None = None, indent: int = 2) -> str:
        """
        Export to JSON format.

        Args:
            filepath: If provided, save to file. Otherwise return string.
            indent: JSON indentation level

        Returns:
            JSON string if filepath is None
        """
        json_str = json.dumps(self.to_dict(), indent=indent)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            return f"Saved to {filepath}"

        return json_str

    @classmethod
    def from_json(cls, filepath: str) -> "BacktestResult":
        """
        Reconstruct BacktestResult from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Reconstructed BacktestResult instance
        """
        with open(filepath) as f:
            data = json.load(f)

        # Reconstruct metadata
        metadata = data["metadata"]
        summary = data["summary"]

        # Parse dates
        start_date = datetime.fromisoformat(metadata["backtest_period"]["start_date"])
        end_date = datetime.fromisoformat(metadata["backtest_period"]["end_date"])
        run_timestamp = datetime.fromisoformat(metadata["run_timestamp"])

        # Reconstruct composed objects
        bet_stats = BetStatistics.from_dict(summary["bet_statistics"])
        risk_metrics = RiskMetrics.from_dict(summary["risk_metrics"])
        breakdowns = PerformanceBreakdown.from_dict(
            {
                "market_breakdown": summary["market_breakdown"],
                "bookmaker_breakdown": summary["bookmaker_breakdown"],
                "monthly_performance": summary["monthly_performance"],
            }
        )

        equity_curve = [EquityPoint.from_dict(e) for e in data["equity_curve"]]
        bets = [BetRecord.from_dict(b) for b in data["bets"]]

        return cls(
            strategy_name=metadata["strategy_name"],
            strategy_params=metadata["strategy_params"],
            start_date=start_date,
            end_date=end_date,
            total_days=metadata["backtest_period"]["total_days"],
            initial_bankroll=metadata["config"]["initial_bankroll"],
            final_bankroll=summary["final_bankroll"],
            total_profit=summary["total_profit"],
            roi=summary["roi"],
            bet_stats=bet_stats,
            risk_metrics=risk_metrics,
            breakdowns=breakdowns,
            equity_curve=equity_curve,
            bets=bets,
            total_events=metadata["data_quality"]["total_events"],
            events_with_complete_data=metadata["data_quality"]["events_with_complete_data"],
            data_quality_issues=metadata["data_quality"]["data_quality_issues"],
            run_timestamp=run_timestamp,
            execution_time_seconds=metadata["execution_time_seconds"],
        )

    def to_csv(self, filepath: str) -> str:
        """
        Export bets to CSV format for spreadsheet analysis.

        Args:
            filepath: Path to save CSV file

        Returns:
            Status message
        """
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "bet_id",
                    "timestamp",
                    "event_id",
                    "event_date",
                    "home_team",
                    "away_team",
                    "market",
                    "outcome",
                    "bookmaker",
                    "odds",
                    "line",
                    "stake",
                    "bankroll_before",
                    "result",
                    "profit",
                    "bankroll_after",
                    "home_score",
                    "away_score",
                    "roi",
                    "cumulative_profit",
                    "strategy_confidence",
                ],
            )
            writer.writeheader()

            cumulative_profit = 0.0
            for bet in self.bets:
                if bet.profit is not None:
                    cumulative_profit += bet.profit
                    bet_roi = (bet.profit / bet.stake * 100) if bet.stake > 0 else 0.0
                else:
                    bet_roi = 0.0

                writer.writerow(
                    {
                        "bet_id": bet.bet_id,
                        "timestamp": bet.decision_time.isoformat(),
                        "event_id": bet.event_id,
                        "event_date": bet.event_date.isoformat(),
                        "home_team": bet.home_team,
                        "away_team": bet.away_team,
                        "market": bet.market,
                        "outcome": bet.outcome,
                        "bookmaker": bet.bookmaker,
                        "odds": bet.odds,
                        "line": bet.line if bet.line is not None else "",
                        "stake": round(bet.stake, 2),
                        "bankroll_before": round(bet.bankroll_before, 2),
                        "result": bet.result or "",
                        "profit": round(bet.profit, 2) if bet.profit is not None else "",
                        "bankroll_after": round(bet.bankroll_after, 2)
                        if bet.bankroll_after is not None
                        else "",
                        "home_score": bet.home_score if bet.home_score is not None else "",
                        "away_score": bet.away_score if bet.away_score is not None else "",
                        "roi": round(bet_roi, 2),
                        "cumulative_profit": round(cumulative_profit, 2),
                        "strategy_confidence": bet.strategy_confidence
                        if bet.strategy_confidence is not None
                        else "",
                    }
                )

        return f"Saved {len(self.bets)} bets to {filepath}"

    def to_summary_text(self) -> str:
        """
        Generate Rich-formatted summary text for CLI display.

        Returns:
            Formatted summary string (to be rendered with Rich)
        """
        profit_indicator = "✓" if self.total_profit > 0 else "✗"

        summary = f"""
╔═══════════════════════════════════════════════════════════════╗
║         Backtest Results - {self.strategy_name:<31s}║
╠═══════════════════════════════════════════════════════════════╣
║ Period:           {self.start_date.date()} to {self.end_date.date()}                   ║
║ Total Days:       {self.total_days:<47d}║
║ Initial Bankroll: ${self.initial_bankroll:,.2f}{' ' * (43 - len(f'{self.initial_bankroll:,.2f}'))}║
║ Final Bankroll:   ${self.final_bankroll:,.2f}{' ' * (43 - len(f'{self.final_bankroll:,.2f}'))}║
║ Total Profit:     ${self.total_profit:,.2f} {profit_indicator}{' ' * (40 - len(f'{self.total_profit:,.2f}'))}║
║                                                               ║
║ PERFORMANCE METRICS                                           ║
║ ├─ ROI:              {self.roi:.2f}%{' ' * (42 - len(f'{self.roi:.2f}'))}║
║ ├─ Total Bets:       {self.total_bets:<42d}║
║ ├─ Win Rate:         {self.win_rate:.2f}% ({self.winning_bets}W / {self.losing_bets}L){' ' * (25 - len(f'{self.win_rate:.2f}% ({self.winning_bets}W / {self.losing_bets}L)'))}║
║ ├─ Avg Stake:        ${self.average_stake:,.2f}{' ' * (39 - len(f'{self.average_stake:,.2f}'))}║
║ └─ Total Wagered:    ${self.total_wagered:,.2f}{' ' * (39 - len(f'{self.total_wagered:,.2f}'))}║
║                                                               ║
║ RISK METRICS                                                  ║
║ ├─ Max Drawdown:     ${abs(self.max_drawdown):,.2f} ({self.max_drawdown_percentage:.2f}%){' ' * (25 - len(f'{abs(self.max_drawdown):,.2f} ({self.max_drawdown_percentage:.2f}%)'))}║
║ ├─ Sharpe Ratio:     {self.sharpe_ratio:.2f}{' ' * (42 - len(f'{self.sharpe_ratio:.2f}'))}║
║ ├─ Sortino Ratio:    {self.sortino_ratio:.2f}{' ' * (42 - len(f'{self.sortino_ratio:.2f}'))}║
║ ├─ Profit Factor:    {self.profit_factor:.2f}{' ' * (42 - len(f'{self.profit_factor:.2f}'))}║
║ └─ Longest Streak:   {self.longest_winning_streak} wins / {self.longest_losing_streak} losses{' ' * (29 - len(f'{self.longest_winning_streak} wins / {self.longest_losing_streak} losses'))}║
╚═══════════════════════════════════════════════════════════════╝
"""
        return summary


@dataclass
class BetOpportunity:
    """Represents a betting opportunity identified by a strategy."""

    event_id: str
    market: str
    outcome: str
    bookmaker: str
    odds: int
    line: float | None
    confidence: float  # 0-1, strategy's confidence in this bet
    rationale: str  # Why this bet was selected
    recommended_stake: float | None = None  # Strategy can suggest stake


@dataclass
class BetSizingConfig:
    """Configuration for bet sizing strategy."""

    method: str = "fractional_kelly"  # "fractional_kelly", "flat", "percentage"
    kelly_fraction: float = 0.25
    flat_stake_amount: float = 100.0
    percentage_stake: float = 0.02  # 2% of bankroll

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "kelly_fraction": self.kelly_fraction,
            "flat_stake_amount": self.flat_stake_amount,
            "percentage_stake": self.percentage_stake,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BetSizingConfig":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class BetConstraintsConfig:
    """Configuration for bet constraints and filters."""

    # Size constraints
    min_bet_size: float = 10.0
    max_bet_size: float | None = None
    max_bet_percentage: float = 0.05  # Max 5% of bankroll per bet

    # Odds filters
    min_odds: int | None = None  # e.g., -200 (don't bet heavy favorites)
    max_odds: int | None = None  # e.g., +300 (don't bet long shots)

    # Market/bookmaker filters
    allowed_markets: list[str] | None = None
    allowed_bookmakers: list[str] | None = None

    # Transaction costs
    include_transaction_costs: bool = False
    transaction_cost_rate: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "min_bet_size": self.min_bet_size,
            "max_bet_size": self.max_bet_size,
            "max_bet_percentage": self.max_bet_percentage,
            "min_odds": self.min_odds,
            "max_odds": self.max_odds,
            "allowed_markets": self.allowed_markets,
            "allowed_bookmakers": self.allowed_bookmakers,
            "include_transaction_costs": self.include_transaction_costs,
            "transaction_cost_rate": self.transaction_cost_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BetConstraintsConfig":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class BacktestConfig:
    """Configuration for running a backtest."""

    # Core settings
    initial_bankroll: float
    start_date: datetime
    end_date: datetime

    # Decision timing (when to "place" bets before game)
    decision_hours_before_game: float = 1.0  # Place bets 1 hour before

    # Composed configuration groups
    sizing: BetSizingConfig = field(default_factory=BetSizingConfig)
    constraints: BetConstraintsConfig = field(default_factory=BetConstraintsConfig)

    # Convenience properties for backward compatibility
    @property
    def bet_sizing_method(self) -> str:
        """Bet sizing method."""
        return self.sizing.method

    @property
    def kelly_fraction(self) -> float:
        """Kelly fraction."""
        return self.sizing.kelly_fraction

    @property
    def flat_stake_amount(self) -> float:
        """Flat stake amount."""
        return self.sizing.flat_stake_amount

    @property
    def percentage_stake(self) -> float:
        """Percentage stake."""
        return self.sizing.percentage_stake

    @property
    def min_bet_size(self) -> float:
        """Minimum bet size."""
        return self.constraints.min_bet_size

    @property
    def max_bet_size(self) -> float | None:
        """Maximum bet size."""
        return self.constraints.max_bet_size

    @property
    def max_bet_percentage(self) -> float:
        """Maximum bet percentage."""
        return self.constraints.max_bet_percentage

    @property
    def min_odds(self) -> int | None:
        """Minimum odds."""
        return self.constraints.min_odds

    @property
    def max_odds(self) -> int | None:
        """Maximum odds."""
        return self.constraints.max_odds

    @property
    def allowed_markets(self) -> list[str] | None:
        """Allowed markets."""
        return self.constraints.allowed_markets

    @property
    def allowed_bookmakers(self) -> list[str] | None:
        """Allowed bookmakers."""
        return self.constraints.allowed_bookmakers

    @property
    def include_transaction_costs(self) -> bool:
        """Whether to include transaction costs."""
        return self.constraints.include_transaction_costs

    @property
    def transaction_cost_rate(self) -> float:
        """Transaction cost rate."""
        return self.constraints.transaction_cost_rate

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "initial_bankroll": self.initial_bankroll,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "decision_hours_before_game": self.decision_hours_before_game,
            "sizing": self.sizing.to_dict(),
            "constraints": self.constraints.to_dict(),
        }


class BettingStrategy(ABC):
    """Abstract base class for all betting strategies."""

    def __init__(self, name: str, **params):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            **params: Strategy-specific parameters
        """
        self.name = name
        self.params = params

    @abstractmethod
    async def evaluate_opportunity(
        self,
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        """
        Evaluate an event and return betting opportunities.

        Args:
            event: BacktestEvent with game details and guaranteed final scores
            odds_snapshot: All available odds at decision time
            config: Backtest configuration

        Returns:
            List of BetOpportunity objects (empty if no bets)
        """
        pass

    def get_name(self) -> str:
        """Get strategy name."""
        return self.name

    def get_params(self) -> dict:
        """Get strategy parameters."""
        return self.params


class BacktestEngine:
    """Execute backtests of betting strategies against historical data."""

    def __init__(
        self,
        strategy: BettingStrategy,
        config: BacktestConfig,
        reader: OddsReader,
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Betting strategy to test
            config: Backtest configuration
            reader: Data reader for events and odds
        """
        self.strategy = strategy
        self.config = config
        self.reader = reader

    async def run(self) -> BacktestResult:
        """
        Execute backtest and return complete results.

        Returns:
            BacktestResult with all metrics and data

        Process:
            1. Query all events in date range with final results
            2. For each event, get odds at decision time
            3. Apply strategy to identify bets
            4. Calculate results and track bankroll
            5. Compute all performance metrics
        """
        start_time = time.time()

        logger.info(
            "backtest_starting",
            strategy=self.strategy.get_name(),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

        # Query events with results
        events = await self._get_events_with_results()

        if not events:
            logger.warning("no_events_found")
            return self._create_empty_result(time.time() - start_time)

        # Process events and place bets
        bets = []
        bankroll = self.config.initial_bankroll
        bet_id = 1

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(
                f"Backtesting {len(events)} events...",
                total=len(events),
            )

            for event in events:
                # Get odds at decision time (e.g., 1 hour before game)
                decision_time = event.commence_time - timedelta(
                    hours=self.config.decision_hours_before_game
                )

                odds_snapshot = await self.reader.get_odds_at_time(
                    event.id,
                    decision_time,
                    tolerance_minutes=30,  # Allow 30min tolerance
                )

                if not odds_snapshot:
                    logger.debug(
                        "no_odds_at_decision_time",
                        event_id=event.id,
                        decision_time=decision_time,
                    )
                    progress.advance(task)
                    continue

                # Ask strategy for betting opportunities
                opportunities = await self.strategy.evaluate_opportunity(
                    event, odds_snapshot, self.config
                )

                # Place bets for each opportunity
                for opp in opportunities:
                    # Calculate stake
                    stake = self._calculate_stake(opp, bankroll)

                    if stake < self.config.min_bet_size:
                        logger.debug(
                            "stake_below_minimum",
                            stake=stake,
                            min_bet=self.config.min_bet_size,
                        )
                        continue

                    # Create bet record
                    bet = BetRecord(
                        bet_id=bet_id,
                        event_id=event.id,
                        event_date=event.commence_time,
                        home_team=event.home_team,
                        away_team=event.away_team,
                        market=opp.market,
                        outcome=opp.outcome,
                        bookmaker=opp.bookmaker,
                        odds=opp.odds,
                        line=opp.line,
                        decision_time=decision_time,
                        stake=stake,
                        bankroll_before=bankroll,
                        strategy_confidence=opp.confidence,
                        bet_rationale=opp.rationale,
                    )

                    # Determine result
                    result, profit = self._evaluate_bet_result(bet, event)

                    bet.result = result
                    bet.profit = profit
                    bet.home_score = event.home_score
                    bet.away_score = event.away_score

                    # Update bankroll
                    bankroll += profit
                    bet.bankroll_after = bankroll

                    bets.append(bet)
                    bet_id += 1

                progress.advance(task)

        # Calculate all metrics
        result = self._calculate_metrics(
            events,
            bets,
            execution_time=time.time() - start_time,
        )

        logger.info(
            "backtest_complete",
            total_bets=result.total_bets,
            final_bankroll=result.final_bankroll,
            roi=result.roi,
        )

        return result

    async def _get_events_with_results(self) -> list[BacktestEvent]:
        """
        Query events in date range that have final results.

        Returns:
            List of BacktestEvent objects with guaranteed scores
        """
        events = await self.reader.get_events_by_date_range(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            status=EventStatus.FINAL,
        )

        # Convert to BacktestEvent, filtering out events without scores
        backtest_events = []
        for event in events:
            bt_event = BacktestEvent.from_db_event(event)
            if bt_event is not None:
                backtest_events.append(bt_event)

        logger.info(
            "events_loaded",
            total_events=len(events),
            events_with_results=len(backtest_events),
        )

        return backtest_events

    def _calculate_stake(self, opportunity: BetOpportunity, bankroll: float) -> float:
        """
        Calculate stake size based on config and opportunity.

        Args:
            opportunity: Betting opportunity
            bankroll: Current bankroll

        Returns:
            Stake amount
        """
        if self.config.bet_sizing_method == "flat":
            stake = self.config.flat_stake_amount

        elif self.config.bet_sizing_method == "percentage":
            stake = bankroll * self.config.percentage_stake

        elif self.config.bet_sizing_method == "fractional_kelly":
            # For Kelly, we need estimated win probability
            # Strategy confidence is used as probability estimate
            stake = calculate_kelly_stake(
                bet_probability=opportunity.confidence,
                american_odds=opportunity.odds,
                bankroll=bankroll,
                kelly_fraction=self.config.kelly_fraction,
                max_stake_percentage=self.config.max_bet_percentage,
            )

        else:
            raise ValueError(f"Unknown bet sizing method: {self.config.bet_sizing_method}")

        # Apply constraints
        if self.config.max_bet_size:
            stake = min(stake, self.config.max_bet_size)

        stake = max(stake, self.config.min_bet_size)

        return round(stake, 2)

    def _evaluate_bet_result(self, bet: BetRecord, event: BacktestEvent) -> tuple[str, float]:
        """
        Determine bet result and profit/loss.

        Args:
            bet: Bet record
            event: BacktestEvent with guaranteed final scores

        Returns:
            Tuple of (result_string, profit_amount)
        """
        # No need to check for None - BacktestEvent guarantees scores exist!

        # Determine if bet won based on market
        if bet.market == "h2h":
            won = self._evaluate_moneyline(bet, event)
        elif bet.market == "spreads":
            won = self._evaluate_spread(bet, event)
        elif bet.market == "totals":
            won = self._evaluate_total(bet, event)
        else:
            logger.warning("unknown_market", market=bet.market)
            return ("unknown", 0.0)

        if won is None:
            return ("push", 0.0)

        # Calculate profit
        profit = calculate_profit_from_odds(bet.stake, bet.odds, won)

        result_str = "win" if won else "loss"
        return (result_str, profit)

    def _evaluate_moneyline(self, bet: BetRecord, event: BacktestEvent) -> bool | None:
        """Evaluate moneyline bet."""
        # Determine winner
        if event.home_score > event.away_score:
            winner = event.home_team
        elif event.away_score > event.home_score:
            winner = event.away_team
        else:
            return None  # Push (tie)

        return bet.outcome == winner

    def _evaluate_spread(self, bet: BetRecord, event: BacktestEvent) -> bool | None:
        """Evaluate spread bet."""
        if bet.line is None:
            return None

        # Apply spread
        if bet.outcome == event.home_team:
            # Betting on home team with spread
            adjusted_score = event.home_score + bet.line
            won = adjusted_score > event.away_score
            push = adjusted_score == event.away_score
        else:
            # Betting on away team with spread
            # Note: away line is inverse of home line
            away_line = -bet.line
            adjusted_score = event.away_score + away_line
            won = adjusted_score > event.home_score
            push = adjusted_score == event.home_score

        if push:
            return None

        return won

    def _evaluate_total(self, bet: BetRecord, event: BacktestEvent) -> bool | None:
        """Evaluate totals (over/under) bet."""
        if bet.line is None:
            return None

        total_points = event.home_score + event.away_score

        if bet.outcome.lower() == "over":
            won = total_points > bet.line
            push = total_points == bet.line
        else:  # Under
            won = total_points < bet.line
            push = total_points == bet.line

        if push:
            return None

        return won

    def _calculate_metrics(
        self,
        events: list[BacktestEvent],
        bets: list[BetRecord],
        execution_time: float,
    ) -> BacktestResult:
        """
        Calculate all backtest metrics.

        Args:
            events: All events processed
            bets: All bets placed
            execution_time: Execution time in seconds

        Returns:
            Complete BacktestResult
        """
        if not bets:
            return self._create_empty_result(execution_time)

        # Basic statistics
        total_bets = len(bets)
        winning_bets = sum(1 for b in bets if b.result == "win")
        losing_bets = sum(1 for b in bets if b.result == "loss")
        push_bets = sum(1 for b in bets if b.result == "push")

        total_wagered = sum(b.stake for b in bets)
        total_profit = sum(b.profit for b in bets if b.profit is not None)

        final_bankroll = self.config.initial_bankroll + total_profit
        roi = (total_profit / self.config.initial_bankroll) * 100

        win_rate = (
            (winning_bets / (winning_bets + losing_bets) * 100)
            if (winning_bets + losing_bets) > 0
            else 0.0
        )

        average_stake = total_wagered / total_bets if total_bets > 0 else 0.0

        odds_list = [b.odds for b in bets]
        average_odds = sum(odds_list) / len(odds_list) if odds_list else 0
        median_odds = sorted(odds_list)[len(odds_list) // 2] if odds_list else 0

        # Risk metrics
        equity_curve_values = [self.config.initial_bankroll]
        for bet in bets:
            if bet.bankroll_after is not None:
                equity_curve_values.append(bet.bankroll_after)

        max_dd, max_dd_pct = calculate_max_drawdown(equity_curve_values)

        # Calculate daily returns for Sharpe/Sortino
        daily_returns = self._calculate_daily_returns(bets)
        sharpe = calculate_sharpe_ratio(daily_returns)
        sortino = calculate_sortino_ratio(daily_returns)

        # Calmar ratio: ROI / abs(max drawdown %)
        calmar = abs(roi / max_dd_pct) if max_dd_pct != 0 else 0.0

        # Profit factor
        winning_profit = sum(b.profit for b in bets if b.result == "win" and b.profit is not None)
        losing_loss = sum(
            abs(b.profit) for b in bets if b.result == "loss" and b.profit is not None
        )
        profit_factor = calculate_profit_factor(winning_profit, losing_loss)

        # Streaks
        longest_win_streak, longest_loss_streak = self._calculate_streaks(bets)

        # Win/loss averages
        wins = [b.profit for b in bets if b.result == "win" and b.profit is not None]
        losses = [b.profit for b in bets if b.result == "loss" and b.profit is not None]

        average_win = sum(wins) / len(wins) if wins else 0.0
        average_loss = sum(losses) / len(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        # Breakdowns
        market_breakdown = self._calculate_market_breakdown(bets)
        bookmaker_breakdown = self._calculate_bookmaker_breakdown(bets)
        monthly_performance = self._calculate_monthly_performance(bets)

        # Equity curve
        equity_curve = self._build_equity_curve(bets)

        # Data quality
        events_with_complete_data = len([e for e in events if e.home_score is not None])
        data_quality_issues = []
        if events_with_complete_data < len(events):
            data_quality_issues.append(
                f"{len(events) - events_with_complete_data} events missing scores"
            )

        total_days = (self.config.end_date - self.config.start_date).days

        # Create composed metric groups
        bet_stats = BetStatistics(
            total_bets=total_bets,
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            push_bets=push_bets,
            win_rate=win_rate,
            total_wagered=total_wagered,
            average_stake=average_stake,
            average_odds=average_odds,
            median_odds=median_odds,
        )

        risk_metrics_obj = RiskMetrics(
            max_drawdown=max_dd,
            max_drawdown_percentage=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            profit_factor=profit_factor,
            longest_winning_streak=longest_win_streak,
            longest_losing_streak=longest_loss_streak,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
        )

        breakdowns_obj = PerformanceBreakdown(
            market_breakdown=market_breakdown,
            bookmaker_breakdown=bookmaker_breakdown,
            monthly_performance=monthly_performance,
        )

        return BacktestResult(
            strategy_name=self.strategy.get_name(),
            strategy_params=self.strategy.get_params(),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            total_days=total_days,
            initial_bankroll=self.config.initial_bankroll,
            final_bankroll=final_bankroll,
            total_profit=total_profit,
            roi=roi,
            bet_stats=bet_stats,
            risk_metrics=risk_metrics_obj,
            breakdowns=breakdowns_obj,
            equity_curve=equity_curve,
            bets=bets,
            total_events=len(events),
            events_with_complete_data=events_with_complete_data,
            data_quality_issues=data_quality_issues,
            run_timestamp=datetime.utcnow(),
            execution_time_seconds=execution_time,
        )

    def _calculate_daily_returns(self, bets: list[BetRecord]) -> list[float]:
        """Calculate daily profit/loss for Sharpe/Sortino."""
        if not bets:
            return []

        # Group bets by date
        daily_profits: dict[str, float] = defaultdict(float)
        for bet in bets:
            if bet.profit is not None:
                date_key = bet.event_date.date().isoformat()
                daily_profits[date_key] += bet.profit

        return list(daily_profits.values())

    def _calculate_streaks(self, bets: list[BetRecord]) -> tuple[int, int]:
        """Calculate longest winning and losing streaks."""
        if not bets:
            return (0, 0)

        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for bet in bets:
            if bet.result == "win":
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif bet.result == "loss":
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        return (max_win_streak, max_loss_streak)

    def _calculate_market_breakdown(self, bets: list[BetRecord]) -> dict[str, PerformanceStats]:
        """Calculate statistics by market type."""
        markets: dict[str, list[BetRecord]] = defaultdict(list)
        for bet in bets:
            markets[bet.market].append(bet)

        breakdown = {}
        for market, market_bets in markets.items():
            total_bets = len(market_bets)
            wins = sum(1 for b in market_bets if b.result == "win")
            losses = sum(1 for b in market_bets if b.result == "loss")
            profit = sum(b.profit for b in market_bets if b.profit is not None)
            wagered = sum(b.stake for b in market_bets)

            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
            roi = (profit / wagered * 100) if wagered > 0 else 0.0

            breakdown[market] = PerformanceStats(
                bets=total_bets,
                profit=profit,
                roi=roi,
                win_rate=win_rate,
                total_wagered=wagered,
            )

        return breakdown

    def _calculate_bookmaker_breakdown(self, bets: list[BetRecord]) -> dict[str, PerformanceStats]:
        """Calculate statistics by bookmaker."""
        bookmakers: dict[str, list[BetRecord]] = defaultdict(list)
        for bet in bets:
            bookmakers[bet.bookmaker].append(bet)

        breakdown = {}
        for bookmaker, book_bets in bookmakers.items():
            total_bets = len(book_bets)
            wins = sum(1 for b in book_bets if b.result == "win")
            losses = sum(1 for b in book_bets if b.result == "loss")
            profit = sum(b.profit for b in book_bets if b.profit is not None)
            wagered = sum(b.stake for b in book_bets)

            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
            roi = (profit / wagered * 100) if wagered > 0 else 0.0

            breakdown[bookmaker] = PerformanceStats(
                bets=total_bets,
                profit=profit,
                roi=roi,
                win_rate=win_rate,
                total_wagered=wagered,
            )

        return breakdown

    def _calculate_monthly_performance(self, bets: list[BetRecord]) -> list[MonthlyStats]:
        """Calculate statistics by month."""
        if not bets:
            return []

        months: dict[str, list[BetRecord]] = defaultdict(list)
        for bet in bets:
            month_key = bet.event_date.strftime("%Y-%m")
            months[month_key].append(bet)

        monthly_stats = []
        cumulative_bankroll = self.config.initial_bankroll

        for month in sorted(months.keys()):
            month_bets = months[month]
            total_bets = len(month_bets)
            wins = sum(1 for b in month_bets if b.result == "win")
            losses = sum(1 for b in month_bets if b.result == "loss")
            profit = sum(b.profit for b in month_bets if b.profit is not None)
            wagered = sum(b.stake for b in month_bets)

            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
            roi = (profit / wagered * 100) if wagered > 0 else 0.0

            start_bankroll = cumulative_bankroll
            end_bankroll = cumulative_bankroll + profit
            cumulative_bankroll = end_bankroll

            monthly_stats.append(
                MonthlyStats(
                    month=month,
                    bets=total_bets,
                    profit=profit,
                    roi=roi,
                    win_rate=win_rate,
                    start_bankroll=start_bankroll,
                    end_bankroll=end_bankroll,
                )
            )

        return monthly_stats

    def _build_equity_curve(self, bets: list[BetRecord]) -> list[EquityPoint]:
        """Build daily equity curve."""
        if not bets:
            return []

        # Group bets by date
        daily_data: dict[str, tuple[float, int]] = {}  # date -> (profit, bet_count)
        for bet in bets:
            date_key = bet.event_date.date()
            if date_key not in daily_data:
                daily_data[date_key] = (0.0, 0)

            profit, count = daily_data[date_key]
            if bet.profit is not None:
                daily_data[date_key] = (profit + bet.profit, count + 1)

        # Build curve
        equity_curve = []
        cumulative_profit = 0.0
        cumulative_bets = 0

        for date in sorted(daily_data.keys()):
            daily_profit, daily_bets = daily_data[date]
            cumulative_profit += daily_profit
            cumulative_bets += daily_bets

            equity_curve.append(
                EquityPoint(
                    date=datetime.combine(date, datetime.min.time()),
                    bankroll=self.config.initial_bankroll + cumulative_profit,
                    cumulative_profit=cumulative_profit,
                    bets_to_date=cumulative_bets,
                )
            )

        return equity_curve

    def _create_empty_result(self, execution_time: float) -> BacktestResult:
        """Create result object when no bets were placed."""
        # Create empty composed metric groups
        bet_stats = BetStatistics(
            total_bets=0,
            winning_bets=0,
            losing_bets=0,
            push_bets=0,
            win_rate=0.0,
            total_wagered=0.0,
            average_stake=0.0,
            average_odds=0,
            median_odds=0,
        )

        risk_metrics_obj = RiskMetrics(
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
        )

        breakdowns_obj = PerformanceBreakdown(
            market_breakdown={},
            bookmaker_breakdown={},
            monthly_performance=[],
        )

        return BacktestResult(
            strategy_name=self.strategy.get_name(),
            strategy_params=self.strategy.get_params(),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            total_days=(self.config.end_date - self.config.start_date).days,
            initial_bankroll=self.config.initial_bankroll,
            final_bankroll=self.config.initial_bankroll,
            total_profit=0.0,
            roi=0.0,
            bet_stats=bet_stats,
            risk_metrics=risk_metrics_obj,
            breakdowns=breakdowns_obj,
            equity_curve=[],
            bets=[],
            total_events=0,
            events_with_complete_data=0,
            data_quality_issues=["No bets placed"],
            run_timestamp=datetime.utcnow(),
            execution_time_seconds=execution_time,
        )
