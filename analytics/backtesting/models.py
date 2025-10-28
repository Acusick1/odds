"""Core data models used by the backtesting framework."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

from core.models import Event, EventStatus

__all__ = [
    "BacktestEvent",
    "BetRecord",
    "EquityPoint",
    "PerformanceStats",
    "MonthlyStats",
    "RiskMetrics",
    "BetStatistics",
    "PerformanceBreakdown",
    "BacktestResult",
    "BetOpportunity",
]


class BacktestEvent(BaseModel):
    """Event validated for backtesting - guaranteed to have final scores."""

    id: str
    commence_time: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    status: EventStatus

    @classmethod
    def from_db_event(cls, event: Event) -> BacktestEvent | None:
        """Convert database Event to BacktestEvent if scores are present."""
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


@dataclass(frozen=True)
class BetRecord:
    """Complete record of a single bet placed during backtest (immutable)."""

    bet_id: int
    event_id: str
    event_date: datetime
    home_team: str
    away_team: str
    market: str
    outcome: str
    bookmaker: str
    odds: int
    line: float | None
    decision_time: datetime
    stake: float
    bankroll_before: float
    strategy_confidence: float | None = None
    result: str | None = None
    profit: float | None = None
    bankroll_after: float | None = None
    home_score: int | None = None
    away_score: int | None = None
    opening_odds: int | None = None
    closing_odds: int | None = None
    our_odds: int = field(init=False)
    bet_rationale: str | None = None

    def __post_init__(self) -> None:
        """Set our_odds to match odds field."""
        object.__setattr__(self, "our_odds", self.odds)

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
    def from_dict(cls, data: dict) -> BetRecord:
        """Reconstruct from dictionary."""
        payload = data.copy()
        payload["event_date"] = datetime.fromisoformat(payload["event_date"])
        payload["decision_time"] = datetime.fromisoformat(payload["decision_time"])
        payload.pop("our_odds", None)
        return cls(**payload)


@dataclass(frozen=True)
class EquityPoint:
    """Single point on equity curve (daily snapshot, immutable)."""

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
    def from_dict(cls, data: dict) -> EquityPoint:
        """Reconstruct from dictionary."""
        payload = data.copy()
        payload["date"] = datetime.fromisoformat(payload["date"])
        return cls(**payload)


@dataclass(frozen=True)
class PerformanceStats:
    """Performance statistics for any grouping (immutable)."""

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
    def from_dict(cls, data: dict) -> PerformanceStats:
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class MonthlyStats:
    """Performance statistics for a specific month (immutable)."""

    month: str
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
    def from_dict(cls, data: dict) -> MonthlyStats:
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class RiskMetrics:
    """Risk and volatility metrics for backtest performance (immutable)."""

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
    def from_dict(cls, data: dict) -> RiskMetrics:
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class BetStatistics:
    """Statistics about bets placed during backtest (immutable)."""

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
    def from_dict(cls, data: dict) -> BetStatistics:
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class PerformanceBreakdown:
    """Performance breakdowns by different dimensions (immutable)."""

    market_breakdown: dict[str, PerformanceStats]
    bookmaker_breakdown: dict[str, PerformanceStats]
    monthly_performance: tuple[MonthlyStats, ...]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "market_breakdown": {k: v.to_dict() for k, v in self.market_breakdown.items()},
            "bookmaker_breakdown": {k: v.to_dict() for k, v in self.bookmaker_breakdown.items()},
            "monthly_performance": [m.to_dict() for m in self.monthly_performance],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PerformanceBreakdown:
        """Reconstruct from dictionary."""
        return cls(
            market_breakdown={
                key: PerformanceStats.from_dict(value)
                for key, value in data["market_breakdown"].items()
            },
            bookmaker_breakdown={
                key: PerformanceStats.from_dict(value)
                for key, value in data["bookmaker_breakdown"].items()
            },
            monthly_performance=tuple(
                MonthlyStats.from_dict(item) for item in data["monthly_performance"]
            ),
        )


@dataclass(frozen=True)
class BacktestResult:
    """Complete backtesting results with all metrics and data (immutable)."""

    strategy_name: str
    strategy_params: dict
    start_date: datetime
    end_date: datetime
    total_days: int
    initial_bankroll: float
    final_bankroll: float
    total_profit: float
    roi: float
    bet_stats: BetStatistics
    risk_metrics: RiskMetrics
    breakdowns: PerformanceBreakdown
    equity_curve: tuple[EquityPoint, ...]
    bets: tuple[BetRecord, ...]
    total_events: int
    events_with_complete_data: int
    data_quality_issues: tuple[str, ...]
    run_timestamp: datetime
    execution_time_seconds: float

    @property
    def total_bets(self) -> int:
        return self.bet_stats.total_bets

    @property
    def winning_bets(self) -> int:
        return self.bet_stats.winning_bets

    @property
    def losing_bets(self) -> int:
        return self.bet_stats.losing_bets

    @property
    def push_bets(self) -> int:
        return self.bet_stats.push_bets

    @property
    def win_rate(self) -> float:
        return self.bet_stats.win_rate

    @property
    def total_wagered(self) -> float:
        return self.bet_stats.total_wagered

    @property
    def average_stake(self) -> float:
        return self.bet_stats.average_stake

    @property
    def average_odds(self) -> float:
        return self.bet_stats.average_odds

    @property
    def median_odds(self) -> float:
        return self.bet_stats.median_odds

    @property
    def max_drawdown(self) -> float:
        return self.risk_metrics.max_drawdown

    @property
    def max_drawdown_percentage(self) -> float:
        return self.risk_metrics.max_drawdown_percentage

    @property
    def sharpe_ratio(self) -> float:
        return self.risk_metrics.sharpe_ratio

    @property
    def sortino_ratio(self) -> float:
        return self.risk_metrics.sortino_ratio

    @property
    def calmar_ratio(self) -> float:
        return self.risk_metrics.calmar_ratio

    @property
    def profit_factor(self) -> float:
        return self.risk_metrics.profit_factor

    @property
    def longest_winning_streak(self) -> int:
        return self.risk_metrics.longest_winning_streak

    @property
    def longest_losing_streak(self) -> int:
        return self.risk_metrics.longest_losing_streak

    @property
    def average_win(self) -> float:
        return self.risk_metrics.average_win

    @property
    def average_loss(self) -> float:
        return self.risk_metrics.average_loss

    @property
    def largest_win(self) -> float:
        return self.risk_metrics.largest_win

    @property
    def largest_loss(self) -> float:
        return self.risk_metrics.largest_loss

    @property
    def market_breakdown(self) -> dict[str, PerformanceStats]:
        return self.breakdowns.market_breakdown

    @property
    def bookmaker_breakdown(self) -> dict[str, PerformanceStats]:
        return self.breakdowns.bookmaker_breakdown

    @property
    def monthly_performance(self) -> list[MonthlyStats]:
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
            "equity_curve": [point.to_dict() for point in self.equity_curve],
            "bets": [bet.to_dict() for bet in self.bets],
        }

    def to_json(self, filepath: str | None = None, indent: int = 2) -> str:
        """Export to JSON format."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if filepath:
            with open(filepath, "w") as file_object:
                file_object.write(json_str)
            return f"Saved to {filepath}"
        return json_str

    @classmethod
    def from_json(cls, filepath: str) -> BacktestResult:
        """Reconstruct BacktestResult from JSON file."""
        with open(filepath) as file_object:
            data = json.load(file_object)

        metadata = data["metadata"]
        summary = data["summary"]

        start_date = datetime.fromisoformat(metadata["backtest_period"]["start_date"])
        end_date = datetime.fromisoformat(metadata["backtest_period"]["end_date"])
        run_timestamp = datetime.fromisoformat(metadata["run_timestamp"])

        bet_stats = BetStatistics.from_dict(summary["bet_statistics"])
        risk_metrics = RiskMetrics.from_dict(summary["risk_metrics"])
        breakdowns = PerformanceBreakdown.from_dict(
            {
                "market_breakdown": summary["market_breakdown"],
                "bookmaker_breakdown": summary["bookmaker_breakdown"],
                "monthly_performance": summary["monthly_performance"],
            }
        )

        equity_curve = tuple(EquityPoint.from_dict(item) for item in data["equity_curve"])
        bets = tuple(BetRecord.from_dict(item) for item in data["bets"])

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
            data_quality_issues=tuple(metadata["data_quality"]["data_quality_issues"]),
            run_timestamp=run_timestamp,
            execution_time_seconds=metadata["execution_time_seconds"],
        )

    def to_csv(self, filepath: str) -> str:
        """Export bets to CSV format for spreadsheet analysis."""
        with open(filepath, "w", newline="") as file_object:
            writer = csv.DictWriter(
                file_object,
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
        """Generate Rich-formatted summary text for CLI display."""
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


@dataclass(frozen=True)
class BetOpportunity:
    """Represents a betting opportunity identified by a strategy (immutable)."""

    event_id: str
    market: str
    outcome: str
    bookmaker: str
    odds: int
    line: float | None
    confidence: float
    rationale: str
    recommended_stake: float | None = None
