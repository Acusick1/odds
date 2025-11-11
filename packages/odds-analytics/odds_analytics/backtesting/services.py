"""Services responsible for executing backtests."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import structlog
from rich.progress import Progress, SpinnerColumn, TextColumn

from odds_core.models import EventStatus, Odds
from odds_lambda.storage.readers import OddsReader

from ..utils import (
    calculate_kelly_stake,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_profit_from_odds,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from .config import BacktestConfig
from .models import (
    BacktestEvent,
    BacktestResult,
    BetOpportunity,
    BetRecord,
    BetStatistics,
    EquityPoint,
    MonthlyStats,
    PerformanceBreakdown,
    PerformanceStats,
    RiskMetrics,
)

__all__ = ["BettingStrategy", "BacktestEngine"]

logger = structlog.get_logger()


class BettingStrategy(ABC):
    """Abstract base class for all betting strategies."""

    __slots__ = ("name", "params")

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    async def evaluate_opportunity(
        self,
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        """Evaluate an event and return betting opportunities."""

    def get_name(self) -> str:
        """Return strategy name for reporting."""
        return self.name

    def get_params(self) -> dict:
        """Expose strategy parameters."""
        return self.params


class BacktestEngine:
    """
    Execute backtests of betting strategies against historical data.

    Supports dependency injection for better testability and flexibility.
    """

    def __init__(
        self,
        strategy: BettingStrategy,
        config: BacktestConfig,
        reader: OddsReader,
        *,
        progress_callback: Callable[[str], None] | None = None,
        logger_instance: structlog.BoundLogger = logger,
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Betting strategy to test
            config: Backtest configuration
            reader: Database reader for odds data
            progress_callback: Optional callback for progress updates (for testing)
            logger_instance: Optional logger instance (for testing)
        """
        self.strategy = strategy
        self.config = config
        self.reader = reader
        self._progress_callback = progress_callback
        self._logger = logger_instance

    async def run(self) -> BacktestResult:
        start_time = time.time()

        self._logger.info(
            "backtest_starting",
            strategy=self.strategy.get_name(),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

        events = await self._get_events_with_results()

        if not events:
            self._logger.warning("no_events_found")
            return self._create_empty_result(time.time() - start_time)

        bets: list[BetRecord] = []
        bankroll = self.config.initial_bankroll
        bet_id = 1

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}")
        ) as progress:
            task = progress.add_task(f"Backtesting {len(events)} events...", total=len(events))

            for event in events:
                decision_time = event.commence_time - timedelta(
                    hours=self.config.decision_hours_before_game
                )

                odds_snapshot = await self.reader.get_odds_at_time(
                    event.id,
                    decision_time,
                    tolerance_minutes=30,
                )

                if not odds_snapshot:
                    self._logger.debug(
                        "no_odds_at_decision_time",
                        event_id=event.id,
                        decision_time=decision_time,
                    )
                    progress.advance(task)
                    continue

                opportunities = await self.strategy.evaluate_opportunity(
                    event, odds_snapshot, self.config
                )

                for opportunity in opportunities:
                    stake = self._calculate_stake(opportunity, bankroll)

                    if stake < self.config.min_bet_size:
                        self._logger.debug(
                            "stake_below_minimum", stake=stake, min_bet=self.config.min_bet_size
                        )
                        continue

                    # Evaluate bet result before creating the immutable record
                    result, profit = self._evaluate_bet_result_for_opportunity(
                        opportunity, event, stake
                    )

                    # Calculate final bankroll after this bet
                    bankroll_after = bankroll + profit

                    # Create immutable bet record with all fields populated
                    bet = BetRecord(
                        bet_id=bet_id,
                        event_id=event.id,
                        event_date=event.commence_time,
                        home_team=event.home_team,
                        away_team=event.away_team,
                        market=opportunity.market,
                        outcome=opportunity.outcome,
                        bookmaker=opportunity.bookmaker,
                        odds=opportunity.odds,
                        line=opportunity.line,
                        decision_time=decision_time,
                        stake=stake,
                        bankroll_before=bankroll,
                        strategy_confidence=opportunity.confidence,
                        result=result,
                        profit=profit,
                        bankroll_after=bankroll_after,
                        home_score=event.home_score,
                        away_score=event.away_score,
                        bet_rationale=opportunity.rationale,
                    )

                    bankroll = bankroll_after
                    bets.append(bet)
                    bet_id += 1

                progress.advance(task)

        result = self._calculate_metrics(events, bets, execution_time=time.time() - start_time)

        self._logger.info(
            "backtest_complete",
            total_bets=result.total_bets,
            final_bankroll=result.final_bankroll,
            roi=result.roi,
        )

        return result

    async def _get_events_with_results(self) -> list[BacktestEvent]:
        events = await self.reader.get_events_by_date_range(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            status=EventStatus.FINAL,
        )

        backtest_events: list[BacktestEvent] = []
        for event in events:
            bt_event = BacktestEvent.from_db_event(event)
            if bt_event is not None:
                backtest_events.append(bt_event)

        self._logger.info(
            "events_loaded",
            total_events=len(events),
            events_with_results=len(backtest_events),
        )

        return backtest_events

    def _calculate_stake(self, opportunity: BetOpportunity, bankroll: float) -> float:
        if self.config.bet_sizing_method == "flat":
            stake = self.config.flat_stake_amount
        elif self.config.bet_sizing_method == "percentage":
            stake = bankroll * self.config.percentage_stake
        elif self.config.bet_sizing_method == "fractional_kelly":
            stake = calculate_kelly_stake(
                bet_probability=opportunity.confidence,
                american_odds=opportunity.odds,
                bankroll=bankroll,
                kelly_fraction=self.config.kelly_fraction,
                max_stake_percentage=self.config.max_bet_percentage,
            )
        else:
            raise ValueError(f"Unknown bet sizing method: {self.config.bet_sizing_method}")

        if self.config.max_bet_size:
            stake = min(stake, self.config.max_bet_size)

        stake = max(stake, self.config.min_bet_size)

        return round(stake, 2)

    def _evaluate_bet_result_for_opportunity(
        self, opportunity: BetOpportunity, event: BacktestEvent, stake: float
    ) -> tuple[str, float]:
        """Evaluate bet result using opportunity details and event outcome."""
        if opportunity.market == "h2h":
            won = self._evaluate_moneyline_outcome(opportunity.outcome, event)
        elif opportunity.market == "spreads":
            won = self._evaluate_spread_outcome(opportunity.outcome, opportunity.line, event)
        elif opportunity.market == "totals":
            won = self._evaluate_total_outcome(opportunity.outcome, opportunity.line, event)
        else:
            self._logger.warning("unknown_market", market=opportunity.market)
            return ("unknown", 0.0)

        if won is None:
            return ("push", 0.0)

        profit = calculate_profit_from_odds(stake, opportunity.odds, won)

        result_str = "win" if won else "loss"
        return (result_str, profit)

    def _evaluate_bet_result(self, bet: BetRecord, event: BacktestEvent) -> tuple[str, float]:
        """Kept for backward compatibility, delegates to outcome-based evaluation."""
        if bet.market == "h2h":
            won = self._evaluate_moneyline_outcome(bet.outcome, event)
        elif bet.market == "spreads":
            won = self._evaluate_spread_outcome(bet.outcome, bet.line, event)
        elif bet.market == "totals":
            won = self._evaluate_total_outcome(bet.outcome, bet.line, event)
        else:
            self._logger.warning("unknown_market", market=bet.market)
            return ("unknown", 0.0)

        if won is None:
            return ("push", 0.0)

        profit = calculate_profit_from_odds(bet.stake, bet.odds, won)

        result_str = "win" if won else "loss"
        return (result_str, profit)

    def _evaluate_moneyline_outcome(self, outcome: str, event: BacktestEvent) -> bool | None:
        """Evaluate moneyline bet based on outcome name."""
        if event.home_score > event.away_score:
            winner = event.home_team
        elif event.away_score > event.home_score:
            winner = event.away_team
        else:
            return None

        return outcome == winner

    def _evaluate_spread_outcome(
        self, outcome: str, line: float | None, event: BacktestEvent
    ) -> bool | None:
        """Evaluate spread bet based on outcome and line."""
        if line is None:
            return None

        if outcome == event.home_team:
            adjusted_score = event.home_score + line
            won = adjusted_score > event.away_score
            push = adjusted_score == event.away_score
        else:
            away_line = -line
            adjusted_score = event.away_score + away_line
            won = adjusted_score > event.home_score
            push = adjusted_score == event.home_score

        if push:
            return None

        return won

    def _evaluate_total_outcome(
        self, outcome: str, line: float | None, event: BacktestEvent
    ) -> bool | None:
        """Evaluate totals bet based on outcome and line."""
        if line is None:
            return None

        total_points = event.home_score + event.away_score

        if outcome.lower() == "over":
            won = total_points > line
            push = total_points == line
        else:
            won = total_points < line
            push = total_points == line

        if push:
            return None

        return won

    def _calculate_metrics(
        self,
        events: list[BacktestEvent],
        bets: list[BetRecord],
        execution_time: float,
    ) -> BacktestResult:
        if not bets:
            return self._create_empty_result(execution_time)

        total_bets = len(bets)
        winning_bets = sum(1 for bet in bets if bet.result == "win")
        losing_bets = sum(1 for bet in bets if bet.result == "loss")
        push_bets = sum(1 for bet in bets if bet.result == "push")

        total_wagered = sum(bet.stake for bet in bets)
        total_profit = sum(bet.profit for bet in bets if bet.profit is not None)

        final_bankroll = self.config.initial_bankroll + total_profit
        roi = (total_profit / self.config.initial_bankroll) * 100

        win_rate = (
            (winning_bets / (winning_bets + losing_bets) * 100)
            if (winning_bets + losing_bets) > 0
            else 0.0
        )

        average_stake = total_wagered / total_bets if total_bets > 0 else 0.0

        odds_list = [bet.odds for bet in bets]
        average_odds = sum(odds_list) / len(odds_list) if odds_list else 0
        median_odds = sorted(odds_list)[len(odds_list) // 2] if odds_list else 0

        equity_curve_values = [self.config.initial_bankroll]
        for bet in bets:
            if bet.bankroll_after is not None:
                equity_curve_values.append(bet.bankroll_after)

        max_drawdown, max_drawdown_pct = calculate_max_drawdown(equity_curve_values)

        daily_returns = self._calculate_daily_returns(bets)
        sharpe = calculate_sharpe_ratio(daily_returns)
        sortino = calculate_sortino_ratio(daily_returns)

        calmar = abs(roi / max_drawdown_pct) if max_drawdown_pct != 0 else 0.0

        winning_profit = sum(
            bet.profit for bet in bets if bet.result == "win" and bet.profit is not None
        )
        losing_loss = sum(
            abs(bet.profit) for bet in bets if bet.result == "loss" and bet.profit is not None
        )
        profit_factor = calculate_profit_factor(winning_profit, losing_loss)

        longest_win_streak, longest_loss_streak = self._calculate_streaks(bets)

        wins = [bet.profit for bet in bets if bet.result == "win" and bet.profit is not None]
        losses = [bet.profit for bet in bets if bet.result == "loss" and bet.profit is not None]

        average_win = sum(wins) / len(wins) if wins else 0.0
        average_loss = sum(losses) / len(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        market_breakdown = self._calculate_market_breakdown(bets)
        bookmaker_breakdown = self._calculate_bookmaker_breakdown(bets)
        monthly_performance = self._calculate_monthly_performance(bets)

        equity_curve = self._build_equity_curve(bets)

        events_with_complete_data = len([event for event in events if event.home_score is not None])
        data_quality_issues = []
        if events_with_complete_data < len(events):
            data_quality_issues.append(
                f"{len(events) - events_with_complete_data} events missing scores"
            )

        total_days = (self.config.end_date - self.config.start_date).days

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

        risk_metrics = RiskMetrics(
            max_drawdown=max_drawdown,
            max_drawdown_percentage=max_drawdown_pct,
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

        breakdowns = PerformanceBreakdown(
            market_breakdown=market_breakdown,
            bookmaker_breakdown=bookmaker_breakdown,
            monthly_performance=tuple(monthly_performance),
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
            risk_metrics=risk_metrics,
            breakdowns=breakdowns,
            equity_curve=tuple(equity_curve),
            bets=tuple(bets),
            total_events=len(events),
            events_with_complete_data=events_with_complete_data,
            data_quality_issues=tuple(data_quality_issues),
            run_timestamp=datetime.now(UTC),
            execution_time_seconds=execution_time,
        )

    def _calculate_daily_returns(self, bets: list[BetRecord]) -> list[float]:
        if not bets:
            return []

        daily_profits: dict[str, float] = defaultdict(float)
        for bet in bets:
            if bet.profit is not None:
                date_key = bet.event_date.date().isoformat()
                daily_profits[date_key] += bet.profit

        return list(daily_profits.values())

    def _calculate_streaks(self, bets: list[BetRecord]) -> tuple[int, int]:
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
        markets: dict[str, list[BetRecord]] = defaultdict(list)
        for bet in bets:
            markets[bet.market].append(bet)

        breakdown: dict[str, PerformanceStats] = {}
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
        bookmakers: dict[str, list[BetRecord]] = defaultdict(list)
        for bet in bets:
            bookmakers[bet.bookmaker].append(bet)

        breakdown: dict[str, PerformanceStats] = {}
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
        if not bets:
            return []

        months: dict[str, list[BetRecord]] = defaultdict(list)
        for bet in bets:
            month_key = bet.event_date.strftime("%Y-%m")
            months[month_key].append(bet)

        monthly_stats: list[MonthlyStats] = []
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
        if not bets:
            return []

        daily_data: dict[date, tuple[float, int]] = {}
        for bet in bets:
            date_key = bet.event_date.date()
            if date_key not in daily_data:
                daily_data[date_key] = (0.0, 0)

            profit, count = daily_data[date_key]
            if bet.profit is not None:
                daily_data[date_key] = (profit + bet.profit, count + 1)

        equity_curve: list[EquityPoint] = []
        cumulative_profit = 0.0
        cumulative_bets = 0

        for day in sorted(daily_data.keys()):
            daily_profit, daily_bets = daily_data[day]
            cumulative_profit += daily_profit
            cumulative_bets += daily_bets

            equity_curve.append(
                EquityPoint(
                    date=datetime.combine(day, datetime.min.time()),
                    bankroll=self.config.initial_bankroll + cumulative_profit,
                    cumulative_profit=cumulative_profit,
                    bets_to_date=cumulative_bets,
                )
            )

        return equity_curve

    def _create_empty_result(self, execution_time: float) -> BacktestResult:
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

        risk_metrics = RiskMetrics(
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

        breakdowns = PerformanceBreakdown(
            market_breakdown={},
            bookmaker_breakdown={},
            monthly_performance=(),
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
            risk_metrics=risk_metrics,
            breakdowns=breakdowns,
            equity_curve=(),
            bets=(),
            total_events=0,
            events_with_complete_data=0,
            data_quality_issues=("No bets placed",),
            run_timestamp=datetime.now(UTC),
            execution_time_seconds=execution_time,
        )
