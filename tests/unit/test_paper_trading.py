"""Tests for paper trade model and settlement logic."""

from datetime import UTC, datetime

import pytest
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import PaperTrade, TradeResult
from odds_lambda.paper_trading import (
    _american_to_decimal,
    _compute_pnl,
    _determine_result,
    get_open_trades,
    get_portfolio_summary,
    place_trade,
    settle_trades,
    to_bet_record,
)


class TestDetermineResult:
    def test_home_win_home_selection(self) -> None:
        assert _determine_result("home", 2, 1) == TradeResult.WIN

    def test_home_win_away_selection(self) -> None:
        assert _determine_result("away", 2, 1) == TradeResult.LOSS

    def test_draw_draw_selection(self) -> None:
        assert _determine_result("draw", 1, 1) == TradeResult.WIN

    def test_draw_home_selection(self) -> None:
        assert _determine_result("home", 1, 1) == TradeResult.LOSS

    def test_away_win_away_selection(self) -> None:
        assert _determine_result("away", 0, 3) == TradeResult.WIN

    def test_away_win_draw_selection(self) -> None:
        assert _determine_result("draw", 0, 3) == TradeResult.LOSS


class TestComputePnl:
    def test_win_positive_odds(self) -> None:
        # +200 means 2x profit on stake
        pnl = _compute_pnl(200, 10.0, TradeResult.WIN)
        assert pnl == pytest.approx(20.0)

    def test_win_negative_odds(self) -> None:
        # -200 means 0.5x profit on stake
        pnl = _compute_pnl(-200, 10.0, TradeResult.WIN)
        assert pnl == pytest.approx(5.0)

    def test_loss(self) -> None:
        pnl = _compute_pnl(150, 10.0, TradeResult.LOSS)
        assert pnl == -10.0

    def test_push(self) -> None:
        pnl = _compute_pnl(150, 10.0, TradeResult.PUSH)
        assert pnl == 0.0

    def test_void(self) -> None:
        pnl = _compute_pnl(150, 10.0, TradeResult.VOID)
        assert pnl == 0.0


class TestAmericanToDecimal:
    def test_positive_odds(self) -> None:
        assert _american_to_decimal(200) == pytest.approx(3.0)

    def test_negative_odds(self) -> None:
        assert _american_to_decimal(-200) == pytest.approx(1.5)

    def test_even_money(self) -> None:
        assert _american_to_decimal(100) == pytest.approx(2.0)

    def test_heavy_favorite(self) -> None:
        assert _american_to_decimal(-500) == pytest.approx(1.2)


class TestPlaceTrade:
    @pytest.mark.asyncio
    async def test_place_valid_trade(self, test_session) -> None:
        event = Event(
            id="evt-001",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 15, 15, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
        )
        test_session.add(event)
        await test_session.flush()

        trade = await place_trade(
            test_session,
            event_id="evt-001",
            market="h2h",
            selection="home",
            bookmaker="bet365",
            odds=-150,
            stake=50.0,
            bankroll=1000.0,
            reasoning="Value on home side",
            confidence=0.72,
        )

        assert trade.id is not None
        assert trade.event_id == "evt-001"
        assert trade.selection == "home"
        assert trade.odds == -150
        assert trade.stake == 50.0
        assert trade.bankroll_before == 1000.0
        assert trade.settled_at is None

    @pytest.mark.asyncio
    async def test_stake_exceeds_bankroll(self, test_session) -> None:
        with pytest.raises(ValueError, match="exceeds available bankroll"):
            await place_trade(
                test_session,
                event_id="evt-001",
                market="h2h",
                selection="home",
                bookmaker="bet365",
                odds=-150,
                stake=2000.0,
                bankroll=1000.0,
            )

    @pytest.mark.asyncio
    async def test_invalid_selection(self, test_session) -> None:
        with pytest.raises(ValueError, match="Invalid selection"):
            await place_trade(
                test_session,
                event_id="evt-001",
                market="h2h",
                selection="over",
                bookmaker="bet365",
                odds=-150,
                stake=50.0,
                bankroll=1000.0,
            )

    @pytest.mark.asyncio
    async def test_zero_stake(self, test_session) -> None:
        with pytest.raises(ValueError, match="Stake must be positive"):
            await place_trade(
                test_session,
                event_id="evt-001",
                market="h2h",
                selection="home",
                bookmaker="bet365",
                odds=-150,
                stake=0,
                bankroll=1000.0,
            )


class TestSettleTrades:
    @pytest.mark.asyncio
    async def test_settle_winning_trade(self, test_session) -> None:
        event = Event(
            id="evt-settle-1",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 10, 15, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
            status=EventStatus.FINAL,
            home_score=2,
            away_score=0,
        )
        test_session.add(event)

        trade = PaperTrade(
            event_id="evt-settle-1",
            market="h2h",
            selection="home",
            bookmaker="bet365",
            odds=150,
            stake=100.0,
            bankroll_before=1000.0,
        )
        test_session.add(trade)
        await test_session.flush()

        settled = await settle_trades(test_session)

        assert len(settled) == 1
        assert settled[0].result == TradeResult.WIN
        assert settled[0].pnl == pytest.approx(150.0)
        assert settled[0].bankroll_after == pytest.approx(1150.0)
        assert settled[0].settled_at is not None

    @pytest.mark.asyncio
    async def test_settle_losing_trade(self, test_session) -> None:
        event = Event(
            id="evt-settle-2",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 10, 15, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
            status=EventStatus.FINAL,
            home_score=0,
            away_score=1,
        )
        test_session.add(event)

        trade = PaperTrade(
            event_id="evt-settle-2",
            market="h2h",
            selection="home",
            bookmaker="bet365",
            odds=150,
            stake=100.0,
            bankroll_before=1000.0,
        )
        test_session.add(trade)
        await test_session.flush()

        settled = await settle_trades(test_session)

        assert len(settled) == 1
        assert settled[0].result == TradeResult.LOSS
        assert settled[0].pnl == -100.0

    @pytest.mark.asyncio
    async def test_idempotent_settlement(self, test_session) -> None:
        event = Event(
            id="evt-settle-3",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 10, 15, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
            status=EventStatus.FINAL,
            home_score=1,
            away_score=1,
        )
        test_session.add(event)

        trade = PaperTrade(
            event_id="evt-settle-3",
            market="h2h",
            selection="draw",
            bookmaker="bet365",
            odds=250,
            stake=50.0,
            bankroll_before=1000.0,
        )
        test_session.add(trade)
        await test_session.flush()

        first = await settle_trades(test_session)
        assert len(first) == 1

        second = await settle_trades(test_session)
        assert len(second) == 0

    @pytest.mark.asyncio
    async def test_skip_scheduled_events(self, test_session) -> None:
        event = Event(
            id="evt-settle-4",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 20, 15, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
            status=EventStatus.SCHEDULED,
        )
        test_session.add(event)

        trade = PaperTrade(
            event_id="evt-settle-4",
            market="h2h",
            selection="home",
            bookmaker="bet365",
            odds=-110,
            stake=50.0,
            bankroll_before=1000.0,
        )
        test_session.add(trade)
        await test_session.flush()

        settled = await settle_trades(test_session)
        assert len(settled) == 0


class TestPortfolioQueries:
    @pytest.mark.asyncio
    async def test_empty_portfolio(self, test_session) -> None:
        summary = await get_portfolio_summary(test_session, initial_bankroll=1000.0)

        assert summary.current_bankroll == 1000.0
        assert summary.total_trades == 0
        assert summary.total_pnl == 0.0

    @pytest.mark.asyncio
    async def test_open_trades_query(self, test_session) -> None:
        event = Event(
            id="evt-port-1",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 20, 15, 0, tzinfo=UTC),
            home_team="Liverpool",
            away_team="Man City",
        )
        test_session.add(event)

        trade = PaperTrade(
            event_id="evt-port-1",
            market="h2h",
            selection="away",
            bookmaker="betway",
            odds=200,
            stake=25.0,
            bankroll_before=975.0,
        )
        test_session.add(trade)
        await test_session.flush()

        open_trades = await get_open_trades(test_session)
        assert len(open_trades) == 1
        assert open_trades[0].selection == "away"


class TestToBetRecord:
    def test_conversion(self) -> None:
        event = Event(
            id="evt-conv-1",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 10, 15, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
            home_score=2,
            away_score=1,
        )
        trade = PaperTrade(
            id=42,
            event_id="evt-conv-1",
            market="h2h",
            selection="home",
            bookmaker="bet365",
            odds=-150,
            stake=100.0,
            bankroll_before=1000.0,
            bankroll_after=1066.67,
            result=TradeResult.WIN,
            pnl=66.67,
            reasoning="Value bet",
            confidence=0.8,
            placed_at=datetime(2026, 4, 10, 12, 0, tzinfo=UTC),
            settled_at=datetime(2026, 4, 10, 17, 0, tzinfo=UTC),
        )

        record = to_bet_record(trade, event)

        assert record.bet_id == 42
        assert record.event_id == "evt-conv-1"
        assert record.market == "h2h"
        assert record.outcome == "home"
        assert record.odds == -150
        assert record.stake == 100.0
        assert record.result == "win"
        assert record.profit == 66.67
        assert record.bankroll_before == 1000.0
        assert record.bankroll_after == 1066.67
        assert record.home_score == 2
        assert record.away_score == 1
        assert record.strategy_confidence == 0.8
        assert record.bet_rationale == "Value bet"
