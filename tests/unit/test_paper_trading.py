"""Tests for paper trade model and settlement logic."""

from datetime import UTC, datetime

import pytest
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import PaperTrade, TradeResult
from odds_lambda.paper_trading import (
    get_open_trades,
    get_portfolio_summary,
    place_trade,
    settle_trades,
)


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


class TestPlaceTradeMarkets:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("sport_key", "accepted"), [("soccer_epl", True), ("baseball_mlb", False)]
    )
    async def test_h2h_draw_depends_on_the_sport(
        self, test_session, sport_key: str, accepted: bool
    ) -> None:
        test_session.add(
            Event(
                id="evt-draw",
                sport_key=sport_key,
                sport_title=sport_key,
                commence_time=datetime(2026, 4, 15, 15, 0, tzinfo=UTC),
                home_team="Arsenal",
                away_team="Chelsea",
            )
        )
        await test_session.flush()

        placing = place_trade(
            test_session,
            event_id="evt-draw",
            market="h2h",
            selection="draw",
            bookmaker="bet365",
            odds=300,
            stake=50.0,
            bankroll=1000.0,
        )
        if accepted:
            assert (await placing).selection == "draw"
        else:
            with pytest.raises(ValueError, match="has no draw"):
                await placing

    @pytest.mark.asyncio
    async def test_unsupported_market_is_rejected(self, test_session) -> None:
        with pytest.raises(ValueError, match="Unsupported market"):
            await place_trade(
                test_session,
                event_id="evt-001",
                market="totals",
                selection="home",
                bookmaker="bet365",
                odds=-110,
                stake=50.0,
                bankroll=1000.0,
            )


def _event(event_id: str, status: EventStatus, scores: tuple[int, int] | None = None) -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=datetime(2026, 4, 10, 15, 0, tzinfo=UTC),
        home_team="Arsenal",
        away_team="Chelsea",
        status=status,
        home_score=scores[0] if scores else None,
        away_score=scores[1] if scores else None,
    )


def _trade(event_id: str, market: str, selection: str) -> PaperTrade:
    return PaperTrade(
        event_id=event_id,
        market=market,
        selection=selection,
        bookmaker="bet365",
        odds=200,
        stake=100.0,
        bankroll_before=1000.0,
    )


class TestSettleMarkets:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("selection", "result", "pnl"),
        [
            ("draw", TradeResult.WIN, 200.0),
            ("home", TradeResult.LOSS, -100.0),
            ("away", TradeResult.LOSS, -100.0),
        ],
    )
    async def test_three_way_market_grades_a_draw(
        self, test_session, selection: str, result: TradeResult, pnl: float
    ) -> None:
        test_session.add(_event("evt-1x2", EventStatus.FINAL, (1, 1)))
        test_session.add(_trade("evt-1x2", "1x2", selection))
        await test_session.flush()

        [settled] = await settle_trades(test_session)

        assert (settled.result, settled.pnl) == (result, pytest.approx(pnl))

    @pytest.mark.asyncio
    async def test_trade_on_a_cancelled_event_is_voided(self, test_session) -> None:
        test_session.add(_event("evt-off", EventStatus.CANCELLED))
        test_session.add(_trade("evt-off", "1x2", "home"))
        await test_session.flush()

        [settled] = await settle_trades(test_session)

        assert (settled.result, settled.pnl, settled.bankroll_after) == (
            TradeResult.VOID,
            0.0,
            1000.0,
        )
        assert settled.settled_at is not None
        assert await settle_trades(test_session) == []

    @pytest.mark.asyncio
    async def test_trade_on_a_postponed_event_stays_open(self, test_session) -> None:
        test_session.add(_event("evt-late", EventStatus.POSTPONED))
        test_session.add(_trade("evt-late", "1x2", "home"))
        await test_session.flush()

        assert await settle_trades(test_session) == []
        assert len(await get_open_trades(test_session)) == 1

    @pytest.mark.asyncio
    async def test_unsupported_market_on_a_final_event_stays_open(self, test_session) -> None:
        test_session.add(_event("evt-totals", EventStatus.FINAL, (2, 0)))
        test_session.add(_trade("evt-totals", "totals", "home"))
        await test_session.flush()

        assert await settle_trades(test_session) == []
        assert len(await get_open_trades(test_session)) == 1


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
    async def test_home_bet_on_draw_loses(self, test_session) -> None:
        event = Event(
            id="evt-settle-draw-loss",
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
            event_id="evt-settle-draw-loss",
            market="h2h",
            selection="home",
            bookmaker="bet365",
            odds=-150,
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

        record = trade.to_bet_record(event)

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
