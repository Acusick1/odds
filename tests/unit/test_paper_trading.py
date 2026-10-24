"""Tests for paper trade model and settlement logic."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import PaperTrade, TradeResult
from odds_lambda.paper_trading import (
    get_open_trades,
    get_portfolio_summary,
    place_trade,
    settle_trades,
)


def _write_candidate(directory: Path, **fields: object) -> None:
    data = {
        "name": "cand",
        "description": "d",
        "kind": "agent",
        "sport": "soccer_epl",
        "version": 2,
        "issue": 1,
        "start_date": (datetime.now(UTC) - timedelta(days=1)).date().isoformat(),
        "status": "active",
        **fields,
    }
    (directory / f"{data['name']}.yaml").write_text(yaml.safe_dump(data))


class TestPlaceTradeCandidate:
    @pytest.fixture
    async def event(self, test_session) -> Event:
        event = Event(
            id="evt-c",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 15, 15, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
        )
        test_session.add(event)
        await test_session.flush()
        return event

    @pytest.fixture
    def registry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr("odds_lambda.paper_trading.candidates_dir", lambda: tmp_path)
        return tmp_path

    @staticmethod
    async def _place(session, **candidate: object) -> PaperTrade:
        return await place_trade(
            session,
            event_id="evt-c",
            market="1x2",
            selection="home",
            bookmaker="bet365",
            price_decimal=2.2,
            stake=10.0,
            bankroll=1000.0,
            **candidate,  # type: ignore[arg-type]
        )

    @pytest.mark.asyncio
    async def test_valid_candidate_is_recorded(self, test_session, event, registry) -> None:
        _write_candidate(registry)
        trade = await self._place(test_session, candidate_name="cand", candidate_version=2)
        assert (trade.candidate_name, trade.candidate_version) == ("cand", 2)

    @pytest.mark.asyncio
    async def test_no_candidate_is_allowed(self, test_session, event, registry) -> None:
        trade = await self._place(test_session)
        assert trade.candidate_name is None and trade.candidate_version is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fields", "call", "message"),
        [
            ({}, {"candidate_name": "other", "candidate_version": 2}, "not in the registry"),
            (
                {"status": "killed"},
                {"candidate_name": "cand", "candidate_version": 2},
                "not active",
            ),
            (
                {"status": "promoted"},
                {"candidate_name": "cand", "candidate_version": 2},
                "not active",
            ),
            (
                {"start_date": (datetime.now(UTC) + timedelta(days=2)).date().isoformat()},
                {"candidate_name": "cand", "candidate_version": 2},
                "starts on",
            ),
            ({}, {"candidate_name": "cand", "candidate_version": 1}, "version 2, not 1"),
            ({}, {"candidate_name": "cand"}, "together"),
            ({}, {"candidate_version": 2}, "together"),
        ],
    )
    async def test_invalid_candidate_is_rejected(
        self, test_session, event, registry, fields, call, message
    ) -> None:
        _write_candidate(registry, **fields)
        with pytest.raises(ValueError, match=message):
            await self._place(test_session, **call)

    @pytest.mark.asyncio
    async def test_h2h_is_rejected_on_a_three_way_sport(
        self, test_session, event, registry
    ) -> None:
        _write_candidate(registry)
        with pytest.raises(ValueError, match="use '1x2'"):
            await place_trade(
                test_session,
                event_id="evt-c",
                market="h2h",
                selection="home",
                bookmaker="bet365",
                price_decimal=2.2,
                stake=10.0,
                bankroll=1000.0,
                candidate_name="cand",
                candidate_version=2,
            )

    @pytest.mark.asyncio
    async def test_candidate_sport_must_match_the_event(
        self, test_session, event, registry
    ) -> None:
        _write_candidate(registry, sport="baseball_mlb")
        with pytest.raises(ValueError, match="is for baseball_mlb, not soccer_epl"):
            await self._place(test_session, candidate_name="cand", candidate_version=2)

    @pytest.mark.asyncio
    async def test_candidate_trade_on_a_missing_event_is_rejected(
        self, test_session, registry
    ) -> None:
        _write_candidate(registry)
        with pytest.raises(ValueError, match="not found"):
            await self._place(test_session, candidate_name="cand", candidate_version=2)

    @pytest.mark.asyncio
    async def test_git_sha_is_recorded(self, test_session, event, registry, monkeypatch) -> None:
        monkeypatch.setattr("odds_lambda.paper_trading.current_git_sha", lambda: "abc123")
        trade = await self._place(test_session)
        assert trade.git_sha == "abc123"

    @pytest.mark.asyncio
    async def test_price_must_exceed_one(self, test_session, event, registry) -> None:
        with pytest.raises(ValueError, match="exceed 1.0"):
            await place_trade(
                test_session,
                event_id="evt-c",
                market="1x2",
                selection="home",
                bookmaker="bet365",
                price_decimal=1.0,
                stake=10.0,
                bankroll=1000.0,
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
            price_decimal=1.6,
            stake=50.0,
            bankroll=1000.0,
            reasoning="Value on home side",
            confidence=0.72,
        )

        assert trade.id is not None
        assert trade.event_id == "evt-001"
        assert trade.selection == "home"
        assert trade.price_decimal == 1.6
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
                price_decimal=1.6,
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
                price_decimal=1.6,
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
                price_decimal=1.6,
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
            price_decimal=4.0,
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
                price_decimal=1.9,
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
        price_decimal=3.0,
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
            price_decimal=2.5,
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
            price_decimal=2.5,
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
            price_decimal=1.6,
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
            price_decimal=3.5,
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
            price_decimal=1.9,
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
            price_decimal=3.0,
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
            price_decimal=1 + 2 / 3,
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
