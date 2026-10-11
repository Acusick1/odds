"""Tests for paper trade model and settlement logic."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot
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


KICKOFF = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)
PLACED_SNAPSHOT_TIME = KICKOFF - timedelta(hours=3)
CLOSING_SNAPSHOT_TIME = KICKOFF - timedelta(minutes=5)


def _book(key: str, home: int, draw: int, away: int, market: str = "h2h") -> dict:
    return {
        "key": key,
        "title": key,
        "last_update": "2026-04-15T11:00:00Z",
        "markets": [
            {
                "key": market,
                "outcomes": [
                    {"name": "Arsenal", "price": home},
                    {"name": "Draw", "price": draw},
                    {"name": "Chelsea", "price": away},
                ],
            }
        ],
    }


def _snapshot(when: datetime, *books: dict) -> OddsSnapshot:
    return OddsSnapshot(
        event_id="evt-sharp",
        snapshot_time=when,
        raw_data={"bookmakers": list(books)},
        bookmaker_count=len(books),
    )


async def _add_event(session, *, final: bool = False) -> None:
    session.add(
        Event(
            id="evt-sharp",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=KICKOFF,
            home_team="Arsenal",
            away_team="Chelsea",
            status=EventStatus.FINAL if final else EventStatus.SCHEDULED,
            home_score=2 if final else None,
            away_score=0 if final else None,
        )
    )
    await session.flush()


async def _place_home_bet(session) -> PaperTrade:
    return await place_trade(
        session,
        event_id="evt-sharp",
        market="h2h",
        selection="home",
        bookmaker="bet365",
        odds=105,
        stake=50.0,
        bankroll=1000.0,
    )


class TestSharpContextAtPlacement:
    @pytest.mark.asyncio
    async def test_primary_sharp_book_is_snapshotted(self, test_session) -> None:
        await _add_event(test_session)
        test_session.add(
            _snapshot(
                PLACED_SNAPSHOT_TIME,
                _book("pinnacle", -110, 300, 300),
                _book("betfair_exchange", 100, 250, 250),
            )
        )
        await test_session.flush()

        trade = await _place_home_bet(test_session)

        # Pinnacle -110/+300/+300: 0.52381 / 1.02381 devigged
        assert trade.sharp_prob_at_bet == pytest.approx(0.51163, abs=1e-5)
        assert trade.sharp_bookmaker_at_bet == "pinnacle"
        assert trade.sharp_snapshot_time_at_bet == PLACED_SNAPSHOT_TIME
        # Bet +105 implies 0.487805; spread = bet - sharp
        assert trade.sharp_spread_at_bet == pytest.approx(-0.02382, abs=1e-5)

    @pytest.mark.asyncio
    async def test_fallback_book_used_when_primary_absent(self, test_session) -> None:
        await _add_event(test_session)
        test_session.add(_snapshot(PLACED_SNAPSHOT_TIME, _book("betfair_exchange", 100, 300, 300)))
        await test_session.flush()

        trade = await _place_home_bet(test_session)

        assert trade.sharp_bookmaker_at_bet == "betfair_exchange"
        assert trade.sharp_prob_at_bet == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_no_sharp_data_leaves_fields_null_and_still_places(self, test_session) -> None:
        await _add_event(test_session)
        test_session.add(_snapshot(PLACED_SNAPSHOT_TIME, _book("bet365", 105, 280, 290)))
        await test_session.flush()

        trade = await _place_home_bet(test_session)

        assert trade.id is not None
        assert trade.sharp_prob_at_bet is None
        assert trade.sharp_bookmaker_at_bet is None
        assert trade.sharp_snapshot_time_at_bet is None
        assert trade.sharp_spread_at_bet is None

    @pytest.mark.asyncio
    async def test_no_snapshots_at_all_still_places(self, test_session) -> None:
        await _add_event(test_session)

        trade = await _place_home_bet(test_session)

        assert trade.id is not None
        assert trade.sharp_prob_at_bet is None

    @pytest.mark.asyncio
    async def test_incomplete_sharp_market_is_not_devigged(self, test_session) -> None:
        await _add_event(test_session)
        book = _book("pinnacle", -110, 300, 300)
        book["markets"][0]["outcomes"].pop()  # no Chelsea price
        test_session.add(_snapshot(PLACED_SNAPSHOT_TIME, book))
        await test_session.flush()

        trade = await _place_home_bet(test_session)

        assert trade.sharp_prob_at_bet is None


class TestSharpMarketKeyAndConsistency:
    @pytest.mark.asyncio
    async def test_betfair_1x2_snapshots_resolve_for_h2h_bet(self, test_session) -> None:
        await _add_event(test_session)
        test_session.add(
            _snapshot(PLACED_SNAPSHOT_TIME, _book("betfair_exchange", 100, 300, 300, market="1x2"))
        )
        await test_session.flush()

        trade = await _place_home_bet(test_session)

        assert trade.sharp_bookmaker_at_bet == "betfair_exchange"
        assert trade.sharp_prob_at_bet == pytest.approx(0.5)
        assert trade.sharp_snapshot_time_at_bet == PLACED_SNAPSHOT_TIME

    @pytest.mark.asyncio
    async def test_partial_h2h_does_not_hide_complete_1x2(self, test_session) -> None:
        await _add_event(test_session)
        partial = _book("pinnacle", -110, 300, 300)
        partial["markets"][0]["outcomes"] = partial["markets"][0]["outcomes"][:1]
        test_session.add(_snapshot(PLACED_SNAPSHOT_TIME, partial))
        test_session.add(
            _snapshot(
                PLACED_SNAPSHOT_TIME - timedelta(minutes=10),
                _book("betfair_exchange", 100, 300, 300, market="1x2"),
            )
        )
        await test_session.flush()

        trade = await _place_home_bet(test_session)

        assert trade.sharp_bookmaker_at_bet == "betfair_exchange"

    @pytest.mark.asyncio
    async def test_outcomes_from_different_books_leave_fields_null(self, test_session) -> None:
        await _add_event(test_session)
        pinnacle_home_only = _book("pinnacle", -110, 300, 300)
        pinnacle_home_only["markets"][0]["outcomes"] = pinnacle_home_only["markets"][0]["outcomes"][
            :1
        ]
        test_session.add(
            _snapshot(
                PLACED_SNAPSHOT_TIME,
                pinnacle_home_only,
                _book("betfair_exchange", 100, 300, 300),
            )
        )
        await test_session.flush()

        trade = await _place_home_bet(test_session)

        # Pinnacle quotes only home and Betfair the rest: no single-book market, so null
        assert trade.sharp_prob_at_bet is None
        assert trade.sharp_bookmaker_at_bet is None


class TestSharpContextAtSettlement:
    @pytest.mark.asyncio
    async def test_closing_price_and_clv_filled(self, test_session) -> None:
        await _add_event(test_session, final=True)
        test_session.add(_snapshot(PLACED_SNAPSHOT_TIME, _book("pinnacle", -110, 300, 300)))
        await test_session.flush()
        await _place_home_bet(test_session)
        test_session.add(_snapshot(CLOSING_SNAPSHOT_TIME, _book("pinnacle", -120, 300, 300)))
        # Post-kickoff snapshot must be ignored
        test_session.add(
            _snapshot(KICKOFF + timedelta(minutes=30), _book("pinnacle", -400, 600, 900))
        )
        await test_session.flush()

        (settled,) = await settle_trades(test_session)

        # Pinnacle -120/+300/+300: 0.54545 / 1.04545 devigged
        assert settled.sharp_prob_at_close == pytest.approx(0.52174, abs=1e-5)
        assert settled.sharp_bookmaker_at_close == "pinnacle"
        assert settled.sharp_snapshot_time_at_close == CLOSING_SNAPSHOT_TIME
        # CLV = close prob - bet implied prob (0.487805); positive = beat the close
        assert settled.clv == pytest.approx(0.03393, abs=1e-5)

    @pytest.mark.asyncio
    async def test_fallback_book_at_close(self, test_session) -> None:
        await _add_event(test_session, final=True)
        await _place_home_bet(test_session)
        test_session.add(
            _snapshot(CLOSING_SNAPSHOT_TIME, _book("betfair_exchange", -120, 300, 300))
        )
        await test_session.flush()

        (settled,) = await settle_trades(test_session)

        assert settled.sharp_bookmaker_at_close == "betfair_exchange"
        assert settled.clv == pytest.approx(0.03393, abs=1e-5)

    @pytest.mark.asyncio
    async def test_no_sharp_data_still_settles(self, test_session) -> None:
        await _add_event(test_session, final=True)
        await _place_home_bet(test_session)

        (settled,) = await settle_trades(test_session)

        assert settled.result == TradeResult.WIN
        assert settled.sharp_prob_at_close is None
        assert settled.sharp_bookmaker_at_close is None
        assert settled.sharp_snapshot_time_at_close is None
        assert settled.clv is None
