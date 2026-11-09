"""Close classification over stored snapshots: ``close_quotes`` then ``close_status``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.close import CloseKind, close_quotes, close_status
from odds_core.models import Event
from odds_core.vocabulary import Source
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_nba_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
    totals_book,
)

KICKOFF = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)
AFTER_GRACE = KICKOFF + timedelta(days=8)
PRICES = (2.2, 3.5, 3.6)


def nba_betfair(prices: tuple[float, float]) -> dict[str, Any]:
    runners = [
        {"runner_name": name, "best_back": price}
        for name, price in zip(("Boston Celtics", "New York Knicks"), prices, strict=True)
    ]
    return {
        "key": "betfair_exchange",
        "markets": [{"key": "h2h", "outcomes": []}],
        "betfair_meta": {"inplay": False, "market_status": "OPEN", "runners": runners},
    }


def nba_polymarket(home: float, away: float) -> dict[str, Any]:
    return {
        "key": "polymarket",
        "markets": [{"key": "h2h", "outcomes": []}],
        "polymarket_meta": {
            "yes_probabilities": {"home": home, "away": away},
            "best_ask": {"home": home + 0.01, "away": away + 0.01},
            "best_bid": {"home": home - 0.01, "away": away - 0.01},
        },
    }


async def status_of(
    session: AsyncSession, event: Event, market: str, now: datetime, line: float | None = None
) -> tuple[CloseKind, Source | None]:
    quoted = await close_quotes(session, [event], market)
    assert event.id is not None
    status = close_status(quoted[event.id], event.commence_time, event.sport_key, market, now, line)
    return status.kind, status.source


async def add(session: AsyncSession, event: Event) -> Event:
    session.add(event)
    await session.flush()
    return event


class TestCloseStatus:
    async def test_a_live_betfair_price_in_the_window_is_live(
        self, pglite_async_session: AsyncSession
    ) -> None:
        event = await add(pglite_async_session, make_epl_event(1, KICKOFF))
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                1,
                KICKOFF - timedelta(minutes=5),
                [betfair_exchange_bookmaker(PRICES)],
                "betfair_api",
            ),
        )
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                1,
                KICKOFF,
                [three_way_bookmaker("betfair_exchange", PRICES)],
                "football_data_uk",
                "kickoff",
            ),
        )

        assert await status_of(pglite_async_session, event, "1x2", AFTER_GRACE) == (
            CloseKind.LIVE,
            Source.BETFAIR,
        )

    async def test_without_a_live_price_the_fduk_close_is_the_fallback(
        self, pglite_async_session: AsyncSession
    ) -> None:
        event = await add(pglite_async_session, make_epl_event(1, KICKOFF))
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                1, KICKOFF - timedelta(hours=2), [betfair_exchange_bookmaker(PRICES)], "betfair_api"
            ),
        )
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                1,
                KICKOFF,
                [three_way_bookmaker("betfair_exchange", PRICES)],
                "football_data_uk",
                "kickoff",
            ),
        )

        assert await status_of(pglite_async_session, event, "1x2", AFTER_GRACE) == (
            CloseKind.FALLBACK,
            Source.FOOTBALL_DATA_UK,
        )

    async def test_without_a_live_price_the_polymarket_mid_is_the_nba_fallback(
        self, pglite_async_session: AsyncSession
    ) -> None:
        event = await add(pglite_async_session, make_nba_event(2, KICKOFF))
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                2, KICKOFF - timedelta(minutes=45), [nba_betfair((1.8, 2.1))], "betfair_api"
            ),
        )
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                2, KICKOFF - timedelta(minutes=2), [nba_polymarket(0.55, 0.46)], "polymarket"
            ),
        )

        assert await status_of(pglite_async_session, event, "h2h", KICKOFF) == (
            CloseKind.FALLBACK,
            Source.POLYMARKET,
        )

    @pytest.mark.parametrize(
        ("now", "kind"),
        [
            (KICKOFF - timedelta(hours=1), CloseKind.PENDING),
            (KICKOFF + timedelta(days=6), CloseKind.PENDING),
            (AFTER_GRACE, CloseKind.NONE),
        ],
    )
    async def test_with_no_close_it_is_pending_within_the_grace_and_none_after(
        self, pglite_async_session: AsyncSession, now: datetime, kind: CloseKind
    ) -> None:
        event = await add(pglite_async_session, make_epl_event(1, KICKOFF))

        assert await status_of(pglite_async_session, event, "1x2", now) == (kind, None)

    async def test_nba_without_a_close_has_no_grace(
        self, pglite_async_session: AsyncSession
    ) -> None:
        event = await add(pglite_async_session, make_nba_event(2, KICKOFF))

        assert await status_of(pglite_async_session, event, "h2h", KICKOFF) == (
            CloseKind.NONE,
            None,
        )

    async def test_a_market_the_sport_has_no_close_for_is_unsupported(
        self, pglite_async_session: AsyncSession
    ) -> None:
        event = await add(
            pglite_async_session, make_nba_event(3, KICKOFF, sport_key="baseball_mlb")
        )

        assert await status_of(pglite_async_session, event, "h2h", AFTER_GRACE) == (
            CloseKind.UNSUPPORTED,
            None,
        )

    async def test_totals_are_classified_per_line(self, pglite_async_session: AsyncSession) -> None:
        event = await add(pglite_async_session, make_epl_event(1, KICKOFF))
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                1,
                KICKOFF - timedelta(minutes=5),
                [totals_book("betfair_exchange", {2.5: (1.9, 2.0)}, betfair_runners=True)],
                "betfair_api",
            ),
        )

        assert await status_of(pglite_async_session, event, "totals", AFTER_GRACE, 2.5) == (
            CloseKind.LIVE,
            Source.BETFAIR,
        )
        assert await status_of(pglite_async_session, event, "totals", AFTER_GRACE, 3.5) == (
            CloseKind.NONE,
            None,
        )
