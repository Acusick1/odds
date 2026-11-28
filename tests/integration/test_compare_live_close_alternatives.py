"""The live-close alternatives comparison over stored snapshots, on the test database."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from odds_core.models import Event, EventStatus
from odds_core.vocabulary import Market, Source
from sqlalchemy.ext.asyncio import AsyncSession

from scripts import compare_live_close_alternatives as cmp
from tests.test_helpers import betfair_exchange_bookmaker, make_epl_event, store_tagged, totals_book
from tests.test_helpers import make_tagged_snapshot as tagged

LIVE_PRICES = (2.0, 3.6, 4.2)
ALT_PRICES = (2.1, 3.5, 4.0)
START, END = date(2026, 11, 1), date(2026, 11, 30)


def kickoff(day: int) -> datetime:
    return datetime(2026, 11, day, 15, 0, tzinfo=UTC)


async def add_event(
    session: AsyncSession, event_id: int, when: datetime, status: EventStatus = EventStatus.FINAL
) -> Event:
    event = make_epl_event(event_id, when, f"Home {event_id}", f"Away {event_id}", status=status)
    session.add(event)
    await session.flush()
    return event


def three_way(event: Event, prices: tuple[float, float, float]) -> dict[str, Any]:
    return betfair_exchange_bookmaker(prices, home_team=event.home_team, away_team=event.away_team)


async def add(
    session: AsyncSession,
    event: Event,
    source: str | None,
    minutes_before: float,
    books: list[dict[str, Any]],
) -> None:
    when = event.commence_time - timedelta(minutes=minutes_before)
    await store_tagged(session, tagged(event.id, when, books, source))


async def load(
    session: AsyncSession, max_age: timedelta | None = None
) -> dict[cmp.Pairing, dict[int, cmp.EventComparison]]:
    found = await cmp.load_comparisons(session, START, END, max_age)
    return {p: {c.event_id: c for c in comparisons} for p, comparisons in found.items()}


ODDSPORTAL_1X2 = cmp.Pairing(Source.ODDSPORTAL, Market.ONE_X_TWO, None)
ODDSPORTAL_TOTALS = cmp.Pairing(Source.ODDSPORTAL, Market.TOTALS, cmp.TOTALS_LINE)
ODDS_API_1X2 = cmp.Pairing(Source.ODDS_API, Market.ONE_X_TWO, None)


class TestLoadComparisons:
    @pytest.fixture
    async def season(self, pglite_async_session: AsyncSession) -> AsyncSession:
        session = pglite_async_session
        both = await add_event(session, 1, kickoff(1))
        await add(
            session,
            both,
            "betfair_api",
            5,
            [
                three_way(both, LIVE_PRICES),
                totals_book("betfair_exchange", {2.5: (1.9, 2.1)}, betfair_runners=True),
            ],
        )
        await add(
            session,
            both,
            "oddsportal_live",
            3,
            [three_way(both, ALT_PRICES), totals_book("betfair_exchange", {2.5: (1.8, 2.2)})],
        )
        await add(session, both, "oddsportal_live", -2, [three_way(both, (9.0, 9.0, 9.0))])
        await add(session, both, None, 25, [three_way(both, ALT_PRICES)])

        no_alternative = await add_event(session, 2, kickoff(2))
        await add(
            session, no_alternative, "betfair_api", 5, [three_way(no_alternative, LIVE_PRICES)]
        )

        # The Betfair job's last price is outside the live close window.
        no_live = await add_event(session, 3, kickoff(3))
        await add(session, no_live, "betfair_api", 45, [three_way(no_live, LIVE_PRICES)])
        await add(session, no_live, "oddsportal_live", 3, [three_way(no_live, ALT_PRICES)])

        postponed = await add_event(session, 4, kickoff(4), EventStatus.POSTPONED)
        await add(session, postponed, "betfair_api", 5, [three_way(postponed, LIVE_PRICES)])
        await session.commit()
        return session

    async def test_each_final_event_is_compared_per_pairing_or_says_what_it_lacks(
        self, season: AsyncSession
    ) -> None:
        statuses = {p: {i: c.status for i, c in f.items()} for p, f in (await load(season)).items()}
        assert statuses == {
            ODDSPORTAL_1X2: {
                1: cmp.Status.COMPARED,
                2: cmp.Status.NO_SNAPSHOT,
                3: cmp.Status.NO_LIVE_CLOSE,
            },
            ODDSPORTAL_TOTALS: {
                1: cmp.Status.COMPARED,
                2: cmp.Status.NO_LIVE_CLOSE,
                3: cmp.Status.NO_LIVE_CLOSE,
            },
            ODDS_API_1X2: {
                1: cmp.Status.COMPARED,
                2: cmp.Status.NO_SNAPSHOT,
                3: cmp.Status.NO_LIVE_CLOSE,
            },
        }

    async def test_each_price_is_its_sources_latest_at_or_before_kickoff(
        self, season: AsyncSession
    ) -> None:
        loaded = await load(season)
        assert loaded[ODDSPORTAL_1X2][1].age_minutes == pytest.approx(3)
        assert loaded[ODDS_API_1X2][1].age_minutes == pytest.approx(25)
        totals = loaded[ODDSPORTAL_TOTALS][1]
        assert totals.alternative is not None and totals.live is not None
        implied = [1 / 1.8, 1 / 2.2]
        assert list(totals.alternative.fair.values()) == pytest.approx(
            [p / sum(implied) for p in implied]
        )

    async def test_max_age_leaves_out_older_snapshots(self, season: AsyncSession) -> None:
        loaded = await load(season, timedelta(minutes=10))
        assert loaded[ODDS_API_1X2][1].status is cmp.Status.NO_SNAPSHOT
        assert loaded[ODDSPORTAL_1X2][1].status is cmp.Status.COMPARED
