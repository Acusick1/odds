"""The substitute-close comparison over stored snapshots, on the test database."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_shaped_snapshot
from sqlalchemy.ext.asyncio import AsyncSession

from scripts import compare_substitute_close as cmp
from tests.test_helpers import betfair_exchange_bookmaker, make_epl_event, store_tagged
from tests.test_helpers import make_tagged_snapshot as tagged

FDUK_PRICES = (2.0, 3.6, 4.2)
API_PRICES = (2.1, 3.5, 4.0)
START, END = date(2025, 3, 1), date(2025, 3, 31)


def kickoff(day: int) -> datetime:
    return datetime(2025, 3, day, 15, 0, tzinfo=UTC)


async def add_event(
    session: AsyncSession, event_id: int, when: datetime, status: EventStatus = EventStatus.FINAL
) -> Event:
    event = make_epl_event(event_id, when, f"Home {event_id}", f"Away {event_id}", status=status)
    session.add(event)
    await session.flush()
    return event


async def add_fduk(session: AsyncSession, event: Event) -> None:
    book = betfair_exchange_bookmaker(
        FDUK_PRICES, home_team=event.home_team, away_team=event.away_team
    )
    await store_tagged(
        session, tagged(event.id, event.commence_time, [book], "football_data_uk", "kickoff")
    )


async def add_oddsapi(
    session: AsyncSession,
    event: Event,
    minutes_before: float,
    prices: tuple[float, float, float] = API_PRICES,
    time_basis: TimeBasis = TimeBasis.SOURCE_STAMPED,
) -> None:
    book = betfair_exchange_bookmaker(prices, home_team=event.home_team, away_team=event.away_team)
    await store_shaped_snapshot(
        session,
        event,
        {"bookmakers": [book]},
        event.commence_time - timedelta(minutes=minutes_before),
        SnapshotShape(Source.ODDS_API, time_basis, None),
    )


async def load(
    session: AsyncSession, max_age: timedelta | None = None
) -> dict[int, cmp.EventComparison]:
    found = await cmp.load_comparisons(session, "soccer_epl", START, END, max_age)
    return {c.event_id: c for c in found}


class TestLoadComparisons:
    @pytest.fixture
    async def season(self, pglite_async_session: AsyncSession) -> AsyncSession:
        session = pglite_async_session
        compared = await add_event(session, 1, kickoff(1))
        await add_fduk(session, compared)
        await add_oddsapi(session, compared, 40, (3.0, 3.0, 3.0))
        await add_oddsapi(session, compared, 4)
        # A live fetch is not a historical snapshot, and a stamp after kick-off is not known yet.
        await add_oddsapi(session, compared, 1, (9.0, 9.0, 9.0), TimeBasis.OBSERVED)
        await add_oddsapi(session, compared, -5, (9.0, 9.0, 9.0))

        no_snapshot = await add_event(session, 2, kickoff(2))
        await add_fduk(session, no_snapshot)

        wide = await add_event(session, 3, kickoff(3))
        await add_fduk(session, wide)
        await add_oddsapi(session, wide, 3, (2.0, 3.0, 2.0))

        no_close = await add_event(session, 4, kickoff(4))
        await add_oddsapi(session, no_close, 3)

        outside = await add_event(session, 5, datetime(2025, 4, 1, 15, 0, tzinfo=UTC))
        await add_fduk(session, outside)
        await add_oddsapi(session, outside, 3)

        postponed = await add_event(session, 6, kickoff(6), EventStatus.POSTPONED)
        await add_fduk(session, postponed)
        await add_oddsapi(session, postponed, 3)
        await session.commit()
        return session

    async def test_each_final_event_in_range_is_compared_or_says_what_it_lacks(
        self, season: AsyncSession
    ) -> None:
        statuses = {i: c.status for i, c in (await load(season)).items()}
        assert statuses == {
            1: cmp.Status.COMPARED,
            2: cmp.Status.NO_SNAPSHOT,
            3: cmp.Status.SNAPSHOT_UNUSABLE,
            4: cmp.Status.NO_FDUK_CLOSE,
        }

    async def test_the_price_is_the_latest_historical_snapshot_at_or_before_kickoff(
        self, season: AsyncSession
    ) -> None:
        found = (await load(season))[1]
        assert found.age_minutes == pytest.approx(4)
        (row_home, *_), implied = found.oddsapi.fair.values(), [1 / p for p in API_PRICES]  # type: ignore[union-attr]
        assert row_home == pytest.approx(implied[0] / sum(implied))
        deltas = cmp.deltas([found])
        assert [d.outcome.value for d in deltas] == ["home", "draw", "away"]
        assert sum(d.dp_pp for d in deltas) == pytest.approx(0)

    async def test_max_age_leaves_out_older_snapshots(self, season: AsyncSession) -> None:
        loaded = await load(season, timedelta(minutes=3))
        assert loaded[1].status is cmp.Status.NO_SNAPSHOT
        assert (await load(season, timedelta(minutes=4)))[1].status is cmp.Status.COMPARED

    async def test_the_report_counts_matches_with_an_fduk_close(self, season: AsyncSession) -> None:
        found = await cmp.load_comparisons(season, "soccer_epl", START, END, None)
        report = cmp.build_report(found, {})
        assert report["matches"]["with_fduk_close"] == 3  # type: ignore[index]
        assert report["within_minutes_of_kickoff"]["5"] == {"matches": 1, "share": 1 / 3}  # type: ignore[index]
