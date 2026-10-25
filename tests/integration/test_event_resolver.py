"""Integration tests for the locked event resolver."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import DataQualityLog, Event, EventExternalId
from odds_core.vocabulary import Source
from odds_lambda.mlb_game_resolver import ResolutionRule, ResolvedGame
from odds_lambda.storage.event_resolver import (
    EventIdTaken,
    ListedGame,
    Resolution,
    event_lock_key,
    resolve_event,
)
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import TWIN_KICKOFF

MLB = "baseball_mlb"
YANKEES, RED_SOX = "New York Yankees", "Boston Red Sox"
GAME_1_START = (datetime.now(UTC) + timedelta(days=1)).replace(
    hour=17, minute=5, second=0, microsecond=0
)
GAME_2_START = GAME_1_START.replace(hour=23, minute=10)


def epl_listing(
    source: Source = Source.ODDSPORTAL,
    external_id: str | None = "op-1",
    commence_time: datetime = TWIN_KICKOFF,
    legacy_id: str = "op_live_ARS_CHE",
) -> ListedGame:
    return ListedGame(
        source=source,
        sport_key="soccer_epl",
        sport_title="EPL",
        home_team="Arsenal",
        away_team="Chelsea",
        commence_time=commence_time,
        legacy_id=legacy_id,
        external_id=external_id,
    )


def mlb_listing(
    game_pk: int,
    start: datetime,
    external_id: str | None = None,
    sibling_starts: tuple[datetime, ...] = (),
    legacy_id: str = "op_live_NYY_BOS",
) -> ListedGame:
    return ListedGame(
        source=Source.BETFAIR,
        sport_key=MLB,
        sport_title="MLB",
        home_team=YANKEES,
        away_team=RED_SOX,
        commence_time=start,
        legacy_id=legacy_id,
        external_id=external_id,
        mlb_game=ResolvedGame(game_pk, ResolutionRule.GAME_NUMBER, start, sibling_starts),
    )


async def external_ids(session: AsyncSession, event: Event) -> dict[Source, str]:
    rows = await session.execute(
        select(EventExternalId).where(EventExternalId.event_id == event.surrogate_id)
    )
    return {row.source: row.external_id for row in rows.scalars()}


async def refusals(session: AsyncSession) -> list[DataQualityLog]:
    await session.flush()
    rows = await session.execute(
        select(DataQualityLog).where(DataQualityLog.issue_type == "external_id_refused")
    )
    return list(rows.scalars())


class TestResolveEvent:
    @pytest.mark.asyncio
    async def test_matches_within_two_hours_and_records_the_source_id(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        listing = epl_listing(commence_time=TWIN_KICKOFF + timedelta(minutes=90))

        resolved = await resolve_event(pglite_async_session, listing)

        assert resolved.resolution is Resolution.MATCHED
        assert resolved.event.id == stored_epl_event.id
        assert await external_ids(pglite_async_session, stored_epl_event) == {
            Source.ODDSPORTAL: "op-1"
        }

    @pytest.mark.asyncio
    async def test_looks_up_a_recorded_id_before_matching(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        await resolve_event(pglite_async_session, epl_listing())

        # Far outside the match window: only the id can find the event.
        moved = epl_listing(commence_time=TWIN_KICKOFF + timedelta(days=3))
        resolved = await resolve_event(pglite_async_session, moved)

        assert resolved.resolution is Resolution.BY_EXTERNAL_ID
        assert resolved.event.id == stored_epl_event.id

    @pytest.mark.asyncio
    async def test_creates_an_event_only_when_nothing_matches(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        listing = epl_listing(commence_time=TWIN_KICKOFF + timedelta(days=7))

        resolved = await resolve_event(pglite_async_session, listing)

        assert resolved.resolution is Resolution.CREATED
        assert resolved.event.id == "op_live_ARS_CHE"
        assert resolved.event.surrogate_id is not None
        assert resolved.event.surrogate_id != stored_epl_event.surrogate_id
        assert await external_ids(pglite_async_session, resolved.event) == {
            Source.ODDSPORTAL: "op-1"
        }

    @pytest.mark.asyncio
    async def test_a_source_without_ids_matches_and_records_nothing(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        listing = epl_listing(source=Source.FOOTBALL_DATA_UK, external_id=None)

        resolved = await resolve_event(pglite_async_session, listing)

        assert resolved.event.id == stored_epl_event.id
        assert await external_ids(pglite_async_session, stored_epl_event) == {}

    @pytest.mark.asyncio
    async def test_refuses_and_logs_a_second_id_from_one_source(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        await resolve_event(pglite_async_session, epl_listing(external_id="op-1"))

        resolved = await resolve_event(pglite_async_session, epl_listing(external_id="op-2"))

        assert resolved.event.id == stored_epl_event.id
        assert await external_ids(pglite_async_session, stored_epl_event) == {
            Source.ODDSPORTAL: "op-1"
        }
        [log] = await refusals(pglite_async_session)
        assert log.event_id == stored_epl_event.id
        assert log.raw_data is not None and log.raw_data["external_id"] == "op-2"

    @pytest.mark.asyncio
    async def test_ids_from_different_sources_sit_on_one_event(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        await resolve_event(pglite_async_session, epl_listing(Source.ODDSPORTAL, "op-1"))
        await resolve_event(pglite_async_session, epl_listing(Source.BETFAIR, "bf-1"))

        assert await external_ids(pglite_async_session, stored_epl_event) == {
            Source.ODDSPORTAL: "op-1",
            Source.BETFAIR: "bf-1",
        }

    @pytest.mark.asyncio
    async def test_takes_the_fixture_advisory_lock(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        await resolve_event(pglite_async_session, epl_listing())

        key = event_lock_key("soccer_epl", "Arsenal", "Chelsea")
        held = await pglite_async_session.execute(
            text(
                "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted"
                " AND ((classid::bigint << 32) | objid::bigint) = :key"
            ),
            {"key": key},
        )
        assert held.scalar_one() == 1

    @pytest.mark.asyncio
    async def test_a_taken_legacy_id_without_a_suffix_raises(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        listing = epl_listing(
            external_id=None,
            commence_time=TWIN_KICKOFF + timedelta(days=7),
            legacy_id=stored_epl_event.id,
        )

        with pytest.raises(EventIdTaken):
            await resolve_event(pglite_async_session, listing)


class TestEventLockKey:
    def test_ignores_kick_off_and_separates_fixtures(self) -> None:
        key = event_lock_key("soccer_epl", "Arsenal", "Chelsea")

        assert key == event_lock_key("soccer_epl", "Arsenal", "Chelsea")
        assert key != event_lock_key("soccer_epl", "Chelsea", "Arsenal")
        assert -(2**63) <= key < 2**63


class TestListedGame:
    def test_mlb_needs_a_game_pk(self) -> None:
        with pytest.raises(ValueError, match="gamePk"):
            ListedGame(Source.BETFAIR, MLB, "MLB", YANKEES, RED_SOX, GAME_1_START, "id")

    def test_a_source_that_issues_no_ids_takes_none(self) -> None:
        with pytest.raises(ValueError, match="issues no event ids"):
            epl_listing(source=Source.FOOTBALL_DATA_UK, external_id="row-7")


class TestResolveMlbEvent:
    @pytest.mark.asyncio
    async def test_doubleheader_games_get_one_event_each(self, pglite_async_session) -> None:
        game_1 = await resolve_event(
            pglite_async_session,
            mlb_listing(1001, GAME_1_START, sibling_starts=(GAME_2_START,), legacy_id="g1"),
        )
        game_2 = await resolve_event(
            pglite_async_session,
            mlb_listing(1002, GAME_2_START, sibling_starts=(GAME_1_START,), legacy_id="g2"),
        )

        assert game_1.resolution is Resolution.CREATED
        assert game_2.resolution is Resolution.CREATED
        assert game_1.event.id != game_2.event.id
        assert await external_ids(pglite_async_session, game_2.event) == {
            Source.MLB_STATS_API: "1002"
        }
        assert game_2.event.mlb_game_pk == 1002

    @pytest.mark.asyncio
    async def test_finds_a_doubleheader_game_by_its_game_pk(self, pglite_async_session) -> None:
        await resolve_event(pglite_async_session, mlb_listing(1001, GAME_1_START, legacy_id="g1"))
        await resolve_event(pglite_async_session, mlb_listing(1002, GAME_2_START, legacy_id="g2"))

        listing = mlb_listing(1002, GAME_2_START, external_id="bf-2", legacy_id="x")
        resolved = await resolve_event(pglite_async_session, listing)

        assert resolved.resolution is Resolution.BY_GAME_PK
        assert resolved.event.id == "g2"
        assert await external_ids(pglite_async_session, resolved.event) == {
            Source.MLB_STATS_API: "1002",
            Source.BETFAIR: "bf-2",
        }

    @pytest.mark.asyncio
    async def test_finds_an_event_holding_its_game_pk_only_in_the_column(
        self, pglite_async_session
    ) -> None:
        stored = Event(
            id="legacy",
            sport_key=MLB,
            sport_title="MLB",
            commence_time=GAME_1_START,
            home_team=YANKEES,
            away_team=RED_SOX,
            mlb_game_pk=1001,
        )
        pglite_async_session.add(stored)
        await pglite_async_session.flush()

        resolved = await resolve_event(pglite_async_session, mlb_listing(1001, GAME_1_START))

        assert resolved.resolution is Resolution.BY_GAME_PK
        assert resolved.event.id == "legacy"
        assert await external_ids(pglite_async_session, stored) == {Source.MLB_STATS_API: "1001"}

    @pytest.mark.asyncio
    async def test_claims_a_nearby_event_holding_no_game_pk(self, pglite_async_session) -> None:
        stored = Event(
            id="unkeyed",
            sport_key=MLB,
            sport_title="MLB",
            commence_time=GAME_1_START + timedelta(minutes=1),
            home_team=YANKEES,
            away_team=RED_SOX,
        )
        pglite_async_session.add(stored)
        await pglite_async_session.flush()

        resolved = await resolve_event(pglite_async_session, mlb_listing(1001, GAME_1_START))

        assert resolved.resolution is Resolution.MATCHED
        assert resolved.event.id == "unkeyed"
        assert resolved.event.mlb_game_pk == 1001
        assert resolved.event.commence_time == GAME_1_START

    @pytest.mark.asyncio
    async def test_never_claims_an_event_holding_another_game_pk(
        self, pglite_async_session
    ) -> None:
        await resolve_event(pglite_async_session, mlb_listing(1001, GAME_1_START, legacy_id="g1"))

        resolved = await resolve_event(
            pglite_async_session,
            mlb_listing(1002, GAME_1_START + timedelta(minutes=5), legacy_id="g2"),
        )

        assert resolved.resolution is Resolution.CREATED
        assert resolved.event.id == "g2"

    @pytest.mark.asyncio
    async def test_a_taken_legacy_id_takes_the_game_pk_as_suffix(
        self, pglite_async_session
    ) -> None:
        await resolve_event(pglite_async_session, mlb_listing(1001, GAME_1_START, legacy_id="g"))

        resolved = await resolve_event(
            pglite_async_session, mlb_listing(1002, GAME_2_START, legacy_id="g")
        )

        assert resolved.event.id == "g_1002"
        count = await pglite_async_session.execute(select(func.count()).select_from(Event))
        assert count.scalar_one() == 2
