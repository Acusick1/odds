"""Integration tests for the locked event resolver."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import DataQualityLog, Event, EventExternalId
from odds_core.vocabulary import Source
from odds_lambda.mlb_game_resolver import ResolutionRule, ResolvedGame
from odds_lambda.storage import event_resolver
from odds_lambda.storage.event_resolver import (
    EventIdTaken,
    ListedGame,
    LockOrderError,
    Resolution,
    StaleListing,
    attach_event,
    attach_events,
    event_lock_key,
    resolve_event,
    resolve_events,
)
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import TWIN_KICKOFF, make_epl_event

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

    @pytest.mark.asyncio
    async def test_ties_go_to_the_oldest_event(self, pglite_async_session) -> None:
        first = make_epl_event("zz_first", TWIN_KICKOFF - timedelta(minutes=30))
        second = make_epl_event("aa_second", TWIN_KICKOFF + timedelta(minutes=30))
        pglite_async_session.add(first)
        await pglite_async_session.flush()
        pglite_async_session.add(second)
        await pglite_async_session.flush()

        resolved = await resolve_event(pglite_async_session, epl_listing())

        assert resolved.event.id == "zz_first"

    @pytest.mark.asyncio
    async def test_a_taken_suffixed_id_raises_and_leaves_the_transaction_usable(
        self, pglite_async_session
    ) -> None:
        far = TWIN_KICKOFF + timedelta(days=30)
        pglite_async_session.add_all(
            [make_epl_event("op_live_ARS_CHE", far), make_epl_event("op_live_ARS_CHE_op-1", far)]
        )
        await pglite_async_session.flush()

        with pytest.raises(EventIdTaken):
            await resolve_event(pglite_async_session, epl_listing())

        count = await pglite_async_session.execute(select(func.count()).select_from(Event))
        assert count.scalar_one() == 2


class TestListedStartMovesTheEvent:
    @pytest.mark.asyncio
    async def test_a_future_start_from_the_authority_moves_the_event(
        self, pglite_async_session
    ) -> None:
        kickoff = (datetime.now(UTC) + timedelta(days=2)).replace(microsecond=0)
        event = make_epl_event("odds_api_hex_1", kickoff)
        pglite_async_session.add(event)
        await pglite_async_session.flush()

        await resolve_event(
            pglite_async_session, epl_listing(commence_time=kickoff + timedelta(minutes=30))
        )

        assert event.commence_time == kickoff + timedelta(minutes=30)

    @pytest.mark.asyncio
    async def test_a_game_brought_forward_to_a_time_already_past_moves(
        self, pglite_async_session
    ) -> None:
        kickoff = (datetime.now(UTC) + timedelta(minutes=20)).replace(microsecond=0)
        event = make_epl_event("odds_api_hex_1", kickoff)
        pglite_async_session.add(event)
        await pglite_async_session.flush()
        brought_forward = kickoff - timedelta(minutes=40)

        await resolve_event(pglite_async_session, epl_listing(commence_time=brought_forward))

        assert event.commence_time == brought_forward

    @pytest.mark.asyncio
    async def test_an_in_play_start_after_the_stored_kick_off_never_moves(
        self, pglite_async_session
    ) -> None:
        kickoff = (datetime.now(UTC) - timedelta(minutes=20)).replace(microsecond=0)
        event = make_epl_event("odds_api_hex_1", kickoff)
        pglite_async_session.add(event)
        await pglite_async_session.flush()

        await resolve_event(
            pglite_async_session, epl_listing(commence_time=kickoff + timedelta(minutes=5))
        )

        assert event.commence_time == kickoff


class TestStaleListing:
    @pytest.mark.asyncio
    async def test_a_live_listing_too_far_past_creates_nothing(self, pglite_async_session) -> None:
        listing = ListedGame(
            source=Source.ODDSPORTAL,
            sport_key="soccer_epl",
            sport_title="EPL",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=TWIN_KICKOFF,
            legacy_id="op_live_ARS_CHE",
            earliest_created_start=TWIN_KICKOFF + timedelta(hours=1),
        )

        with pytest.raises(StaleListing):
            await resolve_event(pglite_async_session, listing)

        count = await pglite_async_session.execute(select(func.count()).select_from(Event))
        assert count.scalar_one() == 0


class TestLockOrder:
    @staticmethod
    def listing(home: str, away: str) -> ListedGame:
        return ListedGame(
            Source.ODDSPORTAL, "soccer_epl", "EPL", home, away, TWIN_KICKOFF, f"{home}-{away}"
        )

    def ordered(self) -> list[ListedGame]:
        games = [self.listing("Arsenal", "Chelsea"), self.listing("Liverpool", "Everton")]
        return sorted(games, key=lambda g: event_lock_key(g.sport_key, g.home_team, g.away_team))

    @pytest.mark.asyncio
    async def test_a_lower_new_lock_in_one_transaction_raises(self, pglite_async_session) -> None:
        low, high = self.ordered()
        await resolve_event(pglite_async_session, high)
        await resolve_event(pglite_async_session, high)

        with pytest.raises(LockOrderError):
            await resolve_event(pglite_async_session, low)

    @pytest.mark.asyncio
    async def test_a_new_transaction_starts_a_new_order(self, pglite_async_session) -> None:
        low, high = self.ordered()
        await resolve_event(pglite_async_session, high)
        await pglite_async_session.commit()

        resolved = await resolve_event(pglite_async_session, low)

        assert resolved.resolution is Resolution.CREATED

    @pytest.mark.asyncio
    async def test_a_batch_resolves_in_key_order_and_answers_in_input_order(
        self, pglite_async_session
    ) -> None:
        low, high = self.ordered()

        resolved = await resolve_events(pglite_async_session, [high, low])

        assert [r.event.id for r in resolved] == [high.legacy_id, low.legacy_id]


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
    async def test_moves_once_to_the_schedules_start(
        self, pglite_async_session, monkeypatch
    ) -> None:
        moves: list[tuple[Source, datetime | None]] = []
        real_move = event_resolver.move_kickoff

        async def recording_move(session, event, source, start, **kwargs):  # type: ignore[no-untyped-def]
            moves.append((source, start))
            return await real_move(session, event, source, start, **kwargs)

        monkeypatch.setattr(event_resolver, "move_kickoff", recording_move)
        listing = ListedGame(
            source=Source.MLB_STATS_API,
            sport_key=MLB,
            sport_title="MLB",
            home_team=YANKEES,
            away_team=RED_SOX,
            commence_time=GAME_1_START + timedelta(minutes=20),
            legacy_id="g1",
            mlb_game=ResolvedGame(1001, ResolutionRule.SINGLE_GAME, GAME_1_START),
        )

        resolved = await resolve_event(pglite_async_session, listing)

        assert moves == [(Source.MLB_STATS_API, GAME_1_START)]
        assert resolved.event.commence_time == GAME_1_START

    @pytest.mark.asyncio
    async def test_refuses_a_source_id_another_event_holds(self, pglite_async_session) -> None:
        await resolve_event(
            pglite_async_session,
            mlb_listing(1001, GAME_1_START, external_id="bf-1", legacy_id="g1"),
        )
        await resolve_event(pglite_async_session, mlb_listing(1002, GAME_2_START, legacy_id="g2"))

        # The schedule says game 2, but the source reuses game 1's id.
        resolved = await resolve_event(
            pglite_async_session, mlb_listing(1002, GAME_2_START, external_id="bf-1")
        )

        assert resolved.event.id == "g2"
        assert await external_ids(pglite_async_session, resolved.event) == {
            Source.MLB_STATS_API: "1002"
        }
        [log] = await refusals(pglite_async_session)
        assert log.event_id == "g2"
        assert log.raw_data is not None and "held by event" in log.raw_data["reason"]

    @pytest.mark.asyncio
    async def test_refuses_a_game_pk_when_the_event_holds_another(
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
        pglite_async_session.add(
            EventExternalId(
                source=Source.MLB_STATS_API, external_id="999", event_id=stored.surrogate_id
            )
        )
        await pglite_async_session.flush()

        resolved = await resolve_event(pglite_async_session, mlb_listing(1001, GAME_1_START))

        assert resolved.event.id == "legacy"
        assert await external_ids(pglite_async_session, stored) == {Source.MLB_STATS_API: "999"}
        [log] = await refusals(pglite_async_session)
        assert log.raw_data is not None and log.raw_data["external_id"] == "1001"

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


def polymarket_listing(
    external_id: str | None = "900",
    commence_time: datetime = TWIN_KICKOFF,
    home: str = "Arsenal",
    away: str = "Chelsea",
) -> ListedGame:
    return ListedGame(
        source=Source.POLYMARKET,
        sport_key="soccer_epl",
        sport_title="EPL",
        home_team=home,
        away_team=away,
        commence_time=commence_time,
        external_id=external_id,
    )


class TestAttachEvent:
    @pytest.mark.asyncio
    async def test_attaches_by_teams_and_records_the_source_id(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        attached = await attach_event(pglite_async_session, polymarket_listing())

        assert attached is not None
        assert (attached.event.id, attached.resolution) == (
            stored_epl_event.id,
            Resolution.MATCHED,
        )
        assert await external_ids(pglite_async_session, stored_epl_event) == {
            Source.POLYMARKET: "900"
        }

    @pytest.mark.asyncio
    async def test_creates_nothing_when_no_event_matches(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        far = polymarket_listing(commence_time=TWIN_KICKOFF + timedelta(days=7))

        assert await attach_event(pglite_async_session, far) is None

        count = await pglite_async_session.execute(select(func.count()).select_from(Event))
        assert count.scalar_one() == 1
        assert await external_ids(pglite_async_session, stored_epl_event) == {}

    @pytest.mark.asyncio
    async def test_a_recorded_id_attaches_across_a_moved_kickoff(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        await attach_event(pglite_async_session, polymarket_listing())

        moved = polymarket_listing(commence_time=TWIN_KICKOFF + timedelta(days=3))
        attached = await attach_event(pglite_async_session, moved)

        assert attached is not None
        assert attached.resolution is Resolution.BY_EXTERNAL_ID

    @pytest.mark.asyncio
    async def test_does_not_move_the_kickoff(self, pglite_async_session, stored_epl_event) -> None:
        later = TWIN_KICKOFF + timedelta(minutes=90)
        listing = epl_listing(commence_time=later)

        await attach_event(pglite_async_session, listing)

        assert stored_epl_event.commence_time == TWIN_KICKOFF

    @pytest.mark.asyncio
    async def test_a_second_id_for_the_event_is_refused_and_logged_once(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        await attach_event(pglite_async_session, polymarket_listing("900"))

        first = await attach_event(pglite_async_session, polymarket_listing("901"))
        again = await attach_event(pglite_async_session, polymarket_listing("901"))

        assert (first, again) == (None, None)
        assert await external_ids(pglite_async_session, stored_epl_event) == {
            Source.POLYMARKET: "900"
        }
        assert len(await refusals(pglite_async_session)) == 1

    @pytest.mark.asyncio
    async def test_a_held_id_wins_over_a_nearer_event_of_the_fixture(
        self, pglite_async_session
    ) -> None:
        pglite_async_session.add(make_epl_event("e1", TWIN_KICKOFF, "Arsenal", "Chelsea"))
        pglite_async_session.add(
            make_epl_event("e2", TWIN_KICKOFF + timedelta(days=30), "Arsenal", "Chelsea")
        )
        await pglite_async_session.commit()
        await attach_event(pglite_async_session, polymarket_listing("900"))

        later = polymarket_listing("900", commence_time=TWIN_KICKOFF + timedelta(days=30))
        attached = await attach_event(pglite_async_session, later)

        assert attached is not None
        assert attached.event.id == "e1"

    @pytest.mark.asyncio
    async def test_creating_an_event_needs_a_legacy_id(self, pglite_async_session) -> None:
        with pytest.raises(ValueError, match="legacy_id"):
            await resolve_event(pglite_async_session, polymarket_listing())


class TestAttachEvents:
    @pytest.mark.asyncio
    async def test_a_mixed_batch_attaches_in_key_order_and_answers_in_input_order(
        self, pglite_async_session
    ) -> None:
        fixtures = [("Arsenal", "Chelsea"), ("Liverpool", "Everton"), ("Fulham", "Brentford")]
        for i, (home, away) in enumerate(fixtures):
            pglite_async_session.add(make_epl_event(f"e{i}", TWIN_KICKOFF, home, away))
        pglite_async_session.add(
            make_epl_event("elsewhere", TWIN_KICKOFF + timedelta(days=9), "Spurs", "Wolves")
        )
        await pglite_async_session.commit()
        games = [
            polymarket_listing(str(i), home=home, away=away)
            for i, (home, away) in enumerate(fixtures)
        ]
        games.append(polymarket_listing("99", home="Spurs", away="Wolves"))
        keys = [event_lock_key(g.sport_key, g.home_team, g.away_team) for g in games]
        assert keys != sorted(keys)

        attached = await attach_events(pglite_async_session, games)

        assert [a.event.id if a else None for a in attached] == ["e0", "e1", "e2", None]
