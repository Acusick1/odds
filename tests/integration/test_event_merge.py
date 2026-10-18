"""Integration tests for collapsing several events for one match into one."""

from __future__ import annotations

from datetime import timedelta

import pytest
from odds_cli.commands import events as events_cli
from odds_core.api_models import EventsResponse
from odds_core.match_brief_models import MatchBrief
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot, utc_now
from odds_core.paper_trade_models import PaperTrade
from odds_core.prediction_models import Prediction
from odds_lambda.event_sync import EventSyncService
from odds_lambda.oddsportal_common import find_existing_event
from odds_lambda.storage.event_merge import EventMerger
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select

from tests.test_helpers import (
    API_EVENT_ID,
    TWIN_EVENT_ID,
    TWIN_KICKOFF,
    StubOddsClient,
    make_epl_event,
)

CHILD_MODELS = (OddsSnapshot, Odds, Prediction, PaperTrade, MatchBrief)


async def _event_ids(session, model) -> set[str]:
    return set((await session.execute(select(model.event_id))).scalars().all())


def _session_maker(session):
    class _Maker:
        def __call__(self):
            return self

        async def __aenter__(self):
            return session

        async def __aexit__(self, *args) -> bool:
            return False

    return _Maker()


async def _stored_event_ids(session) -> set[str]:
    return set((await session.execute(select(Event.id))).scalars().all())


class TestAdoptTwins:
    @pytest.mark.asyncio
    async def test_moves_every_row_and_deletes_twin(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        api_event = make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(minutes=30))
        session.add(api_event)
        await session.flush()

        adopted = await EventMerger(session).adopt_twins([api_event])

        assert adopted == 1
        for model in CHILD_MODELS:
            assert await _event_ids(session, model) == {API_EVENT_ID}, model
        assert await session.get(Event, TWIN_EVENT_ID) is None

    @pytest.mark.asyncio
    async def test_reads_every_table_referencing_events_from_the_catalog(
        self, live_twin_with_rows
    ) -> None:
        columns = await EventMerger(live_twin_with_rows)._event_reference_columns()
        tables = {table for _, table, _ in columns}
        assert {m.__tablename__ for m in CHILD_MODELS} <= tables

    @pytest.mark.asyncio
    async def test_carries_result_when_api_event_has_none(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        twin = await session.get(Event, TWIN_EVENT_ID)
        twin.status = EventStatus.FINAL
        twin.home_score, twin.away_score = 2, 1
        twin.completed_at = utc_now()
        api_event = make_epl_event(API_EVENT_ID)
        session.add(api_event)
        await session.flush()

        await EventMerger(session).adopt_twins([api_event])

        stored = await session.get(Event, API_EVENT_ID)
        assert stored.status == EventStatus.FINAL
        assert (stored.home_score, stored.away_score) == (2, 1)
        assert stored.completed_at is not None

    @pytest.mark.asyncio
    async def test_keeps_api_result_when_present(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        twin = await session.get(Event, TWIN_EVENT_ID)
        twin.status = EventStatus.FINAL
        twin.home_score, twin.away_score = 2, 1
        api_event = make_epl_event(
            API_EVENT_ID, status=EventStatus.FINAL, home_score=3, away_score=0
        )
        session.add(api_event)
        await session.flush()

        await EventMerger(session).adopt_twins([api_event])

        stored = await session.get(Event, API_EVENT_ID)
        assert (stored.home_score, stored.away_score) == (3, 0)

    @pytest.mark.asyncio
    async def test_no_twin_is_a_noop(self, pglite_async_session) -> None:
        api_event = make_epl_event(API_EVENT_ID)
        pglite_async_session.add(api_event)
        await pglite_async_session.flush()

        assert await EventMerger(pglite_async_session).adopt_twins([api_event]) == 0

    @pytest.mark.asyncio
    async def test_twin_outside_window_untouched(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        api_event = make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(hours=3))
        session.add(api_event)
        await session.flush()

        assert await EventMerger(session).adopt_twins([api_event]) == 0
        assert await session.get(Event, TWIN_EVENT_ID) is not None
        assert await _event_ids(session, OddsSnapshot) == {TWIN_EVENT_ID}

    @pytest.mark.asyncio
    async def test_different_teams_untouched(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        api_event = make_epl_event(API_EVENT_ID, away_team="Liverpool")
        session.add(api_event)
        await session.flush()

        assert await EventMerger(session).adopt_twins([api_event]) == 0
        assert await session.get(Event, TWIN_EVENT_ID) is not None

    @pytest.mark.asyncio
    async def test_event_sync_adopts_twin_of_existing_api_event(
        self, live_twin_with_rows, mock_session_factory, test_engine
    ) -> None:
        session = live_twin_with_rows
        api_event = make_epl_event(API_EVENT_ID)
        session.add(api_event)
        await session.commit()

        client = StubOddsClient(
            None,
            EventsResponse(
                events=[make_epl_event(API_EVENT_ID)],
                response_time_ms=1,
                quota_remaining=None,
                timestamp=utc_now(),
            ),
        )
        service = EventSyncService(client=client, session_factory=mock_session_factory)  # type: ignore

        await service.sync_sport("soccer_epl")

        session.expire_all()
        assert await session.get(Event, TWIN_EVENT_ID) is None
        assert await _event_ids(session, PaperTrade) == {API_EVENT_ID}


class TestFindOrCreateEventPrefersApiEvent:
    @pytest.mark.asyncio
    async def test_prefers_non_live_candidate(self, pglite_async_session) -> None:
        # The API id sorts after the twin id, so ordering alone would pick the twin.
        pglite_async_session.add_all([make_epl_event(TWIN_EVENT_ID), make_epl_event(API_EVENT_ID)])
        await pglite_async_session.flush()

        event_id, created = await OddsWriter(pglite_async_session).find_or_create_event(
            "Arsenal", "Chelsea", TWIN_KICKOFF, "soccer_epl", "EPL"
        )

        assert (event_id, created) == (API_EVENT_ID, False)


FDUK_EVENT_ID = "fduk_2025-2026_ARS_CHE_2026-04-15"
HISTORICAL_EVENT_ID = "op_2025-2026_ARS_CHE_2026-04-15"


class TestMergeGroup:
    @pytest.mark.asyncio
    async def test_keeps_api_event_over_every_other_source(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(i)
                for i in (FDUK_EVENT_ID, TWIN_EVENT_ID, HISTORICAL_EVENT_ID, API_EVENT_ID)
            ]
        )
        await session.flush()

        groups = await EventMerger(session).find_twin_groups("soccer_epl")
        report = await EventMerger(session).merge_group(groups[0])
        await session.flush()

        assert report.kept_id == API_EVENT_ID
        assert set(report.merged_ids) == {FDUK_EVENT_ID, TWIN_EVENT_ID, HISTORICAL_EVENT_ID}
        assert await _stored_event_ids(session) == {API_EVENT_ID}

    @pytest.mark.asyncio
    async def test_without_api_event_keeps_historical_then_live_then_fduk(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all([make_epl_event(FDUK_EVENT_ID), make_epl_event(TWIN_EVENT_ID)])
        await session.flush()

        report = await EventMerger(session).merge_group(
            (await EventMerger(session).find_twin_groups())[0]
        )

        assert report.kept_id == TWIN_EVENT_ID

    @pytest.mark.asyncio
    async def test_moves_rows_from_every_duplicate(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        session.add_all([make_epl_event(API_EVENT_ID), make_epl_event(FDUK_EVENT_ID)])
        await session.flush()
        session.add(
            OddsSnapshot(
                event_id=FDUK_EVENT_ID,
                snapshot_time=TWIN_KICKOFF,
                raw_data={},
                bookmaker_count=0,
            )
        )
        await session.flush()

        await EventMerger(session).merge_all_twins("soccer_epl")
        await session.flush()

        assert await _event_ids(session, OddsSnapshot) == {API_EVENT_ID}
        snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()
        assert len(snapshots) == 2

    @pytest.mark.asyncio
    async def test_distinct_matches_are_not_groups(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(API_EVENT_ID),
                make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF + timedelta(days=7)),
                make_epl_event(FDUK_EVENT_ID, away_team="Liverpool"),
            ]
        )
        await session.flush()

        assert await EventMerger(session).find_twin_groups() == []


class TestFindersAreReadOnly:
    @pytest.mark.asyncio
    async def test_find_or_create_returns_canonical_and_deletes_nothing(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all([make_epl_event(TWIN_EVENT_ID), make_epl_event(API_EVENT_ID)])
        await session.flush()

        event_id, created = await OddsWriter(session).find_or_create_event(
            "Arsenal", "Chelsea", TWIN_KICKOFF, "soccer_epl", "EPL"
        )
        await session.flush()

        assert (event_id, created) == (API_EVENT_ID, False)
        assert await _stored_event_ids(session) == {API_EVENT_ID, TWIN_EVENT_ID}

    @pytest.mark.asyncio
    async def test_find_or_create_never_creates_beside_twins(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all([make_epl_event(TWIN_EVENT_ID), make_epl_event(FDUK_EVENT_ID)])
        await session.flush()

        event_id, created = await OddsWriter(session).find_or_create_event(
            "Arsenal", "Chelsea", TWIN_KICKOFF, "soccer_epl", "EPL"
        )

        assert (event_id, created) == (TWIN_EVENT_ID, False)

    @pytest.mark.asyncio
    async def test_historical_twins_resolve_to_canonical_without_merging(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all([make_epl_event(FDUK_EVENT_ID), make_epl_event(API_EVENT_ID)])
        await session.flush()

        event_id = await find_existing_event(
            session, "Arsenal", "Chelsea", TWIN_KICKOFF, "soccer_epl"
        )

        assert event_id == API_EVENT_ID
        assert await _stored_event_ids(session) == {API_EVENT_ID, FDUK_EVENT_ID}

    @pytest.mark.asyncio
    async def test_historical_back_to_back_fixtures_are_never_conflated(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        next_day = TWIN_KICKOFF + timedelta(hours=20)
        session.add_all(
            [make_epl_event(API_EVENT_ID), make_epl_event(HISTORICAL_EVENT_ID, next_day)]
        )
        await session.flush()

        near = await find_existing_event(session, "Arsenal", "Chelsea", TWIN_KICKOFF, "soccer_epl")
        between = await find_existing_event(
            session, "Arsenal", "Chelsea", TWIN_KICKOFF + timedelta(hours=10), "soccer_epl"
        )

        assert near == API_EVENT_ID
        assert between is None
        assert await _stored_event_ids(session) == {API_EVENT_ID, HISTORICAL_EVENT_ID}

    @pytest.mark.asyncio
    async def test_historical_match_is_scoped_to_sport(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add(make_epl_event(API_EVENT_ID))
        await session.flush()

        assert (
            await find_existing_event(session, "Arsenal", "Chelsea", TWIN_KICKOFF, "baseball_mlb")
            is None
        )


class TestFindTwinGroups:
    @pytest.mark.asyncio
    async def test_chain_of_events_is_not_one_group(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event("a", TWIN_KICKOFF),
                make_epl_event("b", TWIN_KICKOFF + timedelta(hours=1.5)),
                make_epl_event("c", TWIN_KICKOFF + timedelta(hours=3)),
            ]
        )
        await session.flush()

        groups = await EventMerger(session).find_twin_groups()

        assert [[e.id for e in g] for g in groups] == [["a", "b"]]


class TestIdentityChecks:
    @pytest.mark.asyncio
    async def test_flags_twins_with_their_sport_and_clears_after_merge(
        self, pglite_async_session
    ) -> None:
        from odds_core.config import get_settings
        from odds_lambda.health_monitor import HealthMonitor

        session = pglite_async_session
        monitor = HealthMonitor(session, get_settings())
        assert await monitor.check_event_identity() == []

        upcoming = utc_now() + timedelta(days=3)
        session.add_all(
            [make_epl_event(TWIN_EVENT_ID, upcoming), make_epl_event(API_EVENT_ID, upcoming)]
        )
        await session.flush()
        assert await monitor.check_event_identity() == [
            "1 soccer_epl matches stored as more than one event"
        ]

        await EventMerger(session).merge_all_twins()
        await session.flush()
        assert await monitor.check_event_identity() == []

    @pytest.mark.asyncio
    async def test_flags_a_rescheduled_fixture_stored_twice_in_a_season(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF),
                make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(days=2)),
            ]
        )
        await session.flush()

        repeated = await EventMerger(session).find_repeated_fixtures("soccer_epl")

        assert [[e.id for e in g] for g in repeated] == [[TWIN_EVENT_ID, API_EVENT_ID]]
        assert await EventMerger(session).find_twin_groups() == []

    @pytest.mark.asyncio
    async def test_same_fixture_in_two_seasons_is_not_repeated(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF),
                make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(days=365)),
            ]
        )
        await session.flush()

        assert await EventMerger(session).find_repeated_fixtures("soccer_epl") == []

    @pytest.mark.asyncio
    async def test_flags_recent_scheduled_events_long_past_kickoff(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add(make_epl_event(API_EVENT_ID, utc_now() - timedelta(days=3)))
        session.add(make_epl_event(TWIN_EVENT_ID, utc_now() + timedelta(days=3)))
        session.add(make_epl_event(FDUK_EVENT_ID, utc_now() - timedelta(days=30)))
        await session.flush()

        stale = await EventMerger(session).find_stale_scheduled(
            timedelta(hours=24), timedelta(days=7), utc_now()
        )

        assert [e.id for e in stale] == [API_EVENT_ID]


class TestMergeTwinsCommand:
    @pytest.mark.asyncio
    async def test_apply_merges_and_reports_counts(
        self, pglite_async_session, monkeypatch, capsys
    ) -> None:
        session = pglite_async_session
        session.add_all([make_epl_event(TWIN_EVENT_ID), make_epl_event(API_EVENT_ID)])
        await session.commit()

        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(session))

        await events_cli._twins("soccer_epl", apply=False)
        assert await session.get(Event, TWIN_EVENT_ID) is not None

        await events_cli._twins("soccer_epl", apply=True)
        assert await session.get(Event, TWIN_EVENT_ID) is None
        assert "Merged 1 events into 1; matches with twins now: 0" in capsys.readouterr().out


class TestMergeCommand:
    @pytest.mark.asyncio
    async def test_dry_run_then_apply_keeps_the_named_event(
        self, pglite_async_session, monkeypatch
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF, status=EventStatus.FINAL),
                make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(days=2)),
            ]
        )
        await session.commit()
        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(session))

        await events_cli._merge(TWIN_EVENT_ID, API_EVENT_ID, apply=False)
        assert await session.get(Event, API_EVENT_ID) is not None

        await events_cli._merge(TWIN_EVENT_ID, API_EVENT_ID, apply=True)
        assert await session.get(Event, API_EVENT_ID) is None
        assert await session.get(Event, TWIN_EVENT_ID) is not None

    @pytest.mark.asyncio
    async def test_refuses_different_sports(self, pglite_async_session, monkeypatch) -> None:
        import typer

        session = pglite_async_session
        other = make_epl_event(API_EVENT_ID)
        other.sport_key = "baseball_mlb"
        session.add_all([make_epl_event(TWIN_EVENT_ID), other])
        await session.commit()
        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(session))

        with pytest.raises(typer.BadParameter):
            await events_cli._merge(TWIN_EVENT_ID, API_EVENT_ID, apply=True)


class TestMergeRebasesSnapshotTiming:
    @pytest.mark.asyncio
    async def test_moved_snapshots_follow_the_kept_events_kickoff(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        later = TWIN_KICKOFF + timedelta(days=2)
        session.add_all(
            [make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF), make_epl_event(API_EVENT_ID, later)]
        )
        await session.flush()
        snapshot = OddsSnapshot(
            event_id=TWIN_EVENT_ID,
            snapshot_time=later - timedelta(hours=1),
            raw_data={},
            bookmaker_count=0,
            hours_until_commence=-47.0,
            fetch_tier="closing",
        )
        session.add(snapshot)
        await session.flush()

        await EventMerger(session).merge(
            await session.get(Event, API_EVENT_ID), [await session.get(Event, TWIN_EVENT_ID)]
        )
        await session.flush()
        await session.refresh(snapshot)

        assert snapshot.event_id == API_EVENT_ID
        assert snapshot.hours_until_commence == pytest.approx(1.0)
        assert snapshot.fetch_tier == "closing"

    @pytest.mark.asyncio
    async def test_same_kickoff_leaves_snapshots_untouched(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        session.add(make_epl_event(API_EVENT_ID))
        await session.flush()
        before = (await session.execute(select(OddsSnapshot.hours_until_commence))).scalar_one()

        await EventMerger(session).merge_all_twins()
        await session.flush()

        after = (await session.execute(select(OddsSnapshot.hours_until_commence))).scalar_one()
        assert after == before

    @pytest.mark.asyncio
    async def test_merge_command_refuses_the_same_id(
        self, pglite_async_session, monkeypatch
    ) -> None:
        import typer

        session = pglite_async_session
        session.add(make_epl_event(API_EVENT_ID))
        await session.commit()
        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(session))

        with pytest.raises(typer.BadParameter):
            await events_cli._merge(API_EVENT_ID, API_EVENT_ID, apply=True)
        assert await session.get(Event, API_EVENT_ID) is not None
