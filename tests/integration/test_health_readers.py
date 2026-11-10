"""The read-only health views the monitor and the web UI's status strip share."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from odds_core.config import get_settings
from odds_core.models import AlertHistory, Event, EventStatus, OddsSnapshot
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda.health_readers import (
    Freshness,
    last_heartbeats,
    schedule_heartbeat_name,
    source_freshness,
)
from odds_lambda.scheduling.schedule_store import (
    StoredSchedule,
    read_outstanding_job_names,
    read_schedules,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import create_scheduler_tables

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


async def add_event(
    session: AsyncSession,
    event_id: int,
    kickoff: datetime,
    snapshots: tuple[tuple[Source, datetime], ...] = (),
) -> None:
    session.add(
        Event(
            id=event_id,
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=kickoff,
            home_team=f"Home {event_id}",
            away_team=f"Away {event_id}",
            status=EventStatus.SCHEDULED,
        )
    )
    await session.flush()
    for source, taken in snapshots:
        session.add(
            OddsSnapshot(
                event_id=event_id,
                snapshot_time=taken,
                source=source,
                market=Market.ONE_X_TWO,
                time_basis=TimeBasis.OBSERVED,
            )
        )
    await session.commit()


def oddsportal_epl(found: list) -> dict[Market, tuple[Freshness, float | None]]:
    return {
        f.market: (f.status, f.active_hours)
        for f in found
        if f.source_key == "oddsportal" and f.sport_key == "soccer_epl"
    }


class TestSourceFreshness:
    async def test_a_market_past_its_threshold_is_stale_and_one_never_written_is_unknown(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await add_event(
            pglite_async_session,
            1,
            NOW + timedelta(hours=4),
            ((Source.ODDSPORTAL, NOW - timedelta(hours=5)),),
        )

        found = await source_freshness(pglite_async_session, get_settings(), NOW)

        assert oddsportal_epl(found) == {
            Market.ONE_X_TWO: (Freshness.STALE, pytest.approx(5.0)),
            Market.TOTALS: (Freshness.UNKNOWN, None),
        }

    async def test_a_market_inside_its_cadence_is_fresh(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await add_event(
            pglite_async_session,
            1,
            NOW + timedelta(hours=3),
            ((Source.ODDSPORTAL, NOW - timedelta(hours=2)),),
        )

        found = await source_freshness(pglite_async_session, get_settings(), NOW)

        assert oddsportal_epl(found)[Market.ONE_X_TWO][0] is Freshness.FRESH

    async def test_without_a_fixture_in_the_horizon_a_source_is_idle_and_keeps_its_last_time(
        self, pglite_async_session: AsyncSession
    ) -> None:
        last = NOW - timedelta(days=10)
        await add_event(
            pglite_async_session, 1, NOW - timedelta(days=9), ((Source.ODDSPORTAL, last),)
        )

        found = await source_freshness(pglite_async_session, get_settings(), NOW)

        [entry] = [
            f
            for f in found
            if (f.source_key, f.sport_key, f.market)
            == ("oddsportal", "soccer_epl", Market.ONE_X_TWO)
        ]
        assert (entry.status, entry.last_snapshot) == (Freshness.IDLE, last)
        assert entry.job_name == "fetch-oddsportal-epl"


class TestHeartbeats:
    async def test_last_heartbeats_are_the_newest_row_per_job(
        self, pglite_async_session: AsyncSession
    ) -> None:
        for job, hours_ago in (("check-health", 3), ("check-health", 1), ("fetch-odds-epl", 5)):
            pglite_async_session.add(
                AlertHistory(
                    alert_type=f"heartbeat:{job}",
                    severity="info",
                    message="ok",
                    sent_at=NOW - timedelta(hours=hours_ago),
                )
            )
        await pglite_async_session.commit()

        beats = await last_heartbeats(
            pglite_async_session, ["check-health", "fetch-odds-epl", "never-ran"]
        )

        assert beats == {
            "check-health": NOW - timedelta(hours=1),
            "fetch-odds-epl": NOW - timedelta(hours=5),
        }

    @pytest.mark.parametrize(
        ("schedule_id", "heartbeat"),
        [
            ("fetch-oddsportal-epl", "fetch-oddsportal-epl"),
            ("fetch-espn-fixtures-epl", "fetch-espn-fixtures"),
            ("fetch-mlb-probables-mlb", "fetch-mlb-probables"),
            ("check-health", "check-health"),
            ("run-candidates-epl", "run-candidates-epl"),
        ],
    )
    def test_a_schedule_maps_to_the_heartbeat_its_job_records(
        self, schedule_id: str, heartbeat: str
    ) -> None:
        assert schedule_heartbeat_name(schedule_id) == heartbeat


class TestReadSchedules:
    async def test_reads_each_schedule_and_counts_its_acquired_jobs(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await create_scheduler_tables(pglite_async_session)
        for task in ("check-health", "fetch-odds-epl"):
            await pglite_async_session.execute(
                text(
                    "INSERT INTO tasks (id, func, job_executor, metadata) "
                    "VALUES (:id, 'f', 'async', '{}')"
                ),
                {"id": task},
            )
            await pglite_async_session.execute(
                text(
                    "INSERT INTO schedules (id, task_id, coalesce, job_executor, metadata, "
                    "next_fire_time) VALUES (:id, :id, 'latest', 'async', '{}', :next)"
                ),
                {"id": task, "next": NOW + timedelta(minutes=30)},
            )
        await pglite_async_session.execute(
            text(
                "INSERT INTO jobs (id, task_id, args, kwargs, schedule_id, executor, metadata, "
                "created_at, acquired_by) VALUES (:id, 'check-health', '', '', 'check-health', "
                "'async', '{}', :now, 'worker')"
            ),
            {"id": uuid.uuid4(), "now": NOW},
        )
        await pglite_async_session.commit()

        schedules = await read_schedules(pglite_async_session)

        assert schedules == [
            StoredSchedule("check-health", NOW + timedelta(minutes=30), None, False, 1),
            StoredSchedule("fetch-odds-epl", NOW + timedelta(minutes=30), None, False, 0),
        ]

    async def test_a_database_the_scheduler_never_booted_on_has_none(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await pglite_async_session.execute(
            text("DROP TABLE IF EXISTS job_results, jobs, schedules, tasks, metadata CASCADE")
        )
        await pglite_async_session.commit()

        assert await read_schedules(pglite_async_session) == []


class TestReadOutstandingJobNames:
    async def test_a_released_job_is_outstanding_under_its_schedule_id(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await create_scheduler_tables(pglite_async_session)
        await pglite_async_session.execute(
            text(
                "INSERT INTO tasks (id, func, job_executor, metadata) "
                "VALUES ('check-health', 'f', 'async', '{}')"
            )
        )
        for schedule_id in ("check-health", "check-health", None):
            await pglite_async_session.execute(
                text(
                    "INSERT INTO jobs (id, task_id, args, kwargs, schedule_id, executor, "
                    "metadata, created_at) VALUES (:id, 'check-health', '', '', :schedule, "
                    "'async', '{}', :now)"
                ),
                {"id": uuid.uuid4(), "schedule": schedule_id, "now": NOW},
            )
        await pglite_async_session.commit()

        assert await read_outstanding_job_names(pglite_async_session) == {"check-health"}

    async def test_a_database_the_scheduler_never_booted_on_has_none(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await pglite_async_session.execute(
            text("DROP TABLE IF EXISTS job_results, jobs, schedules, tasks, metadata CASCADE")
        )
        await pglite_async_session.commit()

        assert await read_outstanding_job_names(pglite_async_session) == set()
