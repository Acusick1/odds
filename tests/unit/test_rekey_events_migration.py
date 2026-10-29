"""Tests for the 75c92251aa25 migration: events re-keyed to the integer id, string ids dropped."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from odds_core.match_brief_models import BriefDecision, MatchBrief
from odds_core.models import (
    AlertHistory,
    DataQualityLog,
    Event,
    EventExternalId,
    EventStatus,
    OddsSnapshot,
)
from odds_core.vocabulary import Market, Source, TimeBasis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "75c92251aa25_rekey_events_to_integer_id.py"
    )
    spec = importlib.util.spec_from_file_location("m75c92251aa25", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(connection: Any, migration: ModuleType, step: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, step)()


async def _step(session: AsyncSession, migration: ModuleType, step: str) -> None:
    connection = await session.connection()
    await connection.run_sync(lambda c: _run(c, migration, step))


async def _rows(session: AsyncSession, query: str) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in (await session.execute(text(query))).all()]


def _event(event_id: int | None, home: str) -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=KICKOFF,
        home_team=home,
        away_team="Chelsea",
        status=EventStatus.SCHEDULED,
    )


@pytest.fixture
async def session(pglite_async_session: AsyncSession, migration: ModuleType) -> AsyncSession:
    """Integer-keyed events 1 to 4 with rows referencing them; events 1 and 2 have the string
    ids an upgrade kept, 3 holds an Odds API id and 4 nothing."""
    session = pglite_async_session
    session.add_all([_event(1, "Arsenal"), _event(2, "Brentford"), _event(3, "Burnley")])
    session.add(_event(4, "Everton"))
    await session.flush()
    session.add_all(
        [
            EventExternalId(source=Source.ODDS_API, external_id="abc0123", event_id=2),
            EventExternalId(source=Source.ODDS_API, external_id="def4567", event_id=3),
            OddsSnapshot(
                event_id=1,
                snapshot_time=KICKOFF - timedelta(hours=1),
                source=Source.BETFAIR,
                market=Market.ONE_X_TWO,
                time_basis=TimeBasis.OBSERVED,
            ),
            MatchBrief(event_id=3, decision=BriefDecision.WATCHING, summary="s", brief_text="t"),
            DataQualityLog(event_id=None, severity="warning", issue_type="x", description="d"),
            DataQualityLog(event_id=4, severity="warning", issue_type="x", description="d"),
            AlertHistory(
                alert_type="unsettled_events:soccer_epl",
                severity="warning",
                message="m",
                context={"sport_key": "soccer_epl", "event_ids": [1, 3, 99, "op_gone"]},
            ),
        ]
    )
    await session.flush()
    await session.execute(text(migration.LEGACY_TABLE_DDL))
    await session.execute(
        text(
            "INSERT INTO legacy_event_ids (event_id, legacy_id) VALUES"
            " (1, 'op_live_ARS_CHE_2026-11-07T1500'), (2, 'abc0123')"
        )
    )
    session.expunge_all()
    return session


class TestDowngrade:
    async def test_restores_the_string_ids(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        await _step(session, migration, "downgrade")

        assert await _rows(
            session, "SELECT surrogate_id, id FROM events ORDER BY surrogate_id"
        ) == [
            (1, "op_live_ARS_CHE_2026-11-07T1500"),
            (2, "abc0123"),
            (3, "def4567"),
            (4, "event_4"),
        ]
        assert await _rows(session, "SELECT event_id FROM odds_snapshots") == [
            ("op_live_ARS_CHE_2026-11-07T1500",)
        ]
        assert await _rows(session, "SELECT event_id FROM match_briefs") == [("def4567",)]
        assert await _rows(
            session, "SELECT event_id FROM data_quality_logs ORDER BY event_id NULLS FIRST"
        ) == [(None,), ("event_4",)]
        assert await _rows(session, "SELECT event_id FROM event_external_ids ORDER BY 1") == [
            (2,),
            (3,),
        ]
        context = (await session.execute(text("SELECT context FROM alert_history"))).scalar_one()
        assert context["event_ids"] == ["op_live_ARS_CHE_2026-11-07T1500", "def4567", 99, "op_gone"]
        assert await _rows(session, "SELECT to_regclass('legacy_event_ids')") == [(None,)]


class TestUpgrade:
    async def test_rekeys_every_reference_to_the_integer_id(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        await _step(session, migration, "downgrade")

        await _step(session, migration, "upgrade")

        assert await _rows(session, "SELECT id FROM events ORDER BY id") == [(1,), (2,), (3,), (4,)]
        assert await _rows(session, "SELECT event_id FROM odds_snapshots") == [(1,)]
        assert await _rows(session, "SELECT event_id FROM match_briefs") == [(3,)]
        assert await _rows(
            session, "SELECT event_id FROM data_quality_logs ORDER BY event_id NULLS FIRST"
        ) == [(None,), (4,)]
        context = (await session.execute(text("SELECT context FROM alert_history"))).scalar_one()
        assert context["event_ids"] == [1, 3, 99, "op_gone"]
        assert await _rows(
            session, "SELECT event_id, legacy_id FROM legacy_event_ids ORDER BY event_id"
        ) == [
            (1, "op_live_ARS_CHE_2026-11-07T1500"),
            (2, "abc0123"),
            (3, "def4567"),
            (4, "event_4"),
        ]

    async def test_new_events_are_numbered_after_the_kept_ids(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        await _step(session, migration, "downgrade")
        await _step(session, migration, "upgrade")

        created = _event(None, "Fulham")
        session.add(created)
        await session.flush()

        assert created.id is not None and created.id > 4

    async def test_records_an_odds_api_key_as_the_events_external_id(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        await _step(session, migration, "downgrade")
        await session.execute(
            text(
                "INSERT INTO events (id, sport_key, sport_title, commence_time, home_team,"
                " away_team, status, created_at, updated_at)"
                " VALUES ('fedcba98', 'soccer_epl', 'EPL', :k, 'Leeds', 'Chelsea', 'SCHEDULED',"
                " now(), now())"
            ),
            {"k": KICKOFF},
        )

        await _step(session, migration, "upgrade")

        assert await _rows(
            session,
            "SELECT x.source, x.external_id FROM event_external_ids x"
            " JOIN events e ON e.id = x.event_id WHERE e.home_team = 'Leeds'",
        ) == [("odds_api", "fedcba98")]

    async def test_built_ids_are_not_recorded_as_external_ids(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        await _step(session, migration, "downgrade")

        await _step(session, migration, "upgrade")

        assert await _rows(
            session, "SELECT event_id, external_id FROM event_external_ids ORDER BY 1"
        ) == [(2, "abc0123"), (3, "def4567")]


class TestUpgradeGuards:
    async def test_refuses_an_odds_api_id_another_event_holds(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        await _step(session, migration, "downgrade")
        await session.execute(
            text(
                "INSERT INTO events (id, sport_key, sport_title, commence_time, home_team,"
                " away_team, status, created_at, updated_at)"
                " VALUES ('fedcba98', 'soccer_epl', 'EPL', :k, 'Leeds', 'Chelsea', 'SCHEDULED',"
                " now(), now())"
            ),
            {"k": KICKOFF},
        )
        await session.execute(
            text(
                "INSERT INTO event_external_ids (source, external_id, event_id, created_at)"
                " VALUES ('odds_api', 'fedcba98', 1, now())"
            )
        )

        with pytest.raises(RuntimeError, match="Odds API id is not their odds_api external id"):
            await _step(session, migration, "upgrade")

    async def test_refuses_an_unknown_foreign_key_to_events(
        self, migration: ModuleType, session: AsyncSession
    ) -> None:
        await _step(session, migration, "downgrade")
        await session.execute(text("CREATE TABLE stray (event_id VARCHAR REFERENCES events (id))"))

        with pytest.raises(RuntimeError, match="stray"):
            await _step(session, migration, "upgrade")


class TestCheckRekeyed:
    def test_a_changed_reference_fails(self, migration: ModuleType) -> None:
        with pytest.raises(RuntimeError, match="odds_snapshots"):
            migration._check_rekeyed({"odds_snapshots": (2, 2, 3)}, {"odds_snapshots": (2, 2, 4)})

    def test_unchanged_references_pass(self, migration: ModuleType) -> None:
        migration._check_rekeyed({"odds_snapshots": (2, 2, 3)}, {"odds_snapshots": (2, 2, 3)})
