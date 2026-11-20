"""The health checks on skipped candidates and on outcome names a source keeps leaving
unmatched."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.candidate_registry_models import CandidateRegistryCheck
from odds_core.config import Settings, get_settings
from odds_core.models import Event
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.candidate_registry_checks import record_registry_check
from odds_lambda.health_monitor import UNMATCHED_ALERT_TYPE, HealthMonitor
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_shaped_snapshot
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import make_epl_event

pytestmark = pytest.mark.integration

NOW = datetime.now(UTC).replace(microsecond=0)
EPL = "soccer_epl"
SERIE_A = "soccer_italy_serie_a"


def _settings(*, bootstrap: bool = True) -> Settings:
    settings = get_settings().model_copy(deep=True)
    jobs = [job for job in settings.scheduler.bootstrap_jobs if job != "run-candidates"]
    settings.scheduler.bootstrap_jobs = [*jobs, "run-candidates"] if bootstrap else jobs
    settings.data_collection.sports = [EPL, SERIE_A]
    return settings


async def _check_candidates(
    session: AsyncSession, settings: Settings
) -> list[CandidateRegistryCheck]:
    return await HealthMonitor(session, settings).check_candidate_registry()


class TestCandidateRegistryCheck:
    async def test_names_the_league_and_each_skipped_candidate(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await record_registry_check(
            session, EPL, NOW, {"bad-hash": "hash mismatch", "bad-rule": "no such rule"}, None
        )
        await record_registry_check(session, SERIE_A, NOW, {}, None)
        await session.commit()
        monitor = HealthMonitor(session, _settings())

        with patch.object(monitor, "_send_alert", new=AsyncMock(return_value=True)) as send:
            status = await monitor.check_system_health(NOW)

        assert not status.overall_healthy
        [issue] = [i for i in status.issues_detected if "run-candidates" in i]
        assert EPL in issue
        assert "bad-hash: hash mismatch" in issue
        assert "bad-rule: no such rule" in issue
        alert_types = [call.kwargs["alert_type"] for call in send.await_args_list]
        assert "candidates_invalid:bad-hash,bad-rule" in alert_types

    async def test_a_registry_error_is_reported(self, pglite_async_session: AsyncSession) -> None:
        session = pglite_async_session
        await record_registry_check(session, EPL, NOW, {}, "broken.yaml: bad YAML")
        await session.commit()

        [check] = await _check_candidates(session, _settings())

        assert check.registry_error == "broken.yaml: bad YAML"

    async def test_clean_runs_and_unscheduled_leagues_report_nothing(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await record_registry_check(session, EPL, NOW, {}, None)
        await record_registry_check(session, "soccer_efl_champ", NOW, {"x": "invalid"}, None)
        await session.commit()

        assert await _check_candidates(session, _settings()) == []

    async def test_reports_nothing_when_run_candidates_is_not_bootstrapped(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await record_registry_check(session, EPL, NOW, {"x": "invalid"}, None)
        await session.commit()

        assert await _check_candidates(session, _settings(bootstrap=False)) == []


def _envelope(away_name: str) -> dict[str, Any]:
    return {
        "bookmakers": [
            {
                "key": "bet365",
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Arsenal", "price_decimal": 2.0},
                            {"name": "Draw", "price_decimal": 3.4},
                            {"name": away_name, "price_decimal": 4.0},
                        ],
                    }
                ],
            }
        ]
    }


async def _scrape(
    session: AsyncSession, event_id: int, minutes_ago: int, away_name: str = "Chelsae"
) -> None:
    event = await session.get(Event, event_id)
    assert event is not None
    await store_shaped_snapshot(
        session,
        event,
        _envelope(away_name),
        NOW - timedelta(minutes=minutes_ago),
        SnapshotShape(Source.ODDSPORTAL, TimeBasis.OBSERVED, None),
    )
    await session.commit()


async def _add_event(session: AsyncSession, event_id: int, kickoff: datetime) -> None:
    session.add(make_epl_event(event_id, kickoff))
    await session.commit()


class TestPersistentUnmatchedOutcomes:
    async def test_an_outcome_unmatched_in_the_last_two_scrapes_is_reported(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await _add_event(session, 1, NOW + timedelta(hours=5))
        await _scrape(session, 1, 90)
        await _scrape(session, 1, 30)
        monitor = HealthMonitor(session, _settings())

        with patch.object(monitor, "_send_alert", new=AsyncMock(return_value=True)) as send:
            status = await monitor.check_system_health(NOW)

        assert not status.overall_healthy
        [issue] = [i for i in status.issues_detected if "unmatched" in i]
        assert "oddsportal" in issue
        assert "Arsenal v Chelsea" in issue
        assert "'Chelsae'" in issue
        [call] = [c for c in send.await_args_list if c.kwargs["alert_type"] == UNMATCHED_ALERT_TYPE]
        assert call.kwargs["context"] == {"event_ids": [1]}

    async def test_a_later_scrape_that_matches_clears_it(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await _add_event(session, 1, NOW + timedelta(hours=5))
        await _scrape(session, 1, 150)
        await _scrape(session, 1, 90)
        await _scrape(session, 1, 30, away_name="Chelsea")

        assert (
            await HealthMonitor(session, _settings()).check_persistent_unmatched_outcomes(NOW) == []
        )

    async def test_one_scrape_or_a_started_event_reports_nothing(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await _add_event(session, 1, NOW + timedelta(hours=5))
        await _scrape(session, 1, 30)
        await _add_event(session, 2, NOW - timedelta(minutes=10))
        await _scrape(session, 2, 90)
        await _scrape(session, 2, 30)

        assert (
            await HealthMonitor(session, _settings()).check_persistent_unmatched_outcomes(NOW) == []
        )
