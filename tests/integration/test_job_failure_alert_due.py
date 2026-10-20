"""job_failure_alert_due reads the alert ledger the jobs write (#571)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from odds_core.alerts import job_failure_alert_due
from odds_core.models import AlertHistory

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


@pytest.fixture
def ledger(test_session, mock_session_factory):
    async def add(alert_type: str, age: timedelta) -> None:
        test_session.add(
            AlertHistory(alert_type=alert_type, severity="info", message="m", sent_at=NOW - age)
        )
        await test_session.commit()

    with patch("odds_core.database.async_session_maker", mock_session_factory):
        yield add


class TestJobFailureAlertDue:
    @pytest.mark.asyncio
    async def test_no_history_alerts(self, ledger) -> None:
        assert await job_failure_alert_due("fetch-odds-epl", NOW)

    @pytest.mark.asyncio
    async def test_failure_without_a_later_heartbeat_is_suppressed(self, ledger) -> None:
        await ledger("job_failure:fetch-odds-epl", timedelta(minutes=30))
        await ledger("heartbeat:fetch-odds-epl", timedelta(hours=3))

        assert not await job_failure_alert_due("fetch-odds-epl", NOW)

    @pytest.mark.asyncio
    async def test_heartbeat_after_the_failure_ends_the_outage(self, ledger) -> None:
        await ledger("job_failure:fetch-odds-epl", timedelta(hours=2))
        await ledger("heartbeat:fetch-odds-epl", timedelta(hours=1))

        assert await job_failure_alert_due("fetch-odds-epl", NOW)

    @pytest.mark.asyncio
    async def test_other_jobs_do_not_interfere(self, ledger) -> None:
        await ledger("job_failure:fetch-scores-epl", timedelta(minutes=30))

        assert await job_failure_alert_due("fetch-odds-epl", NOW)
