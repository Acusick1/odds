"""Outage alerts read the alert ledger the jobs write (#571)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.alerts import outage_alert_due, send_outage_warning
from odds_core.models import AlertHistory

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
JOB = "fetch-oddsportal-results-epl"
FAILURE = f"job_failure:{JOB}"
REBOOKED = f"job_chain_rebooked:{JOB}"


@pytest.fixture
def ledger(test_session, mock_session_factory):
    async def add(alert_type: str, age: timedelta) -> None:
        test_session.add(
            AlertHistory(alert_type=alert_type, severity="info", message="m", sent_at=NOW - age)
        )
        await test_session.commit()

    with patch("odds_core.database.async_session_maker", mock_session_factory):
        yield add


class TestOutageAlertDue:
    @pytest.mark.asyncio
    async def test_no_history_alerts(self, ledger) -> None:
        assert await outage_alert_due(JOB, FAILURE, NOW)

    @pytest.mark.asyncio
    async def test_alert_within_the_reminder_interval_is_suppressed(self, ledger) -> None:
        await ledger(FAILURE, timedelta(minutes=30))
        await ledger(f"heartbeat:{JOB}", timedelta(hours=3))

        assert not await outage_alert_due(JOB, FAILURE, NOW)

    @pytest.mark.asyncio
    async def test_heartbeat_after_the_alert_ends_the_outage(self, ledger) -> None:
        await ledger(FAILURE, timedelta(hours=2))
        await ledger(f"heartbeat:{JOB}", timedelta(hours=1))

        assert await outage_alert_due(JOB, FAILURE, NOW)

    @pytest.mark.asyncio
    async def test_reminder_is_due_after_the_interval(self, ledger) -> None:
        await ledger(FAILURE, timedelta(hours=7))

        assert await outage_alert_due(JOB, FAILURE, NOW)

    @pytest.mark.asyncio
    async def test_other_jobs_do_not_interfere(self, ledger) -> None:
        await ledger("job_failure:fetch-scores-epl", timedelta(minutes=30))

        assert await outage_alert_due(JOB, FAILURE, NOW)


class TestSendOutageWarning:
    @pytest.fixture
    def manager(self):
        manager = MagicMock(enabled=True, alert=AsyncMock())
        with (
            patch("odds_core.alerts.alert_manager", manager),
            patch("odds_core.alerts.record_to_alert_history", AsyncMock()),
        ):
            yield manager

    @pytest.mark.asyncio
    async def test_persistent_failure_warns_once_not_every_half_hour(self, ledger, manager) -> None:
        await ledger(REBOOKED, timedelta(minutes=45))

        sent = await send_outage_warning(JOB, REBOOKED, "re-booked", now=NOW)

        assert not sent
        manager.alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_first_warning_of_an_outage_is_sent(self, ledger, manager) -> None:
        await ledger(REBOOKED, timedelta(hours=3))
        await ledger(f"heartbeat:{JOB}", timedelta(hours=1))

        sent = await send_outage_warning(JOB, REBOOKED, "re-booked", now=NOW)

        assert sent
        manager.alert.assert_awaited_once_with("re-booked", "warning")
