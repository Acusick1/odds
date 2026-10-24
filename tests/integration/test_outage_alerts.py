"""Outage alerts read the alert ledger the jobs write (#571)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.alerts import send_outage_warning, send_recorded_alert
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

    @pytest.mark.asyncio
    async def test_reminder_is_sent_after_the_interval(self, ledger, manager) -> None:
        await ledger(REBOOKED, timedelta(hours=7))

        assert await send_outage_warning(JOB, REBOOKED, "re-booked", now=NOW)

    @pytest.mark.asyncio
    async def test_other_jobs_do_not_end_the_outage(self, ledger, manager) -> None:
        await ledger(REBOOKED, timedelta(minutes=30))
        await ledger("heartbeat:fetch-scores-epl", timedelta(minutes=5))

        assert not await send_outage_warning(JOB, REBOOKED, "re-booked", now=NOW)


class TestSendRecordedAlert:
    EVERY = timedelta(hours=6)

    @pytest.fixture
    def recorded(self):
        record = AsyncMock()
        with patch("odds_core.alerts.record_to_alert_history", record):
            yield record

    @pytest.mark.asyncio
    async def test_a_failing_send_returns_false_and_records_nothing(self, ledger, recorded) -> None:
        manager = MagicMock(enabled=True, alert=AsyncMock(side_effect=RuntimeError("down")))
        with patch("odds_core.alerts.alert_manager", manager):
            sent = await send_recorded_alert("x", "warning", "m", every=self.EVERY, now=NOW)

        assert sent is False
        recorded.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_timed_out_send_returns_false_and_records_nothing(
        self, ledger, recorded
    ) -> None:
        async def hang(*_: object) -> None:
            await asyncio.sleep(10)

        manager = MagicMock(enabled=True, alert=hang)
        with (
            patch("odds_core.alerts.alert_manager", manager),
            patch("odds_core.alerts.ALERT_SEND_TIMEOUT_SECONDS", 0.01),
        ):
            sent = await send_recorded_alert("x", "warning", "m", every=self.EVERY, now=NOW)

        assert sent is False
        recorded.assert_not_called()

    @pytest.mark.asyncio
    async def test_quota_crossing_from_low_to_critical_sends_at_once(
        self, ledger, recorded
    ) -> None:
        await ledger("quota_low", timedelta(minutes=10))
        manager = MagicMock(enabled=True, alert=AsyncMock())
        with patch("odds_core.alerts.alert_manager", manager):
            sent = await send_recorded_alert(
                "quota_critical", "critical", "m", every=self.EVERY, now=NOW
            )

        assert sent is True
        manager.alert.assert_awaited_once_with("m", "critical")

    @pytest.mark.asyncio
    async def test_a_failed_insert_rolls_back_the_callers_session(self, ledger) -> None:
        session = MagicMock(rollback=AsyncMock())
        manager = MagicMock(enabled=True, alert=AsyncMock())
        with (
            patch("odds_core.alerts.alert_manager", manager),
            patch("odds_core.alerts.open_alerted_at", AsyncMock(return_value=None)),
            patch("odds_core.alerts.record_to_alert_history", AsyncMock(side_effect=RuntimeError)),
        ):
            sent = await send_recorded_alert(
                "x", "warning", "m", every=self.EVERY, session=session, now=NOW
            )

        assert sent is False
        session.rollback.assert_awaited_once()
