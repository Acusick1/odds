from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.config import get_settings
from odds_core.models import Event, EventStatus
from odds_lambda.health_monitor import UNSETTLED_ALERT_MINUTES, HealthMonitor

from tests.test_helpers import make_epl_event

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def _event(
    event_id: str,
    kickoff: datetime,
    status: EventStatus = EventStatus.SCHEDULED,
    sport_key: str = "soccer_epl",
) -> Event:
    event = make_epl_event(event_id, kickoff, home_team=event_id, status=status)
    event.sport_key = sport_key
    return event


async def _check(session, now: datetime = NOW) -> tuple[list[str], AsyncMock]:
    monitor = HealthMonitor(session, get_settings())
    with patch.object(monitor, "_send_alert", new=AsyncMock(return_value=True)) as send:
        return await monitor.check_unsettled_events(now), send


class TestUnsettledEvents:
    @pytest.mark.asyncio
    async def test_alerts_naming_each_event_once_a_day(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("stuck-scheduled", NOW - timedelta(days=44)),
                _event("stuck-live", NOW - timedelta(days=2), EventStatus.LIVE),
            ]
        )
        await session.flush()

        issues, send = await _check(session)

        assert len(issues) == 1
        assert "stuck-scheduled" in issues[0] and "stuck-live" in issues[0]
        send.assert_awaited_once()
        kwargs = send.await_args.kwargs
        assert kwargs["alert_type"] == "unsettled_events:soccer_epl"
        assert kwargs["severity"] == "warning"
        assert kwargs["every"] == timedelta(minutes=UNSETTLED_ALERT_MINUTES) == timedelta(hours=24)
        assert kwargs["context"]["event_ids"] == ["stuck-scheduled", "stuck-live"]

    @pytest.mark.asyncio
    async def test_ignores_settled_recent_other_season_and_unchecked_sport_events(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("final", NOW - timedelta(days=3), EventStatus.FINAL),
                _event("postponed", NOW - timedelta(days=3), EventStatus.POSTPONED),
                _event("cancelled", NOW - timedelta(days=3), EventStatus.CANCELLED),
                _event("just-kicked-off", NOW - timedelta(hours=3)),
                _event("last-season", NOW - timedelta(days=200)),
                _event("nba", NOW - timedelta(days=3), sport_key="basketball_nba"),
            ]
        )
        await session.flush()

        issues, send = await _check(session)

        assert issues == []
        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_alerts_on_stuck_mlb_events_of_the_calendar_year(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        stuck = _event("mlb-april", datetime(2026, 4, 10, 23, 5, tzinfo=UTC), EventStatus.LIVE)
        stuck.sport_key = "baseball_mlb"
        last_year = _event("mlb-2025", datetime(2025, 9, 10, 23, 5, tzinfo=UTC))
        last_year.sport_key = "baseball_mlb"
        session.add_all([stuck, last_year])
        await session.flush()

        issues, send = await _check(session)

        assert len(issues) == 1 and "mlb-april" in issues[0]
        assert send.await_args.kwargs["alert_type"] == "unsettled_events:baseball_mlb"
        assert send.await_args.kwargs["context"]["event_ids"] == ["mlb-april"]

    @pytest.mark.asyncio
    async def test_ignores_unassignable_and_postponed_mlb_events(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        unassignable = _event("op_live_BOSSOX_TAMRAY_2026-05-17T1735", NOW - timedelta(days=150))
        unassignable.sport_key = "baseball_mlb"
        unassignable.no_game_pk_reason = "MLB has BOS at ATL that day"
        postponed = _event("mlb-postponed", NOW - timedelta(days=3), EventStatus.POSTPONED)
        postponed.sport_key = "baseball_mlb"
        session.add_all([unassignable, postponed])
        await session.flush()

        issues, send = await _check(session)

        assert issues == []
        send.assert_not_awaited()
