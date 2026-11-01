"""check-health pings an external dead-man's switch after each completed run (#489)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from odds_core.alerts import ping_dead_mans_switch
from odds_core.config import AlertConfig, SchedulerConfig, Settings
from odds_lambda.jobs import check_health
from odds_lambda.scheduling.jobs import JobContext


class PingServer:
    """Stands in for the external monitor: records each ping and answers with a set status."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.status = 200

    async def handle(self, request: web.Request) -> web.Response:
        self.paths.append(request.path)
        return web.Response(status=self.status, text="OK")


@pytest.fixture
async def monitor() -> AsyncIterator[tuple[PingServer, str]]:
    ping = PingServer()
    app = web.Application()
    app.router.add_route("*", "/{uuid}", ping.handle)
    server = TestServer(app)
    await server.start_server()
    try:
        yield ping, str(server.make_url("/check-uuid"))
    finally:
        await server.close()


def _settings(url: str | None, *, dry_run: bool = False) -> Settings:
    return Settings(
        alerts=AlertConfig(dead_mans_switch_url=url),
        scheduler=SchedulerConfig(dry_run=dry_run),
    )


class TestPingDeadMansSwitch:
    async def test_pings_the_configured_url(self, monitor: tuple[PingServer, str]) -> None:
        ping, url = monitor

        await ping_dead_mans_switch(_settings(url))

        assert ping.paths == ["/check-uuid"]

    @pytest.mark.parametrize("url", [None, ""])
    async def test_skips_when_unset(self, url: str | None) -> None:
        with patch("aiohttp.ClientSession") as session:
            await ping_dead_mans_switch(_settings(url))

        session.assert_not_called()

    async def test_skips_under_dry_run(self, monitor: tuple[PingServer, str]) -> None:
        ping, url = monitor

        await ping_dead_mans_switch(_settings(url, dry_run=True))

        assert ping.paths == []

    async def test_error_status_does_not_raise(self, monitor: tuple[PingServer, str]) -> None:
        ping, url = monitor
        ping.status = 500

        await ping_dead_mans_switch(_settings(url))

        assert ping.paths == ["/check-uuid"]

    async def test_unreachable_monitor_does_not_raise(self) -> None:
        await ping_dead_mans_switch(_settings("http://127.0.0.1:9/check-uuid"))


@pytest.fixture
def health_job() -> Iterator[SimpleNamespace]:
    settings = MagicMock()
    settings.scheduler.backend = "local"
    settings.scheduler.dry_run = False
    status = MagicMock(overall_healthy=True, issues_detected=[], alerts_sent=[])
    backend = MagicMock()
    backend.schedule_next_execution = AsyncMock()
    with (
        patch.object(check_health, "get_settings", return_value=settings),
        patch.object(check_health, "check_system_health", new_callable=AsyncMock) as check,
        patch.object(check_health, "get_scheduler_backend", return_value=backend),
        patch("odds_core.alerts.record_to_alert_history", new_callable=AsyncMock),
        patch("odds_core.alerts.send_job_failure", new_callable=AsyncMock),
        patch("odds_core.alerts.ping_dead_mans_switch", new_callable=AsyncMock) as ping,
    ):
        check.return_value = status
        yield SimpleNamespace(check=check, status=status, ping=ping, backend=backend)


class TestCheckHealthPing:
    async def test_completed_run_pings(self, health_job: SimpleNamespace) -> None:
        await check_health.main(JobContext())

        health_job.ping.assert_awaited_once()
        health_job.backend.schedule_next_execution.assert_awaited_once()

    async def test_run_that_finds_issues_still_pings(self, health_job: SimpleNamespace) -> None:
        health_job.status.overall_healthy = False
        health_job.status.issues_detected = ["stale_data:oddsportal:soccer_epl"]

        await check_health.main(JobContext())

        health_job.ping.assert_awaited_once()

    async def test_run_that_raises_does_not_ping(self, health_job: SimpleNamespace) -> None:
        health_job.check.side_effect = RuntimeError("database down")

        with pytest.raises(RuntimeError):
            await check_health.main(JobContext())

        health_job.ping.assert_not_awaited()
