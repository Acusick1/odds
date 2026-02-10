"""Tests for fetch_polymarket job orchestration (main function)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from freezegun import freeze_time
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.jobs import fetch_polymarket
from odds_lambda.polymarket_ingestion import PolymarketIngestionResult

FROZEN_NOW = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    enabled: bool = True,
    price_poll_interval: int = 300,
    dry_run: bool = False,
    alert_enabled: bool = False,
) -> MagicMock:
    s = MagicMock()
    s.polymarket.enabled = enabled
    s.polymarket.price_poll_interval = price_poll_interval
    s.scheduler.dry_run = dry_run
    s.scheduler.backend = "local"
    s.alerts.alert_enabled = alert_enabled
    return s


def _make_async_session() -> AsyncMock:
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.__aexit__.return_value = None
    session.commit = AsyncMock()
    return session


def _make_active_event(hours_until: float) -> MagicMock:
    return MagicMock(
        id=1,
        pm_event_id="event-123",
        title="Lakers vs Celtics",
        start_date=datetime.now(UTC) + timedelta(hours=hours_until),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class JobMocks:
    """Mocked dependencies for fetch_polymarket.main()."""

    def __init__(self) -> None:
        self.client: AsyncMock = AsyncMock()
        self.client.__aenter__.return_value = self.client
        self.client.__aexit__.return_value = None

        self.main_session: AsyncMock = _make_async_session()
        self.log_session: AsyncMock = _make_async_session()

        self.reader: AsyncMock = AsyncMock()
        self.writer: AsyncMock = AsyncMock()
        self.log_writer: AsyncMock = AsyncMock()

        self.service: MagicMock = MagicMock()
        self.service.discover_and_upsert_events = AsyncMock()
        self.service.collect_snapshots = AsyncMock()

        self.backend: MagicMock = MagicMock()
        self.backend.schedule_next_execution = AsyncMock()
        self.backend.get_backend_name.return_value = "local"

        self.settings: MagicMock = _make_settings()


@pytest.fixture
def job():
    """Patch all fetch_polymarket.main() dependencies with frozen time."""
    m = JobMocks()
    with (
        freeze_time(FROZEN_NOW),
        patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=m.settings),
        patch("odds_lambda.jobs.fetch_polymarket.PolymarketClient", return_value=m.client),
        patch(
            "odds_lambda.jobs.fetch_polymarket.async_session_maker",
            side_effect=[m.main_session, m.log_session],
        ),
        patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=m.reader),
        patch(
            "odds_lambda.jobs.fetch_polymarket.PolymarketWriter",
            side_effect=[m.writer, m.log_writer],
        ),
        patch(
            "odds_lambda.jobs.fetch_polymarket.build_ingestion_service",
            return_value=m.service,
        ),
        patch(
            "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
            return_value=m.backend,
        ),
    ):
        yield m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchPolymarketMain:
    # -- gating / early returns --

    @pytest.mark.asyncio
    async def test_early_return_when_polymarket_disabled(self, job: JobMocks) -> None:
        job.settings.polymarket.enabled = False
        await fetch_polymarket.main()
        job.client.get_nba_events.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_api_events_schedules_from_db(self, job: JobMocks) -> None:
        db_event = _make_active_event(hours_until=6)
        job.client.get_nba_events.return_value = []
        job.reader.get_active_events.return_value = [db_event]

        await fetch_polymarket.main()

        job.backend.schedule_next_execution.assert_called_once()
        next_time = job.backend.schedule_next_execution.call_args.kwargs["next_time"]
        assert next_time == FROZEN_NOW + timedelta(seconds=300)

    @pytest.mark.asyncio
    async def test_no_api_events_no_db_events_schedules_daily(self, job: JobMocks) -> None:
        job.client.get_nba_events.return_value = []
        job.reader.get_active_events.return_value = []

        await fetch_polymarket.main()

        job.backend.schedule_next_execution.assert_called_once()
        next_time = job.backend.schedule_next_execution.call_args.kwargs["next_time"]
        assert next_time == FROZEN_NOW + timedelta(hours=24)

    # -- happy path delegates to service --

    @pytest.mark.asyncio
    async def test_happy_path_calls_service(self, job: JobMocks) -> None:
        job.client.get_nba_events.return_value = [{"id": "ev-1"}]

        discovery_result = PolymarketIngestionResult(events_processed=1, markets_discovered=2)
        snapshot_result = PolymarketIngestionResult(
            price_snapshots_stored=3, fetch_tier=FetchTier.PREGAME
        )
        job.service.discover_and_upsert_events.return_value = discovery_result
        job.service.collect_snapshots.return_value = snapshot_result

        await fetch_polymarket.main()

        job.service.discover_and_upsert_events.assert_called_once_with([{"id": "ev-1"}])
        job.service.collect_snapshots.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_active_events_after_upsert_schedules_daily(self, job: JobMocks) -> None:
        """When collect_snapshots returns no fetch_tier, schedule daily."""
        job.client.get_nba_events.return_value = [{"id": "ev-1"}]

        discovery_result = PolymarketIngestionResult(events_processed=1)
        snapshot_result = PolymarketIngestionResult(fetch_tier=None)
        job.service.discover_and_upsert_events.return_value = discovery_result
        job.service.collect_snapshots.return_value = snapshot_result

        await fetch_polymarket.main()

        job.backend.schedule_next_execution.assert_called_once()
        next_time = job.backend.schedule_next_execution.call_args.kwargs["next_time"]
        assert next_time == FROZEN_NOW + timedelta(hours=24)

    # -- fetch logging --

    @pytest.mark.asyncio
    async def test_fetch_log_written_on_success(self, job: JobMocks) -> None:
        job.client.get_nba_events.return_value = [{"id": "ev-1"}]

        discovery_result = PolymarketIngestionResult(events_processed=1, markets_discovered=2)
        snapshot_result = PolymarketIngestionResult(
            price_snapshots_stored=1, fetch_tier=FetchTier.PREGAME
        )
        job.service.discover_and_upsert_events.return_value = discovery_result
        job.service.collect_snapshots.return_value = snapshot_result

        await fetch_polymarket.main()

        job.log_writer.log_fetch.assert_called_once()
        log_arg = job.log_writer.log_fetch.call_args.args[0]
        assert log_arg.success is True
        assert log_arg.error_message is None
        assert log_arg.events_count == 1
        assert log_arg.snapshots_stored == 1
        job.log_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_log_written_on_failure(self, job: JobMocks) -> None:
        job.client.get_nba_events.side_effect = Exception("API down")

        with pytest.raises(Exception, match="API down"):
            await fetch_polymarket.main()

        job.log_writer.log_fetch.assert_called_once()
        log_arg = job.log_writer.log_fetch.call_args.args[0]
        assert log_arg.success is False
        assert "API down" in log_arg.error_message
        job.log_session.commit.assert_called_once()

    # -- scheduling --

    @pytest.mark.asyncio
    async def test_self_schedules_on_success(self, job: JobMocks) -> None:
        job.client.get_nba_events.return_value = [{"id": "ev-1"}]

        discovery_result = PolymarketIngestionResult(events_processed=1)
        snapshot_result = PolymarketIngestionResult(
            price_snapshots_stored=1, fetch_tier=FetchTier.PREGAME
        )
        job.service.discover_and_upsert_events.return_value = discovery_result
        job.service.collect_snapshots.return_value = snapshot_result

        await fetch_polymarket.main()

        job.backend.schedule_next_execution.assert_called_once()
        next_time = job.backend.schedule_next_execution.call_args.kwargs["next_time"]
        assert next_time == FROZEN_NOW + timedelta(seconds=300)

    @pytest.mark.asyncio
    async def test_scheduling_failure_does_not_fail_job(self, job: JobMocks) -> None:
        job.client.get_nba_events.return_value = [{"id": "ev-1"}]

        discovery_result = PolymarketIngestionResult(events_processed=1)
        snapshot_result = PolymarketIngestionResult(
            price_snapshots_stored=1, fetch_tier=FetchTier.PREGAME
        )
        job.service.discover_and_upsert_events.return_value = discovery_result
        job.service.collect_snapshots.return_value = snapshot_result
        job.backend.schedule_next_execution.side_effect = Exception("Scheduler unreachable")

        await fetch_polymarket.main()  # must not raise
