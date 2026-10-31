"""fetch-odds and fetch-scores report through job_alert_context and cannot burn quota on retry (#571)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.config import AlertConfig
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.jobs import fetch_odds, fetch_scores
from odds_lambda.scheduling.decision import ScheduleDecision
from odds_lambda.scheduling.jobs import JobContext
from odds_lambda.scheduling.runner import CHAIN_RETRY_DELAY, run_chain_job

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
EPL = "soccer_epl"


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.scheduler.backend = "local"
    settings.scheduler.dry_run = False
    settings.scheduler.lookahead_days = 7
    settings.data_collection.sports = [EPL]
    settings.alerts.quota_critical_threshold = 0.1
    settings.alerts.quota_warning_threshold = 0.2
    return settings


def _decision(*, should_execute: bool) -> ScheduleDecision:
    return ScheduleDecision(
        should_execute=should_execute,
        reason="test",
        next_execution=NOW + timedelta(minutes=30),
        tier=FetchTier.CLOSING if should_execute else None,
    )


class FakeBackend:
    """Stands in for the scheduler store: records the run the chain has booked."""

    def __init__(self) -> None:
        self.booked: datetime | None = None
        self.schedule_next_execution = AsyncMock(side_effect=self._book)

    async def _book(self, job_name: str, next_time: datetime) -> None:
        self.booked = next_time

    async def get_one_off_run(self, job_name: str) -> datetime | None:
        return self.booked


@pytest.fixture
def alerts() -> Iterator[SimpleNamespace]:
    with (
        patch("odds_core.alerts.send_job_failure", new_callable=AsyncMock) as failure,
        patch("odds_core.alerts.send_critical", new_callable=AsyncMock) as critical,
        patch("odds_core.alerts.record_to_alert_history", new_callable=AsyncMock) as record,
    ):
        yield SimpleNamespace(failure=failure, critical=critical, record=record)


def _heartbeats(alerts: SimpleNamespace) -> list[str]:
    return [call.kwargs["alert_type"] for call in alerts.record.call_args_list]


@pytest.fixture
def odds_job() -> Iterator[SimpleNamespace]:
    backend = FakeBackend()
    ingestion = MagicMock()
    ingestion.ingest_sports = AsyncMock()
    client = AsyncMock()
    client.__aenter__.return_value = client

    async def book(**kwargs: Any) -> None:
        await backend.schedule_next_execution(kwargs["job_name"], kwargs["next_time"])

    with (
        patch.object(fetch_odds, "get_settings", return_value=_settings()),
        patch.object(fetch_odds, "TheOddsAPIClient", return_value=client),
        patch.object(fetch_odds, "build_event_sync_service") as sync,
        patch.object(fetch_odds, "build_ingestion_service", return_value=ingestion),
        patch.object(fetch_odds, "self_schedule", side_effect=book),
        patch.object(fetch_odds, "decide_forward_resilient", new_callable=AsyncMock) as decide,
    ):
        sync.return_value.sync_sports = AsyncMock(return_value=[])
        decide.return_value = _decision(should_execute=True)
        yield SimpleNamespace(backend=backend, ingestion=ingestion, decide=decide)


@pytest.mark.usefixtures("alerts")
class TestHeartbeatExpectations:
    def test_cover_the_longest_gap_of_each_job(self) -> None:
        expectations = AlertConfig().heartbeat_expectations

        assert expectations["fetch-odds-epl"] > fetch_odds.CADENCE.opening
        assert expectations["fetch-scores-epl"] > fetch_scores.SCORES_IDLE_INTERVAL


class TestFetchOdds:
    @pytest.mark.asyncio
    async def test_failure_sends_the_job_failure_alert_only(
        self, odds_job: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        odds_job.ingestion.ingest_sports.side_effect = RuntimeError("401 from the API")

        with pytest.raises(RuntimeError, match="401"):
            await fetch_odds.main(JobContext(sport=EPL))

        alerts.failure.assert_awaited_once()
        assert alerts.failure.call_args.args[0] == "fetch-odds-epl"
        alerts.critical.assert_not_called()
        assert _heartbeats(alerts) == []

    @pytest.mark.asyncio
    async def test_success_records_a_heartbeat(
        self, odds_job: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        odds_job.ingestion.ingest_sports.return_value = SimpleNamespace(
            sport_results=[], total_processed=0, total_events=0, total_failures=0
        )

        await fetch_odds.main(JobContext(sport=EPL))

        assert _heartbeats(alerts) == ["heartbeat:fetch-odds-epl"]

    @pytest.mark.asyncio
    async def test_gated_run_still_records_a_heartbeat(
        self, odds_job: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        odds_job.decide.return_value = _decision(should_execute=False)

        await fetch_odds.main(JobContext(sport=EPL))

        odds_job.ingestion.ingest_sports.assert_not_called()
        assert _heartbeats(alerts) == ["heartbeat:fetch-odds-epl"]

    @pytest.mark.asyncio
    async def test_failed_run_keeps_its_pre_booked_run_instead_of_a_15_minute_retry(
        self, odds_job: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        odds_job.ingestion.ingest_sports.side_effect = RuntimeError("API down")
        booked_by_job = NOW + timedelta(minutes=30)
        odds_job.decide.return_value = ScheduleDecision(
            should_execute=True, reason="t", next_execution=booked_by_job, tier=FetchTier.CLOSING
        )

        with pytest.raises(RuntimeError):
            await run_chain_job(
                odds_job.backend,  # type: ignore[arg-type]
                "fetch-odds-epl",
                lambda: fetch_odds.main(JobContext(sport=EPL)),
                now=lambda: NOW,
            )

        assert odds_job.backend.booked == booked_by_job
        assert odds_job.backend.booked > NOW + CHAIN_RETRY_DELAY
        odds_job.backend.schedule_next_execution.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_pre_schedule_warns_and_failed_post_booking_is_final(
        self, odds_job: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        odds_job.ingestion.ingest_sports.return_value = SimpleNamespace(
            sport_results=[], total_processed=0, total_events=0, total_failures=0
        )

        with (
            patch.object(fetch_odds, "self_schedule", side_effect=RuntimeError("no backend")),
            patch.object(fetch_odds, "send_scheduling_failure", new_callable=AsyncMock) as failure,
        ):
            await fetch_odds.main(JobContext(sport=EPL))

        assert [c.args[0] for c in failure.call_args_list] == ["fetch-odds-epl"] * 2
        assert [c.kwargs["final"] for c in failure.call_args_list] == [False, True]

    @pytest.mark.asyncio
    async def test_failed_only_booking_of_a_skipped_run_is_final(
        self, odds_job: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        odds_job.decide.return_value = _decision(should_execute=False)

        with (
            patch.object(fetch_odds, "self_schedule", side_effect=RuntimeError("no backend")),
            patch.object(fetch_odds, "send_scheduling_failure", new_callable=AsyncMock) as failure,
        ):
            await fetch_odds.main(JobContext(sport=EPL))

        assert failure.call_args_list[0].kwargs["final"] is True

    @pytest.mark.asyncio
    async def test_failed_post_booking_is_not_final_when_the_pre_schedule_stands(
        self, odds_job: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        odds_job.ingestion.ingest_sports.return_value = SimpleNamespace(
            sport_results=[], total_processed=0, total_events=0, total_failures=0
        )

        with (
            patch.object(
                fetch_odds, "self_schedule", side_effect=[None, RuntimeError("no backend")]
            ),
            patch.object(fetch_odds, "send_scheduling_failure", new_callable=AsyncMock) as failure,
        ):
            await fetch_odds.main(JobContext(sport=EPL))

        failure.assert_awaited_once()
        assert failure.call_args.kwargs["final"] is False


class TestFetchScores:
    @pytest.fixture
    def scores_job(self) -> Iterator[SimpleNamespace]:
        with (
            patch.object(fetch_scores, "get_settings", return_value=_settings()),
            patch.object(fetch_scores, "self_schedule", new_callable=AsyncMock),
            patch.object(fetch_scores, "_scores_decision", new_callable=AsyncMock) as decide,
            patch.object(fetch_scores, "_fetch_and_update_scores", new_callable=AsyncMock) as work,
        ):
            decide.return_value = _decision(should_execute=True)
            yield SimpleNamespace(decide=decide, work=work)

    @pytest.mark.asyncio
    async def test_failure_sends_the_job_failure_alert_only(
        self, scores_job: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        scores_job.work.side_effect = RuntimeError("API down")

        with pytest.raises(RuntimeError):
            await fetch_scores.main(JobContext(sport=EPL))

        alerts.failure.assert_awaited_once()
        assert alerts.failure.call_args.args[0] == "fetch-scores-epl"
        alerts.critical.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_and_gated_runs_record_a_heartbeat(
        self, scores_job: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        await fetch_scores.main(JobContext(sport=EPL))
        scores_job.decide.return_value = _decision(should_execute=False)
        await fetch_scores.main(JobContext(sport=EPL))

        assert _heartbeats(alerts) == ["heartbeat:fetch-scores-epl"] * 2
