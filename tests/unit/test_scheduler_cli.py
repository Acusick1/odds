"""Unit tests for scheduler CLI rendering helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from odds_cli.commands.scheduler import _print_scheduled_jobs
from odds_lambda.scheduling.backends.base import JobStatus, ScheduledJob
from rich.console import Console


def _render(jobs: list[ScheduledJob]) -> str:
    console = Console(record=True, width=200)
    _print_scheduled_jobs(console, jobs)
    return console.export_text()


class TestPrintScheduledJobs:
    """The Schedule column appears only when a job carries a raw expression."""

    def test_schedule_column_shown_when_expression_present(self) -> None:
        jobs = [
            ScheduledJob(
                job_name="fetch-odds-epl",
                next_run_time=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
                status=JobStatus.SCHEDULED,
                schedule_expression="cron(0 14 1 6 ? 2026)",
            )
        ]
        out = _render(jobs)
        assert "Schedule" in out
        assert "cron(0 14 1 6 ? 2026)" in out

    def test_schedule_column_hidden_when_no_expression(self) -> None:
        jobs = [
            ScheduledJob(
                job_name="agent-run",
                next_run_time=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
                status=JobStatus.SCHEDULED,
            )
        ]
        out = _render(jobs)
        assert "Schedule" not in out

    def test_empty_jobs_message(self) -> None:
        out = _render([])
        assert "No jobs currently scheduled" in out
