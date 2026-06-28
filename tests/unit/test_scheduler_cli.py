"""Unit tests for scheduler CLI rendering helpers and the smoke command."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import typer
from odds_cli.commands.scheduler import _print_scheduled_jobs, smoke
from odds_lambda.scheduling.backends.base import JobStatus, ScheduledJob
from rich.console import Console
from typer.testing import CliRunner

runner = CliRunner()


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


def _settings(
    bootstrap_jobs: list[str],
    sports: list[str],
    *,
    dry_run: bool = True,
    backend: str = "local",
) -> MagicMock:
    """Build a settings double controlling the job-set derivation inputs."""
    settings = MagicMock()
    settings.scheduler.bootstrap_jobs = bootstrap_jobs
    settings.scheduler.dry_run = dry_run
    settings.scheduler.backend = backend
    settings.data_collection.sports = sports
    return settings


class _JobRecorder:
    """Stand-in for ``get_job_function`` that records runs without touching the DB.

    Returns an async job that appends ``(base_name, ctx.sport)`` on execution and
    raises for any base name in ``fail_bases`` (to exercise failure exit codes).
    """

    def __init__(self, fail_bases: set[str] | None = None) -> None:
        self.fail_bases = fail_bases or set()
        self.ran: list[tuple[str, str | None]] = []

    def __call__(self, base_name: str):
        recorder = self

        async def _job(ctx) -> None:
            recorder.ran.append((base_name, ctx.sport))
            if base_name in recorder.fail_bases:
                raise RuntimeError(f"{base_name} boom")

        return _job

    @property
    def ran_bases(self) -> set[str]:
        return {base for base, _ in self.ran}


def _invoke_smoke(args: list[str], settings: MagicMock, recorder: _JobRecorder):
    """Invoke the smoke command in isolation with patched settings and job runner."""
    test_app = typer.Typer()
    test_app.command()(smoke)
    with (
        patch("odds_cli.commands.scheduler.get_settings", return_value=settings),
        patch("odds_lambda.scheduling.jobs.get_job_function", new=recorder),
    ):
        return runner.invoke(test_app, args)


class TestSmokeCommand:
    """Behaviour of ``odds scheduler smoke`` job selection and exit codes."""

    def test_default_run_skips_agent_run(self) -> None:
        settings = _settings(["fetch-oddsportal", "agent-run"], ["soccer_epl"])
        recorder = _JobRecorder()
        result = _invoke_smoke([], settings, recorder)

        assert result.exit_code == 0
        assert "fetch-oddsportal" in recorder.ran_bases
        assert "agent-run" not in recorder.ran_bases
        assert "SKIP" in result.output

    def test_include_agent_runs_agent(self) -> None:
        settings = _settings(["fetch-oddsportal", "agent-run"], ["soccer_epl"])
        recorder = _JobRecorder()
        result = _invoke_smoke(["--include-agent"], settings, recorder)

        assert result.exit_code == 0
        assert "agent-run" in recorder.ran_bases

    def test_daily_digest_runs_by_default(self) -> None:
        settings = _settings(["fetch-oddsportal", "daily-digest"], ["soccer_epl"])
        recorder = _JobRecorder()
        result = _invoke_smoke([], settings, recorder)

        assert result.exit_code == 0
        assert "daily-digest" in recorder.ran_bases

    def test_no_side_effects_drops_daily_digest(self) -> None:
        settings = _settings(["fetch-oddsportal", "daily-digest"], ["soccer_epl"])
        recorder = _JobRecorder()
        result = _invoke_smoke(["--no-side-effects"], settings, recorder)

        assert result.exit_code == 0
        assert "fetch-oddsportal" in recorder.ran_bases
        assert "daily-digest" not in recorder.ran_bases

    def test_only_by_base_name_matches_all_compounds(self) -> None:
        settings = _settings(["fetch-oddsportal", "fetch-scores"], ["soccer_epl", "baseball_mlb"])
        recorder = _JobRecorder()
        result = _invoke_smoke(["--only", "fetch-oddsportal"], settings, recorder)

        assert result.exit_code == 0
        # Jobs run in sorted compound-name order: -epl before -mlb.
        assert recorder.ran == [
            ("fetch-oddsportal", "soccer_epl"),
            ("fetch-oddsportal", "baseball_mlb"),
        ]

    def test_only_by_compound_name_matches_single(self) -> None:
        settings = _settings(["fetch-oddsportal", "fetch-scores"], ["soccer_epl", "baseball_mlb"])
        recorder = _JobRecorder()
        result = _invoke_smoke(["--only", "fetch-oddsportal-epl"], settings, recorder)

        assert result.exit_code == 0
        assert recorder.ran == [("fetch-oddsportal", "soccer_epl")]

    def test_exclude_removes_job(self) -> None:
        settings = _settings(["fetch-oddsportal", "fetch-scores"], ["soccer_epl"])
        recorder = _JobRecorder()
        result = _invoke_smoke(["--exclude", "fetch-scores"], settings, recorder)

        assert result.exit_code == 0
        assert "fetch-oddsportal" in recorder.ran_bases
        assert "fetch-scores" not in recorder.ran_bases

    def test_unknown_only_token_errors(self) -> None:
        settings = _settings(["fetch-oddsportal"], ["soccer_epl"])
        recorder = _JobRecorder()
        result = _invoke_smoke(["--only", "no-such-job"], settings, recorder)

        assert result.exit_code != 0
        assert "Unknown job" in result.output
        assert recorder.ran == []

    def test_unknown_exclude_token_errors(self) -> None:
        settings = _settings(["fetch-oddsportal"], ["soccer_epl"])
        recorder = _JobRecorder()
        result = _invoke_smoke(["--exclude", "no-such-job"], settings, recorder)

        assert result.exit_code != 0
        assert "Unknown job" in result.output
        assert recorder.ran == []

    def test_failing_job_yields_nonzero_exit(self) -> None:
        settings = _settings(["fetch-oddsportal", "fetch-scores"], ["soccer_epl"])
        recorder = _JobRecorder(fail_bases={"fetch-scores"})
        result = _invoke_smoke([], settings, recorder)

        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_exit_code_counts_failures(self) -> None:
        settings = _settings(["fetch-scores"], ["soccer_epl", "baseball_mlb"])
        recorder = _JobRecorder(fail_bases={"fetch-scores"})
        result = _invoke_smoke([], settings, recorder)

        # Both per-sport expansions fail -> exit code equals the failure count.
        assert result.exit_code == 2

    def test_clean_run_exits_zero(self) -> None:
        settings = _settings(["fetch-oddsportal", "fetch-scores"], ["soccer_epl"])
        recorder = _JobRecorder()
        result = _invoke_smoke([], settings, recorder)

        assert result.exit_code == 0
        assert recorder.ran_bases == {"fetch-oddsportal", "fetch-scores"}
