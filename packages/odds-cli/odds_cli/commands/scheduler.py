"""Scheduler management CLI commands."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
import typer
from odds_core.config import get_settings
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from odds_lambda.scheduling.backends.base import ScheduledJob

app = typer.Typer()
console = Console()
logger = structlog.get_logger()


def _print_scheduled_jobs(con: Console, jobs: list[ScheduledJob]) -> None:
    if not jobs:
        con.print("[yellow]No jobs currently scheduled[/yellow]")
        return
    jobs = sorted(jobs, key=lambda j: (j.next_run_time is None, j.next_run_time))
    # The raw EventBridge schedule expression is only populated by the AWS
    # backend; show the column only when present so local listings stay clean.
    show_schedule = any(job.schedule_expression for job in jobs)
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Job Name", style="white")
    table.add_column("Next Run Time", style="yellow")
    table.add_column("Status", style="green")
    if show_schedule:
        table.add_column("Schedule", style="cyan")
    for job in jobs:
        next_run = (
            job.next_run_time.strftime("%Y-%m-%d %H:%M:%S UTC")
            if job.next_run_time
            else "[dim]Not scheduled[/dim]"
        )
        row = [job.job_name, next_run, job.status.value]
        if show_schedule:
            row.append(job.schedule_expression or "[dim]—[/dim]")
        table.add_row(*row)
    con.print(table)
    con.print(f"[dim]Total: {len(jobs)} jobs[/dim]")


@app.command("start")
def start_local(
    db: str | None = typer.Option(
        None,
        "--db",
        help="Override DATABASE_URL's dbname (e.g. 'odds_test' for a sandbox scheduler).",
    ),
    bootstrap: bool | None = typer.Option(
        None,
        "--bootstrap/--no-bootstrap",
        help=(
            "Bootstrap scheduled jobs on start. Defaults to on when --db is unset "
            "(prod-like) and off when --db is set (sidecar mode — prevents a "
            "concurrent agent-run from clashing with an `odds agent run` you're "
            "driving manually)."
        ),
    ),
) -> None:
    """
    Start local scheduler for testing (APScheduler backend).

    Simulates AWS Lambda + EventBridge behavior locally using APScheduler.
    Event-driven jobs self-schedule as they run; fixed-schedule jobs listed
    in ``_JOB_CRON_MAP`` install recurring cron triggers at bootstrap.

    Requirements:
    - SCHEDULER_BACKEND=local in .env
    - Database running and accessible

    Usage:
        odds scheduler start                          # prod-like (bootstrap on)
        odds scheduler start --db odds_test           # dev sidecar (bootstrap off)
        odds scheduler start --db odds_test --bootstrap   # dev full-loop smoke test

    Press Ctrl+C to stop the scheduler.
    """
    if db is not None:
        from odds_cli.db_override import override_database_url

        override_database_url(db)

    effective_bootstrap = (db is None) if bootstrap is None else bootstrap

    console.print("[bold blue]Starting local scheduler...[/bold blue]")
    console.print("[dim]Backend: APScheduler (local testing mode)[/dim]")
    if db is not None:
        console.print(f"[dim]Database override: {db}[/dim]")
    console.print(f"[dim]Bootstrap: {'enabled' if effective_bootstrap else 'disabled'}[/dim]")
    console.print()

    app_settings = get_settings()

    # Verify we're using local backend
    if app_settings.scheduler.backend != "local":
        console.print(
            f"[bold red]Error:[/bold red] SCHEDULER_BACKEND is '{app_settings.scheduler.backend}', "
            f"expected 'local'"
        )
        console.print(
            "\n[yellow]Set SCHEDULER_BACKEND=local in your .env file to use local scheduler[/yellow]"
        )
        raise typer.Exit(1)

    # Import backend
    try:
        from odds_lambda.scheduling.backends.local import LocalSchedulerBackend
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Failed to import local backend: {e}")
        raise typer.Exit(1) from e

    async def run_scheduler():
        """Run scheduler using async context manager."""
        from odds_lambda.scheduling.jobs import (
            JobContext,
            expected_compound_job_names,
            get_bootstrap_function,
            get_job_cron,
            is_per_sport_job,
            make_compound_job_name,
            resolve_target_sports,
        )

        bootstrap_jobs = app_settings.scheduler.bootstrap_jobs if effective_bootstrap else []
        configured_sports = app_settings.data_collection.sports

        # Start scheduler first so bootstrap jobs can schedule via the backend
        async with LocalSchedulerBackend() as _backend:
            from apscheduler import Event, JobReleased

            async def _on_job_released(_event: Event) -> None:
                try:
                    jobs = await _backend.get_scheduled_jobs()
                except Exception as e:
                    logger.debug("job_released_refresh_failed", error=str(e))
                    return
                console.print()
                _print_scheduled_jobs(console, jobs)

            _backend.subscribe(_on_job_released, {JobReleased})

            # Prune schedules that are no longer in bootstrap_jobs. Without this,
            # self-scheduling jobs removed from the config persist indefinitely in
            # the Postgres data store and keep firing after restart.
            if effective_bootstrap:
                expected = expected_compound_job_names(bootstrap_jobs, configured_sports)
                existing_jobs = await _backend.get_scheduled_jobs()
                stale = [j for j in existing_jobs if j.job_name not in expected]
                if stale:
                    console.print("[yellow]Removing stale schedules...[/yellow]")
                    for job in stale:
                        try:
                            await _backend.cancel_scheduled_execution(job.job_name)
                            console.print(f"[dim]  {job.job_name} removed[/dim]")
                        except Exception as e:
                            console.print(f"[yellow]  {job.job_name} removal failed: {e}[/yellow]")
                    console.print()

            async def _bootstrap_cron(compound: str, cron_expr: str) -> None:
                """Install a recurring CronTrigger, re-applying on every start.

                Cron jobs intentionally do NOT short-circuit on an existing
                schedule — APScheduler's data store persists triggers across
                restarts, so a stale cron expression would otherwise win over
                an edit to ``_JOB_CRON_MAP``. ``ConflictPolicy.replace`` inside
                ``LocalSchedulerBackend.schedule_cron`` makes this idempotent.
                """
                existing = await _backend.get_job_status(compound)
                if existing and existing.next_run_time:
                    console.print(
                        f"[dim]  {compound} existing schedule "
                        f"at {existing.next_run_time:%H:%M:%S UTC} — re-applying[/dim]"
                    )
                try:
                    await _backend.schedule_cron(compound, cron_expr)
                    console.print(f"[green]  {compound} cron scheduled ({cron_expr})[/green]")
                except Exception as e:
                    console.print(f"[yellow]  {compound} cron schedule failed: {e}[/yellow]")

            async def _bootstrap_dynamic(compound: str, ctx: JobContext, bootstrap_fn) -> None:
                """Run a job's bootstrap entry point once, skipping if already scheduled.

                Unlike cron jobs, dynamic (self-scheduling) jobs own their next
                fire time — re-running bootstrap would clobber an agent-requested
                override or an in-flight proximity tier. The skip is the right
                call here.
                """
                existing = await _backend.get_job_status(compound)
                if existing and existing.next_run_time:
                    console.print(
                        f"[dim]  {compound} already scheduled "
                        f"at {existing.next_run_time:%H:%M:%S UTC} — skipping[/dim]"
                    )
                    return
                try:
                    await bootstrap_fn(ctx)
                    console.print(f"[green]  {compound} bootstrapped[/green]")
                except Exception as e:
                    console.print(f"[yellow]  {compound} bootstrap failed: {e}[/yellow]")

            if bootstrap_jobs:
                console.print("[green]Bootstrapping jobs...[/green]")
            else:
                console.print(
                    "[dim]Skipping bootstrap — scheduler will idle until jobs arrive[/dim]"
                )

            for job_name in bootstrap_jobs:
                cron_expr = get_job_cron(job_name)
                per_sport = is_per_sport_job(job_name)

                if cron_expr is not None:
                    if per_sport:
                        target_sports = resolve_target_sports(job_name, configured_sports)
                        skipped_sports = [s for s in configured_sports if s not in target_sports]
                        for skipped in skipped_sports:
                            skipped_compound = make_compound_job_name(job_name, skipped)
                            console.print(
                                f"[dim]  {skipped_compound} skipped — "
                                f"{job_name} does not support {skipped}[/dim]"
                            )
                        for sport_key in target_sports:
                            await _bootstrap_cron(
                                make_compound_job_name(job_name, sport_key), cron_expr
                            )
                    else:
                        await _bootstrap_cron(job_name, cron_expr)
                    continue

                bootstrap_fn = get_bootstrap_function(job_name)

                if per_sport:
                    for sport_key in configured_sports:
                        await _bootstrap_dynamic(
                            make_compound_job_name(job_name, sport_key),
                            JobContext(sport=sport_key),
                            bootstrap_fn,
                        )
                else:
                    await _bootstrap_dynamic(job_name, JobContext(), bootstrap_fn)

            console.print()

            # Display status
            console.print("[bold green]Scheduler started![/bold green]")
            console.print("[dim]Jobs will self-schedule based on game proximity[/dim]")
            console.print("[dim]Press Ctrl+C to stop[/dim]\n")

            jobs = await _backend.get_scheduled_jobs()
            _print_scheduled_jobs(console, jobs)
            console.print()

            try:
                logger.info("local_scheduler_running", message="Press Ctrl+C to stop")
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

    # Run async scheduler
    try:
        asyncio.run(run_scheduler())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down scheduler...[/yellow]")
        console.print("[green]✓ Scheduler stopped[/green]")


@app.command("run-once")
def run_once(
    job: str = typer.Argument(..., help="Job name: fetch-odds, fetch-scores, or update-status"),
):
    """
    Execute a single job once and exit.

    Works with any backend (local, aws, railway).
    Useful for:
    - Manual job execution
    - Testing individual jobs
    - CI/CD pipelines
    - Railway cron invocation

    Usage:
        odds scheduler run-once fetch-odds
        odds scheduler run-once fetch-scores
        odds scheduler run-once update-status
    """
    app_settings = get_settings()

    console.print(f"[bold blue]Executing {job}...[/bold blue]")
    console.print(f"[dim]Backend: {app_settings.scheduler.backend}[/dim]\n")

    try:
        # Use centralized job registry
        from odds_lambda.scheduling.jobs import (
            JobContext,
            get_job_function,
            list_available_jobs,
            resolve_job_name,
        )

        base_name, sport = resolve_job_name(job)
        job_func = get_job_function(base_name)
        ctx = JobContext(sport=sport) if sport else JobContext()

        # Run job
        asyncio.run(job_func(ctx))

        console.print(f"\n[bold green]✓ {job} completed[/bold green]")

    except KeyError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        console.print("\n[yellow]Available jobs:[/yellow]")
        from odds_lambda.scheduling.jobs import list_available_jobs

        for job_name in list_available_jobs():
            console.print(f"  - {job_name}")
        raise typer.Exit(1) from e

    except Exception as e:
        console.print(f"\n[bold red]✗ {job} failed:[/bold red] {e}")
        logger.error("cli_job_failed", job=job, error=str(e), exc_info=True)
        raise typer.Exit(1) from e


@app.command("smoke")
def smoke(
    include_agent: bool = typer.Option(
        False,
        "--include-agent",
        help="Also run smoke-unsafe jobs (agent-run). Off by default so a deploy "
        "check never places paper bets.",
    ),
    no_side_effects: bool = typer.Option(
        False,
        "--no-side-effects",
        help="Drop outward-posting jobs (daily-digest) so the check stays silent "
        "on shared channels.",
    ),
    only: list[str] | None = typer.Option(  # noqa: B008 — typer option idiom
        None,
        "--only",
        help="Run only these jobs (base or compound name). Repeatable.",
    ),
    exclude: list[str] | None = typer.Option(  # noqa: B008 — typer option idiom
        None,
        "--exclude",
        help="Skip these jobs (base or compound name). Repeatable.",
    ),
) -> None:
    """
    Run each bootstrapped job once and report pass/fail (deploy validation).

    Derives the job set from ``scheduler.bootstrap_jobs`` (the same single
    source of truth ``start`` uses), expands per-sport jobs into one run per
    configured sport, runs each once, and prints a per-job pass/fail table.
    Exits non-zero (exit code = number of failed jobs) if any job fails.

    ``agent-run`` is skipped by default (never place paper bets during a deploy
    check); pass ``--include-agent`` to run it. ``daily-digest`` posts to Discord;
    pass ``--no-side-effects`` to drop it.

    Run with ``SCHEDULER_DRY_RUN=true`` (as ``deploy.sh`` does) so self-scheduling
    is a no-op and the live schedule store is untouched.

    Coverage caveat: smoke always exercises each job's import / config / decision /
    schema path, but the full fetch+ingest body only runs when the job's cadence
    gate (``decision.should_execute``) says execute. A ``--force`` gate bypass is
    out of scope.

    Usage:
        SCHEDULER_DRY_RUN=true odds scheduler smoke
        odds scheduler smoke --no-side-effects
        odds scheduler smoke --only fetch-oddsportal --only fetch-espn-fixtures
        odds scheduler smoke --include-agent
    """
    app_settings = get_settings()

    from odds_lambda.scheduling.jobs import (
        JobContext,
        expected_compound_job_names,
        get_job_function,
        is_outward_posting,
        is_smoke_unsafe,
        resolve_job_name,
    )

    bootstrap_jobs = app_settings.scheduler.bootstrap_jobs
    configured_sports = app_settings.data_collection.sports
    candidates = sorted(expected_compound_job_names(bootstrap_jobs, configured_sports))

    if not candidates:
        console.print("[yellow]No jobs to smoke — bootstrap_jobs is empty.[/yellow]")
        return

    if not app_settings.scheduler.dry_run:
        console.print(
            "[yellow]⚠ SCHEDULER_DRY_RUN is not set — jobs may write real schedules "
            "to the store. Set SCHEDULER_DRY_RUN=true (deploy.sh does this).[/yellow]"
        )

    base_of = {c: resolve_job_name(c)[0] for c in candidates}

    def _match_tokens(tokens: list[str]) -> tuple[set[str], list[str]]:
        """Resolve ``--only``/``--exclude`` tokens against the candidate set.

        A token matches a candidate when it equals the compound name or the
        base name. Returns (matched compound names, unknown tokens).
        """
        matched: set[str] = set()
        unknown: list[str] = []
        for token in tokens:
            hits = [c for c in candidates if c == token or base_of[c] == token]
            if hits:
                matched.update(hits)
            else:
                unknown.append(token)
        return matched, unknown

    unknown_tokens: list[str] = []
    if only:
        only_set, unknown = _match_tokens(only)
        unknown_tokens.extend(unknown)
        selected = only_set
    else:
        selected = set(candidates)

    excluded_names: set[str] = set()
    if exclude:
        exclude_set, unknown = _match_tokens(exclude)
        unknown_tokens.extend(unknown)
        excluded_names = exclude_set

    if unknown_tokens:
        console.print(f"[bold red]Error:[/bold red] Unknown job(s): {', '.join(unknown_tokens)}")
        console.print(f"[dim]Available: {', '.join(candidates)}[/dim]")
        raise typer.Exit(1)

    # Classify each candidate into run / skip with a reason.
    to_run: list[str] = []
    skipped: list[tuple[str, str]] = []
    for name in sorted(selected):
        if name in excluded_names:
            skipped.append((name, "excluded (--exclude)"))
        elif is_smoke_unsafe(name) and not include_agent:
            skipped.append((name, "smoke-unsafe (pass --include-agent)"))
        elif is_outward_posting(name) and no_side_effects:
            skipped.append((name, "outward-posting (--no-side-effects)"))
        else:
            to_run.append(name)

    console.print("[bold blue]Smoke-testing bootstrapped jobs...[/bold blue]")
    console.print(f"[dim]Backend: {app_settings.scheduler.backend} | ", end="")
    console.print(f"dry_run: {app_settings.scheduler.dry_run}[/dim]\n")

    async def _run_all() -> list[tuple[str, bool, str]]:
        results: list[tuple[str, bool, str]] = []
        for name in to_run:
            base_name, sport = resolve_job_name(name)
            ctx = JobContext(sport=sport) if sport else JobContext()
            try:
                job_func = get_job_function(base_name)
                await job_func(ctx)
                results.append((name, True, ""))
                console.print(f"[green]  ✓ {name}[/green]")
            except Exception as e:  # noqa: BLE001 — one failure must not abort the rest
                logger.error("smoke_job_failed", job=name, error=str(e), exc_info=True)
                results.append((name, False, str(e)))
                console.print(f"[red]  ✗ {name}: {e}[/red]")
        return results

    results = asyncio.run(_run_all()) if to_run else []

    # Render results table.
    console.print()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Job", style="white")
    table.add_column("Result", style="white")
    table.add_column("Detail", style="dim")
    for name, ok, detail in results:
        result_cell = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(name, result_cell, detail)
    for name, reason in skipped:
        table.add_row(name, "[yellow]SKIP[/yellow]", reason)
    console.print(table)

    failed = [name for name, ok, _ in results if not ok]
    passed = len(results) - len(failed)
    console.print(f"\n[bold]{passed} passed, {len(failed)} failed, {len(skipped)} skipped[/bold]")

    if failed:
        console.print(f"[bold red]✗ Smoke check failed: {', '.join(failed)}[/bold red]")
        # Exit code = number of failed jobs (clamped to a valid 1-255 range).
        raise typer.Exit(min(len(failed), 255))

    console.print("[bold green]✓ Smoke check passed[/bold green]")


@app.command("info")
def info():
    app_settings = get_settings()

    """
    Display current scheduler configuration.

    Shows:
    - Active backend
    - Configuration settings
    - Environment variables
    """
    console.print("[bold blue]Scheduler Configuration[/bold blue]\n")

    # Create configuration table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="white")
    table.add_column("Value", style="yellow")

    table.add_row("Scheduler Backend", app_settings.scheduler.backend)
    table.add_row("Dry-Run Mode", "Enabled" if app_settings.scheduler.dry_run else "Disabled")
    table.add_row("Lookahead Days", str(app_settings.scheduler.lookahead_days))
    table.add_row("Sports", ", ".join(app_settings.data_collection.sports))
    table.add_row("Markets", ", ".join(app_settings.data_collection.markets))
    table.add_row("Bookmakers", f"{len(app_settings.data_collection.bookmakers)} configured")

    if app_settings.scheduler.backend == "aws":
        table.add_row("AWS Region", app_settings.aws.region or "[red]Not set[/red]")
        table.add_row("Lambda ARN", app_settings.aws.lambda_arn or "[red]Not set[/red]")

    console.print(table)

    # Backend-specific info
    console.print("\n[bold]Backend Info:[/bold]")
    if app_settings.scheduler.backend == "local":
        console.print("  • Uses APScheduler for local testing")
        console.print("  • Jobs self-schedule dynamically")
        console.print("  • Start with: [cyan]odds scheduler start[/cyan]")

    elif app_settings.scheduler.backend == "aws":
        console.print("  • Uses AWS Lambda + EventBridge")
        console.print("  • Dynamic one-time schedules")
        console.print("  • Deploy with: [cyan]terraform apply[/cyan]")

    elif app_settings.scheduler.backend == "railway":
        console.print("  • Uses Railway cron (static schedules)")
        console.print("  • Configure in railway.json")
        console.print("  • Jobs use smart gating logic")

    if app_settings.scheduler.dry_run:
        console.print("\n[bold yellow]⚠ DRY-RUN MODE ENABLED[/bold yellow]")
        console.print("  Scheduling operations will be logged but not executed")


@app.command("list-jobs")
def list_jobs(
    backend: str | None = typer.Option(
        None,
        "--backend",
        help=(
            "Override the configured scheduler backend ('aws', 'local', 'railway'). "
            "Use '--backend aws' to query the live deployed EventBridge schedule "
            "without editing .env (requires AWS_REGION, AWS_LAMBDA_ARN, AWS_RULE_PREFIX "
            "and AWS credentials)."
        ),
    ),
):
    """
    List all currently scheduled jobs.

    Shows:
    - Job name
    - Next run time
    - Status
    - Schedule (raw EventBridge expression, AWS backend only)

    Note: Not supported on Railway backend (static cron schedules).
    """
    app_settings = get_settings()
    effective_backend = backend or app_settings.scheduler.backend
    console.print("[bold blue]Scheduled Jobs[/bold blue]")
    console.print(f"[dim]Backend: {effective_backend}[/dim]\n")

    try:
        from odds_lambda.scheduling.backends import BackendUnavailableError, get_scheduler_backend

        backend_instance = get_scheduler_backend(backend_type=backend)

        async def get_jobs():
            return await backend_instance.get_scheduled_jobs()

        jobs = asyncio.run(get_jobs())

        _print_scheduled_jobs(console, jobs)

    except BackendUnavailableError as e:
        console.print(f"[yellow]⚠ Not supported:[/yellow] {e}")
        console.print(f"\n[dim]Backend {effective_backend} does not support job listing[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Failed to list jobs:[/bold red] {e}")
        logger.error("list_jobs_failed", error=str(e), exc_info=True)
        raise typer.Exit(1) from e


@app.command("health")
def health():
    """
    Run comprehensive health check on scheduler backend.

    Shows:
    - Overall health status
    - Individual checks (passed/failed)
    - Detailed information

    Useful for diagnosing issues.
    """
    console.print("[bold blue]Scheduler Health Check[/bold blue]\n")

    try:
        from odds_lambda.scheduling.backends import get_scheduler_backend

        backend = get_scheduler_backend()

        async def run_health_check():
            return await backend.health_check()

        health_result = asyncio.run(run_health_check())

        # Overall status
        if health_result.is_healthy:
            console.print("[bold green]✓ Backend Healthy[/bold green]")
        else:
            console.print("[bold red]✗ Backend Unhealthy[/bold red]")

        console.print(f"Backend: {health_result.backend_name}\n")

        # Checks passed
        if health_result.checks_passed:
            console.print("[bold green]Checks Passed:[/bold green]")
            for check in health_result.checks_passed:
                console.print(f"  [green]✓[/green] {check}")

        # Checks failed
        if health_result.checks_failed:
            console.print("\n[bold red]Checks Failed:[/bold red]")
            for check in health_result.checks_failed:
                console.print(f"  [red]✗[/red] {check}")

        # Details
        if health_result.details:
            console.print("\n[bold]Additional Details:[/bold]")
            for key, value in health_result.details.items():
                console.print(f"  • {key}: {value}")

        # Exit code based on health
        if not health_result.is_healthy:
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]✗ Health check failed:[/bold red] {e}")
        logger.error("health_check_failed", error=str(e), exc_info=True)
        raise typer.Exit(1) from e


@app.command("test-backend")
def test_backend():
    """
    Test scheduler backend connection and permissions.

    Verifies:
    - Configuration validation
    - Comprehensive health checks
    - Database connection
    - Backend-specific permissions

    Uses new health check and validation methods.
    """
    console.print("[bold blue]Testing scheduler backend...[/bold blue]\n")

    try:
        from odds_lambda.scheduling.backends import get_scheduler_backend

        backend = get_scheduler_backend()

        console.print(f"[green]✓ Backend initialized:[/green] {backend.get_backend_name()}")

        # Test configuration validation
        console.print("\n[bold]Validating configuration...[/bold]")
        validation = backend.validate_configuration()

        if validation.is_valid:
            console.print("[green]✓ Configuration valid[/green]")
        else:
            console.print("[red]✗ Configuration invalid:[/red]")
            for error in validation.errors:
                console.print(f"  • {error}")

        if validation.warnings:
            console.print("[yellow]Warnings:[/yellow]")
            for warning in validation.warnings:
                console.print(f"  • {warning}")

        # Test comprehensive health check
        console.print("\n[bold]Running health checks...[/bold]")

        async def run_health_check():
            return await backend.health_check()

        health = asyncio.run(run_health_check())

        if health.is_healthy:
            console.print("[green]✓ Backend healthy[/green]")
        else:
            console.print("[red]✗ Backend unhealthy[/red]")

        console.print("\n[bold]Checks passed:[/bold]")
        for check in health.checks_passed:
            console.print(f"  [green]✓[/green] {check}")

        if health.checks_failed:
            console.print("\n[bold]Checks failed:[/bold]")
            for check in health.checks_failed:
                console.print(f"  [red]✗[/red] {check}")

        if health.details:
            console.print("\n[bold]Details:[/bold]")
            for key, value in health.details.items():
                console.print(f"  • {key}: {value}")

        # Test database connection
        console.print("\n[bold]Testing database connection...[/bold]")
        from odds_core.database import async_session_maker

        async def test_db():
            async with async_session_maker() as session:
                from sqlalchemy import text

                result = await session.execute(text("SELECT 1"))
                return result.scalar()

        asyncio.run(test_db())
        console.print("[green]✓ Database connection successful[/green]")

        # Summary
        if health.is_healthy and validation.is_valid:
            console.print("\n[bold green]✓ All tests passed![/bold green]")
        else:
            console.print("\n[bold yellow]⚠ Some tests failed - check details above[/bold yellow]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]✗ Backend test failed:[/bold red] {e}")
        logger.error("backend_test_failed", error=str(e), exc_info=True)
        raise typer.Exit(1) from e
