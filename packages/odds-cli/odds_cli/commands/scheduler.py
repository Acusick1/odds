"""Scheduler management CLI commands."""

import asyncio

import structlog
import typer
from odds_core.config import get_settings
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()
logger = structlog.get_logger()


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
            get_bootstrap_function,
            get_job_cron,
            get_job_cron_sports,
            is_per_sport_job,
            make_compound_job_name,
        )

        bootstrap_jobs = app_settings.scheduler.bootstrap_jobs if effective_bootstrap else []

        # Start scheduler first so bootstrap jobs can schedule via the backend
        async with LocalSchedulerBackend() as _backend:
            if bootstrap_jobs:
                console.print("[green]Bootstrapping jobs...[/green]")
            else:
                console.print(
                    "[dim]Skipping bootstrap — scheduler will idle until jobs arrive[/dim]"
                )

            for job_name in bootstrap_jobs:
                cron_expr = get_job_cron(job_name)

                if cron_expr is not None:
                    # Cron-triggered jobs: install a recurring CronTrigger
                    # instead of going through the dynamic bootstrap path.
                    if is_per_sport_job(job_name):
                        allowed_sports = get_job_cron_sports(job_name)
                        target_sports = (
                            list(allowed_sports)
                            if allowed_sports is not None
                            else list(app_settings.data_collection.sports)
                        )
                        skipped_sports = (
                            [
                                s
                                for s in app_settings.data_collection.sports
                                if s not in target_sports
                            ]
                            if allowed_sports is not None
                            else []
                        )
                        for skipped in skipped_sports:
                            skipped_compound = make_compound_job_name(job_name, skipped)
                            console.print(
                                f"[dim]  {skipped_compound} skipped — "
                                f"{job_name} does not support {skipped}[/dim]"
                            )
                        for sport_key in target_sports:
                            compound = make_compound_job_name(job_name, sport_key)
                            existing = await _backend.get_job_status(compound)
                            if existing and existing.next_run_time:
                                console.print(
                                    f"[dim]  {compound} already scheduled "
                                    f"at {existing.next_run_time:%H:%M:%S UTC} — skipping[/dim]"
                                )
                                continue
                            try:
                                await _backend.schedule_cron(compound, cron_expr)
                                console.print(
                                    f"[green]  {compound} cron scheduled ({cron_expr})[/green]"
                                )
                            except Exception as e:
                                console.print(
                                    f"[yellow]  {compound} cron schedule failed: {e}[/yellow]"
                                )
                    else:
                        existing = await _backend.get_job_status(job_name)
                        if existing and existing.next_run_time:
                            console.print(
                                f"[dim]  {job_name} already scheduled "
                                f"at {existing.next_run_time:%H:%M:%S UTC} — skipping[/dim]"
                            )
                            continue
                        try:
                            await _backend.schedule_cron(job_name, cron_expr)
                            console.print(
                                f"[green]  {job_name} cron scheduled ({cron_expr})[/green]"
                            )
                        except Exception as e:
                            console.print(
                                f"[yellow]  {job_name} cron schedule failed: {e}[/yellow]"
                            )
                    continue

                bootstrap_fn = get_bootstrap_function(job_name)

                if is_per_sport_job(job_name):
                    for sport_key in app_settings.data_collection.sports:
                        compound = make_compound_job_name(job_name, sport_key)
                        existing = await _backend.get_job_status(compound)
                        if existing and existing.next_run_time:
                            console.print(
                                f"[dim]  {compound} already scheduled "
                                f"at {existing.next_run_time:%H:%M:%S UTC} — skipping[/dim]"
                            )
                            continue
                        try:
                            await bootstrap_fn(JobContext(sport=sport_key))
                            console.print(f"[green]  {compound} bootstrapped[/green]")
                        except Exception as e:
                            console.print(f"[yellow]  {compound} bootstrap failed: {e}[/yellow]")
                else:
                    existing = await _backend.get_job_status(job_name)
                    if existing and existing.next_run_time:
                        console.print(
                            f"[dim]  {job_name} already scheduled "
                            f"at {existing.next_run_time:%H:%M:%S UTC} — skipping[/dim]"
                        )
                        continue
                    try:
                        await bootstrap_fn(JobContext())
                        console.print(f"[green]  {job_name} bootstrapped[/green]")
                    except Exception as e:
                        console.print(f"[yellow]  {job_name} bootstrap failed: {e}[/yellow]")

            console.print()

            # Display status
            console.print("[bold green]Scheduler started![/bold green]")
            console.print("[dim]Jobs will self-schedule based on game proximity[/dim]")
            console.print("[dim]Press Ctrl+C to stop[/dim]\n")

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
def list_jobs():
    app_settings = get_settings()

    """
    List all currently scheduled jobs.

    Shows:
    - Job name
    - Next run time
    - Status

    Note: Not supported on Railway backend (static cron schedules).
    """
    console.print("[bold blue]Scheduled Jobs[/bold blue]\n")

    try:
        from odds_lambda.scheduling.backends import BackendUnavailableError, get_scheduler_backend

        backend = get_scheduler_backend()

        async def get_jobs():
            return await backend.get_scheduled_jobs()

        jobs = asyncio.run(get_jobs())

        if not jobs:
            console.print("[yellow]No jobs currently scheduled[/yellow]")
            return

        # Create jobs table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Job Name", style="white")
        table.add_column("Next Run Time", style="yellow")
        table.add_column("Status", style="green")

        for job in jobs:
            next_run = (
                job.next_run_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                if job.next_run_time
                else "[dim]Not scheduled[/dim]"
            )
            table.add_row(job.job_name, next_run, job.status.value)

        console.print(table)
        console.print(f"\n[dim]Total: {len(jobs)} jobs[/dim]")

    except BackendUnavailableError as e:
        console.print(f"[yellow]⚠ Not supported:[/yellow] {e}")
        console.print(
            f"\n[dim]Backend {app_settings.scheduler.backend} does not support job listing[/dim]"
        )

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
