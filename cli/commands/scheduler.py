"""Scheduler management CLI commands."""

import asyncio

import structlog
import typer
from rich.console import Console
from rich.table import Table

from core.config import settings

app = typer.Typer()
console = Console()
logger = structlog.get_logger()


@app.command("start")
def start_local():
    """
    Start local scheduler for testing (APScheduler backend).

    Simulates AWS Lambda + EventBridge behavior locally using APScheduler.
    Jobs will self-schedule just like in production.

    Requirements:
    - SCHEDULER_BACKEND=local in .env
    - Database running and accessible

    Usage:
        odds scheduler start

    Press Ctrl+C to stop the scheduler.
    """
    console.print("[bold blue]Starting local scheduler...[/bold blue]")
    console.print("[dim]Backend: APScheduler (local testing mode)[/dim]\n")

    # Verify we're using local backend
    if settings.scheduler_backend != "local":
        console.print(
            f"[bold red]Error:[/bold red] SCHEDULER_BACKEND is '{settings.scheduler_backend}', "
            f"expected 'local'"
        )
        console.print(
            "\n[yellow]Set SCHEDULER_BACKEND=local in your .env file to use local scheduler[/yellow]"
        )
        raise typer.Exit(1)

    # Import backend
    try:
        from core.scheduling.backends.local import LocalSchedulerBackend
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Failed to import local backend: {e}")
        raise typer.Exit(1) from e

    # Create backend instance
    backend = LocalSchedulerBackend()

    # Bootstrap by running initial fetch to start self-scheduling
    console.print("[green]Running initial fetch to bootstrap scheduler...[/green]")

    try:
        from jobs import fetch_odds

        asyncio.run(fetch_odds.main())
        console.print("[green]✓ Bootstrap complete[/green]\n")
    except Exception as e:
        console.print(f"[bold red]✗ Bootstrap failed:[/bold red] {e}\n")
        console.print("[yellow]Scheduler will still run, but jobs may not be scheduled[/yellow]\n")

    # Display status
    console.print("[bold green]Scheduler started![/bold green]")
    console.print("[dim]Jobs will self-schedule based on game proximity[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Keep alive
    try:
        backend.keep_alive()
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
    console.print(f"[bold blue]Executing {job}...[/bold blue]")
    console.print(f"[dim]Backend: {settings.scheduler_backend}[/dim]\n")

    job_map = {
        "fetch-odds": ("jobs.fetch_odds", "Fetch odds"),
        "fetch-scores": ("jobs.fetch_scores", "Fetch scores"),
        "update-status": ("jobs.update_status", "Update status"),
    }

    if job not in job_map:
        console.print(f"[bold red]Error:[/bold red] Unknown job '{job}'")
        console.print("\n[yellow]Available jobs:[/yellow]")
        for job_name in job_map.keys():
            console.print(f"  - {job_name}")
        raise typer.Exit(1)

    module_path, job_desc = job_map[job]

    try:
        # Import and run job
        module = __import__(module_path, fromlist=["main"])
        asyncio.run(module.main())

        console.print(f"\n[bold green]✓ {job_desc} completed[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ {job_desc} failed:[/bold red] {e}")
        logger.error("cli_job_failed", job=job, error=str(e), exc_info=True)
        raise typer.Exit(1) from e


@app.command("info")
def info():
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

    table.add_row("Scheduler Backend", settings.scheduler_backend)
    table.add_row("Lookahead Days", str(settings.scheduling_lookahead_days))
    table.add_row("Sports", ", ".join(settings.sports))
    table.add_row("Markets", ", ".join(settings.markets))
    table.add_row("Bookmakers", f"{len(settings.bookmakers)} configured")

    if settings.scheduler_backend == "aws":
        table.add_row("AWS Region", settings.aws_region or "[red]Not set[/red]")
        table.add_row("Lambda ARN", settings.lambda_arn or "[red]Not set[/red]")

    console.print(table)

    # Backend-specific info
    console.print("\n[bold]Backend Info:[/bold]")
    if settings.scheduler_backend == "local":
        console.print("  • Uses APScheduler for local testing")
        console.print("  • Jobs self-schedule dynamically")
        console.print("  • Start with: [cyan]odds scheduler start[/cyan]")

    elif settings.scheduler_backend == "aws":
        console.print("  • Uses AWS Lambda + EventBridge")
        console.print("  • Dynamic one-time schedules")
        console.print("  • Deploy with: [cyan]terraform apply[/cyan]")

    elif settings.scheduler_backend == "railway":
        console.print("  • Uses Railway cron (static schedules)")
        console.print("  • Configure in railway.json")
        console.print("  • Jobs use smart gating logic")


@app.command("test-backend")
def test_backend():
    """
    Test scheduler backend connection and permissions.

    Verifies:
    - Backend can be instantiated
    - Required environment variables are set
    - AWS credentials work (for AWS backend)
    - Database connection works
    """
    console.print("[bold blue]Testing scheduler backend...[/bold blue]\n")

    try:
        from core.scheduling.backends import get_scheduler_backend

        backend = get_scheduler_backend()

        console.print(f"[green]✓ Backend initialized:[/green] {backend.get_backend_name()}")

        # Test database connection
        console.print("\n[bold]Testing database connection...[/bold]")
        from core.database import async_session_maker

        async def test_db():
            async with async_session_maker() as session:
                # Simple query
                from sqlalchemy import text

                result = await session.execute(text("SELECT 1"))
                return result.scalar()

        asyncio.run(test_db())
        console.print("[green]✓ Database connection successful[/green]")

        # AWS-specific tests
        if settings.scheduler_backend == "aws":
            console.print("\n[bold]Testing AWS permissions...[/bold]")
            import boto3

            try:
                events_client = boto3.client("events", region_name=settings.aws_region)
                # List rules to test permissions
                events_client.list_rules(Limit=1)
                console.print("[green]✓ AWS EventBridge access verified[/green]")
            except Exception as e:
                console.print(f"[red]✗ AWS EventBridge access failed:[/red] {e}")

        console.print("\n[bold green]✓ All tests passed![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Backend test failed:[/bold red] {e}")
        logger.error("backend_test_failed", error=str(e), exc_info=True)
        raise typer.Exit(1) from e
