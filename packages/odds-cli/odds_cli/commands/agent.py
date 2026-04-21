"""Run the sport agent subprocess outside the scheduler wrapper."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from odds_cli.db_override import override_database_url

app = typer.Typer()
console = Console()


@app.command("run")
def run(
    sport: str = typer.Option(
        ..., "--sport", "-s", help="Sport key (e.g. 'soccer_epl', 'baseball_mlb')"
    ),
    db: str = typer.Option(
        "odds",
        "--db",
        help="Database name to run against. Swapped into DATABASE_URL.",
    ),
) -> None:
    """Run the sport agent subprocess directly, bypassing the scheduler wrapper.

    No pre-schedule, no agent_wakeups consumption, no reschedule — so repeated
    runs don't disturb the scheduler. By default the agent runs against the
    local ``odds`` database (same host/credentials as the configured
    DATABASE_URL, just a different dbname), which is also where the local
    scheduler writes scraper output and scored predictions.
    """
    override_database_url(db)

    from odds_lambda.jobs.agent_run import _run_claude_agent

    console.print(f"[bold blue]Running agent for {sport} against db={db}...[/bold blue]")
    exit_code = asyncio.run(_run_claude_agent(sport))

    if exit_code < 0:
        console.print("[bold red]Agent timed out[/bold red]")
        raise typer.Exit(1)
    if exit_code != 0:
        console.print(f"[yellow]Agent exited with code {exit_code}[/yellow]")
        raise typer.Exit(exit_code)

    console.print("[green]Agent finished[/green]")
