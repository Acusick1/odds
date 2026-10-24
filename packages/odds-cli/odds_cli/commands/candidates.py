"""CLI commands for the betting candidate registry."""

from pathlib import Path
from typing import Annotated

import typer
from odds_core.candidates import CandidateStatus, candidates_dir, load_candidates
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("list")
def list_candidates(
    status: Annotated[
        CandidateStatus | None, typer.Option("--status", help="Filter by status")
    ] = None,
    directory: Annotated[
        Path | None, typer.Option("--dir", help="Registry directory", hidden=True)
    ] = None,
) -> None:
    """List registered candidates."""
    candidates = load_candidates(directory or candidates_dir())
    if status is not None:
        candidates = [c for c in candidates if c.status == status]

    if not candidates:
        console.print("[yellow]No candidates registered[/yellow]")
        return

    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Kind")
    table.add_column("Sport")
    table.add_column("Version", justify="right")
    table.add_column("Status")
    table.add_column("Start date")
    table.add_column("Issue", justify="right")
    for c in candidates:
        table.add_row(
            c.name,
            c.kind,
            c.sport,
            str(c.version),
            c.status,
            c.start_date.isoformat(),
            f"#{c.issue}",
        )
    console.print(table)
