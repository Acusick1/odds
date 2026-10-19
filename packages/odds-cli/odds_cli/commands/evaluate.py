"""CLI commands for pre-registered model evaluations."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from odds_analytics.retail_ev_evaluation import EvaluationReport, run_evaluation
from odds_core.database import async_session_maker
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


async def _run(
    spec: Path, output_dir: Path, force: bool, allow_incomplete: list[str]
) -> EvaluationReport:
    async with async_session_maker() as session:
        return await run_evaluation(
            session,
            spec,
            output_dir,
            now=datetime.now(UTC),
            force=force,
            allow_incomplete=allow_incomplete,
        )


@app.command("retail-ev")
def evaluate_retail_ev(
    spec: str = typer.Option(..., "--spec", "-s", help="Evaluation spec (YAML)"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Directory for the outputs"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Run a spec that was already evaluated, or overwrite an output directory; "
        "recorded in the manifest",
    ),
    allow_incomplete: Annotated[
        list[str] | None,
        typer.Option(
            "--allow-incomplete",
            help="Score this window event although it is not FINAL or has no FDUK closing "
            "snapshot (repeatable); recorded in the manifest",
        ),
    ] = None,
) -> None:
    """Score a frozen retail-EV model once on its window and print the keep or retire verdict."""
    report = asyncio.run(_run(Path(spec), Path(output_dir), force, allow_incomplete or []))
    table = Table(title=f"Verdict: {report.verdict.value.upper()}")
    for column in ("condition", "point", "95% CI low", "95% CI high"):
        table.add_column(column)
    for name, estimate in (
        ("MSE gain", report.keep_bar.mse_gain),
        ("pick EV gain", report.keep_bar.pick_ev_gain),
    ):
        table.add_row(
            name, f"{estimate.point:.4g}", f"{estimate.ci_low:.4g}", f"{estimate.ci_high:.4g}"
        )
    console.print(table)
    counts = report.manifest["counts"]
    console.print(
        f"{counts['scored_events']} events and {counts['scored_rows']:,} rows scored in "
        f"{counts['blocks']} blocks; {counts['unscored_window_events']} window events had no "
        f"row. Git {report.manifest['git_sha'][:12]}"
        f"{' (dirty)' if report.manifest['git_dirty'] else ''}. Outputs in {output_dir}."
    )
