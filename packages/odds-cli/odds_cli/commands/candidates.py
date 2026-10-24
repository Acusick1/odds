"""CLI commands for the betting candidate registry."""

import asyncio
from typing import Annotated

import typer
from odds_analytics.candidate_report import CandidateSummary, Verdict, candidate_report
from odds_core.candidates import Candidate, CandidateStatus, candidates_dir, load_candidates
from odds_core.database import async_session_maker
from odds_lambda.candidate_rules import RULES, rule_sha256, validate_candidate_rules
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


def _load_registry() -> list[Candidate]:
    try:
        candidates = load_candidates(candidates_dir())
        validate_candidate_rules(candidates)
        return candidates
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]", soft_wrap=True)
        raise typer.Exit(1) from exc


@app.command("list")
def list_candidates(
    status: Annotated[
        CandidateStatus | None, typer.Option("--status", help="Filter by status")
    ] = None,
) -> None:
    """List registered candidates."""
    candidates = _load_registry()
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


@app.command("rule-hash")
def rule_hash(rule: Annotated[str, typer.Argument(help="Rule name")]) -> None:
    """Print the rule_sha256 a mechanical candidate records for the rule."""
    if rule not in RULES:
        console.print(
            f"[red]unknown rule {rule!r}, known rules {sorted(RULES)}[/red]", soft_wrap=True
        )
        raise typer.Exit(1)
    console.print(rule_sha256(rule), soft_wrap=True)


_VERDICT_STYLE = {
    Verdict.PROMOTE: "green",
    Verdict.KILL: "red",
    Verdict.CONTINUE: "yellow",
    Verdict.NOT_YET: "dim",
}


def _interval(low_high: tuple[float, float]) -> str:
    return f"[{low_high[0]:+.3f}, {low_high[1]:+.3f}]"


@app.command("report")
def report() -> None:
    """Settled results and checkpoint verdict per candidate and version."""
    registry = _load_registry()
    summaries = asyncio.run(_report(registry))
    if not summaries:
        console.print("[yellow]No candidates registered and no candidate trades[/yellow]")
        return
    console.print(_report_table(summaries))


async def _report(registry: list[Candidate]) -> list[CandidateSummary]:
    async with async_session_maker() as session:
        return await candidate_report(session, registry)


def _report_table(summaries: list[CandidateSummary]) -> Table:
    table = Table()
    table.add_column("Candidate", style="cyan")
    table.add_column("Version", justify="right")
    table.add_column("Status")
    table.add_column("Settled", justify="right")
    table.add_column("With close", justify="right")
    table.add_column("No close", justify="right")
    table.add_column("Pending", justify="right")
    table.add_column("Staked", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("ROI", justify="right")
    table.add_column("Checkpoint n", justify="right")
    table.add_column("Mean EV", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("99% CI", justify="right")
    table.add_column("Verdict")
    for s in summaries:
        cp = s.checkpoint
        style = _VERDICT_STYLE[s.verdict]
        table.add_row(
            s.name,
            str(s.version),
            s.status if s.status is not None else "not current",
            str(s.settled_bets),
            str(s.bets_with_close),
            str(s.bets_without_close),
            str(s.pending_bets),
            f"{s.total_stake:.2f}",
            f"{s.pnl:+.2f}",
            f"{s.roi:+.1%}",
            str(cp.n) if cp else "-",
            f"{cp.mean_ev:+.4f}" if cp else "-",
            _interval(cp.ci95) if cp else "-",
            _interval(cp.ci99) if cp else "-",
            f"[{style}]{s.verdict}[/{style}]",
        )
    return table
