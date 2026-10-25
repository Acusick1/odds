"""CLI commands for the betting candidate registry."""

import asyncio
import dataclasses
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

import typer
from odds_analytics.candidate_backtest import BacktestResult, HashStatus, replay
from odds_analytics.candidate_parity import ParityReport, ParityStatus, check_parity
from odds_analytics.candidate_report import (
    CandidateSummary,
    Checkpoint,
    Verdict,
    candidate_report,
    family_size_at,
    family_starts,
    load_decision_starts,
)
from odds_core.candidates import (
    Candidate,
    CandidateStatus,
    MechanicalCandidate,
    candidates_dir,
    load_candidate_file,
    load_candidates,
)
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


def _backtest_verdict(verdict: Verdict) -> str:
    """Development data never promotes, so the promote bar is reported as met, not as a verdict."""
    return "meets_promote_bar" if verdict is Verdict.PROMOTE else verdict.value


def _interval(low_high: tuple[float, float]) -> str:
    return f"[{low_high[0]:+.3f}, {low_high[1]:+.3f}]"


def _promote_interval(cp: Checkpoint) -> str:
    return f"{cp.promote_level:.2%} {_interval(cp.promote_ci)}"


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
    table.add_column("Promote CI", justify="right")
    table.add_column("k", justify="right")
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
            _promote_interval(cp) if cp else "-",
            str(s.family_versions),
            f"[{style}]{s.verdict}[/{style}]",
        )
    return table


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]{message}[/red]", soft_wrap=True)
    return typer.Exit(1)


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise _fail(f"{value!r} is not a YYYY-MM-DD date") from exc


def _mechanical(candidate: Candidate) -> MechanicalCandidate:
    if not isinstance(candidate, MechanicalCandidate):
        raise _fail(f"candidate {candidate.name!r} is {candidate.kind}, not mechanical")
    return candidate


def _resolve(name_or_path: str) -> MechanicalCandidate:
    """A registered candidate by name, or a candidate file (possibly a draft) by path."""
    try:
        path = Path(name_or_path)
        if path.is_file():
            return _mechanical(load_candidate_file(path))
        registry = load_candidates(candidates_dir())
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    for candidate in registry:
        if candidate.name == name_or_path:
            return _mechanical(candidate)
    raise _fail(f"no registered candidate {name_or_path!r} and no such file")


def _checkpoint_row(table: Table, label: str, cp: Checkpoint | None) -> None:
    if cp is None:
        table.add_row(label, "-", "-", "-", "-")
        return
    table.add_row(label, str(cp.n), f"{cp.mean_ev:+.4f}", _interval(cp.ci95), _promote_interval(cp))


def _backtest_tables(result: BacktestResult) -> list[Table]:
    cov, summary = result.coverage, result.summary
    coverage = Table(
        title=f"{result.name} v{result.version} ({result.rule}), {result.start} to {result.end}"
    )
    coverage.add_column("Events", justify="right")
    coverage.add_column("With exchange", justify="right")
    coverage.add_column("With UK retail", justify="right")
    coverage.add_column("With a bet", justify="right")
    coverage.add_column("Bets", justify="right")
    coverage.add_column("With close", justify="right")
    coverage.add_column("No close", justify="right")
    coverage.add_column("Pending", justify="right")
    coverage.add_row(
        str(cov.events),
        str(cov.with_exchange),
        str(cov.with_retail),
        str(cov.with_bet),
        str(cov.bets),
        str(cov.bets_with_close),
        str(summary.bets_without_close),
        str(summary.pending_bets),
    )
    verdict = Table(title="Checkpoint verdict")
    verdict.add_column("Estimate")
    verdict.add_column("n", justify="right")
    verdict.add_column("Mean EV", justify="right")
    verdict.add_column("95% CI", justify="right")
    verdict.add_column("Promote CI", justify="right")
    _checkpoint_row(verdict, "Checkpoint", summary.checkpoint)
    _checkpoint_row(verdict, "All bets with close", result.overall)
    style = _VERDICT_STYLE[summary.verdict]
    label = _backtest_verdict(summary.verdict)
    if summary.verdict is Verdict.PROMOTE:
        label += " (development data, not a promotion)"
    verdict.caption = (
        f"[{style}]{label}[/{style}]  family versions k={summary.family_versions}  "
        f"staked {summary.total_stake:.2f}  P&L {summary.pnl:+.2f}  ROI {summary.roi:+.1%}"
    )
    return [coverage, verdict]


@app.command("backtest")
def backtest(
    candidate: Annotated[str, typer.Argument(help="Registered candidate name, or a YAML path")],
    start: Annotated[str, typer.Option("--from", help="First kick-off day, YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--to", help="Last kick-off day, YYYY-MM-DD")],
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Write the summary to this file")
    ] = None,
) -> None:
    """Replay a mechanical candidate over completed events and score it like the report.

    Development data can kill an idea, never promote one.
    """
    resolved = _resolve(candidate)
    try:
        # Not _load_registry: a rule edited since its hash was recorded must stay a warning.
        registry = load_candidates(candidates_dir())
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    first, last = _parse_day(start), _parse_day(end)
    if last < first:
        raise _fail("--to is before --from")

    async def run() -> BacktestResult:
        async with async_session_maker() as session:
            now = datetime.now(UTC)
            starts = family_starts(
                resolved.rule,
                resolved.sport,
                registry,
                await load_decision_starts(session),
                draft=(resolved.name, resolved.version),
                draft_start=now,
            )
            k = family_size_at(starts, now)
            return await replay(session, resolved, first, last, k)

    try:
        result = asyncio.run(run())
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    if result.hash_status is not HashStatus.MATCH:
        console.print(
            f"[yellow]rule_sha256 {result.hash_status}: this is a draft, not a registrable "
            f"candidate (`odds candidates rule-hash {result.rule}` prints the current hash)"
            f"[/yellow]",
            soft_wrap=True,
        )
    for table in _backtest_tables(result):
        console.print(table)
    if json_path is not None:
        payload = dataclasses.asdict(result)
        payload["summary"]["verdict"] = _backtest_verdict(result.summary.verdict)
        json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


@app.command("parity")
def parity(
    name: Annotated[str, typer.Argument(help="Registered candidate name")],
    version: Annotated[
        int | None, typer.Option("--version", help="Candidate version; default the registry's")
    ] = None,
) -> None:
    """Replay each recorded decision from its recorded input snapshots and compare the bets.

    Run on the production host: decisions and trades are not copied by `odds copy from-prod`.
    Exits 1 when a replay differs from the bets placed. Decisions whose replay from the
    database as it is now differs are listed as data differences and never affect the exit code.
    """
    resolved = _mechanical(
        next((c for c in _load_registry() if c.name == name), None) or _missing(name)
    )

    async def run() -> ParityReport:
        async with async_session_maker() as session:
            return await check_parity(session, resolved, version)

    try:
        report = asyncio.run(run())
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    if report.checked == 0:
        console.print(
            f"[yellow]{report.name} v{report.version}: no decisions to check "
            f"(this database holds none; run on the production host)[/yellow]",
            soft_wrap=True,
        )
        return
    console.print(
        f"{report.name} v{report.version}: {report.checked} decisions checked, "
        f"{len(report.of(ParityStatus.MATCH))} matched, "
        f"{len(report.of(ParityStatus.MISMATCH))} mismatches, "
        f"{len(report.data_differences)} data differences",
        soft_wrap=True,
    )
    for d in report.of(ParityStatus.MISMATCH):
        console.print(
            f"[red]mismatch decision {d.decision_id} ({d.event_id}, "
            f"{d.decision_time.isoformat()})[/red]\n  live:     {list(d.live)}\n"
            f"  replayed: {list(d.replayed)}"
            + (
                f"\n  recorded snapshots not in the database: {list(d.missing_snapshot_ids)}"
                if d.missing_snapshot_ids
                else ""
            ),
            soft_wrap=True,
        )
    for d in report.data_differences:
        console.print(
            f"[yellow]data difference decision {d.decision_id} ({d.event_id}, "
            f"{d.decision_time.isoformat()})[/yellow]\n  recorded inputs: {list(d.replayed)}\n"
            f"  database now:    {list(d.db_replayed)}",
            soft_wrap=True,
        )
    if not report.ok:
        raise typer.Exit(1)


def _missing(name: str) -> Candidate:
    raise _fail(f"no registered candidate {name!r}")
