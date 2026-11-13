"""CLI commands for the betting candidate registry."""

import asyncio
import dataclasses
import json
import subprocess
from collections.abc import Collection
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from odds_analytics import cell_selection, holdout, holdout_run
from odds_analytics.candidate_backtest import BacktestResult, Coverage, HashStatus, replay
from odds_analytics.candidate_parity import ParityReport, ParityStatus, check_parity
from odds_analytics.candidate_rehash import rehash as record_rehash
from odds_analytics.candidate_report import (
    CandidateSummary,
    Checkpoint,
    LeagueBreakdown,
    ScoredBet,
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
from odds_core.vocabulary import Source
from odds_lambda.candidate_rules import ModelRuleSpec
from odds_lambda.candidate_rules.pinning import rule_sha256, validate_candidate_rules
from odds_lambda.candidate_rules.registry import RULES
from rich.console import Console
from rich.table import Table
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

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
    table.add_column("Leagues")
    table.add_column("Version", justify="right")
    table.add_column("Status")
    table.add_column("Start date")
    table.add_column("Issue", justify="right")
    for c in candidates:
        table.add_row(
            c.name,
            c.kind,
            ", ".join(c.sports),
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


def _bet_json(bet: ScoredBet) -> dict[str, Any]:
    """A placed bet as ``--json`` writes it; ``p_close``, ``ev`` and ``close_source`` are null
    without a close."""
    return {
        "event_id": bet.event_id,
        "league": bet.sport_key,
        "kickoff": bet.commence_time,
        "decision_time": bet.placed_at,
        "outcome": bet.selection,
        "line": bet.line,
        "bookmaker": bet.bookmaker,
        "price": bet.price_decimal,
        "p_close": bet.p_close,
        "ev": bet.ev,
        "close_source": bet.close_source,
    }


def _backtest_json(result: BacktestResult) -> dict[str, Any]:
    """The ``--json`` payload: each summary carries its registry ``status`` and a backtest
    verdict, not the ``registered`` entry the status comes from, and each placed bet."""
    payload = dataclasses.asdict(dataclasses.replace(result, comparison=None, bets=()))
    summary = payload["summary"]
    del summary["registered"]
    summary["status"] = result.summary.status
    summary["verdict"] = _backtest_verdict(result.summary.verdict)
    payload["comparison"] = None if result.comparison is None else _backtest_json(result.comparison)
    payload["bets"] = [_bet_json(b) for b in result.bets]
    return payload


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
    for s in summaries:
        if len(s.leagues) > 1:
            console.print(_league_table(f"{s.name} v{s.version}", s.leagues))


async def _report(registry: list[Candidate]) -> list[CandidateSummary]:
    async with async_session_maker() as session:
        return (await candidate_report(session, registry)).summaries


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


def _league_table(
    label: str,
    leagues: tuple[LeagueBreakdown, ...],
    coverage: dict[str, Coverage] | None = None,
) -> Table:
    table = Table(title=f"{label} per league, for information: the verdict is the pooled series'")
    table.add_column("League")
    if coverage is not None:
        table.add_column("Decisions", justify="right")
        table.add_column("With input", justify="right")
        table.add_column("Bets", justify="right")
    table.add_column("Settled", justify="right")
    table.add_column("With close", justify="right")
    table.add_column("Mean EV", justify="right")
    table.add_column("P&L", justify="right")
    scored = {b.sport_key: b for b in leagues}
    for league in coverage if coverage is not None else scored:
        b = scored.get(league)
        cells = [str(b.settled_bets), str(b.bets_with_close)] if b else ["0", "0"]
        cells += [
            f"{b.mean_ev:+.4f}" if b and b.mean_ev is not None else "-",
            f"{b.pnl:+.2f}" if b else "-",
        ]
        if coverage is not None:
            cov = coverage[league]
            cells = [str(cov.decisions), str(cov.with_input), str(cov.bets), *cells]
        table.add_row(league, *cells)
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


def _close_sources(closes: dict[Source, int]) -> str:
    return ", ".join(f"{source} {n}" for source, n in closes.items()) or "none"


def _checkpoint_row(table: Table, label: str, cp: Checkpoint | None) -> None:
    if cp is None:
        table.add_row(label, "-", "-", "-", "-")
        return
    table.add_row(label, str(cp.n), f"{cp.mean_ev:+.4f}", _interval(cp.ci95), _promote_interval(cp))


def _backtest_tables(result: BacktestResult) -> list[Table]:
    cov, summary = result.coverage, result.summary
    leads = ", ".join(map(str, cast("list[int]", result.params["minutes_before_kickoff"])))
    coverage = Table(
        title=(
            f"{result.name} v{result.version} ({result.rule}), {result.start} to {result.end}, "
            f"decisions at {leads} min before kick-off, "
            f"max input age {result.params['max_input_age_minutes']} min"
        )
    )
    coverage.add_column("Events", justify="right")
    coverage.add_column("Decisions", justify="right")
    coverage.add_column("With input", justify="right")
    coverage.add_column("With exchange", justify="right")
    coverage.add_column("With Polymarket", justify="right")
    coverage.add_column("With UK retail", justify="right")
    coverage.add_column("With a bet", justify="right")
    coverage.add_column("Bets", justify="right")
    coverage.add_column("With close", justify="right")
    coverage.add_column("No close", justify="right")
    coverage.add_column("Pending", justify="right")
    coverage.add_row(
        str(cov.events),
        str(cov.decisions),
        str(cov.with_input),
        str(cov.with_exchange),
        str(cov.with_polymarket),
        str(cov.with_retail),
        str(cov.with_bet),
        str(cov.bets),
        str(cov.bets_with_close),
        str(summary.bets_without_close),
        str(summary.pending_bets),
    )
    coverage.caption = f"Bets with a close by close source: {_close_sources(cov.closes)}"
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


def _refit_table(result: BacktestResult) -> Table:
    table = Table(
        title=(
            f"Model refits: {result.range_events} matches in range, the first "
            f"{result.range_events - len(result.decided_event_ids or ())} training only"
        )
    )
    table.add_column("Refit", justify="right")
    table.add_column("At")
    table.add_column("Training matches", justify="right")
    table.add_column("Training rows", justify="right")
    table.add_column("Scored matches", justify="right")
    for refit in result.refits:
        table.add_row(
            str(refit.index),
            f"{refit.at:%Y-%m-%d %H:%M}",
            str(refit.training_matches),
            str(refit.training_rows),
            str(len(refit.event_ids)),
        )
    return table


def _tuning_table(result: BacktestResult) -> Table:
    tuning = result.tuning
    assert tuning is not None  # noqa: S101
    table = Table(
        title=(
            f"Hyperparameters tuned on the training matches: {tuning.trials} trials, "
            f"{tuning.folds} walk-forward folds"
        )
    )
    table.add_column("Parameter")
    table.add_column("Value", justify="right")
    for name, value in tuning.params.items():
        table.add_row(name, f"{value:.4g}" if isinstance(value, float) else str(value))
    versions = ", ".join(f"{k} {v}" for k, v in (result.library_versions or {}).items())
    table.caption = (
        f"{tuning.tuning_matches} tuning matches, validation rows per fold "
        f"{list(tuning.validation_rows)}; mean fold weighted MSE {tuning.cv_wmse:.6f}, "
        f"baseline {tuning.baseline_cv_wmse:.6f}; {versions}"
    )
    return table


def _comparison_table(result: BacktestResult, comparison: BacktestResult) -> Table:
    table = Table(title="Mean EV on the same decided matches, every bet with a close")
    table.add_column("Candidate")
    table.add_column("Events", justify="right")
    table.add_column("Bets with close", justify="right")
    table.add_column("Mean EV", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Verdict")
    for r in (result, comparison):
        cp = r.overall
        table.add_row(
            f"{r.name} v{r.version}",
            str(r.coverage.events),
            str(r.coverage.bets_with_close),
            f"{cp.mean_ev:+.4f}" if cp else "-",
            _interval(cp.ci95) if cp else "-",
            _backtest_verdict(r.summary.verdict),
        )
    return table


def _print_backtest(result: BacktestResult) -> None:
    if result.hash_status is not HashStatus.MATCH:
        console.print(
            f"[yellow]{result.name}: rule_sha256 {result.hash_status}: this is a draft, not a "
            f"registrable candidate (`odds candidates rule-hash {result.rule}` prints the "
            f"current hash)[/yellow]",
            soft_wrap=True,
        )
    if result.tuning is not None:
        console.print(_tuning_table(result))
    if result.decided_event_ids is not None:
        console.print(_refit_table(result))
    for table in _backtest_tables(result):
        console.print(table)
    if result.league_coverage is not None and len(result.league_coverage) > 1:
        console.print(
            _league_table(
                f"{result.name} v{result.version}", result.summary.leagues, result.league_coverage
            )
        )


async def _family_size(
    session: AsyncSession, candidate: MechanicalCandidate, registry: list[Candidate], now: datetime
) -> int:
    starts = family_starts(
        candidate,
        registry,
        await load_decision_starts(session),
        draft=(candidate.name, candidate.version),
        draft_start=now,
    )
    return family_size_at(starts, now)


@app.command("backtest")
def backtest(
    candidate: Annotated[str, typer.Argument(help="Registered candidate name, or a YAML path")],
    start: Annotated[str, typer.Option("--from", help="First kick-off day, YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--to", help="Last kick-off day, YYYY-MM-DD")],
    json_path: Annotated[
        Path | None, typer.Option("--json", help="Write the summary to this file")
    ] = None,
    compare: Annotated[
        str | None,
        typer.Option(
            "--compare",
            help="Also replay this candidate (name or YAML path) on the same decided matches",
        ),
    ] = None,
) -> None:
    """Replay a mechanical candidate over completed events and score it like the report.

    Development data can kill an idea, never promote one.
    """
    resolved = _resolve(candidate)
    other = _resolve(compare) if compare is not None else None
    if other is not None and isinstance(RULES.get(other.rule), ModelRuleSpec):
        raise _fail(
            f"--compare {other.name!r} is a model rule, which trains on its own range and so "
            "cannot be replayed on another candidate's matches"
        )
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
            k = await _family_size(session, resolved, registry, now)
            result = await replay(session, resolved, first, last, k)
            if other is None:
                return result
            compared = await replay(
                session,
                other,
                first,
                last,
                await _family_size(session, other, registry, now),
                only_events=result.decided_event_ids,
            )
            return dataclasses.replace(result, comparison=compared)

    try:
        result = asyncio.run(run())
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    _print_backtest(result)
    if result.comparison is not None:
        console.print(f"[bold]Compared with {result.comparison.name}[/bold]")
        _print_backtest(result.comparison)
        console.print(_comparison_table(result, result.comparison))
    if json_path is not None:
        payload = _backtest_json(result)
        json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


@app.command("holdout")
def holdout_command(
    selection: Annotated[
        Path,
        typer.Argument(help="candidates/selections/<issue>-<market>-v<version>.yaml, committed"),
    ],
) -> None:
    """Run a pre-registered idea's selected cell once on the sealed holdout and print the gate.

    The argument is the selection file; its grid file, `candidates/preregistrations/` of the
    same name, is read with it.

    The only command that replays the holdout. It reserves the idea's version in
    candidates/holdout.jsonl before reading any holdout data: a failure after that leaves the
    reservation, and the run is spent. A holdout result never promotes.
    """

    async def run() -> holdout_run.HoldoutRun:
        async with async_session_maker() as session:
            return await holdout_run.run_holdout(session, selection, candidates_dir())

    try:
        done = asyncio.run(run())
    except holdout.HoldoutRefused as exc:
        raise _fail(str(exc)) from exc
    except Exception as exc:
        raise _fail(
            f"the run failed after reserving its version, which is spent: {exc!r}. "
            "Commit the reservation in candidates/holdout.jsonl."
        ) from exc
    table = Table(title=f"Holdout of #{done.row.issue} {done.row.name} v{done.row.version}")
    for column in ("Window", "Events", "Decisions", "Bets", "With close", "Close sources"):
        table.add_column(
            column, justify="left" if column in ("Window", "Close sources") else "right"
        )
    for league in done.leagues:
        cov = league.coverage
        table.add_row(
            str(league.window),
            str(cov.events),
            str(cov.decisions),
            str(cov.bets),
            str(cov.bets_with_close),
            _close_sources(cov.closes),
        )
    console.print(table)
    if done.row.leagues_without_window:
        console.print(
            f"[yellow]No sealed window, so no holdout bets: "
            f"{', '.join(done.row.leagues_without_window)}.[/yellow]",
            soft_wrap=True,
        )
    _print_gate(done.row)


@app.command("select")
def select_command(
    preregistration: Annotated[
        Path,
        typer.Argument(help="candidates/preregistrations/<issue>-<market>-v<version>.yaml"),
    ],
    json_path: Annotated[
        Path | None,
        typer.Option("--json", help="Write every cell's score and placed bets to this file"),
    ] = None,
) -> None:
    """Pick a pre-registered grid's cell on development data by round 1's selection rule.

    Replays every cell as one pooled draft over the file's leagues on each development block,
    prints each cell's score and, when a cell is eligible, writes the picked cell to
    candidates/selections/ under the grid file's name. No eligible cell kills the idea, and
    nothing is written. Reads the database only.
    """
    directory = candidates_dir()
    today = datetime.now(UTC).date()

    async def run() -> cell_selection.GridSelection:
        async with async_session_maker() as session:
            await session.execute(text("SET TRANSACTION READ ONLY"))
            return await cell_selection.select_cell(session, preregistration, directory, today)

    try:
        result = asyncio.run(run())
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    console.print(_selection_table(result))
    if json_path is not None:
        payload = _selection_json(result)
        json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    prereg, picked, chosen = result.prereg, result.picked, result.selection
    if picked is None or chosen is None:
        console.print(
            f"[red]No eligible cell: #{prereg.issue} {prereg.name} v{prereg.version} is killed on "
            "development data and gets no holdout run.[/red]",
            soft_wrap=True,
        )
        return
    try:
        path = cell_selection.write_selection(chosen, directory)
    except FileExistsError as exc:
        raise _fail(f"{exc.filename} already exists; a selection is never changed") from exc
    console.print(
        f"[green]Picked cell {picked.index + 1}: {_cell_label(picked.cell)}[/green]. Wrote "
        f"{path}: commit it alone, then run `odds candidates holdout {path}`.",
        soft_wrap=True,
    )


def _cell_label(cell: dict[str, object], keys: Collection[str] | None = None) -> str:
    """``cell`` as ``key=value`` pairs, of ``keys`` only when given."""
    return " ".join(
        f"{k}={json.dumps(v, separators=(',', ':'))}"
        for k, v in cell.items()
        if keys is None or k in keys
    )


def _ev(value: float | None) -> str:
    return "-" if value is None else f"{value:+.4f}"


def _selection_table(result: cell_selection.GridSelection) -> Table:
    prereg = result.prereg
    cells = [s.cell for s in result.scores]
    varying = {k for k in cells[0] if any(c[k] != cells[0][k] for c in cells)}
    fixed = {k for k in cells[0] if k not in varying}
    table = Table(
        title=(
            f"#{prereg.issue} {prereg.name} v{prereg.version} ({prereg.rule}, {prereg.market}): "
            "bets with a close per cell on development data"
            + (f", every cell {_cell_label(cells[0], fixed)}" if fixed else "")
        )
    )
    for column in ("#", "Cell", "Bets", "Events"):
        table.add_column(column, justify="left" if column == "Cell" else "right")
    for block in result.blocks:
        table.add_column(f"{block.name}\nbets / mean EV", justify="right")
    table.add_column("Mean EV", justify="right")
    table.add_column(f"{cell_selection.INTERVAL_LEVEL:.0%} lower", justify="right")
    if result.longshots:
        table.add_column("Power-devig mean EV", justify="right")
    table.add_column("Close sources")
    table.add_column("Eligible")
    for score in result.scores:
        picked = result.picked is not None and score.index == result.picked.index
        row = [
            str(score.index + 1),
            _cell_label(score.cell, varying),
            str(score.bets),
            str(score.events),
            *(f"{b.bets} / {_ev(b.mean_ev)}" for b in score.blocks),
            _ev(score.mean_ev),
            _ev(score.lower_bound),
            *([_ev(score.power_mean_ev)] if result.longshots else []),
            _close_sources(score.closes),
            "picked" if picked else "yes" if score.eligible else "no",
        ]
        table.add_row(*row, style="green" if picked else None)
    table.caption = (
        f"Eligible: at least {cell_selection.MIN_BETS} bets in {cell_selection.MIN_EVENTS} "
        f"events, mean EV above 0 in every block with {cell_selection.MIN_BLOCK_BETS} or more"
        + (
            f". The power-devig mean EV is for information (bets above "
            f"{cell_selection.LONGSHOT_PRICE}) and never changes the pick"
            if result.longshots
            else ""
        )
    )
    return table


def _selection_json(result: cell_selection.GridSelection) -> dict[str, Any]:
    """The ``select --json`` payload: every cell's score in grid order with its placed bets,
    each labelled with its block."""
    cells = []
    for score, cell in zip(result.scores, result.replays, strict=True):
        cells.append(
            {
                **dataclasses.asdict(score),
                "placed": [
                    {"block": block.name, **_bet_json(bet)}
                    for block, bets in cell.blocks
                    for bet in bets
                ],
            }
        )
    return {
        "preregistration": result.prereg.stem,
        "grid_sha256": result.grid_sha256,
        "selection_rule": result.prereg.selection_rule,
        "blocks": [dataclasses.asdict(b) for b in result.blocks],
        "longshots": result.longshots,
        "picked": None if result.picked is None else result.picked.index,
        "cells": cells,
    }


def _print_gate(row: holdout.HoldoutResult) -> None:
    g = row.result
    style = "green" if g.verdict is holdout.GateVerdict.PASS else "red"
    if g.mean_ev is None:
        scored = "no bets with a close"
    else:
        scored = (
            f"pooled n={g.bets} mean EV {g.mean_ev:+.4f} "
            f"95% CI [{g.ci_lower:+.4f}, {g.ci_upper:+.4f}]{' (killed)' if g.killed else ''}"
        )
    console.print(f"[{style}]Holdout gate: {g.verdict.value}[/{style}]  {scored}", soft_wrap=True)
    if g.low_bets:
        console.print(
            f"[yellow]Under {holdout.MIN_GATE_BETS} holdout bets: not killed alone passes. "
            f"Flag this on issue {row.issue}.[/yellow]",
            soft_wrap=True,
        )
    console.print(
        f"Recorded in candidates/{holdout.HOLDOUT_LEDGER}: commit it. A holdout result never "
        "promotes; the forward test decides.",
        soft_wrap=True,
    )


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
            )
            + (
                f"\n  recorded events not in the database: {list(d.missing_event_ids)}"
                if d.missing_event_ids
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


@app.command("rehash")
def rehash(
    rule: Annotated[str, typer.Argument(help="Rule name")],
    database_url: Annotated[
        str, typer.Option("--database-url", help="Database to replay against, read-only")
    ],
    start: Annotated[str, typer.Option("--from", help="First kick-off day, YYYY-MM-DD")],
    end: Annotated[str, typer.Option("--to", help="Last kick-off day, YYYY-MM-DD")],
    link: Annotated[str, typer.Option("--evidence", help="The PR or issue of the change")],
    base: Annotated[
        str | None,
        typer.Option(
            "--base",
            help="Revision whose code recorded the old hash; default the merge-base with origin/main",
        ),
    ] = None,
) -> None:
    """Record a rule's new hash when the change leaves every candidate's bets the same.

    Replays every registered candidate of the rule under the base revision's code and the
    committed working tree's, on one read-only database snapshot, and checks parity; refuses
    unless the candidates record the base's hash, the replays are byte-identical with decisions
    and bets, and parity holds. Then updates each candidate file's rule_sha256 and appends the
    rehash ledger.
    """
    if rule not in RULES:
        raise _fail(f"unknown rule {rule!r}; known rules: {', '.join(sorted(RULES))}")
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    revision = (
        base
        or subprocess.run(
            ["git", "merge-base", "HEAD", "origin/main"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    try:
        result = asyncio.run(
            record_rehash(
                root, rule, database_url, _parse_day(start), _parse_day(end), link, revision
            )
        )
    except subprocess.CalledProcessError as exc:
        raise _fail(f"{exc}\n{exc.stderr}") from exc
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    if result.refused is not None:
        raise _fail(f"{rule}: not rehashed, {result.refused}")
    if result.rehash is None:
        console.print(f"{rule}: every candidate already records {result.new}", soft_wrap=True)
        return
    entry = result.rehash
    console.print(
        f"{rule}: {entry.old} -> {entry.new} (base {entry.base}, head {entry.head}, "
        f"{entry.start}..{entry.end} on {entry.database}, replay sha256 {entry.replay_sha256})",
        soft_wrap=True,
    )
    for c in entry.candidates:
        console.print(
            f"  {c.name} v{c.version}: {c.decisions} decisions, {c.bets} bets, "
            f"parity {c.parity_matched}/{c.parity_total}",
            soft_wrap=True,
        )


def _missing(name: str) -> Candidate:
    raise _fail(f"no registered candidate {name!r}")
