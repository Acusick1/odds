"""CLI commands that move stored history into the new snapshot and price shape, and verify it."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import typer
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_lambda.shape_backfill.events import MergeDropsRefused
from odds_lambda.shape_backfill.run import BackfillRun, run_backfill
from odds_lambda.shape_backfill.snapshots import SplitSnapshotReferenced, capture_cutoff
from odds_lambda.shape_backfill.verify import VerifyReport, verify_shape
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

app = typer.Typer()
console = Console()


@asynccontextmanager
async def _read_only_session() -> AsyncIterator[AsyncSession]:
    """A session on connections whose every transaction is read-only, so a dry run cannot
    write even through a bug."""
    engine = create_async_engine(
        get_settings().database.url,
        poolclass=NullPool,
        connect_args={"server_settings": {"default_transaction_read_only": "on"}},
    )
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield session
    finally:
        await engine.dispose()


def _counts(title: str, rows: list[tuple[str, object]]) -> None:
    table = Table(title=title)
    table.add_column("What")
    table.add_column("Count", justify="right")
    for label, value in rows:
        table.add_row(label, str(value))
    console.print(table)


def _print_run(run: BackfillRun, apply: bool) -> None:
    verb = "written" if apply else "would write"
    console.print(
        f"cutoff: {run.cutoff.isoformat()} ({'apply' if apply else 'preflight, read-only'})"
    )

    merged = [m for m in run.twins if m.error is None]
    console.print(f"[bold]Twin groups[/bold]: {len(run.twins)} ({len(merged)} merged or mergeable)")
    for merge in run.twins:
        line = (
            f"  {'FAILED ' if merge.error else ''}keep {merge.kept_id} <- {', '.join(merge.merged_ids)}"
            f" kick-off {merge.commence_from.isoformat()} -> {merge.commence_to.isoformat()}"
            f" ({merge.kickoff_source.value if merge.kickoff_source else 'kept event'})"
        )
        if merge.rows_moved:
            line += f" moved={merge.rows_moved}"
        if merge.rows_deleted:
            line += f" deleted={merge.rows_deleted}"
        if merge.error:
            line += f" error={merge.error}"
        console.print(line)

    console.print(
        f"[bold]UNMATCHED[/bold]: {len(run.unmatched.marked)} {verb}, "
        f"{run.unmatched.already} already acknowledged"
        + (f": {', '.join(run.unmatched.marked)}" if run.unmatched.marked else "")
    )

    _counts(
        "External ids from stored data",
        [
            (f"{source.value} {outcome.value}", count)
            for (source, outcome), count in sorted(run.external_ids.outcomes.items())
        ],
    )
    for refused in run.external_ids.refused[:50]:
        console.print(
            f"  refused {refused.source.value} {refused.external_id} on {refused.event_id}"
        )
    if run.oddsportal_history is None:
        console.print("[bold]Historical OddsPortal ids[/bold]: skipped (no --oddsportal-dir)")
    else:
        h = run.oddsportal_history
        console.print(
            f"[bold]Historical OddsPortal ids[/bold]: {h.games} games with an id; "
            f"attached {h.attached}, unattached {h.unattached}"
        )

    s = run.snapshots
    _counts(
        f"Snapshots ({verb})",
        [
            ("legacy snapshots shaped", s.shaped),
            ("  of which split", s.split),
            ("new snapshots from splits", s.new_snapshots),
            ("empty envelopes (no market)", s.empty),
            ("price rows", s.prices),
            ("payloads", s.payloads),
            ("shaped snapshots given a payload", s.payload_only),
            ("duplicates over the new key (markets)", len(s.conflicts)),
            ("snapshots deleted (every market a duplicate)", s.deleted),
            ("odds rows moved", s.odds_moved),
            ("odds rows deleted", s.odds_deleted),
            ("odds rows of no stored market", s.odds_unplaced),
            ("referenced by decisions (split or deleted)", len(s.referenced)),
            ("malformed raw_data (left, not shaped)", len(s.malformed)),
            ("left pending (locked by a live writer)", s.pending_left),
        ]
        + [(f"kind {src.value}/{basis.value}", n) for (src, basis), n in sorted(s.by_kind.items())]
        + [(f"market {src.value}/{m.value}", n) for (src, m), n in sorted(s.by_market.items())]
        + [(f"issue {kind.value}", n) for kind, n in sorted(s.issues.items())],
    )
    for conflict in s.conflicts[:50]:
        console.print(
            f"  duplicate {conflict.event_id} {conflict.source.value} {conflict.market.value} "
            f"{conflict.snapshot_time.isoformat()}: snapshot {conflict.snapshot_id}, "
            f"kept {conflict.held_by}"
        )
    if s.unmatched_outcome_events:
        console.print(
            f"  {len(s.unmatched_outcome_events)} events have unmatched outcome names; "
            "run `odds events repair-names` first:"
        )
        for (event_id, source), n in s.unmatched_outcome_events.most_common(50):
            console.print(f"    {event_id} ({source.value}): {n}")
    if s.referenced:
        console.print(f"  [red]referenced snapshots: {s.referenced[:50]}[/red]")
    if s.malformed:
        console.print(f"  [red]malformed snapshots: {s.malformed[:50]}[/red]")

    p = run.legacy_polymarket
    _counts(
        f"Legacy Polymarket ({verb})",
        [
            ("games", p.pm_events),
            ("games with no event (skipped)", p.unmatched_games),
            ("snapshots", p.snapshots),
            ("price rows", p.prices),
            ("already stored", p.existing),
            ("rows with no probability strictly between 0 and 1", p.invalid_rows),
            ("rows at a time already taken by another market", p.duplicate_rows),
            ("moneyline markets not naming the event's teams", len(p.unmapped_markets)),
            ("order books archived", p.order_books),
            ("order books already archived", p.order_books_existing),
            ("errors", len(p.errors)),
        ]
        + [(f"{kind} markets skipped", n) for kind, n in sorted(p.markets_skipped.items())]
        + [(f"{kind} price rows skipped", n) for kind, n in sorted(p.rows_skipped.items())],
    )
    for error in p.errors[:20]:
        console.print(f"  [red]{error}[/red]")


async def _backfill(
    apply: bool,
    batch_size: int,
    limit: int | None,
    oddsportal_dir: Path | None,
    map_out: Path | None,
    allow_merge_drops: bool,
) -> None:
    started = time.monotonic()
    if apply:
        async with async_session_maker() as session:
            run = await run_backfill(
                session,
                apply=True,
                batch_size=batch_size,
                limit=limit,
                oddsportal_dir=oddsportal_dir,
                map_out=map_out,
                allow_merge_drops=allow_merge_drops,
            )
    else:
        async with _read_only_session() as session:
            run = await run_backfill(
                session,
                apply=False,
                batch_size=batch_size,
                limit=limit,
                oddsportal_dir=oddsportal_dir,
            )
    _print_run(run, apply)
    console.print(f"elapsed: {time.monotonic() - started:.0f} s")


@app.command("backfill-shape")
def backfill_shape(
    apply: Annotated[
        bool, typer.Option("--apply", help="Write; without it the run is a read-only preflight")
    ] = False,
    batch_size: Annotated[int, typer.Option(help="Snapshots per committed batch")] = 1000,
    limit: Annotated[
        int | None, typer.Option(help="Stop the snapshot step after this many")
    ] = None,
    oddsportal_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory of the historical OddsPortal scrapes whose match links give event ids"
        ),
    ] = None,
    map_out: Annotated[
        Path | None,
        typer.Option(help="Append each split snapshot's old and new ids here as JSON lines"),
    ] = None,
    allow_merge_drops: Annotated[
        bool,
        typer.Option(
            "--allow-merge-drops",
            help="Let twin merges delete clashing candidate decisions and paper trades",
        ),
    ] = False,
) -> None:
    """Move stored history into the new snapshot, price and event-identity shape.

    Merges event twins, marks events with no MLB game UNMATCHED, records the external ids
    stored data holds, fills every snapshot stored before the start (source, market, time basis,
    payload, price rows; Odds API snapshots split per market) and migrates the legacy Polymarket
    NBA prices. Idempotent and resumable. Without --apply it writes nothing and prints what it
    would do, duplicates over the new snapshot key included.

    Examples:
        odds data backfill-shape
        odds data backfill-shape --apply --map-out split-map.jsonl
    """
    if oddsportal_dir is not None and not oddsportal_dir.is_dir():
        raise typer.BadParameter(
            f"{oddsportal_dir} is not a directory", param_hint="--oddsportal-dir"
        )
    try:
        asyncio.run(_backfill(apply, batch_size, limit, oddsportal_dir, map_out, allow_merge_drops))
    except (SplitSnapshotReferenced, MergeDropsRefused) as exc:
        console.print(f"[red]Stopped: {exc}[/red]")
        raise typer.Exit(2) from exc


def _print_verify(report: VerifyReport) -> None:
    _counts(
        "Snapshots checked",
        [
            ("snapshots", report.snapshots),
            ("price rows", report.prices),
            ("odds rows", report.odds_rows),
            ("empty envelopes (no market)", report.empty),
            ("archived envelopes checked", report.payloads_checked),
            ("payloads not checked (run lists, unreferenced)", report.payloads_unchecked),
            ("odds rows without a snapshot", report.orphan_odds),
        ]
        + [
            (f"{source.value if source else 'no source'}/{market.value if market else '-'}", n)
            for (source, market), n in sorted(
                report.by_source.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
            )
        ],
    )
    if report.ok:
        console.print("[green]No mismatches[/green]")
        return
    _counts(
        "Mismatches",
        [(check.value, count) for check, count in sorted(report.mismatches.items())],
    )
    for mismatch in report.examples:
        console.print(
            f"  {mismatch.check.value} snapshot={mismatch.snapshot_id} "
            f"event={mismatch.event_id}: {mismatch.detail}"
        )


async def _verify(batch_size: int, examples: int) -> VerifyReport:
    started = time.monotonic()
    async with _read_only_session() as session:
        cutoff = await capture_cutoff(session)
        report = await verify_shape(
            session, cutoff=cutoff, batch_size=batch_size, max_examples=examples
        )
    console.print(f"cutoff: {cutoff.isoformat()}")
    _print_verify(report)
    console.print(f"elapsed: {time.monotonic() - started:.0f} s")
    return report


@app.command("verify-shape")
def verify_shape_command(
    batch_size: Annotated[int, typer.Option(help="Snapshots read per batch")] = 2000,
    examples: Annotated[int, typer.Option(help="Mismatches listed per check")] = 20,
) -> None:
    """Check every snapshot's new-shape columns and price rows against its raw_data.

    Read-only. Exits 1 when anything differs.

    Examples:
        odds data verify-shape
    """
    report = asyncio.run(_verify(batch_size, examples))
    if not report.ok:
        raise typer.Exit(1)
