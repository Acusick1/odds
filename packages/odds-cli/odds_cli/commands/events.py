"""CLI commands for events that exist more than once for one match."""

import asyncio
from collections import Counter

import typer
from odds_core.database import async_session_maker
from odds_core.models import Event
from odds_lambda.storage.event_merge import EventMerger, source_pair
from odds_lambda.storage.event_name_repair import repair_annotated_names
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


async def _twins(sport: str | None, apply: bool) -> None:
    async with async_session_maker() as session:
        merger = EventMerger(session)
        groups = await merger.find_twin_groups(sport)
        counts = Counter(" + ".join(s.name for s in source_pair(group)) for group in groups)

        table = Table(title="Matches with more than one event")
        table.add_column("Sources")
        table.add_column("Matches", justify="right")
        for sources, count in sorted(counts.items()):
            table.add_row(sources, str(count))
        console.print(table)
        console.print(f"Matches with twins: {len(groups)}")

        if not apply or not groups:
            return
        reports = [await merger.merge_group(group) for group in groups]
        await session.commit()
        remaining = len(await merger.find_twin_groups(sport))
        console.print(
            f"Merged {sum(len(r.merged_ids) for r in reports)} events into "
            f"{len(reports)}; matches with twins now: {remaining}"
        )


@app.command("merge-twins")
def merge_twins(
    sport: str | None = typer.Option(None, "--sport", "-s", help="Sport key; default all sports"),
    apply: bool = typer.Option(False, "--apply", help="Merge; without it only counts twins"),
) -> None:
    """Count, and with --apply merge, events that are twins of one match.

    Examples:
        odds events merge-twins --sport soccer_epl

        odds events merge-twins --sport soccer_epl --apply
    """
    asyncio.run(_twins(sport, apply))


async def _merge(keep_id: str, drop_id: str, apply: bool) -> None:
    async with async_session_maker() as session:
        if keep_id == drop_id:
            raise typer.BadParameter("keep_id and drop_id are the same event")
        keep = await session.get(Event, keep_id)
        drop = await session.get(Event, drop_id)
        if keep is None or drop is None:
            raise typer.BadParameter("both events must exist")
        if keep.sport_key != drop.sport_key:
            raise typer.BadParameter(f"different sports: {keep.sport_key} and {drop.sport_key}")
        if (keep.home_team, keep.away_team) != (drop.home_team, drop.away_team):
            raise typer.BadParameter("the events have different teams")
        console.print(
            f"Keep {keep.id} ({keep.commence_time}, {keep.status}); "
            f"merge in {drop.id} ({drop.commence_time}, {drop.status})"
        )
        if not apply:
            return
        report = await EventMerger(session).merge(keep, [drop])
        await session.commit()
        console.print(f"Merged {drop.id} into {report.kept_id}; rows moved: {report.rows_moved}")


@app.command("merge")
def merge(
    keep_id: str = typer.Argument(..., help="Event id that stays"),
    drop_id: str = typer.Argument(..., help="Event id merged into it and deleted"),
    apply: bool = typer.Option(False, "--apply", help="Merge; without it only prints the plan"),
) -> None:
    """Merge one event into another of the same sport and teams, e.g. a rescheduled match.

    Example:
        odds events merge KEEP_ID DROP_ID --apply
    """
    asyncio.run(_merge(keep_id, drop_id, apply))


async def _repair_names(sport: str, apply: bool) -> None:
    async with async_session_maker() as session:
        report = await repair_annotated_names(session, sport)
        statuses = {s.value: n for s, n in report.renamed_in_place_statuses.items()}
        console.print(
            f"Events with a non-canonical team name: {len(report.renamed_ids)}; "
            f"merged into a real event: {report.merged_count}; "
            f"renamed in place: {len(report.renamed_in_place_ids)} "
            f"({statuses or 'none'})"
        )
        for line in report.describe_merges():
            console.print(f"  {line}")
        if not apply:
            await session.rollback()
            console.print("Dry run: nothing written. Re-run with --apply.")
            return
        await session.commit()


@app.command("repair-names")
def repair_names(
    sport: str = typer.Option("baseball_mlb", "--sport", "-s", help="Sport key"),
    apply: bool = typer.Option(False, "--apply", help="Write; without it the plan is rolled back"),
) -> None:
    """Give events the canonical team names (no pitcher annotation, truncation or game suffix).

    Betfair-created events take the full runner names from their snapshots. Events that
    then duplicate a real event are merged into it; the rest are renamed in place so the
    results job can settle them. Doubleheader games stay separate. Safe to re-run.

    Example:
        odds events repair-names --sport baseball_mlb --apply
    """
    asyncio.run(_repair_names(sport, apply))
