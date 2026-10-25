"""CLI commands that repair events: duplicates, team names, scores and MLB gamePks."""

import asyncio
from collections import Counter
from datetime import UTC, date, datetime

import typer
from odds_core.database import async_session_maker
from odds_core.models import Event
from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher
from odds_lambda.storage.event_merge import EventMerger, source_pair
from odds_lambda.storage.event_name_repair import repair_team_names
from odds_lambda.storage.game_pk_backfill import (
    AssignmentOutcome,
    GamePkRefused,
    OutcomeStatus,
    ScoreCarry,
    Settlement,
    delete_ghosts,
    mark_no_game_pk,
    plan_game_pks,
    plan_hand_assignment,
    run_assignments,
)
from odds_lambda.storage.mlb_score_repair import ScoreOutcome, repair_scores
from odds_lambda.storage.same_time_pairs import (
    GroupAction,
    GroupResult,
    delete_same_time_pairs,
    find_same_time_groups,
    unlinked_odds,
)
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
        report = await repair_team_names(session, sport)
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


async def _repair_scores(apply: bool) -> None:
    async with async_session_maker() as session, MlbStatsFetcher() as fetcher:
        checks = await repair_scores(session, fetcher, apply)
        counts = Counter(check.outcome for check in checks)
        console.print(
            f"FINAL baseball_mlb events: {len(checks)}; "
            + "; ".join(f"{o.value}: {counts[o]}" for o in ScoreOutcome)
        )
        for check in checks:
            event = check.event
            label = f"{event.id} {event.commence_time:%Y-%m-%d %H:%M} {event.away_team} @ {event.home_team}"
            if check.outcome is ScoreOutcome.MISMATCH and check.mlb_score is not None:
                console.print(
                    f"  MISMATCH {label}: stored {event.home_score}-{event.away_score}, "
                    f"MLB {check.mlb_score[0]}-{check.mlb_score[1]} (game {event.mlb_game_pk})"
                )
            elif check.outcome is ScoreOutcome.UNMATCHED:
                console.print(f"  UNMATCHED {label}: {check.reason}")
        if not apply:
            await session.rollback()
            console.print("Dry run: nothing written. Re-run with --apply.")
            return
        await session.commit()
        console.print(f"Corrected {counts[ScoreOutcome.MISMATCH]} scores.")


@app.command("repair-scores")
def repair_scores_command(
    apply: bool = typer.Option(False, "--apply", help="Write; without it only the diff is shown"),
) -> None:
    """Compare FINAL MLB events' scores with the MLB Stats API and correct the wrong ones.

    Pairs each event with the game its gamePk names and reports mismatches, plus events
    without a gamePk or whose game is not Final, which it never changes. --apply writes
    home_score and away_score only, in one transaction. Safe to re-run.

    Example:
        odds events repair-scores --apply
    """
    asyncio.run(_repair_scores(apply))


def _counts(counts: dict[str, int]) -> str:
    return ",".join(f"{table}:{n}" for table, n in sorted(counts.items())) or "none"


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def _outcome_line(tag: str, outcome: AssignmentOutcome) -> str:
    line = f"{tag} game_pk={outcome.game_pk} kept={outcome.kept_id}"
    if outcome.rule is not None:
        line += f" rule={outcome.rule.value}"
    merged = ",".join(outcome.merged_ids) or "none"
    if outcome.status is OutcomeStatus.REFUSED:
        return (
            f"{line} merged={merged} reason={outcome.detail}"
            f" would_delete={_counts(outcome.rows_deleted)}"
        )
    if outcome.status is OutcomeStatus.FAILED:
        return f"{line} merged={merged} error={outcome.detail!r}"
    if outcome.merged_ids:
        line += f" merged={merged} rows={_counts(outcome.rows_moved)}"
    if outcome.moved_from is not None and outcome.moved_to is not None:
        line += f" moved_from={_utc(outcome.moved_from)}→{_utc(outcome.moved_to)}"
    else:
        line += " moved=no"
    if outcome.start_tbd:
        line += " start=tbd"
    line += f" score={outcome.score.value}"
    if outcome.score_detail:
        line += f":{outcome.score_detail}"
    line += f" settled={outcome.settled.value if outcome.settled else 'no'}"
    if outcome.settled_detail:
        line += f":{outcome.settled_detail}"
    return (
        f"{line} same_time_pairs_identical={outcome.pairs.identical}"
        f" same_time_pairs_conflicting={outcome.pairs.conflicting}"
    )


def _print_outcomes(outcomes: list[AssignmentOutcome]) -> None:
    done = [o for o in outcomes if o.status is OutcomeStatus.APPLIED]
    for tag, group in (
        ("SINGLE", [o for o in done if not o.merged_ids]),
        ("MERGE", [o for o in done if o.merged_ids and not o.read_only_rows]),
        ("MERGE_READ_ONLY", [o for o in done if o.merged_ids and o.read_only_rows]),
        ("REFUSED", [o for o in outcomes if o.status is OutcomeStatus.REFUSED]),
        ("FAILED", [o for o in outcomes if o.status is OutcomeStatus.FAILED]),
    ):
        for outcome in group:
            typer.echo(_outcome_line(tag, outcome))


async def _assign_game_pks(since: date | None, apply: bool, delete_ghosts_: bool) -> None:
    async with async_session_maker() as session, MlbStatsFetcher() as fetcher:
        plan = await plan_game_pks(session, fetcher, since, datetime.now(UTC).date())
        outcomes = await run_assignments(session, plan.assignments, commit=apply)
        deleted = 0
        if apply and delete_ghosts_:
            deleted = await delete_ghosts(session, plan.ghost_ids)
            await session.commit()
        else:
            await session.rollback()

        done = [o for o in outcomes if o.status is OutcomeStatus.APPLIED]
        merges = [o for o in done if o.merged_ids]
        by_status = Counter(o.status for o in outcomes)
        unresolved = Counter(u.reason.value for u in plan.unresolved)
        typer.echo(
            f"MLB events without a gamePk: {plan.pending} (since {since or 'any date'})\n"
            f"singles: {len(done) - len(merges)}\n"
            f"merge groups: {len(merges)} (read-only: {sum(1 for o in merges if o.read_only_rows)};"
            f" events merged: {sum(len(o.merged_ids) for o in merges)})\n"
            f"settled from MLB: {sum(1 for o in done if o.settled is Settlement.MLB)};"
            f" score conflicts: {sum(1 for o in done if o.score is ScoreCarry.CONFLICT or o.settled is Settlement.CONFLICT)}\n"
            f"refused: {by_status[OutcomeStatus.REFUSED]}\n"
            f"failed: {by_status[OutcomeStatus.FAILED]}\n"
            f"unresolved: {len(plan.unresolved)} ("
            + (", ".join(f"{r}: {n}" for r, n in sorted(unresolved.items())) or "none")
            + ")\n"
            f"ghosts: {len(plan.ghost_ids)}\n"
            f"marked unassignable: {len(plan.marked)}"
        )
        _print_outcomes(outcomes)
        for item in plan.unresolved:
            detail = f" detail={item.detail!r}" if item.detail else ""
            typer.echo(f"UNRESOLVED event={item.event_id} reason={item.reason.value}{detail}")
        for event_id in plan.ghost_ids:
            typer.echo(f"GHOST event={event_id}")
        for event_id, reason in sorted(plan.marked.items()):
            typer.echo(f"UNASSIGNABLE event={event_id} reason={reason!r}")
        if not apply:
            typer.echo("Dry run: nothing written. Re-run with --apply.")
            if delete_ghosts_:
                typer.echo(
                    f"--apply --delete-ghosts would delete {len(plan.ghost_ids)} ghost events."
                )
        elif delete_ghosts_:
            typer.echo(f"Deleted {deleted} ghost events.")


@app.command("assign-game-pks")
def assign_game_pks(
    apply: bool = typer.Option(False, "--apply", help="Write; without it the plan is rolled back"),
    since: str | None = typer.Option(
        None, "--since", help="Only events on or after this MLB date (YYYY-MM-DD)"
    ),
    delete_ghosts_: bool = typer.Option(
        False,
        "--delete-ghosts",
        help="With --apply, delete events with no MLB game that nothing references",
    ),
) -> None:
    """Give every MLB event without a gamePk its MLB Stats API gamePk, merging events of one game.

    Prints a summary, then one line per assignment (SINGLE, MERGE, MERGE_READ_ONLY for merges
    that move paper trades, match briefs or candidate decisions; REFUSED for merges that would
    delete a candidate decision or paper trade; FAILED), then UNRESOLVED, GHOST and UNASSIGNABLE
    events. --apply commits each game in its own transaction. Safe to re-run.

    Examples:
        odds events assign-game-pks --since 2026-03-01

        odds events assign-game-pks --since 2026-03-01 --apply
    """
    try:
        since_date = date.fromisoformat(since) if since else None
    except ValueError as e:
        raise typer.BadParameter(f"--since must be YYYY-MM-DD: {since}") from e
    asyncio.run(_assign_game_pks(since_date, apply, delete_ghosts_))


async def _set_game_pk(event_id: str, game_pk: int, merge: bool, apply: bool) -> None:
    async with async_session_maker() as session, MlbStatsFetcher() as fetcher:
        try:
            assignment = await plan_hand_assignment(session, fetcher, event_id, game_pk, merge)
        except GamePkRefused as e:
            raise typer.BadParameter(str(e)) from e
        [outcome] = await run_assignments(session, [assignment], commit=apply)
        if not apply:
            await session.rollback()
        _print_outcomes([outcome])
        if outcome.status is not OutcomeStatus.APPLIED:
            raise typer.Exit(1)
        if not apply:
            typer.echo("Dry run: nothing written. Re-run with --apply.")


@app.command("set-game-pk")
def set_game_pk(
    event_id: str = typer.Argument(..., help="MLB event without a gamePk"),
    game_pk: int = typer.Argument(..., help="MLB Stats API gamePk of its game"),
    merge: bool = typer.Option(
        False, "--merge", help="Merge the event into the event already holding the gamePk"
    ),
    apply: bool = typer.Option(False, "--apply", help="Write; without it the plan is rolled back"),
) -> None:
    """Give one MLB event a gamePk by hand, moving it to the game's scheduled start.

    Refuses a game of other teams, and a gamePk another event holds unless --merge.

    Example:
        odds events set-game-pk op_live_PITPIR_MILBRE_2026-07-10T2240 823357 --merge --apply
    """
    asyncio.run(_set_game_pk(event_id, game_pk, merge, apply))


async def _mark_unassignable(event_id: str, reason: str, apply: bool) -> None:
    async with async_session_maker() as session:
        try:
            await mark_no_game_pk(session, event_id, reason)
        except GamePkRefused as e:
            raise typer.BadParameter(str(e)) from e
        typer.echo(f"UNASSIGNABLE event={event_id} reason={reason.strip()!r}")
        if not apply:
            await session.rollback()
            typer.echo("Dry run: nothing written. Re-run with --apply.")
            return
        await session.commit()


@app.command("mark-unassignable")
def mark_unassignable(
    event_id: str = typer.Argument(..., help="MLB event without a gamePk"),
    reason: str = typer.Option(..., "--reason", help="Why the event has no MLB game"),
    apply: bool = typer.Option(False, "--apply", help="Write; without it nothing is stored"),
) -> None:
    """Record that an MLB event has no gamePk, so the missing_game_pk warning leaves it out.

    Setting a gamePk later clears the mark.

    Example:
        odds events mark-unassignable op_live_ATLBRA_SANGIA_2026-06-17T1605 --reason "no such game" --apply
    """
    asyncio.run(_mark_unassignable(event_id, reason, apply))


def _ids(ids: tuple[int, ...]) -> str:
    return ",".join(map(str, ids)) or "none"


def _pair_line(result: GroupResult) -> str:
    group = result.group
    line = (
        f"{group.action.value.upper()} event={group.event_id} sport={group.sport_key}"
        f" time={_utc(group.snapshot_time)}"
        f" sources={','.join(s or 'odds_api' for s in group.sources)}"
        f" identical={group.pairs.identical} conflicting={group.pairs.conflicting}"
        f" snapshots={_ids(group.snapshot_ids)}"
    )
    if group.action is GroupAction.REFUSED:
        return f"{line} used_by_decision={_ids(group.referenced_ids)}"
    return (
        f"{line} deleted={_ids(group.deleted_ids)}"
        f" odds_deleted={result.odds_deleted} odds_linked={result.odds_linked}"
    )


async def _delete_same_time_pairs(apply: bool) -> None:
    async with async_session_maker() as session:
        groups = await find_same_time_groups(session)
        results = await delete_same_time_pairs(session, groups)
        unlinked = await unlinked_odds(session)
        if apply:
            await session.commit()
        else:
            await session.rollback()

        by_action = Counter(r.group.action for r in results)
        deleted = {i for r in results for i in r.group.deleted_ids}
        conflicting = deleted.intersection(i for r in results for i in r.group.conflicting_ids)
        typer.echo(
            f"event-times with same-time pairs: {len(results)}\n"
            + "".join(f"{action.value}: {by_action[action]}\n" for action in GroupAction)
            + f"snapshots deleted: {len(deleted)} (conflicting: {len(conflicting)},"
            f" duplicates: {len(deleted - conflicting)})\n"
            f"odds rows deleted: {sum(r.odds_deleted for r in results)}\n"
            f"odds rows linked to their snapshot: {sum(r.odds_linked for r in results)}\n"
            f"odds rows without a snapshot: {unlinked.total}"
            f" (orphans no snapshot quotes: {unlinked.orphans};"
            f" quoted by several snapshots: {unlinked.total - unlinked.orphans})"
        )
        for result in results:
            typer.echo(_pair_line(result))
        if not apply:
            typer.echo("Dry run: nothing written. Re-run with --apply.")


@app.command("delete-same-time-pairs")
def delete_same_time_pairs_command(
    apply: bool = typer.Option(False, "--apply", help="Write; without it the plan is rolled back"),
) -> None:
    """Delete snapshots of one event taken at one time by one source that cannot both stand.

    A pair is two such snapshots quoting a bookmaker, market, outcome and point in common.
    CONFLICTING (OddsPortal live, MLB): a shared key has two prices, i.e. two games on one
    event; both snapshots go. CONFLICTING_LEFT: any other conflict; left. DUPLICATE:
    identical prices; the lowest id stays. REFUSED: a candidate decision used a snapshot that
    would go; left. Deleted snapshots take their odds rows; the copies of a kept snapshot's
    rows collapse to one, linked to it. One transaction. Safe to re-run.

    Examples:
        odds events delete-same-time-pairs

        odds events delete-same-time-pairs --apply
    """
    asyncio.run(_delete_same_time_pairs(apply))
