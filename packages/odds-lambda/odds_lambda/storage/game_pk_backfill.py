"""Give stored MLB events their MLB Stats API ``gamePk`` and merge events that share one (#612)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum

import structlog
from odds_core.event_source import canonical_event_id
from odds_core.models import Event, EventStatus
from odds_core.team import normalize_team_name
from odds_core.time import ensure_utc
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.mlb_game_resolver import (
    MAX_SINGLE_GAME_GAP,
    MLB_SPORT_KEY,
    START_MATCH_WINDOW,
    GameRow,
    PlayedSlot,
    ResolutionRule,
    ResolvedGame,
    Score,
    UnresolvedGame,
    UnresolvedReason,
    final_scores,
    mlb_date,
    played_slots,
    resolve_game_pks,
)
from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher
from odds_lambda.storage.event_kickoff import move_kickoff
from odds_lambda.storage.event_merge import EventMerger
from odds_lambda.storage.same_time_pairs import SnapshotPairs, event_same_time_pairs
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

READ_ONLY_TABLES = frozenset({"paper_trades", "match_briefs", "candidate_decisions"})

_PairDate = tuple[tuple[str, str], date]


class BackfillSkip(StrEnum):
    HELD_GAME_MISMATCH = "held_game_mismatch"


class GamePkRefused(ValueError):
    """A hand assignment or mark that would leave the events inconsistent."""


class OutcomeStatus(StrEnum):
    APPLIED = "applied"
    REFUSED = "refused"
    FAILED = "failed"


class RefusalReason(StrEnum):
    PAPER_TRADE_CLASH = "paper_trade_clash"


class ScoreCarry(StrEnum):
    """The kept event's score after the merge: none, its own, a duplicate's, or its own while a
    duplicate holds another."""

    NONE = "none"
    KEPT = "kept"
    CARRIED = "carried"
    CONFLICT = "conflict"


class Settlement(StrEnum):
    """What the MLB Final score did to the kept event: settled it, matched its stored score,
    or differed from it (left alone)."""

    MLB = "mlb"
    MATCHES = "matches"
    CONFLICT = "conflict"


class MergeClash(Exception):
    """The merge would delete candidate decisions or paper trades."""

    def __init__(self, rows_deleted: dict[str, int]) -> None:
        super().__init__(f"merge would delete {rows_deleted}")
        self.rows_deleted = rows_deleted


@dataclass(frozen=True)
class GamePkAssignment:
    """Events of one game: ``kept`` takes ``game_pk`` and ``merged`` are merged into it.

    ``played`` is the slot the game is (or was) played in, ``None`` while it has only postponed
    slots; ``rule`` is ``None`` for a hand assignment. ``final_score`` is the MLB (home, away)
    score when the game is Final.
    """

    game_pk: int
    kept: Event
    merged: tuple[Event, ...]
    played: PlayedSlot | None
    rule: ResolutionRule | None
    final_score: Score | None = None

    def target_start(self) -> datetime | None:
        """Where ``kept`` moves (``PlayedSlot.start_for``); nowhere without a played slot."""
        if self.played is None:
            return None
        return self.played.start_for(self.kept.commence_time)


@dataclass(frozen=True)
class UnresolvedEvent:
    event_id: str
    reason: UnresolvedReason | BackfillSkip
    detail: str = ""


@dataclass(frozen=True)
class GamePkPlan:
    """``ghost_ids`` are unresolved ``no_game`` events nothing references; ``unresolved`` leaves
    them out. ``marked`` maps events marked as having no gamePk to the reason given."""

    pending: int
    assignments: tuple[GamePkAssignment, ...]
    unresolved: tuple[UnresolvedEvent, ...]
    ghost_ids: tuple[str, ...]
    marked: dict[str, str]


@dataclass(frozen=True)
class AssignmentOutcome:
    game_pk: int
    kept_id: str
    merged_ids: tuple[str, ...]
    rule: ResolutionRule | None
    status: OutcomeStatus = OutcomeStatus.APPLIED
    detail: str = ""
    rows_moved: dict[str, int] = field(default_factory=dict)
    rows_deleted: dict[str, int] = field(default_factory=dict)
    pairs: SnapshotPairs = SnapshotPairs()
    moved_from: datetime | None = None
    moved_to: datetime | None = None
    start_tbd: bool = False
    score: ScoreCarry = ScoreCarry.NONE
    score_detail: str = ""
    settled: Settlement | None = None
    settled_detail: str = ""

    @property
    def read_only_rows(self) -> dict[str, int]:
        """Paper-trade, brief and candidate-decision rows the merge moved or deleted."""
        touched: dict[str, int] = {}
        for rows in (self.rows_moved, self.rows_deleted):
            for table, count in rows.items():
                if table in READ_ONLY_TABLES:
                    touched[table] = touched.get(table, 0) + count
        return dict(sorted(touched.items()))


def _score(event: Event) -> Score | None:
    if event.home_score is None or event.away_score is None:
        return None
    return (event.home_score, event.away_score)


def _fmt(score: Score) -> str:
    return f"{score[0]}-{score[1]}"


def _pair_date(event: Event) -> _PairDate:
    return (
        (normalize_team_name(event.home_team), normalize_team_name(event.away_team)),
        mlb_date(ensure_utc(event.commence_time)),
    )


def _rows_by_event(events: Sequence[Event]) -> dict[str, GameRow]:
    """One resolver row per cluster of a team pair's events on an MLB date whose starts lie
    within ``START_MATCH_WINDOW`` of the cluster's first, at that first start. Twins listed a
    few minutes apart are then one row, so they cannot be paired with two games by start order.
    """
    by_key: dict[_PairDate, list[Event]] = defaultdict(list)
    for event in events:
        by_key[_pair_date(event)].append(event)
    rows: dict[str, GameRow] = {}
    for group in by_key.values():
        row: GameRow | None = None
        for event in sorted(group, key=lambda e: (ensure_utc(e.commence_time), e.id)):
            start = ensure_utc(event.commence_time)
            if row is None or start - row.start > START_MATCH_WINDOW:
                row = GameRow(event.home_team, event.away_team, start)
            rows[event.id] = row
    return rows


def _choose_kept(
    events: Sequence[Event], holder: Event | None, played: PlayedSlot | None
) -> tuple[Event, tuple[Event, ...]]:
    """The event holding the gamePk; else, among the events on the played slot's MLB date when
    there are any (so a postponed slot's event never wins over one on the played date), the one
    nearest the played slot's start (MLB's placeholder when TBD), ties going to the latest and
    then to ``canonical_event_id``; with no played slot, ``canonical_event_id``."""
    if holder is not None:
        return holder, tuple(sorted(events, key=lambda e: e.id))
    candidates = list(events)
    if played is not None:
        on_day = [e for e in events if mlb_date(ensure_utc(e.commence_time)) == played.day]
        candidates = on_day or candidates
        gaps = {e.id: abs(ensure_utc(e.commence_time) - played.start) for e in candidates}
        nearest = min(gaps.values())
        candidates = [e for e in candidates if gaps[e.id] == nearest]
        latest = max(ensure_utc(e.commence_time) for e in candidates)
        candidates = [e for e in candidates if ensure_utc(e.commence_time) == latest]
    kept_id = canonical_event_id([e.id for e in candidates])
    kept = next(e for e in events if e.id == kept_id)
    return kept, tuple(sorted((e for e in events if e.id != kept_id), key=lambda e: e.id))


async def plan_game_pks(
    session: AsyncSession, fetcher: MlbStatsFetcher, since: date | None, today: date
) -> GamePkPlan:
    """Resolve every MLB event without a ``gamePk`` and not marked as having none.

    ``since`` limits the events to MLB dates on or after it. The schedule is fetched once, from
    the earliest of their MLB dates to ``today`` (so a postponed game's played slot is in it).
    Events that already hold a ``gamePk`` on the same team pair and MLB date are resolved with
    them, so a doubleheader pairs by start order; when one of those resolves to a game other
    than its own, the date's events without a ``gamePk`` are left unresolved
    (``held_game_mismatch``). Events resolved to one ``gamePk`` form one assignment, with the
    event already holding it when there is one. Read-only.
    """
    query = select(Event).where(Event.sport_key == MLB_SPORT_KEY)
    if since is not None:
        query = query.where(
            Event.commence_time >= datetime.combine(since - timedelta(days=1), time(), UTC)
        )
    events = [
        e
        for e in (await session.execute(query.order_by(Event.commence_time, Event.id))).scalars()
        if since is None or mlb_date(ensure_utc(e.commence_time)) >= since
    ]
    marked = {
        e.id: e.no_game_pk_reason
        for e in events
        if e.mlb_game_pk is None and e.no_game_pk_reason is not None
    }
    pending = [e for e in events if e.mlb_game_pk is None and e.no_game_pk_reason is None]
    if not pending:
        return GamePkPlan(0, (), (), (), marked)

    pending_keys = {_pair_date(e) for e in pending}
    held = [e for e in events if e.mlb_game_pk is not None and _pair_date(e) in pending_keys]
    row_of = _rows_by_event([*pending, *held])
    rows = list(dict.fromkeys(row_of.values()))
    first_day = min(mlb_date(row.start) for row in rows)
    last_day = max(max(mlb_date(row.start) for row in rows), today)
    schedule = await fetcher.fetch_results(first_day, last_day)
    resolutions = dict(zip(rows, resolve_game_pks(rows, schedule), strict=True))
    scores = final_scores(schedule)
    played = played_slots(schedule)

    mismatched: dict[_PairDate, str] = {}
    for event in held:
        resolution = resolutions[row_of[event.id]]
        if not (isinstance(resolution, ResolvedGame) and resolution.game_pk == event.mlb_game_pk):
            mismatched[_pair_date(event)] = f"{event.id} holds {event.mlb_game_pk}"

    unresolved: list[UnresolvedEvent] = []
    by_pk: dict[int, list[Event]] = defaultdict(list)
    resolved_by_pk: dict[int, ResolvedGame] = {}
    for event in pending:
        resolution = resolutions[row_of[event.id]]
        key = _pair_date(event)
        if key in mismatched:
            unresolved.append(
                UnresolvedEvent(event.id, BackfillSkip.HELD_GAME_MISMATCH, mismatched[key])
            )
        elif isinstance(resolution, UnresolvedGame):
            unresolved.append(UnresolvedEvent(event.id, resolution.reason, resolution.detail))
        else:
            by_pk[resolution.game_pk].append(event)
            resolved_by_pk[resolution.game_pk] = resolution

    holders = {
        e.mlb_game_pk: e
        for e in (
            await session.execute(select(Event).where(Event.mlb_game_pk.in_(list(by_pk))))
        ).scalars()
    }
    assignments: list[GamePkAssignment] = []
    for game_pk in sorted(by_pk):
        resolution = resolved_by_pk[game_pk]
        kept, merged = _choose_kept(by_pk[game_pk], holders.get(game_pk), played.get(game_pk))
        assignments.append(
            GamePkAssignment(
                game_pk, kept, merged, played.get(game_pk), resolution.rule, scores.get(game_pk)
            )
        )

    merger = EventMerger(session)
    ghosts: list[str] = []
    still_unresolved: list[UnresolvedEvent] = []
    for item in unresolved:
        if item.reason is UnresolvedReason.NO_GAME and not await merger.reference_counts(
            item.event_id
        ):
            ghosts.append(item.event_id)
        else:
            still_unresolved.append(item)
    return GamePkPlan(
        len(pending),
        tuple(assignments),
        tuple(sorted(still_unresolved, key=lambda u: u.event_id)),
        tuple(sorted(ghosts)),
        marked,
    )


async def _settle(
    session: AsyncSession, kept: Event, final: Score | None
) -> tuple[Settlement | None, str]:
    """Settle the kept event from the MLB Final score when it has no score of its own."""
    if final is None:
        return None, ""
    stored = _score(kept)
    if stored is None:
        await OddsWriter(session).update_event_status(kept.id, EventStatus.FINAL, *final)
        return Settlement.MLB, ""
    if stored == final:
        return Settlement.MATCHES, ""
    return Settlement.CONFLICT, f"stored_{_fmt(stored)}_vs_mlb_{_fmt(final)}"


async def apply_assignment(
    session: AsyncSession, assignment: GamePkAssignment
) -> AssignmentOutcome:
    """Merge the assignment's events into ``kept``, give it the ``gamePk``, move it to the
    scheduled start re-timing its snapshots, and settle it from MLB's Final score when it has
    none. Raises ``MergeClash`` when the merge would delete candidate decisions or paper
    trades. Runs in the caller's transaction."""
    kept = assignment.kept
    kept_id = kept.id
    moved_from = ensure_utc(kept.commence_time)
    kept_score = _score(kept)
    duplicate_scores = [s for s in (_score(e) for e in assignment.merged) if s is not None]
    rows_moved: dict[str, int] = {}
    if assignment.merged:
        report = await EventMerger(session).merge(kept, list(assignment.merged))
        if report.rows_deleted:
            raise MergeClash(report.rows_deleted)
        rows_moved = report.rows_moved
    if kept.mlb_game_pk is None:
        kept.assign_game_pk(assignment.game_pk)
    await move_kickoff(session, kept, Source.MLB_STATS_API, assignment.target_start())

    merged_score = _score(kept)
    others = sorted({s for s in duplicate_scores if s != merged_score})
    if merged_score is not None and others:
        score, score_detail = (
            ScoreCarry.CONFLICT,
            "_vs_".join(_fmt(s) for s in [merged_score, *others]),
        )
    elif merged_score is not None and kept_score is None:
        score, score_detail = ScoreCarry.CARRIED, ""
    elif merged_score is not None:
        score, score_detail = ScoreCarry.KEPT, ""
    else:
        score, score_detail = ScoreCarry.NONE, ""
    settled, settled_detail = await _settle(session, kept, assignment.final_score)

    await session.flush()
    moved_to = ensure_utc(kept.commence_time)
    return AssignmentOutcome(
        assignment.game_pk,
        kept_id,
        tuple(e.id for e in assignment.merged),
        assignment.rule,
        rows_moved=rows_moved,
        pairs=await event_same_time_pairs(session, kept_id),
        moved_from=moved_from if moved_to != moved_from else None,
        moved_to=moved_to if moved_to != moved_from else None,
        start_tbd=assignment.played is not None and assignment.played.tbd,
        score=score,
        score_detail=score_detail,
        settled=settled,
        settled_detail=settled_detail,
    )


async def run_assignments(
    session: AsyncSession, assignments: Sequence[GamePkAssignment], commit: bool
) -> list[AssignmentOutcome]:
    """Apply each assignment in its own savepoint, committing after each when ``commit``.

    An assignment whose merge would delete candidate decisions or paper trades is rolled back
    and reported ``REFUSED``; one that raises anything else (a ``gamePk`` taken meanwhile, say)
    is rolled back, logged and reported ``FAILED``. The others go ahead either way.
    """
    outcomes: list[AssignmentOutcome] = []
    for assignment in assignments:
        kept_id = assignment.kept.id
        merged_ids = tuple(e.id for e in assignment.merged)
        try:
            async with session.begin_nested():
                outcome = await apply_assignment(session, assignment)
        except MergeClash as e:
            logger.warning(
                "game_pk_assignment_refused",
                game_pk=assignment.game_pk,
                kept_id=kept_id,
                merged_ids=merged_ids,
                would_delete=e.rows_deleted,
            )
            outcome = AssignmentOutcome(
                assignment.game_pk,
                kept_id,
                merged_ids,
                assignment.rule,
                status=OutcomeStatus.REFUSED,
                detail=RefusalReason.PAPER_TRADE_CLASH,
                rows_deleted=e.rows_deleted,
            )
        except Exception as e:
            logger.exception(
                "game_pk_assignment_failed",
                game_pk=assignment.game_pk,
                kept_id=kept_id,
                merged_ids=merged_ids,
            )
            outcome = AssignmentOutcome(
                assignment.game_pk,
                kept_id,
                merged_ids,
                assignment.rule,
                status=OutcomeStatus.FAILED,
                detail=(str(e).splitlines() or [type(e).__name__])[0],
            )
        if commit:
            await session.commit()
        outcomes.append(outcome)
    return outcomes


async def delete_ghosts(session: AsyncSession, event_ids: Sequence[str]) -> int:
    """Delete the given events, refusing any that something now references. Caller commits."""
    merger = EventMerger(session)
    deleted = 0
    for event_id in event_ids:
        event = await session.get(Event, event_id)
        if event is None or await merger.reference_counts(event_id):
            continue
        await session.delete(event)
        deleted += 1
    await session.flush()
    return deleted


async def _mlb_event_without_game_pk(session: AsyncSession, event_id: str) -> Event:
    event = await session.get(Event, event_id)
    if event is None:
        raise GamePkRefused(f"no event {event_id}")
    if event.sport_key != MLB_SPORT_KEY:
        raise GamePkRefused(f"{event_id} is a {event.sport_key} event")
    if event.mlb_game_pk is not None:
        raise GamePkRefused(f"{event_id} already holds gamePk {event.mlb_game_pk}")
    return event


async def plan_hand_assignment(
    session: AsyncSession, fetcher: MlbStatsFetcher, event_id: str, game_pk: int, merge: bool
) -> GamePkAssignment:
    """The assignment of ``game_pk`` to an event without one, after checking that the game has
    the event's teams and a slot on its MLB date or within ``MAX_SINGLE_GAME_GAP`` of its start.

    When another event holds ``game_pk`` it is refused unless ``merge``, which merges the event
    into the holder. Read-only.
    """
    event = await _mlb_event_without_game_pk(session, event_id)
    slots = await fetcher.fetch_game(game_pk)
    if not slots:
        raise GamePkRefused(f"the MLB schedule has no game {game_pk}")
    game_teams = (normalize_team_name(slots[0].home_team), normalize_team_name(slots[0].away_team))
    if game_teams != (event.home_team, event.away_team):
        raise GamePkRefused(
            f"game {game_pk} is {game_teams[1]} @ {game_teams[0]}, "
            f"{event_id} is {event.away_team} @ {event.home_team}"
        )
    event_day = mlb_date(ensure_utc(event.commence_time))
    if not any(
        (slot.schedule_date or slot.day) == event_day
        or abs(slot.start_time - ensure_utc(event.commence_time)) <= MAX_SINGLE_GAME_GAP
        for slot in slots
    ):
        raise GamePkRefused(
            f"game {game_pk} has no slot on {event_day} or within {MAX_SINGLE_GAME_GAP} of "
            f"{event_id}'s start"
        )
    played = played_slots(slots).get(game_pk)
    final = final_scores(slots).get(game_pk)
    holder = (
        await session.execute(select(Event).where(Event.mlb_game_pk == game_pk))
    ).scalar_one_or_none()
    if holder is None:
        return GamePkAssignment(game_pk, event, (), played, None, final)
    if not merge:
        raise GamePkRefused(f"{holder.id} holds gamePk {game_pk}; pass --merge to merge into it")
    return GamePkAssignment(game_pk, holder, (event,), played, None, final)


async def mark_no_game_pk(session: AsyncSession, event_id: str, reason: str) -> Event:
    """Record why an MLB event has no ``gamePk``, so the health check stops listing it. Caller
    commits."""
    if not reason.strip():
        raise GamePkRefused("a reason is required")
    event = await _mlb_event_without_game_pk(session, event_id)
    event.no_game_pk_reason = reason.strip()
    await session.flush()
    return event
