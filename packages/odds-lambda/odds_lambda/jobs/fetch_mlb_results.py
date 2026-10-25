"""Settle MLB events from the MLB Stats API schedule by their ``gamePk`` (#612).

A pending event with a ``gamePk`` follows its game: FINAL with MLB's score once the game is
Final, CANCELLED or POSTPONED when its only slots are, and moved to the played slot when the
game was rescheduled. An event without a ``gamePk`` is never touched.

Idempotency: only events past kick-off that are SCHEDULED, LIVE, POSTPONED, or FINAL without
scores are pending, so re-runs after settling find nothing to process.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import structlog
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus
from odds_core.time import ensure_utc
from odds_core.vocabulary import Source

from odds_lambda.mlb_game_resolver import (
    MLB_SPORT_KEY,
    Score,
    final_scores,
    mlb_date,
    played_slots,
)
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbGameState, MlbStatsFetcher
from odds_lambda.results_run import (
    SCHEDULED_LOOKBACK,
    PendingEvent,
    ResultsStats,
    get_pending_events,
    run_daily_results_job,
)
from odds_lambda.scheduling.jobs import JobContext
from odds_lambda.storage.event_kickoff import move_kickoff
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

JOB_NAME = "fetch-mlb-results"


class Applied(StrEnum):
    """What a run did to one event; ``UNCHANGED`` leaves it pending and unresolved."""

    FINAL = "final"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"
    MOVED = "moved"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class GameOutcome:
    """What the schedule says about one event's game: the status to set (``None`` to leave it),
    MLB's (home, away) score when Final, and the start to move the event to."""

    status: EventStatus | None
    score: Score | None = None
    start: datetime | None = None


def game_outcome(
    commence_time: datetime, game_pk: int, slots: Sequence[MlbGameResult]
) -> GameOutcome:
    """The outcome of game ``game_pk`` for an event starting at ``commence_time``, from every
    schedule slot fetched for that game.

    A Final game settles FINAL; a game with a played slot not yet Final only moves the event
    there; a game with no played slot is CANCELLED or POSTPONED per its slots; a game absent
    from the slots leaves the event alone.
    """
    played = played_slots(slots).get(game_pk)
    start = played.start_for(commence_time) if played is not None else None
    final = final_scores(slots).get(game_pk)
    if final is not None:
        return GameOutcome(EventStatus.FINAL, final, start)
    if played is not None:
        return GameOutcome(None, start=start)
    states = {slot.state for slot in slots if slot.game_pk == game_pk}
    if MlbGameState.CANCELLED in states:
        return GameOutcome(EventStatus.CANCELLED)
    if MlbGameState.POSTPONED in states:
        return GameOutcome(EventStatus.POSTPONED)
    return GameOutcome(None)


async def _fetch_slots(
    fetcher: MlbStatsFetcher, events: Sequence[PendingEvent], today: datetime
) -> dict[int, list[MlbGameResult]]:
    """Every slot of the events' games: one schedule call from the earliest event's MLB date to
    today, then one call per game with no played slot in it, which finds a replay booked later."""
    start = min(mlb_date(ensure_utc(event.commence_time)) for event in events) - timedelta(days=1)
    schedule = await fetcher.fetch_results(start, mlb_date(today) + timedelta(days=1))
    by_pk: dict[int, list[MlbGameResult]] = defaultdict(list)
    for slot in schedule:
        by_pk[slot.game_pk].append(slot)
    played = played_slots(schedule)
    for game_pk in {event.mlb_game_pk for event in events if event.mlb_game_pk is not None}:
        if game_pk not in played:
            by_pk[game_pk] = await fetcher.fetch_game(game_pk)
    return by_pk


async def _apply(writer: OddsWriter, event: Event, game_pk: int, outcome: GameOutcome) -> Applied:
    """Move the event to its game's played slot, then set the outcome's status. Caller commits."""
    moves = await move_kickoff(writer.session, event, Source.MLB_STATS_API, outcome.start)
    if outcome.status is EventStatus.FINAL and outcome.score is not None:
        await writer.update_event_status(
            event.id, EventStatus.FINAL, home_score=outcome.score[0], away_score=outcome.score[1]
        )
        return Applied.FINAL
    if outcome.status is EventStatus.CANCELLED or outcome.status is EventStatus.POSTPONED:
        if event.status is not outcome.status:
            if not event.status.can_become(outcome.status):
                return Applied.UNCHANGED
            await writer.update_event_status(event.id, outcome.status)
        return Applied(outcome.status.value)
    return Applied.MOVED if moves else Applied.UNCHANGED


async def process_results(
    lookback: timedelta | None = SCHEDULED_LOOKBACK, fetcher: MlbStatsFetcher | None = None
) -> ResultsStats:
    """Settle pending MLB events that hold a ``gamePk`` from the MLB schedule.

    Reads pending events, closes the session, fetches the schedule, then settles each event in
    its own transaction so no connection is held across the fetch.

    Args:
        lookback: How far back pending events are considered; None for all of them.
        fetcher: MLB Stats API client to use instead of a new one. For testing.
    """
    stats = ResultsStats()
    now = datetime.now(UTC)
    since = now - lookback if lookback is not None else None
    async with async_session_maker() as session:
        pending = await get_pending_events(session, MLB_SPORT_KEY, since=since)
    if not pending:
        logger.info("no_pending_events", sport_key=MLB_SPORT_KEY)
        return stats

    keyed = [event for event in pending if event.mlb_game_pk is not None]
    unresolved = [event for event in pending if event.mlb_game_pk is None]
    slots: dict[int, list[MlbGameResult]] = {}
    if keyed:
        if fetcher is None:
            async with MlbStatsFetcher() as owned:
                slots = await _fetch_slots(owned, keyed, now)
        else:
            slots = await _fetch_slots(fetcher, keyed, now)
    stats.matches_fetched = len(slots)

    async with async_session_maker() as session:
        writer = OddsWriter(session)
        for pending_event in keyed:
            game_pk = pending_event.mlb_game_pk
            assert game_pk is not None
            outcome = game_outcome(pending_event.commence_time, game_pk, slots.get(game_pk, []))
            try:
                event = await session.get(Event, pending_event.id)
                if event is None:
                    continue
                result = await _apply(writer, event, game_pk, outcome)
                await session.commit()
            except Exception as e:
                msg = f"{pending_event.id} (game {game_pk}): {e}"
                stats.errors.append(msg)
                logger.error("mlb_result_failed", error=msg, exc_info=True)
                try:
                    await session.rollback()
                except Exception:
                    logger.warning("session_rollback_failed", exc_info=True)
                continue
            if result is Applied.FINAL:
                stats.events_updated += 1
            elif result is Applied.CANCELLED:
                stats.events_cancelled += 1
            elif result is Applied.POSTPONED:
                stats.events_postponed += 1
            elif result is Applied.MOVED:
                stats.events_moved += 1
            else:
                unresolved.append(pending_event)

    stats.events_unresolved = len(unresolved)
    stats.unresolved_kickoffs = [event.commence_time for event in unresolved]
    logger.info(
        "mlb_results_complete",
        games_fetched=stats.matches_fetched,
        events_updated=stats.events_updated,
        events_cancelled=stats.events_cancelled,
        events_postponed=stats.events_postponed,
        events_moved=stats.events_moved,
        events_unresolved=stats.events_unresolved,
        events_without_game_pk=sum(1 for event in pending if event.mlb_game_pk is None),
        errors=len(stats.errors),
    )
    return stats


async def main(ctx: JobContext) -> None:
    """Settle MLB events, then book tomorrow's run at 08:00 UTC whatever the outcome."""
    await run_daily_results_job(JOB_NAME, MLB_SPORT_KEY, process_results, schedule_sport=None)
