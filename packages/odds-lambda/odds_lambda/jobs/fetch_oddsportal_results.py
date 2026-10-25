"""Settle EPL events from OddsPortal's results pages.

Reads finished matches over HTTP from the results page's archive feed (no
browser; see docs/adr/0009-oddsportal-feed-client.md) and sets matching
unsettled events FINAL with their scores, or CANCELLED for cancelled matches.
The results page path comes from the shared ``LeagueSpec`` registry in
``fetch_oddsportal``. MLB is settled from the MLB Stats API instead
(``fetch_mlb_results``).

Idempotency: only events past kick-off that are SCHEDULED, LIVE, POSTPONED, or FINAL
without scores are pending, so re-runs after settling find nothing to process.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from odds_core.team import normalize_team_name

from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT
from odds_lambda.mlb_game_resolver import MLB_SPORT_KEY
from odds_lambda.oddsportal_client import (
    CancelledMatch,
    FetchedResults,
    FinishedMatch,
    OddsPortalClient,
    OddsPortalError,
)
from odds_lambda.results_run import (
    SCHEDULED_LOOKBACK,
    STALE_AFTER,
    PendingEvent,
    ResultsStats,
    get_pending_events,
    run_daily_results_job,
)
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport
from odds_lambda.storage.event_merge import EVENT_MATCH_WINDOW, find_match_id
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

DEFAULT_SPORT_KEY = "soccer_epl"


def _raise_if_feed_parsed_nothing(fetched: FetchedResults, pending: list[PendingEvent]) -> None:
    """A feed with rows but no finished or cancelled match, while old events wait, means the parser broke."""
    stale_before = datetime.now(UTC) - STALE_AFTER
    stale = sum(1 for event in pending if event.commence_time < stale_before)
    if stale and fetched.rows_seen and not fetched.matches and not fetched.cancelled:
        raise OddsPortalError(
            f"results feed returned {fetched.rows_seen} rows but no finished or cancelled match, "
            f"with {stale} pending events older than {STALE_AFTER}; page format may have changed"
        )


async def process_results(
    sport: str | None = None,
    lookback: timedelta | None = SCHEDULED_LOOKBACK,
    fetched: FetchedResults | None = None,
) -> ResultsStats:
    """Set pending events FINAL with the scores of their finished matches.

    The job runs in three phases with distinct session scopes so no DB session
    is held across the network fetch (Neon closes idle connections, which would
    crash the write phase):

    1. Read phase: open a session, load pending events (detached), close session.
    2. Fetch phase: read the results pages back to the oldest pending kick-off.
    3. Write phase: open a fresh session, update each matched event, commit per event.

    Args:
        sport: Sport key (e.g. "soccer_epl"). Defaults to DEFAULT_SPORT_KEY.
        lookback: How far back pending events are considered; None for all of them.
        fetched: Results replacing the fetch. For testing.
    """
    stats = ResultsStats()
    sport_key = sport or DEFAULT_SPORT_KEY
    if sport_key == MLB_SPORT_KEY:
        raise ValueError(f"{MLB_SPORT_KEY} is settled from the MLB Stats API (fetch-mlb-results)")

    spec = _LEAGUE_SPEC_BY_SPORT.get(sport_key)
    if spec is None:
        logger.error(
            "unknown_sport_key",
            sport_key=sport_key,
            available=list(_LEAGUE_SPEC_BY_SPORT),
        )
        return stats

    since = datetime.now(UTC) - lookback if lookback is not None else None
    async with async_session_maker() as session:
        pending_events = await get_pending_events(session, sport_key=sport_key, since=since)

    if not pending_events:
        logger.info("no_pending_events", sport_key=sport_key)
        return stats

    logger.info("pending_events_found", count=len(pending_events))

    if fetched is None:
        oldest = min(event.commence_time for event in pending_events)
        async with OddsPortalClient() as client:
            fetched = await client.fetch_results(
                f"{spec.listing_path}results/", since=oldest - EVENT_MATCH_WINDOW
            )
    _raise_if_feed_parsed_nothing(fetched, pending_events)
    if not fetched.complete:
        logger.warning("results_fetch_incomplete", matches=len(fetched.matches))
    stats.matches_fetched = len(fetched.matches)

    matches = [
        FinishedMatch(
            home_team=normalize_team_name(m.home_team),
            away_team=normalize_team_name(m.away_team),
            start_time=m.start_time,
            home_score=m.home_score,
            away_score=m.away_score,
        )
        for m in fetched.matches
    ]
    cancelled = [
        CancelledMatch(
            home_team=normalize_team_name(m.home_team),
            away_team=normalize_team_name(m.away_team),
            start_time=m.start_time,
        )
        for m in fetched.cancelled
    ]
    pending_by_id = {event.id: event for event in pending_events}

    async with async_session_maker() as session:
        writer = OddsWriter(session)

        for match in sorted((*matches, *cancelled), key=lambda m: m.start_time):
            event_id = await find_match_id(
                session, match.home_team, match.away_team, match.start_time, sport_key
            )
            event = pending_by_id.get(event_id) if event_id is not None else None
            if event is None:
                continue
            if isinstance(match, CancelledMatch) and not event.status.can_become(
                EventStatus.CANCELLED
            ):
                continue
            try:
                if isinstance(match, FinishedMatch):
                    await writer.update_event_status(
                        event_id=event.id,
                        status=EventStatus.FINAL,
                        home_score=match.home_score,
                        away_score=match.away_score,
                    )
                else:
                    await writer.update_event_status(
                        event_id=event.id, status=EventStatus.CANCELLED
                    )
                await session.commit()
            except Exception as e:
                msg = f"{match.home_team} vs {match.away_team}: {e}"
                stats.errors.append(msg)
                logger.error("result_processing_failed", error=msg, exc_info=True)
                try:
                    await session.rollback()
                except Exception:
                    logger.warning("session_rollback_failed", exc_info=True)
                continue

            if isinstance(match, FinishedMatch):
                stats.events_updated += 1
            else:
                stats.events_cancelled += 1
            del pending_by_id[event.id]

    stats.events_unresolved = len(pending_by_id)
    stats.unresolved_kickoffs = [event.commence_time for event in pending_by_id.values()]
    logger.info(
        "results_collection_complete",
        matches_fetched=stats.matches_fetched,
        events_updated=stats.events_updated,
        events_cancelled=stats.events_cancelled,
        events_unresolved=stats.events_unresolved,
        errors=len(stats.errors),
    )

    return stats


async def main(ctx: JobContext) -> None:
    """Settle the sport's events, then book tomorrow's run at 08:00 UTC whatever the outcome."""
    sport = require_sport(ctx, "fetch-oddsportal-results")
    await run_daily_results_job(
        make_compound_job_name("fetch-oddsportal-results", sport),
        sport,
        lambda: process_results(sport=sport),
        schedule_sport=sport,
    )
