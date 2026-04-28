"""Fetch Match Odds from Betfair Exchange directly.

Polls the Betfair Exchange API for upcoming events in a sport, fetches the
best back/lay for the configured Match Odds market, converts to canonical
``raw_data`` format, and writes a ``betfair_exchange`` snapshot row via
``OddsWriter.store_odds_snapshot``. Self-schedules with proximity-based
intervals matching ``fetch_oddsportal``.

Replaces the previous BFE-via-OddsPortal data path that broke 2026-04-21
when OP moved exchanges into a separate (encrypted) AJAX endpoint.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.alerts import job_alert_context, send_job_warning
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import FetchLog
from odds_core.sports import SportKey

from odds_lambda.betfair import (
    SPORT_CONFIG,
    BetfairBook,
    BetfairExchangeClient,
    SportBetfairConfig,
    betfair_book_to_bookmaker_entry,
    resolve_teams,
)
from odds_lambda.betfair.cert_loader import resolve_cert_paths
from odds_lambda.scheduling.helpers import (
    ProximitySchedule,
    get_next_kickoff,
    self_schedule,
)
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Proximity-based scheduling intervals (hours): cheap delayed API — poll
# more aggressively than browser-driven fetch-oddsportal.
SCHEDULE = ProximitySchedule(closing=0.25, pregame=0.5, far=2.0, db_fallback=1.0)


@dataclass
class IngestionStats:
    """Per-sport ingestion stats for one fetch_betfair_exchange run."""

    sport: SportKey
    events_seen: int = 0
    books_fetched: int = 0
    books_in_play: int = 0
    books_suspended: int = 0
    snapshots_stored: int = 0
    events_matched: int = 0
    events_created: int = 0
    errors: list[str] = field(default_factory=list)


def _should_skip_book(book: BetfairBook, sport_cfg: SportBetfairConfig) -> bool:
    """Skip in-play and clearly-closed markets; we ingest pre-match only."""
    if book.inplay:
        return True
    status = (book.market_status or "").upper()
    if status in {"CLOSED", "INACTIVE"}:
        return True
    if len(book.runners) < (3 if sport_cfg.has_draw else 2):
        return True
    return False


async def ingest_sport(
    sport_cfg: SportBetfairConfig,
    *,
    client: BetfairExchangeClient,
    lookahead_hours: int,
    snapshot_time: datetime,
) -> IngestionStats:
    """Discover events, fetch books, write snapshots for one sport."""
    stats = IngestionStats(sport=sport_cfg.sport_key)

    # API calls are blocking; run in worker thread to avoid stalling the loop.
    events = await asyncio.to_thread(client.list_events, sport_cfg, lookahead_hours)
    stats.events_seen = len(events)
    if not events:
        logger.warning("betfair_no_events_found", sport=sport_cfg.sport_key)
        return stats

    books = await asyncio.to_thread(client.list_match_odds_books, sport_cfg, events)
    stats.books_fetched = len(books)
    if not books:
        logger.warning(
            "betfair_no_books_returned", sport=sport_cfg.sport_key, events_seen=len(events)
        )
        return stats

    async with async_session_maker() as session:
        writer = OddsWriter(session)

        for book in books:
            if _should_skip_book(book, sport_cfg):
                if book.inplay:
                    stats.books_in_play += 1
                continue

            if (book.market_status or "").upper() == "SUSPENDED":
                stats.books_suspended += 1

            teams = resolve_teams(book, sport_cfg)
            if teams is None:
                stats.errors.append(f"unparseable_event_name: {book.betfair_event_name}")
                continue
            home, away = teams

            entry = betfair_book_to_bookmaker_entry(
                book,
                sport_cfg,
                home_team=home,
                away_team=away,
                snapshot_time=snapshot_time,
            )
            if entry is None:
                stats.errors.append(f"unusable_outcomes: {book.betfair_event_name}")
                continue

            try:
                event_id, created = await writer.find_or_create_event(
                    home_team=home,
                    away_team=away,
                    match_date=book.market_start_time,
                    sport_key=sport_cfg.sport_key,
                    sport_title=sport_cfg.sport_title,
                )
                if created:
                    stats.events_created += 1
                else:
                    stats.events_matched += 1

                raw_data = {
                    "bookmakers": [entry],
                    "source": "betfair_api",
                    "betfair_event_id": book.betfair_event_id,
                }
                snapshot, _ = await writer.store_odds_snapshot(
                    event_id=event_id,
                    raw_data=raw_data,
                    snapshot_time=snapshot_time,
                )
                snapshot.api_request_id = f"betfair_api_{snapshot_time.isoformat()}"
                stats.snapshots_stored += 1

            except Exception as e:
                msg = f"{book.betfair_event_name}: {type(e).__name__}: {e}"
                stats.errors.append(msg)
                logger.error("betfair_book_ingestion_failed", error=msg, exc_info=True)

        # FetchLog for observability — sport-level summary.
        if stats.snapshots_stored or stats.events_seen:
            session.add(
                FetchLog(
                    fetch_time=snapshot_time,
                    sport_key=sport_cfg.sport_key,
                    events_count=stats.snapshots_stored,
                    bookmakers_count=1,
                    success=stats.snapshots_stored > 0,
                    error_message=None if not stats.errors else "; ".join(stats.errors[:3]),
                )
            )

        await session.commit()

    logger.info(
        "betfair_sport_ingestion_complete",
        sport=sport_cfg.sport_key,
        events_seen=stats.events_seen,
        books_fetched=stats.books_fetched,
        books_in_play=stats.books_in_play,
        books_suspended=stats.books_suspended,
        snapshots_stored=stats.snapshots_stored,
        events_matched=stats.events_matched,
        events_created=stats.events_created,
        errors=len(stats.errors),
    )
    return stats


async def main(ctx: JobContext) -> None:
    """Job entry. Reads BetfairConfig, ingests one or all configured sports."""
    settings = get_settings()
    bf = settings.betfair

    if not bf.enabled:
        logger.info("betfair_job_disabled", reason="betfair.enabled=False")
        return

    sport: SportKey | None = ctx.sport
    if sport:
        if sport not in SPORT_CONFIG:
            logger.error("betfair_unknown_sport", sport=sport, supported=list(SPORT_CONFIG))
            return
        target_sports: list[SportKey] = [sport]
    else:
        # Default to every adapter-supported sport; honour an explicit override.
        configured = bf.sports if bf.sports is not None else list(SPORT_CONFIG)
        target_sports = [s for s in configured if s in SPORT_CONFIG]

    if not target_sports:
        logger.warning("betfair_no_target_sports")
        return

    compound_job_name = make_compound_job_name("fetch-betfair-exchange", sport)

    # Pre-schedule before doing the work, so a crash mid-fetch doesn't break
    # the chain. Use the soonest kickoff across configured sports.
    pre_interval = SCHEDULE.db_fallback
    try:
        soonest: datetime | None = None
        for sk in target_sports:
            ko = await get_next_kickoff(sk)
            if ko is not None and (soonest is None or ko < soonest):
                soonest = ko
        pre_interval = SCHEDULE.interval_for(soonest)
    except Exception as e:
        logger.warning("betfair_next_kickoff_query_failed", error=str(e), exc_info=True)

    pre_next_time = datetime.now(UTC) + timedelta(hours=pre_interval)
    try:
        await self_schedule(
            job_name=compound_job_name,
            next_time=pre_next_time,
            dry_run=settings.scheduler.dry_run,
            sport=sport,
            interval_hours=pre_interval,
        )
    except Exception as e:
        logger.error("betfair_pre_schedule_failed", error=str(e), exc_info=True)
        raise

    snapshot_time = datetime.now(UTC)
    all_stats: list[IngestionStats] = []
    # BetfairConfig validator guarantees these are non-None when enabled=True.
    assert bf.username and bf.password and bf.app_key
    cert_file, cert_key = resolve_cert_paths(bf)
    async with job_alert_context(compound_job_name):
        client = BetfairExchangeClient(
            username=bf.username,
            password=bf.password,
            app_key=bf.app_key,
            cert_file=cert_file,
            cert_key=cert_key,
        )
        await asyncio.to_thread(client.login)
        try:
            for sport_key in target_sports:
                cfg = SPORT_CONFIG[sport_key]
                try:
                    stats = await ingest_sport(
                        cfg,
                        client=client,
                        lookahead_hours=bf.lookahead_hours,
                        snapshot_time=snapshot_time,
                    )
                except Exception as e:
                    logger.error(
                        "betfair_sport_ingestion_failed",
                        sport=sport_key,
                        error=str(e),
                        exc_info=True,
                    )
                    stats = IngestionStats(sport=sport_key, errors=[str(e)])
                all_stats.append(stats)
        finally:
            await asyncio.to_thread(client.logout)

    # Soft warning when zero snapshots stored across all sports — likely
    # cert/auth/credentials/geo issue worth investigating.
    total_stored = sum(s.snapshots_stored for s in all_stats)
    if total_stored == 0:
        await send_job_warning(
            f"job_empty:{compound_job_name}",
            f"⚠️ {compound_job_name} stored 0 snapshots across {len(all_stats)} sport(s).",
        )

    # Reschedule based on (possibly updated) next kickoff after fetch.
    try:
        soonest = None
        for sk in target_sports:
            ko = await get_next_kickoff(sk)
            if ko is not None and (soonest is None or ko < soonest):
                soonest = ko
        post_interval = SCHEDULE.interval_for(soonest)
    except Exception as e:
        logger.warning("betfair_post_kickoff_query_failed", error=str(e), exc_info=True)
        post_interval = pre_interval

    post_next_time = datetime.now(UTC) + timedelta(hours=post_interval)
    try:
        await self_schedule(
            job_name=compound_job_name,
            next_time=post_next_time,
            dry_run=settings.scheduler.dry_run,
            sport=sport,
            interval_hours=post_interval,
            reason="post_fetch",
        )
    except Exception as e:
        logger.error("betfair_post_schedule_failed", error=str(e), exc_info=True)
