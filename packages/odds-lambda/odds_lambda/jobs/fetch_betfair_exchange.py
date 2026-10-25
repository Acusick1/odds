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
from typing import Any

import structlog
from odds_core.alerts import job_alert_context
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import FetchLog
from odds_core.snapshot_sources import SnapshotSource
from odds_core.sports import SportKey

from odds_lambda.betfair import (
    SPORT_CONFIG,
    BetfairBook,
    BetfairExchangeClient,
    SportBetfairConfig,
    betfair_book_to_bookmaker_entry,
    doubleheader_game_number,
    resolve_teams,
)
from odds_lambda.betfair.cert_loader import resolve_cert_paths
from odds_lambda.collection_calendar import (
    CollectionSource,
    FixtureKey,
    decide_collection_run,
    require_rows_for_fixtures,
)
from odds_lambda.live_events import (
    DroppedGame,
    LiveEventFinder,
    SkipReason,
    alert_dropped_games,
    count_dropped,
)
from odds_lambda.mlb_game_resolver import GameRow
from odds_lambda.scheduling.decision import OVERNIGHT_WINDOWS, CadenceConfig, ScheduleDecision
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Proximity-based polling cadence (hours): cheap delayed API — poll more
# aggressively than browser-driven fetch-oddsportal. 12h+ to kickoff all poll
# at 2h, matching the previous "far" band. ``close_lead``
# adds a run 5 minutes before every kickoff: the latest decision time and
# inside the live close window of ADR-0012.
CADENCE = CadenceConfig(
    closing=0.25,
    pregame=0.5,
    sharp=2.0,
    early=2.0,
    opening=2.0,
    no_game=6.0,
    db_fallback=1.0,
    close_lead=timedelta(minutes=5),
)

LIVE_REQUEST_ID_PREFIX = "betfair_api_"

# Betfair's start time for a game can pass before the game's own, and it differs from the MLB
# schedule's, so a started game stays listed long enough to be seen in play.
LOOKBACK_HOURS = 12

SOURCE = CollectionSource(
    key="betfair_exchange",
    job_name="fetch-betfair-exchange",
    snapshot_source=SnapshotSource.BETFAIR_API,
    cadence=CADENCE,
    fixture_horizons={"soccer_epl": timedelta(hours=72), "baseball_mlb": timedelta(hours=6)},
    skips_overnight=True,
)


@dataclass
class IngestionStats:
    """Per-sport ingestion stats for one fetch_betfair_exchange run."""

    sport: SportKey
    events_seen: int = 0
    books_fetched: int = 0
    books_in_play: int = 0
    accounted_fixtures: set[FixtureKey] = field(default_factory=set)
    books_suspended: int = 0
    snapshots_stored: int = 0
    events_matched: int = 0
    events_created: int = 0
    dropped_games: list[DroppedGame] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def games_unresolved(self) -> int:
        return count_dropped(self.dropped_games, SkipReason.UNRESOLVED)

    @property
    def games_skipped_schedule_unavailable(self) -> int:
        return count_dropped(self.dropped_games, SkipReason.SCHEDULE_UNAVAILABLE)


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


def _account_for(
    accounted: list[GameRow], book: BetfairBook, sport_cfg: SportBetfairConfig
) -> None:
    """Note the game of a book the run saw but did not price (in play, started, or priceless).

    A book with no start time or unparseable teams identifies no fixture, so it accounts for none.
    """
    teams = resolve_teams(book, sport_cfg)
    if teams is None or book.market_start_time is None:
        return
    accounted.append(
        GameRow(*teams, book.market_start_time, doubleheader_game_number(book.betfair_event_name))
    )


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
    events = await asyncio.to_thread(
        client.list_events, sport_cfg, lookahead_hours, lookback_hours=LOOKBACK_HOURS
    )
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

    priced: list[tuple[BetfairBook, GameRow, dict[str, Any]]] = []
    accounted: list[GameRow] = []
    for book in books:
        started = book.market_start_time is not None and book.market_start_time <= snapshot_time
        if book.inplay:
            stats.books_in_play += 1
        if _should_skip_book(book, sport_cfg) or started:
            if book.inplay or started:
                _account_for(accounted, book, sport_cfg)
            continue

        is_suspended = (book.market_status or "").upper() == "SUSPENDED"
        if is_suspended:
            stats.books_suspended += 1

        teams = resolve_teams(book, sport_cfg)
        if teams is None:
            stats.errors.append(f"unparseable_event_name: {book.betfair_event_name}")
            continue
        home, away = teams
        if book.market_start_time is None:
            stats.errors.append(f"no_start_time: {book.betfair_event_name}")
            continue

        entry = betfair_book_to_bookmaker_entry(
            book,
            sport_cfg,
            home_team=home,
            away_team=away,
            snapshot_time=snapshot_time,
        )
        if entry is None:
            stats.errors.append(f"unusable_outcomes: {book.betfair_event_name}")
            if is_suspended:
                _account_for(accounted, book, sport_cfg)
            continue
        game = GameRow(
            home, away, book.market_start_time, doubleheader_game_number(book.betfair_event_name)
        )
        priced.append((book, game, entry))

    finder = await LiveEventFinder.for_run(
        sport_cfg.sport_key,
        sport_cfg.sport_title,
        [*(game for _, game, _ in priced), *accounted],
    )
    if finder.schedule_unavailable:
        stats.errors.append("MLB schedule unavailable: no new events created")

    async with async_session_maker() as session:
        writer = OddsWriter(session)

        for book, game, entry in priced:
            try:
                found = await finder.find_or_create(writer, game)
                if isinstance(found, DroppedGame):
                    stats.dropped_games.append(found)
                    continue
                event_id, created = found
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
                snapshot.api_request_id = f"{LIVE_REQUEST_ID_PREFIX}{snapshot_time.isoformat()}"
                stats.snapshots_stored += 1

            except Exception as e:
                msg = f"{book.betfair_event_name}: {type(e).__name__}: {e}"
                stats.errors.append(msg)
                logger.error("betfair_book_ingestion_failed", error=msg, exc_info=True)

        for game in accounted:
            key = await finder.fixture_key(session, game)
            if key is not None:
                stats.accounted_fixtures.add(key)

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
        games_unresolved=stats.games_unresolved,
        games_skipped_schedule_unavailable=stats.games_skipped_schedule_unavailable,
        errors=len(stats.errors),
    )
    return stats


async def main(ctx: JobContext) -> None:
    """Job entry. Reads BetfairConfig and ingests the context's sport."""
    sport = require_sport(ctx, "fetch-betfair-exchange")
    settings = get_settings()
    bf = settings.betfair

    if not bf.enabled:
        logger.info("betfair_job_disabled", reason="betfair.enabled=False")
        return

    if sport not in SPORT_CONFIG:
        logger.error("betfair_unknown_sport", sport=sport, supported=list(SPORT_CONFIG))
        return

    compound_job_name = make_compound_job_name("fetch-betfair-exchange", sport)
    overnight = OVERNIGHT_WINDOWS.get(sport)

    async def _reschedule(label: str) -> ScheduleDecision:
        decision = await decide_collection_run([sport], CADENCE, overnight=overnight)
        assert decision.next_execution is not None  # noqa: S101
        try:
            await self_schedule(
                job_name=compound_job_name,
                next_time=decision.next_execution,
                dry_run=settings.scheduler.dry_run,
                sport=sport,
                reason=f"{label}: {decision.reason}",
            )
        except Exception as e:
            logger.error("betfair_schedule_failed", error=str(e), exc_info=True)
            if label == "pre-schedule":
                raise
        return decision

    # Pre-schedule before doing the work, so a crash mid-fetch doesn't break
    # the chain.
    decision = await _reschedule("pre-schedule")

    if not decision.should_execute:
        async with job_alert_context(compound_job_name):
            logger.info("betfair_job_dormant", sport=sport, reason=decision.reason)
        return

    snapshot_time = datetime.now(UTC)
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
            try:
                stats = await ingest_sport(
                    SPORT_CONFIG[sport],
                    client=client,
                    lookahead_hours=bf.lookahead_hours,
                    snapshot_time=snapshot_time,
                )
            except Exception as e:
                logger.error(
                    "betfair_sport_ingestion_failed",
                    sport=sport,
                    error=str(e),
                    exc_info=True,
                )
                stats = IngestionStats(sport=sport, errors=[str(e)])
        finally:
            await asyncio.to_thread(client.logout)
        await alert_dropped_games(compound_job_name, stats.dropped_games)
        if stats.accounted_fixtures and not stats.snapshots_stored:
            logger.warning(
                "betfair_empty_run_check_discounts_unpriced",
                sport=stats.sport,
                accounted_fixtures=sorted(map(str, stats.accounted_fixtures)),
            )
        await require_rows_for_fixtures(
            SOURCE,
            {stats.sport: stats.snapshots_stored},
            errors_by_sport={stats.sport: stats.errors},
            accounted_by_sport={stats.sport: stats.accounted_fixtures},
            now=snapshot_time,
        )

    # Reschedule based on (possibly updated) next kickoff after fetch.
    await _reschedule("post-fetch")
