"""Fetch the configured markets from Betfair Exchange directly.

Polls the Betfair Exchange API for upcoming events in a sport, fetches the
best back/lay of each market the sport collects (Match Odds, and Over/Under
2.5 Goals for EPL), converts them to one canonical envelope per
event, and writes it via ``OddsWriter.store_envelope``, one snapshot per
market. Self-schedules with proximity-based intervals matching ``fetch_oddsportal``.

Replaces the previous BFE-via-OddsPortal data path that broke 2026-04-21
when OP moved exchanges into a separate (encrypted) AJAX endpoint.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.job_specs import job_cadence
from odds_core.models import FetchLog
from odds_core.sports import SportKey
from odds_core.vocabulary import Source, TimeBasis

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
from odds_lambda.betfair.constants import BetfairMarketSpec
from odds_lambda.betfair.kickoff import next_betfair_kickoff
from odds_lambda.collection_calendar import (
    CollectionSource,
    FixtureKey,
    SportCollection,
    decide_collection_run,
    require_rows_for_fixtures,
)
from odds_lambda.live_events import (
    DroppedGame,
    LiveEventFinder,
    LiveListing,
    SkipReason,
    alert_dropped_games,
    count_dropped,
)
from odds_lambda.mlb_game_resolver import GameRow
from odds_lambda.scheduling.collection import CollectionResult, run_collection
from odds_lambda.scheduling.decision import (
    OVERNIGHT_WINDOWS,
    CadenceConfig,
    Lead,
    ScheduleDecision,
)
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_payload

logger = structlog.get_logger()

# Proximity-based polling cadence (hours): cheap delayed API — poll more
# aggressively than browser-driven fetch-oddsportal. 12h+ to kickoff all poll
# at 2h, matching the previous "far" band.
CADENCE = CadenceConfig(**asdict(job_cadence("fetch-betfair-exchange")))
# A run this long before every kickoff: the latest decision time and inside the
# live close window of ADR-0012. Kickoffs come from ``next_betfair_kickoff``, so
# the run also lands before an earlier Betfair start.
CLOSE_LEAD = Lead(timedelta(minutes=5))

# Betfair's start time for a game can pass before the game's own, and it differs from the MLB
# schedule's, so a game stays listed, and priced while pre-play, until it is seen in play.
LOOKBACK_HOURS = 12


def betfair_collection(config: SportBetfairConfig) -> SportCollection:
    return SportCollection(config.fixture_horizon, tuple(spec.market for spec in config.markets))


def collection_source() -> CollectionSource:
    return CollectionSource(
        key="betfair_exchange",
        job_name="fetch-betfair-exchange",
        source=Source.BETFAIR,
        cadence=CADENCE,
        sports={
            sport_key: betfair_collection(config) for sport_key, config in SPORT_CONFIG.items()
        },
        skips_overnight=True,
    )


SOURCE = collection_source()


@dataclass
class IngestionStats:
    """Per-sport ingestion stats for one fetch_betfair_exchange run."""

    sport: SportKey
    events_seen: int = 0
    books_fetched: int = 0
    books_in_play: int = 0
    accounted_fixtures: set[FixtureKey] = field(default_factory=set)
    books_suspended: int = 0
    books_pre_play_past_start: int = 0
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


def _should_skip_book(book: BetfairBook, spec: BetfairMarketSpec) -> bool:
    """Skip in-play and clearly-closed markets; we ingest pre-play only, whatever the Betfair start."""
    if book.inplay:
        return True
    status = (book.market_status or "").upper()
    if status in {"CLOSED", "INACTIVE"}:
        return True
    if len(book.runners) < spec.runner_count:
        return True
    return False


def _other_market_entries(
    books: list[BetfairBook],
    sport_cfg: SportBetfairConfig,
    home: str,
    away: str,
    snapshot_time: datetime,
) -> list[dict[str, Any]]:
    """Bookmaker entries of an event's priced books outside its match odds."""
    entries: list[dict[str, Any]] = []
    for book in books:
        spec = sport_cfg.market_spec(book.market_type)
        if spec is None or _should_skip_book(book, spec):
            continue
        entry = betfair_book_to_bookmaker_entry(book, spec, home, away, snapshot_time)
        if entry is not None:
            entries.append(entry)
    return entries


def _account_for(
    accounted: list[GameRow], book: BetfairBook, sport_cfg: SportBetfairConfig
) -> None:
    """Note the game of a book the run saw but did not price (in play, skipped after its Betfair
    start, or suspended with no prices).

    A book with no start time or unparseable teams identifies no fixture, so it accounts for none.
    """
    teams = resolve_teams(book, sport_cfg)
    if teams is None or book.market_start_time is None:
        return
    accounted.append(
        GameRow(*teams, book.market_start_time, doubleheader_game_number(book.betfair_event_name))
    )


def _envelope(book: BetfairBook, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"bookmakers": entries, "betfair_event_id": book.betfair_event_id}


async def ingest_sport(
    sport_cfg: SportBetfairConfig,
    *,
    client: BetfairExchangeClient,
    lookahead_hours: int,
    snapshot_time: datetime,
) -> IngestionStats:
    """Discover events, fetch books, write snapshots for one sport.

    An event's match odds book decides whether it is stored and names its teams; its other
    markets' books ride in the same envelope when priced, one bookmaker entry per market.
    """
    stats = IngestionStats(sport=sport_cfg.sport_key)

    # API calls are blocking; run in worker thread to avoid stalling the loop.
    events = await asyncio.to_thread(
        client.list_events, sport_cfg, lookahead_hours, lookback_hours=LOOKBACK_HOURS
    )
    stats.events_seen = len(events)
    if not events:
        logger.warning("betfair_no_events_found", sport=sport_cfg.sport_key)
        return stats

    books = await asyncio.to_thread(client.list_books, sport_cfg, events)
    stats.books_fetched = len(books)
    if not books:
        logger.warning(
            "betfair_no_books_returned", sport=sport_cfg.sport_key, events_seen=len(events)
        )
        return stats

    match_type = sport_cfg.match_odds.market_type_code
    other_books: dict[str, list[BetfairBook]] = {}
    for book in books:
        if book.market_type != match_type:
            other_books.setdefault(book.betfair_event_id, []).append(book)

    priced: list[tuple[BetfairBook, GameRow, list[dict[str, Any]]]] = []
    accounted: list[GameRow] = []
    for book in books:
        if book.market_type != match_type:
            continue
        started = book.market_start_time is not None and book.market_start_time <= snapshot_time
        if book.inplay:
            stats.books_in_play += 1
        if _should_skip_book(book, sport_cfg.match_odds):
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
            sport_cfg.match_odds,
            home_team=home,
            away_team=away,
            snapshot_time=snapshot_time,
        )
        if entry is None:
            stats.errors.append(f"unusable_outcomes: {book.betfair_event_name}")
            _account_for(accounted, book, sport_cfg)
            continue
        if started:
            stats.books_pre_play_past_start += 1
        game = GameRow(
            home, away, book.market_start_time, doubleheader_game_number(book.betfair_event_name)
        )
        others = _other_market_entries(
            other_books.get(book.betfair_event_id, []), sport_cfg, home, away, snapshot_time
        )
        priced.append((book, game, [entry, *others]))

    finder = await LiveEventFinder.for_run(
        sport_cfg.sport_key,
        sport_cfg.sport_title,
        [*(game for _, game, _ in priced), *accounted],
    )
    if finder.schedule_unavailable:
        stats.errors.append("MLB schedule unavailable: no new events created")

    async with async_session_maker() as session:
        envelopes = [(book, game, _envelope(book, entries)) for book, game, entries in priced]
        payload_id = (
            await store_payload(session, [raw for _, _, raw in envelopes]) if envelopes else None
        )
        writes = await finder.store_listings(
            session,
            [
                LiveListing(
                    game=game,
                    external_id=book.betfair_event_id,
                    envelope=envelope,
                    snapshot_time=snapshot_time,
                    label=book.betfair_event_name,
                )
                for book, game, envelope in envelopes
            ],
            SnapshotShape(Source.BETFAIR, TimeBasis.OBSERVED, payload_id),
            now=snapshot_time,
        )
        stats.events_created += writes.events_created
        stats.events_matched += writes.events_matched
        stats.snapshots_stored += writes.snapshots_stored
        stats.errors.extend(writes.errors)
        stats.dropped_games.extend(writes.dropped.values())

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
        books_pre_play_past_start=stats.books_pre_play_past_start,
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
    started = datetime.now(UTC)

    async def decide() -> ScheduleDecision:
        return await decide_collection_run(
            [sport],
            CADENCE,
            overnight=overnight,
            stored_kickoff=next_betfair_kickoff,
            leads=[CLOSE_LEAD],
            run_started=started,
        )

    snapshot_time: datetime | None = None
    cert_file: str | None = None
    cert_key: str | None = None

    def prepare() -> None:
        nonlocal snapshot_time, cert_file, cert_key
        snapshot_time = datetime.now(UTC)
        assert bf.username and bf.password and bf.app_key
        cert_file, cert_key = resolve_cert_paths(bf)

    async def collect() -> CollectionResult:
        assert snapshot_time is not None
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

        return CollectionResult()

    await run_collection(
        job_name=compound_job_name,
        sport=sport,
        dry_run=settings.scheduler.dry_run,
        decide=decide,
        collect=collect,
        post_label="post-fetch",
        prepare=prepare,
    )
