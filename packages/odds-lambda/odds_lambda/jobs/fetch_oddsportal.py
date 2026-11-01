"""Fetch upcoming odds from OddsPortal over HTTP.

Scrapes pre-match odds for configured leagues, converts to pipeline format,
and stores via OddsWriter. Self-schedules based on proximity to next kickoff.

Scheduling strategy: look up the next kickoff, compute interval from game
proximity, pre-schedule before scraping so the chain survives Lambda
timeouts. After a successful scrape, re-query (scrape may have created new
events) and reschedule at the updated interval. With no fixture in the stored
events or the published schedule within ``DORMANCY_HORIZON_DAYS`` the run only
books its next check-in.

Interval table:
  < 3h  (CLOSING)   → 30 min
  3–12h (PREGAME)   → 1h
  12h+              → 2h
  no fixture in 28d → 6h check-in, no scrape
  DB unreachable    → 1h (fallback)

A run is also added ``SCRAPE_LEAD`` before each active mechanical candidate's
decision time, so the decision reads a snapshot taken just before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.candidates import candidates_dir, load_candidates
from odds_core.database import async_session_maker
from odds_core.vocabulary import Market, Source, TimeBasis

from odds_lambda.candidate_rules import decision_leads
from odds_lambda.collection_calendar import (
    CollectionSource,
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
from odds_lambda.oddsportal_adapter import MatchOdds, convert_market
from odds_lambda.oddsportal_client import FEED_MARKETS, OddsPortalClient, OddsPortalError
from odds_lambda.scheduling.decision import (
    OVERNIGHT_WINDOWS,
    CadenceConfig,
    Lead,
    ScheduleDecision,
)
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_payload

logger = structlog.get_logger()

# Proximity-based polling cadence (hours): feed scrape — moderate cadence.
# 12h+ to kickoff all poll at 2h, matching the previous "far" band.
CADENCE = CadenceConfig(
    closing=0.5,
    pregame=1.0,
    sharp=2.0,
    early=2.0,
    opening=2.0,
    no_game=6.0,
    db_fallback=1.0,
)

# An EPL or MLB scrape takes about a minute (two minutes at most, October 2026), so a run
# this long before a decision time has stored its snapshots by then. An NBA scrape with its
# ladders took about 6 minutes (22 games), so an NBA candidate needs a longer lead.
SCRAPE_LEAD = timedelta(minutes=5)


@dataclass
class LeagueSpec:
    """Configuration for a single league to scrape."""

    league: str  # OddsPortal league slug (e.g. "england-premier-league")
    listing_path: str  # OddsPortal listing page (e.g. "/football/england/premier-league/")
    sport_key: str  # Pipeline sport key (e.g. "soccer_epl")
    sport_title: str  # Display name (e.g. "EPL")
    markets: list[str] = field(default_factory=lambda: ["1x2"])
    primary_market: str = "1x2"
    num_outcomes: int = 3


LEAGUE_SPECS: list[LeagueSpec] = [
    LeagueSpec(
        league="england-premier-league",
        listing_path="/football/england/premier-league/",
        sport_key="soccer_epl",
        sport_title="EPL",
        markets=["1x2", "over_under_2_5"],
    ),
    LeagueSpec(
        league="mlb",
        listing_path="/baseball/usa/mlb/",
        sport_key="baseball_mlb",
        sport_title="MLB",
        markets=["home_away"],
        primary_market="home_away",
        num_outcomes=2,
    ),
    LeagueSpec(
        league="nba",
        listing_path="/basketball/usa/nba/",
        sport_key="basketball_nba",
        sport_title="NBA",
        markets=["home_away", "totals_ladder", "spreads_ladder"],
        primary_market="home_away",
        num_outcomes=2,
    ),
]

# Index by sport_key for fast lookup from event payload
_LEAGUE_SPEC_BY_SPORT: dict[str, LeagueSpec] = {spec.sport_key: spec for spec in LEAGUE_SPECS}

# Index by league name for fast lookup from scrape commands / MCP tools
LEAGUE_SPEC_BY_NAME: dict[str, LeagueSpec] = {spec.league: spec for spec in LEAGUE_SPECS}

SOURCE = CollectionSource(
    key="oddsportal",
    job_name="fetch-oddsportal",
    source=Source.ODDSPORTAL,
    cadence=CADENCE,
    fixture_horizons={
        "soccer_epl": timedelta(hours=48),
        "baseball_mlb": timedelta(hours=48),
        "basketball_nba": timedelta(hours=48),
    },
    markets={
        spec.sport_key: tuple(Market(FEED_MARKETS[key].pipeline_key) for key in spec.markets)
        for spec in LEAGUE_SPECS
    },
    skips_overnight=True,
)


@dataclass
class IngestionStats:
    """Tracks results of a single league ingestion run."""

    league: str = ""
    events_listed: int = 0
    upcoming_listed: int = 0
    latest_listed_start: datetime | None = None
    matches_scraped: int = 0
    matches_converted: int = 0
    events_matched: int = 0
    events_created: int = 0
    dropped_games: list[DroppedGame] = field(default_factory=list)
    snapshots_stored: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def games_unresolved(self) -> int:
        return count_dropped(self.dropped_games, SkipReason.UNRESOLVED)

    @property
    def games_skipped_schedule_unavailable(self) -> int:
        return count_dropped(self.dropped_games, SkipReason.SCHEDULE_UNAVAILABLE)

    @property
    def all_listed_started(self) -> bool:
        return self.events_listed > 0 and self.upcoming_listed == 0


@dataclass
class LeagueScrape:
    """Converted output of one league scrape."""

    matches: list[MatchOdds] = field(default_factory=list)
    events_listed: int = 0
    upcoming_listed: int = 0
    latest_listed_start: datetime | None = None
    errors: list[str] = field(default_factory=list)


def _game_row(match: MatchOdds) -> GameRow:
    return GameRow(match.home_team, match.away_team, match.match_date)


async def scrape_league(spec: LeagueSpec) -> LeagueScrape:
    """Read upcoming events and their odds for a league from OddsPortal.

    Listing and bookmaker-context failures propagate. A failure for one event
    and market is recorded in ``errors`` and the scrape continues.
    """
    scrape = LeagueScrape()
    async with OddsPortalClient() as client:
        events = await client.fetch_listing(spec.listing_path)
        scrape.events_listed = len(events)
        scrape.latest_listed_start = max((e.start_time for e in events), default=None)
        now = datetime.now(UTC)
        upcoming = [e for e in events if e.start_time > now]
        scrape.upcoming_listed = len(upcoming)
        if not upcoming:
            return scrape

        context = await client.fetch_bookmaker_context()
        fetched = 0
        for event in upcoming:
            for market in spec.markets:
                try:
                    bookmakers = await client.fetch_market(event, market, context)
                except OddsPortalError as e:
                    msg = f"{event.home_team} vs {event.away_team} ({market}): {e}"
                    scrape.errors.append(msg)
                    logger.warning("oddsportal_market_failed", error=msg)
                    continue
                fetched += 1
                match = convert_market(event, market, bookmakers, datetime.now(UTC))
                if match is not None:
                    scrape.matches.append(match)
    if fetched == 0 and scrape.errors:
        raise OddsPortalError(
            f"every market fetch failed for {len(upcoming)} upcoming events "
            f"({len(scrape.errors)} errors); first: {scrape.errors[0]}"
        )
    return scrape


async def ingest_league(
    spec: LeagueSpec,
    *,
    scraped: LeagueScrape | None = None,
) -> IngestionStats:
    """Scrape and ingest a single league.

    Args:
        spec: League configuration.
        scraped: Pre-scraped data (skips the network). For testing.

    Returns:
        Ingestion statistics.
    """
    stats = IngestionStats(league=spec.league)

    if scraped is None:
        logger.info("scraping_league", league=spec.league, sport_key=spec.sport_key)
        scraped = await scrape_league(spec)

    stats.events_listed = scraped.events_listed
    stats.upcoming_listed = scraped.upcoming_listed
    stats.latest_listed_start = scraped.latest_listed_start
    all_converted = scraped.matches
    stats.errors.extend(scraped.errors)
    stats.matches_converted = len(all_converted)
    stats.matches_scraped = len({m.match_link for m in all_converted})

    if not all_converted:
        logger.warning(
            "no_matches_scraped", league=spec.league, events_listed=scraped.events_listed
        )
        return stats

    logger.info(
        "matches_scraped",
        league=spec.league,
        count=stats.matches_scraped,
        events_listed=scraped.events_listed,
    )

    # Ingest to database
    finder = await LiveEventFinder.for_run(
        spec.sport_key, spec.sport_title, (_game_row(m) for m in all_converted)
    )
    if finder.schedule_unavailable:
        stats.errors.append("MLB schedule unavailable: no new events created")

    async with async_session_maker() as session:
        payload_id = await store_payload(session, [m.envelope for m in all_converted])
        writes = await finder.store_listings(
            session,
            [
                LiveListing(
                    game=_game_row(match),
                    external_id=match.envelope.get("oddsportal_event_id"),
                    envelope=match.envelope,
                    snapshot_time=match.scraped_date,
                    label=f"{match.home_team} vs {match.away_team} ({match.market})",
                )
                for match in all_converted
            ],
            SnapshotShape(Source.ODDSPORTAL, TimeBasis.OBSERVED, payload_id),
        )
        await session.commit()

    stats.events_created += writes.events_created
    stats.events_matched += writes.events_matched
    stats.snapshots_stored += writes.snapshots_stored
    stats.errors.extend(writes.errors)
    stats.dropped_games = list(writes.dropped.values())
    logger.info(
        "league_ingestion_complete",
        league=spec.league,
        matches_scraped=stats.matches_scraped,
        events_matched=stats.events_matched,
        events_created=stats.events_created,
        games_unresolved=stats.games_unresolved,
        games_skipped_schedule_unavailable=stats.games_skipped_schedule_unavailable,
        snapshots_stored=stats.snapshots_stored,
        errors=len(stats.errors),
    )

    return stats


def candidate_scrape_leads(sport_key: str) -> list[Lead]:
    """A run ``SCRAPE_LEAD`` before each active mechanical candidate's decision time, which
    still runs late until that decision time.

    An unreadable registry gives none, so collection keeps its cadence.
    """
    try:
        candidates = load_candidates(candidates_dir())
    except ValueError as e:
        logger.error("candidate_registry_invalid", error=str(e))
        return []
    return [Lead(lead + SCRAPE_LEAD, until=lead) for lead in decision_leads(candidates, sport_key)]


async def main(ctx: JobContext) -> None:
    """Main job execution — scrapes configured league(s), then self-schedules.

    Scheduling strategy: query DB for next kickoff, compute proximity-based
    interval, pre-schedule before scraping. After successful scrape,
    re-query (new events may have appeared) and reschedule.
    """
    from odds_core.alerts import job_alert_context
    from odds_core.config import get_settings

    settings = get_settings()
    sport = require_sport(ctx, "fetch-oddsportal")

    spec = _LEAGUE_SPEC_BY_SPORT.get(sport)
    if spec is None:
        logger.error("unknown_sport_key", sport=sport, available=list(_LEAGUE_SPEC_BY_SPORT))
        return

    compound_job_name = make_compound_job_name("fetch-oddsportal", sport)
    overnight = OVERNIGHT_WINDOWS[spec.sport_key]
    sport_keys = [spec.sport_key]
    leads = candidate_scrape_leads(spec.sport_key)
    started = datetime.now(UTC)

    async def _reschedule(label: str) -> ScheduleDecision:
        decision = await decide_collection_run(
            sport_keys, CADENCE, overnight=overnight, leads=leads, run_started=started
        )
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
            logger.error("fetch_oddsportal_scheduling_failed", error=str(e), exc_info=True)
            if label == "pre-schedule":
                raise
        return decision

    # Pre-schedule before scraping so the chain survives Lambda timeouts
    # or a hung scrape.
    decision = await _reschedule("pre-schedule")

    if not decision.should_execute:
        async with job_alert_context(compound_job_name):
            logger.info("fetch_oddsportal_dormant", sport=sport, reason=decision.reason)
        return

    async with job_alert_context(compound_job_name):
        logger.info(
            "fetch_oddsportal_started",
            sport=sport,
        )

        try:
            stats = await ingest_league(spec)
        except Exception as e:
            logger.error(
                "league_failed",
                league=spec.league,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(
                f"{compound_job_name} failed: {spec.league}: {type(e).__name__}: {e}"
            ) from e

        total_scraped = stats.matches_scraped
        await alert_dropped_games(compound_job_name, stats.dropped_games)

        logger.info(
            "fetch_oddsportal_completed",
            total_matches_scraped=total_scraped,
            total_snapshots_stored=stats.snapshots_stored,
            total_errors=len(stats.errors),
        )

        if stats.all_listed_started:
            logger.warning(
                "oddsportal_empty_run_check_skipped",
                sport=spec.sport_key,
                reason="every listed event has already started",
                events_listed=stats.events_listed,
                latest_listed_start=stats.latest_listed_start,
            )
        else:
            await require_rows_for_fixtures(
                SOURCE,
                {spec.sport_key: stats.snapshots_stored},
                errors_by_sport={spec.sport_key: stats.errors},
            )

    # On success, re-query next kickoff (scrape may have created new events)
    # and reschedule at the updated interval.
    if total_scraped > 0:
        await _reschedule("post-scrape")
