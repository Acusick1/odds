"""
Fetch odds job - standalone executable or Lambda handler.

This job:
1. Checks if odds fetch should execute (game-aware gating)
2. Fetches current odds for all upcoming games
3. Stores raw snapshots and normalized data
4. Self-schedules next execution based on game proximity
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial

import structlog
from odds_core.alerts import job_alert_context, send_outage_warning
from odds_core.config import Settings, get_settings
from odds_core.vocabulary import Market, Source

from odds_lambda.collection_calendar import CollectionSource, require_rows_for_fixtures
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.event_sync import EventSyncService
from odds_lambda.ingestion import OddsIngestionService
from odds_lambda.live_events import alert_dropped_games
from odds_lambda.scheduling.decision import (
    CadenceConfig,
    KickoffLookup,
    Lead,
    OvernightWindow,
    decide_forward_resilient,
)
from odds_lambda.scheduling.helpers import (
    book_next_run,
    get_next_slate_main_tip,
    get_next_slate_start,
)
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport

logger = structlog.get_logger()

# Proximity-based polling cadence (hours). Each /odds call costs 3 units against
# the 500/month free tier, so the closing tier polls every 2 h and the pregame
# tier every 3 h; the Betfair and OddsPortal jobs carry the close.
CADENCE = CadenceConfig(
    closing=2.0,
    pregame=3.0,
    sharp=12.0,
    early=24.0,
    opening=48.0,
    no_game=24.0,
)

SOURCE = CollectionSource(
    key="odds_api",
    job_name="fetch-odds",
    source=Source.ODDS_API,
    cadence=CADENCE,
    fixture_horizons={"soccer_epl": timedelta(hours=48)},
    markets={"soccer_epl": (Market.ONE_X_TWO, Market.SPREADS, Market.TOTALS)},
    skips_overnight=False,
)

# NBA has a second key of 500 units a month (3 units a call), so about 160 calls. A call prices
# the whole slate, so the schedule is set per slate, not per game: runs 3 h and 1 h before the
# slate's first tip-off and 1 h before its main tip (the leads), plus a morning run that falls out
# of the 12 h interval after the previous night's last lead run, held to NBA_QUIET_HOURS' end.
# Replayed over stored NBA events that is 100-117 calls (300-351 units) in 30 days, and the
# share of games last priced over 3 h before tip-off falls from 36-45% to 14-21%.
NBA_CADENCE = CadenceConfig(
    closing=12.0,
    pregame=12.0,
    sharp=12.0,
    early=24.0,
    opening=24.0,
    no_game=24.0,
)
NBA_LEADS = (
    Lead(before=timedelta(hours=3), until=timedelta(hours=2)),
    Lead(before=timedelta(hours=1), until=timedelta(minutes=20)),
    # A slate's main tip (the one most games share) can be hours after its first, so its games
    # would be priced 3 h or more out; this lead keys off the main tip and has no run to make
    # when the main tip is within an hour of the first.
    Lead(
        before=timedelta(hours=1),
        until=timedelta(minutes=20),
        kickoff_lookup=get_next_slate_main_tip,
    ),
)
# Cadence runs are not taken from 00:00 to 10:00 UTC (the late games are under way and the
# lines of the next slate are posted by the morning); leads ignore the window.
NBA_QUIET_HOURS: OvernightWindow = (0, 10)

# A run that wrote no row fails when an NBA game is within this horizon. NBA stays out of ``SOURCE``
# because the stale-source check derives its threshold from ``CADENCE``, which would fire between
# the sparse NBA runs.
NBA_SOURCE = CollectionSource(
    key="odds_api_nba",
    job_name="fetch-odds",
    source=Source.ODDS_API,
    cadence=NBA_CADENCE,
    fixture_horizons={"basketball_nba": timedelta(hours=14)},
    markets={"basketball_nba": (Market.H2H, Market.SPREADS, Market.TOTALS)},
    skips_overnight=False,
)


@dataclass(frozen=True, slots=True)
class FetchPlan:
    """When a sport's fetch-odds runs and the check that a run wrote rows."""

    cadence: CadenceConfig
    source: CollectionSource
    leads: tuple[Lead, ...] = ()
    overnight: OvernightWindow | None = None
    kickoff_lookup: KickoffLookup | None = None


PLANS: dict[str, FetchPlan] = {
    "soccer_epl": FetchPlan(CADENCE, SOURCE),
    "basketball_nba": FetchPlan(
        NBA_CADENCE,
        NBA_SOURCE,
        leads=NBA_LEADS,
        overnight=NBA_QUIET_HOURS,
        kickoff_lookup=get_next_slate_start,
    ),
}


def build_ingestion_service(client: TheOddsAPIClient, settings: Settings) -> OddsIngestionService:
    """Factory to create ingestion service; exposed for test patching."""
    return OddsIngestionService(client=client, settings=settings)


def build_event_sync_service(client: TheOddsAPIClient) -> EventSyncService:
    """Factory to create event sync service; exposed for test patching."""
    return EventSyncService(client=client)


async def main(ctx: JobContext) -> None:
    """Main job execution flow.

    Flow:
    1. Sync upcoming events from free /events endpoint (discovery)
    2. Check if we should execute odds fetch (smart gating)
    3. If yes: fetch and store odds
    4. Calculate next execution time
    5. Schedule next run via backend
    """
    started = datetime.now(UTC)
    app_settings = get_settings()
    sport = require_sport(ctx, "fetch-odds")
    sports = [sport]
    if sport not in PLANS:
        raise ValueError(f"fetch-odds has no plan for {sport}; add one to PLANS")
    plan = PLANS[sport]

    logger.info(
        "fetch_odds_job_started",
        backend=app_settings.scheduler.backend,
        sport=sport,
        sports=sports,
    )

    # Self-scheduling uses the sport-suffixed job name to match Terraform rules
    schedule_job_name = make_compound_job_name("fetch-odds", sport)

    # Sync upcoming events first (free, 0 quota units).
    # Failures here should not block odds ingestion.
    try:
        async with TheOddsAPIClient() as client:
            event_sync = build_event_sync_service(client)
            sync_results = await event_sync.sync_sports(sports)

        for sync_result in sync_results:
            logger.info(
                "event_sync_result",
                sport=sync_result.sport_key,
                inserted=sync_result.inserted,
                updated=sync_result.updated,
            )
    except Exception as e:
        logger.error("event_sync_failed", error=str(e), exc_info=True)
        await send_outage_warning(
            schedule_job_name,
            f"event_sync_failed:{schedule_job_name}",
            f"🚨 {schedule_job_name} event sync failed: {type(e).__name__}: {e}",
            severity="critical",
        )

    schedule = partial(
        book_next_run,
        job_name=schedule_job_name,
        sport=sport,
        dry_run=app_settings.scheduler.dry_run,
    )

    # Smart execution gating, scoped to this job's sport(s) — fixes the prior
    # all-sports bug where another sport's fixtures kept this job alive. The
    # resilient path keys off the soonest kickoff across this job's sports and
    # falls back to db_fallback cadence (should_execute=True, burning one cheap
    # /odds call) if the kickoff query fails, so the chain never breaks.
    decision = await decide_forward_resilient(
        sports,
        plan.cadence,
        overnight=plan.overnight,
        lookahead_days=app_settings.scheduler.lookahead_days,
        next_kickoff_lookup=plan.kickoff_lookup,
        leads=plan.leads,
        lead_kickoff_lookup=plan.kickoff_lookup,
        run_started=started,
    )

    # Pre-schedule before any work so the chain survives crashes.
    pre_booked = await schedule(decision, label="pre-schedule", final=not decision.should_execute)

    async with job_alert_context(schedule_job_name):
        if not decision.should_execute:
            logger.info(
                "fetch_odds_skipped",
                reason=decision.reason,
                next_check=decision.next_execution.isoformat() if decision.next_execution else None,
            )
            return

        logger.info(
            "fetch_odds_executing",
            reason=decision.reason,
            tier=decision.tier.value if decision.tier else None,
        )

        try:
            async with TheOddsAPIClient() as client:
                ingestion_service = build_ingestion_service(client, app_settings)
                results = await ingestion_service.ingest_sports(sports)

            for sport_result in results.sport_results:
                for failure in sport_result.failures:
                    logger.warning(
                        "event_processing_failed",
                        sport=sport_result.sport_key,
                        odds_api_id=failure.odds_api_id,
                        error=failure.error,
                    )

                await alert_dropped_games(schedule_job_name, sport_result.dropped_games)

                logger.info(
                    "sport_fetch_completed",
                    sport=sport_result.sport_key,
                    processed_events=sport_result.processed_events,
                    total_events=sport_result.total_events,
                    quota_remaining=sport_result.quota_remaining,
                )

            await require_rows_for_fixtures(
                plan.source, {r.sport_key: r.processed_events for r in results.sport_results}
            )

            logger.info(
                "fetch_odds_completed",
                total_processed=results.total_processed,
                total_events=results.total_events,
                total_failures=results.total_failures,
            )

        except Exception as e:
            logger.error("fetch_odds_failed", error=str(e), exc_info=True)
            raise

    # Re-query and reschedule — ingestion may have created new events that move
    # the proximity tier.
    post_decision = await decide_forward_resilient(
        sports,
        plan.cadence,
        overnight=plan.overnight,
        lookahead_days=app_settings.scheduler.lookahead_days,
        next_kickoff_lookup=plan.kickoff_lookup,
        leads=plan.leads,
        lead_kickoff_lookup=plan.kickoff_lookup,
        run_started=started,
    )
    await schedule(post_decision, label="post-fetch", final=not pre_booked)
    logger.info(
        "fetch_odds_next_scheduled",
        next_time=post_decision.next_execution.isoformat()
        if post_decision.next_execution
        else None,
        tier=post_decision.tier.value if post_decision.tier else None,
    )
