"""Fetch current Polymarket match prices as ``polymarket`` odds snapshots.

Reads the open match events of a sport from Gamma, pairs them with scheduled
``events`` rows, and stores each pair's three Yes midpoints through
``OddsWriter.store_odds_snapshot``. Self-schedules with the cadence that
mirrors the backfill grid spacing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.alerts import job_alert_context, send_job_warning
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, FetchLog
from odds_core.snapshot_sources import SnapshotSource
from odds_core.sports import SportKey
from sqlalchemy import select

from odds_lambda.collection_calendar import CollectionSource, decide_collection_run
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.polymarket_fetcher import PolymarketClient
from odds_lambda.polymarket_odds import (
    EVENT_MATCH_WINDOW,
    LIVE_REQUEST_ID_PREFIX,
    MAX_HORIZON,
    MAX_QUOTE_SPREAD,
    TIER_STEPS,
    build_raw_data,
    pair_events,
    parse_match_event,
    sum_is_valid,
)
from odds_lambda.polymarket_sports import MATCH_ODDS_SPORTS, POLYMARKET_SPORTS
from odds_lambda.scheduling.decision import (
    OVERNIGHT_WINDOWS,
    CadenceConfig,
    ScheduleDecision,
)
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

_NO_GAME_HOURS = 6.0
LISTING_CHECK_HORIZON = timedelta(days=5)


def _hours(tier: FetchTier) -> float:
    return TIER_STEPS[tier].total_seconds() / 3600


CADENCE = CadenceConfig(
    closing=_hours(FetchTier.CLOSING),
    pregame=_hours(FetchTier.PREGAME),
    sharp=_hours(FetchTier.SHARP),
    early=_hours(FetchTier.EARLY),
    opening=_hours(FetchTier.OPENING),
    no_game=_NO_GAME_HOURS,
    db_fallback=1.0,
)

SOURCE = CollectionSource(
    key="polymarket",
    job_name="fetch-polymarket-odds",
    snapshot_source=SnapshotSource.POLYMARKET,
    cadence=CADENCE,
    fixture_horizons=dict.fromkeys(MATCH_ODDS_SPORTS, LISTING_CHECK_HORIZON),
    skips_overnight=True,
)


class EmptyListingError(RuntimeError):
    """Gamma lists no upcoming match for a sport that has fixtures in the database."""


@dataclass
class IngestionStats:
    sport: SportKey
    matches_seen: int = 0
    matches_unpaired: int = 0
    prices_missing: int = 0
    wide_spread: int = 0
    sum_rejected: int = 0
    already_started: int = 0
    snapshots_stored: int = 0
    errors: list[str] = field(default_factory=list)


async def ingest_sport(sport_key: SportKey, snapshot_time: datetime) -> IngestionStats:
    sport = POLYMARKET_SPORTS[sport_key]
    stats = IngestionStats(sport=sport_key)

    async with PolymarketClient() as client:
        raw_events = await client.get_all_events(sport_key, active=True, closed=False)

    horizon = snapshot_time + MAX_HORIZON
    matches = [
        m
        for raw in raw_events
        if (m := parse_match_event(raw, sport)) and snapshot_time < m.start_time <= horizon
    ]
    stats.matches_seen = len(matches)

    async with async_session_maker() as session:
        events = list(
            (
                await session.execute(
                    select(Event).where(
                        Event.sport_key == sport_key,
                        Event.commence_time > snapshot_time - EVENT_MATCH_WINDOW,
                        Event.commence_time <= horizon + EVENT_MATCH_WINDOW,
                    )
                )
            )
            .scalars()
            .all()
        )
        listing_error: EmptyListingError | None = None
        scheduled_soon = [
            e
            for e in events
            if e.status == EventStatus.SCHEDULED
            and snapshot_time < e.commence_time <= snapshot_time + LISTING_CHECK_HORIZON
        ]
        if not matches and scheduled_soon:
            listing_error = EmptyListingError(
                f"Gamma lists no upcoming {sport_key} match; "
                f"{len(scheduled_soon)} event(s) are scheduled within {LISTING_CHECK_HORIZON}"
            )
            stats.errors.append(str(listing_error))

        pairing = pair_events(events, matches)
        stats.matches_unpaired = len(pairing.unmatched_matches)

        writer = OddsWriter(session)
        for event, match in pairing.pairs:
            if snapshot_time >= event.commence_time:
                stats.already_started += 1
                continue
            prices = match.midpoints()
            if prices is None:
                stats.prices_missing += 1
                continue
            spread = match.max_spread()
            if spread is not None and spread > MAX_QUOTE_SPREAD:
                stats.wide_spread += 1
                continue
            if not sum_is_valid(prices):
                stats.sum_rejected += 1
                continue
            try:
                snapshot, _ = await writer.store_odds_snapshot(
                    event_id=event.id,
                    raw_data=build_raw_data(
                        match, sport, event, prices, snapshot_time, include_quotes=True
                    ),
                    snapshot_time=snapshot_time,
                )
                snapshot.api_request_id = f"{LIVE_REQUEST_ID_PREFIX}{snapshot_time.isoformat()}"
                stats.snapshots_stored += 1
            except Exception as e:
                stats.errors.append(f"{match.ticker}: {type(e).__name__}: {e}")
                logger.error("polymarket_odds_ingestion_failed", ticker=match.ticker, exc_info=True)

        if stats.matches_seen or stats.errors:
            session.add(
                FetchLog(
                    fetch_time=snapshot_time,
                    sport_key=sport_key,
                    events_count=stats.snapshots_stored,
                    bookmakers_count=1,
                    success=stats.snapshots_stored > 0,
                    error_message="; ".join(stats.errors[:3]) or None,
                )
            )
        await session.commit()

    logger.info(
        "polymarket_odds_sport_ingestion_complete",
        sport=sport_key,
        matches_seen=stats.matches_seen,
        matches_unpaired=stats.matches_unpaired,
        prices_missing=stats.prices_missing,
        wide_spread=stats.wide_spread,
        sum_rejected=stats.sum_rejected,
        already_started=stats.already_started,
        snapshots_stored=stats.snapshots_stored,
        errors=len(stats.errors),
    )
    if listing_error is not None:
        raise listing_error
    return stats


async def _log_failure(sport_key: SportKey, snapshot_time: datetime, error: str) -> None:
    async with async_session_maker() as session:
        session.add(
            FetchLog(
                fetch_time=snapshot_time,
                sport_key=sport_key,
                events_count=0,
                bookmakers_count=1,
                success=False,
                error_message=error[:500],
            )
        )
        await session.commit()


async def main(ctx: JobContext) -> None:
    settings = get_settings()
    if not settings.polymarket.enabled:
        logger.info("polymarket_odds_disabled", reason="polymarket.enabled=False")
        return

    sport: SportKey | None = ctx.sport
    if sport is not None and sport not in MATCH_ODDS_SPORTS:
        logger.error("polymarket_odds_unknown_sport", sport=sport, supported=MATCH_ODDS_SPORTS)
        return
    target_sports: list[SportKey] = [sport] if sport else list(MATCH_ODDS_SPORTS)

    compound_job_name = make_compound_job_name("fetch-polymarket-odds", sport)
    overnight = OVERNIGHT_WINDOWS.get(target_sports[0]) if len(target_sports) == 1 else None

    async def _reschedule(label: str) -> ScheduleDecision:
        decision = await decide_collection_run(
            target_sports, CADENCE, overnight=overnight, stored_only=True
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
            logger.error("polymarket_odds_schedule_failed", error=str(e), exc_info=True)
            if label == "pre-schedule":
                raise
        return decision

    decision = await _reschedule("pre-schedule")

    if not decision.should_execute:
        async with job_alert_context(compound_job_name):
            logger.info("polymarket_odds_dormant", sports=target_sports, reason=decision.reason)
        return

    snapshot_time = datetime.now(UTC)
    all_stats: list[IngestionStats] = []
    failures: list[str] = []
    async with job_alert_context(compound_job_name):
        for sport_key in target_sports:
            try:
                all_stats.append(await ingest_sport(sport_key, snapshot_time))
            except Exception as e:
                logger.error(
                    "polymarket_odds_sport_ingestion_failed",
                    sport=sport_key,
                    error=str(e),
                    exc_info=True,
                )
                failures.append(f"{sport_key}: {type(e).__name__}: {e}")
                if not isinstance(e, EmptyListingError):
                    await _log_failure(sport_key, snapshot_time, failures[-1])

        matches_seen = sum(s.matches_seen for s in all_stats)
        if matches_seen > 0 and sum(s.snapshots_stored for s in all_stats) == 0:
            await send_job_warning(
                f"job_empty:{compound_job_name}",
                f"{compound_job_name} stored 0 snapshots from {matches_seen} match(es).",
            )

        await _reschedule("post-fetch")

        if failures:
            raise RuntimeError(f"{compound_job_name} failed: {'; '.join(failures)}")
