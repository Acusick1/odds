"""Fetch current Polymarket match prices as ``polymarket`` odds snapshots.

Reads the open match events of a sport from Gamma, attaches each to its existing
event through the resolver (Polymarket creates none), and stores the outcome
midpoints and quotes through ``OddsWriter.store_envelope``. Self-schedules
with the cadence that mirrors the backfill grid spacing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from odds_core.alerts import job_alert_context, send_job_warning
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, FetchLog
from odds_core.sports import SportKey
from odds_core.vocabulary import Market, Source, TimeBasis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.collection_calendar import CollectionSource, decide_collection_run
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.polymarket_fetcher import PolymarketClient
from odds_lambda.polymarket_odds import (
    MAX_HORIZON,
    MAX_QUOTE_SPREAD,
    TIER_STEPS,
    PolymarketMatch,
    build_envelope,
    listed_game,
    parse_match_event,
    sum_is_valid,
)
from odds_lambda.polymarket_sports import POLYMARKET_SPORTS
from odds_lambda.scheduling.decision import (
    OVERNIGHT_WINDOWS,
    CadenceConfig,
    ScheduleDecision,
)
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport
from odds_lambda.storage.event_resolver import attach_events
from odds_lambda.storage.shaped_snapshots import SnapshotShape, discard_payload, store_payload
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
    source=Source.POLYMARKET,
    cadence=CADENCE,
    fixture_horizons=dict.fromkeys(POLYMARKET_SPORTS, LISTING_CHECK_HORIZON),
    markets={
        sport_key: (Market(config.market_key),) for sport_key, config in POLYMARKET_SPORTS.items()
    },
    skips_overnight=True,
)


class EmptyListingError(RuntimeError):
    """Gamma lists no upcoming match for a sport that has fixtures in the database."""


@dataclass
class IngestionStats:
    sport: SportKey
    matches_seen: int = 0
    matches_unattached: int = 0
    prices_missing: int = 0
    wide_spread: int = 0
    sum_rejected: int = 0
    already_started: int = 0
    snapshots_stored: int = 0
    errors: list[str] = field(default_factory=list)


async def _scheduled_soon(session: AsyncSession, sport_key: SportKey, now: datetime) -> int:
    result = await session.execute(
        select(func.count())
        .select_from(Event)
        .where(
            Event.sport_key == sport_key,
            Event.status == EventStatus.SCHEDULED,
            Event.commence_time > now,
            Event.commence_time <= now + LISTING_CHECK_HORIZON,
        )
    )
    return result.scalar_one()


async def ingest_sport(sport_key: SportKey, snapshot_time: datetime) -> IngestionStats:
    sport = POLYMARKET_SPORTS[sport_key]
    stats = IngestionStats(sport=sport_key)

    horizon = snapshot_time + MAX_HORIZON
    async with PolymarketClient() as client:
        raw_events = await client.get_all_events(
            sport_key, active=True, closed=False, ends_after=snapshot_time, ends_before=horizon
        )

    matches = [
        m
        for raw in raw_events
        if (m := parse_match_event(raw, sport)) and snapshot_time < m.start_time <= horizon
    ]
    stats.matches_seen = len(matches)

    async with async_session_maker() as session:
        listing_error: EmptyListingError | None = None
        if not matches:
            scheduled_soon = await _scheduled_soon(session, sport_key, snapshot_time)
            if scheduled_soon:
                listing_error = EmptyListingError(
                    f"Gamma lists no upcoming {sport_key} match; "
                    f"{scheduled_soon} event(s) are scheduled within {LISTING_CHECK_HORIZON}"
                )
                stats.errors.append(str(listing_error))

        resolved = await attach_events(session, [listed_game(m, sport) for m in matches])
        attached: list[tuple[PolymarketMatch, Event]] = []
        for match, found in zip(matches, resolved, strict=True):
            if found is None:
                stats.matches_unattached += 1
            else:
                attached.append((match, found.event))

        quoted: list[tuple[PolymarketMatch, Event, dict[str, Any]]] = []
        for match, event in attached:
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
            envelope = build_envelope(match, sport, prices, snapshot_time, include_quotes=True)
            quoted.append((match, event, envelope))

        if quoted:
            payload_id = await store_payload(session, [envelope for _, _, envelope in quoted])
            shape = SnapshotShape(Source.POLYMARKET, TimeBasis.OBSERVED, payload_id)
            writer = OddsWriter(session)
            for match, event, envelope in quoted:
                try:
                    async with session.begin_nested():
                        stored = await writer.store_envelope(
                            event_id=event.id,
                            envelope=envelope,
                            snapshot_time=snapshot_time,
                            shape=shape,
                        )
                    stats.snapshots_stored += int(stored.inserted)
                except Exception as e:
                    stats.errors.append(f"{match.ticker}: {type(e).__name__}: {e}")
                    logger.error(
                        "polymarket_odds_ingestion_failed", ticker=match.ticker, exc_info=True
                    )
            if stats.snapshots_stored == 0:
                await discard_payload(session, payload_id)

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
        matches_unattached=stats.matches_unattached,
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
    sport = require_sport(ctx, "fetch-polymarket-odds")
    settings = get_settings()
    if not settings.polymarket.enabled:
        logger.info("polymarket_odds_disabled", reason="polymarket.enabled=False")
        return

    if sport not in POLYMARKET_SPORTS:
        logger.error(
            "polymarket_odds_unknown_sport", sport=sport, supported=tuple(POLYMARKET_SPORTS)
        )
        return

    compound_job_name = make_compound_job_name("fetch-polymarket-odds", sport)
    overnight = OVERNIGHT_WINDOWS.get(sport)

    async def _reschedule(label: str) -> ScheduleDecision:
        decision = await decide_collection_run(
            [sport], CADENCE, overnight=overnight, stored_only=True
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
            logger.info("polymarket_odds_dormant", sport=sport, reason=decision.reason)
        return

    snapshot_time = datetime.now(UTC)
    stats: IngestionStats | None = None
    failure: str | None = None
    async with job_alert_context(compound_job_name):
        try:
            stats = await ingest_sport(sport, snapshot_time)
        except Exception as e:
            logger.error(
                "polymarket_odds_sport_ingestion_failed",
                sport=sport,
                error=str(e),
                exc_info=True,
            )
            failure = f"{sport}: {type(e).__name__}: {e}"
            if not isinstance(e, EmptyListingError):
                await _log_failure(sport, snapshot_time, failure)

        if stats is not None and stats.matches_seen > 0 and stats.snapshots_stored == 0:
            await send_job_warning(
                f"job_empty:{compound_job_name}",
                f"{compound_job_name} stored 0 snapshots from {stats.matches_seen} match(es).",
            )

        await _reschedule("post-fetch")

        if failure is not None:
            raise RuntimeError(f"{compound_job_name} failed: {failure}")
