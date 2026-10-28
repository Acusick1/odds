"""Backfill ``polymarket`` odds snapshots for finished events from CLOB price history."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.sports import SportKey
from odds_core.vocabulary import Source, TimeBasis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.polymarket_fetcher import MAX_CONCURRENT_REQUESTS, PolymarketClient
from odds_lambda.polymarket_odds import (
    HISTORY_REQUEST_ID,
    MAX_HORIZON,
    MAX_STALENESS,
    SIDES,
    GridPoint,
    Observation,
    PolymarketMatch,
    Resampled,
    Side,
    build_raw_data,
    listed_game,
    parse_match_event,
    resample,
)
from odds_lambda.polymarket_sports import POLYMARKET_SPORTS, PolymarketSportConfig
from odds_lambda.storage.event_resolver import attach_event
from odds_lambda.storage.shaped_snapshots import (
    SnapshotShape,
    discard_payload,
    store_payload,
    store_shaped_snapshot,
)
from odds_lambda.tier_utils import calculate_hours_until_commence, snapshot_tier

logger = structlog.get_logger()


@dataclass
class BackfillReport:
    matches_considered: int = 0
    matches_attached: int = 0
    matches_out_of_scope: int = 0
    events_skipped_existing: int = 0
    events_written: int = 0
    events_without_points: int = 0
    snapshots_written: int = 0
    points_missing: int = 0
    points_off_sum: int = 0
    unmatched_event_ids: list[str] = field(default_factory=list)
    unmatched_pm_tickers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def _fetch_series(
    client: PolymarketClient,
    match: PolymarketMatch,
    start: datetime,
    end: datetime,
    semaphore: asyncio.Semaphore,
) -> dict[Side, list[Observation]]:
    async def one(side: Side) -> list[Observation]:
        async with semaphore:
            return await client.get_price_series(
                match.outcome_market(side).yes_token_id, start, end
            )

    return dict(zip(SIDES, await asyncio.gather(*(one(side) for side in SIDES)), strict=True))


def _snapshot(event: Event, envelope: dict[str, Any], point: GridPoint) -> OddsSnapshot:
    hours = calculate_hours_until_commence(point.time, event.commence_time)
    return OddsSnapshot(
        event_id=event.id,
        snapshot_time=point.time,
        raw_data=envelope,
        bookmaker_count=1,
        api_request_id=HISTORY_REQUEST_ID,
        fetch_tier=snapshot_tier(point.time, event.commence_time, TimeBasis.OBSERVED).value,
        hours_until_commence=hours,
    )


async def backfill_sport(
    sport_key: SportKey,
    *,
    since: datetime,
    limit: int | None = None,
    dry_run: bool = False,
) -> BackfillReport:
    sport = POLYMARKET_SPORTS[sport_key]
    if not sport.three_way_markets:
        raise ValueError(f"{sport_key} has no three-way match markets to snapshot")

    report = BackfillReport()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with PolymarketClient() as client, async_session_maker() as session:
        raw_events = await client.get_all_events(sport_key, active=None, closed=True)
        matches = [m for raw in raw_events if (m := parse_match_event(raw, sport))]

        final_event_ids = list(
            (
                await session.execute(
                    select(Event.id)
                    .where(
                        Event.sport_key == sport_key,
                        Event.status == EventStatus.FINAL,
                        Event.commence_time >= since,
                    )
                    .order_by(Event.commence_time)
                )
            )
            .scalars()
            .all()
        )
        existing = set(
            (
                await session.execute(
                    select(OddsSnapshot.event_id)
                    .where(OddsSnapshot.api_request_id == HISTORY_REQUEST_ID)
                    .distinct()
                )
            )
            .scalars()
            .all()
        )
        report.matches_considered = len(matches)

        attached_ids: set[str] = set()
        written = 0
        for match in matches:
            try:
                async with session.begin_nested():
                    found = await attach_event(session, listed_game(match, sport))
                if not dry_run:
                    await session.commit()
                if found is None:
                    report.unmatched_pm_tickers.append(match.ticker)
                    continue
                event = found.event
                report.matches_attached += 1
                if event.status != EventStatus.FINAL or event.commence_time < since:
                    report.matches_out_of_scope += 1
                    continue
                attached_ids.add(event.id)
                if event.id in existing:
                    report.events_skipped_existing += 1
                    continue
                if limit is not None and written >= limit:
                    continue
                written += 1

                kickoff = event.commence_time
                cutoff = min(kickoff, match.start_time)
                series = await _fetch_series(
                    client, match, kickoff - MAX_HORIZON - MAX_STALENESS, cutoff, semaphore
                )
                resampled = resample(series, kickoff, cutoff)
                report.points_missing += resampled.missing
                report.points_off_sum += resampled.off_sum
                if not resampled.points:
                    report.events_without_points += 1
                    logger.warning(
                        "polymarket_no_grid_points", event_id=event.id, ticker=match.ticker
                    )
                    continue

                stored = len(resampled.points)
                if not dry_run:
                    async with session.begin_nested():
                        stored = await _store_history(session, event, match, sport, resampled)
                    await session.commit()
                report.snapshots_written += stored
                report.events_written += 1
                logger.info(
                    "polymarket_event_backfilled",
                    event_id=event.id,
                    ticker=match.ticker,
                    snapshots=stored,
                    missing=resampled.missing,
                    off_sum=resampled.off_sum,
                )
            except Exception as e:
                await session.rollback()
                report.errors.append(f"{match.ticker}: {type(e).__name__}: {e}")
                logger.error("polymarket_backfill_match_failed", ticker=match.ticker, exc_info=True)
        if dry_run:
            await session.rollback()
        report.unmatched_event_ids = [i for i in final_event_ids if i not in attached_ids]
    return report


async def _store_history(
    session: AsyncSession,
    event: Event,
    match: PolymarketMatch,
    sport: PolymarketSportConfig,
    resampled: Resampled,
) -> int:
    """Store one snapshot per grid point, each with its own payload; the snapshots already
    stored are skipped. Returns the number inserted."""
    inserted = 0
    for point in resampled.points:
        envelope = build_raw_data(match, sport, point.prices, point.time)
        payload_id = await store_payload(session, envelope)
        shape = SnapshotShape(Source.POLYMARKET, TimeBasis.OBSERVED, payload_id)
        written = await store_shaped_snapshot(
            session, _snapshot(event, envelope, point), event, shape
        )
        if written.inserted:
            inserted += 1
        else:
            await discard_payload(session, payload_id)
    return inserted
