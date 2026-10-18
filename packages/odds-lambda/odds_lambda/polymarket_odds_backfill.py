"""Backfill ``polymarket`` odds snapshots for finished events from CLOB price history."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.sports import SportKey
from sqlalchemy import select

from odds_lambda.polymarket_fetcher import MAX_CONCURRENT_REQUESTS, PolymarketClient
from odds_lambda.polymarket_odds import (
    HISTORY_REQUEST_ID,
    MAX_HORIZON,
    MAX_STALENESS,
    SIDES,
    GridPoint,
    Observation,
    PolymarketMatch,
    Side,
    build_raw_data,
    pair_events,
    parse_match_event,
    resample,
)
from odds_lambda.polymarket_sports import POLYMARKET_SPORTS, PolymarketSportConfig
from odds_lambda.tier_utils import calculate_hours_until_commence, calculate_tier

logger = structlog.get_logger()

INSERT_BATCH_SIZE = 50


@dataclass
class BackfillReport:
    events_considered: int = 0
    events_paired: int = 0
    events_skipped_existing: int = 0
    twin_events_skipped: int = 0
    events_written: int = 0
    events_without_points: int = 0
    snapshots_written: int = 0
    points_missing: int = 0
    points_off_sum: int = 0
    unmatched_event_ids: list[str] = field(default_factory=list)
    ambiguous_event_ids: list[str] = field(default_factory=list)
    unmatched_pm_tickers: list[str] = field(default_factory=list)


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


def _snapshot(
    event: Event, match: PolymarketMatch, sport: PolymarketSportConfig, point: GridPoint
) -> OddsSnapshot:
    hours = calculate_hours_until_commence(point.time, event.commence_time)
    return OddsSnapshot(
        event_id=event.id,
        snapshot_time=point.time,
        raw_data=build_raw_data(match, sport, event, point.prices, point.time),
        bookmaker_count=1,
        api_request_id=HISTORY_REQUEST_ID,
        fetch_tier=calculate_tier(hours).value,
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

        events = list(
            (
                await session.execute(
                    select(Event)
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
        report.events_considered = len(events)
        pairing = pair_events(events, matches)
        report.events_paired = len(pairing.pairs)
        report.twin_events_skipped = sum(len(events) - 1 for _, events in pairing.twins)
        report.unmatched_event_ids = [e.id for e in pairing.unmatched_events]
        report.ambiguous_event_ids = [e.id for e in pairing.ambiguous_events]
        report.unmatched_pm_tickers = [m.ticker for m in pairing.unmatched_matches]

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

        todo = [(e, m) for e, m in pairing.pairs if e.id not in existing]
        report.events_skipped_existing = len(pairing.pairs) - len(todo)
        for event, match in todo[:limit]:
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
                logger.warning("polymarket_no_grid_points", event_id=event.id, ticker=match.ticker)
                continue

            snapshots = [_snapshot(event, match, sport, point) for point in resampled.points]
            report.snapshots_written += len(snapshots)
            report.events_written += 1
            if not dry_run:
                for start in range(0, len(snapshots), INSERT_BATCH_SIZE):
                    session.add_all(snapshots[start : start + INSERT_BATCH_SIZE])
                    await session.flush()
                await session.commit()
            logger.info(
                "polymarket_event_backfilled",
                event_id=event.id,
                ticker=match.ticker,
                snapshots=len(snapshots),
                missing=resampled.missing,
                off_sum=resampled.off_sum,
            )
    return report
