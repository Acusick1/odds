"""Backfill ``polymarket`` odds snapshots for finished events from CLOB price history."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot, SnapshotPayload
from odds_core.sports import SportKey
from odds_core.vocabulary import Source, TimeBasis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_lambda.polymarket_fetcher import MAX_CONCURRENT_REQUESTS, PolymarketClient
from odds_lambda.polymarket_odds import (
    MAX_HORIZON,
    MAX_STALENESS,
    Observation,
    PolymarketMatch,
    Resampled,
    Side,
    build_envelope,
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
    unmatched_event_ids: list[int] = field(default_factory=list)
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

    return dict(
        zip(match.sides, await asyncio.gather(*(one(side) for side in match.sides)), strict=True)
    )


async def _history_event_ids(session: AsyncSession) -> set[int]:
    """Events holding a Polymarket history snapshot: one whose payload is its own envelope with
    no quotes, where a live run's envelopes, listed in its payload, always carry the bid and ask
    keys (``build_envelope`` with ``include_quotes``)."""
    meta = SnapshotPayload.payload[("bookmakers", 0, "polymarket_meta")]  # type: ignore[index]
    rows = await session.execute(
        select(OddsSnapshot.event_id)
        .join(SnapshotPayload, col(SnapshotPayload.id) == col(OddsSnapshot.payload_id))
        .where(
            OddsSnapshot.source == Source.POLYMARKET,
            func.jsonb_typeof(SnapshotPayload.payload) == "object",
            ~meta.has_key("best_bid"),
        )
        .distinct()
    )
    return set(rows.scalars())


async def backfill_sport(
    sport_key: SportKey,
    *,
    since: datetime,
    limit: int | None = None,
    dry_run: bool = False,
) -> BackfillReport:
    sport = POLYMARKET_SPORTS[sport_key]
    report = BackfillReport()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with PolymarketClient() as client, async_session_maker() as session:
        raw_events = await client.get_all_events(
            sport_key, active=None, closed=True, ends_after=since
        )
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
        existing = await _history_event_ids(session)
        report.matches_considered = len(matches)

        attached_ids: set[int] = set()
        written = 0
        for match in matches:
            try:
                async with session.begin_nested():
                    found = await attach_event(session, listed_game(match, sport))
                # A rollback expires the ORM event, so read what the loop needs first.
                event = found.event if found is not None else None
                event_id = event.id if event is not None else None
                status = event.status if event is not None else None
                kickoff = event.commence_time if event is not None else None
                # Either way the fixture lock is released, so the next match may take a lower key.
                if dry_run:
                    await session.rollback()
                else:
                    await session.commit()
                if event is None or event_id is None or status is None or kickoff is None:
                    report.unmatched_pm_tickers.append(match.ticker)
                    continue
                report.matches_attached += 1
                if status != EventStatus.FINAL or kickoff < since:
                    report.matches_out_of_scope += 1
                    continue
                attached_ids.add(event_id)
                if event_id in existing:
                    report.events_skipped_existing += 1
                    continue
                if limit is not None and written >= limit:
                    continue
                written += 1

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
                        "polymarket_no_grid_points", event_id=event_id, ticker=match.ticker
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
                    event_id=event_id,
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
        envelope = build_envelope(match, sport, point.prices, point.time)
        payload_id = await store_payload(session, envelope)
        shape = SnapshotShape(Source.POLYMARKET, TimeBasis.OBSERVED, payload_id)
        written = await store_shaped_snapshot(session, event, envelope, point.time, shape)
        if written.inserted:
            inserted += 1
        else:
            await discard_payload(session, payload_id)
    return inserted
