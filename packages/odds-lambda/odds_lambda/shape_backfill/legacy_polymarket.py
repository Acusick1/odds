"""Move the legacy ``polymarket_*`` NBA moneyline prices into snapshots and price rows.

Each price snapshot of a moneyline market on a matched game becomes one ``polymarket`` ``h2h``
snapshot (``observed``) whose ``raw_data`` is a Polymarket envelope in the shape the live job
writes, so the same reader (``odds_lambda.price_rows.expand_envelope``) gives its ``mid`` rows and
the verify step re-reads it. The envelope is archived as the snapshot's payload. The legacy rows
are left in place.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from odds_core.models import Event, OddsSnapshot, Price, SnapshotPayload
from odds_core.odds_math import PRICE_DECIMAL_KEY
from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketMarketType,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)
from odds_core.team import normalize_team_name
from odds_core.time import ensure_utc
from odds_core.vocabulary import Market, Source, TimeBasis
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.polymarket_matching import normalize_team
from odds_lambda.polymarket_odds import BOOKMAKER_KEY, BOOKMAKER_TITLE, SNAPSHOT_SOURCE
from odds_lambda.price_rows import expand_envelope
from odds_lambda.shape_backfill.snapshots import price_values

# The payload key of an archived legacy order book, which also marks it as archived.
ORDER_BOOK_PAYLOAD_KEY = "polymarket_order_book"


@dataclass
class LegacyPolymarketReport:
    pm_events: int = 0
    unmatched_games: int = 0
    markets_skipped: Counter[str] = field(default_factory=Counter)
    rows_skipped: Counter[str] = field(default_factory=Counter)
    unmapped_markets: list[str] = field(default_factory=list)
    invalid_rows: int = 0
    duplicate_rows: int = 0
    existing: int = 0
    snapshots: int = 0
    prices: int = 0
    order_books: int = 0
    order_books_existing: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _PlannedSnapshot:
    snapshot_time: datetime
    envelope: dict[str, Any]
    fetch_tier: str | None
    hours_until_commence: float | None


def _outcome_sides(market: PolymarketMarket, event: Event) -> tuple[str, str] | None:
    """``("home", "away")`` in the order of the market's two outcomes, ``None`` when they do not
    name the event's teams."""
    names = [normalize_team(o) for o in market.outcomes]
    if len(names) != 2 or any(n is None for n in names):
        return None
    teams = [normalize_team_name(n) for n in names if n is not None]
    if teams == [event.home_team, event.away_team]:
        return "home", "away"
    if teams == [event.away_team, event.home_team]:
        return "away", "home"
    return None


def legacy_envelope(
    pm_event: PolymarketEvent,
    market: PolymarketMarket,
    row: PolymarketPriceSnapshot,
    sides: tuple[str, str],
    event: Event,
) -> dict[str, Any] | None:
    """The live job's envelope for one legacy price snapshot, leaving out a probability not
    strictly between 0 and 1 (a settled market); ``None`` when neither is."""
    probabilities = {
        side: p
        for side, p in zip(sides, (row.outcome_0_price, row.outcome_1_price), strict=True)
        if p is not None and 0.0 < p < 1.0
    }
    if not probabilities:
        return None
    names = {"home": event.home_team, "away": event.away_team}
    snapshot_time = ensure_utc(row.snapshot_time)
    return {
        "bookmakers": [
            {
                "key": BOOKMAKER_KEY,
                "title": BOOKMAKER_TITLE,
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": Market.H2H.value,
                        "outcomes": [
                            {"name": names[side], PRICE_DECIMAL_KEY: 1 / p}
                            for side, p in probabilities.items()
                        ],
                    }
                ],
                "polymarket_meta": {
                    "pm_event_id": pm_event.pm_event_id,
                    "ticker": pm_event.ticker,
                    "pm_market_id": market.pm_market_id,
                    "legacy_price_snapshot_id": row.id,
                    "yes_probabilities": probabilities,
                    "probability_sum": sum(probabilities.values()),
                    "volume": row.volume,
                    "liquidity": row.liquidity,
                },
            }
        ],
        "source": SNAPSHOT_SOURCE,
    }


async def _plan_event(
    session: AsyncSession,
    pm_event: PolymarketEvent,
    event: Event,
    taken: set[datetime],
    report: LegacyPolymarketReport,
) -> list[_PlannedSnapshot]:
    """The snapshots to write for one legacy game. ``taken`` holds the times this run already
    planned for the event from another game linked to it, which count as duplicates."""
    markets = list(
        (
            await session.execute(
                select(PolymarketMarket)
                .where(PolymarketMarket.polymarket_event_id == pm_event.id)
                .order_by(PolymarketMarket.id)  # type: ignore[arg-type]
            )
        ).scalars()
    )
    stored = {
        ensure_utc(t)
        for t in (
            await session.execute(
                select(OddsSnapshot.snapshot_time).where(
                    OddsSnapshot.event_id == event.id,
                    OddsSnapshot.source == Source.POLYMARKET,
                    OddsSnapshot.market == Market.H2H,
                )
            )
        ).scalars()
    }
    planned: dict[datetime, _PlannedSnapshot] = {}
    for market in markets:
        rows = list(
            (
                await session.execute(
                    select(PolymarketPriceSnapshot)
                    .where(PolymarketPriceSnapshot.polymarket_market_id == market.id)
                    .order_by(PolymarketPriceSnapshot.snapshot_time)  # type: ignore[arg-type]
                )
            ).scalars()
        )
        if market.market_type is not PolymarketMarketType.MONEYLINE:
            if rows:
                report.markets_skipped[market.market_type.value] += 1
                report.rows_skipped[market.market_type.value] += len(rows)
            continue
        sides = _outcome_sides(market, event)
        if sides is None:
            report.unmapped_markets.append(market.pm_market_id)
            report.rows_skipped["unmapped_moneyline"] += len(rows)
            continue
        for row in rows:
            snapshot_time = ensure_utc(row.snapshot_time)
            if snapshot_time in taken or snapshot_time in planned:
                report.duplicate_rows += 1
                continue
            if snapshot_time in stored:
                report.existing += 1
                continue
            envelope = legacy_envelope(pm_event, market, row, sides, event)
            if envelope is None:
                report.invalid_rows += 1
                continue
            planned[snapshot_time] = _PlannedSnapshot(
                snapshot_time, envelope, row.fetch_tier, row.hours_until_commence
            )
    return list(planned.values())


async def migrate_legacy_polymarket(
    session: AsyncSession, *, apply: bool
) -> LegacyPolymarketReport:
    """Write a snapshot, payload and ``mid`` price rows for every legacy moneyline price snapshot
    of a game linked to an event, one committed transaction per game; archive every legacy
    order book as a payload. Other market types, games with no event, outcomes that do not name
    the event's teams and probabilities of 0 or 1 are counted and skipped. A snapshot already
    stored at the same time is skipped, so a second run writes nothing. Without ``apply`` it
    only counts."""
    report = LegacyPolymarketReport()
    pm_events = list(
        (
            await session.execute(select(PolymarketEvent).order_by(PolymarketEvent.id))  # type: ignore[arg-type]
        ).scalars()
    )
    report.pm_events = len(pm_events)
    taken: dict[str, set[datetime]] = defaultdict(set)
    for pm_event in pm_events:
        event = (
            await session.get(Event, pm_event.event_id) if pm_event.event_id is not None else None
        )
        if event is None:
            report.unmatched_games += 1
            continue
        planned = await _plan_event(session, pm_event, event, taken[event.id], report)
        taken[event.id] |= {s.snapshot_time for s in planned}
        prices = 0
        rows: list[tuple[_PlannedSnapshot, list[dict[str, Any]]]] = []
        for snapshot in planned:
            expanded = expand_envelope(
                snapshot.envelope, Source.POLYMARKET, event.home_team, event.away_team
            )
            specs = [spec for market in expanded.markets for spec in market.prices]
            rows.append((snapshot, [price_values(spec) for spec in specs]))
            prices += len(specs)
        if apply and rows:
            try:
                await _write_event(session, event, rows)
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                report.errors.append(f"{pm_event.ticker}: {exc.orig!r}")
                continue
        report.snapshots += len(rows)
        report.prices += prices
    await _archive_order_books(session, report, apply=apply)
    if not apply:
        await session.rollback()
    return report


async def _write_event(
    session: AsyncSession,
    event: Event,
    rows: list[tuple[_PlannedSnapshot, list[dict[str, Any]]]],
) -> None:
    payload_ids = (
        await session.scalars(
            insert(SnapshotPayload).returning(SnapshotPayload.id, sort_by_parameter_order=True),  # type: ignore[arg-type]
            [{"payload": s.envelope, "received_at": s.snapshot_time} for s, _ in rows],
        )
    ).all()
    snapshot_ids = (
        await session.scalars(
            insert(OddsSnapshot).returning(OddsSnapshot.id, sort_by_parameter_order=True),  # type: ignore[arg-type]
            [
                {
                    "event_id": event.id,
                    "snapshot_time": s.snapshot_time,
                    "raw_data": s.envelope,
                    "bookmaker_count": 1,
                    "fetch_tier": s.fetch_tier,
                    "hours_until_commence": s.hours_until_commence,
                    # The legacy poller stored each row when it fetched it.
                    "created_at": s.snapshot_time,
                    "source": Source.POLYMARKET,
                    "market": Market.H2H,
                    "time_basis": TimeBasis.OBSERVED,
                    "payload_id": payload_id,
                }
                for (s, _), payload_id in zip(rows, payload_ids, strict=True)
            ],
        )
    ).all()
    price_rows = [
        {**price, "snapshot_id": snapshot_id}
        for (_, prices), snapshot_id in zip(rows, snapshot_ids, strict=True)
        for price in prices
    ]
    if price_rows:
        await session.execute(insert(Price), price_rows)


async def _archive_order_books(
    session: AsyncSession, report: LegacyPolymarketReport, *, apply: bool
) -> None:
    archived = {
        int(i)
        for i in (
            await session.execute(
                select(
                    SnapshotPayload.payload[ORDER_BOOK_PAYLOAD_KEY]["id"].as_integer()  # type: ignore[index]
                ).where(SnapshotPayload.payload.has_key(ORDER_BOOK_PAYLOAD_KEY))  # type: ignore[union-attr]
            )
        ).scalars()
    }
    books = list(
        (
            await session.execute(
                select(PolymarketOrderBookSnapshot, PolymarketMarket.pm_market_id)
                .join(
                    PolymarketMarket,
                    PolymarketMarket.id == PolymarketOrderBookSnapshot.polymarket_market_id,  # type: ignore[arg-type]
                )
                .order_by(PolymarketOrderBookSnapshot.id)  # type: ignore[arg-type]
            )
        ).all()
    )
    pending = [(book, pm_market_id) for book, pm_market_id in books if book.id not in archived]
    report.order_books_existing += len(books) - len(pending)
    report.order_books += len(pending)
    if not apply or not pending:
        return
    await session.execute(
        insert(SnapshotPayload),
        [
            {
                "payload": {
                    ORDER_BOOK_PAYLOAD_KEY: {
                        "id": book.id,
                        "pm_market_id": pm_market_id,
                        "token_id": book.token_id,
                        "snapshot_time": ensure_utc(book.snapshot_time).isoformat(),
                        "raw_book": book.raw_book,
                    }
                },
                "received_at": book.snapshot_time,
            }
            for book, pm_market_id in pending
        ],
    )
    await session.commit()
