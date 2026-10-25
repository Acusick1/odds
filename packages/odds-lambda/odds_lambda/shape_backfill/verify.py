"""Check every snapshot's new-shape columns and price rows against its ``raw_data``, read with
the writers' reader (``odds_lambda.price_rows.expand_envelope``)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from odds_core.models import Odds, Price
from odds_core.vocabulary import Market, PriceOutcome, Side, Source
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.price_rows import ODDS_API_LAY_MARKET, PriceSpec, expand_envelope
from odds_lambda.shape_backfill.legacy_polymarket import migrate_legacy_polymarket
from odds_lambda.shape_backfill.snapshots import (
    StoredSnapshot,
    UnknownSnapshotTag,
    classify,
    load_snapshots,
)

PriceKey = tuple[str, PriceOutcome, Decimal | None, Side]
PriceValue = tuple[Decimal, Decimal | None, Decimal | None, datetime | None]

# The snapshot markets an ``odds`` row's ``market_key`` may sit on.
_ODDS_KEY_MARKETS: Mapping[str, frozenset[Market]] = {
    **{m.value: frozenset({m}) for m in Market},
    ODDS_API_LAY_MARKET: frozenset({Market.ONE_X_TWO, Market.H2H}),
}


class Check(StrEnum):
    NO_SOURCE = "no_source"
    NO_TIME_BASIS = "no_time_basis"
    NO_PAYLOAD = "no_payload"
    # The source or time basis differs from what the ``raw_data`` tags say.
    WRONG_SOURCE = "wrong_source"
    # ``raw_data`` quotes more than one market.
    UNSPLIT = "unsplit"
    WRONG_MARKET = "wrong_market"
    # Betfair's in-play flag, status, total matched or start differs from ``raw_data``.
    MARKET_META = "market_meta"
    PRICES_DIFFER = "prices_differ"
    ODDS_MARKET = "odds_market"
    LEGACY_POLYMARKET_MISSING = "legacy_polymarket_missing"


@dataclass(frozen=True)
class Mismatch:
    check: Check
    snapshot_id: int | None
    event_id: str | None
    detail: str


@dataclass
class VerifyReport:
    snapshots: int = 0
    prices: int = 0
    odds_rows: int = 0
    empty: int = 0
    by_source: Counter[tuple[Source | None, Market | None]] = field(default_factory=Counter)
    mismatches: Counter[Check] = field(default_factory=Counter)
    examples: list[Mismatch] = field(default_factory=list)
    max_examples: int = 20

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def add(self, mismatch: Mismatch) -> None:
        self.mismatches[mismatch.check] += 1
        if sum(1 for m in self.examples if m.check is mismatch.check) < self.max_examples:
            self.examples.append(mismatch)


def _value(spec: PriceSpec) -> PriceValue:
    return (spec.price, spec.size, spec.matched, spec.changed_at)


def check_snapshot(
    snapshot: StoredSnapshot,
    stored: Mapping[PriceKey, PriceValue],
    odds_keys: Mapping[str, int],
) -> list[Mismatch]:
    """What differs between a snapshot's columns, price rows and ``odds`` rows and what its
    ``raw_data`` says."""
    found: list[Mismatch] = []

    def miss(check: Check, detail: str) -> None:
        found.append(Mismatch(check, snapshot.id, snapshot.event_id, detail))

    if snapshot.source is None:
        miss(Check.NO_SOURCE, "no source")
        return found
    if snapshot.time_basis is None:
        miss(Check.NO_TIME_BASIS, "no time basis")
    if snapshot.payload_id is None:
        miss(Check.NO_PAYLOAD, "no payload")
    try:
        expected_kind = classify(snapshot)
        if expected_kind != (snapshot.source, snapshot.time_basis):
            miss(
                Check.WRONG_SOURCE,
                f"stored {snapshot.source}/{snapshot.time_basis}, tags say "
                f"{expected_kind[0]}/{expected_kind[1]}",
            )
    except UnknownSnapshotTag as exc:
        miss(Check.WRONG_SOURCE, str(exc))

    expanded = expand_envelope(
        snapshot.raw_data, snapshot.source, snapshot.home_team, snapshot.away_team
    ).markets
    for key, count in odds_keys.items():
        allowed = _ODDS_KEY_MARKETS.get(key, frozenset())
        if snapshot.market not in allowed:
            miss(
                Check.ODDS_MARKET,
                f"{count} odds rows of market {key!r} on a {snapshot.market} snapshot",
            )
    if len(expanded) > 1:
        miss(Check.UNSPLIT, f"raw_data quotes {[m.market.value for m in expanded]}")
        return found
    market = expanded[0] if expanded else None
    if (market.market if market else None) != snapshot.market:
        miss(
            Check.WRONG_MARKET,
            f"stored {snapshot.market}, raw_data quotes {market.market if market else 'nothing'}",
        )
        return found
    if market is not None and market.meta != snapshot.meta:
        miss(Check.MARKET_META, f"stored {snapshot.meta}, raw_data says {market.meta}")
    expected = {spec.key: _value(spec) for spec in (market.prices if market else ())}
    missing = expected.keys() - stored.keys()
    extra = stored.keys() - expected.keys()
    differing = [k for k in expected.keys() & stored.keys() if expected[k] != stored[k]]
    if missing or extra or differing:
        miss(
            Check.PRICES_DIFFER,
            f"{len(expected)} expected, {len(stored)} stored: {len(missing)} missing, "
            f"{len(extra)} extra, {len(differing)} differing"
            + (
                f" (e.g. {differing[0]}: {expected[differing[0]]} vs {stored[differing[0]]})"
                if differing
                else ""
            ),
        )
    return found


async def _stored_prices(
    session: AsyncSession, snapshot_ids: Sequence[int]
) -> dict[int, dict[PriceKey, PriceValue]]:
    rows = await session.execute(
        select(
            Price.snapshot_id,
            Price.bookmaker,
            Price.outcome,
            Price.line,
            Price.side,
            Price.price,
            Price.size,
            Price.matched,
            Price.changed_at,
        ).where(Price.snapshot_id.in_(snapshot_ids))  # type: ignore[union-attr]
    )
    prices: dict[int, dict[PriceKey, PriceValue]] = defaultdict(dict)
    for row in rows:
        prices[row.snapshot_id][(row.bookmaker, row.outcome, row.line, row.side)] = (
            row.price,
            row.size,
            row.matched,
            row.changed_at,
        )
    return prices


async def _odds_keys(
    session: AsyncSession, snapshot_ids: Sequence[int]
) -> dict[int, dict[str, int]]:
    rows = await session.execute(
        select(Odds.snapshot_id, Odds.market_key, func.count())
        .where(Odds.snapshot_id.in_(snapshot_ids))  # type: ignore[union-attr]
        .group_by(Odds.snapshot_id, Odds.market_key)
    )
    keys: dict[int, dict[str, int]] = defaultdict(dict)
    for snapshot_id, market_key, count in rows:
        keys[snapshot_id][market_key] = count
    return keys


async def verify_shape(
    session: AsyncSession, *, cutoff: datetime, batch_size: int, max_examples: int = 20
) -> VerifyReport:
    """Check every snapshot stored before ``cutoff``: its source, time basis and payload are
    set and match its ``raw_data`` tags, its ``raw_data`` quotes one market (the snapshot's, or
    none for an empty envelope), its price rows equal those the reader reads from ``raw_data``
    (per book, outcome, line and side: price, size, matched and change time), and each of its
    ``odds`` rows is of its market. Also checks that no legacy Polymarket price is left to
    migrate. Read-only."""
    report = VerifyReport(max_examples=max_examples)
    after_id = 0
    while True:
        batch = await load_snapshots(
            session,
            cutoff=cutoff,
            after_id=after_id,
            size=batch_size,
            pending_only=False,
            lock=False,
        )
        if not batch:
            break
        ids = [s.id for s in batch]
        prices = await _stored_prices(session, ids)
        odds = await _odds_keys(session, ids)
        for snapshot in batch:
            stored = prices.get(snapshot.id, {})
            keys = odds.get(snapshot.id, {})
            report.snapshots += 1
            report.prices += len(stored)
            report.odds_rows += sum(keys.values())
            report.by_source[(snapshot.source, snapshot.market)] += 1
            if snapshot.source is not None and snapshot.market is None:
                report.empty += 1
            for mismatch in check_snapshot(snapshot, stored, keys):
                report.add(mismatch)
        after_id = batch[-1].id
        await session.rollback()
    legacy = await migrate_legacy_polymarket(session, apply=False)
    if legacy.snapshots:
        report.mismatches[Check.LEGACY_POLYMARKET_MISSING] += legacy.snapshots
        report.examples.append(
            Mismatch(
                Check.LEGACY_POLYMARKET_MISSING,
                None,
                None,
                f"{legacy.snapshots} legacy Polymarket price snapshots have no snapshot",
            )
        )
    return report
