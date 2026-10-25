"""Check every snapshot's new-shape columns and price rows against its ``raw_data``, read with
the writers' reader (``odds_lambda.price_rows.expand_envelope``)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from odds_core.models import DataQualityLog, Event, Odds, OddsSnapshot, Price, SnapshotPayload
from odds_core.time import ensure_utc
from odds_core.vocabulary import Market, PriceOutcome, Side, Source
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.price_rows import ODDS_API_LAY_MARKET, PriceSpec, RowIssueType, expand_envelope
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
    # ``raw_data`` is not a JSON object, so no reader can expand it.
    MALFORMED = "malformed"
    # The snapshots sharing an archived envelope do not hold the prices the envelope gives.
    PAYLOAD_PRICES = "payload_prices"
    # A market of an archived envelope is on no snapshot, and no logged key conflict names a
    # kept snapshot holding its key.
    PAYLOAD_LOST = "payload_lost"
    # An odds row with no snapshot written after the live writers archived their first run.
    NEW_ORPHAN_ODDS = "new_orphan_odds"


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
    payloads_checked: int = 0
    # Run payloads (a list of envelopes) and payloads no snapshot or conflict references.
    payloads_unchecked: int = 0
    orphan_odds: int = 0
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

    if snapshot.raw_data is None:
        miss(Check.MALFORMED, "raw_data is not a JSON object")
        return found
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
        snapshot.envelope, snapshot.source, snapshot.home_team, snapshot.away_team
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
    _compare_prices(expected, stored, lambda detail: miss(Check.PRICES_DIFFER, detail))
    return found


def _compare_prices(
    expected: Mapping[PriceKey, PriceValue],
    stored: Mapping[PriceKey, PriceValue],
    report: Callable[[str], None],
) -> None:
    missing = expected.keys() - stored.keys()
    extra = stored.keys() - expected.keys()
    differing = [k for k in expected.keys() & stored.keys() if expected[k] != stored[k]]
    if missing or extra or differing:
        report(
            f"{len(expected)} expected, {len(stored)} stored: {len(missing)} missing, "
            f"{len(extra)} extra, {len(differing)} differing"
            + (
                f" (e.g. {differing[0]}: {expected[differing[0]]} vs {stored[differing[0]]})"
                if differing
                else ""
            )
        )


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


@dataclass(frozen=True)
class _LoggedConflict:
    market: Market
    kept_snapshot_id: int
    event_id: str
    source: Source
    snapshot_time: datetime


async def _logged_conflicts(session: AsyncSession) -> dict[int, list[_LoggedConflict]]:
    """The backfill's key conflicts per archived payload: each dropped market and the snapshot
    the log says holds its key."""
    rows = await session.execute(
        select(DataQualityLog.event_id, DataQualityLog.raw_data).where(
            DataQualityLog.issue_type == RowIssueType.SNAPSHOT_KEY_CONFLICT.value
        )
    )
    conflicts: dict[int, list[_LoggedConflict]] = defaultdict(list)
    for event_id, context in rows.all():
        if not isinstance(context, dict) or "payload_id" not in context:
            continue
        conflicts[int(context["payload_id"])].append(
            _LoggedConflict(
                Market(context["market"]),
                int(context["kept_snapshot_id"]),
                event_id,
                Source(context["source"]),
                datetime.fromisoformat(context["snapshot_time"]),
            )
        )
    return conflicts


async def _check_payloads(session: AsyncSession, report: VerifyReport, *, batch_size: int) -> None:
    """Every archived envelope (a payload holding one snapshot's ``raw_data``) read again: each
    of its markets' prices are on exactly one snapshot sharing the payload, or the market was
    dropped as a logged key conflict and the snapshot the log names holds its key."""
    conflicts = await _logged_conflicts(session)
    after_id = 0
    while True:
        payloads = (
            await session.execute(
                select(SnapshotPayload.id, SnapshotPayload.payload)
                .where(
                    SnapshotPayload.id > after_id,  # type: ignore[operator]
                    func.jsonb_typeof(SnapshotPayload.payload) == "object",
                )
                .order_by(SnapshotPayload.id)  # type: ignore[arg-type]
                .limit(batch_size)
            )
        ).all()
        if not payloads:
            break
        after_id = payloads[-1].id
        ids = [p.id for p in payloads]
        snapshots = (
            await session.execute(
                select(
                    OddsSnapshot.id,
                    OddsSnapshot.payload_id,
                    OddsSnapshot.event_id,
                    OddsSnapshot.source,
                    OddsSnapshot.market,
                    OddsSnapshot.snapshot_time,
                ).where(OddsSnapshot.payload_id.in_(ids))  # type: ignore[union-attr]
            )
        ).all()
        by_payload: dict[int, list[Any]] = defaultdict(list)
        for row in snapshots:
            by_payload[row.payload_id].append(row)
        kept_ids = {c.kept_snapshot_id for i in ids for c in conflicts.get(i, [])}
        kept = {
            row.id: row
            for row in (
                await session.execute(
                    select(
                        OddsSnapshot.id,
                        OddsSnapshot.event_id,
                        OddsSnapshot.source,
                        OddsSnapshot.market,
                        OddsSnapshot.snapshot_time,
                    ).where(OddsSnapshot.id.in_(list(kept_ids)))  # type: ignore[union-attr]
                )
            ).all()
        }
        event_ids = {r.event_id for r in snapshots} | {
            c.event_id for i in ids for c in conflicts.get(i, [])
        }
        teams = {
            row.id: (row.home_team, row.away_team)
            for row in (
                await session.execute(
                    select(Event.id, Event.home_team, Event.away_team).where(
                        Event.id.in_(list(event_ids))  # type: ignore[attr-defined]
                    )
                )
            ).all()
        }
        prices = await _stored_prices(session, [r.id for r in snapshots])
        for payload in payloads:
            group = by_payload.get(payload.id, [])
            logged = conflicts.get(payload.id, [])
            if not group and not logged:
                report.payloads_unchecked += 1
                continue
            report.payloads_checked += 1
            for mismatch in _check_payload(
                payload.id, payload.payload, group, logged, kept, teams, prices
            ):
                report.add(mismatch)
        await session.rollback()
    report.payloads_unchecked += (
        await session.execute(
            select(func.count()).where(func.jsonb_typeof(SnapshotPayload.payload) != "object")
        )
    ).scalar_one()
    await session.rollback()


def _check_payload(
    payload_id: int,
    envelope: dict[str, Any],
    group: Sequence[Any],
    logged: Sequence[_LoggedConflict],
    kept: Mapping[int, Any],
    teams: Mapping[str, tuple[str, str]],
    prices: Mapping[int, Mapping[PriceKey, PriceValue]],
) -> list[Mismatch]:
    found: list[Mismatch] = []
    first = group[0] if group else None
    event_id = first.event_id if first is not None else logged[0].event_id
    source = first.source if first is not None else logged[0].source

    def miss(check: Check, snapshot_id: int | None, detail: str) -> None:
        found.append(Mismatch(check, snapshot_id, event_id, f"payload {payload_id}: {detail}"))

    if source is None:
        return found
    if any(r.event_id != event_id or r.source != source for r in group):
        miss(Check.PAYLOAD_PRICES, None, "its snapshots name different events or sources")
        return found
    home, away = teams[event_id]
    expected = {m.market: m for m in expand_envelope(envelope, source, home, away).markets}
    on_snapshots: dict[Market, list[Any]] = defaultdict(list)
    for row in group:
        if row.market is not None:
            on_snapshots[row.market].append(row)
    dropped = {c.market: c for c in logged}
    for market, read in expected.items():
        holders = on_snapshots.get(market, [])
        if len(holders) == 1:
            _compare_prices(
                {spec.key: _value(spec) for spec in read.prices},
                prices.get(holders[0].id, {}),
                lambda detail, sid=holders[0].id: miss(Check.PAYLOAD_PRICES, sid, detail),
            )
        elif len(holders) > 1:
            miss(Check.PAYLOAD_PRICES, None, f"{market.value} is on {len(holders)} snapshots")
        elif market in dropped:
            conflict = dropped[market]
            holder = kept.get(conflict.kept_snapshot_id)
            if holder is None or (
                holder.event_id,
                holder.source,
                holder.market,
                ensure_utc(holder.snapshot_time),
            ) != (event_id, source, market, ensure_utc(conflict.snapshot_time)):
                miss(
                    Check.PAYLOAD_LOST,
                    conflict.kept_snapshot_id,
                    f"{market.value} was dropped for snapshot {conflict.kept_snapshot_id}, "
                    "which does not hold its key",
                )
        else:
            miss(Check.PAYLOAD_LOST, None, f"{market.value} is on no snapshot")
    for market, rows in on_snapshots.items():
        if market not in expected:
            miss(Check.PAYLOAD_PRICES, rows[0].id, f"{market.value} is not in the envelope")
    return found


async def _orphan_odds(session: AsyncSession, report: VerifyReport) -> None:
    """Count ``odds`` rows with no snapshot. Those written after the live writers archived their
    first run (the earliest payload holding a list of envelopes), which link every row, are
    mismatches."""
    report.orphan_odds = (
        await session.execute(
            select(func.count()).where(Odds.snapshot_id.is_(None))  # type: ignore[union-attr]
        )
    ).scalar_one()
    first_run = (
        await session.execute(
            select(func.min(SnapshotPayload.received_at)).where(
                func.jsonb_typeof(SnapshotPayload.payload) == "array"
            )
        )
    ).scalar_one()
    if first_run is not None:
        new = (
            await session.execute(
                select(func.count()).where(
                    Odds.snapshot_id.is_(None),  # type: ignore[union-attr]
                    Odds.created_at > first_run,  # type: ignore[operator]
                )
            )
        ).scalar_one()
        if new:
            report.mismatches[Check.NEW_ORPHAN_ODDS] += new
            report.examples.append(
                Mismatch(
                    Check.NEW_ORPHAN_ODDS,
                    None,
                    None,
                    f"{new} odds rows without a snapshot written after {first_run.isoformat()}",
                )
            )
    await session.rollback()


async def verify_shape(
    session: AsyncSession, *, cutoff: datetime, batch_size: int, max_examples: int = 20
) -> VerifyReport:
    """Check every snapshot stored before ``cutoff``: its source, time basis and payload are
    set and match its ``raw_data`` tags, its ``raw_data`` quotes one market (the snapshot's, or
    none for an empty envelope), its price rows equal those the reader reads from ``raw_data``
    (per book, outcome, line and side: price, size, matched and change time), and each of its
    ``odds`` rows is of its market. Then reads every archived envelope again
    (``_check_payloads``), counts ``odds`` rows with no snapshot (``_orphan_odds``) and checks
    that no legacy Polymarket price is left to migrate. Read-only."""
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
    await _check_payloads(session, report, batch_size=batch_size)
    await _orphan_odds(session, report)
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
