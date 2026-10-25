"""Find, and delete, snapshots of one event taken at one time by one source (#535).

Two snapshots form a pair when they share the event, ``snapshot_time`` and ``raw_data``
source and quote at least one bookmaker, market, outcome and point in common. A pair is
identical when both quote the same keys at the same prices, and conflicts when a shared key
has two prices. A conflicting OddsPortal live pair on an MLB event is two games' prices
collapsed onto one event before events were keyed to the gamePk (#612), so it cannot be
attributed and both snapshots go. Other conflicts have no established cause and are reported
and left. Of identical snapshots one is kept.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from itertools import combinations

from odds_core.models import Event, Odds, OddsSnapshot
from odds_core.paper_trade_models import CandidateDecision
from odds_core.snapshot_sources import SnapshotSource
from odds_core.snapshot_utils import extract_odds_from_snapshot
from sqlalchemy import delete, func, select, text, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.mlb_game_resolver import MLB_SPORT_KEY

PriceKey = tuple[str | None, str | None, str | None, float | None]


class GroupAction(StrEnum):
    CONFLICTING = "conflicting"
    CONFLICTING_LEFT = "conflicting_left"
    DUPLICATE = "duplicate"
    REFUSED = "refused"


@dataclass(frozen=True)
class SnapshotPairs:
    identical: int = 0
    conflicting: int = 0


@dataclass(frozen=True)
class SameTimeGroup:
    """The pairs among one event's snapshots at one time. ``conflicting_ids`` are the
    conflicting snapshots the cleanup deletes, ``conflicting_left_ids`` those it leaves, and
    ``duplicate_ids`` all but the lowest id of each set of identical snapshots."""

    event_id: str
    sport_key: str
    snapshot_time: datetime
    sources: tuple[str | None, ...]
    snapshot_ids: tuple[int, ...]
    pairs: SnapshotPairs
    conflicting_ids: tuple[int, ...]
    conflicting_left_ids: tuple[int, ...]
    duplicate_ids: tuple[int, ...]
    referenced_ids: tuple[int, ...] = ()

    @property
    def action(self) -> GroupAction:
        if self.referenced_ids:
            return GroupAction.REFUSED
        if self.conflicting_ids:
            return GroupAction.CONFLICTING
        if self.conflicting_left_ids:
            return GroupAction.CONFLICTING_LEFT
        return GroupAction.DUPLICATE

    @property
    def doomed_ids(self) -> tuple[int, ...]:
        """The snapshots to delete unless a candidate decision used one of them."""
        return tuple(sorted({*self.conflicting_ids, *self.duplicate_ids}))

    @property
    def deleted_ids(self) -> tuple[int, ...]:
        return () if self.referenced_ids else self.doomed_ids


@dataclass(frozen=True)
class GroupResult:
    group: SameTimeGroup
    odds_deleted: int = 0
    odds_linked: int = 0


def snapshot_prices(snapshot: OddsSnapshot) -> dict[PriceKey, float]:
    """Prices by key, with the bookmaker key stripped as the writer stores it."""
    return {
        (str(o.bookmaker_key or "").strip(), o.market_key, o.outcome_name, o.point): o.price_decimal
        for o in extract_odds_from_snapshot(snapshot, snapshot.event_id)
    }


def _deletes_conflicts(sport_key: str, source: str | None) -> bool:
    return sport_key == MLB_SPORT_KEY and source == SnapshotSource.ODDSPORTAL_LIVE


def _source(snapshot: OddsSnapshot) -> str | None:
    raw = snapshot.raw_data
    return raw.get("source") if isinstance(raw, dict) else None


def _pair_snapshots(
    prices: dict[int, dict[PriceKey, float]],
) -> tuple[SnapshotPairs, set[int], set[int]]:
    """Pair counts, conflicting ids and identical duplicates among one source's snapshots."""
    identical = conflicting_pairs = 0
    conflicting: set[int] = set()
    for a, b in combinations(sorted(prices), 2):
        shared = prices[a].keys() & prices[b].keys()
        if not shared:
            continue
        if any(prices[a][k] != prices[b][k] for k in shared):
            conflicting_pairs += 1
            conflicting |= {a, b}
        elif prices[a] == prices[b]:
            identical += 1

    same_prices: dict[frozenset[tuple[PriceKey, float]], list[int]] = defaultdict(list)
    for snapshot_id in sorted(prices.keys() - conflicting):
        if prices[snapshot_id]:
            same_prices[frozenset(prices[snapshot_id].items())].append(snapshot_id)
    duplicates = {i for ids in same_prices.values() for i in ids[1:]}
    return SnapshotPairs(identical, conflicting_pairs), conflicting, duplicates


def group_snapshots(event: Event, snapshots: Sequence[OddsSnapshot]) -> list[SameTimeGroup]:
    """One group per time at which the event's snapshots form a pair."""
    by_time: dict[datetime, dict[str | None, dict[int, dict[PriceKey, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for snapshot in snapshots:
        assert snapshot.id is not None
        by_time[snapshot.snapshot_time][_source(snapshot)][snapshot.id] = snapshot_prices(snapshot)

    groups = []
    for at in sorted(by_time):
        identical = conflicting_pairs = 0
        paired: set[int] = set()
        sources: list[str | None] = []
        conflicting: set[int] = set()
        conflicting_left: set[int] = set()
        duplicates: set[int] = set()
        for source, prices in by_time[at].items():
            pairs, source_conflicting, source_duplicates = _pair_snapshots(prices)
            if not pairs.identical and not pairs.conflicting:
                continue
            identical += pairs.identical
            conflicting_pairs += pairs.conflicting
            sources.append(source)
            paired |= prices.keys()
            if _deletes_conflicts(event.sport_key, source):
                conflicting |= source_conflicting
            else:
                conflicting_left |= source_conflicting
            duplicates |= source_duplicates
        if not sources:
            continue
        groups.append(
            SameTimeGroup(
                event_id=event.id,
                sport_key=event.sport_key,
                snapshot_time=at,
                sources=tuple(sources),
                snapshot_ids=tuple(sorted(paired)),
                pairs=SnapshotPairs(identical, conflicting_pairs),
                conflicting_ids=tuple(sorted(conflicting)),
                conflicting_left_ids=tuple(sorted(conflicting_left)),
                duplicate_ids=tuple(sorted(duplicates)),
            )
        )
    return groups


async def _shared_time_snapshots(
    session: AsyncSession, event_ids: Sequence[str] | None = None
) -> dict[str, list[OddsSnapshot]]:
    shared = (
        select(OddsSnapshot.event_id, OddsSnapshot.snapshot_time)
        .group_by(OddsSnapshot.event_id, OddsSnapshot.snapshot_time)
        .having(func.count() > 1)
    )
    if event_ids is not None:
        shared = shared.where(OddsSnapshot.event_id.in_(event_ids))
    rows = (
        await session.execute(
            select(OddsSnapshot)
            .where(tuple_(OddsSnapshot.event_id, OddsSnapshot.snapshot_time).in_(shared))
            .order_by(OddsSnapshot.id)
        )
    ).scalars()
    by_event: dict[str, list[OddsSnapshot]] = defaultdict(list)
    for snapshot in rows:
        by_event[snapshot.event_id].append(snapshot)
    return by_event


async def event_same_time_pairs(session: AsyncSession, event_id: str) -> SnapshotPairs:
    """Count the event's same-time pairs, identical and conflicting. Read-only."""
    event = await session.get(Event, event_id)
    snapshots = (await _shared_time_snapshots(session, [event_id])).get(event_id)
    if event is None or not snapshots:
        return SnapshotPairs()
    groups = group_snapshots(event, snapshots)
    return SnapshotPairs(
        sum(g.pairs.identical for g in groups), sum(g.pairs.conflicting for g in groups)
    )


async def find_same_time_groups(session: AsyncSession) -> list[SameTimeGroup]:
    """Every event's same-time pairs. A group whose deletions include a snapshot a candidate
    decision recorded as an input is refused, so its replay keeps working."""
    referenced = set(
        (await session.execute(select(func.unnest(CandidateDecision.input_snapshot_ids))))
        .scalars()
        .all()
    )
    by_event = await _shared_time_snapshots(session)
    events = {
        e.id: e
        for e in (await session.execute(select(Event).where(Event.id.in_(by_event)))).scalars()
    }
    groups = []
    for event_id in sorted(by_event):
        for group in group_snapshots(events[event_id], by_event[event_id]):
            used = referenced.intersection(group.doomed_ids)
            groups.append(replace(group, referenced_ids=tuple(sorted(used))) if used else group)
    return groups


async def _resolve_odds(
    session: AsyncSession, event_id: str, at: datetime, deleted: Sequence[int]
) -> tuple[int, int]:
    """Delete the deleted snapshots' odds rows at one event and time, and link the unlinked
    rows a kept snapshot quotes to it.

    An unlinked row is one of several copies of a key, one per snapshot that quoted it
    (#535). A key only deleted snapshots quote loses its copies. Of a key one kept snapshot
    quotes, the copy at that snapshot's price stays and the rest go; prices compare within
    float rounding, because stored rows were converted to decimal apart from ``raw_data``.
    Any other unlinked row is left alone.
    """
    snapshots = (
        await session.execute(
            select(OddsSnapshot).where(
                OddsSnapshot.event_id == event_id, OddsSnapshot.snapshot_time == at
            )
        )
    ).scalars()
    kept: dict[PriceKey, list[tuple[int, float]]] = defaultdict(list)
    deleted_keys: set[PriceKey] = set()
    for snapshot in snapshots:
        assert snapshot.id is not None
        prices = snapshot_prices(snapshot)
        if snapshot.id in deleted:
            deleted_keys |= prices.keys()
        else:
            for key, price in prices.items():
                kept[key].append((snapshot.id, price))

    linked_deleted = (
        await session.execute(delete(Odds).where(Odds.snapshot_id.in_(deleted)))
    ).rowcount
    unlinked = (
        await session.execute(
            select(Odds)
            .where(
                Odds.event_id == event_id,
                Odds.odds_timestamp == at,
                Odds.snapshot_id.is_(None),
            )
            .order_by(Odds.id)
        )
    ).scalars()
    copies: dict[PriceKey, list[Odds]] = defaultdict(list)
    for row in unlinked:
        copies[(row.bookmaker_key, row.market_key, row.outcome_name, row.point)].append(row)

    drop: list[Odds] = []
    linked = 0
    for key, rows in copies.items():
        quoted_by = kept.get(key, [])
        if not quoted_by:
            if key in deleted_keys:
                drop.extend(rows)
            continue
        if len(quoted_by) > 1 or key not in deleted_keys:
            continue
        [(snapshot_id, price)] = quoted_by
        keep = next((r for r in rows if math.isclose(r.price_decimal, price)), None)
        if keep is None and all(math.isclose(r.price_decimal, rows[0].price_decimal) for r in rows):
            keep = rows[0]
        if keep is None:
            continue
        keep.snapshot_id = snapshot_id
        linked += 1
        drop.extend(r for r in rows if r is not keep)
    if drop:
        await session.execute(delete(Odds).where(Odds.id.in_([r.id for r in drop])))
    await session.flush()
    return linked_deleted + len(drop), linked


async def delete_same_time_pairs(
    session: AsyncSession, groups: Sequence[SameTimeGroup]
) -> list[GroupResult]:
    """Delete each group's deleted snapshots with their odds rows. The caller commits."""
    results = []
    for group in groups:
        if not group.deleted_ids:
            results.append(GroupResult(group))
            continue
        odds_deleted, odds_linked = await _resolve_odds(
            session, group.event_id, group.snapshot_time, group.deleted_ids
        )
        await session.execute(delete(OddsSnapshot).where(OddsSnapshot.id.in_(group.deleted_ids)))
        results.append(GroupResult(group, odds_deleted, odds_linked))
    return results


@dataclass(frozen=True)
class UnlinkedOdds:
    """Odds rows without a snapshot: ``orphans`` are those no snapshot at their event and time
    quotes, which this cleanup never resolves; the rest are quoted by several."""

    total: int
    orphans: int


_ORPHAN_ODDS = text(
    r"""
    SELECT count(*) FROM odds o
    WHERE o.snapshot_id IS NULL AND NOT EXISTS (
        SELECT 1
        FROM odds_snapshots s
        CROSS JOIN LATERAL json_array_elements(s.raw_data->'bookmakers') b
        CROSS JOIN LATERAL json_array_elements(b->'markets') m
        CROSS JOIN LATERAL json_array_elements(m->'outcomes') x
        WHERE s.event_id = o.event_id AND s.snapshot_time = o.odds_timestamp
          AND btrim(b->>'key', E' \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f') = o.bookmaker_key
          AND m->>'key' = o.market_key AND x->>'name' = o.outcome_name
          AND (x->>'point')::float IS NOT DISTINCT FROM o.point
    )
    """
)


async def unlinked_odds(session: AsyncSession) -> UnlinkedOdds:
    total = (
        await session.execute(select(func.count()).where(Odds.snapshot_id.is_(None)))
    ).scalar_one()
    orphans = (await session.execute(_ORPHAN_ODDS)).scalar_one() if total else 0
    return UnlinkedOdds(total, orphans)
