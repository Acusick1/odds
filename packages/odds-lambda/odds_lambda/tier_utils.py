"""Utilities for calculating fetch tiers based on game timing.

This module provides reusable functions for determining fetch tiers (opening, early,
sharp, pregame, closing) based on how much time remains until a game starts. These
functions are used by both the live data collection pipeline and historical backfill
operations to ensure consistent tier classification across all data.
"""

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime

from odds_core.vocabulary import Source, TimeBasis

from odds_lambda.fetch_tier import FetchTier

_NEARER_TO_FARTHER = (
    FetchTier.CLOSING,
    FetchTier.PREGAME,
    FetchTier.SHARP,
    FetchTier.EARLY,
    FetchTier.OPENING,
)


def calculate_tier(hours_until: float) -> FetchTier:
    """The tier of a time ``hours_until`` hours before kick-off; negative is in play.

    This is the one hours-to-tier rule. A time exactly on a boundary
    (``FetchTier.max_hours``: 3, 12, 24 or 72 h) belongs to the nearer tier. What a
    snapshot's time covers also depends on its time basis (``snapshot_tier``).

    Example:
        >>> calculate_tier(3.0)
        <FetchTier.CLOSING: 'closing'>
        >>> calculate_tier(48.0)
        <FetchTier.EARLY: 'early'>
    """
    if hours_until < 0:
        return FetchTier.IN_PLAY
    return next(tier for tier in _NEARER_TO_FARTHER if hours_until <= tier.max_hours)


def snapshot_tier(
    snapshot_time: datetime, commence_time: datetime, time_basis: TimeBasis | None
) -> FetchTier:
    """The tier a snapshot covers, by what its time means (``time_basis``).

    A ``close`` snapshot is the closing price, whatever its stamp. An ``upper_bound`` time
    says the price was taken at or before it, so one exactly on a boundary covers the farther
    tier only (an FDUK opening stamped 3.0 h out is not a closing price, and the Odds API
    backfill's fetches at 72, 24, 12 and 3 h cover OPENING, EARLY, SHARP and PREGAME). Every
    other time is bucketed by ``calculate_tier``.
    """
    if time_basis == TimeBasis.CLOSE:
        return FetchTier.CLOSING
    hours_until = (commence_time - snapshot_time).total_seconds() / 3600
    tier = calculate_tier(hours_until)
    if time_basis == TimeBasis.UPPER_BOUND and tier in _NEARER_TO_FARTHER[:-1]:
        if hours_until == tier.max_hours:
            return _NEARER_TO_FARTHER[_NEARER_TO_FARTHER.index(tier) + 1]
    return tier


def snapshot_tiers(
    snapshots: Sequence[tuple[datetime, Source | None, TimeBasis | None, str | None]],
    commence_time: datetime,
) -> list[FetchTier]:
    """The tier of each of an event's snapshots, given as (snapshot time, source, time basis,
    market).

    Each takes ``snapshot_tier``, with two rules that need the event's other snapshots:

    - A market's latest OddsPortal history (``source_stamped``) time is its closing price,
      stamped with the latest change among the books, so it is CLOSING however near or after
      kick-off the stamp lies. A market with a single stamp before kick-off may hold only an
      opening whose closing was never scraped, so that stamp keeps ``snapshot_tier``; an
      opening price is never stamped after kick-off.
    - When a source has a ``close`` snapshot for a market, that is the source's closing price,
      and none of the source's other snapshots of the market is CLOSING: one that would be
      covers the farther tier (an FDUK opening stamped under 3 h out is PREGAME).
    """
    stamps: dict[str | None, list[datetime]] = defaultdict(list)
    closes: set[tuple[Source | None, str | None]] = set()
    for snapshot_time, source, time_basis, market in snapshots:
        if source == Source.ODDSPORTAL and time_basis == TimeBasis.SOURCE_STAMPED:
            stamps[market].append(snapshot_time)
        if time_basis == TimeBasis.CLOSE:
            closes.add((source, market))
    source_closes = {
        market: max(times)
        for market, times in stamps.items()
        if len(set(times)) > 1 or max(times) >= commence_time
    }

    def tier(
        snapshot_time: datetime,
        source: Source | None,
        time_basis: TimeBasis | None,
        market: str | None,
    ) -> FetchTier:
        if (
            source == Source.ODDSPORTAL
            and time_basis == TimeBasis.SOURCE_STAMPED
            and snapshot_time == source_closes.get(market)
        ):
            return FetchTier.CLOSING
        covered = snapshot_tier(snapshot_time, commence_time, time_basis)
        if (
            covered is FetchTier.CLOSING
            and time_basis != TimeBasis.CLOSE
            and (source, market) in closes
        ):
            return FetchTier.PREGAME
        return covered

    return [tier(*snapshot) for snapshot in snapshots]
