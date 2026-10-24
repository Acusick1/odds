"""Which stored prices count as known at a decision time t."""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta

from odds_core.snapshot_sources import SnapshotKind

MAX_EXCHANGE_AGE = timedelta(hours=2)

RETAIL_KINDS: frozenset[SnapshotKind] = frozenset(
    {
        SnapshotKind.ODDS_API,
        SnapshotKind.ODDSPORTAL,
        SnapshotKind.ODDSPORTAL_LIVE,
        SnapshotKind.FDUK_OPENING,
    }
)
# Stamped at FDUK's latest documented collection time, an upper bound on when the price was taken.
UPPER_BOUND_TIME_KINDS: frozenset[SnapshotKind] = frozenset({SnapshotKind.FDUK_OPENING})
EXCHANGE_AT_T_KINDS: frozenset[SnapshotKind] = frozenset(
    {SnapshotKind.BETFAIR_LIVE, SnapshotKind.FDUK_OPENING}
)


def latest_at_or_before(times: list[datetime], at: datetime, max_age: timedelta) -> int | None:
    """Index of the last of the sorted ``times`` that is <= ``at`` and at most ``max_age`` older."""
    idx = bisect.bisect_right(times, at) - 1
    if idx < 0 or at - times[idx] > max_age:
        return None
    return idx
