"""Which stored quotes count as known at a decision time t."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime

from odds_core.quotes import Quote


def by_snapshot(quoted: Iterable[Quote]) -> list[list[Quote]]:
    """Quotes grouped by snapshot, in snapshot time then id order."""
    groups: dict[int, list[Quote]] = {}
    for quote in quoted:
        groups.setdefault(quote.snapshot_id, []).append(quote)
    return sorted(groups.values(), key=lambda g: (g[0].snapshot_time, g[0].snapshot_id))


def latest_snapshot(
    quoted: Iterable[Quote],
    at: datetime,
    accept: Callable[[Sequence[Quote]], bool] = lambda _: True,
) -> list[Quote] | None:
    """The quotes of the latest snapshot stamped at or before ``at`` whose quotes ``accept``
    takes, as a whole; a tie in time goes to the higher snapshot id. None when there is none."""
    chosen: list[Quote] | None = None
    for group in by_snapshot(q for q in quoted if q.snapshot_time <= at):
        if accept(group):
            chosen = group
    return chosen
