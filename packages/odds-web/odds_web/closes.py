"""Close coverage by week: how each line the candidates bet on got its close."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import NamedTuple

from odds_core.close import CloseKind, CloseStatus, close_quotes, close_status
from odds_core.models import Event
from odds_core.paper_trade_models import PaperTrade
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

WEEKS = 8
_FALLBACK_NAMES: dict[Source, str] = {
    Source.FOOTBALL_DATA_UK: "FDUK",
    Source.POLYMARKET: "Polymarket",
}
_KIND_LABELS: dict[CloseKind, str] = {
    CloseKind.LIVE: "live Betfair",
    CloseKind.PENDING: "pending",
    CloseKind.NONE: "no close",
    CloseKind.UNSUPPORTED: "unsupported",
}
_KIND_ORDER: tuple[CloseKind, ...] = tuple(CloseKind)


class CloseClass(NamedTuple):
    """How a line got its close: the kind, and for a fallback the source that supplied it."""

    kind: CloseKind
    source: Source | None = None

    @classmethod
    def of(cls, status: CloseStatus) -> CloseClass:
        return cls(status.kind, status.source if status.kind is CloseKind.FALLBACK else None)

    @property
    def label(self) -> str:
        """A fallback is named by the source its ``FallbackSpec`` reads."""
        if self.kind is CloseKind.FALLBACK and self.source is not None:
            return f"{_FALLBACK_NAMES.get(self.source, self.source.value)} fallback"
        return _KIND_LABELS[self.kind]

    @property
    def rank(self) -> tuple[int, str]:
        return _KIND_ORDER.index(self.kind), self.source.value if self.source else ""


def week_of(moment: datetime) -> date:
    """The Monday of the UTC week ``moment`` falls in."""
    day = moment.date()
    return day - timedelta(days=day.weekday())


@dataclass(frozen=True)
class CloseWeeks:
    """One sport and market: per week, the count of bet lines in each close class."""

    sport: str
    market: str
    weeks: list[date]
    counts: dict[date, Counter[CloseClass]]

    @property
    def classes(self) -> list[CloseClass]:
        """The classes present, live first and the ones without a close last."""
        present = {cls for c in self.counts.values() for cls in c}
        return sorted(present, key=lambda cls: cls.rank)

    @property
    def most(self) -> int:
        """The largest weekly total, which the bars are scaled to."""
        return max((c.total() for c in self.counts.values()), default=0)


async def close_coverage(session: AsyncSession, now: datetime) -> list[CloseWeeks]:
    """The close of every (event, market, line) a paper trade is on, for events kicking off in
    the ``WEEKS`` weeks to ``now``'s, by kick-off week (``odds_core.close.close_status``)."""
    weeks = [week_of(now) - timedelta(weeks=n) for n in reversed(range(WEEKS))]
    start = datetime.combine(weeks[0], datetime.min.time(), tzinfo=now.tzinfo)
    end = start + timedelta(weeks=WEEKS)
    keys = (
        await session.execute(
            select(col(PaperTrade.event_id), col(PaperTrade.market), col(PaperTrade.line))
            .join(Event, col(Event.id) == col(PaperTrade.event_id))
            .where(col(Event.commence_time) >= start, col(Event.commence_time) < end)
            .distinct()
        )
    ).all()
    events = {
        e.id: e
        for e in await session.scalars(
            select(Event).where(col(Event.id).in_({event_id for event_id, _, _ in keys}))
        )
    }
    lines: dict[str, dict[int, set[float | None]]] = defaultdict(lambda: defaultdict(set))
    for event_id, market, line in keys:
        lines[market][event_id].add(None if line is None else float(line))
    counts: dict[tuple[str, str], dict[date, Counter[CloseClass]]] = defaultdict(
        lambda: {week: Counter() for week in weeks}
    )
    for market, by_event in lines.items():
        quoted = await close_quotes(session, [events[i] for i in by_event], market)
        for event_id, event_lines in by_event.items():
            event = events[event_id]
            week = week_of(event.commence_time)
            for line in event_lines:
                status = close_status(
                    quoted[event_id], event.commence_time, event.sport_key, market, now, line
                )
                counts[(event.sport_key, market)][week][CloseClass.of(status)] += 1
    return [
        CloseWeeks(sport, market, weeks, by_week)
        for (sport, market), by_week in sorted(counts.items())
    ]
