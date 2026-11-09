"""Close coverage by week: how each line the candidates bet on got its close."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from odds_core.close import CloseKind, CloseStatus, close_quotes, close_status
from odds_core.models import Event
from odds_core.paper_trade_models import PaperTrade
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

WEEKS = 8
LIVE_LABEL = "live Betfair"
PENDING_LABEL = "pending"
NO_CLOSE_LABEL = "no close"
UNSUPPORTED_LABEL = "unsupported"
_FALLBACK_NAMES: dict[Source, str] = {
    Source.FOOTBALL_DATA_UK: "FDUK",
    Source.POLYMARKET: "Polymarket",
}


def close_label(status: CloseStatus) -> str:
    """How a close is shown: a fallback is named by the source its ``FallbackSpec`` reads."""
    if status.kind is CloseKind.FALLBACK and status.source is not None:
        return f"{_FALLBACK_NAMES.get(status.source, status.source.value)} fallback"
    return {
        CloseKind.LIVE: LIVE_LABEL,
        CloseKind.PENDING: PENDING_LABEL,
        CloseKind.NONE: NO_CLOSE_LABEL,
        CloseKind.UNSUPPORTED: UNSUPPORTED_LABEL,
    }[status.kind]


def week_of(moment: datetime) -> date:
    """The Monday of the UTC week ``moment`` falls in."""
    day = moment.date()
    return day - timedelta(days=day.weekday())


@dataclass(frozen=True)
class CloseWeeks:
    """One sport and market: per week, the count of bet lines under each close label."""

    sport: str
    market: str
    weeks: list[date]
    counts: dict[date, Counter[str]]

    @property
    def labels(self) -> list[str]:
        """The labels present, live first and the ones without a close last."""
        order = [LIVE_LABEL, PENDING_LABEL, NO_CLOSE_LABEL, UNSUPPORTED_LABEL]
        present = {label for c in self.counts.values() for label in c}
        fallbacks = sorted(present - set(order))
        return [label for label in [LIVE_LABEL, *fallbacks, *order[1:]] if label in present]

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
    counts: dict[tuple[str, str], dict[date, Counter[str]]] = defaultdict(
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
                counts[(event.sport_key, market)][week][close_label(status)] += 1
    return [
        CloseWeeks(sport, market, weeks, by_week)
        for (sport, market), by_week in sorted(counts.items())
    ]
