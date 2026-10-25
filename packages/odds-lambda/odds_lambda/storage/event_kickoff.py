"""The one way an event's ``commence_time`` moves: on the word of its sport's schedule authority."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType

import structlog
from odds_core.models import Event, EventExternalId, EventStatus
from odds_core.sports import SportKey
from odds_core.time import ensure_utc
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.storage.event_merge import retime_snapshots

logger = structlog.get_logger()

# Per sport, the sources whose kick-off times an event follows, best first. A source moves an
# event only when no source ranked above it holds an id on the event. An empty order means no
# source moves that sport's events.
KICKOFF_AUTHORITY: Mapping[SportKey, tuple[Source, ...]] = MappingProxyType(
    {
        "baseball_mlb": (Source.MLB_STATS_API,),
        "soccer_epl": (Source.ODDSPORTAL, Source.ODDS_API, Source.FOOTBALL_DATA_UK),
        "basketball_nba": (Source.ODDSPORTAL, Source.ODDS_API),
    }
)

# Sources whose schedule dates a game by when it was played, so they may move an event to a start
# in the past. Every other source is held to the after-kick-off guard in move_kickoff.
PLAYED_START_SOURCES: frozenset[Source] = frozenset({Source.MLB_STATS_API, Source.FOOTBALL_DATA_UK})


async def _outranked(session: AsyncSession, event: Event, above: tuple[Source, ...]) -> bool:
    if not above:
        return False
    held = await session.execute(
        select(EventExternalId.source).where(
            EventExternalId.event_id == event.surrogate_id,
            EventExternalId.source.in_(above),  # type: ignore[attr-defined]
        )
    )
    return held.first() is not None


async def move_kickoff(
    session: AsyncSession, event: Event, source: Source, start: datetime | None, *, now: datetime
) -> bool:
    """Move ``event`` to ``start`` as reported by ``source`` when that source is the event's
    schedule authority, re-timing its stored snapshots. A postponed event becomes scheduled, and
    so does a live one moved to a start after ``now``.

    Returns whether the event moved: never for a source outside its sport's authority order, a
    ``None`` start or an unchanged one. Before the stored kick-off (``now`` earlier) a source
    outranked by one holding an id on the event does not move it. Once the stored kick-off has
    passed, a start at or before ``now`` is refused (an in-play "actual start"), and a start still
    ahead is accepted from any source in the order, so a postponement the top source stopped
    listing is still followed. A source in ``PLAYED_START_SOURCES`` skips the
    after-kick-off rule: it dates a postponed game by when it was played, which is in the past.
    """
    order = KICKOFF_AUTHORITY.get(event.sport_key, ())  # type: ignore[call-overload]
    if source not in order or start is None:
        return False
    if ensure_utc(event.commence_time) == ensure_utc(start):
        return False
    if source not in PLAYED_START_SOURCES and ensure_utc(now) >= ensure_utc(event.commence_time):
        if ensure_utc(start) <= ensure_utc(now):
            return False
    elif await _outranked(session, event, order[: order.index(source)]):
        return False
    event_id = event.id
    moved_from = ensure_utc(event.commence_time)
    event.commence_time = start
    event.updated_at = datetime.now(UTC)
    if event.status == EventStatus.POSTPONED or (
        event.status == EventStatus.LIVE and ensure_utc(start) > ensure_utc(now)
    ):
        event.status = EventStatus.SCHEDULED
    retimed = await retime_snapshots(session, event_id, start)
    logger.info(
        "event_rescheduled",
        event_id=event_id,
        source=source.value,
        commence_time=moved_from.isoformat(),
        scheduled_start=ensure_utc(start).isoformat(),
        snapshots_retimed=retimed,
    )
    return True
