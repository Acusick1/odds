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
        "soccer_epl": (Source.ODDSPORTAL, Source.ODDS_API),
        "basketball_nba": (Source.ODDSPORTAL, Source.ODDS_API),
    }
)


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
    session: AsyncSession, event: Event, source: Source, start: datetime | None
) -> bool:
    """Move ``event`` to ``start`` as reported by ``source`` when that source is the event's
    schedule authority, re-timing its stored snapshots; a postponed event becomes scheduled.

    Returns whether the event moved: never for a source outside its sport's authority order, one
    outranked by a source holding an id on the event, a ``None`` start or an unchanged one.
    """
    order = KICKOFF_AUTHORITY.get(event.sport_key, ())  # type: ignore[call-overload]
    if source not in order or start is None:
        return False
    if ensure_utc(event.commence_time) == ensure_utc(start):
        return False
    if await _outranked(session, event, order[: order.index(source)]):
        return False
    event_id = event.id
    moved_from = ensure_utc(event.commence_time)
    event.commence_time = start
    event.updated_at = datetime.now(UTC)
    if event.status == EventStatus.POSTPONED:
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
