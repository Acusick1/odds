"""MLB ``gamePk``s held by events, read from their ``mlb_stats_api`` external ids."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from odds_core.models import Event, EventExternalId
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def game_pks_of(session: AsyncSession, events: Sequence[Event]) -> dict[int, int]:
    """The ``gamePk`` of each of ``events`` that holds one, by event id."""
    ids = [e.id for e in events if e.id is not None]
    if not ids:
        return {}
    rows = await session.execute(
        select(EventExternalId.event_id, EventExternalId.external_id).where(
            EventExternalId.source == Source.MLB_STATS_API,
            EventExternalId.event_id.in_(ids),  # type: ignore[attr-defined]
        )
    )
    return {event_id: int(external_id) for event_id, external_id in rows.all()}


async def events_holding_game_pks(session: AsyncSession, events: Sequence[Event]) -> set[int]:
    """The ids of ``events`` that hold a ``gamePk``."""
    return set(await game_pks_of(session, events))


async def holds_game_pk(session: AsyncSession, event: Event) -> bool:
    """Whether the event holds a ``gamePk``."""
    return bool(await game_pks_of(session, [event]))


async def events_holding_game_pks_by_pk(
    session: AsyncSession, game_pks: Collection[int]
) -> dict[int, Event]:
    """The event holding each of ``game_pks`` that one holds, by ``gamePk``."""
    if not game_pks:
        return {}
    rows = await session.execute(
        select(EventExternalId.external_id, Event)
        .join(Event, EventExternalId.event_id == Event.id)  # type: ignore[arg-type]
        .where(
            EventExternalId.source == Source.MLB_STATS_API,
            EventExternalId.external_id.in_([str(pk) for pk in game_pks]),  # type: ignore[attr-defined]
        )
    )
    return {int(external_id): event for external_id, event in rows.all()}


async def event_holding_game_pk(session: AsyncSession, game_pk: int) -> Event | None:
    """The event holding ``game_pk``, if any."""
    return (await events_holding_game_pks_by_pk(session, [game_pk])).get(game_pk)
