"""MLB ``gamePk``s held by events, read from their ``mlb_stats_api`` external ids."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from odds_core.models import Event, EventExternalId
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def game_pks_of(session: AsyncSession, events: Sequence[Event]) -> dict[str, int]:
    """The ``gamePk`` of each of ``events`` that holds one, by event id."""
    by_key = {e.surrogate_id: e.id for e in events if e.surrogate_id is not None}
    if not by_key:
        return {}
    rows = await session.execute(
        select(EventExternalId.event_id, EventExternalId.external_id).where(
            EventExternalId.source == Source.MLB_STATS_API,
            EventExternalId.event_id.in_(list(by_key)),  # type: ignore[attr-defined]
        )
    )
    return {by_key[key]: int(external_id) for key, external_id in rows.all()}


async def events_holding_game_pks(session: AsyncSession, events: Sequence[Event]) -> set[str]:
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
        .join(Event, EventExternalId.event_id == Event.surrogate_id)  # type: ignore[arg-type]
        .where(
            EventExternalId.source == Source.MLB_STATS_API,
            EventExternalId.external_id.in_([str(pk) for pk in game_pks]),  # type: ignore[attr-defined]
        )
    )
    return {int(external_id): event for external_id, event in rows.all()}


async def event_holding_game_pk(session: AsyncSession, game_pk: int) -> Event | None:
    """The event holding ``game_pk``, if any."""
    return (await events_holding_game_pks_by_pk(session, [game_pk])).get(game_pk)
