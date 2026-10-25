"""Resolve the games The Odds API lists to their events, recording the API's id for each (#687).

Every Odds API writer (event sync, ``odds discover``, ingestion, the historical backfill) goes
through ``LiveEventFinder`` and so through the event resolver: a game attaches to the event of
its match, whichever source stored it first, and creates an event only when none matches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog
from odds_core.models import Event, EventExternalId
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.live_events import DroppedGame, LiveEventFinder, lock_order
from odds_lambda.mlb_game_resolver import GameRow
from odds_lambda.storage.event_resolver import Resolution, ResolvedEvent

logger = structlog.get_logger()


def game_row(event: Event) -> GameRow:
    return GameRow(event.home_team, event.away_team, event.commence_time)


async def odds_api_finder(sport_key: str, events: Sequence[Event]) -> LiveEventFinder:
    """The finder for a run listing ``events``; an MLB run reads the MLB schedule once."""
    sport_title = events[0].sport_title if events else sport_key
    return await LiveEventFinder.for_run(sport_key, sport_title, [game_row(e) for e in events])


async def resolve_odds_api_event(
    finder: LiveEventFinder, session: AsyncSession, event: Event
) -> ResolvedEvent | DroppedGame:
    """The stored event of a game the API lists (``event.id`` is the API's id for it).

    A game that creates an event gives it the API's id as its key. Flushes; the caller commits.
    """
    return await finder.resolve(
        session, game_row(event), Source.ODDS_API, event.id, legacy_id=event.id, create_stale=True
    )


@dataclass
class OddsApiResolutions:
    """What resolving a batch of listed games did."""

    created: int = 0
    matched: int = 0
    dropped: list[DroppedGame] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def resolve_odds_api_events(
    session: AsyncSession, finder: LiveEventFinder, events: Sequence[Event]
) -> OddsApiResolutions:
    """Resolve every listed game in the caller's transaction, in fixture-lock order.

    Each game runs in its own savepoint: one that fails is rolled back and recorded in
    ``errors``, and the others stand. Does not commit.
    """
    result = OddsApiResolutions()
    for event in sorted(events, key=lambda e: lock_order(finder.sport_key, game_row(e))):
        try:
            async with session.begin_nested():
                found = await resolve_odds_api_event(finder, session, event)
        except Exception as e:
            msg = f"{event.id}: {type(e).__name__}: {e}"
            result.errors.append(msg)
            logger.error("odds_api_event_resolution_failed", error=msg, exc_info=True)
            continue
        if isinstance(found, DroppedGame):
            result.dropped.append(found)
            logger.warning("odds_api_game_dropped", event_id=event.id, game=found.describe())
        elif found.resolution is Resolution.CREATED:
            result.created += 1
        else:
            result.matched += 1
    return result


async def find_event_by_odds_api_id(session: AsyncSession, odds_api_id: str) -> Event | None:
    """The event the API's id names: the one recorded as holding it, else, for an event stored
    before external ids were recorded, the one keyed by it."""
    held = await session.execute(
        select(Event)
        .join(EventExternalId, EventExternalId.event_id == Event.surrogate_id)  # type: ignore[arg-type]
        .where(
            EventExternalId.source == Source.ODDS_API, EventExternalId.external_id == odds_api_id
        )
    )
    event = held.scalar_one_or_none()
    return event if event is not None else await session.get(Event, odds_api_id)


async def odds_api_id_of(session: AsyncSession, event_id: str) -> str:
    """The id The Odds API uses for the stored event ``event_id``: the one recorded for it, else,
    for an event stored before external ids were recorded, its own key."""
    held = await session.execute(
        select(EventExternalId.external_id)
        .join(Event, EventExternalId.event_id == Event.surrogate_id)  # type: ignore[arg-type]
        .where(EventExternalId.source == Source.ODDS_API, Event.id == event_id)
    )
    return held.scalar_one_or_none() or event_id
