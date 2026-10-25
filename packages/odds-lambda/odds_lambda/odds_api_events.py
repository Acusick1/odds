"""Resolve the games The Odds API lists to their events (#687)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog
from odds_core.event_ids import is_odds_api_event_id
from odds_core.models import Event, EventExternalId
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.live_events import DroppedGame, LiveEventFinder, lock_order
from odds_lambda.mlb_game_resolver import GameRow
from odds_lambda.storage.event_resolver import (
    Resolution,
    ResolvedEvent,
    event_keyed_by_odds_api_id,
)

logger = structlog.get_logger()


def game_row(event: Event) -> GameRow:
    return GameRow(event.home_team, event.away_team, event.commence_time)


async def odds_api_finder(sport_key: str, events: Sequence[Event]) -> LiveEventFinder:
    sport_title = events[0].sport_title if events else sport_key
    return await LiveEventFinder.for_run(sport_key, sport_title, [game_row(e) for e in events])


async def resolve_odds_api_event(
    finder: LiveEventFinder, session: AsyncSession, event: Event
) -> ResolvedEvent | DroppedGame:
    # The API lists games already under way, so a game with no event creates one however late.
    return await finder.resolve(
        session, game_row(event), Source.ODDS_API, event.id, legacy_id=event.id, create_stale=True
    )


@dataclass(frozen=True)
class ResolutionError:
    odds_api_id: str
    error_type: str
    message: str

    def describe(self) -> str:
        return f"{self.odds_api_id}: {self.error_type}: {self.message}"


@dataclass
class OddsApiResolutions:
    created: int = 0
    matched: int = 0
    dropped: list[DroppedGame] = field(default_factory=list)
    errors: list[ResolutionError] = field(default_factory=list)


async def resolve_odds_api_events(
    session: AsyncSession, finder: LiveEventFinder, events: Sequence[Event]
) -> OddsApiResolutions:
    result = OddsApiResolutions()
    for event in sorted(events, key=lambda e: lock_order(finder.sport_key, game_row(e))):
        try:
            async with session.begin_nested():
                found = await resolve_odds_api_event(finder, session, event)
        except Exception as e:
            error = ResolutionError(event.id, type(e).__name__, str(e))
            result.errors.append(error)
            logger.error("odds_api_event_resolution_failed", error=error.describe(), exc_info=True)
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
    held = await session.execute(
        select(Event)
        .join(EventExternalId, EventExternalId.event_id == Event.surrogate_id)  # type: ignore[arg-type]
        .where(
            EventExternalId.source == Source.ODDS_API, EventExternalId.external_id == odds_api_id
        )
    )
    event = held.scalar_one_or_none()
    return event if event is not None else await event_keyed_by_odds_api_id(session, odds_api_id)


async def odds_api_id_of(session: AsyncSession, event_id: str) -> str | None:
    held = await session.execute(
        select(EventExternalId.external_id)
        .join(Event, EventExternalId.event_id == Event.surrogate_id)  # type: ignore[arg-type]
        .where(EventExternalId.source == Source.ODDS_API, Event.id == event_id)
    )
    recorded = held.scalar_one_or_none()
    if recorded is not None:
        return recorded
    # An event stored before ids were recorded is keyed by the API's id when it came from the API.
    return event_id if is_odds_api_event_id(event_id) else None
