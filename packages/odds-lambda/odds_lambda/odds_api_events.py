"""Resolve the games The Odds API lists to their events (#687)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog
from odds_core.api_models import ApiEvent
from odds_core.models import Event, EventExternalId
from odds_core.vocabulary import Source
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.live_events import DroppedGame, LiveEventFinder, SkipReason, lock_order
from odds_lambda.mlb_game_resolver import GameRow
from odds_lambda.storage.event_resolver import (
    Resolution,
    ResolvedEvent,
    StaleListing,
)

logger = structlog.get_logger()


def game_row(event: ApiEvent) -> GameRow:
    return GameRow(event.home_team, event.away_team, event.commence_time)


async def odds_api_finder(sport_key: str, events: Sequence[ApiEvent]) -> LiveEventFinder:
    sport_title = events[0].sport_title if events else sport_key
    return await LiveEventFinder.for_run(sport_key, sport_title, [game_row(e) for e in events])


async def resolve_odds_api_event(
    finder: LiveEventFinder,
    session: AsyncSession,
    event: ApiEvent,
    *,
    observed_at: datetime | None = None,
) -> ResolvedEvent | DroppedGame:
    """The listed game's event, or the game dropped. A game the API lists at or after its start,
    as of ``observed_at``, is listed at its actual start rather than its schedule, so it matches
    an event but creates none. Without ``observed_at`` (the historical backfill) a game with no
    event creates one however late."""
    try:
        return await finder.resolve(
            session,
            game_row(event),
            Source.ODDS_API,
            event.odds_api_id,
            now=observed_at,
            stale_after=None if observed_at is None else timedelta(0),
        )
    except StaleListing:
        return DroppedGame(game_row(event), SkipReason.UNDER_WAY)


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
    session: AsyncSession,
    finder: LiveEventFinder,
    events: Sequence[ApiEvent],
    *,
    observed_at: datetime,
) -> OddsApiResolutions:
    result = OddsApiResolutions()
    for event in sorted(events, key=lambda e: lock_order(finder.sport_key, game_row(e))):
        try:
            async with session.begin_nested():
                found = await resolve_odds_api_event(
                    finder, session, event, observed_at=observed_at
                )
        except Exception as e:
            error = ResolutionError(event.odds_api_id, type(e).__name__, str(e))
            result.errors.append(error)
            logger.error("odds_api_event_resolution_failed", error=error.describe(), exc_info=True)
            continue
        if isinstance(found, DroppedGame):
            result.dropped.append(found)
            logger.warning(
                "odds_api_game_dropped", odds_api_id=event.odds_api_id, game=found.describe()
            )
        elif found.resolution is Resolution.CREATED:
            result.created += 1
        else:
            result.matched += 1
    return result


async def find_event_by_odds_api_id(session: AsyncSession, odds_api_id: str) -> Event | None:
    held = await session.execute(
        select(Event)
        .join(EventExternalId, EventExternalId.event_id == Event.id)  # type: ignore[arg-type]
        .where(
            EventExternalId.source == Source.ODDS_API, EventExternalId.external_id == odds_api_id
        )
    )
    return held.scalar_one_or_none()
