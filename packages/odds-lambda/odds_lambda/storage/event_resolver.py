"""Resolve a game a source lists to its one event, creating the event only when none matches."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

import structlog
from odds_core.event_source import canonical_event_id
from odds_core.models import DataQualityLog, Event, EventExternalId, EventStatus
from odds_core.team import normalize_team_name
from odds_core.time import ensure_utc
from odds_core.vocabulary import EXTERNAL_ID_SOURCES, Source
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.mlb_game_resolver import MLB_SPORT_KEY, ResolvedGame
from odds_lambda.storage.event_kickoff import move_kickoff
from odds_lambda.storage.event_merge import find_match_candidates

logger = structlog.get_logger()


@dataclass(frozen=True)
class ListedGame:
    """A game as one source lists it.

    ``external_id`` is the id the source issued for the game, if it issues one. An MLB game comes
    resolved to its ``gamePk`` (``mlb_game``). ``legacy_id`` is the string id an event created for
    the game takes while ``events.id`` is still a string.
    """

    source: Source
    sport_key: str
    sport_title: str
    home_team: str
    away_team: str
    commence_time: datetime
    legacy_id: str
    external_id: str | None = None
    mlb_game: ResolvedGame | None = None

    def __post_init__(self) -> None:
        if self.external_id is not None and self.source not in EXTERNAL_ID_SOURCES:
            raise ValueError(f"{self.source} issues no event ids")
        if self.sport_key == MLB_SPORT_KEY and self.mlb_game is None:
            raise ValueError(f"an MLB game needs its gamePk: {self.home_team} v {self.away_team}")


class Resolution(StrEnum):
    BY_EXTERNAL_ID = "by_external_id"
    BY_GAME_PK = "by_game_pk"
    MATCHED = "matched"
    CREATED = "created"


@dataclass(frozen=True)
class ResolvedEvent:
    event: Event
    resolution: Resolution


class EventIdTaken(ValueError):
    """A new event's string id belongs to another event and the game offers no suffix."""


def event_lock_key(sport_key: str, home_team: str, away_team: str) -> int:
    """The advisory-lock key of a fixture. Kick-off is left out: two sources can place one game
    on different UTC dates, and both must wait on the same lock."""
    digest = hashlib.blake2b(
        f"{sport_key}\x1f{home_team}\x1f{away_team}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big", signed=True)


async def resolve_event(session: AsyncSession, game: ListedGame) -> ResolvedEvent:
    """The event of ``game``, recording the source's id on it. Flushes; the caller commits.

    Holds a transaction advisory lock on the fixture, so writers resolving the same teams
    concurrently see each other's events. Looks an MLB game up by ``gamePk``, then any game by
    the source's id; else matches the sport and teams within ±2 h (for MLB, only events holding
    no ``gamePk``); else creates the event. A second id from one source for the event, or an id
    another event holds, is refused and logged rather than recorded.
    """
    home_team = normalize_team_name(game.home_team)
    away_team = normalize_team_name(game.away_team)
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:key)"),
        {"key": event_lock_key(game.sport_key, home_team, away_team)},
    )
    resolved = await _find(session, game, home_team, away_team)
    if resolved is None:
        resolved = ResolvedEvent(
            await _create(session, game, home_team, away_team), Resolution.CREATED
        )
    event = resolved.event
    if game.external_id is not None:
        await _record_external_id(session, event, game.source, game.external_id)
    if game.mlb_game is not None:
        await _record_game_pk(session, event, game.mlb_game.game_pk)
        await move_kickoff(session, event, Source.MLB_STATS_API, game.mlb_game.start)
    await move_kickoff(session, event, game.source, game.commence_time)
    return resolved


async def _find(
    session: AsyncSession, game: ListedGame, home_team: str, away_team: str
) -> ResolvedEvent | None:
    if game.mlb_game is not None:
        held = await _by_game_pk(session, game.mlb_game.game_pk)
        if held is not None:
            return ResolvedEvent(held, Resolution.BY_GAME_PK)
    if game.external_id is not None:
        held = await _by_external_id(session, game.source, game.external_id)
        if held is not None and (game.mlb_game is None or not await _holds_game_pk(session, held)):
            return ResolvedEvent(held, Resolution.BY_EXTERNAL_ID)
    matched = await _match(session, game, home_team, away_team)
    return ResolvedEvent(matched, Resolution.MATCHED) if matched is not None else None


async def _by_external_id(session: AsyncSession, source: Source, external_id: str) -> Event | None:
    result = await session.execute(
        select(Event)
        .join(EventExternalId, EventExternalId.event_id == Event.surrogate_id)  # type: ignore[arg-type]
        .where(EventExternalId.source == source, EventExternalId.external_id == external_id)
    )
    return result.scalar_one_or_none()


async def _by_game_pk(session: AsyncSession, game_pk: int) -> Event | None:
    held = await _by_external_id(session, Source.MLB_STATS_API, str(game_pk))
    if held is not None:
        return held
    # Events keyed before the external-id table existed hold their gamePk only in the column.
    result = await session.execute(select(Event).where(Event.mlb_game_pk == game_pk))
    return result.scalar_one_or_none()


async def _holds_game_pk(session: AsyncSession, event: Event) -> bool:
    return bool(await _events_holding_game_pks(session, [event]))


async def _events_holding_game_pks(session: AsyncSession, events: Sequence[Event]) -> set[str]:
    holding = {e.id for e in events if e.mlb_game_pk is not None}
    by_key = {e.surrogate_id: e.id for e in events}
    result = await session.execute(
        select(EventExternalId.event_id).where(
            EventExternalId.source == Source.MLB_STATS_API,
            EventExternalId.event_id.in_(list(by_key)),  # type: ignore[attr-defined]
        )
    )
    holding.update(by_key[key] for key in result.scalars())
    return holding


async def _match(
    session: AsyncSession, game: ListedGame, home_team: str, away_team: str
) -> Event | None:
    """The nearest same-fixture event within ±2 h. For MLB only an event holding no ``gamePk``
    counts, and only when no other game of the pair that day is scheduled nearer to it."""
    anchor = ensure_utc(_anchor(game))
    candidates = await find_match_candidates(session, home_team, away_team, anchor, game.sport_key)
    if game.mlb_game is not None:
        holding = await _events_holding_game_pks(session, candidates)
        candidates = [e for e in candidates if e.id not in holding]
    if not candidates:
        return None

    def gap(event: Event, start: datetime) -> timedelta:
        return abs(ensure_utc(event.commence_time) - ensure_utc(start))

    nearest_gap = min(gap(e, anchor) for e in candidates)
    nearest_ids = [e.id for e in candidates if gap(e, anchor) == nearest_gap]
    nearest = next(e for e in candidates if e.id == canonical_event_id(nearest_ids))
    if game.mlb_game is not None and any(
        gap(nearest, start) < nearest_gap for start in game.mlb_game.sibling_starts
    ):
        return None
    return nearest


def _anchor(game: ListedGame) -> datetime:
    if game.mlb_game is not None and game.mlb_game.start is not None:
        return game.mlb_game.start
    return game.commence_time


async def _create(session: AsyncSession, game: ListedGame, home_team: str, away_team: str) -> Event:
    event_id = game.legacy_id
    if await session.get(Event, event_id) is not None:
        suffix = game.mlb_game.game_pk if game.mlb_game is not None else game.external_id
        if suffix is None:
            raise EventIdTaken(f"event id {event_id} is taken by an event the game did not match")
        logger.warning("event_id_collision", event_id=event_id, suffix=suffix)
        event_id = f"{event_id}_{suffix}"
    event = Event(
        id=event_id,
        sport_key=game.sport_key,
        sport_title=game.sport_title,
        commence_time=_anchor(game),
        home_team=home_team,
        away_team=away_team,
        status=EventStatus.SCHEDULED,
    )
    session.add(event)
    await session.flush()
    logger.info("event_created", event_id=event_id, source=game.source.value)
    return event


async def _record_game_pk(session: AsyncSession, event: Event, game_pk: int) -> None:
    if await _record_external_id(session, event, Source.MLB_STATS_API, str(game_pk)):
        if event.mlb_game_pk is None:
            event.assign_game_pk(game_pk)
            await session.flush()


async def _record_external_id(
    session: AsyncSession, event: Event, source: Source, external_id: str
) -> bool:
    """Record ``external_id`` as ``source``'s id for ``event``; whether the event now holds it."""
    held = await session.get(EventExternalId, (source, external_id))
    if held is not None:
        if held.event_id == event.surrogate_id:
            return True
        _refuse(session, event, source, external_id, f"held by event key {held.event_id}")
        return False
    result = await session.execute(
        select(EventExternalId.external_id).where(
            EventExternalId.event_id == event.surrogate_id, EventExternalId.source == source
        )
    )
    current = result.scalar_one_or_none()
    if current is not None:
        _refuse(session, event, source, external_id, f"the event holds {source.value} id {current}")
        return False
    assert event.surrogate_id is not None
    session.add(
        EventExternalId(source=source, external_id=external_id, event_id=event.surrogate_id)
    )
    await session.flush()
    return True


def _refuse(
    session: AsyncSession, event: Event, source: Source, external_id: str, reason: str
) -> None:
    logger.warning(
        "external_id_refused",
        event_id=event.id,
        source=source.value,
        external_id=external_id,
        reason=reason,
    )
    session.add(
        DataQualityLog(
            event_id=event.id,
            severity="warning",
            issue_type="external_id_refused",
            description=f"{source.value} id {external_id} refused: {reason}",
            raw_data={"source": source.value, "external_id": external_id, "reason": reason},
        )
    )
