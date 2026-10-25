"""Resolve a game a source lists to its one event, creating the event only when none matches."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import structlog
from odds_core.models import DataQualityLog, Event, EventExternalId, EventStatus
from odds_core.team import normalize_team_name
from odds_core.time import ensure_utc
from odds_core.vocabulary import EXTERNAL_ID_SOURCES, Source
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.mlb_game_resolver import MLB_SPORT_KEY, ResolvedGame
from odds_lambda.storage.data_quality import record_issue
from odds_lambda.storage.event_kickoff import move_kickoff
from odds_lambda.storage.event_merge import (
    EVENT_MATCH_WINDOW,
    find_historical_candidates,
    find_match_candidates,
)
from odds_lambda.storage.event_unmatched import (
    clear_unmatched,
    events_holding_game_pks,
    holds_game_pk,
)

logger = structlog.get_logger()

REFUSED_ID_ISSUE = "external_id_refused"


class MatchWindow(StrEnum):
    """How far a listing's start may be from a stored event's and still be the same game."""

    # ±2 h: a source listing upcoming games.
    LIVE = "live"
    # ±24 h: a historical row whose recorded start may be a day off.
    HISTORICAL = "historical"


@dataclass(frozen=True)
class ListedGame:
    """A game as one source lists it.

    ``external_id`` is the id the source issued for the game, if it issues one. An MLB game comes
    resolved to its ``gamePk`` (``mlb_game``). ``legacy_id`` is the string id an event created for
    the game takes while ``events.id`` is still a string; only a game that may create an event
    needs it. ``match_window`` is how close an event's start must be to match the game
    (``MatchWindow.HISTORICAL`` is not for MLB).
    """

    source: Source
    sport_key: str
    sport_title: str
    home_team: str
    away_team: str
    commence_time: datetime
    legacy_id: str | None = None
    external_id: str | None = None
    mlb_game: ResolvedGame | None = None
    earliest_created_start: datetime | None = None
    match_window: MatchWindow = MatchWindow.LIVE

    def __post_init__(self) -> None:
        if self.external_id is not None and self.source not in EXTERNAL_ID_SOURCES:
            raise ValueError(f"{self.source} issues no event ids")
        if self.sport_key == MLB_SPORT_KEY and self.mlb_game is None:
            raise ValueError(f"an MLB game needs its gamePk: {self.home_team} v {self.away_team}")
        if self.sport_key == MLB_SPORT_KEY and self.match_window is MatchWindow.HISTORICAL:
            raise ValueError("an MLB game matches by gamePk, not by a historical window")


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
    """Every string id a new event could take belongs to another event."""


class StaleListing(ValueError):
    """A live listing matched no event, and its start is too far past to create one."""


class LockOrderError(RuntimeError):
    """A transaction asked for a fixture lock below one it holds, which can deadlock."""


# The highest fixture lock each open transaction holds, keyed on the session.
_HELD_LOCK = "event_resolver_held_lock"


def event_lock_key(sport_key: str, home_team: str, away_team: str) -> int:
    """The advisory-lock key of a fixture. Kick-off is left out: two sources can place one game
    on different UTC dates, and both must wait on the same lock."""
    digest = hashlib.blake2b(
        f"{sport_key}\x1f{home_team}\x1f{away_team}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big", signed=True)


async def resolve_events(session: AsyncSession, games: Sequence[ListedGame]) -> list[ResolvedEvent]:
    """``resolve_event`` for each game, in lock-key order so that concurrent batches cannot
    deadlock; results come back in the order of ``games``."""
    order = sorted(range(len(games)), key=lambda i: _lock_key(games[i]))
    resolved: dict[int, ResolvedEvent] = {}
    for i in order:
        resolved[i] = await resolve_event(session, games[i])
    return [resolved[i] for i in range(len(games))]


async def resolve_event(session: AsyncSession, game: ListedGame) -> ResolvedEvent:
    """The event of ``game``, recording the source's id on it. Flushes; the caller commits.

    Holds a transaction advisory lock on the fixture until the caller's transaction ends, so
    writers resolving the same teams concurrently see each other's events. One transaction
    takes fixture locks in ascending key order (``resolve_events`` sorts a batch) or raises
    ``LockOrderError``. Looks an MLB game up by ``gamePk``, then any game by the source's id;
    else matches the sport and teams within the game's ``match_window`` (±2 h, or for a
    historical listing ±24 h, see ``_match_historical``; for MLB, only events holding no
    ``gamePk``); else creates the event. A second id from one source for the event, or an id another event
    holds, is refused and logged rather than recorded. The listed start is offered to
    ``move_kickoff``, which decides whether the source may move the event.

    Raises:
        StaleListing: when no event matches and the game starts before
            ``game.earliest_created_start``.
        EventIdTaken: when no event matches and every id a new event could take is taken.
    """
    home_team = normalize_team_name(game.home_team)
    away_team = normalize_team_name(game.away_team)
    resolved = await _lock_and_find(session, game, home_team, away_team)
    if resolved is None:
        resolved = ResolvedEvent(
            await _create(session, game, home_team, away_team), Resolution.CREATED
        )
    event = resolved.event
    if game.external_id is not None:
        await _record_external_id(session, event, game.source, game.external_id)
    now = datetime.now(UTC)
    if game.mlb_game is not None:
        await _record_game_pk(session, event, game.mlb_game.game_pk)
        await move_kickoff(session, event, Source.MLB_STATS_API, game.mlb_game.start, now=now)
    else:
        await move_kickoff(session, event, game.source, game.commence_time, now=now)
    return resolved


async def attach_event(session: AsyncSession, game: ListedGame) -> ResolvedEvent | None:
    """The existing event of ``game``, recording the source's id on it; ``None`` when none
    matches, and no event is created. The lock and lookup are ``resolve_event``'s; the listed
    start is not offered to ``move_kickoff``, for a source that is not a schedule authority.

    Also ``None`` when the event already holds another id from the source (or another event
    holds this one): the id is refused, so the game is not the event's to write prices for. The
    refusal is logged once per refused id and event. Flushes; the caller commits.

    Raises:
        LockOrderError: as ``resolve_event``.
    """
    resolved = await _lock_and_find(
        session, game, normalize_team_name(game.home_team), normalize_team_name(game.away_team)
    )
    if resolved is None:
        return None
    if game.external_id is not None and not await _record_external_id(
        session, resolved.event, game.source, game.external_id, log_repeats=False
    ):
        return None
    return resolved


async def attach_events(
    session: AsyncSession, games: Sequence[ListedGame]
) -> list[ResolvedEvent | None]:
    """``attach_event`` for each game, in lock-key order so that a batch holds fixture locks in
    ascending order; results come back in the order of ``games``."""
    order = sorted(range(len(games)), key=lambda i: _lock_key(games[i]))
    attached: dict[int, ResolvedEvent | None] = {}
    for i in order:
        attached[i] = await attach_event(session, games[i])
    return [attached[i] for i in range(len(games))]


async def resolve_unscheduled_mlb_event(
    session: AsyncSession,
    source: Source,
    home_team: str,
    away_team: str,
    start: datetime,
    game_number: int | None,
    external_id: str | None,
) -> ResolvedEvent | None:
    """The MLB event of a game listed while the MLB schedule is unavailable, recording the
    source's id on it under the fixture lock; ``None`` when it matches none. Never creates.

    Only the one event of the pair within ±2 h matches, and only for a row with no doubleheader
    game number: without the schedule either game could be another. Flushes; the caller commits.
    """
    home = normalize_team_name(home_team)
    away = normalize_team_name(away_team)
    await _lock(session, event_lock_key(MLB_SPORT_KEY, home, away))
    if game_number is not None:
        return None
    candidates = await find_match_candidates(session, home, away, start, MLB_SPORT_KEY)
    if len(candidates) != 1:
        return None
    event = candidates[0]
    if external_id is not None:
        await _record_external_id(session, event, source, external_id)
    return ResolvedEvent(event, Resolution.MATCHED)


def _lock_key(game: ListedGame) -> int:
    return event_lock_key(
        game.sport_key, normalize_team_name(game.home_team), normalize_team_name(game.away_team)
    )


@dataclass
class _HeldLocks:
    transaction: object
    keys: set[int]


async def _lock(session: AsyncSession, key: int) -> None:
    await session.connection()  # begins the transaction the lock belongs to
    transaction = session.sync_session.get_transaction()
    held = session.info.get(_HELD_LOCK)
    if not isinstance(held, _HeldLocks) or held.transaction is not transaction:
        held = _HeldLocks(transaction, set())
        session.info[_HELD_LOCK] = held
    # Re-taking a held lock cannot wait; a new one below a held one could deadlock.
    if key not in held.keys and held.keys and key < max(held.keys):
        raise LockOrderError(
            "fixture locks must be taken in ascending key order within one transaction; "
            "resolve a batch with resolve_events or commit between fixtures"
        )
    await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})
    held.keys.add(key)


async def _lock_and_find(
    session: AsyncSession, game: ListedGame, home_team: str, away_team: str
) -> ResolvedEvent | None:
    await _lock(session, _lock_key(game))
    return await _find(session, game, home_team, away_team)


async def _find(
    session: AsyncSession, game: ListedGame, home_team: str, away_team: str
) -> ResolvedEvent | None:
    if game.mlb_game is not None:
        held = await _by_game_pk(session, game.mlb_game.game_pk)
        if held is not None:
            return ResolvedEvent(held, Resolution.BY_GAME_PK)
    if game.external_id is not None:
        held = await _by_external_id(session, game.source, game.external_id)
        if held is None and game.source is Source.ODDS_API:
            held = await event_keyed_by_odds_api_id(session, game.external_id)
        if held is not None and (game.mlb_game is None or not await holds_game_pk(session, held)):
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


async def event_keyed_by_odds_api_id(session: AsyncSession, odds_api_id: str) -> Event | None:
    # Events stored before the external-id table existed are keyed by the API's own id.
    return await session.get(Event, odds_api_id)


async def _by_game_pk(session: AsyncSession, game_pk: int) -> Event | None:
    held = await _by_external_id(session, Source.MLB_STATS_API, str(game_pk))
    if held is not None:
        return held
    # Events keyed before the external-id table existed hold their gamePk only in the column.
    result = await session.execute(select(Event).where(Event.mlb_game_pk == game_pk))
    return result.scalar_one_or_none()


async def _match(
    session: AsyncSession, game: ListedGame, home_team: str, away_team: str
) -> Event | None:
    """The nearest same-fixture event within ±2 h. For MLB only an event holding no ``gamePk``
    counts, and only when no other game of the pair that day is scheduled nearer to it."""
    if game.match_window is MatchWindow.HISTORICAL:
        return await _match_historical(session, game, home_team, away_team)
    anchor = ensure_utc(_anchor(game))
    candidates = await find_match_candidates(session, home_team, away_team, anchor, game.sport_key)
    if game.mlb_game is not None:
        holding = await events_holding_game_pks(session, candidates)
        candidates = [e for e in candidates if e.id not in holding]
    if not candidates:
        return None

    def gap(event: Event, start: datetime) -> timedelta:
        return abs(ensure_utc(event.commence_time) - ensure_utc(start))

    nearest_gap = min(gap(e, anchor) for e in candidates)
    nearest = min(
        (e for e in candidates if gap(e, anchor) == nearest_gap),
        key=_stored_key,
    )
    if game.mlb_game is not None and any(
        gap(nearest, start) < nearest_gap for start in game.mlb_game.sibling_starts
    ):
        return None
    return nearest


async def _match_historical(
    session: AsyncSession, game: ListedGame, home_team: str, away_team: str
) -> Event | None:
    """The event of a historical row: a lone same-fixture event within ±24 h, else (several)
    the nearest of those within ±2 h. When several lie within ±24 h and none within ±2 h the
    row is ambiguous and matches nothing, so neighbouring fixtures of the same teams (an NBA
    back-to-back) are never conflated."""
    anchor = ensure_utc(game.commence_time)
    candidates = await find_historical_candidates(
        session, home_team, away_team, anchor, game.sport_key
    )
    if len(candidates) > 1:
        within = [
            e for e in candidates if abs(ensure_utc(e.commence_time) - anchor) <= EVENT_MATCH_WINDOW
        ]
        if not within:
            logger.warning(
                "ambiguous_historical_match",
                home=home_team,
                away=away_team,
                commence_time=anchor.isoformat(),
                candidates=[e.id for e in candidates],
            )
            return None
        candidates = within
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda e: (abs(ensure_utc(e.commence_time) - anchor), _stored_key(e)),
    )


def _stored_key(event: Event) -> int:
    """The tie-break order of stored events: the one stored first has the lowest key."""
    if event.surrogate_id is None:
        raise ValueError(f"event {event.id} is not stored")
    return event.surrogate_id


def _anchor(game: ListedGame) -> datetime:
    if game.mlb_game is not None and game.mlb_game.start is not None:
        return game.mlb_game.start
    return game.commence_time


async def _create(session: AsyncSession, game: ListedGame, home_team: str, away_team: str) -> Event:
    start = ensure_utc(_anchor(game))
    if game.earliest_created_start is not None and start < ensure_utc(game.earliest_created_start):
        raise StaleListing(
            f"{game.source.value} lists {home_team} v {away_team} at {start.isoformat()}, which "
            "matches no event and is too far past to create one"
        )
    if game.legacy_id is None:
        raise ValueError(
            f"a game that creates an event needs a legacy_id: {home_team} v {away_team}"
        )
    suffix = game.mlb_game.game_pk if game.mlb_game is not None else game.external_id
    choices = [game.legacy_id] + ([f"{game.legacy_id}_{suffix}"] if suffix is not None else [])
    event_id = None
    for choice in choices:
        if await session.get(Event, choice) is None:
            event_id = choice
            break
    if event_id is None:
        raise EventIdTaken(f"event ids {choices} are all taken by events the game did not match")
    if event_id != game.legacy_id:
        logger.warning("event_id_collision", event_id=game.legacy_id, used=event_id)
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
            await clear_unmatched(session, event)
            await session.flush()


async def _record_external_id(
    session: AsyncSession,
    event: Event,
    source: Source,
    external_id: str,
    *,
    log_repeats: bool = True,
) -> bool:
    """Record ``external_id`` as ``source``'s id for ``event``; whether the event now holds it.
    A refusal already logged for the event is logged again only with ``log_repeats``."""
    held = await session.get(EventExternalId, (source, external_id))
    if held is not None:
        if held.event_id == event.surrogate_id:
            return True
        await _refuse(
            session, event, source, external_id, f"held by event key {held.event_id}", log_repeats
        )
        return False
    result = await session.execute(
        select(EventExternalId.external_id).where(
            EventExternalId.event_id == event.surrogate_id, EventExternalId.source == source
        )
    )
    current = result.scalar_one_or_none()
    if current is not None and source is Source.ODDS_API:
        # The API reissues ids; the event's current one replaces the old.
        stale = await session.get(EventExternalId, (source, current))
        assert stale is not None
        await session.delete(stale)
        await session.flush()
        record_issue(
            session,
            event.id,
            "external_id_replaced",
            f"{source.value} id {current} replaced by {external_id}",
            {"source": source.value, "old_external_id": current, "external_id": external_id},
        )
    elif current is not None:
        await _refuse(
            session,
            event,
            source,
            external_id,
            f"the event holds {source.value} id {current}",
            log_repeats,
        )
        return False
    assert event.surrogate_id is not None
    session.add(
        EventExternalId(source=source, external_id=external_id, event_id=event.surrogate_id)
    )
    await session.flush()
    return True


async def _refuse(
    session: AsyncSession,
    event: Event,
    source: Source,
    external_id: str,
    reason: str,
    log_repeats: bool,
) -> None:
    description = f"{source.value} id {external_id} refused: {reason}"
    if not log_repeats:
        logged = await session.execute(
            select(DataQualityLog.id).where(
                DataQualityLog.event_id == event.id,
                DataQualityLog.issue_type == REFUSED_ID_ISSUE,
                DataQualityLog.description == description,
            )
        )
        if logged.first() is not None:
            return
    record_issue(
        session,
        event.id,
        REFUSED_ID_ISSUE,
        description,
        {"source": source.value, "external_id": external_id, "reason": reason},
    )
