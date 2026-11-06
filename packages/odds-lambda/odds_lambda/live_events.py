"""Resolve each game a collection run lists to its event (#612).

MLB games are resolved to their MLB Stats API ``gamePk`` once per run, so an MLB event is
found by, or created with, its ``gamePk``. Every game then goes through the event resolver
(``odds_lambda.storage.event_resolver.resolve_event``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import structlog
from odds_core.alerts import send_recorded_alert
from odds_core.config import get_settings
from odds_core.team import normalize_team_name
from odds_core.vocabulary import Source
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.collection_calendar import EventKey, FixtureKey, MlbGameKey
from odds_lambda.mlb_game_resolver import (
    MLB_SPORT_KEY,
    GameResolution,
    GameRow,
    MlbScheduleUnavailable,
    ResolutionRule,
    ResolvedGame,
    UnresolvedGame,
    resolve_batch,
)
from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher
from odds_lambda.storage.event_merge import find_match_id
from odds_lambda.storage.event_resolver import (
    IdDecision,
    ListedGame,
    RefusedId,
    Resolution,
    ResolvedEvent,
    attach_listing,
    event_lock_key,
    resolve_event,
    resolve_unscheduled_mlb_event,
)
from odds_lambda.storage.shaped_snapshots import SnapshotShape
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

DROPPED_GAMES_SHOWN = 10
# A live listing this far past its start matches an event but creates none.
LIVE_EVENT_STALE_THRESHOLD = timedelta(hours=1)


def lock_order(sport_key: str, game: GameRow) -> int:
    """Sort key that resolves a run's games in fixture-lock order within one transaction."""
    return event_lock_key(
        sport_key, normalize_team_name(game.home_team), normalize_team_name(game.away_team)
    )


class SkipReason(StrEnum):
    UNRESOLVED = "unresolved"
    SCHEDULE_UNAVAILABLE = "schedule_unavailable"
    NO_EVENT = "no_event"
    ID_REFUSED = "id_refused"
    # Listed already under way, at its actual start, and matching no event.
    UNDER_WAY = "under_way"


@dataclass(frozen=True)
class DroppedGame:
    """A listed game that got no event and no snapshot in this run."""

    game: GameRow
    reason: SkipReason
    detail: str = ""

    def describe(self) -> str:
        label = f" game {self.game.game_number}" if self.game.game_number is not None else ""
        why = f"{self.reason}: {self.detail}" if self.detail else str(self.reason)
        return (
            f"{self.game.away_team} @ {self.game.home_team}{label} "
            f"{self.game.start:%Y-%m-%d %H:%M}Z ({why})"
        )


@dataclass(frozen=True)
class LiveListing:
    """One market of one game as a live run read it: the envelope it stores."""

    game: GameRow
    external_id: str | None
    envelope: dict[str, Any]
    snapshot_time: datetime
    label: str


@dataclass
class ListingWrites:
    """What storing a run's listings did; ``dropped`` holds each dropped game once."""

    events_created: int = 0
    events_matched: int = 0
    snapshots_stored: int = 0
    dropped: dict[GameRow, DroppedGame] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def count_dropped(dropped: Iterable[DroppedGame], reason: SkipReason) -> int:
    return sum(1 for d in dropped if d.reason is reason)


@dataclass(frozen=True)
class LiveEventFinder:
    """The run's event lookup; ``resolutions`` is ``None`` when the MLB schedule was unavailable."""

    sport_key: str
    sport_title: str
    resolutions: Mapping[GameRow, GameResolution] | None

    @classmethod
    async def for_run(
        cls, sport_key: str, sport_title: str, games: Iterable[GameRow]
    ) -> LiveEventFinder:
        """Resolve every listed MLB game with one schedule request; other sports need none."""
        if sport_key != MLB_SPORT_KEY:
            return cls(sport_key, sport_title, {})
        rows = list(dict.fromkeys(games))
        try:
            async with MlbStatsFetcher() as fetcher:
                resolutions = await resolve_batch(rows, fetcher)
        except MlbScheduleUnavailable as e:
            logger.warning("mlb_schedule_unavailable", error=str(e), games=len(rows))
            return cls(sport_key, sport_title, None)
        return cls(sport_key, sport_title, dict(zip(rows, resolutions, strict=True)))

    @property
    def schedule_unavailable(self) -> bool:
        return self.resolutions is None

    async def fixture_key(self, session: AsyncSession, game: GameRow) -> FixtureKey | None:
        """The calendar fixture a listed game identifies, never creating one; ``None`` if it matches nothing.

        An MLB game is its resolved ``gamePk``, unless the resolver only inferred it from a started
        sibling: that row can be the started game's own, and naming the next game would excuse it.
        Another sport's game is the event within ±2 h.
        """
        if self.sport_key == MLB_SPORT_KEY:
            resolution = (self.resolutions or {}).get(game)
            if (
                not isinstance(resolution, ResolvedGame)
                or resolution.rule is ResolutionRule.SIBLING_STARTED
            ):
                return None
            return MlbGameKey(resolution.game_pk)
        event_id = await find_match_id(
            session,
            normalize_team_name(game.home_team),
            normalize_team_name(game.away_team),
            game.start,
            self.sport_key,
        )
        return EventKey(event_id) if event_id is not None else None

    async def resolve(
        self,
        session: AsyncSession,
        game: GameRow,
        source: Source,
        external_id: str | None,
        *,
        now: datetime | None = None,
        stale_after: timedelta | None = LIVE_EVENT_STALE_THRESHOLD,
    ) -> ResolvedEvent | DroppedGame:
        """The game's event, recording ``source``'s id for it, or the game dropped from this run.

        A game starting ``stale_after`` or more before ``now`` matches an event but creates none;
        with ``stale_after`` ``None`` a game creates an event however late.

        Without the MLB schedule a game attaches only to the one event of its pair within ±2 h
        and never creates one, since a new event could not carry its ``gamePk``. Flushes; the
        caller commits.

        Raises:
            StaleListing: when the game matches no event and started ``stale_after`` or more
                before ``now``.
        """
        now = now or datetime.now(UTC)
        mlb_game = await self._mlb_game(session, game, source, external_id)
        if isinstance(mlb_game, (ResolvedEvent, DroppedGame)):
            return mlb_game
        return await resolve_event(
            session,
            self._listed(
                game,
                source,
                external_id,
                mlb_game,
                earliest_created_start=None if stale_after is None else now - stale_after,
            ),
        )

    async def attach_historical(
        self, session: AsyncSession, game: GameRow, source: Source, external_id: str | None
    ) -> ResolvedEvent | DroppedGame:
        """The stored event of a historical listing, recording ``source``'s id for it, or the game
        dropped: ``SkipReason.NO_EVENT`` when no event is the listed game, ``SkipReason.ID_REFUSED``
        when its event refused the id (another event holds it). Never creates an event and moves
        no kick-off.

        An MLB game is its ``gamePk``'s event. Any other game is the event of its teams within
        ±2 h of the listed start, never found by the source's id, so a game listed at a kick-off
        its event no longer has (before a postponement) attaches to nothing, whatever the order
        of the calls that listed it. Flushes; the caller commits.
        """
        mlb_game = await self._mlb_game(session, game, source, external_id)
        if isinstance(mlb_game, (ResolvedEvent, DroppedGame)):
            return mlb_game
        found = await attach_listing(
            session,
            self._listed(game, source, external_id, mlb_game),
            by_listed_start=mlb_game is None,
        )
        if isinstance(found, RefusedId):
            return DroppedGame(game, SkipReason.ID_REFUSED, describe_refusal(found))
        return found or DroppedGame(game, SkipReason.NO_EVENT)

    async def _mlb_game(
        self, session: AsyncSession, game: GameRow, source: Source, external_id: str | None
    ) -> ResolvedGame | ResolvedEvent | DroppedGame | None:
        """An MLB game's ``gamePk`` resolution; the attached event or dropped game when the MLB
        schedule was unavailable or lacks the game; ``None`` for another sport."""
        if self.sport_key != MLB_SPORT_KEY:
            return None
        if self.resolutions is None:
            found = await resolve_unscheduled_mlb_event(
                session,
                source,
                game.home_team,
                game.away_team,
                game.start,
                game.game_number,
                external_id,
            )
            return found or DroppedGame(game, SkipReason.SCHEDULE_UNAVAILABLE)
        resolution = self.resolutions[game]
        if isinstance(resolution, UnresolvedGame):
            return DroppedGame(game, SkipReason.UNRESOLVED, resolution.reason)
        return resolution

    def _listed(
        self,
        game: GameRow,
        source: Source,
        external_id: str | None,
        mlb_game: ResolvedGame | None,
        earliest_created_start: datetime | None = None,
    ) -> ListedGame:
        return ListedGame(
            source=source,
            sport_key=self.sport_key,
            sport_title=self.sport_title,
            home_team=game.home_team,
            away_team=game.away_team,
            commence_time=game.start,
            external_id=external_id,
            mlb_game=mlb_game,
            earliest_created_start=earliest_created_start,
        )

    async def store_listings(
        self,
        session: AsyncSession,
        listings: Sequence[LiveListing],
        shape: SnapshotShape,
        *,
        now: datetime | None = None,
    ) -> ListingWrites:
        """Resolve each listing's game and store its snapshot, all in the caller's transaction.

        Listings go in fixture-lock order (``lock_order``), so one transaction never asks for a
        lock below one it holds. Each runs in its own savepoint: a failing listing is rolled
        back and recorded in ``errors``, and the run's other writes stand. Does not commit.
        """
        writer = OddsWriter(session)
        writes = ListingWrites()
        for listing in sorted(listings, key=lambda item: lock_order(self.sport_key, item.game)):
            try:
                async with session.begin_nested():
                    found = await self.resolve(
                        session, listing.game, shape.source, listing.external_id, now=now
                    )
                    if isinstance(found, DroppedGame):
                        writes.dropped[found.game] = found
                        continue
                    if found.resolution is Resolution.CREATED:
                        writes.events_created += 1
                    else:
                        writes.events_matched += 1
                    stored = await writer.store_envelope(
                        event_id=found.event.id,
                        envelope=listing.envelope,
                        snapshot_time=listing.snapshot_time,
                        shape=shape,
                    )
                    writes.snapshots_stored += stored.snapshots_written
            except Exception as e:
                msg = f"{listing.label}: {type(e).__name__}: {e}"
                writes.errors.append(msg)
                logger.error(
                    "listing_ingestion_failed", source=shape.source.value, error=msg, exc_info=True
                )
        return writes


def describe_refusal(refused: RefusedId) -> str:
    source = refused.source.value
    if refused.check.decision is IdDecision.HELD_ELSEWHERE:
        return (
            f"{source} id {refused.external_id} is held by event {refused.check.other}, "
            f"not the matched event {refused.event.id}"
        )
    return f"event {refused.event.id} holds {source} id {refused.check.other}"


async def alert_dropped_games(job_name: str, dropped: Sequence[DroppedGame]) -> bool:
    """Warn that a run dropped listed games, at most every ``stale_reminder_minutes`` per job.
    A game dropped as under way is expected and not alerted."""
    dropped = [d for d in dropped if d.reason is not SkipReason.UNDER_WAY]
    if not dropped:
        return False
    shown = "; ".join(d.describe() for d in dropped[:DROPPED_GAMES_SHOWN])
    more = (
        f"; and {len(dropped) - DROPPED_GAMES_SHOWN} more"
        if len(dropped) > DROPPED_GAMES_SHOWN
        else ""
    )
    return await send_recorded_alert(
        f"games_dropped:{job_name}",
        "warning",
        f"⚠️ {job_name} stored no odds for {len(dropped)} listed games: {shown}{more}",
        every=timedelta(minutes=get_settings().alerts.stale_reminder_minutes),
        context={"job_name": job_name, "games": [d.describe() for d in dropped]},
    )
