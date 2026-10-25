"""Resolve each game a live collection run lists to its event (#612).

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
from odds_core.event_ids import LIVE_EVENT_ID_PREFIX
from odds_core.team import normalize_team_name, team_abbrev
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
    ListedGame,
    Resolution,
    ResolvedEvent,
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


def live_event_id(home_team: str, away_team: str, start: datetime) -> str:
    """The string id an event a live listing creates takes, while ``events.id`` is a string.

    It carries the start to the minute, so a doubleheader's games get distinct ids.
    """
    return (
        f"{LIVE_EVENT_ID_PREFIX}{team_abbrev(home_team)}_{team_abbrev(away_team)}_"
        f"{start:%Y-%m-%dT%H%M}"
    )


def lock_order(sport_key: str, game: GameRow) -> int:
    """Sort key that resolves a run's games in fixture-lock order within one transaction."""
    return event_lock_key(
        sport_key, normalize_team_name(game.home_team), normalize_team_name(game.away_team)
    )


class SkipReason(StrEnum):
    UNRESOLVED = "unresolved"
    SCHEDULE_UNAVAILABLE = "schedule_unavailable"


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
    raw_data: dict[str, Any]
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
    ) -> ResolvedEvent | DroppedGame:
        """The game's event, recording ``source``'s id for it, or the game dropped from this run.

        Without the MLB schedule a game attaches only to the one event of its pair within ±2 h
        and never creates one, since a new event could not carry its ``gamePk``. Flushes; the
        caller commits.

        Raises:
            StaleListing: when the game matches no event and started over
                ``LIVE_EVENT_STALE_THRESHOLD`` before ``now``.
        """
        now = now or datetime.now(UTC)
        mlb_game: ResolvedGame | None = None
        start = game.start
        if self.sport_key == MLB_SPORT_KEY:
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
            mlb_game = resolution
            start = resolution.start or game.start
        return await resolve_event(
            session,
            ListedGame(
                source=source,
                sport_key=self.sport_key,
                sport_title=self.sport_title,
                home_team=game.home_team,
                away_team=game.away_team,
                commence_time=game.start,
                legacy_id=live_event_id(
                    normalize_team_name(game.home_team), normalize_team_name(game.away_team), start
                ),
                external_id=external_id,
                mlb_game=mlb_game,
                earliest_created_start=now - LIVE_EVENT_STALE_THRESHOLD,
            ),
        )

    async def store_listings(
        self,
        session: AsyncSession,
        listings: Sequence[LiveListing],
        shape: SnapshotShape,
        api_request_id: str,
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
                    await writer.store_odds_snapshot(
                        event_id=found.event.id,
                        raw_data=listing.raw_data,
                        snapshot_time=listing.snapshot_time,
                        shape=shape,
                        api_request_id=api_request_id,
                    )
                    writes.snapshots_stored += 1
            except Exception as e:
                msg = f"{listing.label}: {type(e).__name__}: {e}"
                writes.errors.append(msg)
                logger.error(
                    "listing_ingestion_failed", source=shape.source.value, error=msg, exc_info=True
                )
        return writes


async def alert_dropped_games(job_name: str, dropped: Sequence[DroppedGame]) -> bool:
    """Warn that a run dropped listed games, at most every ``stale_reminder_minutes`` per job."""
    if not dropped:
        return False
    shown = "; ".join(d.describe() for d in dropped[:DROPPED_GAMES_SHOWN])
    more = (
        f"; and {len(dropped) - DROPPED_GAMES_SHOWN} more"
        if len(dropped) > DROPPED_GAMES_SHOWN
        else ""
    )
    return await send_recorded_alert(
        f"mlb_games_dropped:{job_name}",
        "warning",
        f"⚠️ {job_name} stored no odds for {len(dropped)} listed MLB games: {shown}{more}",
        every=timedelta(minutes=get_settings().alerts.stale_reminder_minutes),
        context={"job_name": job_name, "games": [d.describe() for d in dropped]},
    )
