"""Find or create the event of each game a live collection run lists (#612).

MLB games are resolved to their MLB Stats API ``gamePk`` once per run, so an MLB event is
found by, or created with, its ``gamePk``. Other sports match on teams within ±2 h.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import structlog
from odds_core.alerts import send_recorded_alert
from odds_core.config import get_settings

from odds_lambda.mlb_game_resolver import (
    MLB_SPORT_KEY,
    GameResolution,
    GameRow,
    MlbScheduleUnavailable,
    UnresolvedGame,
    resolve_batch,
)
from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

DROPPED_GAMES_SHOWN = 10


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

    async def find_or_create(
        self, writer: OddsWriter, game: GameRow
    ) -> tuple[str, bool] | DroppedGame:
        """``(event_id, was_created)``, or the game dropped from this run.

        Without the MLB schedule a game attaches only to the one event of its pair within ±2 h
        and never creates one, since a new event could not carry its ``gamePk``.
        """
        if self.sport_key != MLB_SPORT_KEY:
            return await writer.find_or_create_event(
                game.home_team, game.away_team, game.start, self.sport_key, self.sport_title
            )
        if self.resolutions is None:
            event_id = await writer.find_unscheduled_mlb_event(
                game.home_team, game.away_team, game.start, game.game_number
            )
            if event_id is None:
                return DroppedGame(game, SkipReason.SCHEDULE_UNAVAILABLE)
            return event_id, False
        resolution = self.resolutions[game]
        if isinstance(resolution, UnresolvedGame):
            return DroppedGame(game, SkipReason.UNRESOLVED, resolution.reason)
        return await writer.find_or_create_event(
            game.home_team,
            game.away_team,
            game.start,
            self.sport_key,
            self.sport_title,
            mlb_game=resolution,
        )


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
