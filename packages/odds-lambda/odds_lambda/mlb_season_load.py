"""Played MLB seasons from the MLB Stats API schedule: one FINAL event per game, keyed by ``gamePk``.

The live jobs create MLB events only for games a source prices, so seasons before collection
began hold none, and the Odds API backfill plans its calls from stored FINAL events. Each Final
regular-season or postseason game is resolved through ``odds_lambda.storage.event_resolver``
with its ``gamePk``, so an event that already holds the ``gamePk`` (or matches the game and holds
none) is reused rather than duplicated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime

import structlog
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from odds_core.team import normalize_team_name
from odds_core.vocabulary import Source

from odds_lambda.mlb_game_resolver import (
    MLB_SPORT_KEY,
    ResolutionRule,
    ResolvedGame,
    final_scores,
    played_slots,
)
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbStatsFetcher
from odds_lambda.storage.event_resolver import ListedGame, Resolution, resolve_event
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

SPORT_TITLE = "MLB"
# The regular season and the four postseason rounds (wild card, division series, LCS, World
# Series); spring training, exhibitions and the All-Star Game are left out.
LOADED_GAME_TYPES: frozenset[str] = frozenset({"R", "F", "D", "L", "W"})


@dataclass(frozen=True)
class FinalGame:
    """A Final game at its played slot. ``start`` is the scheduled start, ``None`` when MLB lists
    it as TBD; ``commence_time`` is then the first pitch, else MLB's placeholder."""

    game_pk: int
    season: int
    home_team: str
    away_team: str
    commence_time: datetime
    start: datetime | None
    home_score: int
    away_score: int
    sibling_starts: tuple[datetime, ...] = ()

    def listed(self) -> ListedGame:
        return ListedGame(
            source=Source.MLB_STATS_API,
            sport_key=MLB_SPORT_KEY,
            sport_title=SPORT_TITLE,
            home_team=self.home_team,
            away_team=self.away_team,
            commence_time=self.commence_time,
            mlb_game=ResolvedGame(
                self.game_pk, ResolutionRule.SCHEDULE, self.start, self.sibling_starts
            ),
        )


@dataclass
class LoadReport:
    games: int = 0
    created: int = 0
    matched: int = 0
    finalised: int = 0
    games_by_season: dict[int, int] = field(default_factory=dict)
    score_conflicts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def final_games(schedule: Sequence[MlbGameResult]) -> list[FinalGame]:
    """Each Final regular-season or postseason game of ``schedule`` once, in start order.

    A postponed game is placed at the slot it was played in and a suspended one at the slot it
    began in (``played_slots``); its score is the Final slot's (``final_scores``). The
    ``sibling_starts`` of a doubleheader game are the other games of the pair that MLB date.
    """
    loaded = [slot for slot in schedule if slot.game_type in LOADED_GAME_TYPES]
    played = played_slots(loaded)
    listing = {slot.game_pk: slot for slot in loaded if not slot.resumed}
    placed: list[tuple[date, FinalGame]] = []
    for game_pk, (home_score, away_score) in final_scores(loaded).items():
        slot = played.get(game_pk)
        if slot is None:
            continue
        game = listing[game_pk]
        placed.append(
            (
                slot.day,
                FinalGame(
                    game_pk=game_pk,
                    season=slot.day.year,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    commence_time=(game.first_pitch or slot.start) if slot.tbd else slot.start,
                    start=None if slot.tbd else slot.start,
                    home_score=home_score,
                    away_score=away_score,
                ),
            )
        )

    by_pair: dict[tuple[str, str, date], list[FinalGame]] = defaultdict(list)
    for day, game in placed:
        pair = (normalize_team_name(game.home_team), normalize_team_name(game.away_team))
        by_pair[(*pair, day)].append(game)
    games = [
        replace(
            game,
            sibling_starts=tuple(
                other.commence_time for other in pair_games if other.game_pk != game.game_pk
            ),
        )
        for pair_games in by_pair.values()
        for game in pair_games
    ]
    return sorted(games, key=lambda g: (g.commence_time, g.game_pk))


async def fetch_season(fetcher: MlbStatsFetcher, season: int) -> list[FinalGame]:
    """The Final games of calendar year ``season``."""
    return final_games(await fetcher.fetch_results(date(season, 1, 1), date(season, 12, 31)))


async def load_games(games: Sequence[FinalGame], report: LoadReport, *, apply: bool) -> None:
    """Resolve each game to its event (creating it when none matches) and set it FINAL with MLB's
    score. An event already FINAL with another score is reported and left for
    ``odds events repair-scores``. One transaction per game, rolled back unless ``apply``."""
    for game in games:
        report.games += 1
        report.games_by_season[game.season] = report.games_by_season.get(game.season, 0) + 1
        try:
            async with async_session_maker() as session:
                resolved = await resolve_event(session, game.listed())
                event = resolved.event
                created = resolved.resolution is Resolution.CREATED
                report.created += int(created)
                report.matched += int(not created)
                stored = (event.home_score, event.away_score)
                score = (game.home_score, game.away_score)
                if event.status is EventStatus.FINAL and None not in stored:
                    if stored != score:
                        report.score_conflicts.append(
                            f"event {event.id} (game {game.game_pk}): stored "
                            f"{stored[0]}-{stored[1]}, MLB {score[0]}-{score[1]}"
                        )
                else:
                    await OddsWriter(session).update_event_status(
                        event.id, EventStatus.FINAL, *score
                    )
                    report.finalised += 1
                if apply:
                    await session.commit()
                else:
                    await session.rollback()
        except Exception as e:
            report.errors.append(
                f"game {game.game_pk} {game.away_team} at {game.home_team} "
                f"{game.commence_time:%F}: {e}"
            )
            logger.error("mlb_season_load_failed", game_pk=game.game_pk, exc_info=True)


async def load_seasons(
    seasons: Sequence[int], *, apply: bool, fetcher: MlbStatsFetcher | None = None
) -> LoadReport:
    """Load the Final games of each calendar season. ``fetcher`` replaces the default client."""
    report = LoadReport()
    async with fetcher or MlbStatsFetcher() as client:
        for season in seasons:
            await load_games(await fetch_season(client, season), report, apply=apply)
    return report
