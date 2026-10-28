"""Compare FINAL MLB events' scores with the MLB Stats API by ``gamePk`` and correct the wrong
ones (#578, #612)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

import structlog
from odds_core.models import Event, EventStatus
from odds_core.time import ensure_utc
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.mlb_game_resolver import MLB_SPORT_KEY, Score, final_scores, mlb_date
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbStatsFetcher
from odds_lambda.storage.game_pks import game_pks_of

logger = structlog.get_logger()


class ScoreOutcome(StrEnum):
    CORRECT = "correct"
    MISMATCH = "mismatch"
    UNMATCHED = "unmatched"


@dataclass(frozen=True)
class ScoreCheck:
    """``game_pk`` is the event's ``gamePk``; ``mlb_score`` is the (home, away) score of its game
    when it is Final."""

    event: Event
    outcome: ScoreOutcome
    game_pk: int | None = None
    mlb_score: Score | None = None
    reason: str = ""


def check_scores(
    events: Sequence[Event], game_pks: Mapping[str, int], games: Sequence[MlbGameResult]
) -> list[ScoreCheck]:
    """Compare each event's score with the Final score of the game its ``gamePk`` (``game_pks``,
    by event id) names.

    An event without a ``gamePk``, or whose game is not Final in ``games``, is unmatched.
    """
    finals = final_scores(games)
    checks: list[ScoreCheck] = []
    for event in events:
        game_pk = game_pks.get(event.id)
        if game_pk is None:
            checks.append(ScoreCheck(event, ScoreOutcome.UNMATCHED, reason="no gamePk"))
            continue
        final = finals.get(game_pk)
        if final is None:
            checks.append(
                ScoreCheck(event, ScoreOutcome.UNMATCHED, game_pk, reason="game not final")
            )
            continue
        correct = (event.home_score, event.away_score) == final
        outcome = ScoreOutcome.CORRECT if correct else ScoreOutcome.MISMATCH
        checks.append(ScoreCheck(event, outcome, game_pk, final))
    return checks


async def _fetch_games(
    fetcher: MlbStatsFetcher, events: Sequence[Event], game_pks: Mapping[str, int]
) -> list[MlbGameResult]:
    """The schedule over the events' MLB dates, plus each game not Final in it fetched by
    ``gamePk``, which finds a game played away from the event's date."""
    days = [mlb_date(ensure_utc(event.commence_time)) for event in events]
    games = await fetcher.fetch_results(
        min(days) - timedelta(days=1), max(days) + timedelta(days=1)
    )
    finals = final_scores(games)
    held = {game_pks[event.id] for event in events if event.id in game_pks}
    for game_pk in sorted(held - finals.keys()):
        games.extend(await fetcher.fetch_game(game_pk))
    return games


async def repair_scores(
    session: AsyncSession, fetcher: MlbStatsFetcher, apply: bool
) -> list[ScoreCheck]:
    """Check every FINAL MLB event; with ``apply`` write the API's scores into the mismatches.

    Writes ``home_score`` and ``away_score`` only, directly on the row: status, completion time
    and everything else stay as they are, so the FINAL status guard in
    ``OddsWriter.update_event_status`` is not involved. The caller commits.
    """
    events = list(
        (
            await session.execute(
                select(Event)
                .where(Event.sport_key == MLB_SPORT_KEY, Event.status == EventStatus.FINAL)
                .order_by(Event.commence_time)
            )
        ).scalars()
    )
    game_pks = await game_pks_of(session, events)
    keyed = [event for event in events if event.id in game_pks]
    games = await _fetch_games(fetcher, keyed, game_pks) if keyed else []
    checks = check_scores(events, game_pks, games)
    if apply:
        for check in checks:
            if check.outcome is ScoreOutcome.MISMATCH and check.mlb_score is not None:
                logger.info(
                    "mlb_score_repaired",
                    event_id=check.event.id,
                    game_pk=check.game_pk,
                    old=(check.event.home_score, check.event.away_score),
                    new=check.mlb_score,
                )
                check.event.home_score, check.event.away_score = check.mlb_score
    return checks
