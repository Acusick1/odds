"""Compare FINAL MLB events' scores with the MLB Stats API and correct the wrong ones (#578)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

import structlog
from odds_core.models import Event, EventStatus
from odds_core.team import normalize_team_name
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbStatsFetcher

logger = structlog.get_logger()

SPORT_KEY = "baseball_mlb"
MATCH_WINDOW = timedelta(hours=36)
MAX_START_GAP = timedelta(hours=6)


class ScoreOutcome(StrEnum):
    CORRECT = "correct"
    MISMATCH = "mismatch"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ScoreCheck:
    event: Event
    outcome: ScoreOutcome
    game: MlbGameResult | None = None
    reason: str = ""


def _dedupe(games: Sequence[MlbGameResult]) -> list[MlbGameResult]:
    """One entry per ``game_pk``: a postponed game reappears under its new date, preferring the played one."""
    best: dict[int, MlbGameResult] = {}
    for game in games:
        held = best.get(game.game_pk)
        if held is None or (game.is_final and not held.is_final):
            best[game.game_pk] = game
    return list(best.values())


def check_scores(events: Sequence[Event], games: Sequence[MlbGameResult]) -> list[ScoreCheck]:
    """Match each event to the game of the same teams nearest its kick-off within ``MATCH_WINDOW``.

    Games of a doubleheader (several games of one team pair on one date) are paired with the
    date's events by order, because the second game often carries a placeholder start time;
    the pairing is ambiguous unless the counts match. A match is also ambiguous when the
    start times are more than ``MAX_START_GAP`` apart, two games are equally near, or another
    event claims the same game. An event with no game, or whose game is not Final, is unmatched.
    """
    by_teams: dict[tuple[str, str], list[MlbGameResult]] = {}
    for game in _dedupe(games):
        key = (normalize_team_name(game.home_team), normalize_team_name(game.away_team))
        by_teams.setdefault(key, []).append(game)

    checks: dict[str, ScoreCheck] = {}
    single: dict[str, MlbGameResult] = {}
    grouped: dict[tuple[tuple[str, str], date], list[Event]] = {}
    for event in events:
        key = (normalize_team_name(event.home_team), normalize_team_name(event.away_team))
        near = sorted(
            (
                g
                for g in by_teams.get(key, [])
                if abs(g.start_time - event.commence_time) <= MATCH_WINDOW
            ),
            key=lambda g: abs(g.start_time - event.commence_time),
        )
        if not near:
            checks[event.id] = ScoreCheck(event, ScoreOutcome.UNMATCHED, reason="no game found")
            continue
        day_games = [g for g in by_teams[key] if g.day == near[0].day]
        if len(day_games) > 1:
            grouped.setdefault((key, near[0].day), []).append(event)
        elif len(near) > 1 and abs(near[0].start_time - event.commence_time) == abs(
            near[1].start_time - event.commence_time
        ):
            checks[event.id] = ScoreCheck(
                event, ScoreOutcome.AMBIGUOUS, reason="two games equally near"
            )
        else:
            single[event.id] = near[0]

    paired: dict[str, MlbGameResult] = dict(single)
    for (key, day), day_events in grouped.items():
        day_games = sorted((g for g in by_teams[key] if g.day == day), key=lambda g: g.game_number)
        if len(day_events) != len(day_games):
            for event in day_events:
                checks[event.id] = ScoreCheck(
                    event,
                    ScoreOutcome.AMBIGUOUS,
                    reason=f"doubleheader: {len(day_events)} events, {len(day_games)} games",
                )
            continue
        for event, game in zip(
            sorted(day_events, key=lambda e: e.commence_time), day_games, strict=True
        ):
            paired[event.id] = game

    claimed: dict[int, int] = {}
    for game in paired.values():
        claimed[game.game_pk] = claimed.get(game.game_pk, 0) + 1
    for event in events:
        if event.id in checks:
            continue
        game = paired[event.id]
        gap = abs(game.start_time - event.commence_time)
        if claimed[game.game_pk] > 1:
            checks[event.id] = ScoreCheck(
                event, ScoreOutcome.AMBIGUOUS, game, "game claimed by several events"
            )
        elif gap > MAX_START_GAP:
            checks[event.id] = ScoreCheck(
                event, ScoreOutcome.AMBIGUOUS, game, f"start times {gap} apart"
            )
        elif not game.is_final:
            checks[event.id] = ScoreCheck(event, ScoreOutcome.UNMATCHED, game, "game not final")
        else:
            correct = (event.home_score, event.away_score) == (game.home_score, game.away_score)
            checks[event.id] = ScoreCheck(
                event, ScoreOutcome.CORRECT if correct else ScoreOutcome.MISMATCH, game
            )
    return [checks[event.id] for event in events]


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
                .where(Event.sport_key == SPORT_KEY, Event.status == EventStatus.FINAL)
                .order_by(Event.commence_time)
            )
        ).scalars()
    )
    if not events:
        return []
    start = events[0].commence_time - MATCH_WINDOW
    end = events[-1].commence_time + MATCH_WINDOW
    games = await fetcher.fetch_results(start.date(), end.date())
    checks = check_scores(events, games)
    if apply:
        for check in checks:
            if check.outcome is ScoreOutcome.MISMATCH and check.game is not None:
                logger.info(
                    "mlb_score_repaired",
                    event_id=check.event.id,
                    game_pk=check.game.game_pk,
                    old=(check.event.home_score, check.event.away_score),
                    new=(check.game.home_score, check.game.away_score),
                )
                check.event.home_score = check.game.home_score
                check.event.away_score = check.game.away_score
    return checks
