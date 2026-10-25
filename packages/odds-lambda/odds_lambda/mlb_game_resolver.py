"""Resolve MLB events to their MLB Stats API ``gamePk`` (#612)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from enum import StrEnum

import httpx
import structlog
from odds_core.sports import SportKey
from odds_core.team import normalize_team_name
from odds_core.time import to_eastern

from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbGameState, MlbStatsFetcher

logger = structlog.get_logger()

MLB_SPORT_KEY: SportKey = "baseball_mlb"
MAX_SINGLE_GAME_GAP = timedelta(hours=6)
START_MATCH_WINDOW = timedelta(minutes=15)
STARTED_SIBLING_MARGIN = timedelta(minutes=3)


class MlbScheduleUnavailable(Exception):
    """The MLB schedule could not be read: the request failed after retries or the response was unusable."""


@dataclass(frozen=True)
class GameRow:
    """One listed game to resolve; ``game_number`` is Betfair's "Game N" label when it has one."""

    home_team: str
    away_team: str
    start: datetime
    game_number: int | None = None


class ResolutionRule(StrEnum):
    SINGLE_GAME = "single_game"
    GAME_NUMBER = "game_number"
    START_ORDER = "start_order"
    SIBLING_STARTED = "sibling_started"
    START_TIME = "start_time"


class UnresolvedReason(StrEnum):
    NO_GAME = "no_game"
    START_TOO_FAR = "start_too_far"
    GAME_NUMBER_NOT_FOUND = "game_number_not_found"
    SAME_START_ROWS = "same_start_rows"
    AMBIGUOUS_DOUBLEHEADER = "ambiguous_doubleheader"


@dataclass(frozen=True)
class ResolvedGame:
    """The row's game. ``start`` is its scheduled start from a slot that is neither postponed nor
    TBD (``None`` when the fetched schedule has none); ``sibling_starts`` are the listed starts of
    the other games of that pair on that date, TBD placeholders included."""

    game_pk: int
    rule: ResolutionRule
    start: datetime | None = None
    sibling_starts: tuple[datetime, ...] = ()


@dataclass(frozen=True)
class UnresolvedGame:
    reason: UnresolvedReason
    detail: str = ""


GameResolution = ResolvedGame | UnresolvedGame

_STARTED = (MlbGameState.LIVE, MlbGameState.FINAL)

_PairDate = tuple[tuple[str, str], date]


def mlb_date(start: datetime) -> date:
    """The MLB schedule date of a start time: its US Eastern calendar date."""
    return to_eastern(start).date()


def _pair(home: str, away: str) -> tuple[str, str]:
    return normalize_team_name(home), normalize_team_name(away)


def _day_games(slots: Sequence[MlbGameResult]) -> list[MlbGameResult]:
    """One slot per ``gamePk`` in schedule order, preferring a played slot over a postponed one."""
    by_pk: dict[int, MlbGameResult] = {}
    for slot in slots:
        held = by_pk.get(slot.game_pk)
        if held is None or (
            held.state is MlbGameState.POSTPONED and slot.state is not MlbGameState.POSTPONED
        ):
            by_pk[slot.game_pk] = slot
    return sorted(by_pk.values(), key=lambda g: (g.game_number, g.start_time))


def played_starts(schedule: Sequence[MlbGameResult]) -> dict[int, datetime]:
    """Each game's scheduled start, from a slot that is neither postponed, cancelled, TBD nor a
    resumed continuation; a game with no such slot is left out."""
    return {
        slot.game_pk: slot.start_time
        for slot in schedule
        if not slot.resumed
        and not slot.start_time_tbd
        and slot.state not in (MlbGameState.POSTPONED, MlbGameState.CANCELLED)
    }


def resolve_game_pks(
    rows: Sequence[GameRow], schedule: Sequence[MlbGameResult]
) -> list[GameResolution]:
    """Resolve each row to the ``gamePk`` of its game in ``schedule``, in input order.

    Rows and schedule slots are grouped by normalized team pair and MLB date. Every slot counts,
    postponed ones included; a resumed suspended game's continuation slot does not. Within a group
    the first rule that applies wins:

    1. one game: that game, unless its start is more than ``MAX_SINGLE_GAME_GAP`` from the row's
       (a game whose start is TBD is exempt);
    2. a doubleheader: the game with the row's ``game_number``, else, when the group has as many
       rows without a game number as games, the games paired with those rows by start order;
    3. a lone row whose other games are all Live or Final and started more than
       ``STARTED_SIBLING_MARGIN`` before it: the remaining game, when it is still Scheduled;
    4. exactly one game with a slot of any state starting within ``START_MATCH_WINDOW`` of the
       row, none of them at a TBD placeholder: that game;
    5. otherwise unresolved.
    """
    slots_by_group: dict[_PairDate, list[MlbGameResult]] = {}
    for slot in schedule:
        if slot.resumed:
            continue
        key = (_pair(slot.home_team, slot.away_team), slot.schedule_date or slot.day)
        slots_by_group.setdefault(key, []).append(slot)

    row_keys = [(_pair(row.home_team, row.away_team), mlb_date(row.start)) for row in rows]
    unnumbered_by_group: dict[_PairDate, list[GameRow]] = {}
    for row, key in zip(rows, row_keys, strict=True):
        if row.game_number is None:
            unnumbered_by_group.setdefault(key, []).append(row)

    starts = played_starts(schedule)
    resolutions: list[GameResolution] = []
    for row, key in zip(rows, row_keys, strict=True):
        slots = slots_by_group.get(key, [])
        resolution = _resolve_row(row, slots, unnumbered_by_group.get(key, []))
        if isinstance(resolution, ResolvedGame):
            resolution = replace(
                resolution,
                start=starts.get(resolution.game_pk),
                sibling_starts=tuple(
                    g.start_time for g in _day_games(slots) if g.game_pk != resolution.game_pk
                ),
            )
        resolutions.append(resolution)
    return resolutions


def _resolve_row(
    row: GameRow, slots: Sequence[MlbGameResult], unnumbered: Sequence[GameRow]
) -> GameResolution:
    games = _day_games(slots)
    if not games:
        return UnresolvedGame(UnresolvedReason.NO_GAME)

    if len(games) == 1:
        game = games[0]
        gap = abs(game.start_time - row.start)
        if gap > MAX_SINGLE_GAME_GAP and not game.start_time_tbd:
            return UnresolvedGame(
                UnresolvedReason.START_TOO_FAR, f"game {game.game_pk} starts {gap} away"
            )
        return ResolvedGame(game.game_pk, ResolutionRule.SINGLE_GAME)

    if row.game_number is not None:
        numbered = [g for g in games if g.game_number == row.game_number]
        if len(numbered) == 1:
            return ResolvedGame(numbered[0].game_pk, ResolutionRule.GAME_NUMBER)
        return UnresolvedGame(
            UnresolvedReason.GAME_NUMBER_NOT_FOUND,
            f"{len(numbered)} games numbered {row.game_number}",
        )

    if len(unnumbered) == len(games):
        ordered = sorted(unnumbered, key=lambda r: r.start)
        starts = [r.start for r in ordered]
        if len(set(starts)) != len(starts):
            return UnresolvedGame(UnresolvedReason.SAME_START_ROWS)
        return ResolvedGame(games[ordered.index(row)].game_pk, ResolutionRule.START_ORDER)

    if len(unnumbered) == 1:
        pending = [g for g in games if g.state is MlbGameState.SCHEDULED]
        started = [g for g in games if g.state in _STARTED]
        if (
            len(pending) == 1
            and len(started) == len(games) - 1
            and all(g.start_time + STARTED_SIBLING_MARGIN < row.start for g in started)
        ):
            return ResolvedGame(pending[0].game_pk, ResolutionRule.SIBLING_STARTED)

    # A TBD slot's placeholder start can sit beside game 1, so it makes a match ambiguous but
    # never matches on its own.
    near = [s for s in slots if abs(s.start_time - row.start) <= START_MATCH_WINDOW]
    near_pks = {s.game_pk for s in near}
    if len(near_pks) == 1 and not any(s.start_time_tbd for s in near):
        return ResolvedGame(near_pks.pop(), ResolutionRule.START_TIME)

    return UnresolvedGame(
        UnresolvedReason.AMBIGUOUS_DOUBLEHEADER,
        f"{len(unnumbered)} rows for {len(games)} games",
    )


async def resolve_batch(rows: Sequence[GameRow], fetcher: MlbStatsFetcher) -> list[GameResolution]:
    """Fetch the schedule over the MLB dates ``rows`` span in one range request and resolve them.

    Logs ``mlb_game_unresolved`` for each row left unresolved. Raises ``MlbScheduleUnavailable``
    when the request fails after the fetcher's retries, the response is not a schedule, or it
    lists no games at all for a non-empty batch.
    """
    if not rows:
        return []
    days = [mlb_date(row.start) for row in rows]
    try:
        schedule = await fetcher.fetch_results(min(days), max(days))
    except (httpx.HTTPError, ValueError) as e:
        raise MlbScheduleUnavailable(str(e)) from e
    if not schedule:
        raise MlbScheduleUnavailable(f"no games scheduled from {min(days)} to {max(days)}")
    resolutions = resolve_game_pks(rows, schedule)
    for row, resolution in zip(rows, resolutions, strict=True):
        if isinstance(resolution, UnresolvedGame):
            logger.warning(
                "mlb_game_unresolved",
                home_team=row.home_team,
                away_team=row.away_team,
                start=row.start.isoformat(),
                game_number=row.game_number,
                reason=resolution.reason,
                detail=resolution.detail,
            )
    return resolutions
