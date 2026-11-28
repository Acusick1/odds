"""EPL-specific MCP tools.

Mounted onto the parent ``odds-mcp`` server in :mod:`odds_mcp.server` without a
namespace, so tool names are exposed verbatim to clients.
"""

from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP
from odds_analytics.standings_features import TeamRecord, rank_table
from odds_core.database import async_session_maker
from odds_core.epl_data_models import EspnGame
from odds_core.team import normalize_team_name
from odds_lambda.espn_game_fetcher import current_season, season_label
from sqlalchemy import select

epl_mcp = FastMCP("odds-mcp-epl")


# ---------------------------------------------------------------------------
# ESPN game context helpers (used by ``get_team_context``)
# ---------------------------------------------------------------------------


# ESPN's canonical match-state enum (``status.type.state``). Used for
# categorising games as past/live/upcoming.
_ESPN_STATE_POST = "post"  # completed match (any kind: FT, AET, after pens)
_ESPN_STATE_IN = "in"  # live match (any in-play status)

# Fallback set for rows written before the ``state`` column was introduced
# (``state is None``). Values are ESPN's ``status.type.description`` strings
# that indicate a completed match. Kept narrow — only strings confirmed in
# live ESPN responses. Matches are intentionally exact; partial matches
# (e.g. "Final Score") would risk false positives.
_ESPN_FINAL_STATUS_FALLBACK: frozenset[str] = frozenset(
    {
        "Full Time",
        "Final Score - After Penalties",
        "Final Score - After Extra Time",
        "Final",
    }
)

_PREMIER_LEAGUE = "Premier League"


def _is_final(row: EspnGame) -> bool:
    """Return True if ``row`` represents a completed match.

    Prefers ``state == "post"`` when available; falls back to a curated set of
    ``status`` description strings for rows written before the migration.
    """
    if row.state is not None:
        return row.state == _ESPN_STATE_POST
    return row.status in _ESPN_FINAL_STATUS_FALLBACK


def _is_in_progress(row: EspnGame) -> bool:
    """Return True if ``row`` is currently live.

    Only the ``state`` column can distinguish "in" vs "pre" reliably, so rows
    with ``state is None`` are treated as not-in-progress (the old "In
    Progress" status string was never persisted on historical rows anyway).
    """
    return row.state == _ESPN_STATE_IN


def _parse_score(value: str) -> int | None:
    """Parse an ESPN score string (may be blank or non-numeric).

    Returns ``None`` when the value cannot be interpreted as a non-negative int.
    """
    if not value:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _outcome_for_score(score_team: int, score_opponent: int) -> str:
    if score_team > score_opponent:
        return "W"
    if score_team < score_opponent:
        return "L"
    return "D"


def _game_to_dict(game: EspnGame, *, as_of: datetime) -> dict[str, Any]:
    """Serialise an EspnGame for the upcoming-games view."""
    days_until = (game.date - as_of).total_seconds() / 86400.0
    return {
        "date": game.date.isoformat(),
        "opponent": game.opponent,
        "home_away": game.home_away,
        "competition": game.competition,
        "status": game.status,
        "days_until_ko": round(days_until, 3),
    }


def _result_to_dict(game: EspnGame) -> dict[str, Any] | None:
    """Serialise an EspnGame into a last-results entry.

    Returns ``None`` when the row does not have parseable integer scores (e.g.
    status is Final but scores are blank/malformed — an ESPN quirk).
    """
    score_team = _parse_score(game.score_team)
    score_opponent = _parse_score(game.score_opponent)
    if score_team is None or score_opponent is None:
        return None
    return {
        "date": game.date.isoformat(),
        "opponent": game.opponent,
        "home_away": game.home_away,
        "score_team": score_team,
        "score_opponent": score_opponent,
        "outcome": _outcome_for_score(score_team, score_opponent),
        "competition": game.competition,
    }


def _derive_standings(games: list[EspnGame]) -> dict[str, dict[str, Any]]:
    """Build a points table keyed by team from a season's Premier League rows.

    The input is the set of EspnGame rows for the target season. Only rows
    where ``competition == "Premier League"`` and state is ``"post"`` (or, for
    legacy rows with no state, a matching fallback status) with parseable
    scores count. Each match contributes to both teams' rows (rows are per team
    in ``espn_games``). Positions follow ``rank_table``.
    """
    records: dict[str, TeamRecord] = {}

    for game in games:
        if game.competition != _PREMIER_LEAGUE:
            continue
        if not _is_final(game):
            continue
        score_team = _parse_score(game.score_team)
        score_opponent = _parse_score(game.score_opponent)
        if score_team is None or score_opponent is None:
            continue
        records.setdefault(game.team, TeamRecord(team=game.team)).record_result(
            score_team, score_opponent
        )

    return {
        record.team: {
            "team": record.team,
            "position": position,
            "played": record.played,
            "wins": record.won,
            "draws": record.drawn,
            "losses": record.lost,
            "goals_for": record.goals_for,
            "goals_against": record.goals_against,
            "goal_diff": record.goal_difference,
            "points": record.points,
        }
        for position, record in enumerate(rank_table(records.values()), start=1)
    }


@epl_mcp.tool()
async def get_team_context(
    team: str,
    as_of: str | None = None,
    last_n: int = 5,
    next_n: int = 5,
    include_standings: bool = True,
) -> dict[str, Any]:
    """Fetch form, upcoming games, and league table position for an EPL team.

    Draws from the ``espn_games`` table (refreshed daily by the
    ``fetch-espn-games`` job). Returns:

    - ``last_results``: up to ``last_n`` most recent Premier-League-only
      completed games before ``as_of``, each with score, outcome (W/D/L),
      home/away. PL-only to keep "form" meaningful — cup results sit in
      upcoming_games via rotation risk, not form.
    - ``upcoming_games``: next ``next_n`` games after ``as_of`` across
      all competitions (PL, FA Cup, League Cup, European) with
      ``days_until_ko``. This is the rotation-risk signal.
    - ``standings`` (when ``include_standings=True``): derived on the fly from
      Premier League completed rows in the current season. Position assigned
      by a simplified tiebreak chain: points, goal diff, goals for, team name.
      This omits the official EPL head-to-head step, so ``position`` on a
      tight 3-way tie at the threshold is approximate — do not over-trust it.

    Live (``state="in"``) rows are skipped from both last_results and
    upcoming_games; they neither count as form nor as upcoming rotation.

    Args:
        team: Canonical team name (matches the output of
            ``odds_core.team.normalize_team_name`` — e.g. "Manchester Utd",
            "Tottenham", "Wolves"). Common variants are also accepted.
        as_of: ISO datetime cutoff. Defaults to now. Used to split games
            into past vs upcoming; also used to compute ``days_until_ko``.
        last_n: Max last-results entries to return. Clamped to >= 1.
        next_n: Max upcoming-games entries to return. Clamped to >= 1.
        include_standings: When True, include the derived table. The returned
            ``standings.team_row`` is the target team's row; ``standings.table``
            is the full 20-team ordering.

    Returns:
        Dict with ``team``, ``as_of``, ``season``, ``last_results``,
        ``upcoming_games``, and (optionally) ``standings``.
    """
    canonical_team = normalize_team_name(team)

    if as_of is None:
        resolved_as_of = datetime.now(UTC)
    else:
        parsed = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        resolved_as_of = parsed

    last_n = max(1, last_n)
    next_n = max(1, next_n)

    season = current_season(resolved_as_of)
    season_lbl = season_label(season)

    async with async_session_maker() as session:
        # Team-scoped rows: drive last_results + upcoming_games.
        team_query = select(EspnGame).where(EspnGame.team == canonical_team).order_by(EspnGame.date)
        team_result = await session.execute(team_query)
        team_rows = list(team_result.scalars().all())

        # Season-scoped rows for all teams: drive standings derivation.
        standings_rows: list[EspnGame] = []
        if include_standings:
            standings_query = (
                select(EspnGame).where(EspnGame.season == season_lbl).order_by(EspnGame.date)
            )
            standings_result = await session.execute(standings_query)
            standings_rows = list(standings_result.scalars().all())

    # Split team_rows into past (final, PL only) and upcoming (any competition,
    # not started). Live matches ("in") are skipped on both sides.
    last_results: list[dict[str, Any]] = []
    upcoming: list[dict[str, Any]] = []
    for row in team_rows:
        if _is_in_progress(row):
            continue
        if _is_final(row):
            if row.competition != _PREMIER_LEAGUE:
                continue
            if row.date >= resolved_as_of:
                continue
            entry = _result_to_dict(row)
            if entry is not None:
                last_results.append(entry)
        elif row.date >= resolved_as_of:
            # "pre" rows and unknown-state future-dated rows.
            upcoming.append(_game_to_dict(row, as_of=resolved_as_of))

    # Newest-first on last_results; already date-asc on upcoming.
    last_results.reverse()
    last_results = last_results[:last_n]
    upcoming_games = upcoming[:next_n]

    response: dict[str, Any] = {
        "team": canonical_team,
        "as_of": resolved_as_of.isoformat(),
        "season": season_lbl,
        "last_results": last_results,
        "upcoming_games": upcoming_games,
    }

    if include_standings:
        table = _derive_standings(standings_rows)
        team_row = table.get(canonical_team)
        response["standings"] = {
            "team_row": team_row,
            "table": sorted(table.values(), key=lambda r: r["position"]),
        }

    return response
