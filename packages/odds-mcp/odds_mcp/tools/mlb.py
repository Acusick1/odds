"""MLB-specific MCP tools.

Mounted onto the parent ``odds-mcp`` server in :mod:`odds_mcp.server` without a
namespace, so tool names are exposed verbatim to clients.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from fastmcp import FastMCP
from odds_core.database import async_session_maker
from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher, dates_for_window
from odds_lambda.storage.mlb_pitcher_reader import MlbPitcherReader
from odds_lambda.storage.mlb_pitcher_writer import MlbPitcherWriter

mlb_mcp = FastMCP("odds-mcp-mlb")


@mlb_mcp.tool()
async def get_probable_pitchers(
    lookahead_hours: int = 48,
) -> dict[str, Any]:
    """Return announced probable starting pitchers for upcoming MLB games.

    Write-through-on-every-call: hits the MLB Stats API for the dates
    covering ``[now, now + lookahead_hours]``, appends a snapshot row per
    game to ``mlb_probable_pitchers``, then returns the latest row per
    ``game_pk`` whose ``commence_time`` falls in the lookahead window.

    Each returned game carries ``hours_until_commence`` so the agent can
    apply its own freshness heuristic. ``home_pitcher_*`` / ``away_pitcher_*``
    are nullable: MLBAM omits the ``probablePitcher`` object until the team
    announces, so a null is a meaningful "not yet announced" signal.

    The tool **cannot** distinguish a traditional starter from an opener or
    bulk-role — the MLB Stats API does not expose that flag. If the agent
    suspects an opener, it must web-search to confirm.

    Args:
        lookahead_hours: Hours from now to include in the response. Clamped
            to ``[1, 168]``. Default 48h covers today + tomorrow's slate.

    Returns:
        Dict with ``fetched_at``, ``lookahead_hours``, and ``games`` (a list
        ordered by ``commence_time`` ascending).
    """
    lookahead_hours = max(1, min(int(lookahead_hours), 168))
    now = datetime.now(UTC)
    end = now + timedelta(hours=lookahead_hours)

    target_dates = dates_for_window(now, lookahead_hours)

    async with MlbStatsFetcher() as fetcher:
        records = await fetcher.fetch_dates(target_dates, fetched_at=now)

    async with async_session_maker() as session:
        writer = MlbPitcherWriter(session)
        await writer.insert_snapshots(records)
        await session.commit()

        reader = MlbPitcherReader(session)
        rows = await reader.get_latest_in_window(now, end)

    games: list[dict[str, Any]] = []
    for row in rows:
        hours_until = (row.commence_time - now).total_seconds() / 3600.0
        games.append(
            {
                "game_pk": row.game_pk,
                "commence_time": row.commence_time.isoformat(),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_pitcher_name": row.home_pitcher_name,
                "home_pitcher_id": row.home_pitcher_id,
                "away_pitcher_name": row.away_pitcher_name,
                "away_pitcher_id": row.away_pitcher_id,
                "fetched_at": row.fetched_at.isoformat(),
                "hours_until_commence": round(hours_until, 3),
            }
        )

    return {
        "fetched_at": now.isoformat(),
        "lookahead_hours": lookahead_hours,
        "game_count": len(games),
        "games": games,
    }
