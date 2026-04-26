"""MLB-specific MCP tools.

Mounted onto the parent ``odds-mcp`` server in :mod:`odds_mcp.server` without a
namespace, so tool names are exposed verbatim to clients.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import structlog
from fastmcp import FastMCP
from odds_core.database import async_session_maker
from odds_core.mlb_data_models import MlbProbablePitchersRecord
from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher, dates_for_window
from odds_lambda.storage.mlb_pitcher_reader import MlbPitcherReader
from odds_lambda.storage.mlb_pitcher_writer import MlbPitcherWriter

logger = structlog.get_logger()

mlb_mcp = FastMCP("odds-mcp-mlb")


@mlb_mcp.tool()
async def get_probable_pitchers(
    lookahead_hours: int = 48,
    refresh: bool = True,
) -> dict[str, Any]:
    """Return announced probable starting pitchers for upcoming MLB games.

    By default (``refresh=True``), this is write-through-on-every-call: hits
    the MLB Stats API for the dates covering ``[now, now + lookahead_hours]``,
    appends a snapshot row per game to ``mlb_probable_pitchers``, then
    returns the latest row per ``game_pk`` whose ``commence_time`` falls in
    the lookahead window.

    Pass ``refresh=False`` to skip the live fetch entirely and read straight
    from the cached snapshot table. Useful when the agent only needs a
    quick lookup and an earlier call (or the daily backstop cron) has
    already populated the table.

    ``fetch_status`` distinguishes the three modes:

    - ``"live"`` — successful live fetch, snapshot appended, latest read.
    - ``"stale_db_only"`` — ``refresh=True`` but MLBAM HTTP failed; fell
      back to DB read so the agent can decide whether the cached data is
      fresh enough.
    - ``"db_only"`` — caller passed ``refresh=False``; no live fetch
      attempted.

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
        refresh: When True (default), live-fetch from MLBAM, append a
            snapshot, and return the latest row per game. When False, skip
            the fetch and return the latest cached snapshot per game.

    Returns:
        Dict with ``fetched_at``, ``lookahead_hours``, ``fetch_status``
        (``"live"`` | ``"stale_db_only"`` | ``"db_only"``), and ``games``
        (a list ordered by ``commence_time`` ascending; each entry carries
        its own ``fetched_at`` indicating when that row was last refreshed).
    """
    lookahead_hours = max(1, min(int(lookahead_hours), 168))
    now = datetime.now(UTC)
    end = now + timedelta(hours=lookahead_hours)

    records: list[MlbProbablePitchersRecord] = []
    if refresh:
        target_dates = dates_for_window(now, lookahead_hours)
        fetch_status = "live"
        try:
            async with MlbStatsFetcher() as fetcher:
                records = await fetcher.fetch_dates(target_dates, fetched_at=now)
        except httpx.HTTPError as e:
            logger.warning(
                "get_probable_pitchers_fetch_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            fetch_status = "stale_db_only"
    else:
        fetch_status = "db_only"

    async with async_session_maker() as session:
        if records:
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
        "fetch_status": fetch_status,
        "game_count": len(games),
        "games": games,
    }
