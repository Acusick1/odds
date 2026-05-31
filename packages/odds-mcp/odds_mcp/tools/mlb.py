"""MLB-specific MCP tools.

Mounted onto the parent ``odds-mcp`` server in :mod:`odds_mcp.server` without a
namespace, so tool names are exposed verbatim to clients.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
import structlog
from fastmcp import FastMCP
from odds_core.database import async_session_maker
from odds_core.mlb_data_models import MlbProbablePitchersRecord, select_latest_in_window
from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher, dates_for_window
from odds_lambda.storage.mlb_pitcher_reader import MlbPitcherReader
from pydantic import BaseModel

logger = structlog.get_logger()

mlb_mcp = FastMCP("odds-mcp-mlb")

FetchStatus = Literal["live", "stale_db_only", "db_only"]


class ProbablePitcherGame(BaseModel):
    """One game's probable-pitcher state in the ``get_probable_pitchers`` response."""

    game_pk: int
    commence_time: str
    home_team: str
    away_team: str
    home_pitcher_name: str | None
    home_pitcher_id: int | None
    away_pitcher_name: str | None
    away_pitcher_id: int | None
    fetched_at: str
    hours_until_commence: float

    @classmethod
    def from_record(cls, record: MlbProbablePitchersRecord, now: datetime) -> ProbablePitcherGame:
        """Serialize a domain record into the response shape, stamping freshness."""
        hours_until = (record.commence_time - now).total_seconds() / 3600.0
        return cls(
            game_pk=record.game_pk,
            commence_time=record.commence_time.isoformat(),
            home_team=record.home_team,
            away_team=record.away_team,
            home_pitcher_name=record.home_pitcher_name,
            home_pitcher_id=record.home_pitcher_id,
            away_pitcher_name=record.away_pitcher_name,
            away_pitcher_id=record.away_pitcher_id,
            fetched_at=record.fetched_at.isoformat(),
            hours_until_commence=round(hours_until, 3),
        )


class ProbablePitchersResponse(BaseModel):
    """Validated contract for the ``get_probable_pitchers`` tool output."""

    fetched_at: str
    lookahead_hours: int
    fetch_status: FetchStatus
    game_count: int
    games: list[ProbablePitcherGame]

    @classmethod
    def from_records(
        cls,
        records: list[MlbProbablePitchersRecord],
        now: datetime,
        lookahead_hours: int,
        fetch_status: FetchStatus,
    ) -> ProbablePitchersResponse:
        """Build the response from already-selected, ordered domain records."""
        games = [ProbablePitcherGame.from_record(r, now) for r in records]
        return cls(
            fetched_at=now.isoformat(),
            lookahead_hours=lookahead_hours,
            fetch_status=fetch_status,
            game_count=len(games),
            games=games,
        )


@mlb_mcp.tool()
async def get_probable_pitchers(
    lookahead_hours: int = 48,
    refresh: bool = True,
) -> dict[str, Any]:
    """Return announced probable starting pitchers for upcoming MLB games.

    This tool is read-only: it never writes to ``mlb_probable_pitchers``. The
    ``fetch-mlb-probables`` cron owns the snapshot table.

    By default (``refresh=True``), it read-throughs live: hits the MLB Stats
    API for the dates covering ``[now, now + lookahead_hours]`` and returns the
    current state per game *without persisting it*. This keeps the agent
    late-scratch-aware near game time. On HTTP failure it falls back to the
    cached snapshot table so a transient MLBAM outage doesn't blank the slate.

    Pass ``refresh=False`` to skip the live fetch entirely and read straight
    from the cached snapshot table — useful for a quick lookup when the cron
    (or a recent live call) has already populated current data.

    ``fetch_status`` distinguishes the three modes:

    - ``"live"`` — successful live fetch; current state returned, nothing written.
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
        refresh: When True (default), live-fetch from MLBAM and return the
            current state per game without persisting. When False, skip the
            fetch and return the latest cached snapshot per game.

    Returns:
        Dict with ``fetched_at``, ``lookahead_hours``, ``fetch_status``
        (``"live"`` | ``"stale_db_only"`` | ``"db_only"``), and ``games``
        (a list ordered by ``commence_time`` ascending; each entry carries
        its own ``fetched_at`` indicating when that row was last refreshed).
    """
    lookahead_hours = max(1, min(int(lookahead_hours), 168))
    now = datetime.now(UTC)
    end = now + timedelta(hours=lookahead_hours)

    fetch_status: FetchStatus
    if refresh:
        target_dates = dates_for_window(now, lookahead_hours)
        try:
            async with MlbStatsFetcher() as fetcher:
                records = await fetcher.fetch_dates(target_dates, fetched_at=now)
            selected = select_latest_in_window(records, now, end)
            response = ProbablePitchersResponse.from_records(selected, now, lookahead_hours, "live")
            return response.model_dump()
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
        reader = MlbPitcherReader(session)
        records = await reader.get_latest_in_window(now, end)

    response = ProbablePitchersResponse.from_records(records, now, lookahead_hours, fetch_status)
    return response.model_dump()
