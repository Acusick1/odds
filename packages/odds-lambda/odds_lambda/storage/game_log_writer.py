"""Database write operations for NBA team game log data."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import structlog
from odds_core.game_log_models import NbaTeamGameLog
from odds_core.models import Event
from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.game_log_fetcher import GameLogRecord
from odds_lambda.polymarket_matching import normalize_team

logger = structlog.get_logger(__name__)

# nba_api abbreviation → canonical team name (from nba_api.stats.static.teams)
# Hardcoded to avoid importing nba_api at write time.
_ABBREV_TO_CANONICAL: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


class GameLogWriter:
    """Handles all write operations for game log data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_game_logs(self, records: list[GameLogRecord]) -> int:
        """Insert or update game log records with automatic event matching.

        Uses PostgreSQL ON CONFLICT DO UPDATE on the composite unique constraint
        for idempotent upserts. Event matching is performed per unique
        (team, game_date) pair and cached within the batch.

        Returns:
            Number of rows upserted.
        """
        if not records:
            return 0

        event_cache: dict[tuple[str, date], str | None] = {}

        log_dicts: list[dict] = []
        for record in records:
            canonical = _ABBREV_TO_CANONICAL.get(record.team_abbreviation)
            if canonical is None:
                canonical = normalize_team(record.team_name)
            cache_key = (canonical or record.team_name, record.game_date)

            if cache_key not in event_cache:
                event_cache[cache_key] = await self._match_event(cache_key[0], record.game_date)

            log_dicts.append(
                {
                    "nba_game_id": record.nba_game_id,
                    "team_id": record.team_id,
                    "team_abbreviation": record.team_abbreviation,
                    "game_date": record.game_date,
                    "matchup": record.matchup,
                    "wl": record.wl,
                    "season": record.season,
                    "pts": record.pts,
                    "fgm": record.fgm,
                    "fga": record.fga,
                    "fg3m": record.fg3m,
                    "fg3a": record.fg3a,
                    "ftm": record.ftm,
                    "fta": record.fta,
                    "oreb": record.oreb,
                    "dreb": record.dreb,
                    "reb": record.reb,
                    "ast": record.ast,
                    "stl": record.stl,
                    "blk": record.blk,
                    "tov": record.tov,
                    "pf": record.pf,
                    "plus_minus": record.plus_minus,
                    "event_id": event_cache[cache_key],
                }
            )

        # asyncpg limits query parameters to 32,767.  Each row has ~24 columns,
        # so batch to stay well under the limit.
        batch_size = 1000
        for i in range(0, len(log_dicts), batch_size):
            batch = log_dicts[i : i + batch_size]
            stmt = insert(NbaTeamGameLog).values(batch)
            set_ = {
                col.name: stmt.excluded[col.name]
                for col in NbaTeamGameLog.__table__.columns
                if col.name not in ("id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_game_log_game_team",
                set_=set_,
            )
            await self.session.execute(stmt)
        await self.session.flush()

        matched = sum(1 for v in event_cache.values() if v is not None)
        logger.info(
            "game_logs_upserted",
            count=len(log_dicts),
            events_matched=matched,
            events_unmatched=len(event_cache) - matched,
        )
        return len(log_dicts)

    async def _match_event(self, team_name: str, game_date: date) -> str | None:
        """Match a team + game_date to a sportsbook Event record.

        Uses the same ET window logic as InjuryWriter: game_date 10:00 ET
        to game_date+1 06:00 ET to handle timezone rollovers.

        Returns:
            event_id if exactly one match found, None otherwise.
        """
        from odds_core.time import EASTERN

        day_start_et = datetime(game_date.year, game_date.month, game_date.day, 10, tzinfo=EASTERN)
        day_end_et = day_start_et + timedelta(hours=20)
        window_start = day_start_et.astimezone(UTC)
        window_end = day_end_et.astimezone(UTC)

        query = select(Event.id).where(
            and_(
                Event.commence_time >= window_start,
                Event.commence_time <= window_end,
                or_(Event.home_team == team_name, Event.away_team == team_name),
            )
        )
        result = await self.session.execute(query)
        candidates = list(result.scalars().all())

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            logger.warning(
                "game_log_event_ambiguous_match",
                team=team_name,
                game_date=str(game_date),
                candidates=len(candidates),
            )
        return None
