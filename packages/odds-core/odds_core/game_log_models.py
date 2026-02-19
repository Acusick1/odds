"""NBA team game log database schema definitions."""

from datetime import date, datetime

from sqlalchemy import Column, Date, DateTime, Index, UniqueConstraint
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class NbaTeamGameLog(SQLModel, table=True):
    """Single team's box score for one NBA game.

    Two rows per game (one per team). Sourced from stats.nba.com
    LeagueGameFinder endpoint via Playwright.
    """

    __tablename__ = "nba_team_game_logs"

    id: int | None = Field(default=None, primary_key=True)

    # Game identification
    nba_game_id: str = Field(description="NBA game ID e.g. '0022400123'")
    team_id: int = Field(description="NBA team ID (numeric)")
    team_abbreviation: str = Field(description="Team abbreviation e.g. 'BOS'")
    game_date: date = Field(
        sa_column=Column(Date),
        description="Game date",
    )
    matchup: str = Field(description="Matchup string e.g. 'BOS vs. NYK'")
    wl: str | None = Field(default=None, description="Win/Loss indicator")
    season: str = Field(description="Season string e.g. '2024-25'")

    # Box score stats (all nullable for in-progress or missing data)
    pts: int | None = Field(default=None)
    fgm: int | None = Field(default=None)
    fga: int | None = Field(default=None)
    fg3m: int | None = Field(default=None)
    fg3a: int | None = Field(default=None)
    ftm: int | None = Field(default=None)
    fta: int | None = Field(default=None)
    oreb: int | None = Field(default=None)
    dreb: int | None = Field(default=None)
    reb: int | None = Field(default=None)
    ast: int | None = Field(default=None)
    stl: int | None = Field(default=None)
    blk: int | None = Field(default=None)
    tov: int | None = Field(default=None)
    pf: int | None = Field(default=None)
    plus_minus: int | None = Field(default=None)

    # Event linking (auto-matched at write time via team + game_date)
    event_id: str | None = Field(
        default=None,
        foreign_key="events.id",
        index=True,
        description="Linked sportsbook event",
    )

    # Metadata
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
        description="Record creation time",
    )

    __table_args__ = (
        UniqueConstraint(
            "nba_game_id",
            "team_id",
            name="uq_game_log_game_team",
        ),
        Index("ix_game_log_team_date", "team_abbreviation", "game_date"),
        Index("ix_game_log_season", "season"),
    )
