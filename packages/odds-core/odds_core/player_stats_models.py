"""NBA player season statistics database schema definitions."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Index, UniqueConstraint
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class NbaPlayerSeasonStats(SQLModel, table=True):
    """Player season-level statistics from PBPStats API.

    One row per player per season. Traded players appear under their
    final team (PBPStats aggregates across teams).
    """

    __tablename__ = "nba_player_season_stats"

    id: int | None = Field(default=None, primary_key=True)

    # Identity
    player_id: int = Field(description="PBPStats entity ID")
    player_name: str = Field(description="Player name in 'Last, First' format")
    team_id: int = Field(description="NBA team ID (final team for traded players)")
    team_abbreviation: str = Field(description="Team abbreviation e.g. 'BOS'")
    season: str = Field(description="Season string e.g. '2024-25'")

    # Impact
    minutes: float = Field(description="Total minutes played")
    games_played: int = Field(description="Games played")
    on_off_rtg: float | None = Field(default=None, description="On-court offensive rating")
    on_def_rtg: float | None = Field(default=None, description="On-court defensive rating")
    usage: float | None = Field(default=None, description="Usage rate")

    # Efficiency
    ts_pct: float | None = Field(default=None, description="True shooting percentage")
    efg_pct: float | None = Field(default=None, description="Effective field goal percentage")

    # Playmaking
    assists: int = Field(description="Total assists")
    turnovers: int = Field(description="Total turnovers")

    # Defense / rebounding
    rebounds: int = Field(description="Total rebounds")
    steals: int = Field(description="Total steals")
    blocks: int = Field(description="Total blocks")

    # Volume
    points: int = Field(description="Total points")
    plus_minus: float = Field(description="Plus/minus")

    # Metadata
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), default=utc_now),
    )

    __table_args__ = (
        UniqueConstraint(
            "player_id",
            "season",
            name="uq_player_season_stats_player_season",
        ),
        Index("ix_player_season_stats_team", "team_abbreviation", "season"),
        Index("ix_player_season_stats_season", "season"),
    )
