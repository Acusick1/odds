"""NBA injury report database schema definitions."""

from datetime import date, datetime
from enum import Enum

from sqlalchemy import Column, Date, DateTime, Index, UniqueConstraint
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class InjuryStatus(str, Enum):
    """NBA injury report player status."""

    OUT = "OUT"
    QUESTIONABLE = "QUESTIONABLE"
    DOUBTFUL = "DOUBTFUL"
    PROBABLE = "PROBABLE"
    AVAILABLE = "AVAILABLE"


class InjuryReport(SQLModel, table=True):
    """Single player injury entry from an NBA injury report snapshot."""

    __tablename__ = "nba_injury_reports"

    id: int | None = Field(default=None, primary_key=True)

    # Report snapshot identification
    report_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
        description="UTC time of the report snapshot",
    )

    # Game identification
    game_date: date = Field(
        sa_column=Column(Date),
        description="Game date from the report",
    )
    game_time_et: str = Field(description="Game time in ET e.g. '07:00 PM ET'")
    matchup: str = Field(description="Matchup string e.g. 'BOS@ORL'")

    # Player identification
    team: str = Field(index=True, description="Full team name e.g. 'Boston Celtics'")
    player_name: str = Field(description="Player name in 'Last, First' format")

    # Injury details
    status: InjuryStatus = Field(description="Player injury status")
    reason: str = Field(description="Injury/illness reason")

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
            "report_time",
            "team",
            "player_name",
            "game_date",
            name="uq_injury_report_time_team_player_date",
        ),
        Index("ix_injury_event_report_time", "event_id", "report_time"),
    )
