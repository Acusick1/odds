"""EPL supplementary data tables: ESPN fixtures, ESPN lineups, FPL availability."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Index, UniqueConstraint
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class EspnFixture(SQLModel, table=True):
    """Single team fixture row from ESPN all-competition schedule.

    One row per (team, date, competition) — each match appears twice (once per team).
    Covers Premier League, FA Cup, League Cup, and European competitions.
    """

    __tablename__ = "espn_fixtures"

    id: int | None = Field(default=None, primary_key=True)

    date: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
    )
    team: str = Field(index=True)
    opponent: str = Field()
    competition: str = Field()
    match_round: str = Field(default="")
    home_away: str = Field()
    score_team: str = Field(default="")
    score_opponent: str = Field(default="")
    status: str = Field(default="")
    season: str = Field(index=True)

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
    )

    __table_args__ = (
        UniqueConstraint("date", "team", "competition", name="uq_espn_fixture_date_team_comp"),
        Index("ix_espn_fixture_team_season", "team", "season"),
    )


class EspnLineup(SQLModel, table=True):
    """Single player entry from an ESPN match lineup.

    One row per player per match. Starter flag distinguishes starting XI from subs.
    """

    __tablename__ = "espn_lineups"

    id: int | None = Field(default=None, primary_key=True)

    date: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
    )
    home_team: str = Field()
    away_team: str = Field()
    team: str = Field(index=True)
    player_id: str = Field()
    player_name: str = Field()
    position: str = Field(default="")
    starter: bool = Field()
    formation_place: int = Field(default=0)
    season: str = Field(index=True)

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
    )

    __table_args__ = (
        UniqueConstraint("date", "team", "player_id", name="uq_espn_lineup_date_team_player"),
        Index("ix_espn_lineup_team_season", "team", "season"),
    )


class FplAvailability(SQLModel, table=True):
    """FPL player availability snapshot for a single gameweek.

    One row per (player, snapshot_time, gameweek). Multiple snapshots per gameweek
    capture availability changes as the deadline approaches.
    """

    __tablename__ = "fpl_availability"

    id: int | None = Field(default=None, primary_key=True)

    snapshot_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
    )
    gameweek: int = Field()
    season: str = Field(index=True)
    player_code: int = Field()
    player_name: str = Field()
    team: str = Field(index=True)
    position: str = Field()
    chance_of_playing: float = Field()
    status: str = Field(default="")
    news: str | None = Field(default=None)

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
    )

    __table_args__ = (
        UniqueConstraint(
            "snapshot_time",
            "gameweek",
            "player_code",
            "season",
            name="uq_fpl_avail_snapshot_gw_player_season",
        ),
        Index("ix_fpl_avail_team_season", "team", "season"),
    )
