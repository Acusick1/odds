"""EPL supplementary data tables: ESPN fixtures, ESPN lineups, FPL availability, API-Football."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Index, UniqueConstraint, func
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now

# ---------------------------------------------------------------------------
# Typed ingest records (fetcher → writer boundary)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EspnFixtureRecord:
    """Parsed ESPN fixture record ready for database storage."""

    date: datetime
    team: str
    opponent: str
    competition: str
    match_round: str
    home_away: str
    score_team: str
    score_opponent: str
    status: str
    season: str


@dataclass(slots=True)
class EspnLineupRecord:
    """Parsed ESPN lineup record ready for database storage."""

    date: datetime
    home_team: str
    away_team: str
    team: str
    player_id: str
    player_name: str
    position: str
    starter: bool
    formation_place: int
    season: str


@dataclass(slots=True)
class FplAvailabilityRecord:
    """Parsed FPL availability record ready for database storage."""

    snapshot_time: datetime
    gameweek: int
    season: str
    player_code: int
    player_name: str
    team: str
    position: str
    chance_of_playing: float
    status: str
    news: str | None


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
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
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
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
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
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
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


class ApiFootballLineup(SQLModel, table=True):
    """Confirmed lineup from API-Football for one team in a fixture.

    One row per (fixture_id, team). Stores the full lineup response including
    formation, starting XI, substitutes, and coach as structured JSON.
    """

    __tablename__ = "api_football_lineups"

    id: int | None = Field(default=None, primary_key=True)

    event_id: str = Field(index=True)
    fixture_id: int = Field(index=True)
    team_name: str = Field()
    team_id: int = Field()
    formation: str | None = Field(default=None)
    coach: dict[str, Any] | None = Field(sa_column=Column(JSON), default=None)
    start_xi: list[dict[str, Any]] = Field(sa_column=Column(JSON))
    substitutes: list[dict[str, Any]] = Field(sa_column=Column(JSON))

    fetched_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
        default_factory=utc_now,
    )

    __table_args__ = (
        UniqueConstraint("fixture_id", "team_id", name="uq_api_football_lineup_fixture_team"),
        Index("ix_api_football_lineup_event", "event_id"),
    )
