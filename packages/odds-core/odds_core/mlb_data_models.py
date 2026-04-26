"""MLB supplementary data tables: probable pitchers."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Column, DateTime, UniqueConstraint, func
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now

# ---------------------------------------------------------------------------
# Typed ingest records (fetcher → writer boundary)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MlbProbablePitchersRecord:
    """Parsed MLB probable-pitchers record ready for database storage.

    One record per game per fetch. ``home_pitcher_*`` and ``away_pitcher_*``
    are nullable: MLBAM omits the ``probablePitcher`` object until announced,
    and the null itself is the signal we want to capture point-in-time.
    """

    game_pk: int
    commence_time: datetime
    fetched_at: datetime
    home_team: str
    away_team: str
    game_type: str
    home_pitcher_name: str | None = None
    home_pitcher_id: int | None = None
    away_pitcher_name: str | None = None
    away_pitcher_id: int | None = None


class MlbProbablePitchers(SQLModel, table=True):
    """Snapshot of MLBAM's probable-pitcher state for a single game.

    Append-only: each fetch inserts a new row keyed on ``(game_pk, fetched_at)``.
    MLBAM only exposes the current-state probable pitcher; the only way to
    reconstruct the announcement timeline is to write every snapshot.
    """

    __tablename__ = "mlb_probable_pitchers"

    id: int | None = Field(default=None, primary_key=True)

    game_pk: int = Field(index=True)
    commence_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
    )
    fetched_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
    )

    home_team: str = Field()
    away_team: str = Field()
    game_type: str = Field(default="R")

    home_pitcher_name: str | None = Field(default=None)
    home_pitcher_id: int | None = Field(default=None)
    away_pitcher_name: str | None = Field(default=None)
    away_pitcher_id: int | None = Field(default=None)

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
        default_factory=utc_now,
    )

    __table_args__ = (
        UniqueConstraint(
            "game_pk",
            "fetched_at",
            name="uq_mlb_probable_pitchers_game_fetched",
        ),
    )
