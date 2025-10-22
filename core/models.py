"""SQLModel database schema definitions."""

from datetime import datetime
from enum import Enum

from pydantic import field_validator
from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, SQLModel


class EventStatus(str, Enum):
    """Event status enumeration."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINAL = "final"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"


class Event(SQLModel, table=True):
    """Event (game) model."""

    __tablename__ = "events"

    # Primary identification
    id: str = Field(primary_key=True, description="API event ID")
    sport_key: str = Field(index=True, description="Sport identifier")
    sport_title: str = Field(description="Sport display name")

    # Event details
    commence_time: datetime = Field(index=True, description="Game start time")
    home_team: str = Field(index=True, description="Home team name")
    away_team: str = Field(index=True, description="Away team name")
    status: EventStatus = Field(default=EventStatus.SCHEDULED, description="Event status")

    # Results (populated after game completion)
    home_score: int | None = Field(default=None, description="Final home team score")
    away_score: int | None = Field(default=None, description="Final away team score")
    completed_at: datetime | None = Field(default=None, description="Completion timestamp")

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Record creation time"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Record last update time"
    )

    @field_validator("commence_time", mode="before")
    def parse_commence_time(cls, v):
        """Parse commence_time from ISO string to naive UTC datetime."""
        if isinstance(v, str):
            v = v.replace("Z", "+00:00")
            v = datetime.fromisoformat(v).replace(tzinfo=None)
        return v


class OddsSnapshot(SQLModel, table=True):
    """Raw odds snapshot preserving complete API response."""

    __tablename__ = "odds_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True, description="Event reference")
    snapshot_time: datetime = Field(index=True, description="Time of snapshot capture")

    # Full API response stored as JSON
    raw_data: dict = Field(sa_column=Column(JSON), description="Complete API response")

    # Quick statistics
    bookmaker_count: int = Field(description="Number of bookmakers in snapshot")
    api_request_id: str | None = Field(default=None, description="API request ID for debugging")

    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Record creation time"
    )

    __table_args__ = (Index("ix_event_snapshot_time", "event_id", "snapshot_time"),)


class Odds(SQLModel, table=True):
    """Normalized odds record for efficient querying."""

    __tablename__ = "odds"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True, description="Event reference")

    # Bookmaker information
    bookmaker_key: str = Field(index=True, description="Bookmaker identifier")
    bookmaker_title: str = Field(description="Bookmaker display name")
    market_key: str = Field(index=True, description="Market type: h2h, spreads, totals")

    # Outcome data
    outcome_name: str = Field(description="Team name or Over/Under")
    price: int = Field(description="American odds (e.g., -110, +150)")
    point: float | None = Field(default=None, description="Spread/total line (e.g., -2.5, 218.5)")

    # Timestamps
    odds_timestamp: datetime = Field(index=True, description="When odds were valid")
    last_update: datetime = Field(description="Bookmaker's last update time")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Record creation time"
    )

    # Data quality
    is_valid: bool = Field(default=True, description="Validation flag")
    validation_notes: str | None = Field(default=None, description="Validation issues")

    __table_args__ = (
        Index("ix_event_bookmaker_market", "event_id", "bookmaker_key", "market_key"),
        Index("ix_bookmaker_time", "bookmaker_key", "odds_timestamp"),
    )


class DataQualityLog(SQLModel, table=True):
    """Data quality issue logging."""

    __tablename__ = "data_quality_logs"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str | None = Field(foreign_key="events.id", description="Related event ID")

    severity: str = Field(description="Severity: warning, error, critical")
    issue_type: str = Field(description="Issue type: missing_data, suspicious_odds, etc.")
    description: str = Field(description="Human-readable issue description")
    raw_data: dict | None = Field(sa_column=Column(JSON), default=None, description="Context data")

    created_at: datetime = Field(
        default_factory=datetime.utcnow, index=True, description="Issue timestamp"
    )


class FetchLog(SQLModel, table=True):
    """API fetch operation logging."""

    __tablename__ = "fetch_logs"

    id: int | None = Field(default=None, primary_key=True)
    fetch_time: datetime = Field(
        default_factory=datetime.utcnow, index=True, description="Fetch timestamp"
    )

    sport_key: str = Field(description="Sport that was fetched")
    events_count: int = Field(description="Number of events fetched")
    bookmakers_count: int = Field(description="Number of bookmakers in response")

    success: bool = Field(description="Whether fetch succeeded")
    error_message: str | None = Field(default=None, description="Error message if failed")

    # API quota tracking
    api_quota_remaining: int | None = Field(
        default=None, description="Remaining API quota after request"
    )
    response_time_ms: int | None = Field(default=None, description="API response time")
