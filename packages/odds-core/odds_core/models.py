"""SQLModel database schema definitions."""

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import JSON, Column, DateTime, Index
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    """
    Return current UTC time as timezone-aware datetime.

    The system stores all datetimes as timezone-aware UTC for type safety and best practices.
    Using datetime.now(UTC) instead of deprecated datetime.utcnow().
    """
    return datetime.now(UTC)


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
    commence_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True), description="Game start time"
    )
    home_team: str = Field(index=True, description="Home team name")
    away_team: str = Field(index=True, description="Away team name")
    status: EventStatus = Field(default=EventStatus.SCHEDULED, description="Event status")

    # Results (populated after game completion)
    home_score: int | None = Field(default=None, description="Final home team score")
    away_score: int | None = Field(default=None, description="Final away team score")
    completed_at: datetime | None = Field(
        sa_column=Column(DateTime(timezone=True)), default=None, description="Completion timestamp"
    )

    # Metadata
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
        description="Record creation time",
    )
    updated_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
        description="Record last update time",
    )


class OddsSnapshot(SQLModel, table=True):
    """Raw odds snapshot preserving complete API response."""

    __tablename__ = "odds_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True, description="Event reference")
    snapshot_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
        description="Time of snapshot capture",
    )

    # Full API response stored as JSON
    raw_data: dict = Field(sa_column=Column(JSON), description="Complete API response")

    # Quick statistics
    bookmaker_count: int = Field(description="Number of bookmakers in snapshot")
    api_request_id: str | None = Field(default=None, description="API request ID for debugging")

    # Fetch tier tracking (for adaptive sampling validation and ML features)
    fetch_tier: str | None = Field(
        default=None,
        index=True,
        description="Fetch tier: opening, early, sharp, pregame, closing",
    )
    hours_until_commence: float | None = Field(
        default=None, description="Hours between snapshot and game start"
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
        description="Record creation time",
    )

    __table_args__ = (
        Index("ix_event_snapshot_time", "event_id", "snapshot_time"),
        Index("ix_event_tier", "event_id", "fetch_tier"),
    )


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
    odds_timestamp: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True), description="When odds were valid"
    )
    last_update: datetime = Field(
        sa_column=Column(DateTime(timezone=True)), description="Bookmaker's last update time"
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
        description="Record creation time",
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
        sa_column=Column(DateTime(timezone=True), index=True),
        default_factory=utc_now,
        description="Issue timestamp",
    )


class FetchLog(SQLModel, table=True):
    """API fetch operation logging."""

    __tablename__ = "fetch_logs"

    id: int | None = Field(default=None, primary_key=True)
    fetch_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
        default_factory=utc_now,
        description="Fetch timestamp",
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


class AlertHistory(SQLModel, table=True):
    """Alert deduplication tracking."""

    __tablename__ = "alert_history"

    id: int | None = Field(default=None, primary_key=True)

    alert_type: str = Field(
        index=True, description="Alert type: quota_low, stale_data, consecutive_failures, etc."
    )
    severity: str = Field(description="Severity: info, warning, error, critical")
    message: str = Field(description="Alert message content")

    sent_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
        default_factory=utc_now,
        description="Alert sent timestamp",
    )

    # Context data for debugging
    context: dict | None = Field(
        sa_column=Column(JSON), default=None, description="Additional context data"
    )

    __table_args__ = (Index("ix_alert_type_sent_at", "alert_type", "sent_at"),)
