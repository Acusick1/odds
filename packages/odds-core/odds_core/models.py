"""SQLModel database schema definitions."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from odds_core.team import normalize_team_name
from odds_core.vocabulary import (
    EXTERNAL_ID_SOURCES,
    PRICE_SOURCES,
    HealthCheck,
    Market,
    MarketStatus,
    PriceOutcome,
    Side,
    Source,
    StrEnumText,
    TimeBasis,
    closed_set_check,
)


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
    # An MLB event with no game in MLB's schedule; its reason is a health-check acknowledgement.
    UNMATCHED = "unmatched"

    def can_become(self, target: "EventStatus") -> bool:
        """Whether a stored event in this status may be moved to ``target``."""
        return target is self or target in _STATUS_TRANSITIONS[self]


_STATUS_TRANSITIONS: dict[EventStatus, frozenset[EventStatus]] = {
    EventStatus.SCHEDULED: frozenset(
        {
            EventStatus.LIVE,
            EventStatus.FINAL,
            EventStatus.CANCELLED,
            EventStatus.POSTPONED,
            EventStatus.UNMATCHED,
        }
    ),
    EventStatus.LIVE: frozenset(
        {
            EventStatus.SCHEDULED,
            EventStatus.FINAL,
            EventStatus.CANCELLED,
            EventStatus.POSTPONED,
            EventStatus.UNMATCHED,
        }
    ),
    EventStatus.POSTPONED: frozenset(
        {EventStatus.SCHEDULED, EventStatus.FINAL, EventStatus.CANCELLED, EventStatus.UNMATCHED}
    ),
    EventStatus.CANCELLED: frozenset({EventStatus.FINAL}),
    EventStatus.FINAL: frozenset(),
    # A game found in the schedule later makes the event a scheduled one again.
    EventStatus.UNMATCHED: frozenset({EventStatus.SCHEDULED}),
}


class Event(SQLModel, table=True):
    """Event (game) model."""

    __tablename__ = "events"

    # Numbered by the database; source ids are in ``event_external_ids``.
    id: int | None = Field(default=None, sa_column=Column(BigInteger, Identity(), primary_key=True))
    sport_key: str = Field(index=True, description="Sport identifier")
    sport_title: str = Field(description="Sport display name")

    # Event details
    commence_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True), description="Game start time"
    )
    home_team: str = Field(index=True, description="Home team name")
    away_team: str = Field(index=True, description="Away team name")
    status: EventStatus = Field(default=EventStatus.SCHEDULED, description="Event status")

    def model_post_init(self, __context: Any) -> None:
        self.home_team = normalize_team_name(self.home_team)
        self.away_team = normalize_team_name(self.away_team)

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
    """One source's prices for one market of an event at one time; the prices are ``Price``
    rows and the envelope they were read from is its ``SnapshotPayload``."""

    __tablename__ = "odds_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    event_id: int = Field(
        sa_type=BigInteger, foreign_key="events.id", index=True, description="Event reference"
    )
    snapshot_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
        description="Time of snapshot capture",
    )

    source: Source | None = Field(default=None, sa_column=Column(StrEnumText(Source)))
    market: Market | None = Field(default=None, sa_column=Column(StrEnumText(Market)))
    time_basis: TimeBasis | None = Field(default=None, sa_column=Column(StrEnumText(TimeBasis)))
    in_play: bool | None = Field(default=None, sa_column=Column(Boolean))
    market_status: MarketStatus | None = Field(
        default=None, sa_column=Column(StrEnumText(MarketStatus))
    )
    total_matched: Decimal | None = Field(default=None, sa_column=Column(Numeric))
    market_start_time: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    payload_id: int | None = Field(
        default=None,
        sa_column=Column(BigInteger, ForeignKey("snapshot_payloads.id"), index=True, nullable=True),
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
        description="Record creation time",
    )

    __table_args__ = (
        Index("ix_event_snapshot_time", "event_id", "snapshot_time"),
        UniqueConstraint(
            "event_id",
            "source",
            "market",
            "snapshot_time",
            name="uq_odds_snapshots_event_source_market_time",
        ),
        closed_set_check("source", PRICE_SOURCES, "ck_odds_snapshots_source"),
        closed_set_check("market", Market, "ck_odds_snapshots_market"),
        closed_set_check("time_basis", TimeBasis, "ck_odds_snapshots_time_basis"),
        closed_set_check("market_status", MarketStatus, "ck_odds_snapshots_market_status"),
    )


class SnapshotPayload(SQLModel, table=True):
    """What a source returned, kept only to re-parse snapshots after an adapter bug."""

    __tablename__ = "snapshot_payloads"

    id: int | None = Field(default=None, sa_column=Column(BigInteger, Identity(), primary_key=True))
    payload: dict[str, Any] | list[Any] = Field(sa_column=Column(JSONB, nullable=False))
    received_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False), default_factory=utc_now
    )


class Price(SQLModel, table=True):
    """One price in a snapshot: a bookmaker's quote for one outcome, line and side."""

    __tablename__ = "prices"

    id: int | None = Field(default=None, sa_column=Column(BigInteger, Identity(), primary_key=True))
    snapshot_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer, ForeignKey("odds_snapshots.id", ondelete="CASCADE"), nullable=False
        ),
    )
    bookmaker: str = Field(sa_column=Column(Text, nullable=False))
    outcome: PriceOutcome = Field(sa_column=Column(StrEnumText(PriceOutcome), nullable=False))
    line: Decimal | None = Field(default=None, sa_column=Column(Numeric))
    side: Side = Field(sa_column=Column(StrEnumText(Side), nullable=False))
    price: Decimal = Field(sa_column=Column(Numeric, nullable=False))
    size: Decimal | None = Field(default=None, sa_column=Column(Numeric))
    matched: Decimal | None = Field(default=None, sa_column=Column(Numeric))
    changed_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))

    __table_args__ = (
        UniqueConstraint(
            "snapshot_id",
            "bookmaker",
            "outcome",
            "line",
            "side",
            name="uq_prices_snapshot_book_outcome_line_side",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint("price > 1.0", name="ck_prices_price_gt_one"),
        closed_set_check("outcome", PriceOutcome, "ck_prices_outcome"),
        closed_set_check("side", Side, "ck_prices_side"),
    )


class EventExternalId(SQLModel, table=True):
    """An id a source issued for an event: one per source per event, one event per id."""

    __tablename__ = "event_external_ids"

    source: Source = Field(sa_column=Column(StrEnumText(Source), primary_key=True))
    external_id: str = Field(sa_column=Column(Text, primary_key=True))
    event_id: int = Field(sa_column=Column(BigInteger, ForeignKey("events.id"), nullable=False))
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False), default_factory=utc_now
    )

    __table_args__ = (
        UniqueConstraint("event_id", "source", name="uq_event_external_ids_event_source"),
        closed_set_check("source", EXTERNAL_ID_SOURCES, "ck_event_external_ids_source"),
    )


class HealthCheckAcknowledgement(SQLModel, table=True):
    """An event a health check should no longer report, and why."""

    __tablename__ = "health_check_acknowledgements"

    id: int | None = Field(default=None, sa_column=Column(BigInteger, Identity(), primary_key=True))
    event_id: int = Field(sa_column=Column(BigInteger, ForeignKey("events.id"), nullable=False))
    check_name: HealthCheck = Field(sa_column=Column(StrEnumText(HealthCheck), nullable=False))
    reason: str = Field(sa_column=Column(Text, nullable=False))
    acknowledged_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False), default_factory=utc_now
    )

    __table_args__ = (
        UniqueConstraint(
            "event_id", "check_name", name="uq_health_check_acknowledgements_event_check"
        ),
        closed_set_check("check_name", HealthCheck, "ck_health_check_acknowledgements_check"),
    )


class DataQualityLog(SQLModel, table=True):
    """Data quality issue logging."""

    __tablename__ = "data_quality_logs"

    id: int | None = Field(default=None, primary_key=True)
    event_id: int | None = Field(
        sa_type=BigInteger, foreign_key="events.id", description="Related event ID"
    )

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
        index=True, description="Alert type: quota_low, stale_data, missing_heartbeat:<job>, etc."
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
