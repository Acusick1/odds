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
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from odds_core.odds_math import decimal_to_american, decimal_to_implied_probability
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

    # Primary identification
    id: str = Field(primary_key=True, description="API event ID")
    # The event's key once the string ids are dropped; new tables reference it already.
    surrogate_id: int | None = Field(
        default=None, sa_column=Column(BigInteger, Identity(), nullable=False)
    )
    sport_key: str = Field(index=True, description="Sport identifier")
    sport_title: str = Field(description="Sport display name")

    # Event details
    commence_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True), description="Game start time"
    )
    home_team: str = Field(index=True, description="Home team name")
    away_team: str = Field(index=True, description="Away team name")
    status: EventStatus = Field(default=EventStatus.SCHEDULED, description="Event status")
    mlb_game_pk: int | None = Field(
        default=None, sa_column=Column(BigInteger), description="MLB Stats API gamePk"
    )
    no_game_pk_reason: str | None = Field(
        default=None,
        sa_column=Column(Text),
        description="Why an MLB event was marked as having no gamePk",
    )

    def model_post_init(self, __context: Any) -> None:
        self.home_team = normalize_team_name(self.home_team)
        self.away_team = normalize_team_name(self.away_team)

    def assign_game_pk(self, game_pk: int) -> None:
        self.mlb_game_pk = game_pk
        self.no_game_pk_reason = None

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

    __table_args__ = (
        Index(
            "uq_events_mlb_game_pk",
            "mlb_game_pk",
            unique=True,
            postgresql_where=text("mlb_game_pk IS NOT NULL"),
        ),
        CheckConstraint(
            "mlb_game_pk IS NULL OR no_game_pk_reason IS NULL",
            name="ck_events_game_pk_or_no_game_pk_reason",
        ),
        UniqueConstraint("surrogate_id", name="uq_events_surrogate_id"),
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

    # Nullable until the backfill fills every stored snapshot.
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
        Index("ix_event_tier", "event_id", "fetch_tier"),
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
    event_id: int = Field(
        sa_column=Column(BigInteger, ForeignKey("events.surrogate_id"), nullable=False)
    )
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
    event_id: int = Field(
        sa_column=Column(BigInteger, ForeignKey("events.surrogate_id"), nullable=False)
    )
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


class Odds(SQLModel, table=True):
    """Normalized odds record for efficient querying."""

    __tablename__ = "odds"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True, description="Event reference")
    snapshot_id: int | None = Field(
        default=None,
        foreign_key="odds_snapshots.id",
        index=True,
        description="Snapshot the row was parsed from; null only for legacy rows whose snapshot "
        "was ambiguous",
    )

    # Bookmaker information
    bookmaker_key: str = Field(index=True, description="Bookmaker identifier")
    bookmaker_title: str = Field(description="Bookmaker display name")
    market_key: str = Field(index=True, description="Market type: h2h, spreads, totals")

    # Outcome data
    outcome_name: str = Field(description="Team name or Over/Under")
    price_decimal: float = Field(
        sa_column=Column(Float, nullable=False),
        description="Decimal odds as quoted by the source (e.g., 1.91, 2.5)",
    )
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
        CheckConstraint("price_decimal > 1.0", name="ck_odds_price_decimal_gt_one"),
    )

    @property
    def implied_probability(self) -> float:
        return decimal_to_implied_probability(self.price_decimal)

    @property
    def american_price(self) -> int:
        return decimal_to_american(self.price_decimal)


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
