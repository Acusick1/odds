"""Polymarket prediction market database schema definitions."""

from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, Column, DateTime, Index
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class PolymarketMarketType(str, Enum):
    """Polymarket market type classification."""

    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"
    OTHER = "other"


class PolymarketEvent(SQLModel, table=True):
    """Polymarket event mapping to an NBA game."""

    __tablename__ = "polymarket_events"

    id: int | None = Field(default=None, primary_key=True)
    pm_event_id: str = Field(unique=True, index=True, description="Polymarket event ID")
    ticker: str = Field(index=True, description="Event ticker e.g. nba-dal-mil-2026-01-25")
    slug: str = Field(description="URL slug")
    title: str = Field(description="Event display title")

    # Link to internal event (nullable — matched asynchronously by team+date)
    event_id: str | None = Field(
        default=None, foreign_key="events.id", index=True, description="Linked internal event"
    )

    # Temporal bounds
    start_date: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True), description="Event start time"
    )
    end_date: datetime = Field(
        sa_column=Column(DateTime(timezone=True)), description="Event end time"
    )

    # Status
    active: bool = Field(default=True, description="Event is active")
    closed: bool = Field(default=False, description="Event is closed")

    # Aggregate stats (updated each fetch)
    volume: float | None = Field(default=None, description="Total trading volume")
    liquidity: float | None = Field(default=None, description="Total liquidity")
    markets_count: int | None = Field(default=None, description="Number of markets in event")

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

    __table_args__ = (Index("ix_pm_event_active_closed", "active", "closed"),)


class PolymarketMarket(SQLModel, table=True):
    """Individual tradable market within a Polymarket event."""

    __tablename__ = "polymarket_markets"

    id: int | None = Field(default=None, primary_key=True)
    polymarket_event_id: int = Field(
        foreign_key="polymarket_events.id", index=True, description="Parent event"
    )
    pm_market_id: str = Field(unique=True, index=True, description="Polymarket market ID")
    condition_id: str = Field(description="On-chain condition ID")
    question: str = Field(description="Market question text")

    # Token and outcome data
    clob_token_ids: list[str] = Field(
        sa_column=Column(JSON), description="CLOB token IDs for each outcome"
    )
    outcomes: list[str] = Field(
        sa_column=Column(JSON), description="Outcome labels e.g. ['Mavericks', 'Bucks']"
    )

    # Classification
    market_type: PolymarketMarketType = Field(index=True, description="Market type classification")
    group_item_title: str | None = Field(default=None, description="Group label e.g. 'Spread -6.5'")
    point: float | None = Field(default=None, description="Spread/total line value")

    # Status
    active: bool = Field(default=True, description="Market is active")
    closed: bool = Field(default=False, description="Market is closed")
    accepting_orders: bool = Field(default=True, description="Market is accepting orders")

    # Metadata
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
        description="Record creation time",
    )

    __table_args__ = (Index("ix_pm_market_event_type", "polymarket_event_id", "market_type"),)


class PolymarketPriceSnapshot(SQLModel, table=True):
    """Lightweight price snapshot for Polymarket markets."""

    __tablename__ = "polymarket_price_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    polymarket_market_id: int = Field(
        foreign_key="polymarket_markets.id", index=True, description="Market reference"
    )
    snapshot_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
        description="Time of snapshot capture",
    )

    # Prices (implied probabilities 0.0–1.0)
    outcome_0_price: float = Field(description="Outcome 0 implied probability")
    outcome_1_price: float = Field(description="Outcome 1 implied probability")

    # Order book summary
    best_bid: float | None = Field(default=None, description="Best bid price")
    best_ask: float | None = Field(default=None, description="Best ask price")
    spread: float | None = Field(default=None, description="Bid-ask spread")
    midpoint: float | None = Field(default=None, description="Bid-ask midpoint")

    # Volume and liquidity
    volume: float | None = Field(default=None, description="Market volume")
    liquidity: float | None = Field(default=None, description="Market liquidity")

    # Fetch tier tracking (reuses existing tier system)
    fetch_tier: str | None = Field(default=None, index=True, description="Fetch tier")
    hours_until_commence: float | None = Field(
        default=None, description="Hours between snapshot and game start"
    )

    __table_args__ = (
        Index("ix_pm_price_market_time", "polymarket_market_id", "snapshot_time"),
        Index("ix_pm_price_market_tier", "polymarket_market_id", "fetch_tier"),
    )


class PolymarketOrderBookSnapshot(SQLModel, table=True):
    """Full order book depth snapshot for Polymarket markets."""

    __tablename__ = "polymarket_orderbook_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    polymarket_market_id: int = Field(
        foreign_key="polymarket_markets.id", index=True, description="Market reference"
    )
    snapshot_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
        description="Time of snapshot capture",
    )
    token_id: str = Field(description="CLOB token this book represents")

    # Raw order book
    raw_book: dict = Field(
        sa_column=Column(JSON), description="Complete order book {bids: [...], asks: [...]}"
    )

    # Derived summary
    best_bid: float | None = Field(default=None, description="Best bid price")
    best_ask: float | None = Field(default=None, description="Best ask price")
    spread: float | None = Field(default=None, description="Bid-ask spread")
    midpoint: float | None = Field(default=None, description="Bid-ask midpoint")

    # Depth metrics
    bid_levels: int | None = Field(default=None, description="Number of bid levels")
    ask_levels: int | None = Field(default=None, description="Number of ask levels")
    bid_depth_total: float | None = Field(default=None, description="Total bid depth")
    ask_depth_total: float | None = Field(default=None, description="Total ask depth")

    # Pre-computed ML features
    imbalance: float | None = Field(default=None, description="Order book imbalance [-1, 1]")
    weighted_mid: float | None = Field(default=None, description="Volume-weighted midpoint")

    # Fetch tier tracking
    fetch_tier: str | None = Field(default=None, description="Fetch tier")
    hours_until_commence: float | None = Field(
        default=None, description="Hours between snapshot and game start"
    )

    __table_args__ = (
        Index("ix_pm_orderbook_market_time", "polymarket_market_id", "snapshot_time"),
    )


class PolymarketFetchLog(SQLModel, table=True):
    """Polymarket fetch operation logging."""

    __tablename__ = "polymarket_fetch_logs"

    id: int | None = Field(default=None, primary_key=True)
    fetch_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), index=True),
        default_factory=utc_now,
        description="Fetch timestamp",
    )

    job_type: str = Field(description="Fetch job type")
    events_count: int = Field(description="Number of events fetched")
    markets_count: int = Field(description="Number of markets fetched")
    snapshots_stored: int = Field(description="Number of snapshots stored")

    success: bool = Field(description="Whether fetch succeeded")
    error_message: str | None = Field(default=None, description="Error message if failed")
    response_time_ms: int | None = Field(default=None, description="API response time")
