"""Scrape job queue model for decoupling MCP server from Playwright scraper."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, DateTime, Index
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class ScrapeJobStatus(str, Enum):
    """Lifecycle status of a scrape job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ScrapeJob(SQLModel, table=True):
    """Persistent job queue row for OddsPortal scrape requests."""

    __tablename__ = "scrape_jobs"

    id: int | None = Field(default=None, primary_key=True)

    # Request parameters
    league: str = Field(index=True, description="OddsHarvester league name")
    market: str = Field(description="Market to scrape (e.g. 1x2, over_under_2_5)")

    # Status
    status: ScrapeJobStatus = Field(default=ScrapeJobStatus.PENDING)

    # Timestamps
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
        default_factory=utc_now,
    )
    started_at: datetime | None = Field(
        sa_column=Column(DateTime(timezone=True)),
        default=None,
    )
    completed_at: datetime | None = Field(
        sa_column=Column(DateTime(timezone=True)),
        default=None,
    )

    # Flat IngestionStats result columns (populated on completion)
    matches_scraped: int | None = Field(default=None)
    matches_converted: int | None = Field(default=None)
    events_matched: int | None = Field(default=None)
    events_created: int | None = Field(default=None)
    snapshots_stored: int | None = Field(default=None)

    # Error info (populated on failure)
    error_message: str | None = Field(default=None)

    __table_args__ = (Index("ix_scrape_jobs_league_status", "league", "market", "status"),)
