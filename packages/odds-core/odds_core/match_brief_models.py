"""Match brief model for cross-session agent memory."""

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Index
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now

# Per-outcome sharp price entry: bookmaker key, American odds, implied probability.
SharpPriceEntry = dict[str, Any]  # {"bookmaker": str, "price": int, "implied_prob": float}
# Keyed by outcome name (e.g. "Arsenal", "Draw", "Chelsea").
SharpPriceMap = dict[str, SharpPriceEntry]


class BriefCheckpoint(str, Enum):
    """Checkpoint at which a brief is written."""

    CONTEXT = "context"
    DECISION = "decision"


class MatchBrief(SQLModel, table=True):
    """Agent-authored match analysis brief, written at a workflow checkpoint."""

    __tablename__ = "match_briefs"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True)

    checkpoint: BriefCheckpoint = Field(
        description="Workflow checkpoint: context (day before) or decision (KO-90min)"
    )
    brief_text: str = Field(description="Freeform agent brief content")

    sharp_price_at_brief: SharpPriceMap | None = Field(
        sa_column=Column(JSON),
        default=None,
        description="Sharp bookmaker odds snapshot at time of brief creation",
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
    )

    __table_args__ = (Index("ix_match_briefs_event_checkpoint", "event_id", "checkpoint"),)
