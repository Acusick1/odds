"""Match brief model for cross-session agent memory."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Column, DateTime
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class BriefDecision(str, Enum):
    """Agent's decision at the time of writing a brief."""

    WATCHING = "watching"
    BET = "bet"
    SKIP = "skip"


# Per-outcome sharp price entry: bookmaker key, derived American quote, source decimal
# price, implied probability.
SharpPriceEntry = dict[
    str, Any
]  # {"bookmaker": str, "price": int, "price_decimal": float, "implied_prob": float}
# Keyed by outcome name (e.g. "Arsenal", "Draw", "Chelsea").
SharpPriceMap = dict[str, SharpPriceEntry]


@dataclass
class SharpPriceMeta:
    """Metadata about a sharp price found via lookback search."""

    snapshot_id: int
    snapshot_time: datetime
    age_seconds: float


class MatchBrief(SQLModel, table=True):
    """Agent-authored match analysis brief, one per wake-up per event (append-only)."""

    __tablename__ = "match_briefs"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True)

    decision: BriefDecision = Field(description="Agent decision: watching, bet, or skip")
    summary: str = Field(description="One-line summary of the brief for triage views")
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
