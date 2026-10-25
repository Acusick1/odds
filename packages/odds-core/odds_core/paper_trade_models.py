"""Paper trade model for forward-looking bet tracking and settlement."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    UniqueConstraint,
    text,
)
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class TradeResult(str, Enum):
    """Outcome of a settled paper trade."""

    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"


class CandidateDecision(SQLModel, table=True):
    """One run of a mechanical candidate's rule on one event at one of its decision times,
    with the inputs it used.

    Written even when the rule places no bet, so a decision is never taken twice and a replay
    can compare a zero-bet decision, and records the snapshots it used so a replay needs no
    guess at what the run saw. A missed decision (the run came too late, or no snapshot was
    recent enough to decide from) is recorded with ``missed`` set and places nothing. Keyed
    by the decision's minutes before kick-off rather than its time, so a kick-off that moves
    after the decision does not open a second decision for the same entry.
    """

    __tablename__ = "candidate_decisions"

    id: int | None = Field(default=None, primary_key=True)
    candidate_name: str = Field(index=True)
    candidate_version: int
    event_id: str = Field(foreign_key="events.id", index=True)
    minutes_before_kickoff: int = Field(
        sa_column=Column(Integer, nullable=False),
        description="The entry of the rule's minutes_before_kickoff this decision is for",
    )
    decision_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
        description="Kick-off at decision minus minutes_before_kickoff",
    )
    kickoff_at_decision: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
        description="The event's kick-off when the decision was taken",
    )
    known_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
        description="Run time: only snapshots stored by then were visible to the rule",
    )
    input_snapshot_ids: list[int] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(Integer), nullable=False, server_default=text("'{}'")),
        description="Id of the snapshot the rule's market state was built from; empty when missed",
    )
    bets_placed: int = Field(default=0)
    missed: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default=text("false")),
        description=(
            "The run came later than the allowed lateness, or no snapshot was within the "
            "rule's maximum input age, so nothing was placed"
        ),
    )

    __table_args__ = (
        UniqueConstraint(
            "candidate_name",
            "candidate_version",
            "event_id",
            "minutes_before_kickoff",
            name="uq_candidate_decisions_candidate_event_lead",
        ),
        CheckConstraint("NOT missed OR bets_placed = 0", name="ck_candidate_decisions_missed"),
    )


class PaperTrade(SQLModel, table=True):
    """Persistent record of a paper bet, placed before an event and settled against results."""

    __tablename__ = "paper_trades"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True)

    # Bet details
    market: str = Field(description="Market type: h2h")
    selection: str = Field(description="Selection: home, draw, or away")
    bookmaker: str = Field(index=True, description="Bookmaker key")
    price_decimal: float = Field(
        sa_column=Column(Float, nullable=False),
        description="Decimal price taken at placement (e.g. 1.91, 2.5)",
    )
    stake: float = Field(description="Stake amount")

    # Candidate that placed the bet; both set or neither, and null before the registry existed
    candidate_name: str | None = Field(default=None, index=True)
    candidate_version: int | None = Field(default=None)
    git_sha: str | None = Field(default=None, description="Commit the bet was placed from")
    decision_id: int | None = Field(
        default=None,
        foreign_key="candidate_decisions.id",
        index=True,
        description="The scheduled-run decision that placed this bet; null for the agent's bets",
    )

    # Agent reasoning
    reasoning: str | None = Field(default=None, description="Why this bet was placed")
    confidence: float | None = Field(default=None, description="Agent confidence score (0-1)")

    # Bankroll tracking
    # Bankroll snapshots are per-trade only (bankroll_before at placement, bankroll_after at
    # settlement as bankroll_before + pnl). They do not reflect other trades placed in between.
    # Flat-stake candidate trades record neither.
    bankroll_before: float | None = Field(
        default=None, description="Bankroll before this bet; null for a flat-stake trade"
    )
    bankroll_after: float | None = Field(default=None, description="Bankroll after settlement")

    # Timestamps
    placed_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
        default_factory=utc_now,
        description="When the bet was placed",
    )
    settled_at: datetime | None = Field(
        sa_column=Column(DateTime(timezone=True)),
        default=None,
        description="When the bet was settled",
    )

    # Settlement
    result: TradeResult | None = Field(default=None, description="Settlement outcome")
    pnl: float | None = Field(default=None, description="Profit/loss from this bet")

    __table_args__ = (
        Index(
            "ix_paper_trades_unsettled", "settled_at", postgresql_where=text("settled_at IS NULL")
        ),
        Index("ix_paper_trades_event_selection", "event_id", "selection"),
        Index(
            "uq_paper_trades_candidate_bet",
            "candidate_name",
            "candidate_version",
            "event_id",
            "selection",
            unique=True,
            postgresql_where=text("decision_id IS NOT NULL"),
        ),
        CheckConstraint("price_decimal > 1.0", name="ck_paper_trades_price_decimal_gt_one"),
        CheckConstraint(
            "(candidate_name IS NULL) = (candidate_version IS NULL)",
            name="ck_paper_trades_candidate_pair",
        ),
        CheckConstraint(
            "bankroll_before IS NOT NULL OR candidate_name IS NOT NULL",
            name="ck_paper_trades_bankroll_or_candidate",
        ),
    )
