"""Stored replays of a candidate on completed events: the development and holdout runs behind an
idea, kept so the web UI can show how a strategy performed historically and why a killed idea
was killed."""

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now
from odds_core.vocabulary import Source, StrEnumText, closed_set_check


class WindowKind(StrEnum):
    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


class CandidateBacktest(SQLModel, table=True):
    """One replay of a candidate version over a window, scored over its bets with a close.

    A grid cell scored by ``odds candidates select`` is a run too: ``cell_index`` and
    ``eligible`` are set for it and null for any other, and ``selected`` marks the cell the
    selection rule picked. A holdout run is the selected cell on the sealed window. ``blocks``
    lists a pooled development run's blocks with their bets and mean EV, and is null otherwise.

    The estimate (``mean_ev``, ``ci_lower``, ``ci_upper`` at ``ci_level``) is over the ``n``
    bets with a close, or null with none. Bets are stored only for a run a page shows: a
    replay, the selected cell and a holdout run, never the other cells of a grid.
    """

    __tablename__ = "candidate_backtests"

    id: int | None = Field(default=None, primary_key=True)
    candidate_name: str = Field(sa_column=Column(Text, nullable=False, index=True))
    candidate_version: int = Field(sa_column=Column(Integer, nullable=False))
    issue: int = Field(sa_column=Column(Integer, nullable=False))
    market: str = Field(sa_column=Column(Text, nullable=False))
    rule: str = Field(sa_column=Column(Text, nullable=False))
    rule_sha256: str | None = Field(default=None, sa_column=Column(String(64)))
    params: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    window_kind: WindowKind = Field(sa_column=Column(StrEnumText(WindowKind), nullable=False))
    start_date: date = Field(sa_column=Column(Date, nullable=False))
    end_date: date = Field(sa_column=Column(Date, nullable=False))
    leagues: list[str] = Field(sa_column=Column(ARRAY(Text), nullable=False))
    git_sha: str = Field(sa_column=Column(Text, nullable=False))
    cell_index: int | None = Field(default=None, sa_column=Column(Integer))
    eligible: bool | None = Field(default=None, sa_column=Column(Boolean))
    selected: bool = Field(
        default=False, sa_column=Column(Boolean, nullable=False, server_default=text("false"))
    )
    blocks: list[dict[str, Any]] | None = Field(default=None, sa_column=Column(JSONB))
    events: int = Field(sa_column=Column(Integer, nullable=False))
    bets: int = Field(sa_column=Column(Integer, nullable=False))
    bets_with_close: int = Field(sa_column=Column(Integer, nullable=False))
    stake: float = Field(sa_column=Column(Float, nullable=False))
    pnl: float = Field(sa_column=Column(Float, nullable=False))
    mean_ev: float | None = Field(default=None, sa_column=Column(Float))
    ci_lower: float | None = Field(default=None, sa_column=Column(Float))
    ci_upper: float | None = Field(default=None, sa_column=Column(Float))
    ci_level: float | None = Field(default=None, sa_column=Column(Float))
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False), default_factory=utc_now
    )

    __table_args__ = (
        closed_set_check("window_kind", WindowKind, "ck_candidate_backtests_window_kind"),
        CheckConstraint("start_date <= end_date", name="ck_candidate_backtests_range"),
        CheckConstraint(
            "(cell_index IS NULL) = (eligible IS NULL)", name="ck_candidate_backtests_cell"
        ),
        CheckConstraint(
            "(mean_ev IS NULL) = (ci_lower IS NULL) AND (mean_ev IS NULL) = (ci_upper IS NULL) "
            "AND (mean_ev IS NULL) = (ci_level IS NULL)",
            name="ck_candidate_backtests_estimate",
        ),
        UniqueConstraint(
            "candidate_name",
            "candidate_version",
            "window_kind",
            "start_date",
            "end_date",
            "git_sha",
            "cell_index",
            name="uq_candidate_backtests_run",
            postgresql_nulls_not_distinct=True,
        ),
    )


class CandidateBacktestBet(SQLModel, table=True):
    """A bet a stored run placed, in placement order: ``position`` counts from 1.
    ``p_close``, ``ev`` and ``close_source`` are null without a close. ``estimated_ev`` is the
    EV the rule estimated at decision time, null for a rule without one."""

    __tablename__ = "candidate_backtest_bets"

    id: int | None = Field(default=None, primary_key=True)
    backtest_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("candidate_backtests.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    position: int = Field(sa_column=Column(Integer, nullable=False))
    event_id: int = Field(sa_column=Column(BigInteger, nullable=False))
    league: str = Field(sa_column=Column(Text, nullable=False))
    kickoff: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    decision_time: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    outcome: str = Field(sa_column=Column(Text, nullable=False))
    line: float | None = Field(default=None, sa_column=Column(Float))
    bookmaker: str = Field(sa_column=Column(Text, nullable=False))
    price: float = Field(sa_column=Column(Float, nullable=False))
    stake: float = Field(sa_column=Column(Float, nullable=False))
    pnl: float | None = Field(default=None, sa_column=Column(Float))
    p_close: float | None = Field(default=None, sa_column=Column(Float))
    ev: float | None = Field(default=None, sa_column=Column(Float))
    close_source: Source | None = Field(default=None, sa_column=Column(StrEnumText(Source)))
    estimated_ev: float | None = Field(default=None, sa_column=Column(Float))

    __table_args__ = (
        UniqueConstraint("backtest_id", "position", name="uq_candidate_backtest_bets_position"),
        closed_set_check("close_source", Source, "ck_candidate_backtest_bets_close_source"),
    )
