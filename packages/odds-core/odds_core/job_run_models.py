"""The record of each scheduled job run, which the odds-web performance page charts."""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import BigInteger, CheckConstraint, Column, DateTime, Identity, Index, Text
from sqlmodel import Field, SQLModel

from odds_core.vocabulary import StrEnumText, closed_set_check


class JobOutcome(StrEnum):
    """How a scheduled job run ended."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    # Cancelled at the scheduler's hard timeout.
    TIMED_OUT = "timed_out"


class JobRun(SQLModel, table=True):
    """One run of a scheduled job: when it started and finished, and how it ended."""

    __tablename__ = "job_runs"

    id: int | None = Field(default=None, sa_column=Column(BigInteger, Identity(), primary_key=True))
    job_name: str = Field(sa_column=Column(Text, nullable=False))
    started_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    finished_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    outcome: JobOutcome = Field(sa_column=Column(StrEnumText(JobOutcome), nullable=False))

    __table_args__ = (
        Index("ix_job_runs_job_name_started_at", "job_name", "started_at"),
        Index("ix_job_runs_started_at", "started_at"),
        CheckConstraint("finished_at >= started_at", name="ck_job_runs_finished_after_start"),
        closed_set_check("outcome", JobOutcome, "ck_job_runs_outcome"),
    )
