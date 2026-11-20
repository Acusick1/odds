"""What each run-candidates run found wrong with the candidate registry, which the health check
reads.

Kept out of ``odds_core.bet_models``, which is in every rule's import closure, so a change here
does not change any rule's hash.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class CandidateRegistryCheck(SQLModel, table=True):
    """What the latest run-candidates run of one league found wrong with the candidate registry:
    the candidates it skipped as invalid, or the registry error that skipped them all. One row
    per league, replaced by each run."""

    __tablename__ = "candidate_registry_checks"

    sport_key: str = Field(sa_column=Column(Text, primary_key=True))
    checked_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    registry_error: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    invalid_candidates: dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default=text("'{}'::jsonb")),
        description="The reason each skipped candidate is invalid, by candidate name",
    )
