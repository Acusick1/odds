"""Agent wakeup scheduling model.

Communication channel between the MCP server process and the scheduler process.
One active row per sport — upserted, not inserted.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, UniqueConstraint
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class AgentWakeup(SQLModel, table=True):
    """Requested agent wake-up time, written by MCP and consumed by the scheduler."""

    __tablename__ = "agent_wakeups"

    id: int | None = Field(default=None, primary_key=True)
    sport_key: str = Field(index=True)
    requested_time: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    reason: str = Field()
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
        default_factory=utc_now,
    )

    __table_args__ = (UniqueConstraint("sport_key", name="uq_agent_wakeup_sport_key"),)
