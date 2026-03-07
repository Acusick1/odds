"""Prediction model for CLV signal logging."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Index, UniqueConstraint
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class Prediction(SQLModel, table=True):
    """Predicted CLV delta for an event snapshot, produced by a scoring job."""

    __tablename__ = "predictions"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    snapshot_id: int = Field(foreign_key="odds_snapshots.id", index=True)

    model_name: str = Field(index=True)
    model_version: str = Field()
    predicted_clv: float = Field()

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
    )

    __table_args__ = (
        UniqueConstraint(
            "event_id", "snapshot_id", "model_name", name="uq_prediction_event_snap_model"
        ),
        Index("ix_prediction_model_created", "model_name", "created_at"),
    )
