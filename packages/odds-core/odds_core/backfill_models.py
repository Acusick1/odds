"""What a historical backfill records beside the prices it stores."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Text
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now
from odds_core.vocabulary import Market, StrEnumText, closed_set_check


class OddsApiSnapshotGap(SQLModel, table=True):
    """A planned historical Odds API snapshot the provider does not have: the request returned
    the earlier grid snapshot ``returned_snapshot``, whose ``next_snapshot`` the response named.
    A backfill rerun skips a call it records (``odds_analytics.backfill_executor``)."""

    __tablename__ = "odds_api_snapshot_gaps"

    sport_key: str = Field(sa_column=Column(Text, primary_key=True))
    market: Market = Field(sa_column=Column(StrEnumText(Market), primary_key=True))
    planned_snapshot: datetime = Field(sa_column=Column(DateTime(timezone=True), primary_key=True))
    returned_snapshot: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    next_snapshot: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    recorded_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False), default_factory=utc_now
    )

    __table_args__ = (closed_set_check("market", Market, "ck_odds_api_snapshot_gaps_market"),)
