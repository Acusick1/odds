"""The candidate page's decision timeline: what a mechanical candidate's rule did at each
decision time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum

from odds_core.models import Event
from odds_core.paper_trade_models import CandidateDecision
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

WINDOW = timedelta(days=14)


class DecisionKind(StrEnum):
    BET = "bet"
    NO_BET = "no bet"
    MISSED = "missed"


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    event: str
    decision_time: datetime
    bets_placed: int


@dataclass(frozen=True)
class DecisionDay:
    day: date
    decisions: list[Decision]


@dataclass(frozen=True)
class DecisionHistory:
    """Every decision of the version in ``total`` and ``missed``; ``days`` are the last
    ``WINDOW`` days up to today, oldest first, empty days included."""

    total: int
    missed: int
    days: list[DecisionDay]

    @property
    def missed_share(self) -> float | None:
        return self.missed / self.total if self.total else None


def kind_of(decision: CandidateDecision) -> DecisionKind:
    if decision.missed:
        return DecisionKind.MISSED
    return DecisionKind.BET if decision.bets_placed > 0 else DecisionKind.NO_BET


async def load_decision_history(
    session: AsyncSession, name: str, version: int, now: datetime, window: timedelta = WINDOW
) -> DecisionHistory:
    of_version = (
        CandidateDecision.candidate_name == name,
        CandidateDecision.candidate_version == version,
    )
    counts = (
        await session.execute(
            select(
                func.count(),
                func.coalesce(func.sum(case((CandidateDecision.missed, 1), else_=0)), 0),
            ).where(*of_version)
        )
    ).one()
    by_day: dict[date, list[Decision]] = {
        (now - timedelta(days=ago)).date(): [] for ago in range(window.days - 1, -1, -1)
    }
    since = datetime.combine(min(by_day), time.min, tzinfo=now.tzinfo)
    stmt = (
        select(CandidateDecision, Event)
        .join(Event, CandidateDecision.event_id == Event.id)
        .where(*of_version, CandidateDecision.decision_time >= since)
        .order_by(CandidateDecision.decision_time, CandidateDecision.id)  # type: ignore[arg-type]
    )
    for decision, event in (await session.execute(stmt)).all():
        by_day.get(decision.decision_time.date(), []).append(
            Decision(
                kind_of(decision),
                f"{event.home_team} v {event.away_team}",
                decision.decision_time,
                decision.bets_placed,
            )
        )
    return DecisionHistory(
        total=counts[0],
        missed=int(counts[1]),
        days=[DecisionDay(day, found) for day, found in by_day.items()],
    )
