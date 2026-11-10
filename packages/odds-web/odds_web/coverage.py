"""Decision-window coverage: which of a candidate rule's inputs were fresh at each bucket before
kick-off, built by the rule's own market-state code."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import CandidateDecision
from odds_core.quotes import Quote
from odds_core.vocabulary import Market
from odds_lambda.candidate_rules import (
    SharpInput,
    build_market_state,
    decision_times,
    max_input_age,
    resolve_rule,
)
from odds_lambda.candidate_rules.inputs import rule_quotes
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

BUCKET_MINUTES = 10
# The heatmap starts this long before a candidate's earliest decision time.
LEAD_IN_MINUTES = 60
PAST = timedelta(days=7)
UPCOMING = timedelta(hours=24)
RETAIL = "retail"


@dataclass(frozen=True)
class CoverageCell:
    """The state at ``time``, ``minutes_before`` kick-off, from the snapshots stored by
    ``known_at``: the bucket's own time, or a recorded decision's ``known_at``.

    ``fresh`` names the rule's inputs a state needing only that input would have, and
    ``complete`` is whether the rule's own state has inputs. A time still ahead is not
    ``evaluated``.
    """

    minutes_before: int
    time: datetime
    known_at: datetime
    evaluated: bool
    recorded: bool
    fresh: tuple[str, ...] = ()
    complete: bool = False


@dataclass(frozen=True)
class CoverageRow:
    event_id: int
    label: str
    kickoff: datetime
    cells: list[CoverageCell]


@dataclass(frozen=True)
class CandidateCoverage:
    name: str
    version: int
    sport: str
    market: str
    inputs: tuple[str, ...]
    minutes: list[int]
    decision_minutes: tuple[int, ...]
    rows: list[CoverageRow]


def bucket_minutes(decision_minutes: Sequence[int]) -> list[int]:
    """The columns, minutes before kick-off and earliest first: every bucket from the lead-in
    before the earliest decision to the last bucket before kick-off, and each decision time."""
    top = max(decision_minutes) + LEAD_IN_MINUTES
    top += -top % BUCKET_MINUTES
    return sorted(set(range(top, 0, -BUCKET_MINUTES)) | set(decision_minutes), reverse=True)


@dataclass(frozen=True)
class _EventInputs:
    """One event's rule quotes over every bucket's window, and the rule's input needs."""

    event: Event
    market: Market
    quoted: Sequence[Quote]
    age: timedelta
    sharp_inputs: frozenset[SharpInput]

    def evaluate(self, slot: CoverageCell, now: datetime) -> CoverageCell:
        """``slot`` with its inputs filled in, from the quotes stored by its ``known_at``; a
        slot still ahead of ``now`` is returned as it is."""
        if slot.known_at > now or slot.time > now:
            return slot
        visible = [q for q in self.quoted if q.created_at <= slot.known_at]

        def has_input(needs: frozenset[SharpInput]) -> bool:
            state = build_market_state(self.event, self.market, visible, slot.time, self.age, needs)
            return state.has_input

        fresh = [RETAIL] if has_input(frozenset()) else []
        fresh += [i.value for i in sorted(self.sharp_inputs) if has_input(frozenset({i}))]
        return replace(
            slot, evaluated=True, fresh=tuple(fresh), complete=has_input(self.sharp_inputs)
        )


async def candidate_coverage(
    session: AsyncSession, candidate: MechanicalCandidate, now: datetime
) -> CandidateCoverage:
    """The candidate's coverage over its sport's events kicking off from ``PAST`` before
    ``now`` to ``UPCOMING`` after it.

    Per event, one ``rule_quotes`` read over every bucket's window, then ``build_market_state``
    per bucket on the snapshots stored by the bucket's time. A bucket at a decision the
    candidate version recorded takes the decision's time and ``known_at`` instead.
    """
    spec, params = resolve_rule(candidate)
    age = max_input_age(params)
    decision_minutes = params.minutes_before_kickoff
    minutes = bucket_minutes(decision_minutes)
    events = (
        await session.scalars(
            select(Event)
            .where(
                col(Event.sport_key) == candidate.sport,
                col(Event.commence_time) >= now - PAST,
                col(Event.commence_time) <= now + UPCOMING,
                col(Event.status).notin_((EventStatus.POSTPONED, EventStatus.CANCELLED)),
            )
            .order_by(col(Event.commence_time), col(Event.id))
        )
    ).all()
    decisions = (
        await session.scalars(
            select(CandidateDecision).where(
                col(CandidateDecision.candidate_name) == candidate.name,
                col(CandidateDecision.candidate_version) == candidate.version,
                col(CandidateDecision.event_id).in_([e.id for e in events]),
            )
        )
    ).all()
    recorded = {(d.event_id, d.minutes_before_kickoff): d for d in decisions}
    rows: list[CoverageRow] = []
    for event in events:
        assert event.id is not None
        decided = dict(decision_times(event.commence_time, params))
        slots: list[CoverageCell] = []
        for m in minutes:
            decision = recorded.get((event.id, m))
            if decision is not None:
                slot = CoverageCell(m, decision.decision_time, decision.known_at, False, True)
            else:
                at = decided.get(m) or event.commence_time - timedelta(minutes=m)
                slot = CoverageCell(m, at, at, evaluated=False, recorded=False)
            slots.append(slot)
        quoted = await rule_quotes(
            session,
            event.id,
            candidate.market,
            spec.sharp_inputs,
            since=min(slot.time for slot in slots) - age,
            until=max(slot.time for slot in slots),
        )
        inputs = _EventInputs(event, candidate.market, quoted, age, spec.sharp_inputs)
        cells = [inputs.evaluate(slot, now) for slot in slots]
        rows.append(
            CoverageRow(
                event.id, f"{event.home_team} v {event.away_team}", event.commence_time, cells
            )
        )
    return CandidateCoverage(
        name=candidate.name,
        version=candidate.version,
        sport=candidate.sport,
        market=candidate.market.value,
        inputs=(RETAIL, *(i.value for i in sorted(spec.sharp_inputs))),
        minutes=minutes,
        decision_minutes=tuple(decision_minutes),
        rows=rows,
    )


def coverage_candidates(registry: Sequence[Candidate]) -> list[MechanicalCandidate]:
    """The active mechanical candidates, at the version the registry holds."""
    return [
        c
        for c in registry
        if isinstance(c, MechanicalCandidate) and c.status == CandidateStatus.ACTIVE
    ]
