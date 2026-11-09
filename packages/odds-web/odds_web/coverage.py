"""Decision-window coverage: which of a candidate rule's inputs were fresh at each bucket before
kick-off, built by the rule's own market-state code."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import CandidateDecision
from odds_core.quotes import Quote
from odds_lambda.candidate_rules import (
    SharpInput,
    build_market_state,
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


def _cell(
    event: Event,
    candidate: MechanicalCandidate,
    sharp_inputs: frozenset[SharpInput],
    age: timedelta,
    quoted: Sequence[Quote],
    minutes: int,
    at: datetime,
    known_at: datetime,
    recorded: bool,
    now: datetime,
) -> CoverageCell:
    if known_at > now or at > now:
        return CoverageCell(minutes, at, known_at, evaluated=False, recorded=recorded)
    visible = [q for q in quoted if q.created_at <= known_at]

    def has_input(needs: frozenset[SharpInput]) -> bool:
        return build_market_state(event, candidate.market, visible, at, age, needs).has_input

    fresh = [RETAIL] if has_input(frozenset()) else []
    fresh += [i.value for i in sorted(sharp_inputs) if has_input(frozenset({i}))]
    return CoverageCell(
        minutes,
        at,
        known_at,
        evaluated=True,
        recorded=recorded,
        fresh=tuple(fresh),
        complete=has_input(sharp_inputs),
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
        times: list[tuple[int, datetime, datetime, bool]] = []
        for m in minutes:
            decision = recorded.get((event.id, m))
            if decision is not None:
                times.append((m, decision.decision_time, decision.known_at, True))
            else:
                at = event.commence_time - timedelta(minutes=m)
                times.append((m, at, at, False))
        quoted = await rule_quotes(
            session,
            event.id,
            candidate.market,
            spec.sharp_inputs,
            since=min(at for _, at, _, _ in times) - age,
            until=max(at for _, at, _, _ in times),
        )
        cells = [
            _cell(event, candidate, spec.sharp_inputs, age, quoted, m, at, known, was, now)
            for m, at, known, was in times
        ]
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
