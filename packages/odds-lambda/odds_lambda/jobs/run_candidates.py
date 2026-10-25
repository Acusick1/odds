"""Run the active mechanical candidates of one sport at their decision times.

Each run takes every decision that is due, then settles paper trades, and books the next
run at the earliest future decision time (at most ``MAX_WAIT`` away, to pick up new
fixtures and registrations).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

import structlog
from odds_core.alerts import job_alert_context, send_recorded_alert, stale_reminder_interval
from odds_core.candidates import (
    Candidate,
    CandidateStatus,
    MechanicalCandidate,
    candidates_dir,
    load_candidates,
)
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import CandidateDecision
from odds_core.sports import SportKey
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.candidate_rules import (
    RULES,
    RuleParams,
    decision_time,
    load_market_state,
    validate_candidate_rules,
)
from odds_lambda.paper_trading import place_trade, settle_trades
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport
from odds_lambda.storage.event_merge import find_match_id

logger = structlog.get_logger()

# A decision taken later than this after its time would use a market the bet could not have
# been placed in, so it is recorded as missed instead.
MAX_LATENESS = timedelta(minutes=10)
MAX_WAIT = timedelta(minutes=30)
# Events that kicked off without a decision are recorded as missed for this long.
MISSED_LOOKBACK = timedelta(days=7)
# A run with a failed decision is repeated this soon, to retry within MAX_LATENESS.
RETRY_DELAY = timedelta(minutes=2)


@dataclass(frozen=True)
class RunnableCandidate:
    candidate: MechanicalCandidate
    params: RuleParams


@dataclass
class RunSummary:
    placed: int = 0
    skipped_zero_bet: int = 0
    missed: int = 0
    already_decided: int = 0
    failed: int = 0


async def _alert_invalid(name: str, error: ValueError) -> None:
    await send_recorded_alert(
        f"candidate_invalid:{name}",
        "warning",
        f"⚠️ Candidate {name} is skipped by run-candidates: {error}",
        every=stale_reminder_interval(),
    )


async def runnable_candidates(sport: SportKey, today: datetime) -> list[RunnableCandidate]:
    """Active mechanical candidates of ``sport`` that have started and validate.

    An invalid candidate is skipped with one alert per ``stale_reminder_interval``; an
    unreadable registry skips them all.
    """
    try:
        loaded: Sequence[Candidate] = load_candidates(candidates_dir())
    except ValueError as exc:
        logger.error("candidate_registry_invalid", error=str(exc))
        await _alert_invalid("registry", exc)
        return []

    runnable: list[RunnableCandidate] = []
    for candidate in loaded:
        if (
            not isinstance(candidate, MechanicalCandidate)
            or candidate.sport != sport
            or candidate.status != CandidateStatus.ACTIVE
            or candidate.start_date > today.date()
        ):
            continue
        try:
            validate_candidate_rules([candidate])
        except ValueError as exc:
            logger.error("candidate_invalid", candidate=candidate.name, error=str(exc))
            await _alert_invalid(candidate.name, exc)
            continue
        params = RULES[candidate.rule].params_model.model_validate(candidate.params)
        runnable.append(RunnableCandidate(candidate, params))
    return runnable


async def candidate_events(session: AsyncSession, sport: SportKey, now: datetime) -> list[Event]:
    """Events to decide: scheduled ones yet to kick off, and those kicked off within
    ``MISSED_LOOKBACK``, with twins of one match collapsed to the canonical event.
    """
    result = await session.execute(
        select(Event).where(
            Event.sport_key == sport,
            Event.commence_time > now - MISSED_LOOKBACK,
            (Event.commence_time <= now) | (Event.status == EventStatus.SCHEDULED),
        )
    )
    events: list[Event] = []
    for event in result.scalars().all():
        canonical = await find_match_id(
            session, event.home_team, event.away_team, event.commence_time, sport
        )
        if canonical == event.id:
            events.append(event)
    return events


def due_event_ids(item: RunnableCandidate, events: Sequence[Event], now: datetime) -> list[str]:
    """Events whose decision time is at or before ``now`` and not before the candidate's start."""
    start = datetime.combine(item.candidate.start_date, time.min, tzinfo=UTC)
    return [e.id for e in events if start <= decision_time(e.commence_time, item.params) <= now]


def next_decision_wakeup(
    runnable: Sequence[RunnableCandidate], events: Sequence[Event], now: datetime
) -> datetime:
    """The earliest decision time after ``now``, at most ``MAX_WAIT`` away."""
    times = [
        t
        for item in runnable
        for event in events
        if (t := decision_time(event.commence_time, item.params)) > now
    ]
    return min([now + MAX_WAIT, *times])


async def _decide(
    session: AsyncSession, item: RunnableCandidate, event: Event, now: datetime
) -> CandidateDecision:
    candidate = item.candidate
    t = decision_time(event.commence_time, item.params)
    decision = CandidateDecision(
        candidate_name=candidate.name,
        candidate_version=candidate.version,
        event_id=event.id,
        decision_time=t,
        kickoff_at_decision=event.commence_time,
        known_at=now,
        missed=now - t > MAX_LATENESS or event.commence_time <= now,
    )
    session.add(decision)
    await session.flush()
    if decision.missed:
        logger.warning(
            "candidate_decision_missed",
            candidate=candidate.name,
            event_id=event.id,
            decision_time=t.isoformat(),
            now=now.isoformat(),
        )
        return decision

    state = await load_market_state(session, event, t, known_at=now)
    intents = RULES[candidate.rule].function(state, item.params)
    for intent in intents:
        await place_trade(
            session,
            event_id=event.id,
            market=state.market,
            selection=intent.outcome.value,
            bookmaker=intent.bookmaker,
            price_decimal=intent.price_decimal,
            stake=intent.stake,
            bankroll=None,
            candidate_name=candidate.name,
            candidate_version=candidate.version,
            decision_id=decision.id,
        )
    decision.bets_placed = len(intents)
    return decision


async def _is_decided(session: AsyncSession, item: RunnableCandidate, event_id: str) -> bool:
    """Whether another run committed the decision, which is what a unique violation on the
    decision row means; one from a trade of this decision is a failure."""
    found = await session.execute(
        select(CandidateDecision.id).where(
            CandidateDecision.candidate_name == item.candidate.name,
            CandidateDecision.candidate_version == item.candidate.version,
            CandidateDecision.event_id == event_id,
        )
    )
    return found.first() is not None


async def run_decisions(
    session: AsyncSession,
    runnable: Sequence[RunnableCandidate],
    events: Sequence[Event],
    now: datetime,
) -> RunSummary:
    """Take every decision due at ``now`` that has no record yet; each commits on its own."""
    summary = RunSummary()
    # Ids are read up front: a rollback after a failed decision expires every loaded instance.
    plan = [(item, due_event_ids(item, events, now)) for item in runnable]
    for item, due_ids in plan:
        candidate = item.candidate
        if not due_ids:
            continue
        decided = set(
            (
                await session.execute(
                    select(CandidateDecision.event_id).where(
                        CandidateDecision.candidate_name == candidate.name,
                        CandidateDecision.candidate_version == candidate.version,
                        CandidateDecision.event_id.in_(due_ids),  # type: ignore[attr-defined]
                    )
                )
            )
            .scalars()
            .all()
        )
        for event_id in due_ids:
            if event_id in decided:
                continue
            try:
                event = await session.get(Event, event_id, populate_existing=True)
                assert event is not None  # noqa: S101
                decision = await _decide(session, item, event, now)
                await session.commit()
            except IntegrityError:
                await session.rollback()
                if await _is_decided(session, item, event_id):
                    logger.info(
                        "candidate_decision_already_taken",
                        candidate=candidate.name,
                        event_id=event_id,
                    )
                    summary.already_decided += 1
                    continue
                logger.error(
                    "candidate_decision_failed",
                    candidate=candidate.name,
                    event_id=event_id,
                    exc_info=True,
                )
                summary.failed += 1
                continue
            except Exception:
                await session.rollback()
                logger.error(
                    "candidate_decision_failed",
                    candidate=candidate.name,
                    event_id=event_id,
                    exc_info=True,
                )
                summary.failed += 1
                continue
            if decision.missed:
                summary.missed += 1
            elif decision.bets_placed:
                summary.placed += decision.bets_placed
            else:
                summary.skipped_zero_bet += 1
    return summary


async def _settle(sport: SportKey) -> int:
    """Settle paper trades in a session of its own; a failure is logged and alerted, never raised."""
    try:
        async with async_session_maker() as session:
            settled = await settle_trades(session)
            await session.commit()
        return len(settled)
    except Exception as exc:
        logger.error("candidate_settlement_failed", sport=sport, exc_info=True)
        await send_recorded_alert(
            f"candidate_settlement_failed:{sport}",
            "critical",
            f"Paper trade settlement failed in run-candidates for {sport}: {exc}",
            every=stale_reminder_interval(),
        )
        return 0


async def main(ctx: JobContext) -> None:
    sport = require_sport(ctx, "run-candidates")
    settings = get_settings()
    job_name = make_compound_job_name("run-candidates", sport)
    now = datetime.now(UTC)

    async def book(next_time: datetime, reason: str) -> None:
        await self_schedule(
            job_name=job_name,
            next_time=next_time,
            dry_run=settings.scheduler.dry_run,
            sport=sport,
            reason=reason,
        )

    # A failure in these two steps, or in booking, ends the run before anything is booked;
    # the scheduler runner then re-books the chain.
    async with async_session_maker() as session:
        runnable = await runnable_candidates(sport, now)
        events = await candidate_events(session, sport, now)

    # Booked before the work so a crash leaves the chain intact.
    next_time = next_decision_wakeup(runnable, events, now)
    await book(next_time, "pre-schedule: earliest decision time")

    async with job_alert_context(job_name):
        async with async_session_maker() as session:
            summary = await run_decisions(session, runnable, events, now)
        settled = await _settle(sport)
        logger.info(
            "run_candidates_complete",
            sport=sport,
            candidates=len(runnable),
            settled=settled,
            **vars(summary),
        )
        if summary.failed:
            await book(min(next_time, now + RETRY_DELAY), "retry: failed decision")
            raise RuntimeError(f"{summary.failed} candidate decision(s) failed for {sport}")
