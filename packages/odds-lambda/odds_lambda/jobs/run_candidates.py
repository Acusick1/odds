"""Run the active mechanical candidates of one sport at their decision times.

Each run settles paper trades, takes every decision that is due, and books the next run
at the earliest future decision time (at most ``MAX_WAIT`` away, to pick up new fixtures
and registrations).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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

logger = structlog.get_logger()

# A decision taken later than this after its time would use a market the bet could not have
# been placed in, so it is recorded as missed instead.
MAX_LATENESS = timedelta(minutes=10)
MAX_WAIT = timedelta(hours=6)


@dataclass(frozen=True)
class RunnableCandidate:
    candidate: MechanicalCandidate
    params: RuleParams


@dataclass
class RunSummary:
    placed: int = 0
    skipped_zero_bet: int = 0
    missed: int = 0
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


async def upcoming_events(session: AsyncSession, sport: SportKey, now: datetime) -> list[Event]:
    result = await session.execute(
        select(Event).where(
            Event.sport_key == sport,
            Event.status == EventStatus.SCHEDULED,
            Event.commence_time > now,
        )
    )
    return list(result.scalars().all())


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
        missed=now - t > MAX_LATENESS,
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


async def run_decisions(
    session: AsyncSession,
    runnable: Sequence[RunnableCandidate],
    events: Sequence[Event],
    now: datetime,
) -> RunSummary:
    """Take every decision due at ``now`` that has no record yet; each commits on its own."""
    summary = RunSummary()
    # Ids are read up front: a rollback after a failed decision expires every loaded instance.
    plan = [
        (item, [e.id for e in events if decision_time(e.commence_time, item.params) <= now])
        for item in runnable
    ]
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


async def main(ctx: JobContext) -> None:
    sport = require_sport(ctx, "run-candidates")
    settings = get_settings()
    job_name = make_compound_job_name("run-candidates", sport)
    now = datetime.now(UTC)

    async with async_session_maker() as session:
        runnable = await runnable_candidates(sport, now)
        events = await upcoming_events(session, sport, now)

    # Booked before the work so a crash leaves the chain intact.
    next_time = next_decision_wakeup(runnable, events, now)
    await self_schedule(
        job_name=job_name,
        next_time=next_time,
        dry_run=settings.scheduler.dry_run,
        sport=sport,
        reason="pre-schedule: earliest decision time",
    )

    async with job_alert_context(job_name):
        async with async_session_maker() as session:
            settled = await settle_trades(session)
            await session.commit()
            summary = await run_decisions(session, runnable, events, now)
        logger.info(
            "run_candidates_complete",
            sport=sport,
            candidates=len(runnable),
            settled=len(settled),
            **vars(summary),
        )
        if summary.failed:
            raise RuntimeError(f"{summary.failed} candidate decision(s) failed for {sport}")
