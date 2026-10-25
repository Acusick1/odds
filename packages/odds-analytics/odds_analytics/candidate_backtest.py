"""Replay a mechanical candidate over completed events and score it like the forward test.

Each event is decided at its rule's decision time, from the snapshots stamped at or before
it, and the bets are scored by ``odds_analytics.candidate_report``: the same close lookup,
EV, checkpoint and verdict code as the paper trades. Development data can kill an idea but
never promote one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum

from odds_core.candidates import MechanicalCandidate
from odds_core.market_prices import THREE_WAY_MARKET
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.odds_math import calculate_profit_from_decimal, determine_h2h_winner
from odds_lambda.candidate_rules import decision_time, resolve_rule, rule_sha256, run_rule
from odds_lambda.storage.event_merge import canonical_events
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.candidate_report import (
    BOOTSTRAP_SEED,
    DEFAULT_BOOTSTRAP_SAMPLES,
    CandidateSummary,
    Checkpoint,
    ScoredBet,
    checkpoint_at,
    close_candidate_snapshots,
    promote_level,
    scoreable_prefix,
    selection_close_probability,
    summarize_candidate,
)

# Events decided per batch; bounds the snapshots held in memory.
EVENT_BATCH_SIZE = 50


class HashStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    ABSENT = "absent"


@dataclass(frozen=True)
class Coverage:
    """How much of the range the rule could act on."""

    events: int
    with_exchange: int
    with_retail: int
    with_bet: int
    bets: int
    bets_with_close: int


@dataclass(frozen=True)
class BacktestResult:
    """``overall`` is the estimate over every bet with a close in the scoreable prefix,
    beyond the checkpoint cap of the verdict."""

    name: str
    version: int
    rule: str
    params: dict[str, object]
    start: date
    end: date
    hash_status: HashStatus
    coverage: Coverage
    summary: CandidateSummary
    overall: Checkpoint | None


def hash_status(candidate: MechanicalCandidate) -> HashStatus:
    if candidate.rule_sha256 is None:
        return HashStatus.ABSENT
    if candidate.rule_sha256 == rule_sha256(candidate.rule):
        return HashStatus.MATCH
    return HashStatus.MISMATCH


async def final_events(session: AsyncSession, sport: str, start: date, end: date) -> list[Event]:
    """Canonical FINAL events of ``sport`` with scores, kicked off on ``start`` to ``end`` (UTC days)."""
    result = await session.execute(
        select(Event)
        .where(
            Event.sport_key == sport,
            Event.status == EventStatus.FINAL,
            Event.home_score.is_not(None),  # type: ignore[union-attr]
            Event.away_score.is_not(None),  # type: ignore[union-attr]
            Event.commence_time >= datetime.combine(start, time.min, tzinfo=UTC),
            Event.commence_time < datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC),
        )
        .order_by(Event.commence_time, Event.id)  # type: ignore[arg-type]
    )
    return await canonical_events(session, result.scalars().all())


async def _close_snapshots(
    session: AsyncSession, event_ids: set[str]
) -> dict[str, list[OddsSnapshot]]:
    snapshots: dict[str, list[OddsSnapshot]] = defaultdict(list)
    for snapshot in (await session.execute(close_candidate_snapshots(event_ids))).scalars():
        snapshots[snapshot.event_id].append(snapshot)
    return snapshots


async def replay(
    session: AsyncSession,
    candidate: MechanicalCandidate,
    start: date,
    end: date,
    family_versions: int,
    now: datetime | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> BacktestResult:
    """Replay ``candidate`` over the events kicked off from ``start`` to ``end`` inclusive.

    The decision time is the rule's lead before the event's final kick-off, which is the
    only kick-off a completed event has; a live run saw the kick-off as it was then. State
    is built with ``known_at=None`` because backfilled rows were created long after their
    snapshot time. Bets are scored in kick-off order, all settled.
    """
    spec, params = resolve_rule(candidate)
    events = await final_events(session, candidate.sport, start, end)

    bets: list[ScoredBet] = []
    events_with_exchange = events_with_retail = events_with_bet = 0
    for offset in range(0, len(events), EVENT_BATCH_SIZE):
        batch = events[offset : offset + EVENT_BATCH_SIZE]
        decided = []
        for event in batch:
            state, intents = await run_rule(
                session, spec, params, event, decision_time(event.commence_time, params), None
            )
            events_with_exchange += state.exchange is not None
            events_with_retail += bool(state.retail)
            events_with_bet += bool(intents)
            decided.append((event, state.market, intents))

        has_close = {
            e.id for e, market, intents in decided if intents and market == THREE_WAY_MARKET
        }
        snapshots = await _close_snapshots(session, has_close) if has_close else {}
        for event, market, intents in decided:
            assert event.home_score is not None and event.away_score is not None  # noqa: S101
            winner = determine_h2h_winner(event.home_score, event.away_score)
            for intent in intents:
                selection = intent.outcome.value
                bets.append(
                    ScoredBet(
                        candidate_name=candidate.name,
                        candidate_version=candidate.version,
                        event_id=event.id,
                        commence_time=event.commence_time,
                        stake=intent.stake,
                        pnl=calculate_profit_from_decimal(
                            intent.stake, intent.price_decimal, won=selection == winner
                        ),
                        price_decimal=intent.price_decimal,
                        p_close=selection_close_probability(
                            market, selection, event, snapshots.get(event.id, [])
                        ),
                    )
                )
        # Read-only session: drop the loaded snapshots rather than hold the whole range.
        session.expunge_all()

    coverage = Coverage(
        events=len(events),
        with_exchange=events_with_exchange,
        with_retail=events_with_retail,
        with_bet=events_with_bet,
        bets=len(bets),
        bets_with_close=sum(b.p_close is not None for b in bets),
    )
    return score(
        candidate,
        start,
        end,
        bets,
        coverage,
        family_versions,
        now or datetime.now(UTC),
        n_boot,
        seed,
    )


def score(
    candidate: MechanicalCandidate,
    start: date,
    end: date,
    bets: Sequence[ScoredBet],
    coverage: Coverage,
    family_versions: int,
    now: datetime,
    n_boot: int,
    seed: int,
) -> BacktestResult:
    """Summarise replayed ``bets`` with the forward-test report's code."""
    summary = summarize_candidate(
        candidate.name,
        candidate.version,
        candidate.status,
        bets,
        now,
        family_versions,
        n_boot,
        seed,
    )
    with_close, _ = scoreable_prefix(bets, now)
    return BacktestResult(
        name=candidate.name,
        version=candidate.version,
        rule=candidate.rule,
        params=dict(candidate.params),
        start=start,
        end=end,
        hash_status=hash_status(candidate),
        coverage=coverage,
        summary=summary,
        overall=checkpoint_at(
            with_close, len(with_close), promote_level(family_versions), n_boot, seed
        )
        if with_close
        else None,
    )
