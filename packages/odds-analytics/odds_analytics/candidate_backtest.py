"""Replay a mechanical candidate over completed events and score it like the forward test.

Each event is decided at each of its rule's decision times, from the snapshots stamped at or
before it, betting each selection at most once per event, and the bets are scored by
``odds_analytics.candidate_report``: the same close lookup, EV, checkpoint and verdict code
as the paper trades. Development data can kill an idea but
never promote one.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum

from odds_core.betfair_close import close_quotes
from odds_core.candidates import MechanicalCandidate
from odds_core.models import Event, EventStatus
from odds_core.odds_math import calculate_profit_from_decimal, selection_won
from odds_lambda.candidate_rules import (
    RULES,
    ModelRuleSpec,
    RetailEvModelParams,
    RuleParams,
    RuleSpec,
    Selection,
    decision_times,
    resolve_rule,
    rule_sha256,
    run_rule,
)
from odds_lambda.storage.event_merge import canonical_events
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.candidate_model import ModelReplay, Refit, Tuning, library_versions
from odds_analytics.candidate_report import (
    BOOTSTRAP_SEED,
    CLOSE_MARKETS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    CandidateSummary,
    Checkpoint,
    ScoredBet,
    checkpoint_at,
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
    """How much of the range the rule could act on. ``events`` counts the events decided (a
    model rule's scored matches); the other fields count decisions, one per event and decision
    time, and bets."""

    events: int
    decisions: int
    with_input: int
    with_exchange: int
    with_polymarket: int
    with_retail: int
    with_bet: int
    bets: int
    bets_with_close: int


@dataclass(frozen=True)
class BacktestResult:
    """``overall`` is the estimate over every bet with a close in the scoreable prefix,
    beyond the checkpoint cap of the verdict. ``range_events`` counts the canonical events in
    the range, a model rule's training prefix included; ``tuning`` and ``refits`` record a model
    rule's hyperparameters chosen on that prefix and its fits, and ``library_versions`` the
    versions of the libraries that fitted and tuned it. ``comparison`` is another candidate replayed on the same decided events."""

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
    range_events: int = 0
    tuning: Tuning | None = None
    refits: tuple[Refit, ...] = ()
    library_versions: dict[str, str] | None = None
    comparison: BacktestResult | None = None

    @property
    def decided_event_ids(self) -> tuple[str, ...] | None:
        """A model rule's scored matches; ``None`` for a rule that decides every range event."""
        if not isinstance(RULES[self.rule], ModelRuleSpec):
            return None
        return tuple(event_id for refit in self.refits for event_id in refit.event_ids)


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


@dataclass
class _Counts:
    decisions: int = 0
    with_input: int = 0
    with_exchange: int = 0
    with_polymarket: int = 0
    with_retail: int = 0
    with_bet: int = 0


async def _decide(
    session: AsyncSession,
    candidate: MechanicalCandidate,
    spec: RuleSpec,
    params: RuleParams,
    events: Sequence[Event],
    counts: _Counts,
) -> list[ScoredBet]:
    """Decide ``events`` at each of their decision times and settle the bets."""
    market = candidate.market
    bets: list[ScoredBet] = []
    for offset in range(0, len(events), EVENT_BATCH_SIZE):
        batch = events[offset : offset + EVENT_BATCH_SIZE]
        decided = []
        for event in batch:
            bet_selections: set[Selection] = set()
            for _, t in decision_times(event.commence_time, params):
                state, intents = await run_rule(
                    session, spec, params, event, market, t, None, bet_selections=bet_selections
                )
                bet_selections.update(intent.selection for intent in intents)
                counts.decisions += 1
                counts.with_input += state.has_input
                counts.with_exchange += bool(state.exchange)
                counts.with_polymarket += bool(state.polymarket)
                counts.with_retail += bool(state.retail)
                counts.with_bet += bool(intents)
                decided.append((event, t, intents))

        has_close = {e.id: e for e, _, intents in decided if intents and market in CLOSE_MARKETS}
        closes = await close_quotes(session, list(has_close.values()), market)
        for event, t, intents in decided:
            assert event.home_score is not None and event.away_score is not None  # noqa: S101
            for intent in intents:
                bets.append(
                    ScoredBet(
                        candidate_name=candidate.name,
                        candidate_version=candidate.version,
                        event_id=event.id,
                        commence_time=event.commence_time,
                        stake=intent.stake,
                        pnl=calculate_profit_from_decimal(
                            intent.stake,
                            intent.price_decimal,
                            won=selection_won(
                                intent.outcome.value,
                                intent.line,
                                event.home_score,
                                event.away_score,
                            ),
                        ),
                        price_decimal=intent.price_decimal,
                        p_close=selection_close_probability(
                            market,
                            intent.outcome.value,
                            intent.line,
                            event,
                            closes.get(event.id, []),
                        ),
                        placed_at=t,
                    )
                )
    return bets


async def replay(
    session: AsyncSession,
    candidate: MechanicalCandidate,
    start: date,
    end: date,
    family_versions: int,
    now: datetime | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    only_events: Collection[str] | None = None,
) -> BacktestResult:
    """Replay ``candidate`` over the events kicked off from ``start`` to ``end`` inclusive.

    The decision times are the rule's leads before the event's final kick-off, which is the
    only kick-off a completed event has; a live run saw the kick-off as it was then. An
    event's times are decided earliest first, and an outcome bet at one is not bet again.
    State is built with ``known_at=None`` because backfilled rows were created long after
    their snapshot time. Bets are scored in decision-time order, as the live report orders
    trades by placement, all settled.

    A model rule decides only the matches after its training prefix, on which its
    hyperparameters are tuned, each block with the model refitted before it
    (``odds_analytics.candidate_model``). ``only_events`` restricts a
    rule without a model to those event ids, to replay it on another replay's scored matches.
    """
    spec, params = resolve_rule(candidate)
    events = await final_events(session, candidate.sport, start, end)
    range_events = len(events)
    counts = _Counts()
    refits: list[Refit] = []
    tuning: Tuning | None = None
    if isinstance(spec, ModelRuleSpec):
        if only_events is not None:
            raise ValueError(
                f"candidate {candidate.name!r}: a model rule trains on its own range, so it "
                "cannot be restricted to other events"
            )
        assert isinstance(params, RetailEvModelParams)  # noqa: S101
        bets: list[ScoredBet] = []
        model = ModelReplay(session, spec, params, candidate.market, events)
        tuning = await model.tuned_hyperparameters()
        if tuning is not None:
            async for refit, bound, block in model.blocks(tuning):
                refits.append(refit)
                bets += await _decide(session, candidate, bound, params, block, counts)
        scored_events = sum(len(r.event_ids) for r in refits)
    else:
        if only_events is not None:
            events = [e for e in events if e.id in only_events]
        bets = await _decide(session, candidate, spec, params, events, counts)
        scored_events = len(events)
    # Stable, so bets placed at one time keep kick-off and rule order.
    bets.sort(key=lambda bet: bet.placed_at)

    coverage = Coverage(
        events=scored_events,
        decisions=counts.decisions,
        with_input=counts.with_input,
        with_exchange=counts.with_exchange,
        with_polymarket=counts.with_polymarket,
        with_retail=counts.with_retail,
        with_bet=counts.with_bet,
        bets=len(bets),
        bets_with_close=sum(b.p_close is not None for b in bets),
    )
    result = score(
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
    return replace(
        result,
        range_events=range_events,
        tuning=tuning,
        refits=tuple(refits),
        library_versions=library_versions() if isinstance(spec, ModelRuleSpec) else None,
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
    """Summarise replayed ``bets`` with the forward-test report's code, at one family size for all
    checkpoints: a replay has no placement history to count versions against."""
    summary = summarize_candidate(
        candidate.name,
        candidate.version,
        candidate.status,
        bets,
        now,
        lambda _: family_versions,
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
        overall=checkpoint_at(with_close, len(with_close), family_versions, n_boot, seed)
        if with_close
        else None,
    )
