"""Replay a mechanical candidate over completed events and score it like the forward test.

Each event is decided at each of its rule's decision times, from the snapshots stamped at or
before it, betting each selection at most once per event, and the bets are scored by
``odds_analytics.candidate_report``: the same close lookup, EV, checkpoint and verdict code
as the paper trades. Development data can kill an idea but
never promote one.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum

from odds_core.candidates import MechanicalCandidate
from odds_core.close import close_grace, close_quotes
from odds_core.models import Event, EventStatus
from odds_core.odds_math import calculate_profit_from_decimal, selection_won
from odds_lambda.candidate_rules import (
    RULES,
    ModelRuleSpec,
    RetailEvModelParams,
    RuleParams,
    RuleSpec,
    Selection,
    candidate_leagues,
    decision_times,
    resolve_rule,
    run_rule,
)
from odds_lambda.candidate_rules.pinning import rule_sha256
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.candidate_model import ModelReplay, Refit, Tuning, library_versions
from odds_analytics.candidate_report import (
    BOOTSTRAP_SEED,
    DEFAULT_BOOTSTRAP_SAMPLES,
    CandidateSummary,
    Checkpoint,
    ScoredBet,
    checkpoint_at,
    scoreable_prefix,
    selection_close_probability,
    summarize_candidate,
)
from odds_analytics.holdout import HoldoutAccess, require_clear

# Events decided per batch; bounds the snapshots held in memory.
EVENT_BATCH_SIZE = 50


class HashStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    ABSENT = "absent"
    # The rule's closure has code the hash cannot follow, so no candidate of it can register.
    UNHASHABLE = "unhashable"


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
    beyond the checkpoint cap of the verdict. ``range_events`` counts the events in the
    range, a model rule's training prefix included; ``tuning`` and ``refits`` record a model
    rule's hyperparameters chosen on that prefix and its fits, and ``library_versions`` the
    versions of the libraries that fitted and tuned it. ``league_coverage`` is the coverage of
    each league the candidate bets in, for information: the verdict is the pooled series'.
    ``comparison`` is another candidate replayed on the same decided events."""

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
    league_coverage: dict[str, Coverage] | None = None
    comparison: BacktestResult | None = None

    @property
    def decided_event_ids(self) -> tuple[int, ...] | None:
        """A model rule's scored matches; ``None`` for a rule that decides every range event."""
        if not isinstance(RULES[self.rule], ModelRuleSpec):
            return None
        return tuple(event_id for refit in self.refits for event_id in refit.event_ids)


def hash_status(candidate: MechanicalCandidate) -> HashStatus:
    if candidate.rule_sha256 is None:
        return HashStatus.ABSENT
    try:
        current = rule_sha256(candidate.rule)
    except ValueError:
        return HashStatus.UNHASHABLE
    if candidate.rule_sha256 == current:
        return HashStatus.MATCH
    return HashStatus.MISMATCH


async def final_events(
    session: AsyncSession,
    sport: str,
    start: date,
    end: date,
    access: HoldoutAccess | None = None,
) -> list[Event]:
    """FINAL events of ``sport`` with scores, kicked off on ``start`` to ``end`` (UTC days).

    Every replay selects its events here, so this is where the sealed holdout is enforced: a
    range that intersects it raises ``HoldoutRefused`` without ``access``
    (``odds_analytics.holdout``).
    """
    require_clear(sport, start, end, access)
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
    return list(result.scalars().all())


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
                    session,
                    spec,
                    params,
                    event,
                    market,
                    t,
                    None,
                    bet_selections=bet_selections,
                    fixture_leagues=candidate_leagues(candidate),
                )
                bet_selections.update(intent.selection for intent in intents)
                counts.decisions += 1
                counts.with_input += state.has_input
                counts.with_exchange += bool(state.exchange)
                counts.with_polymarket += bool(state.polymarket)
                counts.with_retail += bool(state.retail)
                counts.with_bet += bool(intents)
                decided.append((event, t, intents))

        betting = {e.id: e for e, _, intents in decided if intents}
        closes = await close_quotes(session, list(betting.values()), market)
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
                        close_grace=close_grace(event.sport_key, market),
                        sport_key=event.sport_key,
                    )
                )
    return bets


# Inclusive UTC kick-off days.
DayRange = tuple[date, date]


@dataclass(frozen=True)
class Replayed:
    """What a replay decided, before scoring: the bets in placement order and, per league, the
    counts behind the coverage and the events decided."""

    bets: list[ScoredBet]
    counts: dict[str, _Counts]
    scored_events: dict[str, int]
    range_events: int
    tuning: Tuning | None
    refits: tuple[Refit, ...]
    model: bool


def _placement_order(bet: ScoredBet) -> tuple[datetime, datetime, int]:
    # Stable, so the bets of one event placed at one time keep rule order.
    return (bet.placed_at, bet.commence_time, bet.event_id)


async def _decide_leagues(
    session: AsyncSession,
    candidate: MechanicalCandidate,
    spec: RuleSpec,
    params: RuleParams,
    events: Sequence[Event],
    counts: dict[str, _Counts],
) -> list[ScoredBet]:
    """``_decide`` per league of ``events``, counting each league apart."""
    bets: list[ScoredBet] = []
    for league in candidate.sports:
        league_events = [e for e in events if e.sport_key == league]
        if league_events:
            bets += await _decide(
                session,
                candidate,
                spec,
                params,
                league_events,
                counts.setdefault(league, _Counts()),
            )
    return bets


async def collect(
    session: AsyncSession,
    candidate: MechanicalCandidate,
    ranges: Mapping[str, Sequence[DayRange]],
    only_events: Collection[int] | None = None,
    access: Collection[HoldoutAccess] = (),
) -> Replayed:
    """Decide ``candidate`` over the events of each of its leagues kicked off in that league's
    ``ranges`` and settle its bets, pooled in decision-time order. A sealed range needs the
    league's ``access``."""
    grants = {a.window.sport: a for a in access}
    spans = [(league, span) for league in candidate.sports for span in ranges.get(league, ())]
    for league in candidate.sports:
        ordered = sorted(ranges.get(league, ()))
        for (_, earlier_last), (later_first, _) in zip(ordered, ordered[1:], strict=False):
            # An event on a shared day would be decided once per range.
            if later_first <= earlier_last:
                raise ValueError(f"{league}: day ranges overlap on {later_first}")
    for league, (first, last) in spans:
        require_clear(league, first, last, grants.get(league))
    spec, params = resolve_rule(candidate)
    events: list[Event] = []
    for league, (first, last) in spans:
        events += await final_events(session, league, first, last, grants.get(league))
    events.sort(key=lambda e: (e.commence_time, e.id))
    range_events = len(events)
    counts: dict[str, _Counts] = {league: _Counts() for league in candidate.sports}
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
                bets += await _decide_leagues(session, candidate, bound, params, block, counts)
        league_of = {e.id: e.sport_key for e in events}
        scored = [league_of[event_id] for r in refits for event_id in r.event_ids]
    else:
        if only_events is not None:
            events = [e for e in events if e.id in only_events]
        bets = await _decide_leagues(session, candidate, spec, params, events, counts)
        scored = [e.sport_key for e in events]
    bets.sort(key=_placement_order)
    return Replayed(
        bets,
        counts,
        {league: scored.count(league) for league in candidate.sports},
        range_events,
        tuning,
        tuple(refits),
        isinstance(spec, ModelRuleSpec),
    )


def _coverage(counts: _Counts, events: int, bets: Sequence[ScoredBet]) -> Coverage:
    return Coverage(
        events=events,
        decisions=counts.decisions,
        with_input=counts.with_input,
        with_exchange=counts.with_exchange,
        with_polymarket=counts.with_polymarket,
        with_retail=counts.with_retail,
        with_bet=counts.with_bet,
        bets=len(bets),
        bets_with_close=sum(b.p_close is not None for b in bets),
    )


def league_coverage(replayed: Replayed) -> dict[str, Coverage]:
    return {
        league: _coverage(
            counts,
            replayed.scored_events.get(league, 0),
            [b for b in replayed.bets if b.sport_key == league],
        )
        for league, counts in replayed.counts.items()
    }


def coverage_of(replayed: Replayed) -> Coverage:
    total = _Counts()
    for counts in replayed.counts.values():
        for field in fields(_Counts):
            setattr(total, field.name, getattr(total, field.name) + getattr(counts, field.name))
    return _coverage(total, sum(replayed.scored_events.values()), replayed.bets)


async def replay(
    session: AsyncSession,
    candidate: MechanicalCandidate,
    start: date,
    end: date,
    family_versions: int,
    now: datetime | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    only_events: Collection[int] | None = None,
    holdout_access: Collection[HoldoutAccess] = (),
) -> BacktestResult:
    """Replay ``candidate`` over the events of its leagues kicked off from ``start`` to ``end``
    inclusive.

    The decision times are the rule's leads before the event's final kick-off, which is the
    only kick-off a completed event has; a live run saw the kick-off as it was then. An
    event's times are decided earliest first, and an outcome bet at one is not bet again.
    State is built with ``known_at=None`` because backfilled rows were created long after
    their snapshot time. Bets of every league are scored as one series in decision-time order,
    as the live report orders trades by placement, all settled.

    A model rule decides only the matches after its training prefix, on which its
    hyperparameters are tuned, each block with the model refitted before it
    (``odds_analytics.candidate_model``). ``only_events`` restricts a
    rule without a model to those event ids, to replay it on another replay's scored matches.

    A range that intersects the sealed holdout raises ``HoldoutRefused`` unless
    ``holdout_access``, issued only by ``odds_analytics.holdout.Reservation``, covers it.
    """
    replayed = await collect(
        session,
        candidate,
        {league: [(start, end)] for league in candidate.sports},
        only_events,
        holdout_access,
    )
    return score(
        candidate,
        start,
        end,
        replayed,
        family_versions,
        now or datetime.now(UTC),
        n_boot,
        seed,
    )


def score(
    candidate: MechanicalCandidate,
    start: date,
    end: date,
    replayed: Replayed,
    family_versions: int,
    now: datetime,
    n_boot: int,
    seed: int,
) -> BacktestResult:
    """Summarise the replayed bets with the forward-test report's code, as one series at one
    family size for all checkpoints: a replay has no placement history to count versions
    against."""
    bets = replayed.bets
    summary = summarize_candidate(
        candidate.name,
        candidate.version,
        candidate,
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
        coverage=coverage_of(replayed),
        summary=summary,
        overall=checkpoint_at(with_close, len(with_close), family_versions, n_boot, seed)
        if with_close
        else None,
        range_events=replayed.range_events,
        tuning=replayed.tuning,
        refits=replayed.refits,
        library_versions=library_versions() if replayed.model else None,
        league_coverage=league_coverage(replayed),
    )
