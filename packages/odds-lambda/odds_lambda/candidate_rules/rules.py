"""The framework every mechanical rule shares: its params, its bets, its spec, the helpers rules
bet through, and the one path by which a rule decides.

Rule modules import this; the registry (``odds_lambda.candidate_rules.registry``) imports the rule
modules, and no rule imports the registry, so adding a rule leaves every other rule's closure
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import partial
from typing import Any

from odds_core.candidates import MechanicalCandidate
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event
from odds_core.vocabulary import Market
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.candidate_rules.fixture_context import FixtureInput, load_fixture_context
from odds_lambda.candidate_rules.inputs import SharpInput
from odds_lambda.candidate_rules.market_state import (
    Line,
    Lookback,
    MarketState,
    load_market_state,
)

# What a bet is on: an outcome on a line, the line None in a market without lines.
Selection = tuple[Outcome, Line]


@dataclass(frozen=True)
class BetIntent:
    outcome: Outcome
    bookmaker: str
    price_decimal: float
    stake: float
    line: Line = None

    @property
    def selection(self) -> Selection:
        return (self.outcome, self.line)


class LookbackParams(BaseModel):
    """A lookback point: ``minutes_before_decision`` before each decision time, whose state may
    be at most ``max_input_age_minutes`` old at that point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    minutes_before_decision: PositiveInt
    max_input_age_minutes: PositiveInt


class RuleParams(BaseModel):
    """Params every rule takes.

    ``minutes_before_kickoff`` lists the decision times, held earliest first, and a decision
    whose latest snapshot is older than ``max_input_age_minutes`` at it has no inputs.
    ``lookbacks`` lists the lookback points, held nearest first, each with its own max input
    age; a decision with no input at any of them has no inputs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    minutes_before_kickoff: tuple[PositiveInt, ...] = Field(min_length=1)
    max_input_age_minutes: PositiveInt
    lookbacks: tuple[LookbackParams, ...] = ()

    @field_validator("lookbacks")
    @classmethod
    def _distinct_nearest_first(
        cls, points: tuple[LookbackParams, ...]
    ) -> tuple[LookbackParams, ...]:
        minutes = [p.minutes_before_decision for p in points]
        if len(set(minutes)) != len(minutes):
            raise ValueError(f"lookback points must be distinct, got {minutes}")
        return tuple(sorted(points, key=lambda p: p.minutes_before_decision))

    @field_validator("minutes_before_kickoff")
    @classmethod
    def _distinct_earliest_first(cls, minutes: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(minutes)) != len(minutes):
            raise ValueError(f"decision times must be distinct, got {list(minutes)}")
        return tuple(sorted(minutes, reverse=True))


def decision_times(kickoff: datetime, params: RuleParams) -> list[tuple[int, datetime]]:
    """Each decision's minutes before kick-off and its time, earliest first."""
    return [
        (minutes, kickoff - timedelta(minutes=minutes)) for minutes in params.minutes_before_kickoff
    ]


def max_input_age(params: RuleParams) -> timedelta:
    return timedelta(minutes=params.max_input_age_minutes)


def lookback_points(params: RuleParams) -> tuple[Lookback, ...]:
    return tuple(
        Lookback(p.minutes_before_decision, timedelta(minutes=p.max_input_age_minutes))
        for p in params.lookbacks
    )


def longest_retail(
    retail: Mapping[str, Mapping[Outcome, float]], outcome: Outcome
) -> tuple[float, str] | None:
    """The longest retail price of ``outcome`` and its book, ties to the first book by name."""
    quotes = [
        (price, book)
        for book, prices in retail.items()
        if (price := prices.get(outcome)) is not None
    ]
    return min(quotes, key=lambda q: (-q[0], q[1])) if quotes else None


def gap_bets(
    state: MarketState,
    fairs: Mapping[Line, Mapping[Outcome, float]],
    threshold: float,
    stake: float,
) -> list[BetIntent]:
    """On each line of ``fairs``, back the longest UK retail price of each outcome on that line
    whose EV against the line's fair price exceeds ``threshold``."""
    bets: list[BetIntent] = []
    for line, fair in fairs.items():
        for outcome in market_outcomes(state.market):
            longest = longest_retail(state.retail.get(line, {}), outcome)
            if longest is None:
                continue
            price, book = longest
            if fair[outcome] * price - 1 > threshold:
                bets.append(BetIntent(outcome, book, price, stake, line))
    return bets


def best_bet(market: Market, scored: Iterable[tuple[BetIntent, float]]) -> list[BetIntent]:
    """The one bet of ``scored``, qualifying bets in ``market`` with their estimated EV, of the
    highest EV; empty when there is none. Ties go to the lowest line, then the outcome first in
    ``market_outcomes`` order, then the first bookmaker key."""
    order = {outcome: i for i, outcome in enumerate(market_outcomes(market))}

    def rank(item: tuple[BetIntent, float]) -> tuple[float, float, int, str]:
        bet, ev = item
        return (
            -ev,
            float("-inf") if bet.line is None else bet.line,
            order[bet.outcome],
            bet.bookmaker,
        )

    best = min(scored, key=rank, default=None)
    return [] if best is None else [best[0]]


@dataclass(frozen=True)
class RuleSpec:
    """``sharp_inputs`` are the sharp prices the rule needs: a decision without each of them
    has no inputs, and the exchange makes the retail snapshot one that also quotes it.
    ``also_markets`` are extra markets the rule reads, read-only, in ``MarketState.also``: a
    decision without an input for each of them has no inputs. ``fixture_inputs`` are the
    fixture inputs the rule reads, in ``MarketState.fixture``: a decision with a team whose
    previous result is not known has no inputs."""

    # Each rule takes its own params model, which a shared Callable cannot express.
    function: Callable[[MarketState, Any], list[BetIntent]]
    params_model: type[RuleParams]
    sharp_inputs: frozenset[SharpInput]
    also_markets: frozenset[Market] = frozenset()
    fixture_inputs: frozenset[FixtureInput] = frozenset()


# Predicted EV of each (outcome, UK retail book) quote in a state of a market without lines.
EvPredictor = Callable[[MarketState], Mapping[tuple[Outcome, str], float]]


@dataclass(frozen=True)
class ModelRuleSpec:
    """A rule whose bets need a fitted model: ``function(state, params, predict)``.

    Only the replay backtest fits one (``odds_analytics.candidate_model``) and binds it; the
    live runner and the parity check refuse a model rule. ``model_roots`` name, as dotted
    paths because this package does not depend on them, the code that builds, tunes, fits and
    queries the model. ``markets`` are the markets the model is defined for.
    """

    function: Callable[[MarketState, Any, EvPredictor], list[BetIntent]]
    params_model: type[RuleParams]
    sharp_inputs: frozenset[SharpInput]
    model_roots: tuple[str, ...]
    markets: frozenset[Market]
    also_markets: frozenset[Market] = frozenset()
    fixture_inputs: frozenset[FixtureInput] = frozenset()

    def bind(self, predict: EvPredictor) -> RuleSpec:
        return RuleSpec(
            partial(self.function, predict=predict),
            self.params_model,
            self.sharp_inputs,
            self.also_markets,
            self.fixture_inputs,
        )


async def run_rule(
    session: AsyncSession,
    spec: RuleSpec,
    params: RuleParams,
    event: Event,
    market: Market,
    at: datetime,
    known_at: datetime | None,
    kickoff: datetime | None = None,
    bet_selections: Collection[Selection] = frozenset(),
    fixture_leagues: Collection[str] | None = None,
) -> tuple[MarketState, list[BetIntent]]:
    """The state of ``event``'s ``market`` at ``at`` as known at ``known_at``, and the bets the
    rule takes.

    The live runner, the replay backtest and the parity check all decide through this.
    ``kickoff`` overrides the event's current kick-off with the one a recorded decision saw.
    ``fixture_leagues`` are the leagues a rule's concurrent kick-offs count, the event's own
    by default (``candidate_leagues``). The returned state's ``snapshot_id`` and
    ``input_event_ids`` are the decision's inputs.
    """
    state = await load_market_state(
        session,
        event,
        market,
        at,
        max_input_age(params),
        spec.sharp_inputs,
        known_at,
        also_markets=spec.also_markets,
        lookbacks=lookback_points(params),
    )
    state = await add_fixture_context(
        session, spec, state, event, at, known_at, kickoff, fixture_leagues
    )
    return apply_rule(spec, params, state, kickoff, bet_selections)


def candidate_leagues(candidate: MechanicalCandidate) -> tuple[str, ...]:
    """The leagues whose kick-offs a candidate's rule counts as concurrent: every league it bets
    in."""
    return tuple(candidate.sports)


async def add_fixture_context(
    session: AsyncSession,
    spec: RuleSpec,
    state: MarketState,
    event: Event,
    at: datetime,
    known_at: datetime | None,
    kickoff: datetime | None = None,
    fixture_leagues: Collection[str] | None = None,
) -> MarketState:
    """``state`` with the fixture inputs ``spec`` declares, as known at ``known_at``; ``state``
    itself for a rule that declares none or when it has no inputs.

    A team whose previous match has no result known at ``at`` leaves the state without inputs,
    with the context kept so the run records why (``MarketState.missing_fixture``).
    """
    if not spec.fixture_inputs or not state.has_input:
        return state
    fixture = await load_fixture_context(
        session,
        event,
        kickoff or event.commence_time,
        at,
        spec.fixture_inputs,
        fixture_leagues or (event.sport_key,),
        known_at,
    )
    if fixture.unknown_results:
        return MarketState(
            event_id=state.event_id,
            sport_key=state.sport_key,
            market=state.market,
            kickoff=state.kickoff,
            decision_time=state.decision_time,
            fixture=fixture,
        )
    return replace(state, fixture=fixture)


def apply_rule(
    spec: RuleSpec,
    params: RuleParams,
    state: MarketState,
    kickoff: datetime | None = None,
    bet_selections: Collection[Selection] = frozenset(),
) -> tuple[MarketState, list[BetIntent]]:
    """The rule's bets on ``state``, whose kick-off ``kickoff`` replaces when given.

    A state without inputs gives no bets, whatever the rule. ``bet_selections`` are the
    selections the candidate version bet on this match at an earlier decision time; each
    selection is bet at most once per match, so the rule's bets on them are dropped.
    """
    if kickoff is not None:
        state = replace(state, kickoff=kickoff)
    if not state.has_input:
        return state, []
    return state, [b for b in spec.function(state, params) if b.selection not in bet_selections]
