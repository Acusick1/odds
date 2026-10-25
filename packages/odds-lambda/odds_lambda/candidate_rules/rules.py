"""Mechanical betting rules: a market state in, bets out."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Literal

from odds_core import as_of, betfair_close, bookmakers, market_prices
from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
)
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.candidate_rules import market_state
from odds_lambda.candidate_rules.market_state import MarketState, load_market_state, sport_market


@dataclass(frozen=True)
class BetIntent:
    outcome: Outcome
    bookmaker: str
    price_decimal: float
    stake: float


class RuleParams(BaseModel):
    """Params every rule takes.

    ``minutes_before_kickoff`` lists the decision times, held earliest first, and a decision
    whose latest snapshot is older than ``max_input_age_minutes`` at it has no inputs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    minutes_before_kickoff: tuple[PositiveInt, ...] = Field(min_length=1)
    max_input_age_minutes: PositiveInt

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


class ExchangeGapParams(RuleParams):
    threshold: PositiveFloat
    stake: PositiveFloat = 1.0


def exchange_gap(state: MarketState, params: ExchangeGapParams) -> list[BetIntent]:
    if state.exchange is None:
        return []
    bets: list[BetIntent] = []
    for outcome in market_outcomes(state.market):
        quotes = [
            (price, book)
            for book, prices in state.retail.items()
            if (price := prices.get(outcome)) is not None
        ]
        if not quotes:
            continue
        price, book = min(quotes, key=lambda q: (-q[0], q[1]))
        if state.exchange.fair[outcome] * price - 1 > params.threshold:
            bets.append(BetIntent(outcome, book, price, params.stake))
    return bets


# Code outside this package that decides which prices a market state holds.
RULE_INPUT_FUNCTIONS: tuple[Callable[..., Any], ...] = (
    betfair_close.exchange_price,
    betfair_close.complete_exchange_prices,
    betfair_close.devig,
    betfair_close.overround,
    market_prices.book_prices,
    bookmakers.canonical_bookmaker_key,
)


def rule_input_constants() -> list[tuple[str, str]]:
    """The values, in a fixed order and a stable text form, that pick a market state's prices."""
    return [
        ("UK_RETAIL_BOOKMAKERS", repr(sorted(bookmakers.UK_RETAIL_BOOKMAKERS))),
        ("BOOKMAKER_ALIASES", repr(sorted(bookmakers.BOOKMAKER_ALIASES.items()))),
        ("ODDS_API_KEY_ALIASES", repr(sorted(bookmakers.ODDS_API_KEY_ALIASES.items()))),
        ("EXCHANGE_BOOKMAKER", repr(betfair_close.EXCHANGE_BOOKMAKER)),
        ("MAX_EXCHANGE_OVERROUND", repr(betfair_close.MAX_EXCHANGE_OVERROUND)),
    ]


@dataclass(frozen=True)
class RuleSpec:
    # Each rule takes its own params model, which a shared Callable cannot express.
    function: Callable[[MarketState, Any], list[BetIntent]]
    params_model: type[RuleParams]


# Predicted EV of each (outcome, UK retail book) quote in a state.
EvPredictor = Callable[[MarketState], Mapping[tuple[Outcome, str], float]]


@dataclass(frozen=True)
class ModelRuleSpec:
    """A rule whose bets need a fitted model: ``function(state, params, predict)``.

    Only the replay backtest fits one (``odds_analytics.candidate_model``) and binds it; the
    live runner and the parity check refuse a model rule. ``model_modules`` name the modules
    that build, fit and query the model. They are hashed by name because this package does not
    depend on them.
    """

    function: Callable[[MarketState, Any, EvPredictor], list[BetIntent]]
    params_model: type[RuleParams]
    model_modules: tuple[str, ...]

    def bind(self, predict: EvPredictor) -> RuleSpec:
        return RuleSpec(partial(self.function, predict=predict), self.params_model)


class RetailEvModelParams(RuleParams):
    """``model`` names the fitted model. A replay leaves the first ``training_matches`` of its
    range to training only and refits before every ``refit_every_matches`` scored matches."""

    model: Literal["retail_ev_price_only"]
    threshold: PositiveFloat
    training_matches: PositiveInt
    refit_every_matches: PositiveInt
    stake: PositiveFloat = 1.0


def retail_ev_model(
    state: MarketState, params: RetailEvModelParams, predict: EvPredictor
) -> list[BetIntent]:
    """Back the longest UK retail price of each outcome whose predicted EV exceeds the threshold."""
    if state.exchange is None:
        return []
    predicted = predict(state)
    bets: list[BetIntent] = []
    for outcome in market_outcomes(state.market):
        quotes = [
            (price, book)
            for book, prices in state.retail.items()
            if (price := prices.get(outcome)) is not None
        ]
        if not quotes:
            continue
        price, book = min(quotes, key=lambda q: (-q[0], q[1]))
        ev = predicted.get((outcome, book))
        if ev is not None and ev > params.threshold:
            bets.append(BetIntent(outcome, book, price, params.stake))
    return bets


RULES: dict[str, RuleSpec | ModelRuleSpec] = {
    "exchange_gap": RuleSpec(exchange_gap, ExchangeGapParams),
    "retail_ev_model": ModelRuleSpec(
        retail_ev_model,
        RetailEvModelParams,
        (
            "odds_analytics.candidate_model",
            "odds_analytics.retail_ev",
            "odds_analytics.xgboost_line_movement",
        ),
    ),
}


async def run_rule(
    session: AsyncSession,
    spec: RuleSpec,
    params: RuleParams,
    event: Event,
    at: datetime,
    known_at: datetime | None,
    kickoff: datetime | None = None,
    bet_outcomes: Collection[Outcome] = frozenset(),
) -> tuple[MarketState, list[BetIntent]]:
    """The state of ``event`` at ``at`` as known at ``known_at``, and the bets the rule takes.

    The live runner, the replay backtest and the parity check all decide through this.
    ``kickoff`` overrides the event's current kick-off with the one a recorded decision saw.
    The returned state's ``snapshot_id`` is the decision's input.
    """
    state = await load_market_state(session, event, at, max_input_age(params), known_at)
    return apply_rule(spec, params, state, kickoff, bet_outcomes)


def apply_rule(
    spec: RuleSpec,
    params: RuleParams,
    state: MarketState,
    kickoff: datetime | None = None,
    bet_outcomes: Collection[Outcome] = frozenset(),
) -> tuple[MarketState, list[BetIntent]]:
    """The rule's bets on ``state``, whose kick-off ``kickoff`` replaces when given.

    A state without inputs gives no bets, whatever the rule. ``bet_outcomes`` are the
    outcomes the candidate version bet on this match at an earlier decision time; each
    outcome is bet at most once per match, so the rule's bets on them are dropped.
    """
    if kickoff is not None:
        state = replace(state, kickoff=kickoff)
    if not state.has_input:
        return state, []
    return state, [b for b in spec.function(state, params) if b.outcome not in bet_outcomes]


def rule_sha256(rule_name: str) -> str:
    """Hash of the code and values a rule's bets depend on, in a fixed order.

    The source of the rule function, its params model, the params every rule takes,
    ``apply_rule`` (which drops outcomes already bet), the market state module, the as-of
    module and the functions that read a snapshot's prices (``RULE_INPUT_FUNCTIONS``), then
    the values of ``rule_input_constants``. A model rule adds the source of its
    ``model_modules``.
    """
    spec = RULES[rule_name]
    digest = hashlib.sha256()
    sources = [
        inspect.getsource(spec.function),
        inspect.getsource(spec.params_model),
        inspect.getsource(RuleParams),
        inspect.getsource(apply_rule),
        inspect.getsource(market_state),
        inspect.getsource(as_of),
        *(inspect.getsource(function) for function in RULE_INPUT_FUNCTIONS),
        *(f"{name}={value}" for name, value in rule_input_constants()),
    ]
    if isinstance(spec, ModelRuleSpec):
        sources += [
            inspect.getsource(importlib.import_module(module)) for module in spec.model_modules
        ]
    for source in sources:
        digest.update(source.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_rule(candidate: MechanicalCandidate) -> tuple[RuleSpec | ModelRuleSpec, RuleParams]:
    """The candidate's rule and validated params; ``ValueError`` names what is invalid.

    Does not check ``rule_sha256``, which a draft may lack.
    """
    spec = RULES.get(candidate.rule)
    if spec is None:
        raise ValueError(
            f"candidate {candidate.name!r}: unknown rule {candidate.rule!r}, "
            f"known rules {sorted(RULES)}"
        )
    try:
        sport_market(candidate.sport)
    except ValueError as exc:
        raise ValueError(f"candidate {candidate.name!r}: {exc}") from exc
    try:
        params = spec.params_model.model_validate(candidate.params)
    except ValidationError as exc:
        raise ValueError(
            f"candidate {candidate.name!r}: invalid params for rule {candidate.rule!r}: {exc}"
        ) from exc
    return spec, params


def validate_candidate_rules(candidates: Sequence[Candidate]) -> None:
    """Raise ``ValueError`` naming a mechanical candidate that is invalid or whose rule code changed."""
    for candidate in candidates:
        if not isinstance(candidate, MechanicalCandidate):
            continue
        resolve_rule(candidate)
        if candidate.rule_sha256 != (current := rule_sha256(candidate.rule)):
            raise ValueError(
                f"candidate {candidate.name!r}: the code of rule {candidate.rule!r} changed "
                f"(recorded rule_sha256 {candidate.rule_sha256}, current {current}); "
                f"a new candidate version must record the new hash "
                f"(`odds candidates rule-hash {candidate.rule}` prints it)"
            )


def decision_leads(candidates: Sequence[Candidate], sport: str) -> list[timedelta]:
    """How long before kick-off the active mechanical candidates of ``sport`` decide, one
    lead per decision time.

    A candidate whose rule or params do not resolve is left out; the run-candidates job
    alerts on it.
    """
    leads: list[timedelta] = []
    for candidate in candidates:
        if (
            not isinstance(candidate, MechanicalCandidate)
            or candidate.sport != sport
            or candidate.status != CandidateStatus.ACTIVE
        ):
            continue
        try:
            _, params = resolve_rule(candidate)
        except ValueError:
            continue
        leads.extend(timedelta(minutes=minutes) for minutes in params.minutes_before_kickoff)
    return leads
