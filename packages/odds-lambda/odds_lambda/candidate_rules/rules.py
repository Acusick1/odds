"""Mechanical betting rules: a market state in, bets out."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import partial
from types import ModuleType
from typing import Any, Literal

from odds_core import as_of, bookmakers, exchange_price
from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event
from odds_core.vocabulary import Market
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.candidate_rules import inputs, market_state
from odds_lambda.candidate_rules.market_state import Line, MarketState, load_market_state

# What a bet is on: an outcome on a line, the line None in a market without lines.
Selection = tuple[Outcome, Line]
# The markets a candidate of each sport can decide.
SPORT_MARKETS: dict[str, frozenset[Market]] = {
    "soccer_epl": frozenset({Market.ONE_X_TWO, Market.TOTALS}),
    "baseball_mlb": frozenset({Market.H2H, Market.TOTALS}),
}
# The markets a candidate's rule can decide, in a fixed order.
RULE_MARKETS: tuple[Market, ...] = tuple(
    sorted({market for markets in SPORT_MARKETS.values() for market in markets})
)


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


def exchange_gap(state: MarketState, params: ExchangeGapParams) -> list[BetIntent]:
    """On each line with an exchange price, back the longest UK retail price of each outcome on
    that line whose EV against the exchange fair price exceeds the threshold."""
    bets: list[BetIntent] = []
    for line, exchange in state.exchange.items():
        for outcome in market_outcomes(state.market):
            longest = longest_retail(state.retail.get(line, {}), outcome)
            if longest is None:
                continue
            price, book = longest
            if exchange.fair[outcome] * price - 1 > params.threshold:
                bets.append(BetIntent(outcome, book, price, params.stake, line))
    return bets


# The modules that load a market state's quotes, pick them as of t and price the exchange.
RULE_INPUT_MODULES: tuple[ModuleType, ...] = (inputs, market_state, as_of, exchange_price)


def rule_input_constants() -> list[tuple[str, str]]:
    """The values outside ``RULE_INPUT_MODULES``, in a fixed order and a stable text form, that
    pick a market state's prices: the UK retail roster, the alias maps the writers give each
    retail source's price rows their book keys by, and the outcomes of each market a rule
    can decide."""
    return [
        ("UK_RETAIL_BOOKMAKERS", repr(sorted(bookmakers.UK_RETAIL_BOOKMAKERS))),
        *(
            (
                f"BOOKMAKER_ALIASES_BY_SOURCE[{source.value}]",
                repr(sorted(bookmakers.BOOKMAKER_ALIASES_BY_SOURCE[source].items())),
            )
            for source in inputs.RETAIL_SOURCES
        ),
        *(
            (f"market_outcomes({market})", repr([o.value for o in market_outcomes(market)]))
            for market in RULE_MARKETS
        ),
    ]


@dataclass(frozen=True)
class RuleSpec:
    # Each rule takes its own params model, which a shared Callable cannot express.
    function: Callable[[MarketState, Any], list[BetIntent]]
    params_model: type[RuleParams]


# Predicted EV of each (outcome, UK retail book) quote in a state of a market without lines.
EvPredictor = Callable[[MarketState], Mapping[tuple[Outcome, str], float]]


@dataclass(frozen=True)
class ModelRuleSpec:
    """A rule whose bets need a fitted model: ``function(state, params, predict)``.

    Only the replay backtest fits one (``odds_analytics.candidate_model``) and binds it; the
    live runner and the parity check refuse a model rule. ``model_modules`` name the modules
    that build, fit and query the model. They are hashed by name because this package does not
    depend on them. ``markets`` are the markets the model is defined for.
    """

    function: Callable[[MarketState, Any, EvPredictor], list[BetIntent]]
    params_model: type[RuleParams]
    model_modules: tuple[str, ...]
    markets: frozenset[Market]

    def bind(self, predict: EvPredictor) -> RuleSpec:
        return RuleSpec(partial(self.function, predict=predict), self.params_model)


class RetailEvModelParams(RuleParams):
    """``model`` names the model. A replay leaves the first ``training_matches`` of its range
    to training only and refits before every ``refit_every_matches`` scored matches.

    Its hyperparameters are tuned once, on the training matches alone: ``tuning_trials``
    trials seeded with ``tuning_seed`` (also the model's random state), scored on expanding
    walk-forward folds that first train on ``tuning_min_train_matches`` matches and then
    validate ``tuning_step_matches`` matches at a time.
    """

    model: Literal["retail_ev_price_only"]
    threshold: PositiveFloat
    training_matches: PositiveInt
    refit_every_matches: PositiveInt
    tuning_trials: PositiveInt
    tuning_seed: int
    tuning_min_train_matches: PositiveInt
    tuning_step_matches: PositiveInt
    stake: PositiveFloat = 1.0

    @model_validator(mode="after")
    def _folds_fit_the_training_matches(self) -> RetailEvModelParams:
        if self.tuning_min_train_matches >= self.training_matches:
            raise ValueError(
                f"tuning_min_train_matches {self.tuning_min_train_matches} leaves no match of "
                f"the {self.training_matches} training matches to validate"
            )
        return self


def retail_ev_model(
    state: MarketState, params: RetailEvModelParams, predict: EvPredictor
) -> list[BetIntent]:
    """Back the longest UK retail price of each outcome whose predicted EV exceeds the threshold."""
    if None not in state.exchange:
        return []
    predicted = predict(state)
    bets: list[BetIntent] = []
    for outcome in market_outcomes(state.market):
        longest = longest_retail(state.retail.get(None, {}), outcome)
        if longest is None:
            continue
        price, book = longest
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
            # The close every training target is measured against.
            "odds_core.betfair_close",
            # The tuning objective, the search-space draws and the search-space model.
            "odds_analytics.retail_ev_metrics",
            "odds_analytics.training.tuner",
            "odds_analytics.training.config",
        ),
        frozenset({Market.ONE_X_TWO, Market.H2H}),
    ),
}


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
) -> tuple[MarketState, list[BetIntent]]:
    """The state of ``event``'s ``market`` at ``at`` as known at ``known_at``, and the bets the
    rule takes.

    The live runner, the replay backtest and the parity check all decide through this.
    ``kickoff`` overrides the event's current kick-off with the one a recorded decision saw.
    The returned state's ``snapshot_id`` is the decision's input.
    """
    state = await load_market_state(session, event, market, at, max_input_age(params), known_at)
    return apply_rule(spec, params, state, kickoff, bet_selections)


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


def rule_sha256(rule_name: str) -> str:
    """Hash of the code and values a rule's bets depend on, in a fixed order.

    The source of the rule function, its params model, the params every rule takes,
    ``longest_retail``, ``BetIntent``, ``apply_rule`` (which drops selections already bet) and
    the ``RULE_INPUT_MODULES``, then the values of ``rule_input_constants``. A model rule adds
    the source of its ``model_modules``.
    ``odds_core.quotes``, which other readers share, is not hashed.
    """
    spec = RULES[rule_name]
    digest = hashlib.sha256()
    sources = [
        inspect.getsource(spec.function),
        inspect.getsource(spec.params_model),
        inspect.getsource(RuleParams),
        inspect.getsource(longest_retail),
        inspect.getsource(BetIntent),
        inspect.getsource(apply_rule),
        *(inspect.getsource(module) for module in RULE_INPUT_MODULES),
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
    sport_markets = SPORT_MARKETS.get(candidate.sport, frozenset())
    if candidate.market not in sport_markets:
        raise ValueError(
            f"candidate {candidate.name!r}: no rule decides market {candidate.market.value!r} "
            f"for sport {candidate.sport!r}, its markets {sorted(m.value for m in sport_markets)}"
        )
    if isinstance(spec, ModelRuleSpec) and candidate.market not in spec.markets:
        raise ValueError(
            f"candidate {candidate.name!r}: rule {candidate.rule!r} has no model for market "
            f"{candidate.market.value!r}, model markets {sorted(m.value for m in spec.markets)}"
        )
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
