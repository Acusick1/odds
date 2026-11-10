"""Mechanical betting rules: a market state in, bets out."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Literal

from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event
from odds_core.sports import SportKey
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
_FOOTBALL_MARKETS = frozenset({Market.ONE_X_TWO, Market.TOTALS})
# The markets a candidate of each sport can decide.
SPORT_MARKETS: dict[SportKey, frozenset[Market]] = {
    "soccer_epl": _FOOTBALL_MARKETS,
    "soccer_efl_champ": _FOOTBALL_MARKETS,
    "soccer_spain_la_liga": _FOOTBALL_MARKETS,
    "soccer_italy_serie_a": _FOOTBALL_MARKETS,
    "soccer_germany_bundesliga": _FOOTBALL_MARKETS,
    "soccer_france_ligue_one": _FOOTBALL_MARKETS,
    "baseball_mlb": frozenset({Market.H2H, Market.TOTALS}),
    "basketball_nba": frozenset({Market.H2H}),
}


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


class GapParams(RuleParams):
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


def gap_bets(
    state: MarketState,
    fairs: Mapping[Line, Mapping[Outcome, float]],
    params: GapParams,
) -> list[BetIntent]:
    """On each line of ``fairs``, back the longest UK retail price of each outcome on that line
    whose EV against the line's fair price exceeds the threshold."""
    bets: list[BetIntent] = []
    for line, fair in fairs.items():
        for outcome in market_outcomes(state.market):
            longest = longest_retail(state.retail.get(line, {}), outcome)
            if longest is None:
                continue
            price, book = longest
            if fair[outcome] * price - 1 > params.threshold:
                bets.append(BetIntent(outcome, book, price, params.stake, line))
    return bets


def exchange_gap(state: MarketState, params: GapParams) -> list[BetIntent]:
    """Bet the gap to the exchange fair price on each line with one."""
    return gap_bets(state, {line: p.fair for line, p in state.exchange.items()}, params)


def polymarket_gap(state: MarketState, params: GapParams) -> list[BetIntent]:
    """Bet the gap to Polymarket's fair price on each line with one."""
    return gap_bets(state, {line: p.fair for line, p in state.polymarket.items()}, params)


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
    "exchange_gap": RuleSpec(exchange_gap, GapParams, frozenset({SharpInput.EXCHANGE})),
    "polymarket_gap": RuleSpec(polymarket_gap, GapParams, frozenset({SharpInput.POLYMARKET})),
    "retail_ev_model": ModelRuleSpec(
        retail_ev_model,
        RetailEvModelParams,
        frozenset({SharpInput.EXCHANGE}),
        ("odds_analytics.candidate_model.ModelReplay",),
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
    """The leagues whose kick-offs a candidate's rule counts as concurrent: its own sport's."""
    return (candidate.sport,)


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
    if candidate.market in spec.also_markets or not spec.also_markets <= sport_markets:
        raise ValueError(
            f"candidate {candidate.name!r}: rule {candidate.rule!r} also reads "
            f"{sorted(m.value for m in spec.also_markets)}, which must be markets other than "
            f"{candidate.market.value!r} that sport {candidate.sport!r} decides, "
            f"{sorted(m.value for m in sport_markets)}"
        )
    try:
        params = spec.params_model.model_validate(candidate.params)
    except ValidationError as exc:
        raise ValueError(
            f"candidate {candidate.name!r}: invalid params for rule {candidate.rule!r}: {exc}"
        ) from exc
    return spec, params


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
