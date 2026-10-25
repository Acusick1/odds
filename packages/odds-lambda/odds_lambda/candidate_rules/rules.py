"""Mechanical betting rules: a market state in, bets out."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from odds_core import as_of
from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, ValidationError
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

    ``minutes_before_kickoff`` sets the decision time, and a decision whose latest snapshot
    is older than ``max_input_age_minutes`` at it has no inputs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    minutes_before_kickoff: PositiveInt
    max_input_age_minutes: PositiveInt


def decision_time(kickoff: datetime, params: RuleParams) -> datetime:
    return kickoff - timedelta(minutes=params.minutes_before_kickoff)


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


@dataclass(frozen=True)
class RuleSpec:
    # Each rule takes its own params model, which a shared Callable cannot express.
    function: Callable[[MarketState, Any], list[BetIntent]]
    params_model: type[RuleParams]


RULES: dict[str, RuleSpec] = {
    "exchange_gap": RuleSpec(exchange_gap, ExchangeGapParams),
}


async def run_rule(
    session: AsyncSession,
    spec: RuleSpec,
    params: RuleParams,
    event: Event,
    at: datetime,
    known_at: datetime | None,
    kickoff: datetime | None = None,
) -> tuple[MarketState, list[BetIntent]]:
    """The state of ``event`` at ``at`` as known at ``known_at``, and the bets the rule takes.

    The live runner, the replay backtest and the parity check all decide through this.
    ``kickoff`` overrides the event's current kick-off with the one a recorded decision saw.
    The returned state's ``snapshot_ids`` are the decision's inputs.
    """
    state = await load_market_state(session, event, at, max_input_age(params), known_at)
    return apply_rule(spec, params, state, kickoff)


def apply_rule(
    spec: RuleSpec, params: RuleParams, state: MarketState, kickoff: datetime | None = None
) -> tuple[MarketState, list[BetIntent]]:
    """The rule's bets on ``state``, whose kick-off ``kickoff`` replaces when given.

    A state without inputs gives no bets, whatever the rule.
    """
    if kickoff is not None:
        state = replace(state, kickoff=kickoff)
    if not state.has_input:
        return state, []
    return state, spec.function(state, params)


def rule_sha256(rule_name: str) -> str:
    """Hash of the code a rule's bets depend on, in a fixed order.

    The rule function, its params model, the params every rule takes, the market state
    module and the as-of module.
    """
    spec = RULES[rule_name]
    digest = hashlib.sha256()
    for source in (
        inspect.getsource(spec.function),
        inspect.getsource(spec.params_model),
        inspect.getsource(RuleParams),
        inspect.getsource(market_state),
        inspect.getsource(as_of),
    ):
        digest.update(source.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_rule(candidate: MechanicalCandidate) -> tuple[RuleSpec, RuleParams]:
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
    """How long before kick-off each active mechanical candidate of ``sport`` decides.

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
        leads.append(timedelta(minutes=params.minutes_before_kickoff))
    return leads
