"""Mechanical betting rules: a market state in, bets out."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from odds_core.candidates import Candidate, MechanicalCandidate
from odds_core.market_prices import Outcome, market_outcomes
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, ValidationError

from odds_lambda.candidate_rules.market_state import MarketState


@dataclass(frozen=True)
class BetIntent:
    outcome: Outcome
    bookmaker: str
    price_decimal: float
    stake: float


class RuleParams(BaseModel):
    """Params every rule takes; ``minutes_before_kickoff`` sets its decision time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    minutes_before_kickoff: PositiveInt


def decision_time(kickoff: datetime, params: RuleParams) -> datetime:
    return kickoff - timedelta(minutes=params.minutes_before_kickoff)


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
            for book, quote in state.retail.items()
            if (price := quote.prices.get(outcome)) is not None
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

# A rule a registered candidate references never changes; a changed rule is a new name.
RULE_SOURCE_HASHES: dict[str, str] = {
    "exchange_gap": "2c7f28457bfe9680dab68fca819fa97fd4bfbf157d671f7c6effc80198451ca7",
}


def rule_source_hash(name: str) -> str:
    return hashlib.sha256(inspect.getsource(RULES[name].function).encode()).hexdigest()


def changed_pinned_rules(names: Iterable[str]) -> list[str]:
    """The rules among ``names`` whose source no longer matches its pinned hash."""
    return sorted(n for n in set(names) if rule_source_hash(n) != RULE_SOURCE_HASHES[n])


def validate_candidate_rules(candidates: Sequence[Candidate]) -> None:
    """Raise ``ValueError`` naming a mechanical candidate whose rule or params are invalid."""
    for candidate in candidates:
        if not isinstance(candidate, MechanicalCandidate):
            continue
        spec = RULES.get(candidate.rule)
        if spec is None:
            raise ValueError(
                f"candidate {candidate.name!r}: unknown rule {candidate.rule!r}, "
                f"known rules {sorted(RULES)}"
            )
        try:
            spec.params_model.model_validate(candidate.params)
        except ValidationError as exc:
            raise ValueError(
                f"candidate {candidate.name!r}: invalid params for rule {candidate.rule!r}: {exc}"
            ) from exc
