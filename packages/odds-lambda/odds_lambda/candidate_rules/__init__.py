"""Mechanical candidate rules, shared by the live runner and the replay backtest."""

from odds_lambda.candidate_rules.market_state import (
    MarketState,
    build_market_state,
    load_market_state,
)
from odds_lambda.candidate_rules.rules import (
    RULES,
    BetIntent,
    ExchangeGapParams,
    RuleParams,
    RuleSpec,
    apply_rule,
    decision_leads,
    decision_time,
    exchange_gap,
    max_input_age,
    resolve_rule,
    rule_sha256,
    run_rule,
    validate_candidate_rules,
)

__all__ = [
    "RULES",
    "BetIntent",
    "ExchangeGapParams",
    "MarketState",
    "RuleParams",
    "RuleSpec",
    "apply_rule",
    "build_market_state",
    "decision_leads",
    "decision_time",
    "exchange_gap",
    "load_market_state",
    "max_input_age",
    "resolve_rule",
    "rule_sha256",
    "run_rule",
    "validate_candidate_rules",
]
