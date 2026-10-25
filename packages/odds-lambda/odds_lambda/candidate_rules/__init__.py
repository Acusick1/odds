"""Mechanical candidate rules, shared by the live runner and the replay backtest."""

from odds_lambda.candidate_rules.market_state import (
    MAX_RETAIL_AGE,
    MarketState,
    RetailQuote,
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
    decision_time,
    exchange_gap,
    resolve_rule,
    rule_sha256,
    run_rule,
    validate_candidate_rules,
)

__all__ = [
    "MAX_RETAIL_AGE",
    "RULES",
    "BetIntent",
    "ExchangeGapParams",
    "MarketState",
    "RetailQuote",
    "RuleParams",
    "RuleSpec",
    "apply_rule",
    "build_market_state",
    "decision_time",
    "exchange_gap",
    "load_market_state",
    "resolve_rule",
    "rule_sha256",
    "run_rule",
    "validate_candidate_rules",
]
