"""Mechanical candidate rules, shared by the live runner and the replay backtest.

This package's ``__init__`` is in every rule's hash, so it exports only the framework every
rule's closure contains: never the registry (``registry``) or a rule module.
"""

from odds_lambda.candidate_rules.fixture_context import (
    FixtureContext,
    FixtureInput,
    PreviousResult,
    TeamResult,
    load_fixture_context,
)
from odds_lambda.candidate_rules.inputs import SharpInput
from odds_lambda.candidate_rules.market_state import (
    Line,
    Lookback,
    MarketState,
    PolymarketPrice,
    build_market_state,
    input_window,
    is_bettable_line,
    load_market_state,
)
from odds_lambda.candidate_rules.rules import (
    BetIntent,
    EstimateBasis,
    EvPredictor,
    LookbackParams,
    ModelRuleSpec,
    RuleParams,
    RuleSpec,
    Selection,
    add_fixture_context,
    apply_rule,
    best_bet,
    candidate_leagues,
    decision_times,
    gap_bets,
    lookback_points,
    max_input_age,
    run_rule,
)

__all__ = [
    "BetIntent",
    "EstimateBasis",
    "EvPredictor",
    "FixtureContext",
    "FixtureInput",
    "LookbackParams",
    "Line",
    "Lookback",
    "MarketState",
    "ModelRuleSpec",
    "PolymarketPrice",
    "PreviousResult",
    "RuleParams",
    "RuleSpec",
    "Selection",
    "SharpInput",
    "TeamResult",
    "add_fixture_context",
    "apply_rule",
    "best_bet",
    "candidate_leagues",
    "build_market_state",
    "decision_times",
    "gap_bets",
    "input_window",
    "is_bettable_line",
    "load_fixture_context",
    "load_market_state",
    "lookback_points",
    "max_input_age",
    "run_rule",
]
