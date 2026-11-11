"""Rule pinning: the hash of the code a rule's bets depend on, and the check that each mechanical
candidate records it.

Kept out of the package's ``__init__`` so the hasher, which reads module sources through
``importlib``, is not itself in any rule's import closure.
"""

from __future__ import annotations

from collections.abc import Sequence

from odds_core.candidates import Candidate, MechanicalCandidate
from odds_core.close import close_spec

from odds_lambda.candidate_rules.code_hash import code_sha256
from odds_lambda.candidate_rules.rules import RULES, ModelRuleSpec, resolve_rule, run_rule

# Imported by every rule's decision: the market state, rule and once-per-selection filter, and
# the alias maps by which the writers give price rows the book keys the state selects.
RULE_ROOTS: tuple[object | str, ...] = (
    run_rule,
    "odds_core.bookmakers.BOOKMAKER_ALIASES_BY_SOURCE",
)


def rule_sha256(rule_name: str) -> str:
    """Hash of every first-party module in the import closure of the rule function, its params
    model, ``RULE_ROOTS`` and a model rule's ``model_roots`` (``code_hash.code_sha256``), then
    the rule's sharp inputs, extra markets and fixture inputs."""
    spec = RULES[rule_name]
    roots: list[object | str] = [spec.function, spec.params_model, *RULE_ROOTS]
    if isinstance(spec, ModelRuleSpec):
        roots += spec.model_roots
    return code_sha256(
        roots,
        [
            f"sharp_inputs={sorted(i.value for i in spec.sharp_inputs)}",
            f"also_markets={sorted(m.value for m in spec.also_markets)}",
            f"fixture_inputs={sorted(i.value for i in spec.fixture_inputs)}",
        ],
    )


def validate_candidate_rules(candidates: Sequence[Candidate]) -> None:
    """Raise ``ValueError`` naming a mechanical candidate that is invalid, bets in a league
    whose market has no close, or whose rule code changed."""
    for candidate in candidates:
        if not isinstance(candidate, MechanicalCandidate):
            continue
        resolve_rule(candidate)
        if unclosed := [s for s in candidate.sports if close_spec(s, candidate.market) is None]:
            raise ValueError(
                f"candidate {candidate.name!r}: {', '.join(unclosed)} has no close for market "
                f"{candidate.market.value!r} (`odds_core.close.CLOSES`), so its bets could not "
                "be scored"
            )
        if candidate.rule_sha256 != (current := rule_sha256(candidate.rule)):
            raise ValueError(
                f"candidate {candidate.name!r}: the code of rule {candidate.rule!r} changed "
                f"(recorded rule_sha256 {candidate.rule_sha256}, current {current}); "
                f"a new candidate version must record the new hash "
                f"(`odds candidates rule-hash {candidate.rule}` prints it)"
            )
