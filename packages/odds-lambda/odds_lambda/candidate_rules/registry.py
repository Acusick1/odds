"""The rule registry: each rule's name, its spec and the markets each sport's rules decide.

No rule imports this module, so a rule's hash does not cover the registry or the other rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.sports import SportKey
from odds_core.vocabulary import Market
from pydantic import ValidationError

from odds_lambda.candidate_rules.crowded_gap import CROWDED_GAP
from odds_lambda.candidate_rules.exchange_steam import EXCHANGE_STEAM
from odds_lambda.candidate_rules.favourite_price import FAVOURITE_PRICE
from odds_lambda.candidate_rules.gap import EXCHANGE_GAP, POLYMARKET_GAP
from odds_lambda.candidate_rules.last_result_overreaction import LAST_RESULT_OVERREACTION
from odds_lambda.candidate_rules.retail_ev_model import RETAIL_EV_MODEL
from odds_lambda.candidate_rules.retail_quorum_lag import RETAIL_QUORUM_LAG
from odds_lambda.candidate_rules.rules import ModelRuleSpec, RuleParams, RuleSpec
from odds_lambda.candidate_rules.single_book_outlier import SINGLE_BOOK_OUTLIER
from odds_lambda.candidate_rules.subsidised_outcome import SUBSIDISED_OUTCOME

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

RULES: dict[str, RuleSpec | ModelRuleSpec] = {
    "exchange_gap": EXCHANGE_GAP,
    "exchange_steam": EXCHANGE_STEAM,
    "team_news_lag": EXCHANGE_STEAM,
    "polymarket_gap": POLYMARKET_GAP,
    "retail_ev_model": RETAIL_EV_MODEL,
    "retail_quorum_lag": RETAIL_QUORUM_LAG,
    "single_book_outlier": SINGLE_BOOK_OUTLIER,
    "subsidised_outcome": SUBSIDISED_OUTCOME,
    "favourite_price": FAVOURITE_PRICE,
    "crowded_gap": CROWDED_GAP,
    "last_result_overreaction": LAST_RESULT_OVERREACTION,
}


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
    for sport in candidate.sports:
        sport_markets = SPORT_MARKETS.get(sport, frozenset())
        if candidate.market not in sport_markets:
            raise ValueError(
                f"candidate {candidate.name!r}: no rule decides market "
                f"{candidate.market.value!r} for sport {sport!r}, its markets "
                f"{sorted(m.value for m in sport_markets)}"
            )
    if isinstance(spec, ModelRuleSpec) and candidate.market not in spec.markets:
        raise ValueError(
            f"candidate {candidate.name!r}: rule {candidate.rule!r} has no model for market "
            f"{candidate.market.value!r}, model markets {sorted(m.value for m in spec.markets)}"
        )
    for sport in candidate.sports:
        sport_markets = SPORT_MARKETS.get(sport, frozenset())
        if candidate.market in spec.also_markets or not spec.also_markets <= sport_markets:
            raise ValueError(
                f"candidate {candidate.name!r}: rule {candidate.rule!r} also reads "
                f"{sorted(m.value for m in spec.also_markets)}, which must be markets other than "
                f"{candidate.market.value!r} that sport {sport!r} decides, "
                f"{sorted(m.value for m in sport_markets)}"
            )
    try:
        params = spec.params_model.model_validate(candidate.params)
    except ValidationError as exc:
        raise ValueError(
            f"candidate {candidate.name!r}: invalid params for rule {candidate.rule!r}: {exc}"
        ) from exc
    return spec, params


def input_leads(candidates: Sequence[Candidate], sport: str) -> list[timedelta]:
    """How long before kick-off the active mechanical candidates that bet in ``sport`` read
    the market: one lead per decision time, and one per lookback point of each decision time.

    A candidate whose rule or params do not resolve is left out; the run-candidates job
    alerts on it.
    """
    leads: list[timedelta] = []
    for candidate in candidates:
        if (
            not isinstance(candidate, MechanicalCandidate)
            or sport not in candidate.sports
            or candidate.status != CandidateStatus.ACTIVE
        ):
            continue
        try:
            _, params = resolve_rule(candidate)
        except ValueError:
            continue
        for minutes in params.minutes_before_kickoff:
            leads.append(timedelta(minutes=minutes))
            leads.extend(
                timedelta(minutes=minutes + point.minutes_before_decision)
                for point in params.lookbacks
            )
    return leads
