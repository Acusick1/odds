"""Unit tests for the retail consensus and the rules of round 1 that read retail books against
each other: retail-quorum lag, subsidised outcome and single-book outlier."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import pytest
from odds_core.exchange_price import ExchangePrice
from odds_core.market_prices import Outcome
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda.candidate_rules import BetIntent, MarketState
from odds_lambda.candidate_rules.retail_consensus import (
    BOOK_GROUPS,
    BookGroup,
    consensus,
    group_probabilities,
)
from odds_lambda.candidate_rules.retail_quorum_lag import (
    RetailQuorumLagParams,
    retail_quorum_lag,
)
from odds_lambda.candidate_rules.single_book_outlier import (
    SingleBookOutlierParams,
    single_book_outlier,
)
from odds_lambda.candidate_rules.subsidised_outcome import (
    SubsidisedOutcomeParams,
    subsidised_outcome,
)
from pydantic import ValidationError

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
HOME, DRAW, AWAY = Outcome.HOME, Outcome.DRAW, Outcome.AWAY
PP, LAD, WH, BETFRED, BV = BOOK_GROUPS
LOOKBACK = 30
# One book from each of four groups; Betfred is absent, as in the backfill.
GROUPED = ("paddypower", "ladbrokes", "williamhill", "betvictor")

Prices = dict[Outcome, float]


def _book(home: float, draw: float, away: float, margin: float = 0.05) -> Prices:
    """Prices whose proportional devig gives these probabilities."""
    return {o: 1 / (p * (1 + margin)) for o, p in ((HOME, home), (DRAW, draw), (AWAY, away))}


def _prices(home: float, draw: float, away: float) -> Prices:
    return {HOME: home, DRAW: draw, AWAY: away}


def _groups(*books: str, p: tuple[float, float, float]) -> dict[str, Prices]:
    return {book: _book(*p) for book in books}


def _exchange(home: float, draw: float, away: float) -> ExchangePrice:
    return ExchangePrice(
        time=T,
        source=Source.ODDSPORTAL,
        time_basis=TimeBasis.OBSERVED,
        in_play=False,
        fair={HOME: home, DRAW: draw, AWAY: away},
        overround=0.0,
    )


def _state(
    retail: dict[str, Prices],
    *,
    exchange: ExchangePrice | None = None,
    lookback: dict[str, Prices] | None = None,
) -> MarketState:
    def one(retail: dict[str, Prices], time: datetime) -> MarketState:
        return MarketState(
            event_id=1,
            sport_key="soccer_epl",
            market=Market.ONE_X_TWO,
            kickoff=KICKOFF,
            decision_time=time,
            snapshot_id=1,
            snapshot_time=time,
            retail=MappingProxyType({None: retail}),
            exchange=MappingProxyType({} if exchange is None else {None: exchange}),
        )

    state = one(retail, T)
    if lookback is None:
        return state
    then = one(lookback, T - timedelta(minutes=LOOKBACK))
    return replace(state, lookbacks=MappingProxyType({LOOKBACK: then}))


def _approx(prices: dict[Any, float]) -> dict[Any, Any]:
    return {k: pytest.approx(v) for k, v in prices.items()}


class TestRetailConsensus:
    def test_a_group_is_the_mean_of_its_books_devigged(self) -> None:
        retail = {
            "paddypower": _book(0.5, 0.3, 0.2, margin=0.05),
            "skybet": _book(0.4, 0.3, 0.3, margin=0.08),
        }
        assert group_probabilities(retail, Market.ONE_X_TWO) == {
            PP: _approx(_prices(0.45, 0.3, 0.25))
        }

    def test_a_book_without_every_outcome_is_left_out(self) -> None:
        retail = {"paddypower": _book(0.5, 0.3, 0.2), "skybet": {HOME: 1.5, DRAW: 4.0}}
        assert group_probabilities(retail, Market.ONE_X_TWO) == {
            PP: _approx(_prices(0.5, 0.3, 0.2))
        }

    def test_only_grouped_books_form_groups(self) -> None:
        retail = {"bet365": _book(0.5, 0.3, 0.2), "betway": _book(0.5, 0.3, 0.2)}
        assert group_probabilities(retail, Market.ONE_X_TWO) == {}

    def test_is_the_median_across_groups_normalised(self) -> None:
        groups: dict[BookGroup, Prices] = {
            PP: _prices(0.4, 0.35, 0.25),
            LAD: _prices(0.5, 0.25, 0.25),
            WH: _prices(0.6, 0.3, 0.1),
        }
        assert consensus(groups, "bet365") == _approx(_prices(0.5 / 1.05, 0.3 / 1.05, 0.25 / 1.05))

    def test_an_even_count_takes_the_mean_of_the_middle_two(self) -> None:
        groups: dict[BookGroup, Prices] = {
            PP: _prices(0.4, 0.3, 0.3),
            LAD: _prices(0.5, 0.3, 0.2),
            WH: _prices(0.6, 0.3, 0.1),
            BV: _prices(0.7, 0.2, 0.1),
        }
        assert consensus(groups, "bet365") == _approx(_prices(0.55, 0.3, 0.15))

    def test_leaves_out_the_targets_own_group(self) -> None:
        groups: dict[BookGroup, Prices] = {
            PP: _prices(0.9, 0.05, 0.05),
            LAD: _prices(0.5, 0.3, 0.2),
            WH: _prices(0.6, 0.25, 0.15),
            BV: _prices(0.4, 0.35, 0.25),
        }
        assert consensus(groups, "skybet") == _approx(_prices(0.5, 0.3, 0.2))
        assert consensus(groups, "bet365") == _approx(_prices(0.55, 0.275, 0.175))

    def test_a_book_in_no_group_leaves_out_none(self) -> None:
        groups: dict[BookGroup, Prices] = {
            PP: _prices(0.5, 0.3, 0.2),
            LAD: _prices(0.5, 0.3, 0.2),
            WH: _prices(0.5, 0.3, 0.2),
        }
        assert consensus(groups, "bet365") == _approx(_prices(0.5, 0.3, 0.2))

    def test_needs_three_groups_other_than_the_targets(self) -> None:
        groups: dict[BookGroup, Prices] = {
            PP: _prices(0.5, 0.3, 0.2),
            LAD: _prices(0.5, 0.3, 0.2),
            WH: _prices(0.5, 0.3, 0.2),
        }
        assert consensus(groups, "coral") is None
        assert consensus({PP: groups[PP], LAD: groups[LAD]}, "bet365") is None


NOW = (0.50, 0.27, 0.23)
THEN = (0.48, 0.28, 0.24)
LAG_PARAMS = RetailQuorumLagParams(
    threshold=0.02,
    lookbacks=[{"minutes_before_decision": LOOKBACK, "max_input_age_minutes": 10}],
    minutes_before_kickoff=(60,),
    max_input_age_minutes=10,
)
LAGGING = _prices(2.2, 3.4, 4.0)


def _lag_state(
    now: dict[str, Prices] | None = None, then: dict[str, Prices] | None = None
) -> MarketState:
    """Four groups that raised the home probability from 0.48 to 0.50, and bet365 at 2.2 at both
    points: EV 0.5 × 2.2 − 1 = 0.10 against the consensus."""
    retail = {**_groups(*GROUPED, p=NOW), "bet365": LAGGING}
    lookback = {**_groups(*GROUPED, p=THEN), "bet365": LAGGING}
    return _state({**retail, **(now or {})}, lookback={**lookback, **(then or {})})


def _without(state: MarketState, *books: str, now: bool = True, then: bool = True) -> MarketState:
    def drop(point: MarketState, dropping: bool) -> dict[str, Prices]:
        return {b: dict(p) for b, p in point.retail[None].items() if not dropping or b not in books}

    return _state(drop(state, now), lookback=drop(state.lookbacks[LOOKBACK], then))


class TestRetailQuorumLag:
    def test_backs_the_unchanged_price_the_groups_moved_past(self) -> None:
        assert retail_quorum_lag(_lag_state(), LAG_PARAMS) == [BetIntent(HOME, "bet365", 2.2, 1.0)]

    def test_a_changed_price_does_not_qualify(self) -> None:
        state = _lag_state(then={"bet365": {**LAGGING, HOME: 2.25}})
        assert retail_quorum_lag(state, LAG_PARAMS) == []

    def test_a_quote_missing_at_the_lookback_is_not_unchanged(self) -> None:
        assert retail_quorum_lag(_without(_lag_state(), "bet365", now=False), LAG_PARAMS) == []
        state = _lag_state(then={"bet365": {DRAW: 3.4, AWAY: 4.0}})
        assert retail_quorum_lag(state, LAG_PARAMS) == []

    def test_needs_three_groups_whose_probability_rose_a_point(self) -> None:
        still = _lag_state(then=_groups("betvictor", p=NOW))
        assert retail_quorum_lag(still, LAG_PARAMS) == [BetIntent(HOME, "bet365", 2.2, 1.0)]
        two = _lag_state(then=_groups("betvictor", "williamhill", p=(0.495, 0.27, 0.235)))
        assert retail_quorum_lag(two, LAG_PARAMS) == []

    def test_a_group_must_be_present_at_both_points(self) -> None:
        state = _lag_state(then=_groups("betvictor", p=NOW))
        assert retail_quorum_lag(_without(state, "williamhill", now=False), LAG_PARAMS) == []

    def test_the_targets_own_group_is_left_out(self) -> None:
        # Paddy Power lags; Sky Bet, in its group, moved with the others.
        now = {"paddypower": LAGGING, "skybet": _book(*NOW)}
        then = {"paddypower": LAGGING, "skybet": _book(*THEN)}
        state = _without(_lag_state(now, then), "bet365")
        assert retail_quorum_lag(state, LAG_PARAMS) == [BetIntent(HOME, "paddypower", 2.2, 1.0)]
        fewer = _without(_lag_state(now, {**then, **_groups("betvictor", p=NOW)}), "bet365")
        assert retail_quorum_lag(fewer, LAG_PARAMS) == []

    def test_needs_three_other_groups_for_the_consensus(self) -> None:
        state = _without(_lag_state(), "betvictor")
        assert retail_quorum_lag(state, LAG_PARAMS) == [BetIntent(HOME, "bet365", 2.2, 1.0)]
        assert retail_quorum_lag(_without(state, "williamhill", then=False), LAG_PARAMS) == []

    def test_ev_must_exceed_the_threshold(self) -> None:
        high = LAG_PARAMS.model_copy(update={"threshold": 0.11})
        assert retail_quorum_lag(_lag_state(), high) == []

    def test_bets_the_highest_ev_then_the_first_book(self) -> None:
        longer = _prices(2.3, 3.4, 4.0)
        state = _lag_state({"betway": longer}, {"betway": longer})
        assert retail_quorum_lag(state, LAG_PARAMS) == [BetIntent(HOME, "betway", 2.3, 1.0)]
        tied = _lag_state({"betway": LAGGING}, {"betway": LAGGING})
        assert retail_quorum_lag(tied, LAG_PARAMS) == [BetIntent(HOME, "bet365", 2.2, 1.0)]

    def test_takes_exactly_one_lookback(self) -> None:
        point = {"minutes_before_decision": 30, "max_input_age_minutes": 10}
        base = {"threshold": 0.02, "minutes_before_kickoff": [60], "max_input_age_minutes": 10}
        for lookbacks in ([], [point, {**point, "minutes_before_decision": 60}]):
            with pytest.raises(ValidationError):
                RetailQuorumLagParams.model_validate({**base, "lookbacks": lookbacks})
        with pytest.raises(ValidationError):
            RetailQuorumLagParams.model_validate(base)

    def test_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            LAG_PARAMS.model_validate({**LAG_PARAMS.model_dump(), "threshold": 0})


SUBSIDY_PARAMS = SubsidisedOutcomeParams(
    threshold=0.02, min_overround=0.04, minutes_before_kickoff=(60,), max_input_age_minutes=10
)
# Overround 0.059; home EV 0.5 × 2.1 − 1 = 0.05; draw and away implied 0.0425 and 0.040 above
# the consensus.
SUBSIDISED = _prices(2.1, 3.2, 3.7)


def _subsidy_state(**books: Prices) -> MarketState:
    return _state({**_groups(*GROUPED, p=NOW), **books})


class TestSubsidisedOutcome:
    def test_backs_the_outcome_priced_above_the_consensus(self) -> None:
        assert subsidised_outcome(_subsidy_state(bet365=SUBSIDISED), SUBSIDY_PARAMS) == [
            BetIntent(HOME, "bet365", 2.1, 1.0)
        ]

    def test_the_overround_must_reach_the_minimum(self) -> None:
        params = SUBSIDY_PARAMS.model_copy(update={"min_overround": 0.06})
        assert subsidised_outcome(_subsidy_state(bet365=SUBSIDISED), params) == []

    def test_ev_must_exceed_the_threshold(self) -> None:
        params = SUBSIDY_PARAMS.model_copy(update={"threshold": 0.06})
        assert subsidised_outcome(_subsidy_state(bet365=SUBSIDISED), params) == []

    def test_each_other_outcome_must_carry_a_point_of_margin(self) -> None:
        # Away implied 0.238 against a consensus of 0.23; overround still 0.059.
        state = _subsidy_state(bet365=_prices(2.1, 2.9, 4.2))
        assert subsidised_outcome(state, SUBSIDY_PARAMS) == []

    def test_the_book_must_price_every_outcome(self) -> None:
        state = _subsidy_state(bet365={HOME: 2.1, DRAW: 3.2})
        assert subsidised_outcome(state, SUBSIDY_PARAMS) == []

    def test_a_grouped_book_is_compared_with_the_other_groups(self) -> None:
        three = {b: _book(*NOW) for b in GROUPED[:3]}
        state = _state({**three, "betvictor": SUBSIDISED})
        assert subsidised_outcome(state, SUBSIDY_PARAMS) == [BetIntent(HOME, "betvictor", 2.1, 1.0)]
        two = {b: _book(*NOW) for b in GROUPED[:2]}
        assert subsidised_outcome(_state({**two, "betvictor": SUBSIDISED}), SUBSIDY_PARAMS) == []

    def test_bets_the_highest_ev_then_the_first_book(self) -> None:
        # Home EV 0.075; draw and away implied 0.053 and 0.048 above the consensus.
        higher = _prices(2.15, 3.1, 3.6)
        state = _subsidy_state(bet365=SUBSIDISED, betway=higher)
        assert subsidised_outcome(state, SUBSIDY_PARAMS) == [BetIntent(HOME, "betway", 2.15, 1.0)]
        tied = _subsidy_state(bet365=SUBSIDISED, betway=SUBSIDISED)
        assert subsidised_outcome(tied, SUBSIDY_PARAMS) == [BetIntent(HOME, "bet365", 2.1, 1.0)]

    @pytest.mark.parametrize("field", ["threshold", "min_overround"])
    def test_params_must_be_positive(self, field: str) -> None:
        with pytest.raises(ValidationError):
            SubsidisedOutcomeParams.model_validate({**SUBSIDY_PARAMS.model_dump(), field: 0})


OUTLIER_PARAMS = SingleBookOutlierParams(
    min_ratio=1.05, gap=0, minutes_before_kickoff=(60, 30), max_input_age_minutes=10
)
FAIR = _exchange(0.45, 0.28, 0.27)
# bet365's home 2.5 is 1.087 times William Hill's 2.3; EV 0.45 × 2.5 − 1 = 0.125.
OUTLIER_RETAIL = {
    "bet365": _prices(2.5, 3.0, 3.0),
    "williamhill": _prices(2.3, 3.0, 3.0),
    "coral": _prices(2.2, 3.1, 3.1),
}


class TestSingleBookOutlier:
    def test_backs_the_book_far_above_every_other(self) -> None:
        assert single_book_outlier(_state(OUTLIER_RETAIL, exchange=FAIR), OUTLIER_PARAMS) == [
            BetIntent(HOME, "bet365", 2.5, 1.0)
        ]

    def test_the_ratio_must_reach_the_minimum(self) -> None:
        params = OUTLIER_PARAMS.model_copy(update={"min_ratio": 1.09})
        assert single_book_outlier(_state(OUTLIER_RETAIL, exchange=FAIR), params) == []

    def test_a_tie_at_the_top_has_ratio_one(self) -> None:
        retail = {**OUTLIER_RETAIL, "williamhill": _prices(2.5, 3.0, 3.0)}
        assert single_book_outlier(_state(retail, exchange=FAIR), OUTLIER_PARAMS) == []

    def test_needs_a_second_book(self) -> None:
        retail = {"bet365": OUTLIER_RETAIL["bet365"]}
        assert single_book_outlier(_state(retail, exchange=FAIR), OUTLIER_PARAMS) == []

    def test_ev_against_the_exchange_must_exceed_the_gap(self) -> None:
        state = _state(OUTLIER_RETAIL, exchange=_exchange(0.39, 0.31, 0.30))
        assert single_book_outlier(state, OUTLIER_PARAMS) == []
        high = OUTLIER_PARAMS.model_copy(update={"gap": 0.13})
        assert single_book_outlier(_state(OUTLIER_RETAIL, exchange=FAIR), high) == []

    def test_no_exchange_no_bets(self) -> None:
        assert single_book_outlier(_state(OUTLIER_RETAIL), OUTLIER_PARAMS) == []

    def test_bets_the_highest_ev_then_the_first_outcome(self) -> None:
        # bet365's draw 2.5 is an outlier too.
        retail = {
            "bet365": _prices(2.5, 2.5, 3.0),
            "williamhill": _prices(2.3, 2.3, 3.0),
            "coral": _prices(2.2, 2.2, 3.1),
        }
        richer_draw = _state(retail, exchange=_exchange(0.45, 0.46, 0.09))
        assert single_book_outlier(richer_draw, OUTLIER_PARAMS) == [
            BetIntent(DRAW, "bet365", 2.5, 1.0)
        ]
        tied = _state(retail, exchange=_exchange(0.45, 0.45, 0.1))
        assert single_book_outlier(tied, OUTLIER_PARAMS) == [BetIntent(HOME, "bet365", 2.5, 1.0)]

    def test_min_ratio_must_exceed_one_and_gap_be_non_negative(self) -> None:
        base = OUTLIER_PARAMS.model_dump()
        for bad in ({"min_ratio": 1.0}, {"gap": -0.01}):
            with pytest.raises(ValidationError):
                SingleBookOutlierParams.model_validate({**base, **bad})
