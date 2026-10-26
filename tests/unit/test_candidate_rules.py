"""Unit tests for the mechanical rule interface and the exchange-gap rule."""

from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.candidates import MechanicalCandidate, candidates_dir, load_candidates
from odds_core.exchange_price import ExchangePrice
from odds_core.market_prices import Outcome
from odds_core.quotes import Quote
from odds_core.snapshot_sources import FdukTimeBasis
from odds_core.vocabulary import Market, Side, Source, TimeBasis
from odds_lambda.candidate_rules import (
    RULES,
    BetIntent,
    ExchangeGapParams,
    MarketState,
    ModelRuleSpec,
    RetailEvModelParams,
    RuleSpec,
    apply_rule,
    build_market_state,
    decision_leads,
    decision_times,
    exchange_gap,
    is_bettable_line,
    max_input_age,
    resolve_rule,
    retail_ev_model,
    rule_sha256,
    validate_candidate_rules,
)
from pydantic import ValidationError

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_quotes,
    three_way_bookmaker,
)

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
EVENT = make_epl_event("evt", KICKOFF)
HOME, DRAW, AWAY = Outcome.HOME, Outcome.DRAW, Outcome.AWAY
OVER, UNDER = Outcome.OVER, Outcome.UNDER
MAX_AGE = timedelta(minutes=10)


def _betfair(prices: tuple[float, float, float], inplay: bool = False) -> dict[str, Any]:
    return betfair_exchange_bookmaker(prices, inplay=inplay)


_ids = iter(range(1, 10_000))


def _snap(
    minutes_before_t: float,
    *books: dict[str, Any],
    source: str | None = "oddsportal_live",
    snapshot_id: int | None = None,
) -> list[Quote]:
    return make_quotes(
        EVENT,
        snapshot_id or next(_ids),
        T - timedelta(minutes=minutes_before_t),
        list(books),
        source,
    )


def _time(snapshot: list[Quote]) -> datetime:
    return snapshot[0].snapshot_time


def _state(snapshots: list[list[Quote]], market: Market = Market.ONE_X_TWO) -> MarketState:
    return build_market_state(EVENT, market, [q for s in snapshots for q in s], T, MAX_AGE)


BET365 = three_way_bookmaker("bet365", (2.0, 3.0, 4.0))
BET365_TOTALS = {
    "key": "bet365",
    "markets": [
        {
            "key": "totals",
            "outcomes": [
                {"name": "Over", "price_decimal": 1.9, "point": 2.5},
                {"name": "Under", "price_decimal": 1.9, "point": 2.5},
            ],
        }
    ],
}


class TestBuildMarketState:
    def test_uses_the_given_market_and_event_fields(self) -> None:
        state = _state([])
        assert (state.event_id, state.sport_key, state.market) == ("evt", "soccer_epl", "1x2")
        assert (state.kickoff, state.decision_time) == (KICKOFF, T)
        assert not state.has_input
        assert state.retail == {} and state.exchange == {} and state.snapshot_id is None
        assert _state([], Market.TOTALS).market is Market.TOTALS

    def test_retail_and_exchange_come_from_one_snapshot(self) -> None:
        snapshot = _snap(3, BET365, _betfair((2.0, 4.0, 4.0)), snapshot_id=7)
        state = _state([snapshot])
        assert state.has_input
        assert (state.snapshot_id, state.snapshot_time) == (7, _time(snapshot))
        assert state.retail == {None: {"bet365": {HOME: 2.0, DRAW: 3.0, AWAY: 4.0}}}
        exchange = state.exchange[None]
        assert exchange.source is Source.ODDSPORTAL
        assert exchange.time == _time(snapshot)
        assert sum(exchange.fair.values()) == pytest.approx(1.0)
        assert exchange.fair[HOME] == pytest.approx(0.5)

    def test_snapshots_after_t_are_ignored(self) -> None:
        assert not _state([_snap(-1, BET365, _betfair((2.0, 3.5, 4.0)))]).has_input

    def test_snapshot_at_t_counts(self) -> None:
        state = _state([_snap(0, BET365, _betfair((2.0, 4.0, 4.0)))])
        assert state.retail[None]["bet365"][HOME] == 2.0

    def test_latest_snapshot_wins_whole(self) -> None:
        old = _snap(8, BET365, _betfair((2.0, 4.0, 4.0)))
        new = _snap(2, three_way_bookmaker("betway", (2.2, 3.2, 4.2)), _betfair((2.1, 4.0, 3.8)))
        state = _state([new, old])
        assert state.snapshot_time == _time(new)
        assert set(state.retail[None]) == {"betway"}

    def test_a_tie_in_time_goes_to_the_higher_snapshot_id(self) -> None:
        low = _snap(2, BET365, _betfair((2.0, 4.0, 4.0)), snapshot_id=5000)
        high = _snap(
            2,
            three_way_bookmaker("betway", (2.2, 3.2, 4.2)),
            _betfair((2.0, 4.0, 4.0)),
            source=None,
            snapshot_id=5001,
        )
        state = _state([high, low])
        assert state.snapshot_id == 5001 and set(state.retail[None]) == {"betway"}

    def test_lay_prices_are_not_read(self) -> None:
        [snapshot] = [_snap(2, BET365, _betfair((2.0, 4.0, 4.0)))]
        lays = [replace(q, side=Side.LAY, price=q.price + 1) for q in snapshot]
        state = _state([snapshot, lays])
        assert state.retail == {None: {"bet365": {HOME: 2.0, DRAW: 3.0, AWAY: 4.0}}}
        assert state.exchange[None].fair[HOME] == pytest.approx(0.5)

    def test_a_later_snapshot_of_another_market_is_skipped(self) -> None:
        one_x_two = _snap(5, BET365, _betfair((2.0, 4.0, 4.0)))
        totals = _snap(
            4,
            {
                "key": "bet365",
                "markets": [
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price_decimal": 1.8, "point": 2.5},
                            {"name": "Under", "price_decimal": 2.0, "point": 2.5},
                        ],
                    }
                ],
            },
        )
        assert _state([one_x_two, totals]).snapshot_time == _time(one_x_two)

    def test_snapshot_at_the_max_age_counts(self) -> None:
        assert _state([_snap(10, BET365, _betfair((2.0, 4.0, 4.0)))]).has_input

    def test_stale_latest_snapshot_gives_no_input(self) -> None:
        state = _state([_snap(10.5, BET365, _betfair((2.0, 4.0, 4.0)))])
        assert not state.has_input
        assert state.retail == {} and state.exchange == {}

    @pytest.mark.parametrize(
        ("source", "shape"),
        [
            (None, (Source.ODDS_API, TimeBasis.OBSERVED)),
            ("oddsportal", (Source.ODDSPORTAL, TimeBasis.SOURCE_STAMPED)),
            ("oddsportal_live", (Source.ODDSPORTAL, TimeBasis.OBSERVED)),
        ],
    )
    def test_any_retail_source_with_both_prices_is_an_input(
        self, source: str | None, shape: tuple[Source, TimeBasis]
    ) -> None:
        state = _state([_snap(2, BET365, _betfair((2.0, 4.0, 4.0)), source=source)])
        assert state.has_input
        assert (state.exchange[None].source, state.exchange[None].time_basis) == shape

    def test_odds_api_exchange_alias_is_the_exchange(self) -> None:
        exchange = three_way_bookmaker("betfair_ex_uk", (2.0, 4.0, 4.0))
        state = _state([_snap(2, BET365, exchange, source=None)])
        assert state.exchange[None].fair[HOME] == pytest.approx(0.5)

    def test_live_betfair_snapshot_is_not_an_input(self) -> None:
        assert not _state(
            [_snap(2, BET365, _betfair((2.0, 4.0, 4.0)), source="betfair_api")]
        ).has_input

    def test_fduk_opening_is_not_an_input(self) -> None:
        snapshot = make_quotes(
            EVENT,
            next(_ids),
            T - timedelta(minutes=2),
            [_betfair((2.0, 4.0, 4.0)), BET365],
            "football_data_uk",
            FdukTimeBasis.COLLECTION_DEADLINE,
        )
        assert not _state([snapshot]).has_input

    def test_snapshot_without_an_exchange_row_is_skipped_for_an_earlier_one(self) -> None:
        both = _snap(6, BET365, _betfair((2.0, 4.0, 4.0)))
        retail_only = _snap(2, three_way_bookmaker("betway", (2.2, 3.2, 4.2)), source=None)
        state = _state([both, retail_only])
        assert state.snapshot_time == _time(both) and set(state.retail[None]) == {"bet365"}

    def test_snapshot_without_retail_is_skipped(self) -> None:
        assert not _state([_snap(2, _betfair((2.0, 4.0, 4.0)))]).has_input

    def test_non_retail_books_are_excluded(self) -> None:
        pinnacle = three_way_bookmaker("pinnacle", (2.0, 3.0, 4.0))
        state = _state([_snap(2, pinnacle, BET365, _betfair((2.0, 4.0, 4.0)))])
        assert set(state.retail[None]) == {"bet365"}

    def test_incomplete_exchange_row_gives_no_exchange(self) -> None:
        partial = {
            "key": "betfair_exchange",
            "markets": [{"key": "1x2", "outcomes": [{"name": "Arsenal", "price_decimal": 2.0}]}],
        }
        state = _state([_snap(2, BET365, partial)])
        assert state.has_input and state.exchange == {}

    def test_wide_exchange_gives_none_without_falling_back(self) -> None:
        tight = _snap(6, BET365, _betfair((2.0, 4.0, 4.0)))
        wide = _snap(2, BET365, _betfair((1.8, 3.5, 3.5)))
        state = _state([tight, wide])
        assert state.snapshot_time == _time(wide) and state.exchange == {}


def _totals_book(key: str, lines: dict[float, tuple[float, float]]) -> dict[str, Any]:
    """A ``raw_data`` bookmaker entry with totals (over, under) prices per line."""
    return {
        "key": key,
        "markets": [
            {
                "key": "totals",
                "outcomes": [
                    outcome
                    for line, (over, under) in lines.items()
                    for outcome in (
                        {"name": "Over", "price_decimal": over, "point": line},
                        {"name": "Under", "price_decimal": under, "point": line},
                    )
                ],
            }
        ],
    }


def _totals_snap(minutes_before_t: float, *books: dict[str, Any]) -> list[Quote]:
    return [q for q in _snap(minutes_before_t, *books) if q.market == "totals"]


class TestTotalsMarketState:
    def test_retail_and_exchange_are_keyed_by_line(self) -> None:
        snapshot = _totals_snap(
            2,
            _totals_book("bet365", {2.5: (1.9, 1.9), 3.5: (3.0, 1.4)}),
            _totals_book("betfair_exchange", {2.5: (2.0, 2.0)}),
        )
        state = _state([snapshot], Market.TOTALS)
        assert state.retail == {
            2.5: {"bet365": {OVER: 1.9, UNDER: 1.9}},
            3.5: {"bet365": {OVER: 3.0, UNDER: 1.4}},
        }
        assert set(state.exchange) == {2.5}
        assert state.exchange[2.5].fair == {
            OVER: pytest.approx(0.5),
            UNDER: pytest.approx(0.5),
        }

    def test_each_line_is_devigged_on_its_own_pair(self) -> None:
        snapshot = _totals_snap(
            2,
            BET365_TOTALS,
            _totals_book("betfair_exchange", {2.5: (2.0, 2.0), 3.5: (4.0, 4.0 / 3)}),
        )
        state = _state([snapshot], Market.TOTALS)
        assert state.exchange[3.5].fair[OVER] == pytest.approx(0.25)
        assert state.exchange[2.5].fair[OVER] == pytest.approx(0.5)

    @pytest.mark.parametrize("line", [2.0, 2.25, 2.75])
    def test_only_half_lines_count(self, line: float) -> None:
        snapshot = _totals_snap(
            2,
            _totals_book("bet365", {line: (1.9, 1.9), 2.5: (1.9, 1.9)}),
            _totals_book("betfair_exchange", {line: (2.0, 2.0), 2.5: (2.0, 2.0)}),
        )
        state = _state([snapshot], Market.TOTALS)
        assert set(state.retail) == {2.5} and set(state.exchange) == {2.5}

    def test_a_wide_line_has_no_exchange_price(self) -> None:
        snapshot = _totals_snap(
            2,
            BET365_TOTALS,
            _totals_book("betfair_exchange", {2.5: (2.0, 2.0), 3.5: (3.0, 1.3)}),
        )
        assert set(_state([snapshot], Market.TOTALS).exchange) == {2.5}

    def test_a_one_sided_line_has_no_exchange_price(self) -> None:
        exchange = _totals_book("betfair_exchange", {2.5: (2.0, 2.0)})
        exchange["markets"][0]["outcomes"].append(
            {"name": "Over", "price_decimal": 3.0, "point": 3.5}
        )
        state = _state([_totals_snap(2, BET365_TOTALS, exchange)], Market.TOTALS)
        assert set(state.exchange) == {2.5}

    def test_a_totals_row_without_a_line_never_counts(self) -> None:
        snapshot = _totals_snap(
            2, BET365_TOTALS, _totals_book("betfair_exchange", {2.5: (2.0, 2.0)})
        )
        lineless = [replace(q, line=None, price=q.price + 1) for q in snapshot]
        state = _state([snapshot + lineless], Market.TOTALS)
        assert set(state.retail) == {2.5} and set(state.exchange) == {2.5}

    @pytest.mark.parametrize(
        ("market", "line", "bettable"),
        [
            (Market.ONE_X_TWO, None, True),
            (Market.ONE_X_TWO, 2.5, False),
            (Market.TOTALS, None, False),
            (Market.TOTALS, 2.5, True),
            (Market.TOTALS, 3.0, False),
            (Market.TOTALS, 2.25, False),
        ],
    )
    def test_bettable_lines_depend_on_the_market(
        self, market: Market, line: float | None, bettable: bool
    ) -> None:
        assert is_bettable_line(market, line) is bettable

    def test_a_snapshot_of_the_other_market_is_not_an_input(self) -> None:
        assert not _state([_snap(2, BET365, _betfair((2.0, 4.0, 4.0)))], Market.TOTALS).has_input


def _exchange(home: float, draw: float, away: float) -> ExchangePrice:
    return ExchangePrice(
        time=T,
        source=Source.ODDSPORTAL,
        time_basis=TimeBasis.OBSERVED,
        in_play=False,
        fair={HOME: home, DRAW: draw, AWAY: away},
        overround=0.0,
    )


def _rule_state(
    retail: dict[str, dict[Outcome, float]], exchange: ExchangePrice | None
) -> MarketState:
    return MarketState(
        event_id="evt",
        sport_key="soccer_epl",
        market="1x2",
        kickoff=KICKOFF,
        decision_time=T,
        snapshot_id=1,
        snapshot_time=T,
        retail={None: retail} if retail else {},
        exchange={} if exchange is None else {None: exchange},
    )


def _totals_exchange(over: float) -> ExchangePrice:
    return replace(_exchange(0, 0, 0), fair={OVER: over, UNDER: 1 - over})


def _totals_state(
    retail: dict[float, dict[str, dict[Outcome, float]]], exchange: dict[float, ExchangePrice]
) -> MarketState:
    return replace(_rule_state({}, None), market=Market.TOTALS, retail=retail, exchange=exchange)


PARAMS = ExchangeGapParams(threshold=0.02, minutes_before_kickoff=(60,), max_input_age_minutes=10)


class TestExchangeGap:
    def test_no_exchange_no_bets(self) -> None:
        assert exchange_gap(_rule_state({"bet365": {HOME: 9.0}}, None), PARAMS) == []

    def test_threshold_is_strict(self) -> None:
        # 0.5 * 2.5 - 1 = 0.25, clearly above; edge exactly at threshold is not a bet.
        state = _rule_state({"bet365": {HOME: 2.5}}, _exchange(0.5, 0.25, 0.25))
        at = ExchangeGapParams(
            threshold=0.25, minutes_before_kickoff=(60,), max_input_age_minutes=10
        )
        below = ExchangeGapParams(
            threshold=0.2499, minutes_before_kickoff=(60,), max_input_age_minutes=10
        )
        assert exchange_gap(state, at) == []
        assert exchange_gap(state, below) == [BetIntent(HOME, "bet365", 2.5, 1.0)]

    def test_longest_price_across_books_is_chosen(self) -> None:
        state = _rule_state(
            {"bet365": {HOME: 2.4}, "betway": {HOME: 2.6}, "coral": {HOME: 2.5}},
            _exchange(0.5, 0.25, 0.25),
        )
        assert exchange_gap(state, PARAMS) == [BetIntent(HOME, "betway", 2.6, 1.0)]

    def test_tie_goes_to_the_alphabetically_first_book(self) -> None:
        state = _rule_state(
            {"coral": {HOME: 2.6}, "betway": {HOME: 2.6}}, _exchange(0.5, 0.25, 0.25)
        )
        assert [b.bookmaker for b in exchange_gap(state, PARAMS)] == ["betway"]

    def test_outcomes_are_independent(self) -> None:
        state = _rule_state(
            {"bet365": {HOME: 1.5, DRAW: 5.0, AWAY: 5.0}, "betway": {DRAW: 4.0, AWAY: 6.0}},
            _exchange(0.5, 0.25, 0.25),
        )
        bets = exchange_gap(state, PARAMS)
        assert bets == [
            BetIntent(DRAW, "bet365", 5.0, 1.0),
            BetIntent(AWAY, "betway", 6.0, 1.0),
        ]

    def test_stake_comes_from_params(self) -> None:
        state = _rule_state({"bet365": {HOME: 2.6}}, _exchange(0.5, 0.25, 0.25))
        params = PARAMS.model_copy(update={"stake": 2.5})
        assert exchange_gap(state, params)[0].stake == 2.5

    def test_no_retail_no_bets(self) -> None:
        assert exchange_gap(_rule_state({}, _exchange(0.5, 0.25, 0.25)), PARAMS) == []

    def test_totals_bets_carry_their_line(self) -> None:
        state = _totals_state(
            {2.5: {"bet365": {OVER: 2.2, UNDER: 1.7}, "betway": {OVER: 2.1}}},
            {2.5: _totals_exchange(0.5)},
        )
        assert exchange_gap(state, PARAMS) == [BetIntent(OVER, "bet365", 2.2, 1.0, 2.5)]

    def test_totals_retail_is_compared_on_the_exchange_line_only(self) -> None:
        state = _totals_state(
            {2.5: {"bet365": {OVER: 1.9}}, 3.5: {"bet365": {OVER: 9.0, UNDER: 9.0}}},
            {2.5: _totals_exchange(0.5)},
        )
        assert exchange_gap(state, PARAMS) == []

    def test_totals_lines_are_independent(self) -> None:
        state = _totals_state(
            {2.5: {"bet365": {UNDER: 2.2}}, 3.5: {"betway": {OVER: 4.5}}},
            {2.5: _totals_exchange(0.5), 3.5: _totals_exchange(0.25)},
        )
        assert exchange_gap(state, PARAMS) == [
            BetIntent(UNDER, "bet365", 2.2, 1.0, 2.5),
            BetIntent(OVER, "betway", 4.5, 1.0, 3.5),
        ]

    @pytest.mark.parametrize("field", ["threshold", "stake", "max_input_age_minutes"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_params_must_be_positive(self, field: str, value: float) -> None:
        raw: dict[str, Any] = {**PARAMS.model_dump(), field: value}
        with pytest.raises(ValidationError):
            ExchangeGapParams(**raw)

    def test_max_input_age_is_required(self) -> None:
        with pytest.raises(ValidationError):
            ExchangeGapParams(threshold=0.02, minutes_before_kickoff=(60,))  # type: ignore[call-arg]

    @pytest.mark.parametrize("minutes", [[], [0], [60, -1], [60, 60]])
    def test_decision_times_are_distinct_positive_and_at_least_one(
        self, minutes: list[int]
    ) -> None:
        with pytest.raises(ValidationError):
            ExchangeGapParams(**{**PARAMS.model_dump(), "minutes_before_kickoff": minutes})

    def test_decision_times_are_held_earliest_first(self) -> None:
        params = ExchangeGapParams(
            **{**PARAMS.model_dump(), "minutes_before_kickoff": [30, 180, 60]}
        )
        assert params.minutes_before_kickoff == (180, 60, 30)

    def test_params_reject_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExchangeGapParams(**PARAMS.model_dump(), surprise=1)


MODEL_PARAMS = RetailEvModelParams(
    model="retail_ev_price_only",
    threshold=0.01,
    training_matches=50,
    refit_every_matches=10,
    tuning_trials=50,
    tuning_seed=42,
    tuning_min_train_matches=20,
    tuning_step_matches=10,
    minutes_before_kickoff=(60,),
    max_input_age_minutes=10,
)


def _predicting(evs: dict[tuple[Outcome, str], float]) -> Any:
    seen: list[MarketState] = []

    def predict(state: MarketState) -> dict[tuple[Outcome, str], float]:
        seen.append(state)
        return evs

    predict.seen = seen  # type: ignore[attr-defined]
    return predict


class TestRetailEvModel:
    RETAIL = {
        "bet365": {HOME: 2.1, DRAW: 3.4, AWAY: 4.0},
        "betway": {HOME: 2.0, DRAW: 3.5, AWAY: 4.0},
    }

    def test_bets_the_longest_price_when_its_predicted_ev_exceeds_the_threshold(self) -> None:
        predict = _predicting(
            {(HOME, "bet365"): 0.02, (HOME, "betway"): 0.5, (DRAW, "betway"): 0.011}
        )
        state = _rule_state(self.RETAIL, _exchange(0.5, 0.25, 0.25))
        assert retail_ev_model(state, MODEL_PARAMS, predict) == [
            BetIntent(HOME, "bet365", 2.1, 1.0),
            BetIntent(DRAW, "betway", 3.5, 1.0),
        ]
        assert predict.seen == [state]

    def test_threshold_is_strict(self) -> None:
        predict = _predicting({(HOME, "bet365"): 0.01})
        state = _rule_state(self.RETAIL, _exchange(0.5, 0.25, 0.25))
        assert retail_ev_model(state, MODEL_PARAMS, predict) == []

    def test_tie_goes_to_the_alphabetically_first_book(self) -> None:
        predict = _predicting({(AWAY, "bet365"): 0.05, (AWAY, "betway"): 0.0})
        state = _rule_state(self.RETAIL, _exchange(0.5, 0.25, 0.25))
        assert retail_ev_model(state, MODEL_PARAMS, predict) == [
            BetIntent(AWAY, "bet365", 4.0, 1.0)
        ]

    def test_no_exchange_no_prediction_and_no_bets(self) -> None:
        predict = _predicting({(HOME, "bet365"): 0.5})
        assert retail_ev_model(_rule_state(self.RETAIL, None), MODEL_PARAMS, predict) == []
        assert predict.seen == []

    @pytest.mark.parametrize(
        "missing",
        [
            "model",
            "threshold",
            "training_matches",
            "refit_every_matches",
            "tuning_trials",
            "tuning_seed",
            "tuning_min_train_matches",
            "tuning_step_matches",
        ],
    )
    def test_params_are_required(self, missing: str) -> None:
        fields = MODEL_PARAMS.model_dump()
        del fields[missing]
        with pytest.raises(ValidationError):
            RetailEvModelParams.model_validate(fields)

    def test_tuning_must_leave_a_training_match_to_validate(self) -> None:
        with pytest.raises(ValidationError, match="leaves no match"):
            RetailEvModelParams.model_validate(
                {**MODEL_PARAMS.model_dump(), "tuning_min_train_matches": 50}
            )

    def test_unknown_model_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetailEvModelParams.model_validate({**MODEL_PARAMS.model_dump(), "model": "other"})

    def test_is_a_model_rule_bound_to_a_predictor(self) -> None:
        spec = RULES["retail_ev_model"]
        assert isinstance(spec, ModelRuleSpec)
        predict = _predicting({(HOME, "bet365"): 0.5})
        state = _rule_state(self.RETAIL, _exchange(0.5, 0.25, 0.25))
        _, bets = apply_rule(spec.bind(predict), MODEL_PARAMS, state, bet_selections={(DRAW, None)})
        assert bets == [BetIntent(HOME, "bet365", 2.1, 1.0)]


class TestApplyRule:
    def test_state_without_input_gives_no_bets_whatever_the_rule(self) -> None:
        always = RuleSpec(
            lambda state, params: [BetIntent(HOME, "bet365", 2.0, 1.0)], ExchangeGapParams
        )
        empty = MarketState("evt", "soccer_epl", "1x2", KICKOFF, T)
        assert apply_rule(always, PARAMS, empty)[1] == []
        assert apply_rule(always, PARAMS, _rule_state({}, None))[1] != []

    def test_selections_already_bet_are_dropped(self) -> None:
        both = RuleSpec(
            lambda state, params: [
                BetIntent(HOME, "bet365", 2.0, 1.0),
                BetIntent(AWAY, "bet365", 4.0, 1.0),
            ],
            ExchangeGapParams,
        )
        state = _rule_state({}, None)
        assert apply_rule(both, PARAMS, state, bet_selections={(HOME, None)})[1] == [
            BetIntent(AWAY, "bet365", 4.0, 1.0)
        ]
        assert apply_rule(both, PARAMS, state, bet_selections={(HOME, None), (AWAY, None)})[1] == []

    def test_a_selection_is_an_outcome_on_a_line(self) -> None:
        overs = RuleSpec(
            lambda state, params: [
                BetIntent(OVER, "bet365", 2.0, 1.0, 2.5),
                BetIntent(OVER, "bet365", 3.0, 1.0, 3.5),
            ],
            ExchangeGapParams,
        )
        state = _totals_state({}, {})
        assert apply_rule(overs, PARAMS, state, bet_selections={(OVER, 2.5)})[1] == [
            BetIntent(OVER, "bet365", 3.0, 1.0, 3.5)
        ]


class TestDecisionTimes:
    def test_are_kickoff_minus_each_lead_earliest_first(self) -> None:
        params = ExchangeGapParams(**{**PARAMS.model_dump(), "minutes_before_kickoff": [60, 1440]})
        assert decision_times(KICKOFF, params) == [
            (1440, KICKOFF - timedelta(days=1)),
            (60, KICKOFF - timedelta(hours=1)),
        ]

    def test_max_input_age_is_in_minutes(self) -> None:
        assert max_input_age(PARAMS) == timedelta(minutes=10)


def _candidate(**overrides: Any) -> MechanicalCandidate:
    fields: dict[str, Any] = {
        "name": "gap",
        "description": "d",
        "kind": "mechanical",
        "market": "1x2",
        "sport": "soccer_epl",
        "version": 1,
        "issue": 626,
        "start_date": "2026-11-01",
        "status": "active",
        "rule": "exchange_gap",
        "rule_sha256": rule_sha256("exchange_gap"),
        "params": {"threshold": 0.02, "minutes_before_kickoff": [60], "max_input_age_minutes": 10},
        **overrides,
    }
    return MechanicalCandidate.model_validate(fields)


class TestValidateCandidateRules:
    def test_valid_candidate_passes(self) -> None:
        validate_candidate_rules([_candidate()])

    def test_unknown_rule_names_the_candidate(self) -> None:
        with pytest.raises(ValueError, match="'gap'.*no_such_rule"):
            validate_candidate_rules([_candidate(rule="no_such_rule")])

    def test_bad_params_name_the_candidate(self) -> None:
        with pytest.raises(ValueError, match="'gap'.*invalid params"):
            validate_candidate_rules([_candidate(params={"threshold": -1})])

    def test_market_is_required(self) -> None:
        fields = _candidate().model_dump(mode="json")
        del fields["market"]
        with pytest.raises(ValidationError, match="market"):
            MechanicalCandidate.model_validate(fields)

    @pytest.mark.parametrize(
        ("sport", "market"),
        [
            ("soccer_epl", "1x2"),
            ("soccer_epl", "totals"),
            ("baseball_mlb", "h2h"),
            ("baseball_mlb", "totals"),
        ],
    )
    def test_each_sport_decides_its_markets(self, sport: str, market: str) -> None:
        resolve_rule(_candidate(sport=sport, market=market))

    @pytest.mark.parametrize(
        ("sport", "market"),
        [("soccer_epl", "h2h"), ("baseball_mlb", "1x2"), ("basketball_nba", "h2h")],
    )
    def test_a_market_the_sport_does_not_decide_fails(self, sport: str, market: str) -> None:
        with pytest.raises(ValueError, match=f"no rule decides market '{market}' for sport"):
            resolve_rule(_candidate(sport=sport, market=market))

    def test_market_without_a_rule_fails(self) -> None:
        with pytest.raises(ValueError, match="'gap'.*no rule decides market 'spreads'"):
            validate_candidate_rules([_candidate(market="spreads")])

    def test_model_rule_without_a_model_for_the_market_fails(self) -> None:
        candidate = _candidate(
            market="totals",
            rule="retail_ev_model",
            rule_sha256=rule_sha256("retail_ev_model"),
            params=MODEL_PARAMS.model_dump(mode="json"),
        )
        with pytest.raises(ValueError, match="'gap'.*no model for market 'totals'"):
            resolve_rule(candidate)

    def test_registered_candidates_are_valid(self) -> None:
        validate_candidate_rules(load_candidates(candidates_dir()))

    def test_model_candidate_resolves_to_its_model_rule(self) -> None:
        candidate = _candidate(
            rule="retail_ev_model",
            rule_sha256=rule_sha256("retail_ev_model"),
            params=MODEL_PARAMS.model_dump(mode="json"),
        )
        validate_candidate_rules([candidate])
        spec, params = resolve_rule(candidate)
        assert isinstance(spec, ModelRuleSpec)
        assert params == MODEL_PARAMS


class TestDecisionLeads:
    def test_active_mechanical_candidates_of_the_sport(self) -> None:
        candidates = [
            _candidate(),
            _candidate(
                name="later", params={**PARAMS.model_dump(), "minutes_before_kickoff": [30, 15]}
            ),
            _candidate(name="mlb", sport="baseball_mlb", market="h2h"),
            _candidate(name="killed", status="killed"),
            _candidate(name="broken", params={"threshold": -1}),
        ]
        assert decision_leads(candidates, "soccer_epl") == [
            timedelta(minutes=60),
            timedelta(minutes=30),
            timedelta(minutes=15),
        ]


class TestRuleSha256:
    def test_is_a_stable_sha256(self) -> None:
        assert rule_sha256("exchange_gap") == rule_sha256("exchange_gap")
        assert len(rule_sha256("exchange_gap")) == 64

    @pytest.mark.parametrize(
        "target",
        [
            "odds_lambda.candidate_rules.rules.exchange_gap",
            "odds_lambda.candidate_rules.rules.ExchangeGapParams",
            "odds_lambda.candidate_rules.rules.RuleParams",
            "odds_lambda.candidate_rules.rules.longest_retail",
            "odds_lambda.candidate_rules.rules.BetIntent",
            "odds_lambda.candidate_rules.rules.apply_rule",
            "odds_lambda.candidate_rules.inputs",
            "odds_lambda.candidate_rules.market_state",
            "odds_core.as_of",
            "odds_core.exchange_price",
        ],
    )
    def test_covers_the_code_that_picks_the_inputs(
        self, monkeypatch: pytest.MonkeyPatch, target: str
    ) -> None:
        real = inspect.getsource

        def name_of(obj: Any) -> str:
            return (
                getattr(obj, "__name__", "")
                if inspect.ismodule(obj)
                else (f"{obj.__module__}.{obj.__qualname__}")
            )

        def fake(obj: Any) -> str:
            return real(obj) + ("# changed" if name_of(obj) == target else "")

        before = rule_sha256("exchange_gap")
        monkeypatch.setattr(inspect, "getsource", fake)
        assert rule_sha256("exchange_gap") != before

    def test_does_not_cover_the_shared_quote_reader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = inspect.getsource

        def fake(obj: Any) -> str:
            changed = inspect.ismodule(obj) and obj.__name__ == "odds_core.quotes"
            return real(obj) + ("# changed" if changed else "")

        before = rule_sha256("exchange_gap")
        monkeypatch.setattr(inspect, "getsource", fake)
        assert rule_sha256("exchange_gap") == before

    def test_covers_the_uk_retail_roster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = rule_sha256("exchange_gap")
        monkeypatch.setattr("odds_core.bookmakers.UK_RETAIL_BOOKMAKERS", frozenset({"bet365"}))
        assert rule_sha256("exchange_gap") != before

    @pytest.mark.parametrize("source", [Source.ODDS_API, Source.ODDSPORTAL])
    def test_covers_the_write_time_alias_map_of_each_retail_source(
        self, monkeypatch: pytest.MonkeyPatch, source: Source
    ) -> None:
        from odds_core import bookmakers

        before = rule_sha256("exchange_gap")
        aliases = dict(bookmakers.BOOKMAKER_ALIASES_BY_SOURCE)
        aliases[source] = {**aliases[source], "bet365_uk": "bet365"}
        monkeypatch.setattr(bookmakers, "BOOKMAKER_ALIASES_BY_SOURCE", aliases)
        assert rule_sha256("exchange_gap") != before

    def test_covers_the_rule_markets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from odds_lambda.candidate_rules import rules

        before = rule_sha256("exchange_gap")
        monkeypatch.setattr(rules, "RULE_MARKETS", rules.RULE_MARKETS[:2])
        assert rule_sha256("exchange_gap") != before

    def test_covers_the_outcomes_of_each_market(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from odds_lambda.candidate_rules import rules

        before = rule_sha256("exchange_gap")
        monkeypatch.setattr(rules, "market_outcomes", lambda market: (HOME, AWAY))
        assert rule_sha256("exchange_gap") != before

    @pytest.mark.parametrize(
        "target",
        [
            "odds_lambda.candidate_rules.rules.retail_ev_model",
            "odds_lambda.candidate_rules.rules.RetailEvModelParams",
            "odds_lambda.candidate_rules.market_state",
            "odds_analytics.candidate_model",
            "odds_analytics.retail_ev",
            "odds_analytics.xgboost_line_movement",
            "odds_core.betfair_close",
            "odds_analytics.retail_ev_metrics",
        ],
    )
    def test_model_rule_covers_the_code_that_fits_and_queries_the_model(
        self, monkeypatch: pytest.MonkeyPatch, target: str
    ) -> None:
        real = inspect.getsource

        def fake(obj: Any) -> str:
            name = obj.__name__ if inspect.ismodule(obj) else f"{obj.__module__}.{obj.__qualname__}"
            return real(obj) + ("# changed" if name == target else "")

        before = rule_sha256("retail_ev_model")
        assert before != rule_sha256("exchange_gap")
        monkeypatch.setattr(inspect, "getsource", fake)
        assert rule_sha256("retail_ev_model") != before

    def test_a_stale_hash_fails_validation_with_the_new_one(self) -> None:
        with pytest.raises(ValueError, match="'gap'.*code of rule 'exchange_gap' changed") as exc:
            validate_candidate_rules([_candidate(rule_sha256="0" * 64)])
        assert rule_sha256("exchange_gap") in str(exc.value)

    def test_hash_must_be_64_hex_chars(self) -> None:
        with pytest.raises(ValidationError):
            _candidate(rule_sha256="abc")
