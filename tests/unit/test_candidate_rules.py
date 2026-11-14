"""Unit tests for the mechanical rule interface and the rules."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.candidates import MechanicalCandidate, candidates_dir, load_candidates
from odds_core.exchange_price import ExchangePrice
from odds_core.market_prices import Outcome
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, Side, Source, TimeBasis
from odds_lambda.candidate_rules import (
    BetIntent,
    EstimateBasis,
    MarketState,
    ModelRuleSpec,
    RuleSpec,
    SharpInput,
    apply_rule,
    best_bet,
    build_market_state,
    decision_times,
    is_bettable_line,
    max_input_age,
    pinning,
)
from odds_lambda.candidate_rules.code_hash import closure, root_module
from odds_lambda.candidate_rules.gap import GapParams, exchange_gap, polymarket_gap
from odds_lambda.candidate_rules.market_state import PolymarketPrice
from odds_lambda.candidate_rules.pinning import RULE_ROOTS, rule_sha256, validate_candidate_rules
from odds_lambda.candidate_rules.registry import RULES, input_leads, resolve_rule
from odds_lambda.candidate_rules.retail_ev_model import RetailEvModelParams, retail_ev_model
from pydantic import ValidationError

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_quotes,
    polymarket_bookmaker,
    three_way_bookmaker,
)
from tests.unit.conftest import FIXTURE_PACKAGE, SourceEdit

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
EVENT = make_epl_event(1, KICKOFF)
HOME, DRAW, AWAY = Outcome.HOME, Outcome.DRAW, Outcome.AWAY
OVER, UNDER = Outcome.OVER, Outcome.UNDER
MAX_AGE = timedelta(minutes=10)
EXCHANGE = frozenset({SharpInput.EXCHANGE})
POLYMARKET = frozenset({SharpInput.POLYMARKET})
BOTH = EXCHANGE | POLYMARKET


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


def _state(
    snapshots: list[list[Quote]],
    market: Market = Market.ONE_X_TWO,
    sharp_inputs: frozenset[SharpInput] = EXCHANGE,
) -> MarketState:
    return build_market_state(
        EVENT,
        market,
        [q for s in snapshots for q in s],
        T,
        MAX_AGE,
        sharp_inputs,
        UK_RETAIL_BOOKMAKERS,
    )


def _polymarket_snap(
    minutes_before_t: float, probabilities: tuple[float, float, float], snapshot_id: int
) -> list[Quote]:
    return _snap(
        minutes_before_t,
        polymarket_bookmaker(probabilities),
        source="polymarket",
        snapshot_id=snapshot_id,
    )


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
        assert (state.event_id, state.sport_key, state.market) == (1, "soccer_epl", "1x2")
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
            "fduk_collection_deadline",
        )
        assert not _state([snapshot]).has_input

    def test_snapshot_without_an_exchange_row_is_skipped_for_an_earlier_one(self) -> None:
        both = _snap(6, BET365, _betfair((2.0, 4.0, 4.0)))
        retail_only = _snap(2, three_way_bookmaker("betway", (2.2, 3.2, 4.2)), source=None)
        state = _state([both, retail_only])
        assert state.snapshot_time == _time(both) and set(state.retail[None]) == {"bet365"}

    def test_snapshot_without_retail_is_skipped(self) -> None:
        assert not _state([_snap(2, _betfair((2.0, 4.0, 4.0)))]).has_input

    def test_retail_is_only_the_given_books(self) -> None:
        betway = three_way_bookmaker("betway", (2.2, 3.2, 4.2))
        snapshot = _snap(2, BET365, betway, _betfair((2.0, 4.0, 4.0)))
        state = build_market_state(
            EVENT, Market.ONE_X_TWO, snapshot, T, MAX_AGE, EXCHANGE, frozenset({"betway"})
        )
        assert set(state.retail[None]) == {"betway"}

    def test_a_snapshot_quoting_none_of_the_books_is_not_an_input(self) -> None:
        both = _snap(6, BET365, _betfair((2.0, 4.0, 4.0)))
        older = _snap(9, three_way_bookmaker("betway", (2.2, 3.2, 4.2)), _betfair((2.0, 4.0, 4.0)))
        books = frozenset({"betway"})
        state = build_market_state(
            EVENT, Market.ONE_X_TWO, [*older, *both], T, MAX_AGE, EXCHANGE, books
        )
        assert state.snapshot_time == _time(older) and set(state.retail[None]) == {"betway"}
        assert not build_market_state(
            EVENT, Market.ONE_X_TWO, both, T, MAX_AGE, EXCHANGE, books
        ).has_input

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


class TestPolymarketInputs:
    def test_fair_price_is_the_mids_normalised(self) -> None:
        retail = _snap(2, BET365, snapshot_id=8000)
        polymarket = _polymarket_snap(4, (0.5, 0.3, 0.25), snapshot_id=8001)
        state = _state([retail, polymarket], sharp_inputs=POLYMARKET)
        assert state.has_input and state.snapshot_id == 8000
        price = state.polymarket[None]
        assert (price.snapshot_id, price.time) == (8001, _time(polymarket))
        assert price.fair[HOME] == pytest.approx(0.5 / 1.05)
        assert sum(price.fair.values()) == pytest.approx(1.0)
        assert state.input_snapshot_ids == [8000, 8001]

    def test_retail_without_the_exchange_counts_when_the_rule_does_not_need_it(self) -> None:
        retail = _snap(2, BET365)
        polymarket = _polymarket_snap(4, (0.5, 0.3, 0.2), snapshot_id=8010)
        assert _state([retail, polymarket], sharp_inputs=POLYMARKET).has_input
        assert not _state([retail, polymarket], sharp_inputs=BOTH).has_input
        assert not _state([retail, polymarket]).has_input

    def test_the_exchange_rule_ignores_polymarket(self) -> None:
        co_timed = _snap(2, BET365, _betfair((2.0, 4.0, 4.0)))
        polymarket = _polymarket_snap(1, (0.5, 0.3, 0.2), snapshot_id=8020)
        state = _state([co_timed, polymarket])
        assert state.has_input and state.polymarket == {}
        assert state.input_snapshot_ids == [co_timed[0].snapshot_id]

    def test_both_inputs_need_a_co_timed_snapshot_and_polymarket(self) -> None:
        co_timed = _snap(2, BET365, _betfair((2.0, 4.0, 4.0)))
        polymarket = _polymarket_snap(3, (0.5, 0.3, 0.2), snapshot_id=8030)
        state = _state([co_timed, polymarket], sharp_inputs=BOTH)
        assert set(state.exchange) == set(state.polymarket) == {None}
        assert not _state([co_timed], sharp_inputs=BOTH).has_input

    def test_polymarket_older_than_the_max_age_is_no_input(self) -> None:
        retail = _snap(2, BET365)
        old = _polymarket_snap(11, (0.5, 0.3, 0.2), snapshot_id=8040)
        assert not _state([retail, old], sharp_inputs=POLYMARKET).has_input

    def test_polymarket_after_t_is_ignored(self) -> None:
        retail = _snap(2, BET365)
        later = _polymarket_snap(-1, (0.5, 0.3, 0.2), snapshot_id=8050)
        assert not _state([retail, later], sharp_inputs=POLYMARKET).has_input

    def test_latest_polymarket_snapshot_wins(self) -> None:
        retail = _snap(2, BET365)
        old = _polymarket_snap(6, (0.5, 0.3, 0.2), snapshot_id=8060)
        new = _polymarket_snap(3, (0.4, 0.3, 0.3), snapshot_id=8061)
        state = _state([retail, old, new], sharp_inputs=POLYMARKET)
        assert state.polymarket[None].snapshot_id == 8061

    def test_an_incomplete_polymarket_snapshot_is_skipped(self) -> None:
        retail = _snap(2, BET365)
        complete = _polymarket_snap(6, (0.5, 0.3, 0.2), snapshot_id=8070)
        partial = [q for q in _polymarket_snap(3, (0.4, 0.3, 0.3), 8071) if q.outcome != AWAY]
        state = _state([retail, complete, partial], sharp_inputs=POLYMARKET)
        assert state.polymarket[None].snapshot_id == 8070

    def test_only_mids_count(self) -> None:
        retail = _snap(2, BET365)
        polymarket = _polymarket_snap(3, (0.5, 0.3, 0.2), snapshot_id=8080)
        backs = [q for q in polymarket if q.side is not Side.MID]
        assert not _state([retail, backs], sharp_inputs=POLYMARKET).has_input


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
        event_id=1,
        sport_key="soccer_epl",
        market="1x2",
        kickoff=KICKOFF,
        decision_time=T,
        snapshot_id=1,
        snapshot_time=T,
        retail={None: retail} if retail else {},
        exchange={} if exchange is None else {None: exchange},
    )


def _polymarket(home: float, draw: float, away: float) -> PolymarketPrice:
    return PolymarketPrice(snapshot_id=2, time=T, fair={HOME: home, DRAW: draw, AWAY: away})


def _totals_exchange(over: float) -> ExchangePrice:
    return replace(_exchange(0, 0, 0), fair={OVER: over, UNDER: 1 - over})


def _totals_state(
    retail: dict[float, dict[str, dict[Outcome, float]]], exchange: dict[float, ExchangePrice]
) -> MarketState:
    return replace(_rule_state({}, None), market=Market.TOTALS, retail=retail, exchange=exchange)


PARAMS = GapParams(threshold=0.02, minutes_before_kickoff=(60,), max_input_age_minutes=10)


class TestExchangeGap:
    def test_no_exchange_no_bets(self) -> None:
        assert exchange_gap(_rule_state({"bet365": {HOME: 9.0}}, None), PARAMS) == []

    def test_threshold_is_strict(self) -> None:
        # 0.5 * 2.5 - 1 = 0.25, clearly above; edge exactly at threshold is not a bet.
        state = _rule_state({"bet365": {HOME: 2.5}}, _exchange(0.5, 0.25, 0.25))
        at = GapParams(threshold=0.25, minutes_before_kickoff=(60,), max_input_age_minutes=10)
        below = GapParams(threshold=0.2499, minutes_before_kickoff=(60,), max_input_age_minutes=10)
        assert exchange_gap(state, at) == []
        assert exchange_gap(state, below) == [
            BetIntent(HOME, "bet365", 2.5, 1.0, estimated_ev=pytest.approx(0.25))
        ]

    def test_longest_price_across_books_is_chosen(self) -> None:
        state = _rule_state(
            {"bet365": {HOME: 2.4}, "betway": {HOME: 2.6}, "coral": {HOME: 2.5}},
            _exchange(0.5, 0.25, 0.25),
        )
        assert exchange_gap(state, PARAMS) == [
            BetIntent(HOME, "betway", 2.6, 1.0, estimated_ev=pytest.approx(0.5 * 2.6 - 1))
        ]

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
            BetIntent(DRAW, "bet365", 5.0, 1.0, estimated_ev=pytest.approx(0.25)),
            BetIntent(AWAY, "betway", 6.0, 1.0, estimated_ev=pytest.approx(0.5)),
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
        assert exchange_gap(state, PARAMS) == [
            BetIntent(OVER, "bet365", 2.2, 1.0, 2.5, estimated_ev=pytest.approx(0.1))
        ]

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
            BetIntent(UNDER, "bet365", 2.2, 1.0, 2.5, estimated_ev=pytest.approx(0.1)),
            BetIntent(OVER, "betway", 4.5, 1.0, 3.5, estimated_ev=pytest.approx(0.125)),
        ]

    @pytest.mark.parametrize("field", ["threshold", "stake", "max_input_age_minutes"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_params_must_be_positive(self, field: str, value: float) -> None:
        raw: dict[str, Any] = {**PARAMS.model_dump(), field: value}
        with pytest.raises(ValidationError):
            GapParams(**raw)

    def test_max_input_age_is_required(self) -> None:
        with pytest.raises(ValidationError):
            GapParams(threshold=0.02, minutes_before_kickoff=(60,))  # type: ignore[call-arg]

    @pytest.mark.parametrize("minutes", [[], [0], [60, -1], [60, 60]])
    def test_decision_times_are_distinct_positive_and_at_least_one(
        self, minutes: list[int]
    ) -> None:
        with pytest.raises(ValidationError):
            GapParams(**{**PARAMS.model_dump(), "minutes_before_kickoff": minutes})

    def test_decision_times_are_held_earliest_first(self) -> None:
        params = GapParams(**{**PARAMS.model_dump(), "minutes_before_kickoff": [30, 180, 60]})
        assert params.minutes_before_kickoff == (180, 60, 30)

    def test_params_reject_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            GapParams(**PARAMS.model_dump(), surprise=1)


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


def _bet(
    outcome: Outcome, book: str = "bet365", line: float | None = None, ev: float | None = 0.05
) -> BetIntent:
    return BetIntent(outcome, book, 2.0, 1.0, line, estimated_ev=ev)


class TestBestBet:
    def test_keeps_the_highest_estimated_ev(self) -> None:
        bets = [_bet(HOME, ev=0.02), _bet(AWAY, ev=0.05), _bet(DRAW, ev=0.03)]
        assert best_bet(Market.ONE_X_TWO, bets) == [_bet(AWAY, ev=0.05)]

    def test_no_qualifying_bet_gives_none(self) -> None:
        assert best_bet(Market.ONE_X_TWO, []) == []

    def test_a_tie_goes_to_the_lowest_line(self) -> None:
        bets = [_bet(OVER, line=3.5), _bet(UNDER, line=2.5)]
        assert best_bet(Market.TOTALS, bets) == [_bet(UNDER, line=2.5)]

    def test_then_to_the_outcome_first_in_market_order(self) -> None:
        assert best_bet(Market.ONE_X_TWO, [_bet(AWAY), _bet(DRAW)]) == [_bet(DRAW)]
        totals = [_bet(UNDER, line=2.5), _bet(OVER, line=2.5)]
        assert best_bet(Market.TOTALS, totals) == [_bet(OVER, line=2.5)]

    def test_then_to_the_first_bookmaker_key(self) -> None:
        bets = [_bet(HOME, "williamhill"), _bet(HOME, "betfred")]
        assert best_bet(Market.ONE_X_TWO, bets) == [_bet(HOME, "betfred")]

    def test_bets_without_an_estimate_tie(self) -> None:
        bets = [_bet(AWAY, ev=None), _bet(HOME, ev=None)]
        assert best_bet(Market.ONE_X_TWO, bets) == [_bet(HOME, ev=None)]


class TestEstimateBasis:
    @pytest.mark.parametrize(
        ("rule", "basis"),
        [
            ("exchange_gap", EstimateBasis.EXCHANGE),
            ("exchange_steam", EstimateBasis.EXCHANGE),
            ("team_news_lag", EstimateBasis.EXCHANGE),
            ("polymarket_gap", EstimateBasis.POLYMARKET),
            ("retail_ev_model", EstimateBasis.MODEL),
            ("retail_quorum_lag", EstimateBasis.RETAIL_CONSENSUS),
            ("single_book_outlier", EstimateBasis.EXCHANGE),
            ("subsidised_outcome", EstimateBasis.RETAIL_CONSENSUS),
            ("favourite_price", EstimateBasis.EXCHANGE),
            ("crowded_gap", EstimateBasis.EXCHANGE),
            ("last_result_overreaction", EstimateBasis.NONE),
        ],
    )
    def test_each_rule_declares_the_fair_price_of_its_estimate(
        self, rule: str, basis: EstimateBasis
    ) -> None:
        assert RULES[rule].estimate_basis is basis

    def test_every_rule_is_covered(self) -> None:
        assert len(RULES) == 11

    def test_a_bound_model_rule_keeps_its_basis(self) -> None:
        spec = RULES["retail_ev_model"]
        assert isinstance(spec, ModelRuleSpec)
        assert spec.bind(_predicting({})).estimate_basis is EstimateBasis.MODEL


class TestPolymarketGap:
    def test_bets_the_gap_to_the_polymarket_fair_price(self) -> None:
        state = replace(
            _rule_state({"bet365": {HOME: 2.5, DRAW: 3.0}}, None),
            polymarket={None: _polymarket(0.5, 0.3, 0.2)},
        )
        assert polymarket_gap(state, PARAMS) == [
            BetIntent(HOME, "bet365", 2.5, 1.0, estimated_ev=pytest.approx(0.25))
        ]

    def test_ignores_the_exchange(self) -> None:
        state = _rule_state({"bet365": {HOME: 2.5}}, _exchange(0.5, 0.25, 0.25))
        assert polymarket_gap(state, PARAMS) == []


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
            BetIntent(HOME, "bet365", 2.1, 1.0, estimated_ev=0.02),
            BetIntent(DRAW, "betway", 3.5, 1.0, estimated_ev=0.011),
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
            BetIntent(AWAY, "bet365", 4.0, 1.0, estimated_ev=0.05)
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
        assert bets == [BetIntent(HOME, "bet365", 2.1, 1.0, estimated_ev=0.5)]


class TestApplyRule:
    def test_state_without_input_gives_no_bets_whatever_the_rule(self) -> None:
        always = RuleSpec(
            lambda state, params: [BetIntent(HOME, "bet365", 2.0, 1.0)],
            GapParams,
            EXCHANGE,
            estimate_basis=EstimateBasis.NONE,
        )
        empty = MarketState(1, "soccer_epl", "1x2", KICKOFF, T)
        assert apply_rule(always, PARAMS, empty)[1] == []
        assert apply_rule(always, PARAMS, _rule_state({}, None))[1] != []

    def test_selections_already_bet_are_dropped(self) -> None:
        both = RuleSpec(
            lambda state, params: [
                BetIntent(HOME, "bet365", 2.0, 1.0),
                BetIntent(AWAY, "bet365", 4.0, 1.0),
            ],
            GapParams,
            EXCHANGE,
            estimate_basis=EstimateBasis.NONE,
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
            GapParams,
            EXCHANGE,
            estimate_basis=EstimateBasis.NONE,
        )
        state = _totals_state({}, {})
        assert apply_rule(overs, PARAMS, state, bet_selections={(OVER, 2.5)})[1] == [
            BetIntent(OVER, "bet365", 3.0, 1.0, 3.5)
        ]


class TestDecisionTimes:
    def test_are_kickoff_minus_each_lead_earliest_first(self) -> None:
        params = GapParams(**{**PARAMS.model_dump(), "minutes_before_kickoff": [60, 1440]})
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
        "books": "uk_retail",
        "market": "1x2",
        "sports": ["soccer_epl"],
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
            ("basketball_nba", "h2h"),
        ],
    )
    def test_each_sport_decides_its_markets(self, sport: str, market: str) -> None:
        resolve_rule(_candidate(sports=[sport], market=market))

    @pytest.mark.parametrize(
        "sport",
        [
            "soccer_efl_champ",
            "soccer_spain_la_liga",
            "soccer_italy_serie_a",
            "soccer_germany_bundesliga",
            "soccer_france_ligue_one",
        ],
    )
    @pytest.mark.parametrize("market", ["1x2", "totals"])
    def test_each_football_league_decides_the_three_way_and_totals(
        self, sport: str, market: str
    ) -> None:
        resolve_rule(_candidate(sports=[sport], market=market))

    @pytest.mark.parametrize(
        ("sport", "market"),
        [("soccer_epl", "h2h"), ("baseball_mlb", "1x2"), ("basketball_nba", "totals")],
    )
    def test_a_market_the_sport_does_not_decide_fails(self, sport: str, market: str) -> None:
        with pytest.raises(ValueError, match=f"no rule decides market '{market}' for sport"):
            resolve_rule(_candidate(sports=[sport], market=market))

    def test_a_pooled_candidate_needs_its_market_in_every_league(self) -> None:
        resolve_rule(_candidate(sports=["soccer_epl", "soccer_italy_serie_a"]))
        with pytest.raises(ValueError, match="for sport 'basketball_nba'"):
            resolve_rule(_candidate(sports=["soccer_epl", "basketball_nba"]))

    def test_a_league_listed_twice_fails(self) -> None:
        with pytest.raises(ValidationError, match="listed twice"):
            _candidate(sports=["soccer_epl", "soccer_epl"])

    def test_registration_accepts_a_league_whose_close_is_none(self) -> None:
        validate_candidate_rules([_candidate(sports=["baseball_mlb"], market="h2h")])

    def test_registration_refuses_a_league_missing_from_closes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closes = {k: v for k, v in pinning.CLOSES.items() if k != ("baseball_mlb", Market.H2H)}
        monkeypatch.setattr(pinning, "CLOSES", closes)
        with pytest.raises(ValueError, match="baseball_mlb has no entry for market 'h2h'"):
            validate_candidate_rules([_candidate(sports=["baseball_mlb"], market="h2h")])

    def test_a_league_without_a_close_still_resolves_as_a_draft(self) -> None:
        resolve_rule(_candidate(sports=["baseball_mlb"], market="h2h"))

    def test_market_without_a_rule_fails(self) -> None:
        with pytest.raises(ValueError, match="'gap'.*no rule decides market 'spreads'"):
            validate_candidate_rules([_candidate(market="spreads")])

    def test_model_rule_without_a_model_for_the_market_fails(self) -> None:
        candidate = _candidate(
            market="totals",
            rule="retail_ev_model",
            rule_sha256="0" * 64,
            params=MODEL_PARAMS.model_dump(mode="json"),
        )
        with pytest.raises(ValueError, match="'gap'.*no model for market 'totals'"):
            resolve_rule(candidate)

    def test_registered_candidates_are_valid(self) -> None:
        validate_candidate_rules(load_candidates(candidates_dir()))

    def test_model_candidate_resolves_to_its_model_rule(self) -> None:
        candidate = _candidate(
            rule="retail_ev_model",
            rule_sha256="0" * 64,
            params=MODEL_PARAMS.model_dump(mode="json"),
        )
        spec, params = resolve_rule(candidate)
        assert isinstance(spec, ModelRuleSpec)
        assert params == MODEL_PARAMS

    def test_a_model_candidate_cannot_register_while_its_rule_is_unhashable(self) -> None:
        candidate = _candidate(
            rule="retail_ev_model",
            rule_sha256="0" * 64,
            params=MODEL_PARAMS.model_dump(mode="json"),
        )
        with pytest.raises(ValueError, match="cannot follow"):
            validate_candidate_rules([candidate])


class TestInputLeads:
    def test_active_mechanical_candidates_of_the_sport(self) -> None:
        candidates = [
            _candidate(),
            _candidate(
                name="later",
                params={**PARAMS.model_dump(mode="json"), "minutes_before_kickoff": [30, 15]},
            ),
            _candidate(name="mlb", sports=["baseball_mlb"], market="h2h"),
            _candidate(name="killed", status="killed"),
            _candidate(name="broken", params={"threshold": -1}),
        ]
        assert input_leads(candidates, "soccer_epl") == [
            timedelta(minutes=60),
            timedelta(minutes=30),
            timedelta(minutes=15),
        ]

    def test_a_pooled_candidate_gives_leads_in_each_of_its_leagues(self) -> None:
        pooled = [_candidate(sports=["soccer_epl", "soccer_italy_serie_a"])]
        assert input_leads(pooled, "soccer_italy_serie_a") == [timedelta(minutes=60)]
        assert input_leads(pooled, "soccer_efl_champ") == []

    def test_each_lookback_point_of_each_decision_time_gives_a_lead(self) -> None:
        steam = _candidate(
            rule="exchange_steam",
            rule_sha256=rule_sha256("exchange_steam"),
            params={
                "min_rise": 0.025,
                "gap": 0,
                "lookbacks": [{"minutes_before_decision": 180, "max_input_age_minutes": 60}],
                "minutes_before_kickoff": [120, 60],
                "max_input_age_minutes": 10,
            },
        )
        assert input_leads([steam], "soccer_epl") == [
            timedelta(minutes=120),
            timedelta(minutes=300),
            timedelta(minutes=60),
            timedelta(minutes=240),
        ]


MECHANICAL_RULES = [name for name, spec in RULES.items() if isinstance(spec, RuleSpec)]


def _replace(old: str, new: str) -> Callable[[str], str]:
    return lambda source: source.replace(old, new)


def _append(text: str) -> Callable[[str], str]:
    return lambda source: source + text


class TestRuleSha256:
    def test_is_a_stable_sha256(self) -> None:
        assert rule_sha256("exchange_gap") == rule_sha256("exchange_gap")
        assert len(rule_sha256("exchange_gap")) == 64

    @pytest.mark.parametrize(
        ("module", "transform"),
        [
            (
                "odds_lambda.candidate_rules.inputs",
                _replace('"""The back prices of UK retail books', '"""Back prices of UK books'),
            ),
            (
                "odds_lambda.candidate_rules.market_state",
                _replace("# A market's line, None", "# The line, None"),
            ),
            (
                "odds_core.exchange_price",
                _replace(
                    "return sum(1 / prices[o] for o in outcomes) - 1",
                    "return (\n        sum(1 / prices[o] for o in outcomes)\n        - 1\n    )",
                ),
            ),
        ],
        ids=["docstring", "comment", "formatting"],
    )
    @pytest.mark.parametrize("rule", MECHANICAL_RULES)
    def test_a_docstring_comment_or_format_edit_leaves_it_unchanged(
        self, edit_source: SourceEdit, module: str, transform: Callable[[str], str], rule: str
    ) -> None:
        before = rule_sha256(rule)
        edit_source(module, transform)
        assert rule_sha256(rule) == before

    @pytest.mark.parametrize(
        ("module", "transform"),
        [
            # The close the report scores against is outside a mechanical rule's imports.
            (
                "odds_core.close",
                _replace("return CloseStatus(CloseKind.UNSUPPORTED)", "return None"),
            ),
            (
                "odds_analytics.retail_ev",
                _replace("LOSS_WEIGHT_SCALE = 1013.0", "LOSS_WEIGHT_SCALE = 1.0"),
            ),
        ],
    )
    @pytest.mark.parametrize("rule", MECHANICAL_RULES)
    def test_a_module_outside_a_mechanical_rules_imports_leaves_it_unchanged(
        self, edit_source: SourceEdit, module: str, transform: Callable[[str], str], rule: str
    ) -> None:
        before = rule_sha256(rule)
        edit_source(module, transform)
        assert rule_sha256(rule) == before

    @pytest.mark.parametrize(
        ("module", "transform"),
        [
            (
                "odds_core.odds_math",
                _replace("return tuple(p / total for p in probs)", "return probs"),
            ),
            (
                "odds_core.market_prices",
                _replace("(Outcome.OVER, Outcome.UNDER)", "(Outcome.UNDER, Outcome.OVER)"),
            ),
            ("odds_core.bookmakers", _replace('        "bet365",\n', "")),
            (
                "odds_core.book_price",
                _replace("POLYMARKET_SIDE = Side.MID", "POLYMARKET_SIDE = Side.BACK"),
            ),
            ("odds_core.as_of", _replace("if accept(group):", "if not accept(group):")),
            ("odds_core.quotes", _replace("    snapshot_id: int\n", "    snapshot_id: str\n")),
            (
                "odds_lambda.candidate_rules.market_state",
                _replace("line % 1 == 0.5", "line % 1 == 0"),
            ),
            (
                "odds_lambda.candidate_rules.rules",
                _replace("b.selection not in bet_selections", "True"),
            ),
            (
                "odds_lambda.candidate_rules.market_state",
                _append("\n\ndef unrelated() -> int:\n    return 1\n"),
            ),
            ("odds_lambda", _append("VERSION = 1\n")),
        ],
    )
    @pytest.mark.parametrize("rule", MECHANICAL_RULES)
    def test_a_code_edit_in_a_closure_module_changes_it(
        self, edit_source: SourceEdit, module: str, transform: Callable[[str], str], rule: str
    ) -> None:
        before = rule_sha256(rule)
        edit_source(module, transform)
        assert rule_sha256(rule) != before

    def test_covers_the_sharp_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from odds_lambda.candidate_rules import registry

        before = rule_sha256("exchange_gap")
        monkeypatch.setitem(
            registry.RULES,
            "exchange_gap",
            replace(registry.RULES["exchange_gap"], sharp_inputs=BOTH),
        )
        assert rule_sha256("exchange_gap") != before

    def test_the_model_rule_fails_closed(self) -> None:
        # Its closure reads attributes by computed name and dunders such as ``__version__``,
        # which the hash cannot follow.
        with pytest.raises(ValueError, match="cannot follow"):
            rule_sha256("retail_ev_model")

    def test_a_stale_hash_fails_validation_with_the_new_one(self) -> None:
        with pytest.raises(ValueError, match="'gap'.*code of rule 'exchange_gap' changed") as exc:
            validate_candidate_rules([_candidate(rule_sha256="0" * 64)])
        assert rule_sha256("exchange_gap") in str(exc.value)

    def test_hash_must_be_64_hex_chars(self) -> None:
        with pytest.raises(ValidationError):
            _candidate(rule_sha256="abc")


NEW_RULE = """
from odds_lambda.candidate_rules import (
    BetIntent,
    EstimateBasis,
    MarketState,
    RuleParams,
    RuleSpec,
    SharpInput,
)


def new_rule(state: MarketState, params: RuleParams) -> list[BetIntent]:
    return []


NEW_RULE = RuleSpec(
    new_rule, RuleParams, frozenset({SharpInput.EXCHANGE}), estimate_basis=EstimateBasis.NONE
)
"""


class TestRuleIndependence:
    @pytest.mark.parametrize("rule", MECHANICAL_RULES)
    def test_no_rule_closure_holds_the_registry_or_another_rule(self, rule: str) -> None:
        spec = RULES[rule]
        found = set(closure([spec.function, spec.params_model, *RULE_ROOTS]))
        rule_modules = {root_module(s.function) for s in RULES.values()}
        assert "odds_lambda.candidate_rules.registry" not in found
        assert found & rule_modules == {root_module(spec.function)}

    def test_a_new_rule_module_and_registry_entry_leave_the_other_hashes_unchanged(
        self,
        fixture_package: Callable[[dict[str, str]], None],
        edit_source: SourceEdit,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        before = {rule: rule_sha256(rule) for rule in MECHANICAL_RULES}
        fixture_package({"new_rule": NEW_RULE})
        module = importlib.import_module(f"{FIXTURE_PACKAGE}.new_rule")
        edit_source(
            "odds_lambda.candidate_rules.registry",
            lambda source: source.replace(
                "\nRULES: dict", f"\nfrom {FIXTURE_PACKAGE}.new_rule import NEW_RULE\n\nRULES: dict"
            ).replace(
                '"retail_ev_model": RETAIL_EV_MODEL,',
                '"retail_ev_model": RETAIL_EV_MODEL,\n    "new_rule": NEW_RULE,',
            ),
        )
        monkeypatch.setitem(RULES, "new_rule", module.NEW_RULE)

        assert {rule: rule_sha256(rule) for rule in MECHANICAL_RULES} == before
        assert rule_sha256("new_rule") not in before.values()
