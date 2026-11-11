"""Unit tests for the exchange-steam rule, registered as ``exchange_steam`` and
``team_news_lag``."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.exchange_price import ExchangePrice
from odds_core.market_prices import Outcome
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.candidate_rules import BetIntent, MarketState
from odds_lambda.candidate_rules.exchange_steam import (
    EXCHANGE_STEAM,
    ExchangeSteamParams,
    exchange_steam,
)
from odds_lambda.candidate_rules.registry import RULES
from pydantic import ValidationError

HOME, DRAW, AWAY = Outcome.HOME, Outcome.DRAW, Outcome.AWAY
KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
LOOKBACK = {"minutes_before_decision": 360, "max_input_age_minutes": 60}


def params(**overrides: Any) -> ExchangeSteamParams:
    fields: dict[str, Any] = {
        "min_rise": 0.025,
        "gap": 0.0,
        "lookbacks": [LOOKBACK],
        "minutes_before_kickoff": [60],
        "max_input_age_minutes": 10,
        **overrides,
    }
    return ExchangeSteamParams.model_validate(fields)


def exchange(home: float, draw: float, away: float) -> ExchangePrice:
    return ExchangePrice(
        time=T,
        source=Source.ODDSPORTAL,
        time_basis=TimeBasis.OBSERVED,
        in_play=False,
        fair={HOME: home, DRAW: draw, AWAY: away},
        overround=0.0,
    )


def state(
    retail: dict[str, dict[Outcome, float]],
    now: ExchangePrice | None,
    before: ExchangePrice | None,
    minutes: int = 360,
) -> MarketState:
    """A decision whose lookback point ``minutes`` before it holds ``before``."""

    def one(price: ExchangePrice | None, retail_prices: dict[str, dict[Outcome, float]]) -> Any:
        return MarketState(
            event_id=1,
            sport_key="soccer_epl",
            market="1x2",
            kickoff=KICKOFF,
            decision_time=T,
            snapshot_id=1,
            snapshot_time=T,
            retail={None: retail_prices},
            exchange={} if price is None else {None: price},
        )

    return replace(one(now, retail), lookbacks={minutes: one(before, retail)})


# Home's fair probability rose by 0.05 to 0.5, draw's and away's fell.
NOW = exchange(0.5, 0.25, 0.25)
BEFORE = exchange(0.45, 0.275, 0.275)


class TestQualifies:
    def test_a_rise_with_a_longer_retail_price_is_bet(self) -> None:
        bets = exchange_steam(state({"bet365": {HOME: 2.5}}, NOW, BEFORE), params())
        assert bets == [BetIntent(HOME, "bet365", 2.5, 1.0)]

    def test_the_longest_price_across_books_is_bet(self) -> None:
        retail = {"bet365": {HOME: 2.4}, "betway": {HOME: 2.6}, "coral": {HOME: 2.5}}
        assert exchange_steam(state(retail, NOW, BEFORE), params()) == [
            BetIntent(HOME, "betway", 2.6, 1.0)
        ]

    def test_equal_prices_go_to_the_first_book_key(self) -> None:
        retail = {"coral": {HOME: 2.5}, "betway": {HOME: 2.5}}
        assert exchange_steam(state(retail, NOW, BEFORE), params()) == [
            BetIntent(HOME, "betway", 2.5, 1.0)
        ]

    def test_the_lookback_read_is_the_declared_point(self) -> None:
        lookback = {"minutes_before_decision": 25, "max_input_age_minutes": 10}
        late = state({"bet365": {HOME: 2.5}}, NOW, BEFORE, minutes=25)
        assert exchange_steam(late, params(lookbacks=[lookback])) == [
            BetIntent(HOME, "bet365", 2.5, 1.0)
        ]


class TestEachConditionFailing:
    def test_a_rise_below_min_rise_is_not_bet(self) -> None:
        small = exchange(0.48, 0.26, 0.26)
        assert exchange_steam(state({"bet365": {HOME: 2.5}}, NOW, small), params()) == []

    def test_a_rise_of_exactly_min_rise_is_bet(self) -> None:
        # 0.5 - 0.25 is exactly 0.25 in binary floating point.
        steam = state({"bet365": {HOME: 2.5}}, NOW, exchange(0.25, 0.375, 0.375))
        assert exchange_steam(steam, params(min_rise=0.25)) == [BetIntent(HOME, "bet365", 2.5, 1.0)]
        assert exchange_steam(steam, params(min_rise=0.2501)) == []

    def test_a_fall_is_not_a_rise(self) -> None:
        assert exchange_steam(state({"bet365": {HOME: 2.5}}, BEFORE, NOW), params()) == []

    def test_a_gap_not_exceeded_is_not_bet(self) -> None:
        # 0.5 * 2.5 - 1 = 0.25: the gap is strict.
        steam = state({"bet365": {HOME: 2.5}}, NOW, BEFORE)
        assert exchange_steam(steam, params(gap=0.25)) == []
        assert exchange_steam(steam, params(gap=0.2499)) == [BetIntent(HOME, "bet365", 2.5, 1.0)]

    def test_a_retail_price_at_the_fair_price_is_not_bet_at_gap_zero(self) -> None:
        assert exchange_steam(state({"bet365": {HOME: 2.0}}, NOW, BEFORE), params()) == []

    def test_a_lookback_without_an_exchange_price_is_not_bet(self) -> None:
        assert exchange_steam(state({"bet365": {HOME: 2.5}}, NOW, None), params()) == []

    def test_a_decision_without_an_exchange_price_is_not_bet(self) -> None:
        assert exchange_steam(state({"bet365": {HOME: 2.5}}, None, BEFORE), params()) == []

    def test_an_outcome_without_a_retail_price_is_not_bet(self) -> None:
        assert exchange_steam(state({"bet365": {DRAW: 9.0}}, NOW, BEFORE), params()) == []

    def test_no_retail_is_not_bet(self) -> None:
        assert exchange_steam(state({}, NOW, BEFORE), params()) == []


class TestOneBetPerDecision:
    # Every outcome rose, so each qualifies on its own.
    BEFORE = exchange(0.25, 0.125, 0.125)

    def test_the_highest_estimated_ev_is_bet(self) -> None:
        # EVs: home 0.5 * 2.6 - 1 = 0.3, draw 0.25 * 6 - 1 = 0.5, away 0.25 * 5 - 1 = 0.25.
        retail = {"bet365": {HOME: 2.6, DRAW: 6.0, AWAY: 5.0}}
        assert exchange_steam(state(retail, NOW, self.BEFORE), params(min_rise=0.1)) == [
            BetIntent(DRAW, "bet365", 6.0, 1.0)
        ]

    def test_an_outcome_below_min_rise_does_not_compete(self) -> None:
        # Home rose 0.05 at EV 0.5 * 3.2 - 1 = 0.6, above draw's 0.5, so only the filter drops it.
        before = exchange(0.45, 0.125, 0.125)
        retail = {"bet365": {HOME: 3.2, DRAW: 6.0}}
        assert exchange_steam(state(retail, NOW, before), params(min_rise=0.1)) == [
            BetIntent(DRAW, "bet365", 6.0, 1.0)
        ]

    def test_equal_ev_goes_to_the_outcome_first_in_market_order(self) -> None:
        # Each EV is exactly 0.25.
        retail = {"bet365": {HOME: 2.5, DRAW: 5.0, AWAY: 5.0}}
        steam = state(retail, NOW, self.BEFORE)
        assert exchange_steam(steam, params(min_rise=0.1)) == [BetIntent(HOME, "bet365", 2.5, 1.0)]
        no_home = state({"bet365": {DRAW: 5.0, AWAY: 5.0}}, NOW, self.BEFORE)
        assert exchange_steam(no_home, params(min_rise=0.1)) == [
            BetIntent(DRAW, "bet365", 5.0, 1.0)
        ]
        only_away = state({"bet365": {AWAY: 5.0}}, NOW, self.BEFORE)
        assert exchange_steam(only_away, params(min_rise=0.1)) == [
            BetIntent(AWAY, "bet365", 5.0, 1.0)
        ]


class TestParams:
    @pytest.mark.parametrize(
        "lookbacks", [[], [LOOKBACK, {**LOOKBACK, "minutes_before_decision": 180}]]
    )
    def test_exactly_one_lookback_is_required(self, lookbacks: list[dict[str, int]]) -> None:
        with pytest.raises(ValidationError):
            params(lookbacks=lookbacks)

    def test_the_lookback_is_required(self) -> None:
        with pytest.raises(ValidationError):
            ExchangeSteamParams.model_validate(
                {
                    "min_rise": 0.01,
                    "gap": 0,
                    "minutes_before_kickoff": [60],
                    "max_input_age_minutes": 10,
                }
            )

    @pytest.mark.parametrize("min_rise", [0, -0.01])
    def test_min_rise_is_positive(self, min_rise: float) -> None:
        with pytest.raises(ValidationError):
            params(min_rise=min_rise)

    def test_gap_is_non_negative(self) -> None:
        assert params(gap=0).gap == 0
        with pytest.raises(ValidationError):
            params(gap=-0.01)

    def test_unknown_params_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            params(stake=2.0)


class TestRegistration:
    def test_both_names_are_one_spec(self) -> None:
        assert RULES["exchange_steam"] is RULES["team_news_lag"] is EXCHANGE_STEAM
