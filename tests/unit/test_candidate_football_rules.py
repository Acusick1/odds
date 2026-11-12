"""The favourite-price, crowded-gap and last-result rules of round 1 on hand-built states."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import pytest
from odds_core.exchange_price import ExchangePrice
from odds_core.market_prices import Outcome
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.candidate_rules import (
    BetIntent,
    FixtureContext,
    FixtureInput,
    MarketState,
    PreviousResult,
    apply_rule,
)
from odds_lambda.candidate_rules.crowded_gap import CrowdedGapParams, crowded_gap
from odds_lambda.candidate_rules.favourite_price import FavouriteParams, favourite_price
from odds_lambda.candidate_rules.last_result_overreaction import (
    LastResultParams,
    last_result_overreaction,
)
from odds_lambda.candidate_rules.registry import RULES
from pydantic import ValidationError

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
HOME, DRAW, AWAY = Outcome.HOME, Outcome.DRAW, Outcome.AWAY
BASE: dict[str, Any] = {"minutes_before_kickoff": (60,), "max_input_age_minutes": 10}

Retail = dict[str, dict[Outcome, float]]


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
    retail: Retail,
    fair: ExchangePrice | None = None,
    fixture: FixtureContext | None = None,
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
        exchange={} if fair is None else {None: fair},
        fixture=fixture or FixtureContext(),
    )


def lost_by(margin: int) -> PreviousResult:
    return PreviousResult(2, KICKOFF - timedelta(days=6), 0, margin)


def previous(
    home: PreviousResult | None = None, away: PreviousResult | None = None
) -> FixtureContext:
    return FixtureContext(previous=MappingProxyType({HOME: home, AWAY: away}))


class TestFavouritePrice:
    PARAMS = FavouriteParams(min_probability=0.6, threshold=0.0, **BASE)

    def test_the_favourite_at_its_longest_price_when_the_gap_is_above_threshold(self) -> None:
        s = state(
            {"bet365": {HOME: 1.8, AWAY: 9.0}, "betway": {HOME: 1.9}}, exchange(0.6, 0.25, 0.15)
        )
        assert favourite_price(s, self.PARAMS) == [BetIntent(HOME, "betway", 1.9, 1.0)]

    def test_an_outcome_below_the_minimum_probability_is_not_bet(self) -> None:
        s = state({"bet365": {HOME: 2.5, DRAW: 9.0}}, exchange(0.55, 0.35, 0.1))
        assert favourite_price(s, self.PARAMS) == []

    def test_the_minimum_probability_is_inclusive(self) -> None:
        s = state({"bet365": {HOME: 1.9}}, exchange(0.6, 0.25, 0.15))
        assert favourite_price(s, self.PARAMS) == [BetIntent(HOME, "bet365", 1.9, 1.0)]

    def test_a_gap_at_or_below_the_threshold_is_not_bet(self) -> None:
        # 0.8 × 1.25 − 1 is exactly zero.
        s = state({"bet365": {HOME: 1.25}}, exchange(0.8, 0.1, 0.1))
        assert favourite_price(s, self.PARAMS) == []
        below = FavouriteParams(min_probability=0.6, threshold=-0.005, **BASE)
        assert favourite_price(s, below) == [BetIntent(HOME, "bet365", 1.25, 1.0)]

    def test_a_longshot_with_a_gap_is_not_bet(self) -> None:
        s = state({"bet365": {HOME: 1.2, AWAY: 20.0}}, exchange(0.6, 0.25, 0.15))
        assert favourite_price(s, self.PARAMS) == []

    def test_a_favourite_no_book_quotes_is_not_bet(self) -> None:
        s = state({"bet365": {AWAY: 20.0}}, exchange(0.6, 0.25, 0.15))
        assert favourite_price(s, self.PARAMS) == []

    def test_no_exchange_price_no_bet(self) -> None:
        assert favourite_price(state({"bet365": {HOME: 9.0}}), self.PARAMS) == []

    def test_a_price_tie_goes_to_the_first_book_by_name(self) -> None:
        s = state({"coral": {HOME: 1.9}, "betway": {HOME: 1.9}}, exchange(0.6, 0.25, 0.15))
        assert [b.bookmaker for b in favourite_price(s, self.PARAMS)] == ["betway"]

    def test_stake_comes_from_params(self) -> None:
        s = state({"bet365": {HOME: 1.9}}, exchange(0.6, 0.25, 0.15))
        params = FavouriteParams(min_probability=0.6, threshold=0.0, stake=2.5, **BASE)
        assert favourite_price(s, params)[0].stake == 2.5

    @pytest.mark.parametrize("min_probability", [0.5, 1.0, 0.3, 1.2])
    def test_the_minimum_probability_is_strictly_between_a_half_and_one(
        self, min_probability: float
    ) -> None:
        with pytest.raises(ValidationError):
            FavouriteParams(min_probability=min_probability, threshold=0.0, **BASE)

    @pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
    def test_the_threshold_is_a_finite_float(self, threshold: float) -> None:
        with pytest.raises(ValidationError):
            FavouriteParams(min_probability=0.6, threshold=threshold, **BASE)

    def test_a_negative_threshold_is_allowed(self) -> None:
        assert FavouriteParams(min_probability=0.55, threshold=-0.005, **BASE).threshold == -0.005


class TestCrowdedGap:
    PARAMS = CrowdedGapParams(min_concurrent_kickoffs=3, threshold=0.0, **BASE)
    FAIR = exchange(0.5, 0.25, 0.25)

    @staticmethod
    def crowd(count: int | None) -> FixtureContext:
        return FixtureContext(concurrent_kickoffs=count)

    def test_the_gap_bet_when_enough_kickoffs_are_concurrent(self) -> None:
        s = state({"bet365": {HOME: 2.4}, "betway": {HOME: 2.6}}, self.FAIR, self.crowd(3))
        assert crowded_gap(s, self.PARAMS) == [BetIntent(HOME, "betway", 2.6, 1.0)]

    def test_too_few_concurrent_kickoffs_no_bet(self) -> None:
        s = state({"bet365": {HOME: 2.6}}, self.FAIR, self.crowd(2))
        assert crowded_gap(s, self.PARAMS) == []

    def test_an_unloaded_count_is_no_bet(self) -> None:
        assert crowded_gap(state({"bet365": {HOME: 2.6}}, self.FAIR), self.PARAMS) == []

    def test_no_gap_no_bet(self) -> None:
        s = state({"bet365": {HOME: 1.9}}, self.FAIR, self.crowd(6))
        assert crowded_gap(s, self.PARAMS) == []

    def test_a_gap_at_the_threshold_is_not_bet(self) -> None:
        # 0.5 × 2.5 − 1 is exactly 0.25.
        s = state({"bet365": {HOME: 2.5}}, self.FAIR, self.crowd(3))
        at = CrowdedGapParams(min_concurrent_kickoffs=3, threshold=0.25, **BASE)
        assert crowded_gap(s, at) == []

    def test_no_exchange_price_no_bet(self) -> None:
        assert crowded_gap(state({"bet365": {HOME: 9.0}}, None, self.crowd(6)), self.PARAMS) == []

    def test_only_the_bet_of_the_highest_ev(self) -> None:
        s = state({"bet365": {HOME: 2.4, DRAW: 5.0, AWAY: 4.4}}, self.FAIR, self.crowd(3))
        assert crowded_gap(s, self.PARAMS) == [BetIntent(DRAW, "bet365", 5.0, 1.0)]

    def test_an_ev_tie_goes_to_the_outcome_first_in_market_order(self) -> None:
        s = state(
            {"bet365": {HOME: 3.0, DRAW: 3.0, AWAY: 1.0}}, exchange(0.4, 0.4, 0.2), self.crowd(3)
        )
        assert crowded_gap(s, self.PARAMS) == [BetIntent(HOME, "bet365", 3.0, 1.0)]

    def test_an_ev_tie_across_books_goes_to_the_first_book_by_name(self) -> None:
        s = state({"coral": {HOME: 2.6}, "betway": {HOME: 2.6}}, self.FAIR, self.crowd(3))
        assert [b.bookmaker for b in crowded_gap(s, self.PARAMS)] == ["betway"]

    @pytest.mark.parametrize(
        "fields",
        [{"min_concurrent_kickoffs": 0}, {"threshold": -0.01}, {"threshold": float("inf")}],
    )
    def test_invalid_params_are_refused(self, fields: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            CrowdedGapParams(**{"min_concurrent_kickoffs": 3, "threshold": 0.0, **BASE, **fields})


class TestLastResultOverreaction:
    PARAMS = LastResultParams(min_margin=2, **BASE)

    def test_a_side_that_lost_by_the_margin_at_its_longest_price(self) -> None:
        s = state({"bet365": {HOME: 2.0}, "betway": {HOME: 2.2}}, fixture=previous(home=lost_by(2)))
        assert last_result_overreaction(s, self.PARAMS) == [BetIntent(HOME, "betway", 2.2, 1.0)]

    def test_the_away_side_qualifies_too(self) -> None:
        s = state({"bet365": {HOME: 2.0, AWAY: 3.5}}, fixture=previous(away=lost_by(4)))
        assert last_result_overreaction(s, self.PARAMS) == [BetIntent(AWAY, "bet365", 3.5, 1.0)]

    def test_a_smaller_loss_does_not_qualify(self) -> None:
        s = state({"bet365": {HOME: 2.0}}, fixture=previous(home=lost_by(1)))
        assert last_result_overreaction(s, self.PARAMS) == []

    @pytest.mark.parametrize("scored", [2, 5])
    def test_a_draw_or_a_win_does_not_qualify(self, scored: int) -> None:
        drawn_or_won = PreviousResult(2, KICKOFF - timedelta(days=6), scored, 2)
        s = state({"bet365": {HOME: 2.0}}, fixture=previous(home=drawn_or_won))
        assert last_result_overreaction(s, self.PARAMS) == []

    def test_a_team_with_no_previous_match_does_not_qualify(self) -> None:
        s = state({"bet365": {HOME: 2.0, AWAY: 3.0}}, fixture=previous(None, None))
        assert last_result_overreaction(s, self.PARAMS) == []

    def test_a_state_without_a_context_does_not_qualify(self) -> None:
        assert last_result_overreaction(state({"bet365": {HOME: 2.0}}), self.PARAMS) == []

    def test_a_side_no_book_quotes_does_not_qualify(self) -> None:
        s = state({"bet365": {AWAY: 3.0}}, fixture=previous(home=lost_by(3)))
        assert last_result_overreaction(s, self.PARAMS) == []

    def test_both_sides_qualifying_bets_home(self) -> None:
        s = state(
            {"bet365": {HOME: 2.0, AWAY: 9.0}}, fixture=previous(home=lost_by(2), away=lost_by(5))
        )
        assert last_result_overreaction(s, self.PARAMS) == [BetIntent(HOME, "bet365", 2.0, 1.0)]

    def test_the_other_side_bets_when_the_home_side_has_no_price(self) -> None:
        s = state({"bet365": {AWAY: 3.0}}, fixture=previous(home=lost_by(2), away=lost_by(2)))
        assert last_result_overreaction(s, self.PARAMS) == [BetIntent(AWAY, "bet365", 3.0, 1.0)]

    def test_a_larger_margin_param_needs_a_larger_loss(self) -> None:
        s = state({"bet365": {HOME: 2.0}}, fixture=previous(home=lost_by(3)))
        assert last_result_overreaction(s, LastResultParams(min_margin=4, **BASE)) == []
        assert last_result_overreaction(s, LastResultParams(min_margin=3, **BASE)) != []

    @pytest.mark.parametrize("min_margin", [0, -1])
    def test_the_margin_is_positive(self, min_margin: int) -> None:
        with pytest.raises(ValidationError):
            LastResultParams(min_margin=min_margin, **BASE)


class TestRegistered:
    @pytest.mark.parametrize(
        ("rule", "fixture_inputs"),
        [
            ("favourite_price", frozenset()),
            ("crowded_gap", frozenset({FixtureInput.CONCURRENT_KICKOFFS})),
            ("last_result_overreaction", frozenset({FixtureInput.PREVIOUS_RESULT})),
        ],
    )
    def test_each_rule_declares_the_fixture_inputs_it_reads(
        self, rule: str, fixture_inputs: frozenset[FixtureInput]
    ) -> None:
        assert RULES[rule].fixture_inputs == fixture_inputs

    def test_a_state_without_inputs_gives_no_bets(self) -> None:
        empty = MarketState(1, "soccer_epl", "1x2", KICKOFF, T)
        spec, params = RULES["last_result_overreaction"], LastResultParams(min_margin=2, **BASE)
        assert apply_rule(spec, params, empty)[1] == []  # type: ignore[arg-type]

    def test_an_already_bet_selection_is_not_bet_again(self) -> None:
        s = state({"bet365": {HOME: 2.0}}, fixture=previous(home=lost_by(2)))
        spec, params = RULES["last_result_overreaction"], LastResultParams(min_margin=2, **BASE)
        assert apply_rule(spec, params, s)[1] != []  # type: ignore[arg-type]
        assert apply_rule(spec, params, s, bet_selections={(HOME, None)})[1] == []  # type: ignore[arg-type]
