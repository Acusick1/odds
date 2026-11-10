"""A rule that declares lookback points reads the market state as of each one."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import pytest
from odds_core.market_prices import Outcome
from odds_core.quotes import Quote
from odds_core.vocabulary import Market
from odds_lambda.candidate_rules import (
    RULES,
    GapParams,
    Lookback,
    MarketState,
    SharpInput,
    build_market_state,
    input_window,
    lookback_points,
)
from pydantic import ValidationError

from tests.test_helpers import make_epl_event
from tests.unit.conftest import CROSS_KICKOFF, CROSS_T, RunQuotes

EVENT = make_epl_event(1, CROSS_KICKOFF)
MAX_AGE = timedelta(minutes=10)
EXCHANGE = frozenset({SharpInput.EXCHANGE})
SIX_HOURS = Lookback(360, timedelta(minutes=60))
THREE_HOURS = Lookback(180, timedelta(minutes=30))


def _state(
    quotes: list[Quote],
    lookbacks: tuple[Lookback, ...] = (SIX_HOURS,),
    also: frozenset[Market] = frozenset(),
) -> MarketState:
    return build_market_state(EVENT, "1x2", quotes, CROSS_T, MAX_AGE, EXCHANGE, also, lookbacks)


class _Base:
    """Snapshots of event 1's 1x2 (and totals) stamped minutes before the decision time."""

    run: Callable[..., list[Quote]]

    @pytest.fixture(autouse=True)
    def _bind(self, cross_market_run: RunQuotes) -> None:
        def run(first_id: int, minutes: float, totals: float | None = None) -> list[Quote]:
            return cross_market_run(first_id, first_id, one_x_two=minutes, totals=totals)

        self.run = run

    def _quotes(self, decision: float = 2, six_hours: float | None = 390) -> list[Quote]:
        quotes = self.run(100, decision)
        return quotes + self.run(110, six_hours) if six_hours is not None else quotes


class TestLookbackIsRead(_Base):
    def test_a_rule_declaring_a_lookback_receives_its_state(self) -> None:
        state = _state(self._quotes())
        point = state.lookbacks[360]
        assert state.has_input and point.has_input
        assert point.snapshot_id == 110 and point.decision_time == CROSS_T - timedelta(hours=6)
        assert point.lookbacks == {} and set(point.retail) == {None}

    def test_a_rule_declaring_nothing_has_no_lookback_state(self) -> None:
        state = _state(self._quotes(six_hours=None), lookbacks=())
        assert state.has_input and state.lookbacks == {} and state.input_snapshot_ids == [100]

    def test_several_lookbacks_are_each_read(self) -> None:
        quotes = self._quotes() + self.run(120, 180 + 5)
        state = _state(quotes, (SIX_HOURS, THREE_HOURS))
        assert {m: s.snapshot_id for m, s in state.lookbacks.items()} == {180: 120, 360: 110}

    def test_the_lookback_takes_the_latest_snapshot_at_or_before_its_point(self) -> None:
        quotes = self._quotes() + self.run(120, 360 + 20) + self.run(130, 360 - 1)
        assert _state(quotes).lookbacks[360].snapshot_id == 120

    def test_the_decision_keeps_its_own_max_age(self) -> None:
        assert not _state(self._quotes(decision=11)).has_input


class TestMissingOrStaleLookback(_Base):
    def test_a_missing_lookback_gives_no_input(self) -> None:
        state = _state(self._quotes(six_hours=None))
        assert not state.has_input and state.lookbacks == {} and state.input_snapshot_ids == []

    def test_a_stale_lookback_gives_no_input(self) -> None:
        assert not _state(self._quotes(six_hours=360 + 61)).has_input

    def test_a_lookback_at_its_max_age_counts(self) -> None:
        assert _state(self._quotes(six_hours=360 + 60)).has_input

    def test_the_lookback_age_is_not_the_decisions(self) -> None:
        # 45 minutes old is stale for the decision (10) and fresh for the lookback (60).
        assert _state(self._quotes(six_hours=360 + 45)).has_input

    def test_one_missing_lookback_of_two_gives_no_input(self) -> None:
        assert not _state(self._quotes(), (SIX_HOURS, THREE_HOURS)).has_input

    def test_a_missing_decision_gives_no_input_though_the_lookback_is_fresh(self) -> None:
        assert not _state(self._quotes(decision=11)).has_input


class TestNoLookAhead(_Base):
    def test_a_snapshot_after_the_lookback_point_is_not_its_state(self) -> None:
        # Fresh for the decision and just after the point: the point has only the older one.
        quotes = self._quotes(six_hours=360 + 50) + self.run(120, 360 - 5)
        point = _state(quotes).lookbacks[360]
        assert point.snapshot_id == 110 and point.snapshot_time <= point.decision_time

    def test_only_a_snapshot_after_the_point_gives_no_input(self) -> None:
        assert not _state(self._quotes(six_hours=360 - 5)).has_input

    def test_every_lookback_is_stamped_at_or_before_its_point(self) -> None:
        quotes = self._quotes() + self.run(120, 180 + 5) + self.run(130, 180 - 5)
        state = _state(quotes, (SIX_HOURS, THREE_HOURS))
        for point in state.lookbacks.values():
            assert point.snapshot_time is not None and point.snapshot_time <= point.decision_time

    def test_the_window_covers_the_furthest_lookback(self) -> None:
        assert input_window(MAX_AGE) == MAX_AGE
        assert input_window(MAX_AGE, (SIX_HOURS, THREE_HOURS)) == timedelta(minutes=420)


class TestRecordedInputs(_Base):
    def test_the_lookback_snapshots_are_recorded_after_the_decisions(self) -> None:
        quotes = self._quotes() + self.run(120, 180 + 5)
        assert _state(quotes, (SIX_HOURS, THREE_HOURS)).input_snapshot_ids == [100, 120, 110]

    def test_with_an_extra_market_each_point_gets_both(self) -> None:
        quotes = self.run(100, 2, totals=2) + self.run(110, 390, totals=390)
        state = _state(quotes, also=frozenset({Market.TOTALS}))
        point = state.lookbacks[360]
        assert point.also[Market.TOTALS].snapshot_id == 111
        assert state.input_snapshot_ids == [100, 101, 110, 111]

    def test_an_extra_market_missing_at_the_point_gives_no_input(self) -> None:
        quotes = self.run(100, 2, totals=2) + self.run(110, 390)
        assert not _state(quotes, also=frozenset({Market.TOTALS})).has_input


class TestReadOnly(_Base):
    def test_a_rule_cannot_change_the_lookbacks(self) -> None:
        state = _state(self._quotes())
        with pytest.raises(TypeError):
            state.lookbacks[360] = state  # type: ignore[index]
        with pytest.raises(AttributeError):
            state.lookbacks.clear()  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            state.input_ids = ()  # type: ignore[misc]
        assert state.input_snapshot_ids == [100, 110]

    def test_a_lookbacks_prices_cannot_be_mutated(self) -> None:
        point = _state(self._quotes()).lookbacks[360]
        with pytest.raises(TypeError):
            point.retail[None]["bet365"][Outcome.HOME] = 9.0  # type: ignore[index]
        with pytest.raises(TypeError):
            point.exchange[None].fair[Outcome.HOME] = 0.9  # type: ignore[index]
        with pytest.raises(TypeError):
            point.lookbacks[1] = point  # type: ignore[index]


class TestDeclaration:
    def _params(self, *points: dict[str, int]) -> GapParams:
        return GapParams(
            threshold=0.01,
            minutes_before_kickoff=[60],
            max_input_age_minutes=10,
            lookbacks=points,  # type: ignore[arg-type]
        )

    def test_a_lookback_is_given_in_params_with_its_own_age(self) -> None:
        params = self._params(
            {"minutes_before_decision": 360, "max_input_age_minutes": 60},
            {"minutes_before_decision": 180, "max_input_age_minutes": 30},
        )
        assert lookback_points(params) == (THREE_HOURS, SIX_HOURS)

    def test_a_candidate_without_lookbacks_declares_none(self) -> None:
        params = GapParams(threshold=0.01, minutes_before_kickoff=[60], max_input_age_minutes=10)
        assert params.lookbacks == () and lookback_points(params) == ()

    def test_points_must_be_distinct_and_positive(self) -> None:
        point = {"minutes_before_decision": 60, "max_input_age_minutes": 10}
        with pytest.raises(ValidationError, match="distinct"):
            self._params(point, point)
        with pytest.raises(ValidationError):
            self._params({"minutes_before_decision": 0, "max_input_age_minutes": 10})

    def test_the_registered_rules_are_unchanged_without_lookbacks(self) -> None:
        assert RULES.keys() >= {"exchange_gap", "polymarket_gap"}
        assert GapParams.model_fields["lookbacks"].default == ()
