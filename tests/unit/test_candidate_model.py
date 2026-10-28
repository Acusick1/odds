"""The model behind a model candidate rule: its training rows, inputs, schedule and frozen fit."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Any

import pytest
from odds_analytics.candidate_model import (
    FIXED_XGBOOST_PARAMS,
    SEARCH_SPACES,
    at_decision_times,
    choose_hyperparameters,
    refit_blocks,
    state_inputs,
    training_prefix,
    training_rows,
    tuning_folds,
)
from odds_analytics.retail_ev import RetailEvInput
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event
from odds_core.quotes import Quote
from odds_core.snapshot_sources import FdukTimeBasis, SnapshotKind
from odds_core.vocabulary import Source
from odds_lambda.candidate_rules import RetailEvModelParams, SharpInput, build_market_state

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_quotes,
    three_way_bookmaker,
)

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
EVENT = make_epl_event("evt", KICKOFF)
EXCHANGE_INPUT = frozenset({SharpInput.EXCHANGE})
EXCHANGE = (2.0, 4.0, 4.0)


_ids = count(1)


def _betfair(prices: tuple[float, float, float] = EXCHANGE) -> dict[str, Any]:
    return betfair_exchange_bookmaker(prices)


def _snap(before_kickoff: timedelta, *books: dict[str, Any], source: str) -> list[Quote]:
    return make_quotes(EVENT, next(_ids), KICKOFF - before_kickoff, list(books), source)


def _time(snapshot: list[Quote]) -> datetime:
    return snapshot[0].snapshot_time


CO_TIMED = _snap(
    timedelta(hours=3),
    three_way_bookmaker("bet365", (2.2, 3.4, 3.8)),
    three_way_bookmaker("betway", (2.1, 3.5, 3.9)),
    _betfair(),
    source="oddsportal_live",
)
CLOSE = _snap(timedelta(minutes=2), _betfair((1.9, 4.2, 4.4)), source="betfair_api")


class TestTrainingRows:
    def test_one_row_per_book_and_outcome_of_each_co_timed_snapshot(self) -> None:
        rows = training_rows(EVENT, "1x2", CO_TIMED + CLOSE)
        assert {(r.book, r.outcome) for r in rows} == {
            (book, outcome) for book in ("bet365", "betway") for outcome in market_outcomes("1x2")
        }
        assert {r.decision_time for r in rows} == {_time(CO_TIMED)}
        assert all(r.retail_kind is SnapshotKind.ODDSPORTAL_LIVE for r in rows)
        assert all(r.exchange.fair[Outcome.HOME] == pytest.approx(0.5) for r in rows)
        assert all(r.close.source is Source.BETFAIR for r in rows)

    def test_the_exchange_price_comes_from_the_rows_own_snapshot(self) -> None:
        live_betfair = _snap(
            timedelta(hours=3, minutes=1), _betfair((1.5, 5.0, 8.0)), source="betfair_api"
        )
        rows = training_rows(EVENT, "1x2", live_betfair + CO_TIMED + CLOSE)
        assert all(r.exchange.time == _time(CO_TIMED) for r in rows)

    @pytest.mark.parametrize(
        "snapshot",
        [
            _snap(
                timedelta(hours=1),
                three_way_bookmaker("bet365", (2.2, 3.4, 3.8)),
                source="oddsportal_live",
            ),
            _snap(timedelta(hours=1), _betfair(), source="betfair_api"),
            _snap(
                timedelta(minutes=4),
                three_way_bookmaker("bet365", (2.2, 3.4, 3.8)),
                _betfair(),
                source="oddsportal_live",
            ),
            _snap(
                timedelta(days=7, minutes=1),
                three_way_bookmaker("bet365", (2.2, 3.4, 3.8)),
                _betfair(),
                source="oddsportal_live",
            ),
        ],
        ids=["no_exchange_row", "no_retail", "inside_the_cutoff", "outside_the_window"],
    )
    def test_snapshots_that_are_not_a_co_timed_decision_give_no_rows(
        self, snapshot: list[Quote]
    ) -> None:
        assert training_rows(EVENT, "1x2", snapshot + CLOSE) == []

    def test_an_event_without_a_close_gives_no_rows(self) -> None:
        assert training_rows(EVENT, "1x2", CO_TIMED) == []

    def test_an_fduk_close_is_not_a_label(self) -> None:
        # Published days after kick-off, so a refit at the next matchday would not have it.
        fduk_close = make_quotes(
            EVENT,
            next(_ids),
            KICKOFF,
            [_betfair((1.9, 4.2, 4.4))],
            "football_data_uk",
            FdukTimeBasis.KICKOFF,
        )
        assert training_rows(EVENT, "1x2", CO_TIMED + fduk_close) == []
        assert training_rows(EVENT, "1x2", CO_TIMED + fduk_close + CLOSE) != []


class TestStateInputs:
    def test_a_decision_sees_the_inputs_its_snapshot_trains_on(self) -> None:
        t = _time(CO_TIMED) + timedelta(minutes=8)
        state = build_market_state(
            EVENT, "1x2", CO_TIMED + CLOSE, t, timedelta(minutes=10), EXCHANGE_INPUT
        )
        rows = training_rows(EVENT, "1x2", CO_TIMED + CLOSE)
        assert state_inputs(state) == [
            RetailEvInput(**{f.name: getattr(r, f.name) for f in fields(RetailEvInput)})
            for r in rows
        ]

    def test_a_state_without_an_exchange_price_has_no_inputs(self) -> None:
        wide = _snap(
            timedelta(hours=3),
            three_way_bookmaker("bet365", (2.2, 3.4, 3.8)),
            _betfair((1.5, 2.5, 3.0)),
            source="oddsportal_live",
        )
        state = build_market_state(
            EVENT, "1x2", wide, _time(wide), timedelta(minutes=10), EXCHANGE_INPUT
        )
        assert state.has_input and state.exchange == {}
        assert state_inputs(state) == []


def _events(n: int) -> list[Event]:
    return [make_epl_event(f"e{i}", KICKOFF + timedelta(hours=i)) for i in range(n)]


def _params(**overrides: Any) -> RetailEvModelParams:
    fields: dict[str, Any] = {
        "model": "retail_ev_price_only",
        "threshold": 0.01,
        "training_matches": 3,
        "refit_every_matches": 2,
        "tuning_trials": 3,
        "tuning_seed": 7,
        "tuning_min_train_matches": 1,
        "tuning_step_matches": 1,
        "minutes_before_kickoff": (90, 30),
        "max_input_age_minutes": 10,
        **overrides,
    }
    return RetailEvModelParams.model_validate(fields)


class TestRefitBlocks:
    def test_scored_matches_follow_the_training_prefix_in_blocks(self) -> None:
        events = _events(8)
        blocks = refit_blocks(events, _params())
        assert [[e.id for e in b.events] for b in blocks] == [["e3", "e4"], ["e5", "e6"], ["e7"]]
        assert [b.index for b in blocks] == [0, 1, 2]

    def test_a_block_refits_at_its_first_decision_time(self) -> None:
        blocks = refit_blocks(_events(5), _params())
        assert blocks[0].at == KICKOFF + timedelta(hours=3) - timedelta(minutes=90)

    def test_a_range_no_longer_than_the_prefix_scores_nothing(self) -> None:
        assert refit_blocks(_events(3), _params()) == []


def _weekly(n: int) -> list[Event]:
    return [make_epl_event(f"w{i}", KICKOFF + timedelta(days=8 * i)) for i in range(n)]


class TestTuningFolds:
    def test_expanding_folds_validate_the_training_matches_in_steps(self) -> None:
        events = _weekly(6)
        folds = tuning_folds(events, _params(tuning_min_train_matches=2, tuning_step_matches=2))
        assert [([e.id for e in f.train], [e.id for e in f.validate]) for f in folds] == [
            (["w0", "w1"], ["w2", "w3"]),
            (["w0", "w1", "w2", "w3"], ["w4", "w5"]),
        ]

    def test_training_matches_end_a_decision_window_before_validation(self) -> None:
        events = [*_events(2), make_epl_event("late", KICKOFF + timedelta(days=9))]
        [first, second] = tuning_folds(events, _params(training_matches=3))
        assert first.train == ()
        assert [e.id for e in second.train] == ["e0", "e1"]


class TestTune:
    def _folds(self) -> list[tuple[list[Any], list[Any]]]:
        rows = []
        for i, home in enumerate((2.2, 2.0, 1.9)):
            event = make_epl_event(f"t{i}", KICKOFF + timedelta(days=8 * i))
            snap = make_quotes(
                event,
                next(_ids),
                event.commence_time - timedelta(hours=3),
                [three_way_bookmaker("bet365", (home, 3.4, 3.8)), _betfair()],
                "oddsportal_live",
            )
            close = make_quotes(
                event,
                next(_ids),
                event.commence_time - timedelta(minutes=2),
                [_betfair((1.9, 4.2, 4.4))],
                "betfair_api",
            )
            rows.append(training_rows(event, "1x2", snap + close))
        return [(rows[0], rows[1]), ([], rows[2]), (rows[0] + rows[1], rows[2])]

    def test_chooses_from_the_search_space_reproducibly(self) -> None:
        params = _params()
        first = choose_hyperparameters("1x2", self._folds(), params)
        assert first == choose_hyperparameters("1x2", self._folds(), params)
        assert set(first.params) == set(SEARCH_SPACES)
        assert not set(first.params) & set(FIXED_XGBOOST_PARAMS)
        assert first.trials == 3
        # The fold without training rows is skipped.
        assert first.folds == 2

    def test_without_a_usable_fold_it_refuses(self) -> None:
        with pytest.raises(ValueError, match="no tuning fold"):
            choose_hyperparameters("1x2", [([], [])], _params())


class TestTrainingPrefix:
    def test_is_the_first_training_matches(self) -> None:
        assert training_prefix(_events(5), _params()) == 3

    def test_takes_matches_tied_with_the_last_one(self) -> None:
        events = [
            *_events(3),
            make_epl_event("tie", KICKOFF + timedelta(hours=2)),
            make_epl_event("later", KICKOFF + timedelta(hours=5)),
        ]
        assert training_prefix(events, _params()) == 4

    def test_is_at_most_the_range(self) -> None:
        assert training_prefix(_events(2), _params()) == 2


class TestAtDecisionTimes:
    @pytest.mark.parametrize(
        ("before_kickoff", "usable"),
        [
            (timedelta(minutes=90), True),
            (timedelta(minutes=100), True),
            (timedelta(minutes=101), False),
            (timedelta(minutes=35), True),
            (timedelta(minutes=29), False),
            (timedelta(hours=3), False),
        ],
    )
    def test_rows_within_the_max_input_age_before_a_decision_time(
        self, before_kickoff: timedelta, usable: bool
    ) -> None:
        snapshot = _snap(
            before_kickoff,
            three_way_bookmaker("bet365", (2.2, 3.4, 3.8)),
            _betfair(),
            source="oddsportal_live",
        )
        [row, *_] = training_rows(EVENT, "1x2", snapshot + CLOSE)
        assert at_decision_times(row, _params()) is usable
