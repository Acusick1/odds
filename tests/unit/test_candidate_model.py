"""The model behind a model candidate rule: its training rows, inputs, schedule and frozen fit."""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_analytics.candidate_model import (
    MODEL_XGBOOST_PARAMS,
    refit_blocks,
    state_inputs,
    training_rows,
)
from odds_analytics.retail_ev import RetailEvInput
from odds_analytics.training.config import XGBoostConfig
from odds_core.config import get_settings
from odds_core.market_prices import Outcome
from odds_core.models import Event, OddsSnapshot
from odds_core.snapshot_sources import SnapshotKind
from odds_lambda.candidate_rules import RetailEvModelParams, build_market_state

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
EVENT = make_epl_event("evt", KICKOFF)
EXCHANGE = (2.0, 4.0, 4.0)


def _betfair(prices: tuple[float, float, float] = EXCHANGE) -> dict[str, Any]:
    return three_way_bookmaker(
        "betfair_exchange", prices, betfair_meta={"inplay": False, "market_status": "OPEN"}
    )


def _snap(before_kickoff: timedelta, *books: dict[str, Any], source: str) -> OddsSnapshot:
    return make_tagged_snapshot(EVENT.id, KICKOFF - before_kickoff, list(books), source)


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
        rows = training_rows(EVENT, [CO_TIMED, CLOSE])
        assert {(r.book, r.outcome) for r in rows} == {
            (book, outcome) for book in ("bet365", "betway") for outcome in Outcome
        }
        assert {r.decision_time for r in rows} == {CO_TIMED.snapshot_time}
        assert all(r.retail_kind is SnapshotKind.ODDSPORTAL_LIVE for r in rows)
        assert all(r.exchange.fair[Outcome.HOME] == pytest.approx(0.5) for r in rows)
        assert all(r.close.kind is SnapshotKind.BETFAIR_LIVE for r in rows)

    def test_the_exchange_price_comes_from_the_rows_own_snapshot(self) -> None:
        live_betfair = _snap(
            timedelta(hours=3, minutes=1), _betfair((1.5, 5.0, 8.0)), source="betfair_api"
        )
        rows = training_rows(EVENT, [live_betfair, CO_TIMED, CLOSE])
        assert all(r.exchange.time == CO_TIMED.snapshot_time for r in rows)

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
        self, snapshot: OddsSnapshot
    ) -> None:
        assert training_rows(EVENT, [snapshot, CLOSE]) == []

    def test_an_event_without_a_close_gives_no_rows(self) -> None:
        assert training_rows(EVENT, [CO_TIMED]) == []


class TestStateInputs:
    def test_a_decision_sees_the_inputs_its_snapshot_trains_on(self) -> None:
        t = CO_TIMED.snapshot_time + timedelta(minutes=8)
        state = build_market_state(EVENT, [CO_TIMED, CLOSE], t, timedelta(minutes=10))
        rows = training_rows(EVENT, [CO_TIMED, CLOSE])
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
        state = build_market_state(EVENT, [wide], wide.snapshot_time, timedelta(minutes=10))
        assert state.has_input and state.exchange is None
        assert state_inputs(state) == []


def _events(n: int) -> list[Event]:
    return [make_epl_event(f"e{i}", KICKOFF + timedelta(hours=i)) for i in range(n)]


def _params(**overrides: Any) -> RetailEvModelParams:
    fields: dict[str, Any] = {
        "model": "retail_ev_price_only",
        "threshold": 0.01,
        "training_matches": 3,
        "refit_every_matches": 2,
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


class TestModel:
    def test_is_the_frozen_price_only_residual_trial(self) -> None:
        summary = json.loads(
            (
                get_settings().project_root
                / "experiments/results/retail_ev_as_of_t_groups/summary.json"
            ).read_text()
        )
        trial = next(
            run["best_params"]
            for run in summary["variants"]["price_only"]["runs"]
            if run["run"] == "residual_s42"
        )
        assert MODEL_XGBOOST_PARAMS["retail_ev_price_only"] == (
            XGBoostConfig(**trial, n_jobs=1, random_state=42).model_dump()
        )
