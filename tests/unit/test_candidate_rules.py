"""Unit tests for the mechanical rule interface and the exchange-gap rule."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.betfair_close import ExchangePrice
from odds_core.candidates import MechanicalCandidate, candidates_dir, load_candidates
from odds_core.market_prices import Outcome
from odds_core.models import OddsSnapshot
from odds_core.snapshot_sources import FdukTimeBasis, SnapshotKind
from odds_lambda.candidate_rules import (
    BetIntent,
    ExchangeGapParams,
    MarketState,
    RuleSpec,
    apply_rule,
    build_market_state,
    decision_leads,
    decision_time,
    exchange_gap,
    max_input_age,
    rule_sha256,
    validate_candidate_rules,
)
from pydantic import ValidationError

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
EVENT = make_epl_event("evt", KICKOFF)
HOME, DRAW, AWAY = Outcome.HOME, Outcome.DRAW, Outcome.AWAY
MAX_AGE = timedelta(minutes=10)


def _betfair(prices: tuple[float, float, float], inplay: bool = False) -> dict[str, Any]:
    return three_way_bookmaker(
        "betfair_exchange", prices, betfair_meta={"inplay": inplay, "market_status": "OPEN"}
    )


def _snap(
    minutes_before_t: float, *books: dict[str, Any], source: str | None = "oddsportal_live"
) -> OddsSnapshot:
    return make_tagged_snapshot(
        EVENT.id, T - timedelta(minutes=minutes_before_t), list(books), source
    )


def _state(snapshots: list[OddsSnapshot]) -> MarketState:
    return build_market_state(EVENT, snapshots, T, MAX_AGE)


BET365 = three_way_bookmaker("bet365", (2.0, 3.0, 4.0))


class TestBuildMarketState:
    def test_uses_the_sports_market_and_event_fields(self) -> None:
        state = _state([])
        assert (state.event_id, state.sport_key, state.market) == ("evt", "soccer_epl", "1x2")
        assert (state.kickoff, state.decision_time) == (KICKOFF, T)
        assert not state.has_input
        assert state.retail == {} and state.exchange is None and state.snapshot_id is None

    def test_mlb_uses_h2h(self) -> None:
        event = make_epl_event("mlb", KICKOFF)
        event.sport_key = "baseball_mlb"
        assert build_market_state(event, [], T, MAX_AGE).market == "h2h"

    def test_retail_and_exchange_come_from_one_snapshot(self) -> None:
        snapshot = _snap(3, BET365, _betfair((2.0, 4.0, 4.0)))
        snapshot.id = 7
        state = _state([snapshot])
        assert state.has_input
        assert (state.snapshot_id, state.snapshot_time) == (7, snapshot.snapshot_time)
        assert state.retail == {"bet365": {HOME: 2.0, DRAW: 3.0, AWAY: 4.0}}
        assert state.exchange is not None
        assert state.exchange.kind is SnapshotKind.ODDSPORTAL_LIVE
        assert state.exchange.time == snapshot.snapshot_time
        assert sum(state.exchange.fair.values()) == pytest.approx(1.0)
        assert state.exchange.fair[HOME] == pytest.approx(0.5)

    def test_snapshots_after_t_are_ignored(self) -> None:
        assert not _state([_snap(-1, BET365, _betfair((2.0, 3.5, 4.0)))]).has_input

    def test_snapshot_at_t_counts(self) -> None:
        state = _state([_snap(0, BET365, _betfair((2.0, 4.0, 4.0)))])
        assert state.retail["bet365"][HOME] == 2.0

    def test_latest_snapshot_wins_whole(self) -> None:
        old = _snap(8, BET365, _betfair((2.0, 4.0, 4.0)))
        new = _snap(2, three_way_bookmaker("betway", (2.2, 3.2, 4.2)), _betfair((2.1, 4.0, 3.8)))
        state = _state([new, old])
        assert state.snapshot_time == new.snapshot_time
        assert set(state.retail) == {"betway"}

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
        assert _state([one_x_two, totals]).snapshot_time == one_x_two.snapshot_time

    def test_snapshot_at_the_max_age_counts(self) -> None:
        assert _state([_snap(10, BET365, _betfair((2.0, 4.0, 4.0)))]).has_input

    def test_stale_latest_snapshot_gives_no_input(self) -> None:
        state = _state([_snap(10.5, BET365, _betfair((2.0, 4.0, 4.0)))])
        assert not state.has_input
        assert state.retail == {} and state.exchange is None

    @pytest.mark.parametrize(
        ("source", "kind"),
        [
            (None, SnapshotKind.ODDS_API),
            ("oddsportal", SnapshotKind.ODDSPORTAL),
            ("oddsportal_live", SnapshotKind.ODDSPORTAL_LIVE),
        ],
    )
    def test_any_retail_kind_with_both_prices_is_an_input(
        self, source: str | None, kind: SnapshotKind
    ) -> None:
        state = _state([_snap(2, BET365, _betfair((2.0, 4.0, 4.0)), source=source)])
        assert state.has_input and state.exchange is not None and state.exchange.kind is kind

    def test_odds_api_exchange_alias_is_the_exchange(self) -> None:
        exchange = three_way_bookmaker("betfair_ex_uk", (2.0, 4.0, 4.0))
        state = _state([_snap(2, BET365, exchange, source=None)])
        assert state.exchange is not None and state.exchange.fair[HOME] == pytest.approx(0.5)

    def test_live_betfair_snapshot_is_not_an_input(self) -> None:
        assert not _state(
            [_snap(2, BET365, _betfair((2.0, 4.0, 4.0)), source="betfair_api")]
        ).has_input

    def test_fduk_opening_is_not_an_input(self) -> None:
        snapshot = make_tagged_snapshot(
            EVENT.id,
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
        assert state.snapshot_time == both.snapshot_time and set(state.retail) == {"bet365"}

    def test_snapshot_without_retail_is_skipped(self) -> None:
        assert not _state([_snap(2, _betfair((2.0, 4.0, 4.0)))]).has_input

    def test_non_retail_books_are_excluded(self) -> None:
        pinnacle = three_way_bookmaker("pinnacle", (2.0, 3.0, 4.0))
        state = _state([_snap(2, pinnacle, BET365, _betfair((2.0, 4.0, 4.0)))])
        assert set(state.retail) == {"bet365"}

    def test_incomplete_exchange_row_gives_no_exchange(self) -> None:
        partial = {
            "key": "betfair_exchange",
            "markets": [{"key": "1x2", "outcomes": [{"name": "Arsenal", "price_decimal": 2.0}]}],
        }
        state = _state([_snap(2, BET365, partial)])
        assert state.has_input and state.exchange is None

    def test_wide_exchange_gives_none_without_falling_back(self) -> None:
        tight = _snap(6, BET365, _betfair((2.0, 4.0, 4.0)))
        wide = _snap(2, BET365, _betfair((1.8, 3.5, 3.5)))
        state = _state([tight, wide])
        assert state.snapshot_time == wide.snapshot_time and state.exchange is None


def _exchange(home: float, draw: float, away: float) -> ExchangePrice:
    return ExchangePrice(
        time=T,
        kind=SnapshotKind.ODDSPORTAL_LIVE,
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
        retail=retail,
        exchange=exchange,
    )


PARAMS = ExchangeGapParams(threshold=0.02, minutes_before_kickoff=60, max_input_age_minutes=10)


class TestExchangeGap:
    def test_no_exchange_no_bets(self) -> None:
        assert exchange_gap(_rule_state({"bet365": {HOME: 9.0}}, None), PARAMS) == []

    def test_threshold_is_strict(self) -> None:
        # 0.5 * 2.5 - 1 = 0.25, clearly above; edge exactly at threshold is not a bet.
        state = _rule_state({"bet365": {HOME: 2.5}}, _exchange(0.5, 0.25, 0.25))
        at = ExchangeGapParams(threshold=0.25, minutes_before_kickoff=60, max_input_age_minutes=10)
        below = ExchangeGapParams(
            threshold=0.2499, minutes_before_kickoff=60, max_input_age_minutes=10
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

    @pytest.mark.parametrize(
        "field", ["threshold", "stake", "minutes_before_kickoff", "max_input_age_minutes"]
    )
    @pytest.mark.parametrize("value", [0, -1])
    def test_params_must_be_positive(self, field: str, value: float) -> None:
        raw: dict[str, Any] = {**PARAMS.model_dump(), field: value}
        with pytest.raises(ValidationError):
            ExchangeGapParams(**raw)

    def test_max_input_age_is_required(self) -> None:
        with pytest.raises(ValidationError):
            ExchangeGapParams(threshold=0.02, minutes_before_kickoff=60)  # type: ignore[call-arg]

    def test_params_reject_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExchangeGapParams(**PARAMS.model_dump(), surprise=1)


class TestApplyRule:
    def test_state_without_input_gives_no_bets_whatever_the_rule(self) -> None:
        always = RuleSpec(
            lambda state, params: [BetIntent(HOME, "bet365", 2.0, 1.0)], ExchangeGapParams
        )
        empty = MarketState("evt", "soccer_epl", "1x2", KICKOFF, T)
        assert apply_rule(always, PARAMS, empty)[1] == []
        assert apply_rule(always, PARAMS, _rule_state({}, None))[1] != []


class TestDecisionTime:
    def test_is_kickoff_minus_minutes(self) -> None:
        params = PARAMS.model_copy(update={"minutes_before_kickoff": 1440})
        assert decision_time(KICKOFF, params) == KICKOFF - timedelta(days=1)

    def test_max_input_age_is_in_minutes(self) -> None:
        assert max_input_age(PARAMS) == timedelta(minutes=10)


def _candidate(**overrides: Any) -> MechanicalCandidate:
    fields: dict[str, Any] = {
        "name": "gap",
        "description": "d",
        "kind": "mechanical",
        "sport": "soccer_epl",
        "version": 1,
        "issue": 626,
        "start_date": "2026-11-01",
        "status": "active",
        "rule": "exchange_gap",
        "rule_sha256": rule_sha256("exchange_gap"),
        "params": {"threshold": 0.02, "minutes_before_kickoff": 60, "max_input_age_minutes": 10},
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

    def test_sport_without_a_market_fails(self) -> None:
        with pytest.raises(ValueError, match="'gap'.*No candidate market defined"):
            validate_candidate_rules([_candidate(sport="basketball_nba")])

    def test_registered_candidates_are_valid(self) -> None:
        validate_candidate_rules(load_candidates(candidates_dir()))


class TestDecisionLeads:
    def test_active_mechanical_candidates_of_the_sport(self) -> None:
        candidates = [
            _candidate(),
            _candidate(name="later", params={**PARAMS.model_dump(), "minutes_before_kickoff": 30}),
            _candidate(name="mlb", sport="baseball_mlb"),
            _candidate(name="killed", status="killed"),
            _candidate(name="broken", params={"threshold": -1}),
        ]
        assert decision_leads(candidates, "soccer_epl") == [
            timedelta(minutes=60),
            timedelta(minutes=30),
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
            "odds_lambda.candidate_rules.market_state",
            "odds_core.as_of",
        ],
    )
    def test_covers_the_rule_its_params_the_state_and_the_as_of_module(
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

    def test_a_stale_hash_fails_validation_with_the_new_one(self) -> None:
        with pytest.raises(ValueError, match="'gap'.*code of rule 'exchange_gap' changed") as exc:
            validate_candidate_rules([_candidate(rule_sha256="0" * 64)])
        assert rule_sha256("exchange_gap") in str(exc.value)

    def test_hash_must_be_64_hex_chars(self) -> None:
        with pytest.raises(ValidationError):
            _candidate(rule_sha256="abc")
