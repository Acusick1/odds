"""Unit tests for the mechanical rule interface and the exchange-gap rule."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.betfair_close import ExchangePrice
from odds_core.candidates import MechanicalCandidate, candidates_dir, load_candidates
from odds_core.market_prices import Outcome
from odds_core.models import OddsSnapshot
from odds_core.snapshot_sources import FdukTimeBasis, SnapshotKind
from odds_lambda.candidate_rules import (
    MAX_RETAIL_AGE,
    RULE_SOURCE_HASHES,
    RULES,
    BetIntent,
    ExchangeGapParams,
    MarketState,
    RetailQuote,
    build_market_state,
    changed_pinned_rules,
    decision_time,
    exchange_gap,
    validate_candidate_rules,
)
from pydantic import ValidationError

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
EVENT = make_epl_event("evt", KICKOFF)
HOME, DRAW, AWAY = Outcome.HOME, Outcome.DRAW, Outcome.AWAY


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
    return build_market_state(EVENT, snapshots, T)


class TestBuildMarketState:
    def test_uses_the_sports_market_and_event_fields(self) -> None:
        state = _state([])
        assert (state.event_id, state.sport_key, state.market) == ("evt", "soccer_epl", "1x2")
        assert (state.kickoff, state.decision_time) == (KICKOFF, T)
        assert state.retail == {} and state.exchange is None

    def test_mlb_uses_h2h(self) -> None:
        event = make_epl_event("mlb", KICKOFF)
        event.sport_key = "baseball_mlb"
        assert build_market_state(event, [], T).market == "h2h"

    def test_snapshots_after_t_are_ignored(self) -> None:
        late = _snap(-1, three_way_bookmaker("bet365", (2.0, 3.0, 4.0)))
        betfair = _snap(-1, _betfair((2.0, 3.5, 4.0)), source="betfair_api")
        state = _state([late, betfair])
        assert state.retail == {} and state.exchange is None

    def test_snapshot_at_t_counts(self) -> None:
        state = _state([_snap(0, three_way_bookmaker("bet365", (2.0, 3.0, 4.0)))])
        assert state.retail["bet365"].prices == {HOME: 2.0, DRAW: 3.0, AWAY: 4.0}

    def test_latest_quote_per_book_wins(self) -> None:
        old = _snap(30, three_way_bookmaker("bet365", (2.0, 3.0, 4.0)))
        new = _snap(10, three_way_bookmaker("bet365", (2.1, 3.1, 4.1)))
        state = _state([new, old])
        assert state.retail["bet365"] == RetailQuote(
            T - timedelta(minutes=10), {HOME: 2.1, DRAW: 3.1, AWAY: 4.1}
        )

    def test_book_missing_from_latest_snapshot_keeps_earlier_quote(self) -> None:
        old = _snap(30, three_way_bookmaker("bet365", (2.0, 3.0, 4.0)))
        new = _snap(10, three_way_bookmaker("betway", (2.2, 3.2, 4.2)))
        assert set(_state([old, new]).retail) == {"bet365", "betway"}

    def test_stale_retail_quote_is_dropped(self) -> None:
        minutes = MAX_RETAIL_AGE.total_seconds() / 60
        edge = _snap(minutes, three_way_bookmaker("bet365", (2.0, 3.0, 4.0)))
        stale = _snap(minutes + 1, three_way_bookmaker("betway", (2.0, 3.0, 4.0)))
        assert set(_state([edge, stale]).retail) == {"bet365"}

    def test_non_retail_books_and_kinds_are_excluded(self) -> None:
        pinnacle = _snap(5, three_way_bookmaker("pinnacle", (2.0, 3.0, 4.0)))
        exchange_only = _snap(5, _betfair((2.0, 3.5, 4.0)), source="betfair_api")
        assert _state([pinnacle, exchange_only]).retail == {}

    def test_live_betfair_exchange_is_devigged(self) -> None:
        state = _state([_snap(10, _betfair((2.0, 4.0, 4.0)), source="betfair_api")])
        assert state.exchange is not None
        assert state.exchange.kind is SnapshotKind.BETFAIR_LIVE
        assert state.exchange.time == T - timedelta(minutes=10)
        assert sum(state.exchange.fair.values()) == pytest.approx(1.0)
        assert state.exchange.fair[HOME] == pytest.approx(0.5)

    def test_in_play_exchange_is_skipped_for_the_earlier_price(self) -> None:
        pre = _snap(30, _betfair((2.0, 4.0, 4.0)), source="betfair_api")
        live = _snap(5, _betfair((1.5, 5.0, 8.0), inplay=True), source="betfair_api")
        state = _state([pre, live])
        assert state.exchange is not None and state.exchange.time == pre.snapshot_time

    def test_wide_exchange_is_skipped_for_the_earlier_price(self) -> None:
        tight = _snap(30, _betfair((2.0, 4.0, 4.0)), source="betfair_api")
        wide = _snap(5, _betfair((1.8, 3.5, 3.5)), source="betfair_api")
        state = _state([tight, wide])
        assert state.exchange is not None and state.exchange.time == tight.snapshot_time

    def test_only_wide_exchange_gives_none(self) -> None:
        assert _state([_snap(5, _betfair((1.8, 3.5, 3.5)), source="betfair_api")]).exchange is None

    def test_stale_exchange_is_dropped_without_falling_back(self) -> None:
        old = _snap(121, _betfair((2.0, 4.0, 4.0)), source="betfair_api")
        assert _state([old]).exchange is None

    def test_fduk_opening_is_usable_as_exchange_and_retail(self) -> None:
        snapshot = make_tagged_snapshot(
            EVENT.id,
            T - timedelta(minutes=20),
            [_betfair((2.0, 4.0, 4.0)), three_way_bookmaker("bet365", (2.1, 3.9, 4.2))],
            "football_data_uk",
            FdukTimeBasis.COLLECTION_DEADLINE,
        )
        state = _state([snapshot])
        assert state.exchange is not None and state.exchange.kind is SnapshotKind.FDUK_OPENING
        assert "bet365" in state.retail

    def test_fduk_closing_is_not_known_at_t(self) -> None:
        snapshot = make_tagged_snapshot(
            EVENT.id,
            T - timedelta(minutes=5),
            [_betfair((2.0, 4.0, 4.0)), three_way_bookmaker("bet365", (2.1, 3.9, 4.2))],
            "football_data_uk",
            FdukTimeBasis.KICKOFF,
        )
        state = _state([snapshot])
        assert state.exchange is None and state.retail == {}


def _exchange(home: float, draw: float, away: float) -> ExchangePrice:
    return ExchangePrice(
        time=T,
        kind=SnapshotKind.BETFAIR_LIVE,
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
        retail={book: RetailQuote(T, prices) for book, prices in retail.items()},
        exchange=exchange,
    )


PARAMS = ExchangeGapParams(threshold=0.02, minutes_before_kickoff=60)


class TestExchangeGap:
    def test_no_exchange_no_bets(self) -> None:
        assert exchange_gap(_rule_state({"bet365": {HOME: 9.0}}, None), PARAMS) == []

    def test_threshold_is_strict(self) -> None:
        # 0.5 * 2.5 - 1 = 0.25, clearly above; edge exactly at threshold is not a bet.
        state = _rule_state({"bet365": {HOME: 2.5}}, _exchange(0.5, 0.25, 0.25))
        at = ExchangeGapParams(threshold=0.25, minutes_before_kickoff=60)
        below = ExchangeGapParams(threshold=0.2499, minutes_before_kickoff=60)
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
        params = ExchangeGapParams(threshold=0.02, minutes_before_kickoff=60, stake=2.5)
        assert exchange_gap(state, params)[0].stake == 2.5

    def test_no_retail_no_bets(self) -> None:
        assert exchange_gap(_rule_state({}, _exchange(0.5, 0.25, 0.25)), PARAMS) == []

    @pytest.mark.parametrize("field", ["threshold", "stake", "minutes_before_kickoff"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_params_must_be_positive(self, field: str, value: float) -> None:
        raw: dict[str, Any] = {"threshold": 0.02, "minutes_before_kickoff": 60, field: value}
        with pytest.raises(ValidationError):
            ExchangeGapParams(**raw)

    def test_params_reject_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExchangeGapParams(threshold=0.02, minutes_before_kickoff=60, surprise=1)  # type: ignore[call-arg]


class TestDecisionTime:
    def test_is_kickoff_minus_minutes(self) -> None:
        params = ExchangeGapParams(threshold=0.02, minutes_before_kickoff=1440)
        assert decision_time(KICKOFF, params) == KICKOFF - timedelta(days=1)


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
        "params": {"threshold": 0.02, "minutes_before_kickoff": 60},
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

    def test_registered_candidates_are_valid(self) -> None:
        validate_candidate_rules(load_candidates(candidates_dir()))


class TestRulePinning:
    def test_every_rule_is_pinned(self) -> None:
        assert set(RULES) == set(RULE_SOURCE_HASHES)

    def test_referenced_rules_are_unchanged(self) -> None:
        referenced = [
            c.rule for c in load_candidates(candidates_dir()) if isinstance(c, MechanicalCandidate)
        ]
        assert changed_pinned_rules(referenced) == [], (
            "a rule a candidate references changed; a changed rule needs a new rule name"
        )

    def test_a_source_change_is_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert changed_pinned_rules(["exchange_gap"]) == []
        monkeypatch.setitem(RULE_SOURCE_HASHES, "exchange_gap", "0" * 64)
        assert changed_pinned_rules(["exchange_gap"]) == ["exchange_gap"]
