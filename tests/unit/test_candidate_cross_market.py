"""A rule that declares an extra market reads its state beside the candidate's own market."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest
from odds_core.candidates import MechanicalCandidate
from odds_core.market_prices import Outcome
from odds_core.quotes import Quote
from odds_core.vocabulary import Market
from odds_lambda.candidate_rules import (
    MarketState,
    RuleSpec,
    SharpInput,
    apply_rule,
    build_market_state,
)
from odds_lambda.candidate_rules.gap import exchange_gap
from odds_lambda.candidate_rules.pinning import rule_sha256
from odds_lambda.candidate_rules.registry import RULES, resolve_rule

from tests.test_helpers import make_epl_event
from tests.unit.conftest import CROSS_KICKOFF, CROSS_T, RunQuotes

EVENT = make_epl_event(1, CROSS_KICKOFF)
MAX_AGE = timedelta(minutes=10)
EXCHANGE = frozenset({SharpInput.EXCHANGE})
TOTALS = frozenset({Market.TOTALS})


def _state(
    quotes: list[Quote],
    also: frozenset[Market] = TOTALS,
    sharp: frozenset[SharpInput] = EXCHANGE,
) -> MarketState:
    return build_market_state(EVENT, "1x2", quotes, CROSS_T, MAX_AGE, sharp, also)


class TestDeclaredMarketIsRead:
    def test_a_rule_declaring_totals_receives_both_states(
        self, cross_market_run: RunQuotes
    ) -> None:
        state = _state(cross_market_run(7, 100))
        assert state.has_input and state.market == "1x2"
        totals = state.also["totals"]
        assert totals.has_input and totals.market == "totals" and totals.snapshot_id == 101
        assert set(totals.retail) == {2.5} and set(totals.exchange) == {2.5}
        assert totals.also == {}
        assert state.input_snapshot_ids == [100, 101]

    def test_a_rule_declaring_nothing_has_no_extra_state(self, cross_market_run: RunQuotes) -> None:
        state = _state(cross_market_run(7, 100), also=frozenset())
        assert state.has_input and state.also == {} and state.input_snapshot_ids == [100]

    def test_the_extra_market_is_not_read_without_the_declaration(
        self, cross_market_run: RunQuotes
    ) -> None:
        quotes = cross_market_run(7, 100, one_x_two=2.0, totals=None)
        assert _state(quotes, also=frozenset()).has_input


class TestStateIsReadOnly:
    def test_a_rule_cannot_erase_the_recorded_inputs(self, cross_market_run: RunQuotes) -> None:
        state = _state(cross_market_run(7, 100))
        with pytest.raises(AttributeError):
            state.also.clear()  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            state.also[Market.H2H] = state  # type: ignore[index]
        assert state.input_snapshot_ids == [100, 101]

    def test_the_recorded_inputs_are_fixed_when_the_state_is_built(
        self, cross_market_run: RunQuotes
    ) -> None:
        state = _state(cross_market_run(7, 100))
        with pytest.raises(AttributeError):
            state.input_ids = ()  # type: ignore[misc]
        assert state.input_snapshot_ids == [100, 101]

    @pytest.mark.parametrize("extra", [False, True])
    def test_nested_prices_cannot_be_mutated(
        self, cross_market_run: RunQuotes, extra: bool
    ) -> None:
        primary = _state(cross_market_run(7, 100))
        state = primary.also[Market.TOTALS] if extra else primary
        line = next(iter(state.retail))
        with pytest.raises(TypeError):
            state.retail[line] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            state.retail[line]["bet365"] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            state.retail[line]["bet365"][Outcome.HOME] = 9.0  # type: ignore[index]
        with pytest.raises(TypeError):
            state.exchange[line] = state.exchange[line]  # type: ignore[index]
        with pytest.raises(TypeError):
            state.exchange[line].fair[Outcome.HOME] = 0.9  # type: ignore[index]


class TestMissingOrStaleExtraMarket:
    def test_a_missing_extra_market_gives_no_input(self, cross_market_run: RunQuotes) -> None:
        state = _state(cross_market_run(7, 100, totals=None))
        assert not state.has_input and state.also == {} and state.input_snapshot_ids == []

    def test_a_stale_extra_market_gives_no_input(self, cross_market_run: RunQuotes) -> None:
        state = _state(cross_market_run(7, 100, totals=10.5))
        assert not state.has_input

    def test_an_extra_market_at_the_max_age_counts(self, cross_market_run: RunQuotes) -> None:
        assert _state(cross_market_run(7, 100, totals=10)).has_input

    def test_a_stale_primary_gives_no_input_though_the_extra_is_fresh(
        self, cross_market_run: RunQuotes
    ) -> None:
        assert not _state(cross_market_run(7, 100, one_x_two=11)).has_input

    def test_apply_rule_bets_nothing_without_the_extra_input(
        self, cross_market_run: RunQuotes
    ) -> None:
        spec = RuleSpec(exchange_gap, RULES["exchange_gap"].params_model, EXCHANGE, TOTALS)
        params = spec.params_model(
            threshold=0.01, minutes_before_kickoff=[60], max_input_age_minutes=10
        )
        state = _state(cross_market_run(7, 100, totals=None))
        assert apply_rule(spec, params, state)[1] == []

    def test_an_extra_market_without_an_exchange_row_gives_no_input_for_an_exchange_rule(
        self, cross_market_run: RunQuotes
    ) -> None:
        quotes = [
            q
            for q in cross_market_run(7, 100)
            if not (q.market == "totals" and q.bookmaker == "betfair_exchange")
        ]
        assert not _state(quotes).has_input
        assert _state(quotes, sharp=frozenset()).has_input


class TestSameSourceRun:
    def test_the_extra_market_comes_from_the_primarys_run(
        self, cross_market_run: RunQuotes
    ) -> None:
        # Run 8 is the later one, but its totals snapshot is not in the same run as the 1x2.
        run_7 = cross_market_run(7, 100, one_x_two=3, totals=3)
        run_8 = cross_market_run(8, 200, one_x_two=1, totals=None)
        run_9 = cross_market_run(9, 300, one_x_two=None, totals=1)
        state = _state(run_7 + run_8 + run_9)
        assert state.snapshot_id == 100 and state.also["totals"].snapshot_id == 101

    def test_a_primary_without_a_paired_extra_is_skipped_for_an_earlier_run(
        self, cross_market_run: RunQuotes
    ) -> None:
        paired = cross_market_run(7, 100, one_x_two=5, totals=5)
        unpaired = cross_market_run(8, 200, one_x_two=1, totals=None)
        assert _state(paired + unpaired).snapshot_id == 100

    def test_a_totals_snapshot_of_another_run_is_not_paired(
        self, cross_market_run: RunQuotes
    ) -> None:
        quotes = cross_market_run(7, 100, totals=None) + cross_market_run(8, 200, one_x_two=None)
        assert not _state(quotes).has_input

    def test_without_a_payload_the_latest_snapshot_of_the_source_is_paired(
        self, cross_market_run: RunQuotes
    ) -> None:
        state = _state(cross_market_run(None, 100, one_x_two=3, totals=1))
        assert state.has_input and state.also["totals"].snapshot_id == 101

    def test_a_snapshot_of_another_source_is_not_paired(self, cross_market_run: RunQuotes) -> None:
        quotes = cross_market_run(None, 100, totals=None) + cross_market_run(
            None, 200, one_x_two=None, source=None
        )
        assert not _state(quotes).has_input


class TestDeclaration:
    def _candidate(self, market: str, rule: str = "exchange_gap") -> MechanicalCandidate:
        return MechanicalCandidate.model_validate(
            {
                "name": "c",
                "description": "d",
                "kind": "mechanical",
                "market": market,
                "sports": ["soccer_epl"],
                "version": 1,
                "issue": 1,
                "start_date": "2026-01-01",
                "status": "active",
                "rule": rule,
                "params": {
                    "threshold": 0.01,
                    "minutes_before_kickoff": [60],
                    "max_input_age_minutes": 10,
                },
            }
        )

    def test_the_declaration_is_part_of_the_rule_hash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from odds_lambda.candidate_rules import registry

        before = rule_sha256("exchange_gap")
        monkeypatch.setitem(
            registry.RULES,
            "exchange_gap",
            replace(registry.RULES["exchange_gap"], also_markets=TOTALS),
        )
        assert rule_sha256("exchange_gap") != before

    def test_a_rule_cannot_also_read_the_candidates_own_market(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from odds_lambda.candidate_rules import registry

        monkeypatch.setitem(
            registry.RULES,
            "exchange_gap",
            replace(registry.RULES["exchange_gap"], also_markets=TOTALS),
        )
        resolve_rule(self._candidate("1x2"))
        with pytest.raises(ValueError, match="also reads"):
            resolve_rule(self._candidate("totals"))

    def test_a_rule_cannot_also_read_a_market_the_sport_does_not_decide(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from odds_lambda.candidate_rules import registry

        monkeypatch.setitem(
            registry.RULES,
            "exchange_gap",
            replace(registry.RULES["exchange_gap"], also_markets=frozenset({Market.H2H})),
        )
        with pytest.raises(ValueError, match="also reads"):
            resolve_rule(self._candidate("1x2"))

    def test_the_registered_rules_declare_nothing(self) -> None:
        assert all(not spec.also_markets for spec in RULES.values())
