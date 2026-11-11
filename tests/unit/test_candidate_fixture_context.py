"""A rule declares fixture inputs on its spec: the declaration, the hash and the state."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest
from odds_core.market_prices import Outcome
from odds_core.time import season_start
from odds_lambda.candidate_rules import (
    BetIntent,
    FixtureContext,
    FixtureInput,
    MarketState,
    PreviousResult,
    RuleSpec,
    SharpInput,
    apply_rule,
)
from odds_lambda.candidate_rules.pinning import rule_sha256
from odds_lambda.candidate_rules.registry import RULES

BOTH = frozenset({FixtureInput.PREVIOUS_RESULT, FixtureInput.CONCURRENT_KICKOFFS})
KICKOFF = datetime(2026, 4, 10, 15, 0, tzinfo=UTC)


class TestDeclaration:
    def test_the_registered_rules_declare_no_fixture_input(self) -> None:
        assert all(not spec.fixture_inputs for spec in RULES.values())

    def test_the_declaration_is_part_of_the_rule_hash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = rule_sha256("exchange_gap")
        monkeypatch.setitem(
            RULES, "exchange_gap", replace(RULES["exchange_gap"], fixture_inputs=BOTH)
        )
        declared = rule_sha256("exchange_gap")
        monkeypatch.setitem(
            RULES,
            "exchange_gap",
            replace(
                RULES["exchange_gap"],
                fixture_inputs=frozenset({FixtureInput.CONCURRENT_KICKOFFS}),
            ),
        )
        assert len({before, declared, rule_sha256("exchange_gap")}) == 3

    def test_a_state_of_a_rule_declaring_nothing_has_an_empty_context(self) -> None:
        state = MarketState(1, "soccer_epl", "1x2", KICKOFF, KICKOFF - timedelta(hours=1))
        assert state.fixture == FixtureContext()
        assert state.input_event_ids == [] and not state.missing_fixture
        assert dict(state.fixture.previous) == {}
        assert state.fixture.concurrent_kickoffs is None


class TestRuleReceivesContext:
    def test_a_rule_reads_the_context_off_its_state(self) -> None:
        seen: list[FixtureContext] = []

        def rule(state: MarketState, params: object) -> list[BetIntent]:
            seen.append(state.fixture)
            return [BetIntent(Outcome.HOME, "bet365", 2.0, 1.0)]

        context = FixtureContext(
            previous=MappingProxyType(
                {
                    Outcome.HOME: PreviousResult(2, KICKOFF - timedelta(days=5), 0, 3),
                    Outcome.AWAY: None,
                }
            ),
            concurrent_kickoffs=4,
            event_ids=(2, 5),
        )
        state = MarketState(
            1,
            "soccer_epl",
            "1x2",
            KICKOFF,
            KICKOFF - timedelta(hours=1),
            snapshot_id=7,
            snapshot_time=KICKOFF - timedelta(hours=1, minutes=2),
            fixture=context,
        )
        spec = RuleSpec(rule, object, frozenset({SharpInput.EXCHANGE}), fixture_inputs=BOTH)  # type: ignore[arg-type]

        _, intents = apply_rule(spec, object(), state)  # type: ignore[arg-type]

        assert len(intents) == 1 and seen == [context]
        assert state.input_event_ids == [2, 5]

    def test_a_state_with_unknown_results_has_no_inputs_and_no_bets(self) -> None:
        state = MarketState(
            1,
            "soccer_epl",
            "1x2",
            KICKOFF,
            KICKOFF - timedelta(hours=1),
            fixture=FixtureContext(overdue_results=(Outcome.HOME,)),
        )
        assert state.missing_fixture and not state.has_input
        assert apply_rule(RULES["exchange_gap"], object(), state)[1] == []  # type: ignore[arg-type]


class TestSeasonStart:
    @pytest.mark.parametrize(
        ("sport", "kickoff", "start"),
        [
            ("soccer_epl", datetime(2026, 4, 10, 15, tzinfo=UTC), datetime(2025, 8, 1, tzinfo=UTC)),
            ("soccer_epl", datetime(2026, 8, 1, 11, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC)),
            ("soccer_epl", datetime(2026, 7, 31, 23, tzinfo=UTC), datetime(2025, 8, 1, tzinfo=UTC)),
            (
                "baseball_mlb",
                datetime(2026, 4, 10, 15, tzinfo=UTC),
                datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ],
    )
    def test_the_season_a_kickoff_falls_in(
        self, sport: str, kickoff: datetime, start: datetime
    ) -> None:
        assert season_start(kickoff, sport) == start


class TestPreviousResult:
    @pytest.mark.parametrize(
        ("scored", "conceded", "difference", "result"),
        [(3, 0, 3, "win"), (0, 2, -2, "loss"), (1, 1, 0, "draw")],
    )
    def test_the_result_and_goal_difference(
        self, scored: int, conceded: int, difference: int, result: str
    ) -> None:
        previous = PreviousResult(2, KICKOFF, scored, conceded)
        assert (previous.goal_difference, previous.result.value) == (difference, result)
