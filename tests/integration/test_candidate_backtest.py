"""The replay backtest: as-of-t state, final kick-off, twins, coverage and shared scoring."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from odds_analytics.candidate_backtest import HashStatus, replay
from odds_analytics.candidate_report import candidate_report, load_decision_starts
from odds_core.candidates import MechanicalCandidate
from odds_core.event_source import canonical_event_id
from odds_core.models import EventStatus
from odds_core.paper_trade_models import CandidateDecision, PaperTrade, TradeResult
from odds_core.snapshot_sources import FdukTimeBasis
from odds_lambda.candidate_rules import rule_sha256

from tests.test_helpers import (
    API_EVENT_ID,
    TWIN_EVENT_ID,
    TWIN_KICKOFF,
    make_epl_event,
    make_tagged_snapshot,
    three_way_bookmaker,
)

pytestmark = pytest.mark.integration

N_BOOT = 200
NOW = datetime(2026, 9, 1, tzinfo=UTC)
KICKOFF = datetime(2026, 4, 10, 15, 0, tzinfo=UTC)
# 60 minutes before kick-off.
T = KICKOFF - timedelta(minutes=60)
# Exchange fair (0.5, 0.25, 0.25) against a retail home price of 2.2: EV +10%.
EXCHANGE = (2.0, 4.0, 4.0)
RETAIL = (2.2, 3.0, 3.0)


def candidate(**overrides: Any) -> MechanicalCandidate:
    fields: dict[str, Any] = {
        "name": "gap",
        "description": "d",
        "kind": "mechanical",
        "sport": "soccer_epl",
        "version": 1,
        "issue": 626,
        "start_date": "2026-01-01",
        "status": "active",
        "rule": "exchange_gap",
        "rule_sha256": rule_sha256("exchange_gap"),
        "params": {"threshold": 0.02, "minutes_before_kickoff": [60], "max_input_age_minutes": 10},
        **overrides,
    }
    return MechanicalCandidate.model_validate(fields)


def betfair(prices: tuple[float, float, float], **teams: str) -> dict[str, Any]:
    return three_way_bookmaker(
        "betfair_exchange",
        prices,
        betfair_meta={"inplay": False, "market_status": "OPEN"},
        **teams,
    )


async def add_event(
    session: Any,
    event_id: str,
    kickoff: datetime = KICKOFF,
    *,
    retail: tuple[float, float, float] | None = RETAIL,
    exchange: tuple[float, float, float] | None = EXCHANGE,
    close: tuple[float, float, float] | None = EXCHANGE,
    snapshot_offset: timedelta = timedelta(0),
    scores: tuple[int, int] | None = (2, 1),
    **fields: Any,
) -> None:
    """An event with an OddsPortal snapshot of its retail and exchange prices 5 minutes before
    the decision, and a Betfair close."""
    t = kickoff - timedelta(minutes=60)
    event = make_epl_event(
        event_id,
        kickoff,
        status=EventStatus.FINAL if scores else EventStatus.SCHEDULED,
        home_score=scores[0] if scores else None,
        away_score=scores[1] if scores else None,
        **fields,
    )
    session.add(event)
    teams = {"home_team": event.home_team, "away_team": event.away_team}
    books = [
        *([three_way_bookmaker("bet365", retail, **teams)] if retail is not None else []),
        *([betfair(exchange, **teams)] if exchange is not None else []),
    ]
    if books:
        session.add(
            make_tagged_snapshot(
                event_id, t - timedelta(minutes=5) + snapshot_offset, books, "oddsportal_live"
            )
        )
    if close is not None:
        session.add(
            make_tagged_snapshot(
                event_id, kickoff - timedelta(minutes=2), [betfair(close, **teams)], "betfair_api"
            )
        )
    await session.commit()


async def run(session: Any, cand: MechanicalCandidate | None = None, **kwargs: Any):
    return await replay(
        session,
        cand or candidate(),
        date(2026, 4, 1),
        date(2026, 4, 30),
        1,
        now=NOW,
        n_boot=N_BOOT,
        **kwargs,
    )


class TestReplay:
    async def test_bets_are_scored_at_the_price_taken_against_the_close(
        self, pglite_async_session
    ) -> None:
        await add_event(pglite_async_session, "e1", close=(2.5, 3.6, 3.6))

        result = await run(pglite_async_session)

        [stats] = [result.summary]
        assert stats.settled_bets == 1 and stats.bets_with_close == 1
        # Home won 2-1 at 2.2 for a unit stake.
        assert stats.pnl == pytest.approx(1.2)

    async def test_only_snapshots_stamped_at_or_before_the_decision_time_count(
        self, pglite_async_session
    ) -> None:
        await add_event(pglite_async_session, "late", snapshot_offset=timedelta(minutes=20))

        result = await run(pglite_async_session)

        assert (result.coverage.with_retail, result.coverage.bets) == (0, 0)

    async def test_decision_time_uses_the_events_final_kickoff(self, pglite_async_session) -> None:
        # A snapshot 30 minutes before the 15:00 kick-off is invisible at 14:00 but would be
        # visible had the event been decided 60 minutes before 15:30.
        await add_event(pglite_async_session, "moved", snapshot_offset=timedelta(minutes=30))
        later = candidate(
            params={"threshold": 0.02, "minutes_before_kickoff": [30], "max_input_age_minutes": 10}
        )

        assert (await run(pglite_async_session, later)).coverage.bets == 1
        assert (await run(pglite_async_session)).coverage.bets == 0

    async def test_each_decision_time_is_decided_and_an_outcome_is_bet_once(
        self, pglite_async_session
    ) -> None:
        await add_event(pglite_async_session, "e1")
        # Inputs at both decision times, each with the home edge.
        pglite_async_session.add(
            make_tagged_snapshot(
                "e1",
                KICKOFF - timedelta(minutes=95),
                [three_way_bookmaker("bet365", RETAIL), betfair(EXCHANGE)],
                "oddsportal_live",
            )
        )
        await pglite_async_session.commit()
        two_times = candidate(
            params={
                "threshold": 0.02,
                "minutes_before_kickoff": [60, 90],
                "max_input_age_minutes": 10,
            }
        )

        coverage = (await run(pglite_async_session, two_times)).coverage

        assert (coverage.events, coverage.decisions, coverage.with_input) == (1, 2, 2)
        assert (coverage.with_bet, coverage.bets) == (1, 1)

    async def test_twins_of_one_match_count_once(self, pglite_async_session) -> None:
        kickoff = TWIN_KICKOFF
        await add_event(pglite_async_session, TWIN_EVENT_ID, kickoff)
        await add_event(pglite_async_session, API_EVENT_ID, kickoff + timedelta(minutes=30))
        canonical = canonical_event_id([TWIN_EVENT_ID, API_EVENT_ID])
        other = API_EVENT_ID if canonical == TWIN_EVENT_ID else TWIN_EVENT_ID

        result = await run(pglite_async_session)

        assert result.coverage.events == 1
        assert canonical != other

    async def test_events_outside_the_range_or_not_final_are_excluded(
        self, pglite_async_session
    ) -> None:
        await add_event(pglite_async_session, "before", datetime(2026, 3, 31, 23, 0, tzinfo=UTC))
        await add_event(pglite_async_session, "last-day", datetime(2026, 4, 30, 23, 0, tzinfo=UTC))
        await add_event(pglite_async_session, "scheduled", scores=None)

        result = await run(pglite_async_session)

        assert result.coverage.events == 1

    async def test_coverage_counts_each_stage(self, pglite_async_session) -> None:
        day = timedelta(days=1)
        await add_event(pglite_async_session, "bet", KICKOFF)
        await add_event(pglite_async_session, "no-exchange", KICKOFF + day, exchange=None)
        await add_event(pglite_async_session, "no-edge", KICKOFF + 2 * day, retail=(1.9, 3.0, 3.0))
        await add_event(
            pglite_async_session, "nothing", KICKOFF + 3 * day, retail=None, exchange=None
        )
        await add_event(
            pglite_async_session, "stale", KICKOFF + 4 * day, snapshot_offset=-timedelta(minutes=6)
        )

        coverage = (await run(pglite_async_session)).coverage

        assert (
            coverage.events,
            coverage.decisions,
            coverage.with_input,
            coverage.with_exchange,
            coverage.with_retail,
            coverage.with_bet,
            coverage.bets,
            coverage.bets_with_close,
        ) == (5, 5, 2, 2, 2, 1, 1, 1)

    async def test_empty_range_has_no_verdict(self, pglite_async_session) -> None:
        result = await run(pglite_async_session)

        assert result.coverage.events == 0
        assert result.overall is None
        assert result.summary.verdict == "not_yet"


class TestHashStatus:
    async def test_reports_whether_the_recorded_hash_matches(self, pglite_async_session) -> None:
        assert (await run(pglite_async_session)).hash_status is HashStatus.MATCH
        stale = candidate(rule_sha256="0" * 64)
        assert (await run(pglite_async_session, stale)).hash_status is HashStatus.MISMATCH
        draft = candidate(rule_sha256=None)
        assert (await run(pglite_async_session, draft)).hash_status is HashStatus.ABSENT


class TestSharedScoring:
    async def test_same_bets_give_the_same_verdict_as_the_forward_report(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        trades = []
        for i in range(100):
            kickoff = KICKOFF + timedelta(hours=3 * i)  # beyond the twin window
            close = (2.0, 4.0, 4.0) if i % 2 else (2.4, 3.6, 3.6)
            await add_event(session, f"e{i}", kickoff, close=close)
            trades.append(
                PaperTrade(
                    event_id=f"e{i}",
                    market="1x2",
                    selection="home",
                    bookmaker="bet365",
                    price_decimal=2.2,
                    stake=1.0,
                    candidate_name="gap",
                    candidate_version=1,
                    placed_at=kickoff - timedelta(minutes=60),
                    settled_at=kickoff,
                    result=TradeResult.WIN,
                    pnl=1.2,
                )
            )
        session.add_all(trades)
        await session.commit()

        result = await run(session)
        [forward] = await candidate_report(session, [], N_BOOT, now=NOW)

        assert result.summary.bets_with_close == 100
        assert result.summary.checkpoint == forward.checkpoint
        assert result.summary.verdict == forward.verdict
        assert result.summary.pnl == pytest.approx(forward.pnl)


class TestFamilySize:
    async def test_report_counts_versions_from_their_first_decision(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        await add_event(session, "e1", KICKOFF)
        await add_event(session, "e2", KICKOFF + timedelta(days=1))
        for version, event_id, when in (
            (1, "e1", KICKOFF),
            (1, "e2", KICKOFF + timedelta(days=1)),
            (2, "e2", KICKOFF + timedelta(days=1)),
        ):
            session.add(
                CandidateDecision(
                    candidate_name="gap",
                    candidate_version=version,
                    event_id=event_id,
                    minutes_before_kickoff=60,
                    decision_time=when,
                    kickoff_at_decision=when,
                    known_at=when,
                )
            )
        await session.commit()

        starts = await load_decision_starts(session)
        assert starts == {("gap", 1): KICKOFF, ("gap", 2): KICKOFF + timedelta(days=1)}

        # The registered gap v3 and the sibling have not decided, so only v1 and v2 count.
        registry = [candidate(version=3), candidate(name="sibling")]
        summaries = await candidate_report(session, registry, N_BOOT, now=NOW)
        assert {(s.name, s.version): s.family_versions for s in summaries} == {
            ("gap", 3): 2,
            ("sibling", 1): 2,
        }


def model_candidate(**params: Any) -> MechanicalCandidate:
    return candidate(
        name="model",
        rule="retail_ev_model",
        rule_sha256=rule_sha256("retail_ev_model"),
        params={
            "model": "retail_ev_price_only",
            "threshold": 0.01,
            "training_matches": 3,
            "refit_every_matches": 1,
            "tuning_trials": 2,
            "tuning_seed": 0,
            "tuning_min_train_matches": 1,
            "tuning_step_matches": 1,
            "minutes_before_kickoff": [60],
            "max_input_age_minutes": 10,
            **params,
        },
    )


# A week apart, so each tuning fold has a training match a full decision window before it.
WEEK = timedelta(days=8)


async def add_weeks(session: Any, n: int) -> list[str]:
    ids = [f"m{i}" for i in range(n)]
    for i, event_id in enumerate(ids):
        await add_event(session, event_id, KICKOFF + i * WEEK, away_team=f"Team {i}")
    return ids


async def run_model(session: Any, cand: MechanicalCandidate, **kwargs: Any):
    return await replay(
        session, cand, date(2026, 4, 1), date(2026, 6, 30), 1, now=NOW, n_boot=N_BOOT, **kwargs
    )


class TestModelReplay:
    async def test_scores_only_the_matches_after_the_training_prefix(
        self, pglite_async_session
    ) -> None:
        ids = await add_weeks(pglite_async_session, 5)

        result = await run_model(pglite_async_session, model_candidate())

        assert result.range_events == 5
        assert result.tuning is not None and result.tuning.folds == 2
        assert result.decided_event_ids == tuple(ids[3:])
        assert result.coverage.events == 2 and result.coverage.decisions == 2
        assert [(r.training_matches, r.event_ids) for r in result.refits] == [
            (3, (ids[3],)),
            (4, (ids[4],)),
        ]

    async def test_a_model_that_learns_nothing_bets_like_the_exchange_gap(
        self, pglite_async_session
    ) -> None:
        # The close equals the exchange price at t, so the residual is zero and the
        # prediction is the baseline: the same EV the exchange-gap rule computes.
        ids = await add_weeks(pglite_async_session, 5)

        model = await run_model(pglite_async_session, model_candidate())
        gap = await run_model(
            pglite_async_session, candidate(), only_events=model.decided_event_ids
        )

        assert model.coverage.bets == gap.coverage.bets == 2
        assert gap.coverage.events == 2 and gap.range_events == 5
        assert model.overall is not None and gap.overall is not None
        assert model.overall.mean_ev == pytest.approx(gap.overall.mean_ev)
        assert gap.decided_event_ids is None and set(ids[3:]) == set(model.decided_event_ids)

    async def test_trains_only_on_matches_kicked_off_before_the_refit(
        self, pglite_async_session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # m2 kicks off 30 minutes before m3 and after m3's KO-60 decision.
        await add_event(pglite_async_session, "m0", KICKOFF, away_team="A")
        await add_event(pglite_async_session, "m1", KICKOFF + WEEK, away_team="B")
        await add_event(
            pglite_async_session, "m2", KICKOFF + 2 * WEEK - timedelta(minutes=30), away_team="C"
        )
        await add_event(pglite_async_session, "m3", KICKOFF + 2 * WEEK, away_team="D")
        from odds_analytics import candidate_model

        real = candidate_model.fit_model
        seen: list[set[str]] = []

        def fit(market: str, rows: Any, params: Any) -> Any:
            seen.append({r.event_id for r in rows})
            return real(market, rows, params)

        monkeypatch.setattr(candidate_model, "fit_model", fit)

        result = await run_model(pglite_async_session, model_candidate())

        assert seen == [{"m0", "m1"}]
        assert result.refits[0].training_matches == 2

    async def test_scored_matches_never_reach_tuning(
        self, pglite_async_session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ids = await add_weeks(pglite_async_session, 5)
        from odds_analytics import candidate_model

        real = candidate_model.choose_hyperparameters
        seen: set[str] = set()

        def tune(market: str, folds: Any, params: Any) -> Any:
            seen.update(r.event_id for train, val in folds for r in [*train, *val])
            return real(market, folds, params)

        monkeypatch.setattr(candidate_model, "choose_hyperparameters", tune)

        await run_model(pglite_async_session, model_candidate())

        assert seen == set(ids[:3])

    async def test_labels_come_only_from_a_live_close(self, pglite_async_session) -> None:
        # Closed only by FDUK, published after the refit, no training match has a label.
        for i in range(5):
            kickoff = KICKOFF + i * WEEK
            await add_event(pglite_async_session, f"m{i}", kickoff, away_team=f"T{i}", close=None)
            pglite_async_session.add(
                make_tagged_snapshot(
                    f"m{i}",
                    kickoff,
                    [betfair(EXCHANGE, home_team="Arsenal", away_team=f"T{i}")],
                    "football_data_uk",
                    FdukTimeBasis.KICKOFF,
                )
            )
        await pglite_async_session.commit()

        with pytest.raises(ValueError, match="no tuning fold"):
            await run_model(pglite_async_session, model_candidate())

    async def test_cannot_be_restricted_to_other_events(self, pglite_async_session) -> None:
        with pytest.raises(ValueError, match="model rule"):
            await run(pglite_async_session, model_candidate(), only_events={"x"})
