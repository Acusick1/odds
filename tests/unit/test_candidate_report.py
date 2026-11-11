"""Unit tests for the per-candidate report and its checkpoint verdict."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_analytics.candidate_report import (
    ScoredBet,
    Verdict,
    candidate_report,
    checkpoint_at,
    checkpoint_progress,
    checkpoints_through_next,
    family_size_at,
    family_starts,
    in_family,
    judge,
    league_breakdown,
    next_checkpoint,
    promote_level,
    report_candidates,
    summarize_candidates,
)
from odds_cli.main import app
from odds_core.candidates import AgentCandidate, CandidateStatus
from odds_core.close import close_quotes, event_close
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import PaperTrade, TradeResult
from odds_core.quotes import quotes
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_nba_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
    totals_book,
)

N_BOOT = 400
NOW = datetime(2026, 6, 1, tzinfo=UTC)
LONG_AGO = NOW - timedelta(days=30)
FDUK_GRACE = timedelta(days=7)


def one(_when: datetime) -> int:
    return 1


def trade(
    index: int,
    ev: float | None = 0.5,
    *,
    event_id: int | None = None,
    stake: float = 1.0,
    pnl: float | None = 0.0,
    name: str = "cand",
    version: int = 1,
    commence_time: datetime = LONG_AGO,
    close_grace: timedelta = FDUK_GRACE,
    sport_key: str = "soccer_epl",
) -> ScoredBet:
    """A trade at price 3.0 whose EV is ``ev`` (p_close = (1 + ev) / 3), None for no close."""
    return ScoredBet(
        candidate_name=name,
        candidate_version=version,
        event_id=event_id or index + 1,
        commence_time=commence_time,
        stake=stake,
        pnl=pnl,
        price_decimal=3.0,
        p_close=None if ev is None else (1 + ev) / 3,
        placed_at=LONG_AGO + timedelta(minutes=index),
        close_grace=close_grace,
        sport_key=sport_key,
    )


def trades(n: int, ev: float = 0.5, start: int = 0) -> list[ScoredBet]:
    return [trade(start + i, ev) for i in range(n)]


def noise(n: int, start: int = 0) -> list[ScoredBet]:
    return [trade(start + i, 1.0 if i % 2 else -1.0) for i in range(n)]


def summary_of(sequence: list[ScoredBet], now: datetime = NOW):
    return summarize_candidates(sequence, [], now, n_boot=N_BOOT)[0]


def registry_entry(name: str = "cand", version: int = 1) -> AgentCandidate:
    return AgentCandidate(
        name=name,
        description="d",
        sports=("soccer_epl",),
        version=version,
        issue=1,
        start_date="2026-01-01",
        status=CandidateStatus.ACTIVE,
        kind="agent",
    )


class TestEv:
    def test_ev_is_close_probability_times_price_minus_one(self) -> None:
        t = ScoredBet(
            "c",
            1,
            1,
            LONG_AGO,
            2.0,
            0.0,
            price_decimal=2.5,
            p_close=0.5,
            placed_at=LONG_AGO,
            close_grace=FDUK_GRACE,
            sport_key="soccer_epl",
        )
        assert t.ev == pytest.approx(0.25)

    def test_no_close_has_no_ev(self) -> None:
        assert trade(0, ev=None).ev is None


class TestCheckpoint:
    def test_under_one_checkpoint_is_not_yet(self) -> None:
        summary = summary_of(trades(99))
        assert summary.checkpoint is None
        assert summary.verdict is Verdict.NOT_YET

    def test_exactly_100_scores_all(self) -> None:
        checkpoint = checkpoint_at(trades(100), 100, 1, N_BOOT, 0)
        assert checkpoint.n == 100
        assert checkpoint.mean_ev == pytest.approx(0.5)

    def test_checkpoint_scores_only_the_earliest_n(self) -> None:
        sequence = trades(200, 0.5) + trades(50, -0.5, start=200)
        assert checkpoint_at(sequence, 200, 1, N_BOOT, 0).mean_ev == pytest.approx(0.5)

    def test_mean_is_stake_weighted(self) -> None:
        sequence = [trade(i, 1.0, stake=3.0) for i in range(50)] + [
            trade(50 + i, -1.0, stake=1.0) for i in range(50)
        ]
        assert checkpoint_at(sequence, 100, 1, N_BOOT, 0).mean_ev == pytest.approx(0.5)

    def test_events_are_resampled_as_clusters(self) -> None:
        sequence = [trade(i, 1.0, event_id=1) for i in range(50)] + [
            trade(50 + i, -1.0, event_id=2) for i in range(50)
        ]
        checkpoint = checkpoint_at(sequence, 100, 1, N_BOOT, 0)
        assert checkpoint.mean_ev == pytest.approx(0.0)
        assert checkpoint.promote_ci == pytest.approx((-1.0, 1.0))

    def test_seed_is_deterministic(self) -> None:
        sequence = [trade(i, 1.0 if i % 3 else -1.0) for i in range(100)]
        assert checkpoint_at(sequence, 100, 1, N_BOOT, 7) == checkpoint_at(
            sequence, 100, 1, N_BOOT, 7
        )


class TestCurrentEstimate:
    """The current estimate is separate from the verdict's checkpoints."""

    def test_below_a_checkpoint_there_is_an_estimate_and_no_checkpoint(self) -> None:
        summary = summary_of(trades(64, 0.5))
        assert summary.checkpoint is None and summary.checkpoints == ()
        assert summary.current is not None and summary.current.n == 64
        assert summary.verdict is Verdict.NOT_YET

    def test_one_event_has_no_estimate(self) -> None:
        sequence = [trade(i, event_id=1) for i in range(5)]
        assert summary_of(sequence).current is None

    def test_no_bets_with_a_close_have_no_estimate(self) -> None:
        assert summary_of([trade(0, None)]).current is None

    def test_the_verdict_checkpoint_is_frozen_while_the_estimate_follows_the_bets(self) -> None:
        summary = summary_of(trades(100, 0.5) + trades(100, -0.5, start=100))
        assert summary.verdict is Verdict.PROMOTE
        assert summary.checkpoint is not None and summary.checkpoint.n == 100
        assert [c.n for c in summary.checkpoints] == [100]
        assert summary.current is not None and summary.current.n == 200
        assert summary.current.mean_ev == pytest.approx(0.0)
        assert summary.checkpoint.mean_ev == pytest.approx(0.5)

    def test_the_estimate_is_the_checkpoint_when_both_score_the_same_bets(self) -> None:
        summary = summary_of(trades(100, 0.5))
        assert summary.current is summary.checkpoint

    def test_the_estimate_uses_the_family_size_now(self) -> None:
        registry = [mechanical("a"), mechanical("b")]
        decisions = {("a", 1): LONG_AGO, ("b", 1): NOW - timedelta(days=1)}
        summaries = summarize_candidates(
            [trade(i, name="a") for i in range(150)], registry, NOW, decisions, N_BOOT
        )
        a = next(s for s in summaries if s.name == "a")
        assert a.checkpoint is not None and a.checkpoint.family_versions == 1
        assert a.current is not None and a.current.family_versions == 2
        assert a.current.promote_level == promote_level(2)


class TestVerdict:
    def test_promote_when_99_lower_bound_above_zero(self) -> None:
        verdict, checkpoints = judge(trades(100, 0.5), one, N_BOOT, 0)
        assert verdict is Verdict.PROMOTE
        assert [c.n for c in checkpoints] == [100]

    def test_kill_when_95_upper_bound_at_or_below_zero(self) -> None:
        assert judge(trades(100, -0.5), one, N_BOOT, 0)[0] is Verdict.KILL

    def test_continue_when_interval_straddles_zero(self) -> None:
        verdict, checkpoints = judge(noise(400), one, N_BOOT, 0)
        assert verdict is Verdict.CONTINUE
        assert [c.n for c in checkpoints] == [100, 200, 300, 400]

    def test_kill_at_500_without_promotion(self) -> None:
        verdict, checkpoints = judge(noise(500), one, N_BOOT, 0)
        assert verdict is Verdict.KILL
        assert checkpoints[-1].n == 500

    def test_promotion_at_500_beats_the_kill_limit(self) -> None:
        assert judge(trades(500, 0.5), one, N_BOOT, 0)[0] is Verdict.PROMOTE

    def test_no_checkpoint_is_not_yet(self) -> None:
        assert judge(trades(99), one, N_BOOT, 0) == (Verdict.NOT_YET, [])

    def test_first_verdict_stands_when_later_bets_reverse_the_estimate(self) -> None:
        sequence = trades(100, 0.5) + trades(300, -0.5, start=100)
        verdict, checkpoints = judge(sequence, one, N_BOOT, 0)
        assert verdict is Verdict.PROMOTE
        assert [c.n for c in checkpoints] == [100]
        assert checkpoint_at(sequence, 400, 1, N_BOOT, 0).promote_ci[0] < 0

    def test_a_kill_stands_when_later_bets_recover(self) -> None:
        sequence = trades(100, -0.5) + trades(300, 0.5, start=100)
        verdict, checkpoints = judge(sequence, one, N_BOOT, 0)
        assert verdict is Verdict.KILL
        assert [c.n for c in checkpoints] == [100]

    def test_summary_carries_the_checkpoint_of_the_verdict(self) -> None:
        summary = summary_of(trades(100, 0.5) + trades(100, -0.5, start=100))
        assert summary.verdict is Verdict.PROMOTE
        assert summary.checkpoint is not None and summary.checkpoint.n == 100


class TestPrefix:
    def test_no_close_is_skipped_once_final_and_not_counted_toward_n(self) -> None:
        sequence = trades(60) + [trade(100 + i, None) for i in range(50)] + trades(60, start=200)
        summary = summary_of(sequence)
        assert summary.settled_bets == 170
        assert summary.bets_with_close == 120
        assert summary.bets_without_close == 50
        assert summary.pending_bets == 0
        assert summary.checkpoint is not None and summary.checkpoint.n == 100

    def test_unsettled_trade_ends_the_prefix(self) -> None:
        sequence = trades(60) + [trade(60, pnl=None)] + trades(60, start=61)
        summary = summary_of(sequence)
        assert summary.bets_with_close == 60
        assert summary.pending_bets == 61
        assert summary.settled_bets == 120
        assert summary.verdict is Verdict.NOT_YET

    def test_recent_no_close_ends_the_prefix_until_the_grace_passes(self) -> None:
        kickoff = NOW - timedelta(days=3)
        sequence = trades(60) + [trade(60, None, commence_time=kickoff)] + trades(60, start=61)
        waiting = summary_of(sequence)
        assert (waiting.bets_with_close, waiting.bets_without_close) == (60, 0)
        assert waiting.pending_bets == 61

        later = summary_of(sequence, now=kickoff + FDUK_GRACE)
        assert (later.bets_with_close, later.bets_without_close) == (120, 1)
        assert later.pending_bets == 0

    def test_a_recent_no_close_is_final_where_no_fallback_arrives_late(self) -> None:
        kickoff = NOW - timedelta(days=3)
        no_wait = trade(60, None, commence_time=kickoff, close_grace=timedelta(0))
        summary = summary_of(trades(60) + [no_wait] + trades(60, start=61))
        assert (summary.bets_with_close, summary.bets_without_close) == (120, 1)
        assert summary.pending_bets == 0

    def test_trades_placed_earlier_count_first(self) -> None:
        """A later-placed bet that settles first never takes an earlier bet's place."""
        sequence = [trade(0, pnl=None), *trades(100, start=1)]
        assert summary_of(sequence).checkpoint is None

    def test_prefix_is_fixed_once_final(self) -> None:
        before = trades(100, 0.5) + [trade(100, None, commence_time=NOW - timedelta(days=1))]
        after = before + trades(50, -0.5, start=101)
        assert summary_of(before).checkpoint == summary_of(after).checkpoint


class TestSummaries:
    def test_totals_and_roi_cover_settled_trades(self) -> None:
        sequence = [
            trade(0, 0.5, stake=10.0, pnl=20.0),
            trade(1, None, stake=5.0, pnl=-5.0),
            trade(2, 0.5, stake=7.0, pnl=None),
        ]
        summary = summary_of(sequence)
        assert summary.total_stake == 15.0
        assert summary.pnl == 15.0
        assert summary.roi == pytest.approx(1.0)

    def test_versions_are_separate_and_unregistered_has_no_status(self) -> None:
        sequence = [trade(0, version=1), trade(1, version=2)]
        by_version = {
            s.version: s
            for s in summarize_candidates(sequence, [registry_entry(version=2)], NOW, n_boot=N_BOOT)
        }
        assert by_version[1].status is None
        assert by_version[2].status is CandidateStatus.ACTIVE
        assert by_version[1].settled_bets == by_version[2].settled_bets == 1

    def test_each_version_carries_its_own_registry_entry(self) -> None:
        sequence = [trade(0, version=1), trade(1, version=2)]
        v1, v2 = registry_entry(version=1), registry_entry(version=2)
        v2 = v2.model_copy(update={"sports": ("basketball_nba",), "status": CandidateStatus.KILLED})

        by_version = {
            s.version: s for s in summarize_candidates(sequence, [v2], NOW, n_boot=N_BOOT)
        }
        both = {s.version: s for s in summarize_candidates(sequence, [v1, v2], NOW, n_boot=N_BOOT)}

        assert by_version[1].registered is None and by_version[2].registered == v2
        assert by_version[2].status is CandidateStatus.KILLED
        assert both[1].registered == v1 and both[1].registered.sports == ("soccer_epl",)
        assert both[2].registered.sports == ("basketball_nba",)  # type: ignore[union-attr]

    def test_registered_candidate_without_trades_is_listed(self) -> None:
        summary = summarize_candidates([], [registry_entry("fresh", 4)], NOW, n_boot=N_BOOT)[0]
        assert (summary.name, summary.version, summary.settled_bets) == ("fresh", 4, 0)
        assert summary.roi == 0.0
        assert summary.verdict is Verdict.NOT_YET

    def test_trades_of_a_candidate_missing_from_registry_are_listed(self) -> None:
        summary = summarize_candidates([trade(0, name="ghost")], [], NOW, n_boot=N_BOOT)[0]
        assert summary.name == "ghost" and summary.status is None


class TestCheckpointProgress:
    @pytest.mark.parametrize(
        ("bets", "target", "fraction"),
        [
            (0, 100, 0.0),
            (30, 100, 0.3),
            (100, 200, 0.0),
            (130, 200, 0.3),
            (499, 500, 0.99),
            (500, 500, 1.0),
            (620, 500, 1.0),
        ],
    )
    def test_progress_is_the_distance_from_the_last_checkpoint_to_the_next(
        self, bets: int, target: int, fraction: float
    ) -> None:
        progress = checkpoint_progress(bets)

        assert next_checkpoint(bets) == progress.next_checkpoint == target
        assert progress.bets_with_close == bets
        assert progress.fraction == pytest.approx(fraction)

    def test_the_checkpoints_through_the_next_stop_at_the_kill_checkpoint(self) -> None:
        assert checkpoints_through_next(0) == [100]
        assert checkpoints_through_next(130) == [100, 200]
        assert checkpoints_through_next(480) == [100, 200, 300, 400, 500]
        assert checkpoints_through_next(900) == [100, 200, 300, 400, 500]


def mechanical(name: str, version: int = 1, start: str = "2026-01-01", **overrides: object):
    from odds_core.candidates import MechanicalCandidate

    return MechanicalCandidate.model_validate(
        {
            "name": name,
            "description": "d",
            "kind": "mechanical",
            "market": "1x2",
            "sports": ["soccer_epl"],
            "version": version,
            "issue": 1,
            "start_date": start,
            "status": "active",
            "rule": "exchange_gap",
            "rule_sha256": "0" * 64,
            **overrides,
        }
    )


def at(day: int) -> datetime:
    return datetime(2026, 5, day, tzinfo=UTC)


class TestPromoteLevel:
    def test_one_version_keeps_the_99_level(self) -> None:
        assert promote_level(1) == pytest.approx(0.99)

    def test_three_versions_is_about_99_7(self) -> None:
        assert promote_level(3) == pytest.approx(0.996667, abs=1e-6)

    def test_k_above_one_widens_the_promote_interval_only(self) -> None:
        sequence = [trade(i, 1.0 if i % 4 else -1.0) for i in range(100)]
        base, wide = (checkpoint_at(sequence, 100, k, N_BOOT, 0) for k in (1, 8))
        assert (base.family_versions, wide.family_versions) == (1, 8)
        assert wide.promote_level > base.promote_level
        assert wide.promote_ci[0] <= base.promote_ci[0]
        assert wide.ci95 == base.ci95

    def test_a_higher_k_can_withhold_a_promotion(self) -> None:
        # 10 big winners in 100 events give a positive 99% lower bound but not a positive 99.999% one.
        sequence = [trade(i, 5.0 if i < 10 else -0.2, event_id=i + 1) for i in range(100)]
        assert judge(sequence, one, N_BOOT, 0)[0] is Verdict.PROMOTE
        assert judge(sequence, lambda _: 1000, N_BOOT, 0)[0] is not Verdict.PROMOTE


class TestFamilyStarts:
    def starts(self, registry, decisions=None, draft=None, draft_start=None):
        return family_starts(
            mechanical("epl"),
            registry,
            decisions or {},
            draft,
            draft_start,
        )

    def test_a_version_counts_from_its_first_decision(self) -> None:
        registry = [mechanical("a", version=3), mechanical("b")]
        decisions = {("a", 1): at(2), ("a", 2): at(9), ("a", 3): at(21), ("b", 1): at(5)}
        assert self.starts(registry, decisions) == [at(2), at(5), at(9), at(21)]

    def test_registered_versions_without_a_decision_do_not_count(self) -> None:
        registry = [mechanical("a"), mechanical("b", start="2026-01-01")]
        assert self.starts(registry, {("a", 1): at(4)}) == [at(4)]

    def test_killed_siblings_count(self) -> None:
        registry = [mechanical("a"), mechanical("b", status="killed")]
        assert len(self.starts(registry, {("a", 1): at(1), ("b", 1): at(2)})) == 2

    def test_other_rule_sport_or_market_or_unregistered_name_is_another_family(self) -> None:
        registry = [
            mechanical("a"),
            mechanical("b", rule="polymarket_gap"),
            mechanical("c", sports=["baseball_mlb"], market="h2h"),
            mechanical("d", market="totals"),
        ]
        decisions = {(name, 1): at(1) for name in ("a", "b", "c", "d", "ghost")}
        assert self.starts(registry, decisions) == [at(1)]

    def test_a_pooled_version_shares_the_family_of_each_league_it_lists(self) -> None:
        registry = [mechanical("pooled", sports=["soccer_epl", "soccer_italy_serie_a"])]
        assert self.starts(registry, {("pooled", 1): at(1)}) == [at(1)]

    def test_draft_counts_once_from_the_given_time(self) -> None:
        registry = [mechanical("a")]
        decisions = {("a", 1): at(3)}
        assert self.starts(registry, decisions, ("new", 1), at(30)) == [at(3), at(30)]
        assert self.starts(registry, decisions, ("a", 1), at(30)) == [at(3)]
        assert self.starts(registry, {}, ("a", 1), at(30)) == [at(30)]

    def test_size_counts_the_versions_started_by_a_time_and_is_at_least_one(self) -> None:
        starts = [at(3), at(10)]
        assert family_size_at(starts, at(1)) == 1
        assert family_size_at(starts, at(3)) == 1
        assert family_size_at(starts, at(10)) == 2


class TestFamilyOfIntersectingLeagues:
    epl = mechanical("epl")
    serie_a = mechanical("serie-a", sports=["soccer_italy_serie_a"])
    pooled = mechanical("pooled", sports=["soccer_epl", "soccer_italy_serie_a"])
    registry = [epl, serie_a, pooled]
    decisions = {("epl", 1): at(1), ("serie-a", 1): at(2), ("pooled", 1): at(3)}

    def k(self, candidate) -> int:
        return family_size_at(family_starts(candidate, self.registry, self.decisions), at(30))

    def test_the_pooled_version_shares_k_with_each_single_league_version(self) -> None:
        assert in_family(self.pooled, self.epl) and in_family(self.pooled, self.serie_a)
        assert self.k(self.pooled) == 3

    def test_single_league_versions_of_different_leagues_do_not_share_k(self) -> None:
        assert not in_family(self.epl, self.serie_a)
        assert self.k(self.epl) == self.k(self.serie_a) == 2

    def test_another_market_with_the_same_leagues_is_another_family(self) -> None:
        totals = mechanical(
            "pooled", sports=["soccer_epl", "soccer_italy_serie_a"], market="totals"
        )
        assert not in_family(totals, self.pooled)


class TestLeagueBreakdown:
    def test_it_splits_the_series_by_league_and_leaves_the_verdict_pooled(self) -> None:
        epl = [trade(i, 0.5, sport_key="soccer_epl") for i in range(0, 120, 2)]
        serie_a = [trade(i, -0.1, sport_key="soccer_italy_serie_a") for i in range(1, 120, 2)]
        sequence = sorted(epl + serie_a, key=lambda t: t.placed_at)

        summary = summary_of(sequence)

        by_league = {b.sport_key: b for b in summary.leagues}
        assert by_league["soccer_epl"].bets_with_close == 60
        assert by_league["soccer_epl"].mean_ev == pytest.approx(0.5)
        assert by_league["soccer_italy_serie_a"].mean_ev == pytest.approx(-0.1)
        assert summary.checkpoint is not None and summary.checkpoint.n == 100
        assert summary.checkpoint.mean_ev == pytest.approx(0.2)

    def test_a_bet_beyond_the_scoreable_prefix_has_no_ev_in_its_league(self) -> None:
        sequence = [
            trade(0, 0.5, sport_key="soccer_epl"),
            trade(1, 0.5, pnl=None, sport_key="soccer_epl"),
            trade(2, 0.5, sport_key="soccer_italy_serie_a"),
        ]
        by_league = {b.sport_key: b for b in league_breakdown(sequence, NOW)}
        assert by_league["soccer_epl"].bets_with_close == 1
        assert by_league["soccer_epl"].settled_bets == 1
        assert by_league["soccer_italy_serie_a"].bets_with_close == 0
        assert by_league["soccer_italy_serie_a"].mean_ev is None


class TestKAtEachCheckpoint:
    registry = [mechanical("a"), mechanical("b")]

    def judged(self, decisions: dict):
        sequence = [trade(i, 0.5, name="a") for i in range(100)]
        summaries = summarize_candidates(sequence, self.registry, NOW, decisions, N_BOOT)
        return {s.name: s for s in summaries}

    def test_a_variant_deciding_after_a_promote_leaves_it_standing(self) -> None:
        # Trade i is placed i minutes after LONG_AGO; b first decides a day later.
        summaries = self.judged({("a", 1): LONG_AGO, ("b", 1): LONG_AGO + timedelta(days=1)})
        assert summaries["a"].verdict is Verdict.PROMOTE
        assert summaries["a"].family_versions == 1
        assert summaries["b"].family_versions == 2

    def test_a_registered_variant_without_a_decision_does_not_change_k(self) -> None:
        summaries = self.judged({("a", 1): LONG_AGO})
        assert summaries["a"].family_versions == summaries["b"].family_versions == 1

    def test_a_variant_deciding_before_the_checkpoint_raises_its_k(self) -> None:
        a = self.judged({("a", 1): LONG_AGO, ("b", 1): LONG_AGO})["a"]
        assert a.family_versions == 2
        assert a.checkpoint is not None and a.checkpoint.family_versions == 2

    def test_each_checkpoint_uses_the_family_at_its_nth_bet(self) -> None:
        sequence = noise(300)
        verdict, checkpoints = judge(
            sequence, lambda when: 1 if when < LONG_AGO + timedelta(minutes=200) else 3, N_BOOT, 0
        )
        assert verdict is Verdict.CONTINUE
        assert [c.family_versions for c in checkpoints] == [1, 1, 3]

    def test_every_version_of_a_name_takes_the_family_of_its_registry_entry(self) -> None:
        registry = [mechanical("a", version=2)]
        sequence = [trade(i, name="a", version=1) for i in range(3)]
        decisions = {("a", 1): LONG_AGO, ("a", 2): at(31)}
        summaries = summarize_candidates(sequence, registry, NOW, decisions, N_BOOT)
        assert {(s.name, s.version): s.family_versions for s in summaries} == {
            ("a", 1): 2,
            ("a", 2): 2,
        }

    def test_unregistered_name_and_agent_have_k_1(self) -> None:
        summaries = summarize_candidates(
            [trade(0, name="ghost"), trade(1, name="cand")],
            [registry_entry("cand"), mechanical("a"), mechanical("b")],
            NOW,
            {("a", 1): at(1), ("b", 1): at(1)},
            n_boot=N_BOOT,
        )
        assert {s.name: s.family_versions for s in summaries}["ghost"] == 1
        assert {s.name: s.family_versions for s in summaries}["cand"] == 1


class TestRunningScore:
    def test_it_is_the_stake_weighted_mean_ev_and_the_running_totals(self) -> None:
        sequence = [trade(0, 0.3, stake=2.0, pnl=4.0), trade(1, -0.1, pnl=-1.0), trade(2, 0.5)]

        running = report_candidates(sequence, [], NOW, n_boot=N_BOOT).running[("cand", 1)]

        assert running.staked == pytest.approx((2.0, 3.0, 4.0))
        assert running.expected == pytest.approx((0.6, 0.5, 1.0))
        assert running.pnl == pytest.approx((4.0, 3.0, 3.0))
        assert running.mean_ev == pytest.approx((0.3, 0.5 / 3, 0.25))

    def test_it_covers_the_summarys_bets_with_a_close(self) -> None:
        sequence = [trade(0), trade(1, None), trade(2), trade(3, pnl=None), trade(4)]

        found = report_candidates(sequence, [], NOW, n_boot=N_BOOT)

        [summary] = found.summaries
        assert len(found.running[("cand", 1)].mean_ev) == summary.bets_with_close == 2

    def test_its_mean_is_the_score_at_every_checkpoint_and_the_current_n(self) -> None:
        sequence = noise(230)

        found = report_candidates(sequence, [], NOW, n_boot=N_BOOT)

        [summary] = found.summaries
        mean_ev = found.running[("cand", 1)].mean_ev
        assert summary.current is not None
        for point in (*summary.checkpoints, summary.current):
            assert mean_ev[point.n - 1] == pytest.approx(point.mean_ev)

    def test_every_summarised_version_has_one_and_the_summaries_are_unchanged(self) -> None:
        sequence = [*trades(3), trade(3, name="ghost", version=2)]
        registry = [registry_entry("fresh", 4)]

        found = report_candidates(sequence, registry, NOW, n_boot=N_BOOT)

        assert found.summaries == summarize_candidates(sequence, registry, NOW, n_boot=N_BOOT)
        assert set(found.running) == {(s.name, s.version) for s in found.summaries}
        assert found.running[("fresh", 4)].mean_ev == ()


KICKOFF = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)
CLOSE_PRICES = (2.0, 4.0, 4.0)


async def _store_close(session: AsyncSession, event: Event) -> None:
    await store_tagged(
        session,
        make_tagged_snapshot(
            event.id,
            KICKOFF - timedelta(minutes=10),
            [betfair_exchange_bookmaker(CLOSE_PRICES)],
            "betfair_api",
        ),
    )


def _paper_trade(event_id: int, minutes: int, **fields) -> PaperTrade:
    defaults = {
        "event_id": event_id,
        "market": "1x2",
        "selection": "home",
        "bookmaker": "bet365",
        "price_decimal": 2.2,
        "stake": 10.0,
        "bankroll_before": 1000.0,
        "candidate_name": "cand",
        "candidate_version": 1,
        "placed_at": KICKOFF - timedelta(days=1) + timedelta(minutes=minutes),
        "settled_at": KICKOFF + timedelta(hours=3),
        "result": TradeResult.WIN,
        "pnl": 12.0,
    }
    return PaperTrade(**{**defaults, **fields})


class TestLoadScoredBets:
    @pytest.mark.asyncio
    async def test_report_from_database(self, test_session: AsyncSession) -> None:
        session = test_session
        e1 = make_epl_event(1, KICKOFF, status=EventStatus.FINAL)
        session.add(e1)
        session.add(make_epl_event(2, KICKOFF, status=EventStatus.FINAL))
        session.add(
            Event(
                id=3,
                sport_key="baseball_mlb",
                sport_title="MLB",
                commence_time=KICKOFF,
                home_team="A",
                away_team="B",
                status=EventStatus.FINAL,
            )
        )
        await session.flush()
        await _store_close(session, e1)
        session.add_all(
            [
                _paper_trade(1, 2),
                _paper_trade(1, 1, selection="away", result=TradeResult.LOSS, pnl=-10.0),
                _paper_trade(1, 3, result=TradeResult.VOID, pnl=0.0),
                _paper_trade(1, 4, settled_at=None, result=None, pnl=None),
                _paper_trade(1, 5, candidate_name=None, candidate_version=None),
                _paper_trade(2, 6),
                _paper_trade(3, 7, market="h2h"),
            ]
        )
        await session.flush()

        summaries = (
            await candidate_report(
                session, [registry_entry()], n_boot=N_BOOT, now=KICKOFF + timedelta(days=30)
            )
        ).summaries

        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.settled_bets == 4
        assert summary.bets_with_close == 2
        # The unsettled trade ends the prefix, so the two trades after it are pending.
        assert summary.bets_without_close == 0
        assert summary.pending_bets == 3
        assert summary.pnl == pytest.approx(12.0 - 10.0 + 12.0 + 12.0)
        assert summary.status is CandidateStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_ev_uses_devigged_close_of_selection(self, test_session: AsyncSession) -> None:
        from odds_analytics.candidate_report import load_scored_trades

        session = test_session
        e1 = make_epl_event(1, KICKOFF, status=EventStatus.FINAL)
        session.add(e1)
        await session.flush()
        await _store_close(session, e1)
        session.add(_paper_trade(1, 1))
        session.add(_paper_trade(1, 2, selection="draw", price_decimal=4.5))
        await session.flush()

        home, draw = await load_scored_trades(session)
        assert home.commence_time == KICKOFF

        assert home.p_close == pytest.approx(0.5)
        assert home.ev == pytest.approx(0.5 * 2.2 - 1)
        assert draw.p_close == pytest.approx(0.25)
        assert draw.ev == pytest.approx(0.25 * 4.5 - 1)

    @pytest.mark.asyncio
    async def test_each_trade_is_scored_against_the_close_on_its_line(
        self, test_session: AsyncSession
    ) -> None:
        from odds_analytics.candidate_report import load_scored_trades

        session = test_session
        e1 = make_epl_event(1, KICKOFF, status=EventStatus.FINAL)
        session.add(e1)
        await session.flush()
        await _store_close(session, e1)
        # The live totals close prices 2.5 only, so the trade on 3.5 has no close.
        await store_tagged(
            session,
            make_tagged_snapshot(
                e1.id,
                KICKOFF - timedelta(minutes=10),
                [totals_book("betfair_exchange", {2.5: (2.1, 1.9)}, betfair_runners=True)],
                "betfair_api",
            ),
        )
        session.add(_paper_trade(1, 1))
        session.add(
            _paper_trade(1, 2, market="totals", selection="over", line=2.5, price_decimal=2.3)
        )
        session.add(
            _paper_trade(1, 3, market="totals", selection="under", line=3.5, price_decimal=1.6)
        )
        await session.flush()

        home, over, under = await load_scored_trades(session)

        assert home.p_close == pytest.approx(0.5)
        over_close = (1 / 2.1) / (1 / 2.1 + 1 / 1.9)
        assert over.p_close == pytest.approx(over_close)
        assert over.ev == pytest.approx(over_close * 2.3 - 1)
        assert under.p_close is None

    @pytest.mark.asyncio
    async def test_nba_h2h_trades_are_scored_against_the_nba_close(
        self, test_session: AsyncSession
    ) -> None:
        from odds_analytics.candidate_report import load_scored_trades

        session = test_session
        with_betfair = make_nba_event(1, KICKOFF, status=EventStatus.FINAL)
        with_polymarket = make_nba_event(2, KICKOFF, status=EventStatus.FINAL)
        with_neither = make_nba_event(3, KICKOFF, status=EventStatus.FINAL)
        mlb = make_nba_event(4, KICKOFF, status=EventStatus.FINAL, sport_key="baseball_mlb")
        session.add_all([with_betfair, with_polymarket, with_neither, mlb])
        await session.flush()
        names = (with_betfair.home_team, with_betfair.away_team)
        runners = [
            {"runner_name": name, "best_back": price}
            for name, price in zip(names, (2.0, 2.0), strict=True)
        ]
        betfair_entry = {
            "key": "betfair_exchange",
            "markets": [{"key": "h2h", "outcomes": []}],
            "betfair_meta": {"inplay": False, "market_status": "OPEN", "runners": runners},
        }
        polymarket_entry = {
            "key": "polymarket",
            "markets": [{"key": "h2h", "outcomes": []}],
            "polymarket_meta": {"yes_probabilities": {"home": 0.6, "away": 0.41}},
        }
        for event in (with_betfair, mlb):
            await store_tagged(
                session,
                make_tagged_snapshot(
                    event.id, KICKOFF - timedelta(minutes=5), [betfair_entry], "betfair_api"
                ),
            )
        for event in (with_polymarket, mlb):
            await store_tagged(
                session,
                make_tagged_snapshot(
                    event.id, KICKOFF - timedelta(minutes=2), [polymarket_entry], "polymarket"
                ),
            )
        for event_id, minutes in ((1, 1), (2, 2), (3, 3), (4, 4)):
            session.add(_paper_trade(event_id, minutes, market="h2h", price_decimal=2.1))
        await session.flush()

        betfair, polymarket, neither, mlb_trade = await load_scored_trades(session)

        assert betfair.p_close == pytest.approx(0.5)
        assert betfair.ev == pytest.approx(0.5 * 2.1 - 1)
        assert polymarket.p_close == pytest.approx(0.6 / 1.01)
        assert neither.p_close is None
        assert mlb_trade.p_close is None

    @pytest.mark.asyncio
    async def test_close_quotes_give_the_close_of_all_quotes(
        self, test_session: AsyncSession
    ) -> None:
        session = test_session
        events = [make_epl_event(i, KICKOFF, status=EventStatus.FINAL) for i in (1, 2)]
        session.add_all(events)
        await session.flush()
        e1, e2 = events

        async def betfair(event: Event, minutes_before: int, prices: tuple[float, float, float]):
            await store_tagged(
                session,
                make_tagged_snapshot(
                    event.id,
                    KICKOFF - timedelta(minutes=minutes_before),
                    [betfair_exchange_bookmaker(prices)],
                    "betfair_api",
                ),
            )

        async def fduk(event: Event, basis: str, prices: tuple[float, float, float]):
            await store_tagged(
                session,
                make_tagged_snapshot(
                    event.id,
                    KICKOFF - timedelta(hours=3 if basis == "fduk_collection_deadline" else 0),
                    [three_way_bookmaker("betfair_exchange", prices)],
                    "football_data_uk",
                    basis,
                ),
            )

        await betfair(e1, 300, (3.0, 3.0, 3.0))
        await betfair(e1, 20, (2.0, 4.0, 4.0))
        await betfair(e1, 5, (2.1, 3.9, 4.0))
        await fduk(e1, "fduk_collection_deadline", (2.5, 3.5, 3.5))
        await fduk(e1, "kickoff", (1.9, 4.0, 4.5))
        await betfair(e2, 300, (3.0, 3.0, 3.0))
        await fduk(e2, "fduk_collection_deadline", (2.5, 3.5, 3.5))
        await fduk(e2, "kickoff", (1.9, 4.0, 4.5))
        await session.flush()

        restricted = await close_quotes(session, events, "1x2")
        for event in events:
            full = await quotes(session, [event.id], "1x2")
            kept = restricted[event.id]
            assert len(kept) < len(full)
            expected = event_close(full, KICKOFF, "soccer_epl", "1x2")
            assert expected is not None
            assert event_close(kept, KICKOFF, "soccer_epl", "1x2") == expected


# The first 100 bets reach the k=100 checkpoint at EV +0.5; the 20 after it pull the current
# estimate down to +0.33, which the command never shows.
COHORT = (
    trades(100)
    + trades(20, ev=-0.5, start=100)
    + [trade(i, name="ghost", version=2) for i in range(3)]
)


EXPECTED_REPORT = """\
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━┳━━━━━━━━━┓
┃ Candidate ┃ Version ┃ Status      ┃ Settled ┃ With close ┃ No close ┃ Pending ┃ Staked ┃   P&L ┃   ROI ┃ Checkpoint n ┃ Mean EV ┃           95% CI ┃              Promote CI ┃ k ┃ Verdict ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━╇━━━━━━━━━┩
│ cand      │       1 │ not current │     120 │        120 │        0 │       0 │ 120.00 │ +0.00 │ +0.0% │          100 │ +0.5000 │ [+0.500, +0.500] │ 99.00% [+0.500, +0.500] │ 1 │ promote │
│ epl-agent │       3 │ promoted    │       0 │          0 │        0 │       0 │   0.00 │ +0.00 │ +0.0% │            - │       - │                - │                       - │ 1 │ not_yet │
│ ghost     │       2 │ not current │       3 │          3 │        0 │       0 │   3.00 │ +0.00 │ +0.0% │            - │       - │                - │                       - │ 1 │ not_yet │
└───────────┴─────────┴─────────────┴─────────┴────────────┴──────────┴─────────┴────────┴───────┴───────┴──────────────┴─────────┴──────────────────┴─────────────────────────┴───┴─────────┘
"""


class TestReportCommand:
    @pytest.fixture
    def registry(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        from tests.unit.test_candidates import AGENT, write

        write(tmp_path, AGENT)
        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: tmp_path)

        async def fake_report(registry):
            return summarize_candidates(COHORT, registry, NOW, n_boot=N_BOOT)

        monkeypatch.setattr("odds_cli.commands.candidates._report", fake_report)
        return tmp_path

    def test_prints_the_verdict_checkpoint_not_the_current_estimate(self, registry) -> None:
        summaries = summarize_candidates(COHORT, [], NOW, n_boot=N_BOOT)
        cand = next(s for s in summaries if s.name == "cand")
        assert cand.checkpoint is not None and cand.current is not None
        assert cand.checkpoint.mean_ev == pytest.approx(0.5)
        assert cand.current.mean_ev == pytest.approx(1 / 3)

        result = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["candidates", "report"])

        assert result.exit_code == 0, result.stdout
        assert result.stdout == EXPECTED_REPORT


@pytest.fixture
def stored_cohort(monkeypatch: pytest.MonkeyPatch) -> None:
    """``candidate_report`` reads ``COHORT`` with no decisions, through a session nothing uses."""

    async def scored(_session: object, _candidate: object = None) -> list[ScoredBet]:
        return COHORT

    async def no_starts(_session: object) -> dict[tuple[str, int], datetime]:
        return {}

    class Session:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr("odds_analytics.candidate_report.load_scored_trades", scored)
    monkeypatch.setattr("odds_analytics.candidate_report.load_decision_starts", no_starts)
    monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", Session)
    monkeypatch.setattr("odds_mcp.server.async_session_maker", Session)


class TestReportOutputThroughCandidateReport:
    def test_the_command_prints_the_same_table(
        self, stored_cohort: None, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.unit.test_candidates import AGENT, write

        write(tmp_path, AGENT)
        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: tmp_path)

        result = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["candidates", "report"])

        assert result.exit_code == 0, result.stdout
        assert result.stdout == EXPECTED_REPORT

    @pytest.mark.asyncio
    async def test_get_portfolio_returns_the_same_candidates(
        self, stored_cohort: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import AsyncMock

        from odds_lambda.paper_trading import PortfolioSummary
        from odds_mcp.server import get_portfolio

        portfolio = PortfolioSummary(1000.0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0)
        monkeypatch.setattr(
            "odds_mcp.server.get_portfolio_summary", AsyncMock(return_value=portfolio)
        )
        monkeypatch.setattr("odds_mcp.server.get_open_trades", AsyncMock(return_value=[]))
        monkeypatch.setattr("odds_mcp.server.load_candidates", lambda _directory: [])

        result = await get_portfolio()

        assert result["candidates"] == [
            {
                "name": "cand",
                "version": 1,
                "status": None,
                "settled_bets": 120,
                "bets_with_close": 120,
                "bets_without_close": 0,
                "pending_bets": 0,
                "total_stake": 120.0,
                "pnl": 0.0,
                "roi": 0.0,
                "checkpoint": {
                    "n": 100,
                    "mean_ev": pytest.approx(0.5),
                    "ci95": [pytest.approx(0.5), pytest.approx(0.5)],
                    "promote_level": 0.99,
                    "promote_ci": [pytest.approx(0.5), pytest.approx(0.5)],
                },
                "family_versions": 1,
                "verdict": "promote",
            },
            {
                "name": "ghost",
                "version": 2,
                "status": None,
                "settled_bets": 3,
                "bets_with_close": 3,
                "bets_without_close": 0,
                "pending_bets": 0,
                "total_stake": 3.0,
                "pnl": 0.0,
                "roi": 0.0,
                "checkpoint": None,
                "family_versions": 1,
                "verdict": "not_yet",
            },
        ]
