"""Unit tests for the per-candidate report and its checkpoint verdict."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_analytics.candidate_report import (
    CLOSE_GRACE,
    ScoredTrade,
    Verdict,
    candidate_report,
    checkpoint_at,
    close_candidate_snapshots,
    judge,
    summarize_candidates,
)
from odds_cli.main import app
from odds_core.betfair_close import event_close
from odds_core.candidates import AgentCandidate, CandidateStatus
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.paper_trade_models import PaperTrade, TradeResult
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

N_BOOT = 400
NOW = datetime(2026, 6, 1, tzinfo=UTC)
LONG_AGO = NOW - timedelta(days=30)


def trade(
    index: int,
    ev: float | None = 0.5,
    *,
    event_id: str | None = None,
    stake: float = 1.0,
    pnl: float | None = 0.0,
    name: str = "cand",
    version: int = 1,
    commence_time: datetime = LONG_AGO,
) -> ScoredTrade:
    """A trade at price 3.0 whose EV is ``ev`` (p_close = (1 + ev) / 3), None for no close."""
    return ScoredTrade(
        candidate_name=name,
        candidate_version=version,
        event_id=event_id or f"e{index}",
        commence_time=commence_time,
        stake=stake,
        pnl=pnl,
        price_decimal=3.0,
        p_close=None if ev is None else (1 + ev) / 3,
    )


def trades(n: int, ev: float = 0.5, start: int = 0) -> list[ScoredTrade]:
    return [trade(start + i, ev) for i in range(n)]


def noise(n: int, start: int = 0) -> list[ScoredTrade]:
    return [trade(start + i, 1.0 if i % 2 else -1.0) for i in range(n)]


def summary_of(sequence: list[ScoredTrade], now: datetime = NOW):
    return summarize_candidates(sequence, [], now, N_BOOT)[0]


def registry_entry(name: str = "cand", version: int = 1) -> AgentCandidate:
    return AgentCandidate(
        name=name,
        description="d",
        sport="soccer_epl",
        version=version,
        issue=1,
        start_date="2026-01-01",
        status=CandidateStatus.ACTIVE,
        kind="agent",
    )


class TestEv:
    def test_ev_is_close_probability_times_price_minus_one(self) -> None:
        t = ScoredTrade("c", 1, "e", LONG_AGO, 2.0, 0.0, price_decimal=2.5, p_close=0.5)
        assert t.ev == pytest.approx(0.25)

    def test_no_close_has_no_ev(self) -> None:
        assert trade(0, ev=None).ev is None


class TestCheckpoint:
    def test_under_one_checkpoint_is_not_yet(self) -> None:
        summary = summary_of(trades(99))
        assert summary.checkpoint is None
        assert summary.verdict is Verdict.NOT_YET

    def test_exactly_100_scores_all(self) -> None:
        checkpoint = checkpoint_at(trades(100), 100, N_BOOT, 0)
        assert checkpoint.n == 100
        assert checkpoint.mean_ev == pytest.approx(0.5)

    def test_checkpoint_scores_only_the_earliest_n(self) -> None:
        sequence = trades(200, 0.5) + trades(50, -0.5, start=200)
        assert checkpoint_at(sequence, 200, N_BOOT, 0).mean_ev == pytest.approx(0.5)

    def test_mean_is_stake_weighted(self) -> None:
        sequence = [trade(i, 1.0, stake=3.0) for i in range(50)] + [
            trade(50 + i, -1.0, stake=1.0) for i in range(50)
        ]
        assert checkpoint_at(sequence, 100, N_BOOT, 0).mean_ev == pytest.approx(0.5)

    def test_events_are_resampled_as_clusters(self) -> None:
        sequence = [trade(i, 1.0, event_id="a") for i in range(50)] + [
            trade(50 + i, -1.0, event_id="b") for i in range(50)
        ]
        checkpoint = checkpoint_at(sequence, 100, N_BOOT, 0)
        assert checkpoint.mean_ev == pytest.approx(0.0)
        assert checkpoint.ci99 == pytest.approx((-1.0, 1.0))

    def test_seed_is_deterministic(self) -> None:
        sequence = [trade(i, 1.0 if i % 3 else -1.0) for i in range(100)]
        assert checkpoint_at(sequence, 100, N_BOOT, 7) == checkpoint_at(sequence, 100, N_BOOT, 7)


class TestVerdict:
    def test_promote_when_99_lower_bound_above_zero(self) -> None:
        verdict, checkpoint = judge(trades(100, 0.5), N_BOOT, 0)
        assert verdict is Verdict.PROMOTE
        assert checkpoint is not None and checkpoint.n == 100

    def test_kill_when_95_upper_bound_at_or_below_zero(self) -> None:
        assert judge(trades(100, -0.5), N_BOOT, 0)[0] is Verdict.KILL

    def test_continue_when_interval_straddles_zero(self) -> None:
        verdict, checkpoint = judge(noise(400), N_BOOT, 0)
        assert verdict is Verdict.CONTINUE
        assert checkpoint is not None and checkpoint.n == 400

    def test_kill_at_500_without_promotion(self) -> None:
        verdict, checkpoint = judge(noise(500), N_BOOT, 0)
        assert verdict is Verdict.KILL
        assert checkpoint is not None and checkpoint.n == 500

    def test_promotion_at_500_beats_the_kill_limit(self) -> None:
        assert judge(trades(500, 0.5), N_BOOT, 0)[0] is Verdict.PROMOTE

    def test_no_checkpoint_is_not_yet(self) -> None:
        assert judge(trades(99), N_BOOT, 0) == (Verdict.NOT_YET, None)

    def test_first_verdict_stands_when_later_bets_reverse_the_estimate(self) -> None:
        sequence = trades(100, 0.5) + trades(300, -0.5, start=100)
        verdict, checkpoint = judge(sequence, N_BOOT, 0)
        assert verdict is Verdict.PROMOTE
        assert checkpoint is not None and checkpoint.n == 100
        assert checkpoint_at(sequence, 400, N_BOOT, 0).ci99[0] < 0

    def test_a_kill_stands_when_later_bets_recover(self) -> None:
        sequence = trades(100, -0.5) + trades(300, 0.5, start=100)
        verdict, checkpoint = judge(sequence, N_BOOT, 0)
        assert verdict is Verdict.KILL
        assert checkpoint is not None and checkpoint.n == 100

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

        later = summary_of(sequence, now=kickoff + CLOSE_GRACE)
        assert (later.bets_with_close, later.bets_without_close) == (120, 1)
        assert later.pending_bets == 0

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
            for s in summarize_candidates(sequence, [registry_entry(version=2)], NOW, N_BOOT)
        }
        assert by_version[1].status is None
        assert by_version[2].status is CandidateStatus.ACTIVE
        assert by_version[1].settled_bets == by_version[2].settled_bets == 1

    def test_registered_candidate_without_trades_is_listed(self) -> None:
        summary = summarize_candidates([], [registry_entry("fresh", 4)], NOW, N_BOOT)[0]
        assert (summary.name, summary.version, summary.settled_bets) == ("fresh", 4, 0)
        assert summary.roi == 0.0
        assert summary.verdict is Verdict.NOT_YET

    def test_trades_of_a_candidate_missing_from_registry_are_listed(self) -> None:
        summary = summarize_candidates([trade(0, name="ghost")], [], NOW, N_BOOT)[0]
        assert summary.name == "ghost" and summary.status is None


KICKOFF = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)
CLOSE_PRICES = (2.0, 4.0, 4.0)


def _close_snapshot(event_id: str):
    return make_tagged_snapshot(
        event_id,
        KICKOFF - timedelta(minutes=10),
        [
            three_way_bookmaker(
                "betfair_exchange",
                CLOSE_PRICES,
                betfair_meta={"inplay": False, "market_status": "OPEN"},
            )
        ],
        "betfair_api",
    )


def _paper_trade(event_id: str, minutes: int, **fields) -> PaperTrade:
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


class TestLoadScoredTrades:
    @pytest.mark.asyncio
    async def test_report_from_database(self, test_session: AsyncSession) -> None:
        session = test_session
        session.add(make_epl_event("e1", KICKOFF, status=EventStatus.FINAL))
        session.add(make_epl_event("e2", KICKOFF, status=EventStatus.FINAL))
        session.add(
            Event(
                id="mlb1",
                sport_key="baseball_mlb",
                sport_title="MLB",
                commence_time=KICKOFF,
                home_team="A",
                away_team="B",
                status=EventStatus.FINAL,
            )
        )
        await session.flush()
        session.add(_close_snapshot("e1"))
        session.add_all(
            [
                _paper_trade("e1", 2),
                _paper_trade("e1", 1, selection="away", result=TradeResult.LOSS, pnl=-10.0),
                _paper_trade("e1", 3, result=TradeResult.VOID, pnl=0.0),
                _paper_trade("e1", 4, settled_at=None, result=None, pnl=None),
                _paper_trade("e1", 5, candidate_name=None, candidate_version=None),
                _paper_trade("e2", 6),
                _paper_trade("mlb1", 7, market="h2h"),
            ]
        )
        await session.flush()

        summaries = await candidate_report(
            session, [registry_entry()], n_boot=N_BOOT, now=KICKOFF + timedelta(days=30)
        )

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
        session.add(make_epl_event("e1", KICKOFF, status=EventStatus.FINAL))
        await session.flush()
        session.add(_close_snapshot("e1"))
        session.add(_paper_trade("e1", 1))
        session.add(_paper_trade("e1", 2, selection="draw", price_decimal=4.5))
        await session.flush()

        home, draw = await load_scored_trades(session)
        assert home.commence_time == KICKOFF

        assert home.p_close == pytest.approx(0.5)
        assert home.ev == pytest.approx(0.5 * 2.2 - 1)
        assert draw.p_close == pytest.approx(0.25)
        assert draw.ev == pytest.approx(0.25 * 4.5 - 1)

    @pytest.mark.asyncio
    async def test_restricted_snapshot_load_gives_the_same_close(
        self, test_session: AsyncSession
    ) -> None:
        session = test_session
        session.add(make_epl_event("e1", KICKOFF, status=EventStatus.FINAL))
        session.add(make_epl_event("e2", KICKOFF, status=EventStatus.FINAL))
        await session.flush()

        def betfair(event_id: str, minutes_before: int, prices: tuple[float, float, float]):
            return make_tagged_snapshot(
                event_id,
                KICKOFF - timedelta(minutes=minutes_before),
                [three_way_bookmaker("betfair_exchange", prices)],
                "betfair_api",
            )

        def fduk(event_id: str, basis: str, prices: tuple[float, float, float]):
            return make_tagged_snapshot(
                event_id,
                KICKOFF,
                [three_way_bookmaker("betfair_exchange", prices)],
                "football_data_uk",
                basis,
            )

        session.add_all(
            [
                betfair("e1", 300, (3.0, 3.0, 3.0)),
                betfair("e1", 20, (2.0, 4.0, 4.0)),
                betfair("e1", 5, (2.1, 3.9, 4.0)),
                fduk("e1", "fduk_collection_deadline", (2.5, 3.5, 3.5)),
                fduk("e1", "kickoff", (1.9, 4.0, 4.5)),
                betfair("e2", 300, (3.0, 3.0, 3.0)),
                fduk("e2", "fduk_collection_deadline", (2.5, 3.5, 3.5)),
                fduk("e2", "kickoff", (1.9, 4.0, 4.5)),
            ]
        )
        await session.flush()

        full = (await session.execute(select(OddsSnapshot))).scalars().all()
        restricted = (await session.execute(close_candidate_snapshots({"e1", "e2"}))).scalars()
        kept = list(restricted)

        assert len(kept) < len(full)
        for event_id in ("e1", "e2"):
            event = await session.get(Event, event_id)
            assert event is not None
            expected = event_close(event, [s for s in full if s.event_id == event_id], "1x2")
            actual = event_close(event, [s for s in kept if s.event_id == event_id], "1x2")
            assert expected is not None
            assert actual == expected


class TestReportCommand:
    @pytest.fixture
    def registry(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        from tests.unit.test_candidates import AGENT, write

        write(tmp_path, AGENT)
        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: tmp_path)

        async def fake_report(registry):
            sequence = trades(100) + [trade(i, name="ghost", version=2) for i in range(3)]
            return summarize_candidates(sequence, registry, NOW, N_BOOT)

        monkeypatch.setattr("odds_cli.commands.candidates._report", fake_report)
        return tmp_path

    def test_lists_registered_and_unregistered_candidates(self, registry) -> None:
        result = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["candidates", "report"])
        assert result.exit_code == 0, result.stdout
        assert "epl-agent" in result.stdout
        assert "ghost" in result.stdout
        assert "not current" in result.stdout
        assert "not_yet" in result.stdout
