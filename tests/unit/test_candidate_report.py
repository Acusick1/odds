"""Unit tests for the per-candidate report and its checkpoint verdict."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_analytics.candidate_report import (
    ScoredTrade,
    Verdict,
    candidate_report,
    checkpoint_of,
    summarize_candidates,
    verdict_of,
)
from odds_cli.main import app
from odds_core.candidates import AgentCandidate, CandidateStatus
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import PaperTrade, TradeResult
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

N_BOOT = 400


def trade(
    index: int,
    ev: float | None = 0.5,
    *,
    event_id: str | None = None,
    stake: float = 1.0,
    pnl: float = 0.0,
    name: str = "cand",
    version: int = 1,
) -> ScoredTrade:
    """A trade at price 3.0 whose EV is ``ev`` (p_close = (1 + ev) / 3), None for no close."""
    return ScoredTrade(
        candidate_name=name,
        candidate_version=version,
        event_id=event_id or f"e{index}",
        stake=stake,
        pnl=pnl,
        price_decimal=3.0,
        p_close=None if ev is None else (1 + ev) / 3,
    )


def trades(n: int, ev: float = 0.5, start: int = 0) -> list[ScoredTrade]:
    return [trade(start + i, ev) for i in range(n)]


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
        t = ScoredTrade("c", 1, "e", 2.0, 0.0, price_decimal=2.5, p_close=0.5)
        assert t.ev == pytest.approx(0.25)

    def test_no_close_has_no_ev(self) -> None:
        assert trade(0, ev=None).ev is None


class TestCheckpoint:
    def test_under_one_checkpoint_is_not_yet(self) -> None:
        summary = summarize_candidates(trades(99), [], N_BOOT)[0]
        assert summary.checkpoint is None
        assert summary.verdict is Verdict.NOT_YET

    def test_exactly_100_scores_all(self) -> None:
        checkpoint = checkpoint_of(trades(100), N_BOOT, 0)
        assert checkpoint is not None
        assert checkpoint.n == 100
        assert checkpoint.mean_ev == pytest.approx(0.5)

    def test_250_scores_first_200_only(self) -> None:
        sequence = trades(200, 0.5) + trades(50, -0.5, start=200)
        checkpoint = checkpoint_of(sequence, N_BOOT, 0)
        assert checkpoint is not None
        assert checkpoint.n == 200
        assert checkpoint.mean_ev == pytest.approx(0.5)

    def test_mean_is_stake_weighted(self) -> None:
        sequence = [trade(i, 1.0, stake=3.0) for i in range(50)] + [
            trade(50 + i, -1.0, stake=1.0) for i in range(50)
        ]
        checkpoint = checkpoint_of(sequence, N_BOOT, 0)
        assert checkpoint is not None
        assert checkpoint.mean_ev == pytest.approx(0.5)

    def test_bets_without_close_do_not_count_toward_n(self) -> None:
        sequence = trades(60) + [trade(100 + i, None) for i in range(50)] + trades(60, start=200)
        summary = summarize_candidates(sequence, [], N_BOOT)[0]
        assert summary.settled_bets == 170
        assert summary.bets_with_close == 120
        assert summary.bets_without_close == 50
        assert summary.checkpoint is not None and summary.checkpoint.n == 100

    def test_events_are_resampled_as_clusters(self) -> None:
        sequence = [trade(i, 1.0, event_id="a") for i in range(50)] + [
            trade(50 + i, -1.0, event_id="b") for i in range(50)
        ]
        checkpoint = checkpoint_of(sequence, N_BOOT, 0)
        assert checkpoint is not None
        assert checkpoint.mean_ev == pytest.approx(0.0)
        assert checkpoint.ci99 == pytest.approx((-1.0, 1.0))

    def test_seed_is_deterministic(self) -> None:
        sequence = [trade(i, 1.0 if i % 3 else -1.0) for i in range(100)]
        assert checkpoint_of(sequence, N_BOOT, 7) == checkpoint_of(sequence, N_BOOT, 7)


class TestVerdict:
    def test_promote_when_99_lower_bound_above_zero(self) -> None:
        assert verdict_of(checkpoint_of(trades(100, 0.5), N_BOOT, 0)) is Verdict.PROMOTE

    def test_kill_when_95_upper_bound_at_or_below_zero(self) -> None:
        assert verdict_of(checkpoint_of(trades(100, -0.5), N_BOOT, 0)) is Verdict.KILL

    @staticmethod
    def _noise(n: int) -> list[ScoredTrade]:
        return [trade(i, 1.0 if i % 2 else -1.0) for i in range(n)]

    def test_continue_when_interval_straddles_zero(self) -> None:
        assert verdict_of(checkpoint_of(self._noise(100), N_BOOT, 0)) is Verdict.CONTINUE
        assert verdict_of(checkpoint_of(self._noise(400), N_BOOT, 0)) is Verdict.CONTINUE

    def test_kill_at_500_without_promotion(self) -> None:
        checkpoint = checkpoint_of(self._noise(500), N_BOOT, 0)
        assert checkpoint is not None and checkpoint.n == 500
        assert verdict_of(checkpoint) is Verdict.KILL

    def test_promotion_at_500_beats_the_kill_limit(self) -> None:
        assert verdict_of(checkpoint_of(trades(500, 0.5), N_BOOT, 0)) is Verdict.PROMOTE

    def test_no_checkpoint_is_not_yet(self) -> None:
        assert verdict_of(None) is Verdict.NOT_YET


class TestSummaries:
    def test_totals_and_roi(self) -> None:
        sequence = [
            trade(0, 0.5, stake=10.0, pnl=20.0),
            trade(1, None, stake=5.0, pnl=-5.0),
        ]
        summary = summarize_candidates(sequence, [], N_BOOT)[0]
        assert summary.total_stake == 15.0
        assert summary.pnl == 15.0
        assert summary.roi == pytest.approx(1.0)

    def test_versions_are_separate_and_unregistered_has_no_status(self) -> None:
        sequence = [trade(0, version=1), trade(1, version=2)]
        by_version = {
            s.version: s
            for s in summarize_candidates(sequence, [registry_entry(version=2)], N_BOOT)
        }
        assert by_version[1].status is None
        assert by_version[2].status is CandidateStatus.ACTIVE
        assert by_version[1].settled_bets == by_version[2].settled_bets == 1

    def test_registered_candidate_without_trades_is_listed(self) -> None:
        summary = summarize_candidates([], [registry_entry("fresh", 4)], N_BOOT)[0]
        assert (summary.name, summary.version, summary.settled_bets) == ("fresh", 4, 0)
        assert summary.roi == 0.0
        assert summary.verdict is Verdict.NOT_YET

    def test_trades_of_a_candidate_missing_from_registry_are_listed(self) -> None:
        summary = summarize_candidates([trade(0, name="ghost")], [], N_BOOT)[0]
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

        summaries = await candidate_report(session, [registry_entry()], n_boot=N_BOOT)

        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.settled_bets == 4
        assert summary.bets_with_close == 2
        assert summary.bets_without_close == 2
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

        assert home.p_close == pytest.approx(0.5)
        assert home.ev == pytest.approx(0.5 * 2.2 - 1)
        assert draw.p_close == pytest.approx(0.25)
        assert draw.ev == pytest.approx(0.25 * 4.5 - 1)


class TestReportCommand:
    @pytest.fixture
    def registry(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        from tests.unit.test_candidates import AGENT, write

        write(tmp_path, AGENT)
        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: tmp_path)

        async def fake_report(registry):
            sequence = trades(100) + [trade(i, name="ghost", version=2) for i in range(3)]
            return summarize_candidates(sequence, registry, N_BOOT)

        monkeypatch.setattr("odds_cli.commands.candidates._report", fake_report)
        return tmp_path

    def test_lists_registered_and_unregistered_candidates(self, registry) -> None:
        result = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["candidates", "report"])
        assert result.exit_code == 0, result.stdout
        assert "epl-agent" in result.stdout
        assert "ghost" in result.stdout
        assert "not current" in result.stdout
        assert "not_yet" in result.stdout
