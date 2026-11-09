"""Report-data builders shared by the odds-web page tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from odds_analytics.candidate_report import CandidateSummary, ScoredBet, summarize_candidates
from odds_core.candidates import AgentCandidate, CandidateStatus, MechanicalCandidate

NOW = datetime(2026, 6, 1, tzinfo=UTC)
LONG_AGO = NOW - timedelta(days=30)
N_BOOT = 400


def bet(index: int, ev: float, *, name: str = "gap", p_close: bool = True) -> ScoredBet:
    """A settled trade at price 3.0 whose EV is ``ev``; without a close when ``p_close`` is False."""
    return ScoredBet(
        candidate_name=name,
        candidate_version=1,
        event_id=index + 1,
        commence_time=LONG_AGO,
        stake=1.0,
        pnl=2.0 if index % 3 == 0 else -1.0,
        price_decimal=3.0,
        p_close=(1 + ev) / 3 if p_close else None,
        placed_at=LONG_AGO + timedelta(minutes=index),
        close_grace=timedelta(0),
    )


def mechanical(
    name: str = "gap", sport: str = "soccer_epl", market: str = "1x2"
) -> MechanicalCandidate:
    return MechanicalCandidate(
        name=name,
        description="d",
        sport=sport,
        version=1,
        issue=1,
        start_date="2026-01-01",
        status=CandidateStatus.ACTIVE,
        kind="mechanical",
        market=market,
        rule="exchange_gap",
        rule_sha256="0" * 64,
    )


def agent(name: str = "mlb-agent") -> AgentCandidate:
    return AgentCandidate(
        name=name,
        description="d",
        sport="baseball_mlb",
        version=1,
        issue=1,
        start_date="2026-01-01",
        status=CandidateStatus.ACTIVE,
        kind="agent",
    )


def summaries(bets: list[ScoredBet], registry: list) -> list[CandidateSummary]:
    return summarize_candidates(bets, registry, NOW, n_boot=N_BOOT)
