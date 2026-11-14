"""Store the runs behind an idea (a replay, a grid's cells, a holdout) so the web UI can show them.

A run is built from the result of the code that replayed it and written by ``store_runs``, which
replaces the earlier run of the same candidate version, window, range, code and grid cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from odds_core.candidate_backtest_models import CandidateBacktest, CandidateBacktestBet, WindowKind
from odds_core.candidates import MechanicalCandidate
from odds_lambda.candidate_rules.pinning import rule_sha256
from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_analytics.candidate_backtest import BacktestResult
from odds_analytics.candidate_report import ScoredBet
from odds_analytics.cell_selection import INTERVAL_LEVEL, MIN_BETS, GridSelection
from odds_analytics.holdout_run import SpentReplay, pooled_draft

# The level of a replay's and a holdout's interval (``odds_analytics.candidate_report.Checkpoint.ci95``).
REPLAY_INTERVAL_LEVEL = 0.95


@dataclass(frozen=True)
class RunRecord:
    """A run as stored: ``bets`` are the bets the run keeps, in placement order, and are empty
    for a grid cell no page shows. The run's own counts, ``stake`` and ``pnl`` cover every bet
    it placed."""

    run: CandidateBacktest
    bets: tuple[ScoredBet, ...]


def _placement_order(bets: Sequence[ScoredBet]) -> list[ScoredBet]:
    return sorted(bets, key=lambda b: (b.placed_at, b.commence_time, b.event_id))


def _run(
    candidate: MechanicalCandidate,
    kind: WindowKind,
    start: date,
    end: date,
    leagues: Sequence[str],
    git_sha: str,
    placed: Sequence[ScoredBet],
    estimate: tuple[float, float, float, float] | None,
    **fields: Any,
) -> CandidateBacktest:
    """A run over ``placed`` bets; ``estimate`` is the mean EV, its lower and upper bounds and
    the interval's level."""
    with_close = [b for b in placed if b.ev is not None]
    mean_ev, lower, upper, level = estimate or (None, None, None, None)
    return CandidateBacktest(
        candidate_name=candidate.name,
        candidate_version=candidate.version,
        issue=candidate.issue,
        market=candidate.market,
        rule=candidate.rule,
        rule_sha256=rule_sha256(candidate.rule),
        params=dict(candidate.params),
        window_kind=kind,
        start_date=start,
        end_date=end,
        leagues=list(leagues),
        git_sha=git_sha,
        events=len({b.event_id for b in with_close}),
        bets=len(placed),
        bets_with_close=len(with_close),
        stake=sum(b.stake for b in placed),
        pnl=sum(b.pnl for b in placed if b.pnl is not None),
        mean_ev=mean_ev,
        ci_lower=lower,
        ci_upper=upper,
        ci_level=level,
        **fields,
    )


def replay_record(
    candidate: MechanicalCandidate, result: BacktestResult, kind: WindowKind, git_sha: str
) -> RunRecord:
    """A replay's run: the bets it placed, estimated over every bet with a close
    (``BacktestResult.overall``)."""
    overall = result.overall
    estimate = (
        None
        if overall is None
        else (overall.mean_ev, overall.ci95[0], overall.ci95[1], REPLAY_INTERVAL_LEVEL)
    )
    bets = tuple(_placement_order(result.bets))
    run = _run(candidate, kind, result.start, result.end, candidate.sports, git_sha, bets, estimate)
    return RunRecord(run, bets)


def holdout_record(replay: SpentReplay, git_sha: str) -> RunRecord:
    """A spent holdout's run: the pooled bets of the replay on the sealed windows. The cell is
    the selected one, so the run is marked selected."""
    record = replay_record(replay.draft, replay.result, WindowKind.HOLDOUT, git_sha)
    record.run.selected = True
    return record


def grid_records(selection: GridSelection, git_sha: str, today: date) -> list[RunRecord]:
    """Every cell of a grid as a development run: its bets pooled over the development blocks,
    estimated as the selection rule scores it (the 90% interval of ``cell_selection``). Only the
    picked cell keeps its bets, since no page shows the others and a best cell picked after the
    fact would bias what a chart of it suggests."""
    prereg = selection.prereg
    blocks = selection.blocks
    records: list[RunRecord] = []
    for replay, score in zip(selection.replays, selection.scores, strict=True):
        placed = _placement_order([bet for _, bets in replay.blocks for bet in bets])
        picked = selection.picked is not None and selection.picked.index == score.index
        candidate = pooled_draft(prereg, score.cell, today, "Cell selection")
        estimate = (
            None
            if score.mean_ev is None or score.lower_bound is None or score.upper_bound is None
            else (score.mean_ev, score.lower_bound, score.upper_bound, INTERVAL_LEVEL)
        )
        run = _run(
            candidate,
            WindowKind.DEVELOPMENT,
            min(b.start for b in blocks),
            max(b.end for b in blocks),
            prereg.leagues,
            git_sha,
            placed,
            estimate,
            cell_index=score.index,
            eligible=score.eligible,
            selected=picked,
            blocks=[
                {
                    "name": block.name,
                    "start": block.start.isoformat(),
                    "end": block.end.isoformat(),
                    "leagues": list(block.leagues),
                    "bets": scored.bets,
                    "mean_ev": scored.mean_ev,
                }
                for block, scored in zip(blocks, score.blocks, strict=True)
            ],
        )
        records.append(RunRecord(run, tuple(placed) if picked else ()))
    return records


async def store_runs(session: AsyncSession, records: Sequence[RunRecord]) -> None:
    """Write ``records`` in the session's transaction, each replacing the stored run of the same
    candidate version, window kind, range, code and grid cell, bets included."""
    for record in records:
        run = record.run
        await session.execute(
            delete(CandidateBacktest).where(
                col(CandidateBacktest.candidate_name) == run.candidate_name,
                col(CandidateBacktest.candidate_version) == run.candidate_version,
                col(CandidateBacktest.window_kind) == run.window_kind,
                col(CandidateBacktest.start_date) == run.start_date,
                col(CandidateBacktest.end_date) == run.end_date,
                col(CandidateBacktest.git_sha) == run.git_sha,
                col(CandidateBacktest.cell_index).is_not_distinct_from(run.cell_index),
            )
        )
        session.add(run)
        await session.flush()
        assert run.id is not None  # noqa: S101
        if record.bets:
            await session.execute(
                insert(CandidateBacktestBet),
                [
                    {
                        "backtest_id": run.id,
                        "position": position,
                        "event_id": bet.event_id,
                        "league": bet.sport_key,
                        "kickoff": bet.commence_time,
                        "decision_time": bet.placed_at,
                        "outcome": bet.selection,
                        "line": bet.line,
                        "bookmaker": bet.bookmaker,
                        "price": bet.price_decimal,
                        "stake": bet.stake,
                        "pnl": bet.pnl,
                        "p_close": bet.p_close,
                        "ev": bet.ev,
                        "close_source": bet.close_source,
                        "estimated_ev": bet.estimated_ev,
                    }
                    for position, bet in enumerate(record.bets, 1)
                ],
            )
    await session.commit()


def stored_with_close(
    run: CandidateBacktest, bets: Sequence[CandidateBacktestBet]
) -> list[ScoredBet]:
    """A stored run's settled bets with a close, in placement order, as the report scores them."""
    scored = [
        ScoredBet(
            candidate_name=run.candidate_name,
            candidate_version=run.candidate_version,
            event_id=b.event_id,
            commence_time=b.kickoff,
            stake=b.stake,
            pnl=b.pnl,
            price_decimal=b.price,
            p_close=b.p_close,
            placed_at=b.decision_time,
            close_grace=timedelta(0),
            sport_key=b.league,
            selection=b.outcome,
            line=b.line,
            bookmaker=b.bookmaker,
            p_close_power=None,
            close_source=b.close_source,
            estimated_ev=b.estimated_ev,
        )
        for b in bets
    ]
    return [b for b in scored if b.ev is not None and b.pnl is not None]


def kill_reason(cells: Sequence[CandidateBacktest]) -> str:
    """Why no cell of a stored grid was eligible, from the cells' counts and means."""
    if all(c.bets_with_close < MIN_BETS for c in cells):
        most = max((c.bets_with_close for c in cells), default=0)
        return f"no cell reached {MIN_BETS} bets with a close (the most is {most})"
    if all(c.mean_ev is None or c.mean_ev <= 0 for c in cells):
        return "no cell had a positive mean EV"
    return (
        "no cell was eligible: too few events, or a block with enough bets had a mean EV of 0 "
        "or less"
    )
