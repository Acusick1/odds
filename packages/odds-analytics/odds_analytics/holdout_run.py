"""The one holdout run of a pre-registered idea: replay the selected cell on each league's sealed
window, pool the bets and judge them once.

Each league is replayed as a single-sport draft; pooling the bets into one sample and one gate
is the seam a pooled draft would replace.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path

from odds_core.candidates import CandidateStatus, MechanicalCandidate
from odds_core.models import OddsSnapshot
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.candidate_rules import resolve_rule
from odds_lambda.candidate_rules.pinning import rule_sha256
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.candidate_backtest import Coverage, Replayed, collect, coverage_of, score
from odds_analytics.candidate_report import (
    BOOTSTRAP_SEED,
    DEFAULT_BOOTSTRAP_SAMPLES,
    ScoredBet,
)
from odds_analytics.holdout import (
    HoldoutRefused,
    HoldoutResult,
    Preregistration,
    Window,
    exclusive_lock,
    gate,
    load_preregistration,
    reserve,
    windows_of,
)


@dataclass(frozen=True)
class LeagueRun:
    window: Window
    coverage: Coverage


@dataclass(frozen=True)
class HoldoutRun:
    row: HoldoutResult
    leagues: list[LeagueRun]


def league_draft(prereg: Preregistration, sport: str, today: date) -> MechanicalCandidate:
    """The pre-registered strategy as a draft of one league, with the selected cell's params."""
    return MechanicalCandidate.model_validate(
        {
            "name": prereg.name,
            "description": f"Holdout run of #{prereg.issue}",
            "kind": "mechanical",
            "market": prereg.market,
            "sport": sport,
            "version": prereg.version,
            "issue": prereg.issue,
            "start_date": today,
            "status": CandidateStatus.ACTIVE,
            "rule": prereg.rule,
            "params": prereg.cell,
        }
    )


def _pooled_coverage(coverages: list[Coverage]) -> Coverage:
    return Coverage(
        **{f.name: sum(getattr(c, f.name) for c in coverages) for f in fields(Coverage)}
    )


async def earliest_backfill(session: AsyncSession) -> datetime | None:
    """When the first Odds API backfill snapshot (``source_stamped``) was stored, if any."""
    result = await session.execute(
        select(func.min(OddsSnapshot.created_at)).where(
            OddsSnapshot.source == Source.ODDS_API,
            OddsSnapshot.time_basis == TimeBasis.SOURCE_STAMPED,
        )
    )
    return result.scalar_one_or_none()


async def run_holdout(
    session: AsyncSession, path: Path, directory: Path, now: datetime | None = None
) -> HoldoutRun:
    """Run the pre-registration at ``path`` (in ``directory/preregistrations``) once.

    Everything that can be checked without sealed data is checked first, so a mistake costs no
    run. The ``reserved`` ledger row then consumes the version: an error after it leaves the row
    and the run is spent.
    """
    now = now or datetime.now(UTC)
    prereg = load_preregistration(path, directory)
    drafts = {sport: league_draft(prereg, sport, now.date()) for sport in prereg.leagues}
    for draft in drafts.values():
        try:
            resolve_rule(draft)
        except ValueError as exc:
            raise HoldoutRefused(str(exc)) from exc
    backfill_start = await earliest_backfill(session)
    with exclusive_lock(directory.parent):
        reservation = reserve(prereg, path, directory, backfill_start, now)
        replays: list[tuple[Window, Replayed]] = []
        for sport, draft in drafts.items():
            (window,) = windows_of(sport)
            replays.append(
                (
                    window,
                    await collect(
                        session,
                        draft,
                        window.start,
                        window.end,
                        access=reservation.access(sport),
                    ),
                )
            )
        bets: list[ScoredBet] = sorted(
            (b for _, r in replays for b in r.bets), key=lambda b: b.placed_at
        )
        first = next(iter(drafts.values()))
        pooled = score(
            first,
            min(w.start for w, _ in replays),
            max(w.end for w, _ in replays),
            bets,
            _pooled_coverage([coverage_of(r) for _, r in replays]),
            1,
            now,
            DEFAULT_BOOTSTRAP_SAMPLES,
            BOOTSTRAP_SEED,
        )
        row = reservation.record(
            rule_sha256(prereg.rule),
            [w for w, _ in replays],
            now.date(),
            gate(pooled.overall),
        )
    return HoldoutRun(row, [LeagueRun(w, coverage_of(r)) for w, r in replays])
