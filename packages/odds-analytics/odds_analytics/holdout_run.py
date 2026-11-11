"""The one holdout run of a pre-registered idea: replay the selected cell as one pooled draft
over the pre-registered leagues, each on its sealed window, and judge the pooled bets once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from odds_core.candidates import CandidateStatus, MechanicalCandidate
from odds_core.models import OddsSnapshot
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.candidate_rules.pinning import rule_sha256
from odds_lambda.candidate_rules.registry import resolve_rule
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.candidate_backtest import Coverage, collect, league_coverage, score
from odds_analytics.candidate_report import BOOTSTRAP_SEED, DEFAULT_BOOTSTRAP_SAMPLES
from odds_analytics.holdout import (
    HoldoutRefused,
    HoldoutResult,
    Preregistration,
    Window,
    exclusive_lock,
    gate,
    load_plan,
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


def pooled_draft(
    prereg: Preregistration,
    cell: dict[str, object],
    today: date,
    purpose: str,
    leagues: Sequence[str] | None = None,
) -> MechanicalCandidate:
    """The pre-registered strategy as one draft over ``leagues`` (by default all its leagues),
    with ``cell`` as params."""
    return MechanicalCandidate.model_validate(
        {
            "name": prereg.name,
            "description": f"{purpose} of #{prereg.issue}",
            "kind": "mechanical",
            "market": prereg.market,
            "sports": list(leagues) if leagues is not None else prereg.leagues,
            "version": prereg.version,
            "issue": prereg.issue,
            "start_date": today,
            "status": CandidateStatus.ACTIVE,
            "rule": prereg.rule,
            "params": cell,
        }
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
    """Run the selection at ``path`` (in ``directory/selections``) on its grid, once.

    Everything that can be checked without sealed data is checked first, so a mistake costs no
    run. The ``reserved`` ledger row then consumes the version: an error after it leaves the row
    and the run is spent.
    """
    now = now or datetime.now(UTC)
    plan = load_plan(path, directory)
    prereg = plan.prereg
    try:
        draft = pooled_draft(prereg, plan.cell, now.date(), "Holdout run", plan.holdout_leagues)
        resolve_rule(draft)
    except ValueError as exc:
        raise HoldoutRefused(str(exc)) from exc
    windows = {sport: windows_of(sport)[0] for sport in draft.sports}
    backfill_start = await earliest_backfill(session)
    with exclusive_lock(directory.parent):
        reservation = reserve(plan, directory, backfill_start, now)
        replayed = await collect(
            session,
            draft,
            {sport: [(w.start, w.end)] for sport, w in windows.items()},
            access=[reservation.access(sport) for sport in draft.sports],
        )
        pooled = score(
            draft,
            min(w.start for w in windows.values()),
            max(w.end for w in windows.values()),
            replayed,
            1,
            now,
            DEFAULT_BOOTSTRAP_SAMPLES,
            BOOTSTRAP_SEED,
        )
        row = reservation.record(
            rule_sha256(prereg.rule),
            list(windows.values()),
            now.date(),
            gate(pooled.overall),
        )
    coverage = league_coverage(replayed)
    return HoldoutRun(row, [LeagueRun(w, coverage[sport]) for sport, w in windows.items()])
