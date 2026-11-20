"""Record what each run-candidates run found wrong with the candidate registry, and describe it
for the run's alert and the health check."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from odds_core.bet_models import CandidateRegistryCheck
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

# Discord rejects an embed description over 4096 characters, which would lose the alert.
MAX_MESSAGE_CHARS = 3800


async def record_registry_check(
    session: AsyncSession,
    sport: str,
    checked_at: datetime,
    invalid: Mapping[str, str],
    registry_error: str | None,
) -> None:
    """Replace ``sport``'s row with this run's findings. Does not commit."""
    values = {
        "sport_key": sport,
        "checked_at": checked_at,
        "registry_error": registry_error,
        "invalid_candidates": dict(invalid),
    }
    statement = insert(CandidateRegistryCheck).values(**values)
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=[CandidateRegistryCheck.__table__.c.sport_key],  # type: ignore[attr-defined]
            set_={key: statement.excluded[key] for key in values if key != "sport_key"},
        )
    )


async def failed_registry_checks(
    session: AsyncSession, sports: Sequence[str]
) -> list[CandidateRegistryCheck]:
    """The latest check of each of ``sports`` that found an invalid candidate or a registry
    error, by league."""
    rows = await session.execute(
        select(CandidateRegistryCheck)
        .where(CandidateRegistryCheck.sport_key.in_(sports))  # type: ignore[attr-defined]
        .order_by(CandidateRegistryCheck.sport_key)
    )
    return [check for check in rows.scalars() if is_failed(check)]


def is_failed(check: CandidateRegistryCheck) -> bool:
    return check.registry_error is not None or bool(check.invalid_candidates)


def candidates_alert_type(check: CandidateRegistryCheck) -> str:
    """Keyed by what failed, not the league, so a registry error or a pooled candidate that every
    league's run finds is one alert, and a newly failing candidate is not muted by an older one."""
    if check.registry_error is not None:
        return "candidates_invalid:registry"
    return "candidates_invalid:" + ",".join(sorted(check.invalid_candidates))


def describe(check: CandidateRegistryCheck) -> str:
    """The league and every failed candidate with its reason, cut at ``MAX_MESSAGE_CHARS``."""
    when = f"{check.checked_at:%Y-%m-%d %H:%M} UTC"
    if check.registry_error is not None:
        text = (
            f"run-candidates for {check.sport_key} ran no candidates at {when}: the registry "
            f"is unreadable: {check.registry_error}"
        )
    else:
        names = sorted(check.invalid_candidates)
        reasons = "; ".join(f"{name}: {check.invalid_candidates[name]}" for name in names)
        # Names come before the reasons so a truncated message still names every candidate.
        text = (
            f"run-candidates for {check.sport_key} skipped {len(names)} invalid "
            f"candidate(s) at {when} ({', '.join(names)}): {reasons}"
        )
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[: MAX_MESSAGE_CHARS - 1] + "…"
    return text
