"""Mark an MLB event that has no game in MLB's schedule, and undo it when it gets a ``gamePk``."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from odds_core.models import Event, EventExternalId, EventStatus, HealthCheckAcknowledgement
from odds_core.vocabulary import HealthCheck, Source
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.mlb_game_resolver import MLB_SPORT_KEY


async def holds_game_pk(session: AsyncSession, event: Event) -> bool:
    """Whether the event holds a ``gamePk``, in ``mlb_game_pk`` or as an external id."""
    return bool(await events_holding_game_pks(session, [event]))


async def events_holding_game_pks(session: AsyncSession, events: Sequence[Event]) -> set[str]:
    """The ids of ``events`` that hold a ``gamePk``, in ``mlb_game_pk`` or as an external id."""
    holding = {e.id for e in events if e.mlb_game_pk is not None}
    by_key = {e.surrogate_id: e.id for e in events}
    result = await session.execute(
        select(EventExternalId.event_id).where(
            EventExternalId.source == Source.MLB_STATS_API,
            EventExternalId.event_id.in_(list(by_key)),  # type: ignore[attr-defined]
        )
    )
    holding.update(by_key[key] for key in result.scalars())
    return holding


async def mark_unmatched(session: AsyncSession, event: Event, reason: str) -> None:
    """Record that an MLB event has no game in MLB's schedule: its status becomes ``UNMATCHED``
    and ``reason`` is stored as its ``missing_mlb_game_pk`` health-check acknowledgement,
    replacing an earlier one. An event whose status cannot become ``UNMATCHED`` (a final one)
    keeps its status and still gets the acknowledgement. Flushes; the caller commits.

    Raises:
        ValueError: for an empty reason, an event that is not stored, not an MLB event, or one
            that holds a ``gamePk``.
    """
    reason = reason.strip()
    if not reason:
        raise ValueError("a reason is required")
    if event.surrogate_id is None:
        raise ValueError(f"event {event.id} is not stored")
    if event.sport_key != MLB_SPORT_KEY:
        raise ValueError(f"{event.id} is a {event.sport_key} event")
    if await holds_game_pk(session, event):
        raise ValueError(f"{event.id} holds a gamePk")
    held = (
        await session.execute(
            select(HealthCheckAcknowledgement).where(
                HealthCheckAcknowledgement.event_id == event.surrogate_id,
                HealthCheckAcknowledgement.check_name == HealthCheck.MISSING_MLB_GAME_PK,
            )
        )
    ).scalar_one_or_none()
    if held is None:
        session.add(
            HealthCheckAcknowledgement(
                event_id=event.surrogate_id,
                check_name=HealthCheck.MISSING_MLB_GAME_PK,
                reason=reason,
            )
        )
    else:
        held.reason = reason
        held.acknowledged_at = datetime.now(UTC)
    if event.status.can_become(EventStatus.UNMATCHED):
        event.status = EventStatus.UNMATCHED
        event.updated_at = datetime.now(UTC)
    await session.flush()


async def clear_unmatched(session: AsyncSession, event: Event) -> None:
    """Undo ``mark_unmatched`` once the event has a game: the acknowledgement is deleted and an
    ``UNMATCHED`` event becomes ``SCHEDULED``. Does not flush."""
    if event.surrogate_id is not None:
        await session.execute(
            delete(HealthCheckAcknowledgement).where(
                HealthCheckAcknowledgement.event_id == event.surrogate_id,
                HealthCheckAcknowledgement.check_name == HealthCheck.MISSING_MLB_GAME_PK,
            )
        )
    if event.status is EventStatus.UNMATCHED:
        event.status = EventStatus.SCHEDULED
        event.updated_at = datetime.now(UTC)
