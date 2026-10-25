"""When the Betfair job's runs fall due: before an event's kickoff, and before an earlier Betfair start."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from odds_core import database
from odds_core.models import Event, EventStatus, OddsSnapshot
from sqlalchemy import func, select

from odds_lambda.betfair.constants import LIVE_REQUEST_ID_PREFIX

# Betfair can turn a market in play at its own start, before the event's kickoff (an hour early
# for the 2026-10-24 World Series game). A start further ahead of the kickoff than this is taken
# as a mismatched listing, not the market's real start.
MAX_BETFAIR_START_LEAD = timedelta(hours=3)


def parse_betfair_start(raw: object) -> datetime | None:
    """A stored ``betfair_meta.market_start_time``, or None when it is missing or malformed."""
    if not isinstance(raw, str):
        return None
    try:
        start = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return start if start.tzinfo is not None else start.replace(tzinfo=UTC)


def next_close_target(
    commence: datetime, betfair_start: datetime | None, now: datetime
) -> datetime:
    """The event's Betfair start while it is still ahead and at most ``MAX_BETFAIR_START_LEAD``
    before ``commence``, else ``commence``."""
    if (
        betfair_start is not None
        and now < betfair_start < commence
        and commence - betfair_start <= MAX_BETFAIR_START_LEAD
    ):
        return betfair_start
    return commence


async def next_betfair_kickoff(sport_key: str, *, now: datetime) -> datetime | None:
    """Earliest time after ``now`` that a run must precede: a scheduled event's ``commence_time``,
    or an earlier Betfair start from the event's latest stored Betfair snapshot.

    Only events starting within ``MAX_BETFAIR_START_LEAD`` of the first upcoming one can have a
    Betfair start before it, so only their snapshots are read.
    """
    upcoming = (
        Event.sport_key == sport_key,
        Event.status == EventStatus.SCHEDULED,
        Event.commence_time > now,  # type: ignore[operator]
    )
    async with database.async_session_maker() as session:
        first = (
            await session.execute(select(func.min(Event.commence_time)).where(*upcoming))
        ).scalar_one_or_none()
        if first is None:
            return None
        events = (
            await session.execute(
                select(Event.id, Event.commence_time).where(
                    *upcoming,
                    Event.commence_time <= first + MAX_BETFAIR_START_LEAD,  # type: ignore[operator]
                )
            )
        ).all()
        start_path = OddsSnapshot.raw_data[  # type: ignore[index]
            ("bookmakers", 0, "betfair_meta", "market_start_time")
        ].as_string()
        latest = (
            await session.execute(
                select(OddsSnapshot.event_id, start_path)
                .where(
                    OddsSnapshot.event_id.in_([event_id for event_id, _ in events]),  # type: ignore[attr-defined]
                    OddsSnapshot.api_request_id.like(f"{LIVE_REQUEST_ID_PREFIX}%"),  # type: ignore[union-attr]
                )
                .distinct(OddsSnapshot.event_id)
                .order_by(OddsSnapshot.event_id, OddsSnapshot.snapshot_time.desc())  # type: ignore[attr-defined]
            )
        ).all()
    starts = {event_id: parse_betfair_start(raw) for event_id, raw in latest}
    return min(
        next_close_target(commence, starts.get(event_id), now) for event_id, commence in events
    )
