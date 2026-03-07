"""Daily Discord digest: predictions for upcoming matches + post-match results.

Queries the Prediction table for recently completed events (last 24h) and
upcoming SCHEDULED events (next 48h), builds a Discord embed, and sends
it via AlertManager. Skips sending when both sections are empty.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.prediction_models import Prediction
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

SPORT_KEY = "soccer_epl"
EMBED_COLOR = 3066993  # Green
MAX_FIELD_CHARS = 1024


async def _get_completed_events_with_predictions(
    session: AsyncSession,
    since: datetime,
    model_name: str,
) -> list[dict[str, Any]]:
    """Get events completed since `since` that have predictions.

    For each event, returns the latest prediction (closest to kickoff).
    """
    # Subquery: latest prediction per event (max snapshot_id = closest to kickoff)
    latest_pred = (
        select(
            Prediction.event_id,
            func.max(Prediction.snapshot_id).label("max_snapshot_id"),
        )
        .where(Prediction.model_name == model_name)
        .group_by(Prediction.event_id)
        .subquery()
    )

    query = (
        select(Event, Prediction)
        .join(latest_pred, Event.id == latest_pred.c.event_id)
        .join(
            Prediction,
            and_(
                Prediction.event_id == latest_pred.c.event_id,
                Prediction.snapshot_id == latest_pred.c.max_snapshot_id,
            ),
        )
        .where(
            and_(
                Event.sport_key == SPORT_KEY,
                Event.status == EventStatus.FINAL,
                Event.completed_at >= since,
            )
        )
        .order_by(Event.commence_time)
    )

    result = await session.execute(query)
    rows = result.all()

    events = []
    for event, prediction in rows:
        events.append(
            {
                "home_team": event.home_team,
                "away_team": event.away_team,
                "home_score": event.home_score,
                "away_score": event.away_score,
                "commence_time": event.commence_time,
                "predicted_clv": prediction.predicted_clv,
            }
        )
    return events


async def _get_upcoming_events_with_predictions(
    session: AsyncSession,
    until: datetime,
    model_name: str,
) -> list[dict[str, Any]]:
    """Get SCHEDULED events before `until` that have at least one prediction.

    Returns the latest prediction per event, ranked by absolute predicted CLV.
    """
    now = datetime.now(UTC)

    latest_pred = (
        select(
            Prediction.event_id,
            func.max(Prediction.snapshot_id).label("max_snapshot_id"),
        )
        .where(Prediction.model_name == model_name)
        .group_by(Prediction.event_id)
        .subquery()
    )

    query = (
        select(Event, Prediction, OddsSnapshot.snapshot_time)
        .join(latest_pred, Event.id == latest_pred.c.event_id)
        .join(
            Prediction,
            and_(
                Prediction.event_id == latest_pred.c.event_id,
                Prediction.snapshot_id == latest_pred.c.max_snapshot_id,
            ),
        )
        .join(OddsSnapshot, OddsSnapshot.id == Prediction.snapshot_id)
        .where(
            and_(
                Event.sport_key == SPORT_KEY,
                Event.status == EventStatus.SCHEDULED,
                Event.commence_time > now,
                Event.commence_time <= until,
            )
        )
        .order_by(func.abs(Prediction.predicted_clv).desc())
    )

    result = await session.execute(query)
    rows = result.all()

    events = []
    for event, prediction, snapshot_time in rows:
        events.append(
            {
                "home_team": event.home_team,
                "away_team": event.away_team,
                "commence_time": event.commence_time,
                "predicted_clv": prediction.predicted_clv,
                "snapshot_time": snapshot_time,
            }
        )
    return events


def _format_results_section(events: list[dict[str, Any]]) -> str:
    """Format post-match results into a Discord field value."""
    if not events:
        return ""

    lines = []
    total_clv = 0.0
    for e in events:
        clv_pct = e["predicted_clv"] * 100
        total_clv += e["predicted_clv"]
        score = f"{e['home_score']}-{e['away_score']}"
        lines.append(
            f"**{e['home_team']}** vs **{e['away_team']}** ({score}) | pred CLV: {clv_pct:+.1f}%"
        )

    mean_clv = (total_clv / len(events)) * 100
    lines.append(f"\n_{len(events)} events | mean pred CLV: {mean_clv:+.1f}%_")

    text = "\n".join(lines)
    if len(text) > MAX_FIELD_CHARS:
        text = text[: MAX_FIELD_CHARS - 3] + "..."
    return text


def _format_upcoming_section(events: list[dict[str, Any]]) -> str:
    """Format upcoming predictions into a Discord field value."""
    if not events:
        return ""

    lines = []
    for e in events:
        clv_pct = e["predicted_clv"] * 100
        kickoff = e["commence_time"].strftime("%a %d %b %H:%M")
        lines.append(
            f"**{e['home_team']}** vs **{e['away_team']}** "
            f"({kickoff} UTC) | pred CLV: {clv_pct:+.1f}%"
        )

    text = "\n".join(lines)
    if len(text) > MAX_FIELD_CHARS:
        text = text[: MAX_FIELD_CHARS - 3] + "..."
    return text


def build_digest_embed(
    results: list[dict[str, Any]],
    upcoming: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a Discord embed dict for the daily digest."""
    now = datetime.now(UTC)
    fields: list[dict[str, str | bool]] = []

    if results:
        fields.append(
            {
                "name": "Post-Match Results (last 24h)",
                "value": _format_results_section(results),
                "inline": False,
            }
        )

    if upcoming:
        fields.append(
            {
                "name": "Upcoming Predictions (next 48h)",
                "value": _format_upcoming_section(upcoming),
                "inline": False,
            }
        )

    return {
        "title": "EPL Daily Digest",
        "color": EMBED_COLOR,
        "fields": fields,
        "timestamp": now.isoformat(),
    }


async def send_digest(model_name: str | None = None) -> dict[str, int]:
    """Query predictions and results, send Discord digest.

    Args:
        model_name: Model to filter predictions by. Defaults to MODEL_NAME env var.

    Returns dict with counts: results_count, upcoming_count, sent (0 or 1).
    """
    model_name = model_name or os.environ.get("MODEL_NAME") or "epl-clv-home"

    now = datetime.now(UTC)
    since = now - timedelta(hours=24)
    until = now + timedelta(hours=48)

    stats = {"results_count": 0, "upcoming_count": 0, "sent": 0}

    async with async_session_maker() as session:
        results = await _get_completed_events_with_predictions(session, since, model_name)
        upcoming = await _get_upcoming_events_with_predictions(session, until, model_name)

    stats["results_count"] = len(results)
    stats["upcoming_count"] = len(upcoming)

    if not results and not upcoming:
        logger.info("daily_digest_empty", reason="no predictions or results")
        return stats

    embed = build_digest_embed(results, upcoming)

    from odds_cli.alerts.base import AlertManager

    manager = AlertManager()
    await manager.send_embed(embed)
    stats["sent"] = 1

    logger.info("daily_digest_sent", **stats)
    return stats


async def main() -> None:
    """Main job entry point."""
    logger.info("daily_digest_started")

    try:
        stats = await send_digest()
        logger.info("daily_digest_complete", **stats)
    except Exception as e:
        logger.error("daily_digest_failed", error=str(e), exc_info=True)

        from odds_core.config import get_settings

        if get_settings().alerts.alert_enabled:
            from odds_cli.alerts.base import send_error

            await send_error(f"Daily digest job failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
