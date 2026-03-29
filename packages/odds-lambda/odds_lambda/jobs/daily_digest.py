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

DEFAULT_SPORT_KEY = "soccer_epl"
EMBED_COLOR = 3066993  # Green
MAX_FIELD_CHARS = 1024


async def _get_completed_events_with_predictions(
    session: AsyncSession,
    since: datetime,
    model_name: str,
    sport_key: str = DEFAULT_SPORT_KEY,
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
                Event.sport_key == sport_key,
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
    sport_key: str = DEFAULT_SPORT_KEY,
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
                Event.sport_key == sport_key,
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


def _value_side(predicted_clv: float) -> tuple[str, float]:
    """Return (side label, absolute CLV) from a home-outcome predicted CLV.

    Positive CLV means the home price is expected to tighten (home undervalued).
    Negative CLV means the away price is expected to tighten (away undervalued).
    """
    if predicted_clv >= 0:
        return "Home", abs(predicted_clv)
    return "Away", abs(predicted_clv)


def _result_hit(predicted_clv: float, home_score: int, away_score: int) -> bool:
    """Check whether the predicted value side won the match."""
    if predicted_clv >= 0:
        return home_score > away_score
    return away_score > home_score


def _format_results_section(events: list[dict[str, Any]]) -> str:
    """Format post-match results into a Discord field value."""
    if not events:
        return ""

    lines = []
    hits = 0
    for e in events:
        side, clv_abs = _value_side(e["predicted_clv"])
        hit = _result_hit(e["predicted_clv"], e["home_score"], e["away_score"])
        if hit:
            hits += 1
        icon = "\u2705" if hit else "\u274c"
        score = f"{e['home_score']}-{e['away_score']}"
        lines.append(
            f"{icon} {e['home_team']} vs {e['away_team']} ({score}) | {side} +{clv_abs * 100:.1f}%"
        )

    lines.append(f"\n_{len(events)} events | {hits}/{len(events)} correct side_")

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
        side, clv_abs = _value_side(e["predicted_clv"])
        kickoff = e["commence_time"].strftime("%a %d %b %H:%M")
        lines.append(
            f"{e['home_team']} vs {e['away_team']} ({kickoff} UTC) | {side} +{clv_abs * 100:.1f}%"
        )

    text = "\n".join(lines)
    if len(text) > MAX_FIELD_CHARS:
        text = text[: MAX_FIELD_CHARS - 3] + "..."
    return text


def _format_window(hours: float) -> str:
    """Format a window duration for section headers."""
    if hours % 24 == 0 and hours >= 24:
        days = int(hours // 24)
        return f"{days}d" if days > 1 else "24h"
    return f"{int(hours)}h"


def _league_display_name(sport_key: str) -> str:
    """Return a human-readable league name for embed titles.

    Derives from LeagueSpec.sport_title when available, otherwise
    strips the ``soccer_`` prefix and title-cases the remainder.
    """
    from odds_lambda.jobs.fetch_oddsportal import LEAGUE_SPECS

    for spec in LEAGUE_SPECS:
        if spec.sport_key == sport_key:
            return spec.sport_title
    # Fallback: "soccer_epl" -> "Epl", "soccer_la_liga" -> "La Liga"
    name = sport_key.removeprefix("soccer_").replace("_", " ").title()
    return name


def build_digest_embed(
    results: list[dict[str, Any]],
    upcoming: list[dict[str, Any]],
    lookback_hours: float = 24,
    lookahead_hours: float = 48,
    sport_key: str = DEFAULT_SPORT_KEY,
) -> dict[str, Any]:
    """Build a Discord embed dict for the daily digest."""
    now = datetime.now(UTC)
    fields: list[dict[str, str | bool]] = []

    if results:
        fields.append(
            {
                "name": f"Post-Match Results (last {_format_window(lookback_hours)})",
                "value": _format_results_section(results),
                "inline": False,
            }
        )

    if upcoming:
        fields.append(
            {
                "name": f"Upcoming Predictions (next {_format_window(lookahead_hours)})",
                "value": _format_upcoming_section(upcoming),
                "inline": False,
            }
        )

    league = _league_display_name(sport_key)
    return {
        "title": f"{league} Daily Digest",
        "color": EMBED_COLOR,
        "fields": fields,
        "timestamp": now.isoformat(),
    }


async def send_digest(
    model_name: str | None = None,
    sport_key: str = DEFAULT_SPORT_KEY,
    lookback_hours: float = 24,
    lookahead_hours: float = 48,
) -> dict[str, int]:
    """Query predictions and results, send Discord digest.

    Args:
        model_name: Model to filter predictions by. Defaults to MODEL_NAME env var.
        sport_key: Sport/league to query events for.
        lookback_hours: How far back to look for completed events.
        lookahead_hours: How far ahead to look for upcoming events.

    Returns dict with counts: results_count, upcoming_count, sent (0 or 1).
    """
    model_name = model_name or os.environ.get("MODEL_NAME") or "epl-clv-home"

    now = datetime.now(UTC)
    since = now - timedelta(hours=lookback_hours)
    until = now + timedelta(hours=lookahead_hours)

    stats = {"results_count": 0, "upcoming_count": 0, "sent": 0}

    async with async_session_maker() as session:
        results = await _get_completed_events_with_predictions(
            session, since, model_name, sport_key
        )
        upcoming = await _get_upcoming_events_with_predictions(
            session, until, model_name, sport_key
        )

    stats["results_count"] = len(results)
    stats["upcoming_count"] = len(upcoming)

    if not results and not upcoming:
        logger.info("daily_digest_empty", reason="no predictions or results", sport_key=sport_key)
        return stats

    embed = build_digest_embed(results, upcoming, lookback_hours, lookahead_hours, sport_key)

    from odds_cli.alerts.base import AlertManager

    manager = AlertManager()
    await manager.send_embed(embed)
    stats["sent"] = 1

    logger.info("daily_digest_sent", **stats)
    return stats


async def main(
    sport_key: str = DEFAULT_SPORT_KEY,
    model_name: str | None = None,
    lookback_hours: float = 24,
    lookahead_hours: float = 48,
) -> None:
    """Main job entry point.

    Args:
        sport_key: Sport/league to query events for.
        model_name: Model to filter predictions by. Defaults to MODEL_NAME env var.
        lookback_hours: How far back to look for completed events.
        lookahead_hours: How far ahead to look for upcoming events.
    """
    logger.info(
        "daily_digest_started",
        sport_key=sport_key,
        model_name=model_name,
        lookback_hours=lookback_hours,
        lookahead_hours=lookahead_hours,
    )

    try:
        stats = await send_digest(
            model_name=model_name,
            sport_key=sport_key,
            lookback_hours=lookback_hours,
            lookahead_hours=lookahead_hours,
        )
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
