"""Score upcoming events against a trained CLV model and store predictions.

Loads a model from S3, extracts features for each unscored snapshot of
upcoming SCHEDULED events, and writes Prediction rows. Safe to re-run:
the unique constraint on (event_id, snapshot_id, model_name) prevents
duplicate predictions via ON CONFLICT DO NOTHING.

Called inline at the end of fetch-oddsportal to score newly arrived data.
The ``main()`` entry point is retained for manual invocation via
``odds scheduler run-once score-predictions-<sport>`` (e.g. for backfills
or ad-hoc scoring runs outside the scrape loop).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import numpy as np
import structlog
from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import TabularFeatureExtractor
from odds_analytics.feature_groups import resolve_outcome_name
from odds_analytics.training.config import FeatureConfig
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.prediction_models import Prediction
from odds_core.snapshot_utils import extract_odds_from_snapshot
from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.model_loader import get_cached_version, load_model
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name

logger = structlog.get_logger()


async def _get_upcoming_events(session: AsyncSession, sport_key: str) -> list[Event]:
    """Get SCHEDULED events for a sport with commence_time in the future."""
    query = select(Event).where(
        and_(
            Event.sport_key == sport_key,
            Event.status == EventStatus.SCHEDULED,
            Event.commence_time > datetime.now(UTC),
        )
    )
    result = await session.execute(query)
    return list(result.scalars().all())


async def _get_unscored_snapshots(
    session: AsyncSession,
    event_id: str,
    model_name: str,
) -> list[OddsSnapshot]:
    """Get snapshots for an event that haven't been scored by this model yet."""
    scored_subq = (
        select(Prediction.snapshot_id)
        .where(
            and_(
                Prediction.event_id == event_id,
                Prediction.model_name == model_name,
            )
        )
        .scalar_subquery()
    )
    query = (
        select(OddsSnapshot)
        .where(
            and_(
                OddsSnapshot.event_id == event_id,
                OddsSnapshot.id.notin_(scored_subq),
            )
        )
        .order_by(OddsSnapshot.snapshot_time)
    )
    result = await session.execute(query)
    return list(result.scalars().all())


def _extract_features(
    event: Event,
    snapshot: OddsSnapshot,
    config: FeatureConfig,
    expected_features: list[str],
) -> np.ndarray | None:
    """Extract feature vector for a single snapshot.

    Returns None if feature extraction fails (e.g. no odds data in snapshot)
    or if the produced feature names don't match what the model expects.
    """
    market = config.primary_market
    outcome_name = resolve_outcome_name(config, event)

    odds = extract_odds_from_snapshot(snapshot, event.id, market=market)
    if not odds:
        return None

    backtest_event = BacktestEvent(
        id=event.id,
        commence_time=event.commence_time,
        home_team=event.home_team,
        away_team=event.away_team,
        home_score=0,
        away_score=0,
        status=event.status,
    )

    extractor = TabularFeatureExtractor.from_config(config)
    tab_feats = extractor.extract_features(
        event=backtest_event,
        odds_data=odds,
        outcome=outcome_name,
        market=market,
    )
    tab_array = tab_feats.to_array()

    hours_until = (event.commence_time - snapshot.snapshot_time).total_seconds() / 3600
    feature_vector = np.concatenate([tab_array, np.array([hours_until])])
    feature_vector = np.nan_to_num(feature_vector, nan=0.0).astype(np.float32)

    # Verify feature alignment — mismatched features produce garbage predictions
    produced_names = [f"tab_{n}" for n in extractor.get_feature_names()] + ["hours_until_event"]
    if produced_names != expected_features:
        logger.error(
            "feature_name_mismatch",
            expected=expected_features,
            produced=produced_names,
        )
        return None

    return feature_vector


async def score_events(
    model_name: str | None = None,
    bucket: str | None = None,
    config: FeatureConfig | None = None,
    sport: str | None = None,
) -> dict[str, int]:
    """Score all unscored snapshots for upcoming events.

    Uses INSERT ... ON CONFLICT DO NOTHING for safe concurrent execution.

    Args:
        model_name: S3 model name. Defaults to ``settings.model.name``
            (``MODEL_NAME`` env var).
        bucket: S3 bucket. Defaults to ``settings.model.bucket``
            (``MODEL_BUCKET`` env var).
        config: Override feature config (mainly for testing). If None, uses
            the config bundled in the model artifact.
        sport: Sport key from event payload. When provided, validated against
            the model's bundled sport_key to prevent mismatched scoring.

    Returns:
        Dict with counts: events_checked, snapshots_scored, snapshots_skipped, errors.
    """
    from odds_core.config import get_settings

    model_name = model_name or get_settings().model.name

    stats = {"events_checked": 0, "snapshots_scored": 0, "snapshots_skipped": 0, "errors": 0}

    try:
        model_data = load_model(model_name=model_name, bucket=bucket)
    except (ValueError, FileNotFoundError) as e:
        logger.error("model_load_failed", error=str(e))
        return stats

    model = model_data["model"]
    feature_names: list[str] = model_data["feature_names"]
    model_version = get_cached_version() or "unknown"

    bundled_config: FeatureConfig | None = model_data.get("feature_config")
    config = config or bundled_config
    if config is None:
        logger.error(
            "no_feature_config", msg="No config provided and none bundled in model artifact"
        )
        return stats

    sport_key = config.sport_key
    if not sport_key:
        logger.error(
            "no_sport_key",
            msg="FeatureConfig has no sport_key — cannot determine which events to score",
        )
        return stats

    # Validate that requested sport matches the model's sport
    if sport and sport != sport_key:
        logger.error(
            "sport_mismatch",
            requested_sport=sport,
            model_sport=sport_key,
            model_name=model_name,
            msg="Requested sport does not match model's sport_key",
        )
        return stats

    async with async_session_maker() as session:
        events = await _get_upcoming_events(session, sport_key)
        stats["events_checked"] = len(events)

        if not events:
            logger.info("no_upcoming_events", sport_key=sport_key)
            return stats

        for event in events:
            snapshots = await _get_unscored_snapshots(session, event.id, model_name)

            for snapshot in snapshots:
                try:
                    features = _extract_features(event, snapshot, config, feature_names)
                    if features is None:
                        stats["snapshots_skipped"] += 1
                        continue

                    predicted_clv = float(model.predict(features.reshape(1, -1))[0])

                    stmt = (
                        pg_insert(Prediction)
                        .values(
                            event_id=event.id,
                            snapshot_id=snapshot.id,
                            model_name=model_name,
                            model_version=model_version,
                            predicted_clv=predicted_clv,
                            created_at=datetime.now(UTC),
                        )
                        .on_conflict_do_nothing(
                            constraint="uq_prediction_event_snap_model",
                        )
                    )
                    result = await session.execute(stmt)
                    if result.rowcount:
                        stats["snapshots_scored"] += 1

                except Exception:
                    stats["errors"] += 1
                    logger.error(
                        "snapshot_scoring_failed",
                        event_id=event.id,
                        snapshot_id=snapshot.id,
                        exc_info=True,
                    )

        await session.commit()

    logger.info("score_predictions_complete", **stats)
    return stats


async def main(ctx: JobContext) -> None:
    """Main job entry point."""
    from odds_core.alerts import job_alert_context

    sport = ctx.sport

    async with job_alert_context(make_compound_job_name("score-predictions", sport)):
        logger.info("score_predictions_started", sport=sport)
        await score_events(sport=sport)


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
