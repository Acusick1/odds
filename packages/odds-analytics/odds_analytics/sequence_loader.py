"""
Sequence Data Loader for LSTM Training.

This module provides utilities to fetch historical line movement data and convert it
into 3D tensor format suitable for LSTM model training and prediction.

Key Functions:
- load_sequences_for_event(): Query odds snapshots and organize chronologically
- prepare_lstm_training_data(): Process multiple events into training tensors

Architecture:
- Queries OddsSnapshot.raw_data to reconstruct historical odds
- Groups snapshots by odds_timestamp for chronological ordering
- Converts to list[list[Odds]] format expected by SequenceFeatureExtractor
- Handles variable-length sequences with attention masks
- Generates binary labels from event outcomes (home win = 1, away win = 0)

Example:
    ```python
    from odds_lambda.storage.readers import OddsReader
    from odds_analytics.sequence_loader import load_sequences_for_event, prepare_lstm_training_data

    # Load sequences for a single event
    async with get_async_session() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id("abc123")
        sequences = await load_sequences_for_event("abc123", session)

        # sequences = [
        #     [Odds(...), Odds(...), ...],  # First timestamp
        #     [Odds(...), Odds(...), ...],  # Second timestamp
        #     ...
        # ]

        # Prepare training data for multiple events
        events = await reader.get_events_by_date_range(start, end, status=EventStatus.FINAL)
        X, y, masks = await prepare_lstm_training_data(
            events=events,
            session=session,
            outcome="home",  # Predict home team wins
            market="h2h",
            lookback_hours=72,
            timesteps=24
        )

        # X: (n_events, 24, num_features)
        # y: (n_events,) - 1 if home won, 0 if away won
        # masks: (n_events, 24) - True for valid timesteps
    ```
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from enum import Enum

import numpy as np
import structlog
from odds_core.models import Event, EventStatus, Odds
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import SequenceFeatureExtractor
from odds_analytics.utils import calculate_implied_probability


class TargetType(str, Enum):
    """Target type for LSTM training."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"

logger = structlog.get_logger()

__all__ = [
    "load_sequences_for_event",
    "prepare_lstm_training_data",
    "TargetType",
]


def _extract_opening_closing_odds(
    odds_sequences: list[list[Odds]],
    outcome: str,
    market: str,
) -> tuple[list[Odds] | None, list[Odds] | None]:
    """
    Extract opening and closing odds from a sequence of snapshots.

    Opening odds are from the first snapshot, closing odds from the last.

    Args:
        odds_sequences: List of snapshots ordered chronologically
        outcome: Target outcome name (team name or Over/Under)
        market: Market type (h2h, spreads, totals)

    Returns:
        Tuple of (opening_odds, closing_odds) where each is a list of Odds
        for the target outcome/market, or None if not found.
    """
    if not odds_sequences:
        return None, None

    opening_odds = None
    closing_odds = None

    # Find opening odds (first snapshot with valid data)
    for snapshot in odds_sequences:
        filtered = [
            o for o in snapshot if o.market_key == market and o.outcome_name == outcome
        ]
        if filtered:
            opening_odds = filtered
            break

    # Find closing odds (last snapshot with valid data)
    for snapshot in reversed(odds_sequences):
        filtered = [
            o for o in snapshot if o.market_key == market and o.outcome_name == outcome
        ]
        if filtered:
            closing_odds = filtered
            break

    return opening_odds, closing_odds


def _calculate_regression_target(
    opening_odds: list[Odds] | None,
    closing_odds: list[Odds] | None,
    market: str,
) -> float | None:
    """
    Calculate the regression target (closing - opening line delta).

    For h2h market: Uses implied probability delta (closing prob - opening prob)
    For spreads/totals: Uses point delta (closing point - opening point)

    Args:
        opening_odds: Opening odds for target outcome
        closing_odds: Closing odds for target outcome
        market: Market type (h2h, spreads, totals)

    Returns:
        Line movement delta or None if cannot be calculated.
        For h2h: Probability change (positive = line moved in favor)
        For spreads/totals: Point change (closing point - opening point)
    """
    if not opening_odds or not closing_odds:
        return None

    if market == "h2h":
        # For h2h, use implied probability delta
        # Average across bookmakers if multiple
        opening_probs = [
            calculate_implied_probability(o.price) for o in opening_odds
        ]
        closing_probs = [
            calculate_implied_probability(o.price) for o in closing_odds
        ]

        if not opening_probs or not closing_probs:
            return None

        opening_prob = float(np.mean(opening_probs))
        closing_prob = float(np.mean(closing_probs))

        return closing_prob - opening_prob

    elif market in ("spreads", "totals"):
        # For spreads/totals, use point delta
        # Filter for odds that have point values
        opening_points = [o.point for o in opening_odds if o.point is not None]
        closing_points = [o.point for o in closing_odds if o.point is not None]

        if not opening_points or not closing_points:
            return None

        opening_point = float(np.mean(opening_points))
        closing_point = float(np.mean(closing_points))

        return closing_point - opening_point

    else:
        # Unknown market
        return None


async def load_sequences_for_event(
    event_id: str,
    session: AsyncSession,
) -> list[list[Odds]]:
    """
    Load odds snapshots for an event and organize them chronologically by timestamp.

    This function queries OddsSnapshot records, parses the raw_data JSON field,
    and reconstructs Odds objects grouped by their odds_timestamp. This produces
    a time series of snapshots suitable for sequence models.

    Args:
        event_id: Event identifier
        session: Async database session

    Returns:
        List of snapshots, where each snapshot is a list of Odds objects.
        Snapshots are ordered chronologically (earliest to latest).

        Example return value:
        [
            [Odds(bookmaker="pinnacle", price=-110, ...), Odds(bookmaker="fanduel", ...)],
            [Odds(bookmaker="pinnacle", price=-115, ...), Odds(bookmaker="fanduel", ...)],
            ...
        ]

    Edge Cases:
        - Empty list if no snapshots found
        - Handles missing bookmakers gracefully (snapshot may have different bookmakers)
        - Skips snapshots with no valid odds data

    Example:
        >>> async with get_async_session() as session:
        ...     sequences = await load_sequences_for_event("abc123", session)
        ...     print(f"Found {len(sequences)} snapshots")
        ...     if sequences:
        ...         print(f"First snapshot has {len(sequences[0])} odds records")
    """
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)

    # Get all snapshots for the event, ordered by time
    snapshots = await reader.get_snapshots_for_event(event_id)

    if not snapshots:
        logger.warning("no_snapshots_found", event_id=event_id)
        return []

    # Group odds by timestamp
    # Key: timestamp, Value: list of Odds objects
    odds_by_timestamp: dict[datetime, list[Odds]] = defaultdict(list)

    for snapshot in snapshots:
        # Parse raw_data to reconstruct Odds objects
        raw_data = snapshot.raw_data

        # Skip if raw_data is empty or invalid
        if not raw_data or "bookmakers" not in raw_data:
            logger.warning(
                "invalid_snapshot_data",
                event_id=event_id,
                snapshot_id=snapshot.id,
                snapshot_time=snapshot.snapshot_time,
            )
            continue

        # Extract bookmakers array from raw data
        bookmakers = raw_data.get("bookmakers", [])

        for bookmaker_data in bookmakers:
            bookmaker_key = bookmaker_data.get("key")
            bookmaker_title = bookmaker_data.get("title")
            last_update_str = bookmaker_data.get("last_update")

            # Parse last_update timestamp
            if last_update_str:
                try:
                    last_update = datetime.fromisoformat(last_update_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    last_update = snapshot.snapshot_time
            else:
                last_update = snapshot.snapshot_time

            # Process each market for this bookmaker
            markets = bookmaker_data.get("markets", [])
            for market in markets:
                market_key = market.get("key")

                # Process each outcome in the market
                outcomes = market.get("outcomes", [])
                for outcome in outcomes:
                    outcome_name = outcome.get("name")
                    price = outcome.get("price")  # American odds
                    point = outcome.get("point")  # Spread/total line

                    # Create Odds object (not persisted to DB, just for in-memory use)
                    odds = Odds(
                        event_id=event_id,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker_title,
                        market_key=market_key,
                        outcome_name=outcome_name,
                        price=price,
                        point=point,
                        odds_timestamp=snapshot.snapshot_time,
                        last_update=last_update,
                        is_valid=True,
                    )

                    # Group by snapshot_time (not last_update)
                    odds_by_timestamp[snapshot.snapshot_time].append(odds)

    # Convert to list of lists, sorted by timestamp
    sorted_timestamps = sorted(odds_by_timestamp.keys())
    sequences = [odds_by_timestamp[ts] for ts in sorted_timestamps]

    logger.info(
        "loaded_sequences",
        event_id=event_id,
        num_snapshots=len(sequences),
        timestamp_range=(
            sorted_timestamps[0].isoformat() if sorted_timestamps else None,
            sorted_timestamps[-1].isoformat() if sorted_timestamps else None,
        ),
    )

    return sequences


async def prepare_lstm_training_data(
    events: list[Event],
    session: AsyncSession,
    outcome: str = "home",
    market: str = "h2h",
    lookback_hours: int = 72,
    timesteps: int = 24,
    sharp_bookmakers: list[str] | None = None,
    retail_bookmakers: list[str] | None = None,
    target_type: TargetType | str = TargetType.CLASSIFICATION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process multiple events to generate LSTM training data.

    This function:
    1. Loads odds sequences for each event
    2. Uses SequenceFeatureExtractor to convert to feature tensors
    3. Generates labels based on target_type:
       - Classification: Binary labels from event outcomes (win/loss)
       - Regression: Continuous line movement delta (closing - opening)
    4. Creates attention masks for variable-length sequences

    Args:
        events: List of Event objects with final scores (status=FINAL required)
        session: Async database session
        outcome: What to predict - "home" (home team wins), "away" (away team wins), or team name
        market: Market to analyze (h2h, spreads, totals)
        lookback_hours: Hours before game to start sequence
        timesteps: Number of timesteps in sequence (fixed length)
        sharp_bookmakers: Sharp bookmakers for features (default: ["pinnacle"])
        retail_bookmakers: Retail bookmakers for features (default: ["fanduel", "draftkings", "betmgm"])
        target_type: Type of target to generate:
            - "classification": Binary win/loss labels (default)
            - "regression": Line movement delta (closing - opening)
              For h2h: implied probability change
              For spreads/totals: point change

    Returns:
        Tuple of (X, y, masks):
        - X: Feature array of shape (n_samples, timesteps, num_features)
        - y: Label array of shape (n_samples,)
            - Classification: int32 binary (1=win, 0=loss)
            - Regression: float32 continuous (line movement delta)
        - masks: Attention mask of shape (n_samples, timesteps) - True for valid timesteps

    Edge Cases:
        - Skips events without final scores (status != FINAL)
        - Skips events with no odds data
        - Returns empty arrays if no valid events
        - Handles variable-length sequences via attention masks
        - For regression: Skips events without opening/closing odds data

    Example:
        >>> from odds_lambda.storage.readers import OddsReader
        >>> async with get_async_session() as session:
        ...     reader = OddsReader(session)
        ...     events = await reader.get_events_by_date_range(
        ...         start_date=datetime(2024, 10, 1, tzinfo=UTC),
        ...         end_date=datetime(2024, 10, 31, tzinfo=UTC),
        ...         status=EventStatus.FINAL
        ...     )
        ...     # Classification mode (default)
        ...     X, y, masks = await prepare_lstm_training_data(
        ...         events=events,
        ...         session=session,
        ...         outcome="home",
        ...         market="h2h"
        ...     )
        ...     print(f"Training data shape: {X.shape}")
        ...     print(f"Labels shape: {y.shape}")  # Binary labels
        ...
        ...     # Regression mode
        ...     X, y, masks = await prepare_lstm_training_data(
        ...         events=events,
        ...         session=session,
        ...         outcome="home",
        ...         market="h2h",
        ...         target_type="regression"
        ...     )
        ...     print(f"Labels shape: {y.shape}")  # Continuous deltas
    """
    # Validate inputs
    if not events:
        logger.warning("no_events_provided")
        return np.array([]), np.array([]), np.array([])

    # Normalize target_type to enum
    if isinstance(target_type, str):
        target_type = TargetType(target_type.lower())

    # Filter for events with final scores
    valid_events = [
        e
        for e in events
        if e.status == EventStatus.FINAL and e.home_score is not None and e.away_score is not None
    ]

    if not valid_events:
        logger.warning("no_valid_events", total_events=len(events))
        return np.array([]), np.array([]), np.array([])

    # Initialize feature extractor
    extractor = SequenceFeatureExtractor(
        lookback_hours=lookback_hours,
        timesteps=timesteps,
        sharp_bookmakers=sharp_bookmakers,
        retail_bookmakers=retail_bookmakers,
    )

    # Get feature dimension from extractor
    feature_names = extractor.get_feature_names()
    num_features = len(feature_names)

    # Pre-allocate arrays
    n_samples = len(valid_events)
    X = np.zeros((n_samples, timesteps, num_features), dtype=np.float32)
    # Use float32 for regression, int32 for classification
    y_dtype = np.float32 if target_type == TargetType.REGRESSION else np.int32
    y = np.zeros(n_samples, dtype=y_dtype)
    masks = np.zeros((n_samples, timesteps), dtype=bool)

    # Process each event
    valid_sample_idx = 0
    skipped_events = 0

    for event in valid_events:
        # Convert Event to BacktestEvent for feature extraction
        backtest_event = BacktestEvent(
            id=event.id,
            commence_time=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            home_score=event.home_score,
            away_score=event.away_score,
            status=event.status,
        )

        # Load odds sequences for this event
        try:
            odds_sequences = await load_sequences_for_event(event.id, session)
        except Exception as e:
            logger.error(
                "failed_to_load_sequences",
                event_id=event.id,
                error=str(e),
            )
            skipped_events += 1
            continue

        # Skip if no data
        if not odds_sequences or all(len(seq) == 0 for seq in odds_sequences):
            logger.warning("empty_sequences", event_id=event.id)
            skipped_events += 1
            continue

        # Determine target outcome for feature extraction
        if outcome == "home":
            target_outcome = event.home_team
        elif outcome == "away":
            target_outcome = event.away_team
        else:
            # Assume outcome is a team name
            target_outcome = outcome

        # Extract features using SequenceFeatureExtractor
        try:
            features_dict = extractor.extract_features(
                event=backtest_event,
                odds_data=odds_sequences,
                outcome=target_outcome,
                market=market,
            )
        except Exception as e:
            logger.error(
                "failed_to_extract_features",
                event_id=event.id,
                error=str(e),
            )
            skipped_events += 1
            continue

        # Extract sequence and mask from result
        sequence = features_dict["sequence"]  # (timesteps, num_features)
        mask = features_dict["mask"]  # (timesteps,)

        # Skip if all timesteps are invalid
        if not mask.any():
            logger.warning("no_valid_timesteps", event_id=event.id)
            skipped_events += 1
            continue

        # Generate label based on target_type and outcome
        if target_type == TargetType.CLASSIFICATION:
            # Binary classification: win/loss
            if outcome == "home":
                label = 1 if event.home_score > event.away_score else 0
            elif outcome == "away":
                label = 1 if event.away_score > event.home_score else 0
            else:
                # Assume outcome is a team name
                if target_outcome == event.home_team:
                    label = 1 if event.home_score > event.away_score else 0
                else:
                    label = 1 if event.away_score > event.home_score else 0
        else:
            # Regression: line movement delta (closing - opening)
            opening_odds, closing_odds = _extract_opening_closing_odds(
                odds_sequences, target_outcome, market
            )
            regression_target = _calculate_regression_target(
                opening_odds, closing_odds, market
            )

            if regression_target is None:
                logger.warning(
                    "missing_regression_target",
                    event_id=event.id,
                    reason="could not calculate opening/closing delta",
                )
                skipped_events += 1
                continue

            label = regression_target

        # Store in arrays
        X[valid_sample_idx] = sequence
        y[valid_sample_idx] = label
        masks[valid_sample_idx] = mask

        valid_sample_idx += 1

    # Trim arrays to actual number of valid samples
    if valid_sample_idx < n_samples:
        logger.info(
            "trimming_arrays",
            original_size=n_samples,
            valid_samples=valid_sample_idx,
            skipped=skipped_events,
        )
        X = X[:valid_sample_idx]
        y = y[:valid_sample_idx]
        masks = masks[:valid_sample_idx]

    logger.info(
        "prepared_training_data",
        num_samples=valid_sample_idx,
        timesteps=timesteps,
        num_features=num_features,
        outcome=outcome,
        market=market,
        target_type=target_type.value,
        skipped_events=skipped_events,
    )

    return X, y, masks
