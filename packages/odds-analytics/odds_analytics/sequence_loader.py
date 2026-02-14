"""
Sequence Data Loader for LSTM Training.

This module provides utilities to fetch historical line movement data and convert it
into 3D tensor format suitable for LSTM model training and prediction.

Key Functions:
- load_sequences_for_event(): Query odds snapshots and organize chronologically
- load_sequences_up_to_tier(): Query odds up to a decision tier
- get_opening_closing_odds_by_tier(): Get odds at specific tiers
- calculate_regression_target(): Calculate line movement for regression

Architecture:
- Queries OddsSnapshot.raw_data to reconstruct historical odds
- Groups snapshots by odds_timestamp for chronological ordering
- Converts to list[list[Odds]] format expected by SequenceFeatureExtractor
- Handles variable-length sequences with attention masks
- Supports tier-based filtering for decision time simulation

Example:
    ```python
    from odds_lambda.storage.readers import OddsReader
    from odds_analytics.sequence_loader import load_sequences_for_event

    # Load sequences for a single event
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id("abc123")
        sequences = await load_sequences_for_event("abc123", session)

        # sequences = [
        #     [Odds(...), Odds(...), ...],  # First timestamp
        #     [Odds(...), Odds(...), ...],  # Second timestamp
        #     ...
        # ]
    ```

For training data preparation, use the composable feature group API in feature_groups.py:
    ```python
    from odds_analytics.feature_groups import prepare_training_data
    from odds_analytics.training.config import FeatureConfig

    config = FeatureConfig(feature_groups=["sequence_full"], ...)
    result = await prepare_training_data(config, session)
    ```
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import structlog
from odds_core.models import Odds, OddsSnapshot
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.utils import calculate_implied_probability, devig_probabilities


class TargetType(str, Enum):
    """Target type for LSTM training."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


logger = structlog.get_logger()

__all__ = [
    "load_sequences_for_event",
    "TargetType",
    "calculate_regression_target",
    "extract_pinnacle_h2h_probs",
    "calculate_devigged_pinnacle_target",
    "get_snapshots_in_time_range",
]


def calculate_regression_target(
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
        opening_probs = [calculate_implied_probability(o.price) for o in opening_odds]
        closing_probs = [calculate_implied_probability(o.price) for o in closing_odds]

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


def extract_pinnacle_h2h_probs(
    odds: list[Odds],
    home_team: str,
    away_team: str,
) -> tuple[float, float] | None:
    """Extract devigged Pinnacle h2h probabilities for home and away teams.

    Filters odds to Pinnacle h2h market, finds home and away outcomes,
    converts to implied probabilities, and applies proportional devigging.

    Returns:
        (fair_home_prob, fair_away_prob) or None if Pinnacle h2h not found
        for both sides.
    """
    pinnacle_h2h = [o for o in odds if o.bookmaker_key == "pinnacle" and o.market_key == "h2h"]
    if not pinnacle_h2h:
        return None

    home_odds: Odds | None = None
    away_odds: Odds | None = None
    for o in pinnacle_h2h:
        if o.outcome_name == home_team:
            home_odds = o
        elif o.outcome_name == away_team:
            away_odds = o

    if home_odds is None or away_odds is None:
        return None

    home_raw = calculate_implied_probability(home_odds.price)
    away_raw = calculate_implied_probability(away_odds.price)
    return devig_probabilities(home_raw, away_raw)


def calculate_devigged_pinnacle_target(
    snapshot_odds: list[Odds],
    closing_odds: list[Odds],
    home_team: str,
    away_team: str,
) -> float | None:
    """Calculate target as devigged Pinnacle close minus devigged Pinnacle at snapshot.

    Returns:
        fair_close_home - fair_snapshot_home, or None if Pinnacle data missing
        on either side.
    """
    snapshot_probs = extract_pinnacle_h2h_probs(snapshot_odds, home_team, away_team)
    closing_probs = extract_pinnacle_h2h_probs(closing_odds, home_team, away_team)

    if snapshot_probs is None or closing_probs is None:
        return None

    fair_snapshot_home = snapshot_probs[0]
    fair_close_home = closing_probs[0]
    return fair_close_home - fair_snapshot_home


async def get_snapshots_in_time_range(
    session: AsyncSession,
    event_id: str,
    commence_time: datetime,
    min_hours_before: float,
    max_hours_before: float,
    max_samples: int,
) -> list[OddsSnapshot]:
    """Get stratified-sampled snapshots within a time range before game start.

    Divides [commence_time - max_hours, commence_time - min_hours] into
    max_samples equal bins and picks the snapshot nearest to each bin's
    midpoint. Bins with no snapshot are skipped.

    Returns snapshots sorted chronologically.
    """
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)
    all_snapshots = await reader.get_snapshots_for_event(event_id)

    range_start = commence_time - timedelta(hours=max_hours_before)
    range_end = commence_time - timedelta(hours=min_hours_before)

    in_range = [s for s in all_snapshots if range_start <= s.snapshot_time <= range_end]

    if not in_range:
        return []

    if len(in_range) <= max_samples:
        return in_range

    # Stratified sampling: divide range into equal bins, pick nearest to midpoint
    range_seconds = (range_end - range_start).total_seconds()
    bin_size = range_seconds / max_samples
    selected: list[OddsSnapshot] = []

    for i in range(max_samples):
        bin_mid = range_start + timedelta(seconds=(i + 0.5) * bin_size)
        best: OddsSnapshot | None = None
        best_diff = float("inf")

        for s in in_range:
            diff = abs((s.snapshot_time - bin_mid).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best = s

        if best is not None and best not in selected:
            selected.append(best)

    selected.sort(key=lambda s: s.snapshot_time)
    return selected


def _extract_odds_from_snapshot(
    snapshot: OddsSnapshot,
    event_id: str,
    market: str | None = None,
    outcome: str | None = None,
) -> list[Odds]:
    """
    Extract Odds objects from a snapshot's raw_data.

    Args:
        snapshot: OddsSnapshot with raw_data JSON
        event_id: Event identifier
        market: Market type to filter (h2h, spreads, totals). If None, includes all markets.
        outcome: Outcome name to filter (team name or Over/Under). If None, includes all outcomes.

    Returns:
        List of Odds objects, optionally filtered by market and/or outcome
    """
    raw_data = snapshot.raw_data
    if not raw_data or "bookmakers" not in raw_data:
        return []

    odds_list: list[Odds] = []
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
        for market_data in markets:
            market_key = market_data.get("key")
            if market is not None and market_key != market:
                continue

            # Process each outcome in the market
            outcomes = market_data.get("outcomes", [])
            for outcome_data in outcomes:
                outcome_name = outcome_data.get("name")
                if outcome is not None and outcome_name != outcome:
                    continue

                price = outcome_data.get("price")
                point = outcome_data.get("point")

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
                odds_list.append(odds)

    return odds_list


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
        >>> async with async_session_maker() as session:
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
