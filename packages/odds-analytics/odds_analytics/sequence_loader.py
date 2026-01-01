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
    async with get_async_session() as session:
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
from odds_lambda.fetch_tier import FetchTier
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.utils import calculate_implied_probability


class TargetType(str, Enum):
    """Target type for LSTM training."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


logger = structlog.get_logger()

__all__ = [
    "load_sequences_for_event",
    "load_sequences_up_to_tier",
    "TargetType",
    "get_opening_closing_odds_by_tier",
    "calculate_regression_target",
]


def extract_opening_closing_odds(
    odds_sequences: list[list[Odds]],
    outcome: str,
    market: str,
    commence_time: datetime | None = None,
    opening_hours_before: float = 48.0,
    closing_hours_before: float = 0.5,
) -> tuple[list[Odds] | None, list[Odds] | None]:
    """
    Extract opening and closing odds from a sequence of snapshots.

    When commence_time is provided, finds snapshots closest to the target times:
    - Opening: snapshot closest to (commence_time - opening_hours_before)
    - Closing: snapshot closest to (commence_time - closing_hours_before)

    When commence_time is None (legacy mode), uses first/last snapshots.

    Args:
        odds_sequences: List of snapshots ordered chronologically
        outcome: Target outcome name (team name or Over/Under)
        market: Market type (h2h, spreads, totals)
        commence_time: Game start time for calculating target windows
        opening_hours_before: Hours before game for opening line (default: 48)
        closing_hours_before: Hours before game for closing line (default: 0.5)

    Returns:
        Tuple of (opening_odds, closing_odds) where each is a list of Odds
        for the target outcome/market, or None if not found.
        Returns (None, None) if opening and closing resolve to same snapshot.
    """
    if not odds_sequences:
        return None, None

    # Build list of valid snapshots with their timestamps
    valid_snapshots: list[tuple[datetime, list[Odds]]] = []
    for snapshot in odds_sequences:
        filtered = [o for o in snapshot if o.market_key == market and o.outcome_name == outcome]
        if filtered:
            # Use odds_timestamp from first odds in snapshot
            timestamp = filtered[0].odds_timestamp
            valid_snapshots.append((timestamp, filtered))

    if not valid_snapshots:
        return None, None

    # Legacy mode: use first/last snapshots
    if commence_time is None:
        if len(valid_snapshots) == 1:
            # Only one snapshot - can't calculate delta
            return None, None
        return valid_snapshots[0][1], valid_snapshots[-1][1]

    # Time-window mode: find snapshots closest to target times
    opening_target = commence_time - timedelta(hours=opening_hours_before)
    closing_target = commence_time - timedelta(hours=closing_hours_before)

    def find_closest_snapshot(
        target_time: datetime,
    ) -> tuple[int, list[Odds]] | None:
        """Find snapshot closest to target time, returning (index, odds)."""
        best_idx = None
        best_diff = float("inf")

        for idx, (timestamp, _odds) in enumerate(valid_snapshots):
            diff = abs((timestamp - target_time).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

        if best_idx is not None:
            return best_idx, valid_snapshots[best_idx][1]
        return None

    opening_result = find_closest_snapshot(opening_target)
    closing_result = find_closest_snapshot(closing_target)

    if opening_result is None or closing_result is None:
        return None, None

    opening_idx, opening_odds = opening_result
    closing_idx, closing_odds = closing_result

    # If opening and closing resolve to same snapshot, can't calculate delta
    if opening_idx == closing_idx:
        return None, None

    return opening_odds, closing_odds


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


async def get_opening_closing_odds_by_tier(
    session: AsyncSession,
    event_id: str,
    opening_tier: FetchTier,
    closing_tier: FetchTier,
    market: str,
    outcome: str,
) -> tuple[list[Odds] | None, list[Odds] | None]:
    """
    Get opening and closing odds by querying specific fetch tiers directly.

    This is more efficient and accurate than time-window based extraction because
    it uses the actual tier stored on each snapshot rather than calculating from
    timestamps.

    Args:
        session: Async database session
        event_id: Event identifier
        opening_tier: Tier for opening line (uses first snapshot in tier)
        closing_tier: Tier for closing line (uses last snapshot in tier)
        market: Market type (h2h, spreads, totals)
        outcome: Outcome name (team name or Over/Under)

    Returns:
        Tuple of (opening_odds, closing_odds) where each is a list of Odds
        for the target outcome/market, or None if not found.

    Example:
        opening, closing = await get_opening_closing_odds_by_tier(
            session=session,
            event_id="abc123",
            opening_tier=FetchTier.EARLY,
            closing_tier=FetchTier.CLOSING,
            market="h2h",
            outcome="Los Angeles Lakers",
        )
    """
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)

    # Query for first snapshot in opening tier
    opening_snapshot = await reader.get_first_snapshot_in_tier(event_id, opening_tier)
    if opening_snapshot is None:
        logger.debug(
            "no_opening_snapshot",
            event_id=event_id,
            tier=opening_tier.value,
        )
        return None, None

    # Query for last snapshot in closing tier
    closing_snapshot = await reader.get_last_snapshot_in_tier(event_id, closing_tier)
    if closing_snapshot is None:
        logger.debug(
            "no_closing_snapshot",
            event_id=event_id,
            tier=closing_tier.value,
        )
        return None, None

    # Ensure we have different snapshots
    if opening_snapshot.id == closing_snapshot.id:
        logger.debug(
            "same_opening_closing_snapshot",
            event_id=event_id,
            snapshot_id=opening_snapshot.id,
        )
        return None, None

    # Extract odds from snapshots
    opening_odds = _extract_odds_from_snapshot(opening_snapshot, event_id, market, outcome)
    closing_odds = _extract_odds_from_snapshot(closing_snapshot, event_id, market, outcome)

    if not opening_odds:
        logger.debug(
            "no_opening_odds_for_outcome",
            event_id=event_id,
            market=market,
            outcome=outcome,
            tier=opening_tier.value,
        )
        return None, None

    if not closing_odds:
        logger.debug(
            "no_closing_odds_for_outcome",
            event_id=event_id,
            market=market,
            outcome=outcome,
            tier=closing_tier.value,
        )
        return None, None

    return opening_odds, closing_odds


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


async def load_sequences_up_to_tier(
    event_id: str,
    session: AsyncSession,
    decision_tier: FetchTier,
) -> list[list[Odds]]:
    """
    Load odds snapshots for an event, filtered to include only snapshots up to decision tier.

    This prevents look-ahead bias by ensuring trajectory features are computed only
    from data that would have been available at the time of the betting decision.

    Args:
        event_id: Event identifier
        session: Async database session
        decision_tier: Maximum tier to include (snapshots at or before this tier)

    Returns:
        List of snapshots filtered to include only those at or before decision_tier.
        Each snapshot is a list of Odds objects.
        Snapshots are ordered chronologically (earliest to latest).

    Example:
        >>> # Load only snapshots up to PREGAME tier (excludes CLOSING tier data)
        >>> sequences = await load_sequences_up_to_tier(
        ...     event_id="abc123",
        ...     session=session,
        ...     decision_tier=FetchTier.PREGAME
        ... )
    """
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)

    # Get all snapshots for the event, ordered by time
    snapshots = await reader.get_snapshots_for_event(event_id)

    if not snapshots:
        logger.warning("no_snapshots_found", event_id=event_id)
        return []

    # Define tier ordering: CLOSING is closest to game, OPENING is furthest
    tier_order = FetchTier.get_priority_order()  # [CLOSING, PREGAME, SHARP, EARLY, OPENING]
    decision_tier_idx = tier_order.index(decision_tier)

    # Filter snapshots to only include those at or before decision tier
    # "Before" means further from game time, which is higher index in tier_order
    filtered_snapshots = []
    for snapshot in snapshots:
        snapshot_tier = None
        if snapshot.fetch_tier:
            try:
                snapshot_tier = FetchTier(snapshot.fetch_tier)
            except ValueError:
                # Unknown tier, include by default
                pass

        if snapshot_tier is None:
            # If tier is unknown, use hours_until_commence to determine
            if snapshot.hours_until_commence is not None:
                # Map hours to approximate tier
                hours = snapshot.hours_until_commence
                if hours > 72:  # OPENING
                    snapshot_tier = FetchTier.OPENING
                elif hours > 24:  # EARLY
                    snapshot_tier = FetchTier.EARLY
                elif hours > 12:  # SHARP
                    snapshot_tier = FetchTier.SHARP
                elif hours > 3:  # PREGAME
                    snapshot_tier = FetchTier.PREGAME
                else:  # CLOSING
                    snapshot_tier = FetchTier.CLOSING
            else:
                # No tier info - include by default (conservative)
                filtered_snapshots.append(snapshot)
                continue

        # Check if snapshot is at or before decision tier
        if snapshot_tier is not None:
            snapshot_tier_idx = tier_order.index(snapshot_tier)
            # Higher index = further from game = earlier tier
            if snapshot_tier_idx >= decision_tier_idx:
                filtered_snapshots.append(snapshot)

    if not filtered_snapshots:
        logger.warning(
            "no_snapshots_after_tier_filter",
            event_id=event_id,
            decision_tier=decision_tier.value,
        )
        return []

    # Group odds by timestamp (same logic as load_sequences_for_event)
    odds_by_timestamp: dict[datetime, list[Odds]] = defaultdict(list)

    for snapshot in filtered_snapshots:
        raw_data = snapshot.raw_data

        if not raw_data or "bookmakers" not in raw_data:
            continue

        bookmakers = raw_data.get("bookmakers", [])

        for bookmaker_data in bookmakers:
            bookmaker_key = bookmaker_data.get("key")
            bookmaker_title = bookmaker_data.get("title")
            last_update_str = bookmaker_data.get("last_update")

            if last_update_str:
                try:
                    last_update = datetime.fromisoformat(last_update_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    last_update = snapshot.snapshot_time
            else:
                last_update = snapshot.snapshot_time

            markets = bookmaker_data.get("markets", [])
            for market_data in markets:
                market_key = market_data.get("key")
                outcomes = market_data.get("outcomes", [])

                for outcome_data in outcomes:
                    outcome_name = outcome_data.get("name")
                    price = outcome_data.get("price")
                    point = outcome_data.get("point")

                    if price is None:
                        continue

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

                    odds_by_timestamp[snapshot.snapshot_time].append(odds)

    # Convert to list of lists, sorted by timestamp
    sorted_timestamps = sorted(odds_by_timestamp.keys())
    sequences = [odds_by_timestamp[ts] for ts in sorted_timestamps]

    logger.info(
        "loaded_sequences_up_to_tier",
        event_id=event_id,
        decision_tier=decision_tier.value,
        num_snapshots=len(sequences),
        total_before_filter=len(snapshots),
    )

    return sequences
