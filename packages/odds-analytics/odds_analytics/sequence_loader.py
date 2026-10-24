"""
Sequence Data Loader and Odds Utilities.

Key Functions:
- load_sequences_for_event(): Query odds snapshots and organize chronologically
- calculate_regression_target(): Calculate line movement for regression
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np
import structlog
from odds_core.models import Odds
from odds_core.odds_math import outcome_price_decimal
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

__all__ = [
    "load_sequences_for_event",
    "calculate_regression_target",
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
        opening_probs = [o.implied_probability for o in opening_odds]
        closing_probs = [o.implied_probability for o in closing_odds]

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
            [Odds(bookmaker="pinnacle", price_decimal=1.91, ...), Odds(bookmaker="fanduel", ...)],
            [Odds(bookmaker="pinnacle", price_decimal=1.87, ...), Odds(bookmaker="fanduel", ...)],
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
                    point = outcome.get("point")  # Spread/total line

                    # Create Odds object (not persisted to DB, just for in-memory use)
                    odds = Odds(
                        event_id=event_id,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker_title,
                        market_key=market_key,
                        outcome_name=outcome_name,
                        price_decimal=outcome_price_decimal(outcome),
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
