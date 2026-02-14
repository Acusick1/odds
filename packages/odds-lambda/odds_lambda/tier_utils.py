"""Utilities for calculating fetch tiers based on game timing.

This module provides reusable functions for determining fetch tiers (opening, early,
sharp, pregame, closing) based on how much time remains until a game starts. These
functions are used by both the live data collection pipeline and historical backfill
operations to ensure consistent tier classification across all data.
"""

from datetime import datetime

from odds_lambda.fetch_tier import FetchTier


def calculate_tier(hours_until: float) -> FetchTier:
    """
    Determine fetch tier based on hours until game commence time.

    The fetch tier determines how frequently odds should be collected:
    - IN_PLAY (<0 hours): Odds collected during the game
    - CLOSING (0-3 hours): Most critical period, fetch every 30 minutes
    - PREGAME (3-12 hours): Active betting period, fetch every 3 hours
    - SHARP (12-24 hours): Professional betting period, fetch every 12 hours
    - EARLY (1-3 days): Opening line establishment, fetch every 24 hours
    - OPENING (3+ days): Initial line release, fetch every 48 hours

    Args:
        hours_until: Hours until game commence time. Can be negative for
                    games that have already started.

    Returns:
        Appropriate FetchTier for the given time window.

    Example:
        >>> calculate_tier(1.5)  # 1.5 hours before game
        <FetchTier.CLOSING: 'closing'>

        >>> calculate_tier(48.0)  # 2 days before game
        <FetchTier.EARLY: 'early'>

        >>> calculate_tier(100.0)  # >3 days before game
        <FetchTier.OPENING: 'opening'>
    """
    if hours_until < 0:
        return FetchTier.IN_PLAY
    if hours_until <= 3:
        return FetchTier.CLOSING
    elif hours_until <= 12:
        return FetchTier.PREGAME
    elif hours_until <= 24:
        return FetchTier.SHARP
    elif hours_until <= 72:  # 3 days
        return FetchTier.EARLY
    else:
        return FetchTier.OPENING


def calculate_tier_from_timestamps(snapshot_time: datetime, commence_time: datetime) -> FetchTier:
    """
    Calculate fetch tier from snapshot and game commence timestamps.

    This is a convenience function that computes hours until game start and
    determines the appropriate tier. Useful for historical data where you have
    timestamps but need to determine the tier.

    Args:
        snapshot_time: When the odds snapshot was captured (timezone-aware)
        commence_time: When the game starts (timezone-aware)

    Returns:
        Appropriate FetchTier for the snapshot timing.

    Example:
        >>> from datetime import datetime, timedelta, UTC
        >>> game_time = datetime.now(UTC) + timedelta(hours=6)
        >>> snapshot_time = datetime.now(UTC)
        >>> calculate_tier_from_timestamps(snapshot_time, game_time)
        <FetchTier.PREGAME: 'pregame'>

    Note:
        Both timestamps should be timezone-aware. If they're not in the same
        timezone, the calculation will still be correct as Python handles
        timezone conversion automatically.
    """
    time_delta = commence_time - snapshot_time
    hours_until = time_delta.total_seconds() / 3600
    return calculate_tier(hours_until)


def calculate_hours_until_commence(snapshot_time: datetime, commence_time: datetime) -> float:
    """
    Calculate hours between snapshot and game commence time.

    Args:
        snapshot_time: When the odds snapshot was captured (timezone-aware)
        commence_time: When the game starts (timezone-aware)

    Returns:
        Hours until game commence (negative if game already started).

    Example:
        >>> from datetime import datetime, timedelta, UTC
        >>> now = datetime.now(UTC)
        >>> game_time = now + timedelta(hours=2.5)
        >>> calculate_hours_until_commence(now, game_time)
        2.5
    """
    time_delta = commence_time - snapshot_time
    return time_delta.total_seconds() / 3600
