"""FetchTier enum for adaptive sampling tiers.

Separated into its own module to avoid circular imports.
"""

from enum import Enum


class FetchTier(Enum):
    """
    Adaptive sampling tiers based on game proximity.

    Each tier represents different data collection frequency needs:
    - CLOSING: Most critical period for line movement
    - PREGAME: Active betting period with frequent updates
    - SHARP: Professional betting period
    - EARLY: Opening line establishment
    - OPENING: Initial line release
    """

    CLOSING = "closing"  # 0-3 hours before: every 30 min
    PREGAME = "pregame"  # 3-12 hours before: every 3 hours
    SHARP = "sharp"  # 12-24 hours before: every 12 hours
    EARLY = "early"  # 1-3 days before: every 24 hours
    OPENING = "opening"  # 3+ days before: every 48 hours

    @property
    def interval_hours(self) -> float:
        """Get interval in hours for this tier."""
        intervals = {
            FetchTier.CLOSING: 0.5,  # 30 minutes
            FetchTier.PREGAME: 3.0,
            FetchTier.SHARP: 12.0,
            FetchTier.EARLY: 24.0,
            FetchTier.OPENING: 48.0,
        }
        return intervals[self]
