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

    IN_PLAY = "in_play"  # Game started: every 30 min (incidental collection)
    CLOSING = "closing"  # 0-3 hours before: every 30 min
    PREGAME = "pregame"  # 3-12 hours before: every 3 hours
    SHARP = "sharp"  # 12-24 hours before: every 12 hours
    EARLY = "early"  # 1-3 days before: every 24 hours
    OPENING = "opening"  # 3+ days before: every 48 hours

    @property
    def max_hours(self) -> float:
        """Upper boundary (hours before kickoff) for this tier.

        A game falls into a tier when hours_before < tier.max_hours (and
        >= the next-lower tier's max_hours).  OPENING has no upper bound
        so returns inf.
        """
        bounds: dict[FetchTier, float] = {
            FetchTier.IN_PLAY: 0.0,
            FetchTier.CLOSING: 3.0,
            FetchTier.PREGAME: 12.0,
            FetchTier.SHARP: 24.0,
            FetchTier.EARLY: 72.0,
            FetchTier.OPENING: float("inf"),
        }
        return bounds[self]

    @property
    def interval_hours(self) -> float:
        """Get interval in hours for this tier."""
        intervals = {
            FetchTier.IN_PLAY: 0.5,  # 30 minutes
            FetchTier.CLOSING: 0.5,  # 30 minutes
            FetchTier.PREGAME: 3.0,
            FetchTier.SHARP: 12.0,
            FetchTier.EARLY: 24.0,
            FetchTier.OPENING: 48.0,
        }
        return intervals[self]

    @classmethod
    def get_priority_order(cls) -> list["FetchTier"]:
        """
        Get tiers in priority order from highest to lowest priority.

        Priority is based on proximity to game start (closest = highest priority).
        Used for gap detection and backfill planning.

        Returns:
            List of FetchTier in descending priority order:
            [CLOSING, PREGAME, SHARP, EARLY, OPENING]
        """
        return [
            cls.CLOSING,  # Highest priority (closest to game start)
            cls.PREGAME,
            cls.SHARP,
            cls.EARLY,
            cls.OPENING,
            cls.IN_PLAY,  # Lowest priority (incidental collection)
        ]
