"""FetchTier enum for adaptive sampling tiers.

Separated into its own module to avoid circular imports.
"""

from enum import Enum


class FetchTier(Enum):
    """
    Adaptive sampling tiers based on game proximity.

    Each tier represents different data collection frequency needs:
    - CLOSING: Most critical period for line movement (0-3 hrs)
    - PREGAME: Active betting period with frequent updates (3-12 hrs)
    - SHARP: Professional betting period, Pinnacle typically opens here (12-24 hrs)
    - EARLY: Line establishment period (1-3 days)
    - OPENING: Soft book opening lines from DraftKings/FanDuel (3+ days)

    Note: Sharp books like Pinnacle typically release lines 12-24 hours before
    game time. The SHARP tier is optimized to capture this "sharp opening" with
    higher frequency sampling.
    """

    IN_PLAY = "in_play"  # Game started: every 30 min (incidental collection)
    CLOSING = "closing"  # 0-3 hours before: every 30 min
    PREGAME = "pregame"  # 3-12 hours before: every 3 hours
    SHARP = "sharp"  # 12-24 hours before: every 6 hours
    EARLY = "early"  # 1-3 days before: every 24 hours
    OPENING = "opening"  # 3+ days before: every 24 hours

    @property
    def interval_hours(self) -> float:
        """Get interval in hours for this tier."""
        intervals = {
            FetchTier.IN_PLAY: 0.5,  # 30 minutes
            FetchTier.CLOSING: 0.5,  # 30 minutes
            FetchTier.PREGAME: 3.0,
            FetchTier.SHARP: 6.0,
            FetchTier.EARLY: 24.0,
            FetchTier.OPENING: 24.0,
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
