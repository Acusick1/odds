"""Observation abstractions.

Observations are time-stamped data points about a prediction problem.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class Observation(ABC):
    """Base class for all observations.

    An observation is a data point captured at a specific time about a prediction problem.

    Examples:
    - Betting: odds snapshot from a bookmaker at a specific time
    - Trading: price/volume data at a specific time
    - Weather: temperature reading at a specific time

    The key property is the timestamp - observations must be ordered in time
    to prevent look-ahead bias in feature extraction.
    """

    problem_id: str  # Links observation to the prediction problem
    observation_time: datetime  # When this observation was captured

    @abstractmethod
    def get_data(self) -> dict[str, Any]:
        """Return the observation data as a dictionary.

        Returns:
            Dictionary containing the observation data. Keys and values
            depend on the observation type.

        Example:
            For odds: {'bookmaker': 'fanduel', 'odds': -110, ...}
            For price: {'price': 150.50, 'volume': 1000000, ...}
        """
        pass

    def is_before(self, time: datetime) -> bool:
        """Check if this observation occurred before a given time.

        Args:
            time: The time to compare against

        Returns:
            True if observation_time is before the given time

        This is critical for preventing look-ahead bias - we can only use
        observations that occurred before our decision time.
        """
        return self.observation_time < time

    def is_at_or_before(self, time: datetime) -> bool:
        """Check if this observation occurred at or before a given time.

        Args:
            time: The time to compare against

        Returns:
            True if observation_time is at or before the given time
        """
        return self.observation_time <= time
