"""Prediction problem abstractions.

Defines what we're trying to predict - can be discrete events or continuous values.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class PredictionProblem(ABC):
    """Base class for all prediction problems.

    A prediction problem represents something we want to predict - either a discrete
    outcome (e.g., which team wins) or a continuous value (e.g., stock return).

    This abstraction works for:
    - Sports betting (discrete outcomes: win/loss)
    - Election markets (discrete outcomes: candidates)
    - Stock trading (continuous: returns, prices)
    - Weather forecasting (continuous: temperature, rain amount)
    - Any time series prediction task
    """

    id: str  # Unique identifier for this problem
    timestamp: datetime  # Primary reference time for this problem

    @abstractmethod
    def get_outcome(self) -> Any:
        """Return the actual outcome/result that occurred.

        Returns:
            The realized outcome (type depends on problem type)
        """
        pass

    @abstractmethod
    def get_problem_type(self) -> str:
        """Return a string identifying the problem type.

        Returns:
            String like "discrete_event" or "continuous_prediction"
        """
        pass


class DiscreteEventProblem(PredictionProblem):
    """Prediction problem for discrete events with categorical outcomes.

    Examples:
    - Sports game: outcome is which team won
    - Election: outcome is which candidate won
    - Contest/competition: outcome is a winner or result category

    Characteristics:
    - Events happen at a specific time
    - Outcomes are categorical (not continuous numbers)
    - Clear start and end
    """

    event_time: datetime  # When the event occurs/completes
    outcome: Any | None = None  # The actual outcome (set after event completes)

    def get_outcome(self) -> Any:
        """Return the event's outcome."""
        return self.outcome

    def get_problem_type(self) -> str:
        """Return problem type identifier."""
        return "discrete_event"


class ContinuousPredictionProblem(PredictionProblem):
    """Prediction problem for continuous values over time windows.

    Examples:
    - Stock returns: predict % return over next N hours/days
    - Price movements: predict price at a future time
    - Metrics: predict continuous value (temperature, demand, etc.)

    Characteristics:
    - Predictions are for time windows (start -> end)
    - Outcomes are continuous numeric values
    - Can have multiple prediction windows for same asset/entity
    """

    prediction_window_start: datetime  # Start of prediction window
    prediction_window_end: datetime  # End of prediction window
    realized_value: float | None = None  # The actual value that occurred

    def get_outcome(self) -> float | None:
        """Return the realized value."""
        return self.realized_value

    def get_problem_type(self) -> str:
        """Return problem type identifier."""
        return "continuous_prediction"
