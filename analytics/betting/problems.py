"""Betting-specific prediction problems."""

from datetime import datetime

from analytics.core import DiscreteEventProblem
from core.models import Event, EventStatus


class BettingEvent(DiscreteEventProblem):
    """A betting event - any event you can bet on.

    Works for:
    - Sports games (NBA, NFL, soccer, etc.)
    - Elections
    - Prediction markets
    - Any contest with a discrete outcome

    This bridges the generic DiscreteEventProblem abstraction with the
    specific database Event model used in this codebase.
    """

    def __init__(
        self,
        event_id: str,
        home_competitor: str,
        away_competitor: str,
        commence_time: datetime,
        home_score: int | None = None,
        away_score: int | None = None,
        status: EventStatus = EventStatus.SCHEDULED,
    ):
        """Initialize a betting event.

        Args:
            event_id: Unique identifier for the event
            home_competitor: Home team/competitor name
            away_competitor: Away team/competitor name
            commence_time: When the event starts
            home_score: Home score (if event completed)
            away_score: Away score (if event completed)
            status: Event status
        """
        self.id = event_id
        self.timestamp = commence_time
        self.event_time = commence_time

        self.home_competitor = home_competitor
        self.away_competitor = away_competitor
        self.commence_time = commence_time
        self.home_score = home_score
        self.away_score = away_score
        self.status = status

        # Determine outcome based on scores
        self.outcome = self._determine_outcome()

    def _determine_outcome(self) -> str | None:
        """Determine the outcome based on scores.

        Returns:
            "home_win", "away_win", "draw", or None if scores not available
        """
        if self.home_score is None or self.away_score is None:
            return None

        if self.home_score > self.away_score:
            return "home_win"
        elif self.away_score > self.home_score:
            return "away_win"
        else:
            return "draw"

    @classmethod
    def from_db_event(cls, event: Event) -> "BettingEvent":
        """Create BettingEvent from database Event model.

        Args:
            event: Database Event instance

        Returns:
            BettingEvent instance
        """
        return cls(
            event_id=event.id,
            home_competitor=event.home_team,
            away_competitor=event.away_team,
            commence_time=event.commence_time,
            home_score=event.home_score,
            away_score=event.away_score,
            status=event.status,
        )

    def has_result(self) -> bool:
        """Check if the event has a final result.

        Returns:
            True if scores are available and event is final
        """
        return (
            self.home_score is not None
            and self.away_score is not None
            and self.status == EventStatus.FINAL
        )

    def get_winner(self) -> str | None:
        """Get the winner of the event.

        Returns:
            Team name of winner, or None if no winner (draw or incomplete)
        """
        if self.outcome == "home_win":
            return self.home_competitor
        elif self.outcome == "away_win":
            return self.away_competitor
        return None

    def get_margin(self) -> int | None:
        """Get the margin of victory.

        Returns:
            Absolute point/goal difference, or None if scores unavailable
        """
        if self.home_score is None or self.away_score is None:
            return None
        return abs(self.home_score - self.away_score)
