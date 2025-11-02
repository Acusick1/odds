"""Betting-specific observations."""

from datetime import datetime

from analytics.core import Observation
from core.models import Odds


class OddsObservation(Observation):
    """An odds observation - odds from a bookmaker at a specific time.

    Represents odds data for a betting market at a single point in time.
    This bridges the generic Observation abstraction with the specific
    database Odds model.
    """

    def __init__(
        self,
        event_id: str,
        observation_time: datetime,
        bookmaker: str,
        market: str,
        outcome: str,
        odds: int,
        line: float | None = None,
        last_update: datetime | None = None,
    ):
        """Initialize an odds observation.

        Args:
            event_id: ID of the event being bet on
            observation_time: When this odds snapshot was captured
            bookmaker: Bookmaker key (e.g., "fanduel", "pinnacle")
            market: Market type ("h2h", "spreads", "totals")
            outcome: Outcome name (team name or "Over"/"Under")
            odds: American odds (e.g., -110, +150)
            line: Point spread or total line (for spreads/totals markets)
            last_update: When bookmaker last updated these odds
        """
        self.problem_id = event_id
        self.observation_time = observation_time

        self.bookmaker = bookmaker
        self.market = market
        self.outcome = outcome
        self.odds = odds
        self.line = line
        self.last_update = last_update or observation_time

    @classmethod
    def from_db_odds(cls, odds: Odds) -> "OddsObservation":
        """Create OddsObservation from database Odds model.

        Args:
            odds: Database Odds instance

        Returns:
            OddsObservation instance
        """
        return cls(
            event_id=odds.event_id,
            observation_time=odds.odds_timestamp,
            bookmaker=odds.bookmaker_key,
            market=odds.market_key,
            outcome=odds.outcome_name,
            odds=odds.price,
            line=odds.point,
            last_update=odds.last_update,
        )

    def get_data(self) -> dict:
        """Return observation data as dictionary.

        Returns:
            Dictionary containing all odds information
        """
        return {
            "bookmaker": self.bookmaker,
            "market": self.market,
            "outcome": self.outcome,
            "odds": self.odds,
            "line": self.line,
            "last_update": self.last_update,
            "observation_time": self.observation_time,
        }

    def is_moneyline(self) -> bool:
        """Check if this is a moneyline (h2h) market."""
        return self.market == "h2h"

    def is_spread(self) -> bool:
        """Check if this is a spread market."""
        return self.market == "spreads"

    def is_total(self) -> bool:
        """Check if this is a totals (over/under) market."""
        return self.market == "totals"

    def get_implied_probability(self) -> float:
        """Calculate implied probability from American odds.

        Returns:
            Implied probability (0-1)
        """
        if self.odds > 0:
            # Positive odds: probability = 100 / (odds + 100)
            return 100 / (self.odds + 100)
        else:
            # Negative odds: probability = |odds| / (|odds| + 100)
            return abs(self.odds) / (abs(self.odds) + 100)


class OddsSnapshot:
    """Collection of odds observations for an event at a specific time.

    Groups all bookmaker odds for all markets at a single point in time.
    Useful for feature engineering that needs to consider multiple bookmakers.
    """

    def __init__(self, event_id: str, snapshot_time: datetime, observations: list[OddsObservation]):
        """Initialize odds snapshot.

        Args:
            event_id: Event ID
            snapshot_time: When this snapshot was taken
            observations: List of odds observations
        """
        self.event_id = event_id
        self.snapshot_time = snapshot_time
        self.observations = observations

    def get_observations_for_market(self, market: str) -> list[OddsObservation]:
        """Get all observations for a specific market.

        Args:
            market: Market type ("h2h", "spreads", "totals")

        Returns:
            List of odds observations for that market
        """
        return [obs for obs in self.observations if obs.market == market]

    def get_observations_for_bookmaker(self, bookmaker: str) -> list[OddsObservation]:
        """Get all observations from a specific bookmaker.

        Args:
            bookmaker: Bookmaker key

        Returns:
            List of odds observations from that bookmaker
        """
        return [obs for obs in self.observations if obs.bookmaker == bookmaker]

    def get_bookmakers(self) -> set[str]:
        """Get set of all bookmakers in this snapshot.

        Returns:
            Set of bookmaker keys
        """
        return {obs.bookmaker for obs in self.observations}

    def get_markets(self) -> set[str]:
        """Get set of all markets in this snapshot.

        Returns:
            Set of market types
        """
        return {obs.market for obs in self.observations}
