"""Game selection logic for historical data collection."""

from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger()


class GameSelector:
    """Select games strategically for historical backfill."""

    # NBA teams for distribution validation
    NBA_TEAMS = {
        "Atlanta Hawks",
        "Boston Celtics",
        "Brooklyn Nets",
        "Charlotte Hornets",
        "Chicago Bulls",
        "Cleveland Cavaliers",
        "Dallas Mavericks",
        "Denver Nuggets",
        "Detroit Pistons",
        "Golden State Warriors",
        "Houston Rockets",
        "Indiana Pacers",
        "LA Clippers",
        "Los Angeles Lakers",
        "Memphis Grizzlies",
        "Miami Heat",
        "Milwaukee Bucks",
        "Minnesota Timberwolves",
        "New Orleans Pelicans",
        "New York Knicks",
        "Oklahoma City Thunder",
        "Orlando Magic",
        "Philadelphia 76ers",
        "Phoenix Suns",
        "Portland Trail Blazers",
        "Sacramento Kings",
        "San Antonio Spurs",
        "Toronto Raptors",
        "Utah Jazz",
        "Washington Wizards",
    }

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        target_games: int = 166,
        games_per_team: int = 5,
    ):
        """
        Initialize game selector.

        Args:
            start_date: Start of date range to select from
            end_date: End of date range to select from
            target_games: Target number of games to select
            games_per_team: Target games per team for balanced distribution
        """
        self.start_date = start_date
        self.end_date = end_date
        self.target_games = target_games
        self.games_per_team = games_per_team

    def generate_sample_dates(self, days_interval: int = 3) -> list[datetime]:
        """
        Generate evenly distributed dates across the date range.

        Args:
            days_interval: Days between samples (default 3 = ~2 games/week)

        Returns:
            List of datetime objects for sampling
        """
        dates = []
        current = self.start_date

        while current <= self.end_date:
            dates.append(current)
            current += timedelta(days=days_interval)

        logger.info(
            "sample_dates_generated",
            start=self.start_date.isoformat(),
            end=self.end_date.isoformat(),
            interval_days=days_interval,
            total_dates=len(dates),
        )

        return dates

    def select_games_from_events(
        self,
        events: list[dict[str, Any]],
        max_games: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Select games from a list of events based on criteria.

        Selection criteria:
        - Balanced team distribution
        - Variety in matchups
        - Prioritize competitive games (when spread data available)

        Args:
            events: List of event dictionaries from API
            max_games: Maximum games to select (defaults to target_games)

        Returns:
            Selected subset of events
        """
        if not events:
            return []

        max_games = max_games or self.target_games

        # Track team appearances
        team_counts: dict[str, int] = {}

        # Score each game
        scored_events = []
        for event in events:
            score = self._score_event(event, team_counts)
            scored_events.append((score, event))

        # Sort by score (higher is better)
        scored_events.sort(key=lambda x: x[0], reverse=True)

        # Select top games while maintaining balance
        selected = []
        for _score, event in scored_events:
            if len(selected) >= max_games:
                break

            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")

            # Check if we should include this game (team balance)
            home_count = team_counts.get(home_team, 0)
            away_count = team_counts.get(away_team, 0)

            # Allow if neither team is over-represented
            if home_count < self.games_per_team and away_count < self.games_per_team:
                selected.append(event)
                team_counts[home_team] = home_count + 1
                team_counts[away_team] = away_count + 1

        logger.info(
            "games_selected",
            total_candidates=len(events),
            selected_count=len(selected),
            max_games=max_games,
            unique_teams=len(team_counts),
        )

        return selected

    def _score_event(
        self,
        event: dict[str, Any],
        team_counts: dict[str, int],
    ) -> float:
        """
        Score an event based on desirability for backtesting.

        Higher scores = more valuable for backtesting.

        Args:
            event: Event dictionary from API
            team_counts: Current count of games per team

        Returns:
            Score (0-100)
        """
        score = 50.0  # Base score

        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        # Penalize if teams already have many games selected
        home_count = team_counts.get(home_team, 0)
        away_count = team_counts.get(away_team, 0)

        # Heavy penalty for over-represented teams
        if home_count >= self.games_per_team:
            score -= 50
        elif home_count > 0:
            score -= home_count * 5

        if away_count >= self.games_per_team:
            score -= 50
        elif away_count > 0:
            score -= away_count * 5

        # Bonus for having bookmaker odds available
        if event.get("bookmakers") and len(event["bookmakers"]) > 0:
            score += 10

            # Bonus for multiple bookmakers (more liquid market)
            bookmaker_count = len(event["bookmakers"])
            score += min(bookmaker_count * 2, 20)

        # Small randomization to avoid always selecting same games
        import random

        score += random.uniform(-2, 2)

        return max(0, score)

    def calculate_snapshot_times(
        self,
        commence_time: datetime,
    ) -> list[datetime]:
        """
        Calculate the 5 optimal snapshot times for a game using adaptive strategy.

        Adaptive Strategy (verified with API testing):
        1. Opening line (3 days before): Initial market set, earliest value opportunities
        2. Early action (24 hours before): Most injury reports out, public betting starts
        3. Sharp action (12 hours before): Sharp bettors get their action in
        4. Pre-game (3 hours before): Final adjustments, late injury news
        5. Closing line (30 minutes before): Market consensus, critical for CLV analysis

        This captures the full arc of line movement from opening to close:
        - Opening vs closing spread (key for value analysis)
        - Sharp money movement patterns
        - Public vs sharp betting behavior
        - Line value at different decision points

        Note: API testing confirms historical data is available going back at least
        14 days before games, so 3-day opening lines are reliably available.

        Args:
            commence_time: Game start time

        Returns:
            List of 5 datetime objects for snapshots (ordered earliest to latest)
        """
        snapshots = [
            commence_time - timedelta(days=3),  # Opening line (3 days before)
            commence_time - timedelta(hours=24),  # Early action (24h before)
            commence_time - timedelta(hours=12),  # Sharp action (12h before)
            commence_time - timedelta(hours=3),  # Pre-game (3h before)
            commence_time - timedelta(minutes=30),  # Closing line (30min before)
        ]

        return snapshots

    def generate_backfill_plan(
        self,
        events_by_date: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Generate a complete backfill execution plan.

        Args:
            events_by_date: Dictionary mapping date strings to event lists

        Returns:
            Backfill plan with games and snapshot times
        """
        selected_games = []
        total_snapshots = 0

        for _date_str, events in events_by_date.items():
            # Select games from this date
            selected = self.select_games_from_events(
                events, max_games=self.target_games - len(selected_games)
            )

            for event in selected:
                commence_time_str = event.get("commence_time", "")
                if not commence_time_str:
                    continue

                # Parse commence time
                commence_time = datetime.fromisoformat(
                    commence_time_str.replace("Z", "+00:00")
                ).replace(tzinfo=None)

                # Calculate snapshot times
                snapshots = self.calculate_snapshot_times(commence_time)

                game_plan = {
                    "event_id": event.get("id"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "commence_time": commence_time,
                    "snapshots": [s.isoformat() for s in snapshots],
                    "snapshot_count": len(snapshots),
                }

                selected_games.append(game_plan)
                total_snapshots += len(snapshots)

                if len(selected_games) >= self.target_games:
                    break

            if len(selected_games) >= self.target_games:
                break

        # Calculate quota usage
        # Each snapshot costs: 10 (historical multiplier) × 1 (region) × 3 (markets) = 30
        # With 5 snapshots per game, that's 150 requests per game
        estimated_quota = total_snapshots * 30

        plan = {
            "total_games": len(selected_games),
            "total_snapshots": total_snapshots,
            "estimated_quota_usage": estimated_quota,
            "games": selected_games,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }

        logger.info(
            "backfill_plan_generated",
            total_games=len(selected_games),
            total_snapshots=total_snapshots,
            estimated_quota=estimated_quota,
        )

        return plan
