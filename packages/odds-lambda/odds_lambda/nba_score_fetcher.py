"""NBA score fetcher using nba_api for historical data backfill."""

from datetime import UTC, datetime

import structlog
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamefinder

logger = structlog.get_logger()


class NBAScoreFetcher:
    """Fetches NBA game scores using nba_api for historical backfill."""

    def __init__(self):
        """Initialize NBA score fetcher."""
        pass

    def get_live_scores(self) -> list[dict]:
        """
        Fetch live scores from NBA API.

        Returns:
            List of game dictionaries with scores

        Example:
            {
                "game_id": "0022400123",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "home_score": 110,
                "away_score": 105,
                "game_status": "Final",
                "game_date": datetime(2024, 1, 15)
            }
        """
        try:
            board = scoreboard.ScoreBoard()
            games_data = board.get_dict()

            results = []
            for game in games_data.get("scoreboard", {}).get("games", []):
                game_id = game.get("gameId")
                home_team = game.get("homeTeam", {})
                away_team = game.get("awayTeam", {})

                # Get game date
                game_time_utc = game.get("gameTimeUTC")
                game_date = None
                if game_time_utc:
                    try:
                        game_date = datetime.fromisoformat(
                            game_time_utc.replace("Z", "+00:00")
                        ).replace(tzinfo=UTC)
                    except Exception:
                        pass

                result = {
                    "game_id": game_id,
                    "home_team": home_team.get("teamTricode", ""),
                    "away_team": away_team.get("teamTricode", ""),
                    "home_score": home_team.get("score", 0),
                    "away_score": away_team.get("score", 0),
                    "game_status": game.get("gameStatusText", ""),
                    "game_date": game_date,
                }
                results.append(result)

            logger.info("live_scores_fetched", games_count=len(results))
            return results

        except Exception as e:
            logger.error("live_scores_fetch_failed", error=str(e), exc_info=True)
            raise

    def get_historical_scores(
        self, start_date: datetime, end_date: datetime, team: str | None = None
    ) -> list[dict]:
        """
        Fetch historical game scores for a date range.

        Args:
            start_date: Start date for search (UTC)
            end_date: End date for search (UTC)
            team: Optional team abbreviation to filter (e.g., "LAL")

        Returns:
            List of game dictionaries with scores

        Example:
            {
                "game_id": "0022300456",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "home_score": 110,
                "away_score": 105,
                "game_date": datetime(2024, 1, 15, tzinfo=UTC)
            }
        """
        try:
            # nba_api requires date format YYYY-MM-DD
            date_from = start_date.strftime("%Y-%m-%d")
            date_to = end_date.strftime("%Y-%m-%d")

            # Create game finder
            game_finder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=date_from,
                date_to_nullable=date_to,
                team_id_nullable=team if team else None,
            )

            # Get games dataframe
            games_df = game_finder.get_data_frames()[0]

            results = []
            # Group by GAME_ID to get both teams' data
            for game_id, game_group in games_df.groupby("GAME_ID"):
                if len(game_group) < 2:
                    # Skip incomplete game data
                    continue

                # Sort by MATCHUP to determine home/away
                # Home team has "vs." in MATCHUP, away has "@"
                home_row = game_group[game_group["MATCHUP"].str.contains("vs.")]
                away_row = game_group[game_group["MATCHUP"].str.contains("@")]

                if home_row.empty or away_row.empty:
                    continue

                home_row = home_row.iloc[0]
                away_row = away_row.iloc[0]

                # Parse game date
                game_date_str = str(home_row["GAME_DATE"])
                try:
                    game_date = datetime.strptime(game_date_str, "%Y-%m-%d").replace(tzinfo=UTC)
                except Exception:
                    game_date = None

                result = {
                    "game_id": str(game_id),
                    "home_team": str(home_row["TEAM_ABBREVIATION"]),
                    "away_team": str(away_row["TEAM_ABBREVIATION"]),
                    "home_score": int(home_row["PTS"]) if home_row["PTS"] else 0,
                    "away_score": int(away_row["PTS"]) if away_row["PTS"] else 0,
                    "game_date": game_date,
                }
                results.append(result)

            logger.info(
                "historical_scores_fetched",
                games_count=len(results),
                date_from=date_from,
                date_to=date_to,
            )
            return results

        except Exception as e:
            logger.error(
                "historical_scores_fetch_failed",
                error=str(e),
                date_from=date_from,
                date_to=date_to,
                exc_info=True,
            )
            raise

    def match_game_by_teams_and_date(
        self, home_team: str, away_team: str, game_date: datetime, tolerance_hours: int = 24
    ) -> dict | None:
        """
        Find a game by team names and date with fuzzy matching.

        This is useful for matching events in our database (which use The Odds API team names)
        with NBA API game data (which uses official NBA team names/abbreviations).

        Args:
            home_team: Home team name (can be full name or abbreviation)
            away_team: Away team name (can be full name or abbreviation)
            game_date: Expected game date
            tolerance_hours: Search window around game_date (default 24 hours)

        Returns:
            Game dictionary if found, None otherwise
        """
        # Search within tolerance window
        from datetime import timedelta

        start_date = game_date - timedelta(hours=tolerance_hours)
        end_date = game_date + timedelta(hours=tolerance_hours)

        games = self.get_historical_scores(start_date, end_date)

        # Try to match by team names
        # This is a simple implementation - could be enhanced with fuzzy matching
        for game in games:
            if self._teams_match(home_team, game["home_team"]) and self._teams_match(
                away_team, game["away_team"]
            ):
                return game

        return None

    def _teams_match(self, team1: str, team2: str) -> bool:
        """
        Check if two team names/abbreviations match.

        Args:
            team1: First team identifier
            team2: Second team identifier

        Returns:
            True if teams match, False otherwise
        """
        # Normalize strings for comparison
        t1 = team1.lower().strip()
        t2 = team2.lower().strip()

        # Direct match
        if t1 == t2:
            return True

        # Check if one contains the other (e.g., "Lakers" in "Los Angeles Lakers")
        if t1 in t2 or t2 in t1:
            return True

        # TODO: Add more sophisticated matching (team name mapping, fuzzy matching)
        return False
