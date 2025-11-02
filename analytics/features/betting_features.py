"""Betting-specific feature computation functions.

These functions can be used with both TabularFeatureExtractor and SequentialFeatureExtractor.
Each function signature: (problem, observations, decision_time) -> dict[str, float]
"""

from datetime import datetime

from analytics.betting import BettingEvent, OddsObservation
from analytics.core import Observation, PredictionProblem


def compute_sharp_retail_diff(
    problem: PredictionProblem,
    observations: list[Observation],
    decision_time: datetime,
) -> dict[str, float]:
    """Compute difference between sharp and retail bookmaker odds.

    This is a key feature for EV betting - sharp books like Pinnacle have lower margins
    and more accurate odds, so differences between sharp and retail indicate value.

    Args:
        problem: Betting event
        observations: Odds observations
        decision_time: Decision time

    Returns:
        Dictionary with sharp_retail_diff_home and sharp_retail_diff_away
    """
    if not isinstance(problem, BettingEvent):
        return {"sharp_retail_diff_home": 0.0, "sharp_retail_diff_away": 0.0}

    # Filter to odds observations for h2h market
    odds_obs = [
        obs for obs in observations if isinstance(obs, OddsObservation) and obs.market == "h2h"
    ]

    if not odds_obs:
        return {"sharp_retail_diff_home": 0.0, "sharp_retail_diff_away": 0.0}

    # Get sharp book odds (Pinnacle)
    sharp_obs = [obs for obs in odds_obs if obs.bookmaker == "pinnacle"]
    sharp_home_prob = 0.5
    sharp_away_prob = 0.5

    if sharp_obs:
        home_obs = [obs for obs in sharp_obs if obs.outcome == problem.home_competitor]
        away_obs = [obs for obs in sharp_obs if obs.outcome == problem.away_competitor]

        if home_obs:
            sharp_home_prob = home_obs[0].get_implied_probability()
        if away_obs:
            sharp_away_prob = away_obs[0].get_implied_probability()

    # Get retail average (FanDuel, DraftKings, BetMGM)
    retail_books = ["fanduel", "draftkings", "betmgm"]
    retail_obs = [obs for obs in odds_obs if obs.bookmaker in retail_books]

    retail_home_probs = []
    retail_away_probs = []

    for book in retail_books:
        book_obs = [obs for obs in retail_obs if obs.bookmaker == book]
        home_obs = [obs for obs in book_obs if obs.outcome == problem.home_competitor]
        away_obs = [obs for obs in book_obs if obs.outcome == problem.away_competitor]

        if home_obs:
            retail_home_probs.append(home_obs[0].get_implied_probability())
        if away_obs:
            retail_away_probs.append(away_obs[0].get_implied_probability())

    retail_home_prob = sum(retail_home_probs) / len(retail_home_probs) if retail_home_probs else 0.5
    retail_away_prob = sum(retail_away_probs) / len(retail_away_probs) if retail_away_probs else 0.5

    return {
        "sharp_retail_diff_home": sharp_home_prob - retail_home_prob,
        "sharp_retail_diff_away": sharp_away_prob - retail_away_prob,
    }


def compute_market_hold(
    problem: PredictionProblem,
    observations: list[Observation],
    decision_time: datetime,
) -> dict[str, float]:
    """Compute market hold (bookmaker's edge/vig).

    Lower hold = more efficient market. Typical hold is 4-5% for two-way markets.

    Args:
        problem: Betting event
        observations: Odds observations
        decision_time: Decision time

    Returns:
        Dictionary with market_hold for each bookmaker
    """
    if not isinstance(problem, BettingEvent):
        return {"market_hold_avg": 0.05}

    odds_obs = [
        obs for obs in observations if isinstance(obs, OddsObservation) and obs.market == "h2h"
    ]

    if not odds_obs:
        return {"market_hold_avg": 0.05}

    # Calculate hold for each bookmaker
    holds = []
    bookmakers = {obs.bookmaker for obs in odds_obs}

    for book in bookmakers:
        book_obs = [obs for obs in odds_obs if obs.bookmaker == book]
        home_obs = [obs for obs in book_obs if obs.outcome == problem.home_competitor]
        away_obs = [obs for obs in book_obs if obs.outcome == problem.away_competitor]

        if home_obs and away_obs:
            total_prob = (
                home_obs[0].get_implied_probability() + away_obs[0].get_implied_probability()
            )
            hold = total_prob - 1.0
            holds.append(hold)

    avg_hold = sum(holds) / len(holds) if holds else 0.05

    return {"market_hold_avg": avg_hold}


def compute_line_movement(
    problem: PredictionProblem,
    observations: list[Observation],
    decision_time: datetime,
) -> dict[str, float]:
    """Compute how much the odds have moved.

    Line movement can indicate sharp money coming in on one side.

    Args:
        problem: Betting event
        observations: Odds observations
        decision_time: Decision time

    Returns:
        Dictionary with line_movement features
    """
    if not isinstance(problem, BettingEvent):
        return {"line_movement_home": 0.0}

    odds_obs = [
        obs for obs in observations if isinstance(obs, OddsObservation) and obs.market == "h2h"
    ]

    if len(odds_obs) < 2:
        return {"line_movement_home": 0.0}

    # Sort by time
    odds_obs.sort(key=lambda o: o.observation_time)

    # Get earliest and latest for home team (using a common bookmaker)
    common_books = ["fanduel", "draftkings", "pinnacle"]
    for book in common_books:
        home_obs = [
            obs
            for obs in odds_obs
            if obs.bookmaker == book and obs.outcome == problem.home_competitor
        ]

        if len(home_obs) >= 2:
            earliest_prob = home_obs[0].get_implied_probability()
            latest_prob = home_obs[-1].get_implied_probability()
            movement = latest_prob - earliest_prob

            return {"line_movement_home": movement}

    return {"line_movement_home": 0.0}


def compute_consensus_odds(
    problem: PredictionProblem,
    observations: list[Observation],
    decision_time: datetime,
) -> dict[str, float]:
    """Compute consensus (average) odds across bookmakers.

    Args:
        problem: Betting event
        observations: Odds observations
        decision_time: Decision time

    Returns:
        Dictionary with consensus probabilities
    """
    if not isinstance(problem, BettingEvent):
        return {"consensus_home_prob": 0.5, "consensus_away_prob": 0.5}

    odds_obs = [
        obs for obs in observations if isinstance(obs, OddsObservation) and obs.market == "h2h"
    ]

    if not odds_obs:
        return {"consensus_home_prob": 0.5, "consensus_away_prob": 0.5}

    home_probs = []
    away_probs = []

    for obs in odds_obs:
        if obs.outcome == problem.home_competitor:
            home_probs.append(obs.get_implied_probability())
        elif obs.outcome == problem.away_competitor:
            away_probs.append(obs.get_implied_probability())

    consensus_home = sum(home_probs) / len(home_probs) if home_probs else 0.5
    consensus_away = sum(away_probs) / len(away_probs) if away_probs else 0.5

    return {
        "consensus_home_prob": consensus_home,
        "consensus_away_prob": consensus_away,
    }


def compute_timing_features(
    problem: PredictionProblem,
    observations: list[Observation],
    decision_time: datetime,
) -> dict[str, float]:
    """Compute time-based features.

    Args:
        problem: Betting event
        observations: Odds observations
        decision_time: Decision time

    Returns:
        Dictionary with timing features
    """
    if not isinstance(problem, BettingEvent):
        return {"hours_to_game": 0.0, "day_of_week": 0.0}

    hours_to_game = (problem.commence_time - decision_time).total_seconds() / 3600
    day_of_week = problem.commence_time.weekday()  # 0=Monday, 6=Sunday

    return {
        "hours_to_game": hours_to_game,
        "day_of_week": float(day_of_week),
    }
