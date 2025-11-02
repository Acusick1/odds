"""Rule-based betting strategies using the new domain infrastructure.

These strategies use logic and statistical calculations instead of ML models.
They work with domain objects (BettingEvent, OddsObservation) for clean separation.
"""

from typing import Any

from analytics.backtesting import BetOpportunity
from analytics.betting.observations import OddsObservation
from analytics.betting.problems import BettingEvent
from analytics.betting.strategies import RuleBasedStrategy
from analytics.utils import (
    calculate_ev,
    calculate_implied_probability,
    detect_arbitrage,
)


class FlatBettingStrategy(RuleBasedStrategy):
    """Baseline strategy: Bet on every game matching a pattern.

    Useful as a baseline to compare other strategies against.
    Always bets regardless of odds value.
    """

    def __init__(
        self,
        market: str = "h2h",
        outcome_pattern: str = "home",  # "home", "away", or "favorite"
        bookmaker: str = "fanduel",
    ):
        """Initialize flat betting strategy.

        Args:
            market: Which market to bet (h2h, spreads, totals)
            outcome_pattern: How to select outcome ("home", "away", "favorite")
            bookmaker: Which bookmaker to use
        """
        super().__init__(
            name="FlatBetting",
            market=market,
            outcome_pattern=outcome_pattern,
            bookmaker=bookmaker,
        )

    async def evaluate_opportunity(
        self,
        event: BettingEvent,
        observations: list[OddsObservation],
        config: Any,
    ) -> list[BetOpportunity]:
        """Find opportunities based on simple rules.

        Args:
            event: Betting event to evaluate
            observations: Odds observations for this event
            config: Backtest configuration

        Returns:
            List of betting opportunities (0 or 1)
        """
        market = self.params["market"]
        bookmaker = self.params["bookmaker"]
        pattern = self.params["outcome_pattern"]

        # Filter observations for our market and bookmaker
        relevant_obs = [
            obs for obs in observations if obs.market == market and obs.bookmaker == bookmaker
        ]

        if not relevant_obs:
            return []

        # Select outcome based on pattern
        if pattern == "home":
            target_outcome = event.home_competitor
        elif pattern == "away":
            target_outcome = event.away_competitor
        elif pattern == "favorite":
            # Find favorite (most negative odds)
            favorite_obs = min(relevant_obs, key=lambda o: o.odds)
            target_outcome = favorite_obs.outcome
        else:
            return []

        # Find observation for target outcome
        target_obs = next(
            (obs for obs in relevant_obs if obs.outcome == target_outcome),
            None,
        )

        if not target_obs:
            return []

        return [
            BetOpportunity(
                event_id=event.id,
                market=target_obs.market,
                outcome=target_obs.outcome,
                bookmaker=target_obs.bookmaker,
                odds=target_obs.odds,
                line=target_obs.line,
                confidence=0.5,  # Flat confidence (no edge)
                rationale=f"Flat bet on {pattern}",
            )
        ]


class BasicEVStrategy(RuleBasedStrategy):
    """Expected Value strategy using sharp vs retail bookmaker comparison.

    Uses sharp books (Pinnacle, Circa) as "true" probability baseline
    and looks for positive expected value at retail books.
    """

    def __init__(
        self,
        sharp_book: str = "pinnacle",
        retail_books: list[str] | None = None,
        min_ev_threshold: float = 0.03,  # 3% minimum EV
        markets: list[str] | None = None,
    ):
        """Initialize EV strategy.

        Args:
            sharp_book: Sharp bookmaker for baseline odds (default: Pinnacle)
            retail_books: Retail books to find +EV bets at
            min_ev_threshold: Minimum EV required to bet (default: 3%)
            markets: Markets to consider (default: h2h, spreads, totals)
        """
        if retail_books is None:
            retail_books = ["fanduel", "draftkings", "betmgm", "caesars"]

        if markets is None:
            markets = ["h2h", "spreads", "totals"]

        super().__init__(
            name="BasicEV",
            sharp_book=sharp_book,
            retail_books=retail_books,
            min_ev_threshold=min_ev_threshold,
            markets=markets,
        )

    async def evaluate_opportunity(
        self,
        event: BettingEvent,
        observations: list[OddsObservation],
        config: Any,
    ) -> list[BetOpportunity]:
        """Find +EV opportunities by comparing sharp vs retail odds.

        Args:
            event: Betting event to evaluate
            observations: Odds observations for this event
            config: Backtest configuration

        Returns:
            List of betting opportunities with positive EV
        """
        opportunities = []

        sharp_book = self.params["sharp_book"]
        retail_books = self.params["retail_books"]
        min_ev = self.params["min_ev_threshold"]
        markets = self.params["markets"]

        # Process each market
        for market in markets:
            market_obs = [obs for obs in observations if obs.market == market]

            if not market_obs:
                continue

            # Get sharp odds for this market
            sharp_obs = [obs for obs in market_obs if obs.bookmaker == sharp_book]

            if not sharp_obs:
                continue

            # For each outcome, compare sharp vs retail
            outcomes = {obs.outcome for obs in sharp_obs}

            for outcome in outcomes:
                # Get sharp odds for this outcome
                sharp_outcome_obs = next(
                    (obs for obs in sharp_obs if obs.outcome == outcome),
                    None,
                )

                if not sharp_outcome_obs:
                    continue

                # Calculate "true" probability from sharp odds
                true_prob = calculate_implied_probability(sharp_outcome_obs.odds)

                # Check each retail book for +EV
                for retail_book in retail_books:
                    retail_obs = next(
                        (
                            obs
                            for obs in market_obs
                            if obs.bookmaker == retail_book and obs.outcome == outcome
                        ),
                        None,
                    )

                    if not retail_obs:
                        continue

                    # Calculate expected value
                    ev = calculate_ev(true_prob, retail_obs.odds)

                    if ev >= min_ev:
                        opportunities.append(
                            BetOpportunity(
                                event_id=event.id,
                                market=market,
                                outcome=outcome,
                                bookmaker=retail_book,
                                odds=retail_obs.odds,
                                line=retail_obs.line,
                                confidence=true_prob,
                                rationale=f"+{ev:.1%} EV vs {sharp_book}",
                            )
                        )

        return opportunities


class ArbitrageStrategy(RuleBasedStrategy):
    """Risk-free arbitrage betting across bookmakers.

    Identifies situations where betting both sides of a market across
    different bookmakers guarantees profit regardless of outcome.
    """

    def __init__(
        self,
        min_profit_margin: float = 0.01,  # 1% minimum profit
        max_hold: float = 0.10,  # 10% max market hold to filter bad lines
        bookmakers: list[str] | None = None,
        markets: list[str] | None = None,
    ):
        """Initialize arbitrage strategy.

        Args:
            min_profit_margin: Minimum profit margin required (default: 1%)
            max_hold: Maximum market hold to consider (default: 10%)
            bookmakers: Bookmakers to consider (default: all major books)
            markets: Markets to search for arbs (default: h2h, spreads, totals)
        """
        if bookmakers is None:
            bookmakers = [
                "pinnacle",
                "circa",
                "fanduel",
                "draftkings",
                "betmgm",
                "caesars",
                "betrivers",
                "bovada",
            ]

        if markets is None:
            markets = ["h2h", "spreads", "totals"]

        super().__init__(
            name="Arbitrage",
            min_profit_margin=min_profit_margin,
            max_hold=max_hold,
            bookmakers=bookmakers,
            markets=markets,
        )

    async def evaluate_opportunity(
        self,
        event: BettingEvent,
        observations: list[OddsObservation],
        config: Any,
    ) -> list[BetOpportunity]:
        """Find arbitrage opportunities across bookmakers.

        Args:
            event: Betting event to evaluate
            observations: Odds observations for this event
            config: Backtest configuration

        Returns:
            List of betting opportunities (typically 2 legs for arb)
        """
        opportunities = []

        markets = self.params["markets"]
        bookmakers = self.params["bookmakers"]
        min_profit = self.params["min_profit_margin"]
        max_hold = self.params["max_hold"]

        # Process each market
        for market in markets:
            market_obs = [
                obs for obs in observations if obs.market == market and obs.bookmaker in bookmakers
            ]

            if len(market_obs) < 2:
                continue

            # Get all unique outcomes for this market
            outcomes = list({obs.outcome for obs in market_obs})

            # For 2-way markets, check for arbitrage
            if len(outcomes) == 2:
                # Find best odds for each outcome across all bookmakers
                outcome_odds = {}
                for outcome in outcomes:
                    outcome_obs = [obs for obs in market_obs if obs.outcome == outcome]
                    if outcome_obs:
                        # Best odds = highest odds (most positive or least negative)
                        best_obs = max(outcome_obs, key=lambda o: o.odds)
                        outcome_odds[outcome] = best_obs

                if len(outcome_odds) == 2:
                    # Check if this is an arbitrage opportunity
                    obs_list = list(outcome_odds.values())
                    # detect_arbitrage expects list of (bookmaker_key, american_odds) tuples
                    has_arb, profit_pct, stake_dist = detect_arbitrage(
                        [(obs.bookmaker, obs.odds) for obs in obs_list]
                    )

                    if has_arb and (profit_pct / 100) >= min_profit:
                        # Calculate market hold to filter suspicious lines
                        implied_probs = [
                            calculate_implied_probability(obs.odds) for obs in obs_list
                        ]
                        hold = sum(implied_probs) - 1.0

                        if hold <= max_hold:
                            # Add both legs of the arbitrage
                            for obs in obs_list:
                                opportunities.append(
                                    BetOpportunity(
                                        event_id=event.id,
                                        market=market,
                                        outcome=obs.outcome,
                                        bookmaker=obs.bookmaker,
                                        odds=obs.odds,
                                        line=obs.line,
                                        confidence=1.0,  # Risk-free
                                        rationale=f"Arb: {profit_pct:.2f}% profit",
                                    )
                                )

        return opportunities
