"""Example betting strategies for backtesting."""

from odds_analytics.backtesting import BacktestConfig, BacktestEvent, BetOpportunity, BettingStrategy
from odds_analytics.utils import (
    calculate_ev,
    calculate_implied_probability,
    calculate_market_hold,
    detect_arbitrage,
)
from odds_core.models import Odds


class FlatBettingStrategy(BettingStrategy):
    """
    Baseline strategy: Bet fixed amount on every game.

    Useful as a baseline to compare other strategies against.
    """

    def __init__(
        self,
        market: str = "h2h",
        outcome_pattern: str = "home",  # "home", "away", or "favorite"
        bookmaker: str = "fanduel",
    ):
        """
        Initialize flat betting strategy.

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
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        """Find opportunities based on simple rules."""
        opportunities = []

        # Filter for our market and bookmaker
        relevant_odds = [
            o
            for o in odds_snapshot
            if o.market_key == self.params["market"] and o.bookmaker_key == self.params["bookmaker"]
        ]

        if not relevant_odds:
            return []

        # Select outcome based on pattern
        pattern = self.params["outcome_pattern"]

        if pattern == "home":
            target_outcome = event.home_team
        elif pattern == "away":
            target_outcome = event.away_team
        elif pattern == "favorite":
            # Find favorite (lowest odds / most negative)
            min_odds = min(relevant_odds, key=lambda o: o.price)
            target_outcome = min_odds.outcome_name
        else:
            return []

        # Find odds for target outcome
        target_odds = next(
            (o for o in relevant_odds if o.outcome_name == target_outcome),
            None,
        )

        if target_odds:
            opportunities.append(
                BetOpportunity(
                    event_id=event.id,
                    market=target_odds.market_key,
                    outcome=target_odds.outcome_name,
                    bookmaker=target_odds.bookmaker_key,
                    odds=target_odds.price,
                    line=target_odds.point,
                    confidence=0.5,  # Flat confidence (50%)
                    rationale=f"Flat bet on {pattern} team",
                )
            )

        return opportunities


class BasicEVStrategy(BettingStrategy):
    """
    Expected Value strategy: Bet when retail book differs from sharp book.

    Uses Pinnacle as "true" probability and looks for +EV at other books.
    """

    def __init__(
        self,
        sharp_book: str = "pinnacle",
        retail_books: list[str] | None = None,
        min_ev_threshold: float = 0.03,  # 3% minimum EV
        markets: list[str] | None = None,
    ):
        """
        Initialize EV strategy.

        Args:
            sharp_book: Sharp bookmaker for "true" odds (default: Pinnacle)
            retail_books: Retail books to find +EV bets at (default: FanDuel, DraftKings)
            min_ev_threshold: Minimum EV required to bet (default: 3%)
            markets: Markets to consider (default: all)
        """
        if retail_books is None:
            retail_books = ["fanduel", "draftkings", "betmgm"]

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
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        """Find +EV opportunities by comparing sharp vs retail odds."""
        opportunities = []

        sharp_book = self.params["sharp_book"]
        retail_books = self.params["retail_books"]
        min_ev = self.params["min_ev_threshold"]
        markets = self.params["markets"]

        # Group odds by market and outcome
        for market in markets:
            # Get sharp odds for this market
            sharp_odds = [
                o for o in odds_snapshot if o.bookmaker_key == sharp_book and o.market_key == market
            ]

            if not sharp_odds:
                continue

            # For each outcome in sharp book
            for sharp_odd in sharp_odds:
                # Calculate "true" probability from sharp book
                true_prob = calculate_implied_probability(sharp_odd.price)

                # Check retail books for better odds
                for retail_book in retail_books:
                    retail_odd = next(
                        (
                            o
                            for o in odds_snapshot
                            if o.bookmaker_key == retail_book
                            and o.market_key == market
                            and o.outcome_name == sharp_odd.outcome_name
                            and (
                                o.point == sharp_odd.point if sharp_odd.point is not None else True
                            )
                        ),
                        None,
                    )

                    if not retail_odd:
                        continue

                    # Calculate EV at retail odds using sharp probability
                    ev = calculate_ev(
                        bet_probability=true_prob,
                        american_odds=retail_odd.price,
                        stake=100.0,  # Per $100
                    )

                    ev_percentage = ev / 100.0  # Convert to percentage

                    # If EV exceeds threshold, add opportunity
                    if ev_percentage >= min_ev:
                        opportunities.append(
                            BetOpportunity(
                                event_id=event.id,
                                market=retail_odd.market_key,
                                outcome=retail_odd.outcome_name,
                                bookmaker=retail_odd.bookmaker_key,
                                odds=retail_odd.price,
                                line=retail_odd.point,
                                confidence=true_prob,  # Use sharp implied prob as confidence
                                rationale=f"+EV: {ev_percentage:.2%} edge over {sharp_book} "
                                f"(Sharp: {sharp_odd.price}, Retail: {retail_odd.price})",
                            )
                        )

        return opportunities


class ArbitrageStrategy(BettingStrategy):
    """
    Arbitrage strategy: Find risk-free profit opportunities.

    Looks for situations where you can bet both sides and guarantee profit.
    """

    def __init__(
        self,
        min_profit_margin: float = 0.01,  # 1% minimum profit
        max_hold: float = 0.10,  # Don't arb markets with >10% hold
        bookmakers: list[str] | None = None,
    ):
        """
        Initialize arbitrage strategy.

        Args:
            min_profit_margin: Minimum profit margin to pursue (default: 1%)
            max_hold: Maximum market hold to consider (default: 10%)
            bookmakers: Bookmakers to consider (default: all major ones)
        """
        if bookmakers is None:
            bookmakers = [
                "pinnacle",
                "fanduel",
                "draftkings",
                "betmgm",
                "williamhill_us",
                "betrivers",
            ]

        super().__init__(
            name="Arbitrage",
            min_profit_margin=min_profit_margin,
            max_hold=max_hold,
            bookmakers=bookmakers,
        )

    async def evaluate_opportunity(
        self,
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        """Find arbitrage opportunities across bookmakers."""
        opportunities = []

        min_profit = self.params["min_profit_margin"]
        max_hold = self.params["max_hold"]
        bookmakers = self.params["bookmakers"]

        # Check each market type
        for market in ["h2h", "spreads", "totals"]:
            market_odds = [
                o for o in odds_snapshot if o.market_key == market and o.bookmaker_key in bookmakers
            ]

            if not market_odds:
                continue

            # For two-way markets (h2h in NBA rarely ties)
            if market == "h2h":
                # Find best odds for each team
                home_odds = [o for o in market_odds if o.outcome_name == event.home_team]
                away_odds = [o for o in market_odds if o.outcome_name == event.away_team]

                if home_odds and away_odds:
                    best_home = max(home_odds, key=lambda o: o.price)
                    best_away = max(away_odds, key=lambda o: o.price)

                    # Skip if market hold is too high (unlikely to have arb)
                    hold = calculate_market_hold([best_home.price, best_away.price])
                    if hold > max_hold:
                        continue

                    # Check for arbitrage
                    has_arb, profit_pct, stakes = detect_arbitrage(
                        [
                            (best_home.bookmaker_key, best_home.price),
                            (best_away.bookmaker_key, best_away.price),
                        ]
                    )

                    if has_arb and profit_pct >= (min_profit * 100):
                        # Add both sides of the arb
                        opportunities.append(
                            BetOpportunity(
                                event_id=event.id,
                                market=market,
                                outcome=best_home.outcome_name,
                                bookmaker=best_home.bookmaker_key,
                                odds=best_home.price,
                                line=None,
                                confidence=1.0,  # Arb is risk-free
                                rationale=f"Arbitrage: {profit_pct:.2%} profit "
                                f"({best_home.bookmaker_key} vs {best_away.bookmaker_key})",
                            )
                        )
                        opportunities.append(
                            BetOpportunity(
                                event_id=event.id,
                                market=market,
                                outcome=best_away.outcome_name,
                                bookmaker=best_away.bookmaker_key,
                                odds=best_away.price,
                                line=None,
                                confidence=1.0,
                                rationale=f"Arbitrage: {profit_pct:.2%} profit "
                                f"({best_home.bookmaker_key} vs {best_away.bookmaker_key})",
                            )
                        )

            # For spreads and totals, check each line separately
            elif market in ["spreads", "totals"]:
                # Group by line value
                lines = {}
                for odd in market_odds:
                    if odd.point is not None:
                        if odd.point not in lines:
                            lines[odd.point] = []
                        lines[odd.point].append(odd)

                for line_value, line_odds in lines.items():
                    if len(line_odds) < 2:
                        continue

                    # For spreads: home vs away at same line
                    # For totals: over vs under at same line
                    if market == "spreads":
                        home_line_odds = [o for o in line_odds if o.outcome_name == event.home_team]
                        away_line_odds = [o for o in line_odds if o.outcome_name == event.away_team]

                        if home_line_odds and away_line_odds:
                            best_home = max(home_line_odds, key=lambda o: o.price)
                            best_away = max(away_line_odds, key=lambda o: o.price)

                            # Skip if market hold is too high
                            hold = calculate_market_hold([best_home.price, best_away.price])
                            if hold > max_hold:
                                continue

                            has_arb, profit_pct, stakes = detect_arbitrage(
                                [
                                    (best_home.bookmaker_key, best_home.price),
                                    (best_away.bookmaker_key, best_away.price),
                                ]
                            )

                            if has_arb and profit_pct >= (min_profit * 100):
                                opportunities.extend(
                                    [
                                        BetOpportunity(
                                            event_id=event.id,
                                            market=market,
                                            outcome=best_home.outcome_name,
                                            bookmaker=best_home.bookmaker_key,
                                            odds=best_home.price,
                                            line=best_home.point,
                                            confidence=1.0,
                                            rationale=f"Spread arb: {profit_pct:.2%} profit at {line_value}",
                                        ),
                                        BetOpportunity(
                                            event_id=event.id,
                                            market=market,
                                            outcome=best_away.outcome_name,
                                            bookmaker=best_away.bookmaker_key,
                                            odds=best_away.price,
                                            line=best_away.point,
                                            confidence=1.0,
                                            rationale=f"Spread arb: {profit_pct:.2%} profit at {line_value}",
                                        ),
                                    ]
                                )

                    elif market == "totals":
                        over_odds = [o for o in line_odds if o.outcome_name.lower() == "over"]
                        under_odds = [o for o in line_odds if o.outcome_name.lower() == "under"]

                        if over_odds and under_odds:
                            best_over = max(over_odds, key=lambda o: o.price)
                            best_under = max(under_odds, key=lambda o: o.price)

                            # Skip if market hold is too high
                            hold = calculate_market_hold([best_over.price, best_under.price])
                            if hold > max_hold:
                                continue

                            has_arb, profit_pct, stakes = detect_arbitrage(
                                [
                                    (best_over.bookmaker_key, best_over.price),
                                    (best_under.bookmaker_key, best_under.price),
                                ]
                            )

                            if has_arb and profit_pct >= (min_profit * 100):
                                opportunities.extend(
                                    [
                                        BetOpportunity(
                                            event_id=event.id,
                                            market=market,
                                            outcome=best_over.outcome_name,
                                            bookmaker=best_over.bookmaker_key,
                                            odds=best_over.price,
                                            line=best_over.point,
                                            confidence=1.0,
                                            rationale=f"Total arb: {profit_pct:.2%} profit at {line_value}",
                                        ),
                                        BetOpportunity(
                                            event_id=event.id,
                                            market=market,
                                            outcome=best_under.outcome_name,
                                            bookmaker=best_under.bookmaker_key,
                                            odds=best_under.price,
                                            line=best_under.point,
                                            confidence=1.0,
                                            rationale=f"Total arb: {profit_pct:.2%} profit at {line_value}",
                                        ),
                                    ]
                                )

        return opportunities
