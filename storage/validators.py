"""Data quality validation for odds data."""

from datetime import datetime

import structlog

logger = structlog.get_logger()


class OddsValidator:
    """Validator for odds data quality checks."""

    # Validation thresholds
    MIN_ODDS = -10000
    MAX_ODDS = 10000
    MIN_VIG_PERCENT = 2.0  # Minimum expected vig/juice
    MAX_VIG_PERCENT = 15.0  # Maximum reasonable vig/juice
    MAX_SPREAD_MOVEMENT = 10.0  # Maximum point spread change
    MAX_TOTAL_MOVEMENT = 20.0  # Maximum total points change

    @staticmethod
    def american_to_probability(odds: int) -> float:
        """
        Convert American odds to implied probability.

        Args:
            odds: American odds (e.g., -110, +150)

        Returns:
            Implied probability (0.0 to 1.0)
        """
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    @classmethod
    def validate_odds_snapshot(cls, data: dict, event_id: str) -> tuple[bool, list[str]]:
        """
        Validate a complete odds snapshot.

        Args:
            data: Raw API response data
            event_id: Event identifier

        Returns:
            Tuple of (is_valid, list of warning messages)

        Behavior:
            - Returns warnings but does not reject data
            - Logs all validation issues
        """
        warnings = []

        # Check if data is a list (multiple events) or dict (single event)
        if isinstance(data, list):
            if not data:
                warnings.append("Empty odds data received")
                return False, warnings

            # Find the specific event
            event_data = None
            for event in data:
                if event.get("id") == event_id:
                    event_data = event
                    break

            if not event_data:
                warnings.append(f"Event {event_id} not found in response")
                return False, warnings
        else:
            event_data = data

        # Validate bookmakers exist
        bookmakers = event_data.get("bookmakers", [])
        if not bookmakers:
            warnings.append("No bookmakers data found")
            return False, warnings

        # Validate each bookmaker
        for bookmaker in bookmakers:
            bookmaker_key = bookmaker.get("key", "unknown")

            # Check last update timestamp
            last_update = bookmaker.get("last_update")
            if last_update:
                try:
                    update_time = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
                    if update_time > datetime.now(update_time.tzinfo):
                        warnings.append(f"Future timestamp for {bookmaker_key}: {last_update}")
                except Exception as e:
                    warnings.append(f"Invalid timestamp format for {bookmaker_key}: {str(e)}")

            # Validate markets
            markets = bookmaker.get("markets", [])
            for market in markets:
                market_key = market.get("key", "unknown")
                outcomes = market.get("outcomes", [])

                if not outcomes:
                    warnings.append(f"No outcomes for {bookmaker_key} {market_key}")
                    continue

                # Validate each outcome
                for outcome in outcomes:
                    price = outcome.get("price")
                    if price is not None:
                        price_warnings = cls.validate_odds_value(
                            price, bookmaker_key, market_key, outcome.get("name", "unknown")
                        )
                        warnings.extend(price_warnings)

                # Validate vig for two-way markets
                if market_key in ["h2h", "spreads", "totals"] and len(outcomes) == 2:
                    vig_warnings = cls.validate_vig(outcomes, bookmaker_key, market_key)
                    warnings.extend(vig_warnings)

        is_valid = len(warnings) == 0

        if warnings:
            logger.warning(
                "odds_validation_warnings",
                event_id=event_id,
                warning_count=len(warnings),
                warnings=warnings[:5],  # Log first 5 warnings
            )

        return is_valid, warnings

    @classmethod
    def validate_odds_value(
        cls, price: int, bookmaker: str, market: str, outcome: str
    ) -> list[str]:
        """
        Validate a single odds value.

        Args:
            price: American odds value
            bookmaker: Bookmaker key
            market: Market key
            outcome: Outcome name

        Returns:
            List of warning messages
        """
        warnings = []

        # Check odds range
        if not isinstance(price, int | float):
            warnings.append(f"{bookmaker} {market} {outcome}: Invalid price type {type(price)}")
            return warnings

        price = int(price)

        if price < cls.MIN_ODDS or price > cls.MAX_ODDS:
            warnings.append(
                f"{bookmaker} {market} {outcome}: Odds {price} out of valid range "
                f"({cls.MIN_ODDS} to {cls.MAX_ODDS})"
            )

        # Warn on odds that are too close to even (might indicate stale data)
        if -105 < price < 105 and price != 100:
            warnings.append(f"{bookmaker} {market} {outcome}: Suspicious odds near even: {price}")

        return warnings

    @classmethod
    def validate_vig(cls, outcomes: list[dict], bookmaker: str, market: str) -> list[str]:
        """
        Validate vig/juice for two-way markets.

        Args:
            outcomes: List of outcome dictionaries
            bookmaker: Bookmaker key
            market: Market key

        Returns:
            List of warning messages

        Note:
            Vig should typically be between 2-15% for legitimate markets
        """
        warnings = []

        if len(outcomes) != 2:
            return warnings

        try:
            prob1 = cls.american_to_probability(outcomes[0].get("price", 0))
            prob2 = cls.american_to_probability(outcomes[1].get("price", 0))

            total_prob = prob1 + prob2
            vig_percent = (total_prob - 1.0) * 100

            if vig_percent < cls.MIN_VIG_PERCENT:
                warnings.append(
                    f"{bookmaker} {market}: Vig too low ({vig_percent:.2f}%), "
                    "possible arbitrage or data error"
                )
            elif vig_percent > cls.MAX_VIG_PERCENT:
                warnings.append(
                    f"{bookmaker} {market}: Vig too high ({vig_percent:.2f}%), "
                    "possible data error"
                )

        except Exception as e:
            warnings.append(f"{bookmaker} {market}: Error calculating vig: {str(e)}")

        return warnings

    @classmethod
    def validate_line_movement(
        cls,
        old_point: float | None,
        new_point: float | None,
        market: str,
        bookmaker: str,
    ) -> list[str]:
        """
        Validate line movement is not excessive.

        Args:
            old_point: Previous line value
            new_point: New line value
            market: Market key
            bookmaker: Bookmaker key

        Returns:
            List of warning messages
        """
        warnings = []

        if old_point is None or new_point is None:
            return warnings

        movement = abs(new_point - old_point)

        if market == "spreads" and movement > cls.MAX_SPREAD_MOVEMENT:
            warnings.append(
                f"{bookmaker} {market}: Excessive spread movement "
                f"({old_point} -> {new_point}, {movement} points)"
            )
        elif market == "totals" and movement > cls.MAX_TOTAL_MOVEMENT:
            warnings.append(
                f"{bookmaker} {market}: Excessive total movement "
                f"({old_point} -> {new_point}, {movement} points)"
            )

        return warnings

    @staticmethod
    def validate_event_data(event_data: dict) -> tuple[bool, list[str]]:
        """
        Validate basic event data structure.

        Args:
            event_data: Event data dictionary

        Returns:
            Tuple of (is_valid, list of warning messages)
        """
        warnings = []
        required_fields = ["id", "sport_key", "commence_time", "home_team", "away_team"]

        for field in required_fields:
            if field not in event_data or event_data[field] is None:
                warnings.append(f"Missing required field: {field}")

        # Validate commence_time is in future or recent past
        if "commence_time" in event_data:
            try:
                commence_time = datetime.fromisoformat(
                    event_data["commence_time"].replace("Z", "+00:00")
                )
                now = datetime.now(commence_time.tzinfo)

                # Allow events up to 24 hours in the past (for final results)
                if commence_time < now.replace(hour=0, minute=0, second=0):
                    warnings.append(
                        f"Event commence time is more than 24h in past: {commence_time}"
                    )
            except Exception as e:
                warnings.append(f"Invalid commence_time format: {str(e)}")

        is_valid = len(warnings) == 0
        return is_valid, warnings
