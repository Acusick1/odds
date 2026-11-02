"""Data quality validation for odds data."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from odds_core.bookmakers import ZERO_MARGIN_BOOKMAKERS
from odds_core.odds_math import (
    decimal_to_implied_probability,
    is_valid_decimal_price,
    outcome_price_decimal,
)
from odds_core.time import parse_api_datetime

logger = structlog.get_logger()


class OddsValidator:
    """Validator for odds data quality checks."""

    # Validation thresholds
    MIN_VIG_PERCENT = 2.0  # Minimum expected vig for non-exchange bookmakers
    MAX_VIG_PERCENT = 15.0  # Maximum reasonable vig/juice

    @classmethod
    def validate_odds_snapshot(cls, data: dict, event_id: int) -> tuple[bool, list[str]]:
        """
        Validate a complete odds snapshot.

        Args:
            data: One event's envelope
            event_id: The event's id, for the log

        Returns:
            Tuple of (is_valid, list of warning messages)

        Behavior:
            - Returns warnings but does not reject data
            - Logs all validation issues
        """
        errors: list[str] = []
        vig_warnings: list[str] = []

        # Validate bookmakers exist
        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            errors.append("No bookmakers data found")
            return False, errors

        # Validate each bookmaker
        for bookmaker in bookmakers:
            bookmaker_key = bookmaker.get("key", "unknown")

            # Check last update timestamp
            last_update = bookmaker.get("last_update")
            if isinstance(last_update, str):
                try:
                    update_time = parse_api_datetime(last_update)
                    if update_time > datetime.now(UTC):
                        errors.append(f"Future timestamp for {bookmaker_key}: {last_update}")
                except Exception as e:
                    errors.append(f"Invalid timestamp format for {bookmaker_key}: {str(e)}")

            # Validate markets
            markets = bookmaker.get("markets", [])
            for market in markets:
                market_key = market.get("key", "unknown")
                outcomes = market.get("outcomes", [])

                if not outcomes:
                    errors.append(f"No outcomes for {bookmaker_key} {market_key}")
                    continue

                # Validate each outcome
                for outcome in outcomes:
                    outcome_name = outcome.get("name", "unknown")
                    try:
                        price = outcome_price_decimal(outcome)
                    except (KeyError, TypeError, ValueError):
                        errors.append(
                            f"{bookmaker_key} {market_key} {outcome_name}: Missing or invalid price"
                        )
                        continue
                    errors.extend(
                        cls.validate_odds_value(price, bookmaker_key, market_key, outcome_name)
                    )

                # Validate vig for two-way markets (informational, not an error)
                if market_key in ["h2h", "spreads", "totals"] and len(outcomes) == 2:
                    vig_warnings.extend(cls.validate_vig(outcomes, bookmaker_key, market_key))

        is_valid = len(errors) == 0
        all_warnings = errors + vig_warnings

        if all_warnings:
            logger.warning(
                "odds_validation_warnings",
                event_id=event_id,
                warning_count=len(all_warnings),
                warnings=all_warnings[:5],  # Log first 5 warnings
            )

        return is_valid, all_warnings

    @classmethod
    def validate_odds_value(
        cls, price: float, bookmaker: str, market: str, outcome: str
    ) -> list[str]:
        """
        Validate a single decimal odds value against `is_valid_decimal_price`.

        Args:
            price: Decimal odds value
            bookmaker: Bookmaker key
            market: Market key
            outcome: Outcome name

        Returns:
            List of warning messages
        """
        if not isinstance(price, int | float):
            return [f"{bookmaker} {market} {outcome}: Invalid price type {type(price)}"]

        if not is_valid_decimal_price(price):
            return [f"{bookmaker} {market} {outcome}: Odds {price} not above 1.0"]
        return []

    @classmethod
    def validate_vig(cls, outcomes: list[dict], bookmaker: str, market: str) -> list[str]:
        """
        Validate vig/juice for two-way markets.

        Exchange bookmakers (Betfair, Betdaq, etc.) legitimately operate near 0% vig,
        so only negative vig and excessive vig are checked. All other bookmakers use
        the standard MIN_VIG_PERCENT floor.

        Args:
            outcomes: List of outcome dictionaries
            bookmaker: Bookmaker key
            market: Market key

        Returns:
            List of warning messages
        """
        warnings = []

        if len(outcomes) != 2:
            return warnings

        try:
            prob1 = decimal_to_implied_probability(outcome_price_decimal(outcomes[0]))
            prob2 = decimal_to_implied_probability(outcome_price_decimal(outcomes[1]))

            total_prob = prob1 + prob2
            vig_percent = (total_prob - 1.0) * 100

            if vig_percent > cls.MAX_VIG_PERCENT:
                warnings.append(
                    f"{bookmaker} {market}: Vig too high ({vig_percent:.2f}%), possible data error"
                )
            elif vig_percent < 0:
                warnings.append(
                    f"{bookmaker} {market}: Negative vig ({vig_percent:.2f}%), "
                    "possible arbitrage or data error"
                )
            elif bookmaker not in ZERO_MARGIN_BOOKMAKERS and vig_percent < cls.MIN_VIG_PERCENT:
                warnings.append(
                    f"{bookmaker} {market}: Vig too low ({vig_percent:.2f}%), "
                    "possible arbitrage or data error"
                )

        except Exception as e:
            warnings.append(f"{bookmaker} {market}: Error calculating vig: {str(e)}")

        return warnings
