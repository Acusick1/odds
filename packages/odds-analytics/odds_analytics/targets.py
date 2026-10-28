"""
Training-label maths: closing-line movement targets.

Key Functions:
- calculate_regression_target(): Calculate line movement for regression
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import structlog
from odds_core.quotes import Quote

logger = structlog.get_logger()

__all__ = [
    "calculate_regression_target",
]


def calculate_regression_target(
    opening_quotes: Sequence[Quote] | None,
    closing_quotes: Sequence[Quote] | None,
    market: str,
) -> float | None:
    """
    Calculate the regression target (closing - opening line delta).

    For h2h market: Uses implied probability delta (closing prob - opening prob)
    For spreads/totals: Uses point delta (closing point - opening point)

    Args:
        opening_quotes: Opening quotes for target outcome
        closing_quotes: Closing quotes for target outcome
        market: Market type (h2h, spreads, totals)

    Returns:
        Line movement delta or None if cannot be calculated.
        For h2h: Probability change (positive = line moved in favor)
        For spreads/totals: Point change (closing point - opening point)
    """
    if not opening_quotes or not closing_quotes:
        return None

    if market == "h2h":
        # For h2h, use implied probability delta
        # Average across bookmakers if multiple
        opening_probs = [1 / q.price for q in opening_quotes]
        closing_probs = [1 / q.price for q in closing_quotes]

        if not opening_probs or not closing_probs:
            return None

        opening_prob = float(np.mean(opening_probs))
        closing_prob = float(np.mean(closing_probs))

        return closing_prob - opening_prob

    elif market in ("spreads", "totals"):
        # For spreads/totals, use point delta
        # Filter for odds that have point values
        opening_points = [q.line for q in opening_quotes if q.line is not None]
        closing_points = [q.line for q in closing_quotes if q.line is not None]

        if not opening_points or not closing_points:
            return None

        opening_point = float(np.mean(opening_points))
        closing_point = float(np.mean(closing_points))

        return closing_point - opening_point

    else:
        # Unknown market
        return None
