#!/usr/bin/env python
"""Check tier distribution in database for recent snapshots."""

import asyncio
import sys
from datetime import datetime

from odds_core.database import async_session_maker
from odds_core.models import OddsSnapshot
from sqlalchemy import func, select


async def check_tier_distribution(days_back: int = 7):
    """
    Check tier distribution for recent odds snapshots.

    Args:
        days_back: Number of days to look back (default: 7)
    """
    async with async_session_maker() as session:
        # Calculate start date
        start_date = datetime.now().date()
        if days_back > 0:
            from datetime import timedelta

            start_date = (datetime.now() - timedelta(days=days_back)).date()

        result = await session.execute(
            select(
                func.date(OddsSnapshot.snapshot_time).label("date"),
                OddsSnapshot.fetch_tier,
                func.count().label("count"),
                func.min(OddsSnapshot.hours_until_commence).label("min_hours"),
                func.max(OddsSnapshot.hours_until_commence).label("max_hours"),
            )
            .where(func.date(OddsSnapshot.snapshot_time) >= start_date)
            .group_by(func.date(OddsSnapshot.snapshot_time), OddsSnapshot.fetch_tier)
            .order_by(func.date(OddsSnapshot.snapshot_time).desc(), OddsSnapshot.fetch_tier)
        )

        print(f"\nTier Distribution (last {days_back} days)")
        print("=" * 70)
        print(f"{'Date':<12} | {'Tier':<8} | {'Count':>6} | {'Min Hours':>10} | {'Max Hours':>10}")
        print("-" * 70)

        for row in result:
            tier = row.fetch_tier or "NULL"
            print(
                f"{row.date!s:<12} | {tier:<8} | {row.count:>6} | {row.min_hours:>10.1f} | {row.max_hours:>10.1f}"
            )

        print("\nExpected Tier Ranges:")
        print("  closing: 0-3 hours")
        print("  pregame: 3-12 hours")
        print("  sharp:   12-24 hours")
        print("  early:   24-72 hours")
        print("  opening: >72 hours")


if __name__ == "__main__":
    days = 7
    if len(sys.argv) > 1:
        days = int(sys.argv[1])

    asyncio.run(check_tier_distribution(days))
