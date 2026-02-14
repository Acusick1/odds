#!/usr/bin/env python
"""
Check upcoming scheduled games IN THE DATABASE and their expected tiers.

IMPORTANT: This shows what games exist in the database, NOT the live NBA schedule.

To see actual NBA games happening today/this week, visit: https://www.nba.com/schedule

Common issue: If this shows no games but NBA schedule shows games today,
the fetch-odds job hasn't run recently and the database is stale.
"""

import asyncio
import sys
from datetime import UTC, datetime

from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus
from odds_lambda.tier_utils import calculate_tier
from sqlalchemy import and_, select


async def check_upcoming_games(limit: int = 10):
    """
    List upcoming scheduled games with tier information.

    Args:
        limit: Number of games to show (default: 10)
    """
    async with async_session_maker() as session:
        now = datetime.now(UTC)
        result = await session.execute(
            select(Event)
            .where(and_(Event.commence_time > now, Event.status == EventStatus.SCHEDULED))
            .order_by(Event.commence_time)
            .limit(limit)
        )

        events = result.scalars().all()

        print(f"\nCurrent Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"\nNext {limit} Scheduled Games:")
        print("=" * 100)
        print(f"{'Commence Time (UTC)':<20} | {'Hours':>7} | {'Tier':<8} | {'Matchup':<60}")
        print("-" * 100)

        if not events:
            print("No upcoming games found in database")
            return

        for event in events:
            hours_until = (event.commence_time - now).total_seconds() / 3600
            tier = calculate_tier(hours_until)
            matchup = f"{event.away_team} @ {event.home_team}"

            print(
                f"{event.commence_time.strftime('%Y-%m-%d %H:%M'):<20} | {hours_until:>7.1f} | {tier.value:<8} | {matchup:<60}"
            )

        # Show next expected fetch time
        if events:
            closest = events[0]
            hours_until = (closest.commence_time - now).total_seconds() / 3600
            tier = calculate_tier(hours_until)
            next_fetch = now.timestamp() + (tier.interval_hours * 3600)
            next_fetch_dt = datetime.fromtimestamp(next_fetch, UTC)

            print(f"\nClosest Game Tier: {tier.value}")
            print(f"Fetch Interval: {tier.interval_hours} hours")
            print(f"Expected Next Fetch: {next_fetch_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")


if __name__ == "__main__":
    game_limit = 10
    if len(sys.argv) > 1:
        game_limit = int(sys.argv[1])

    asyncio.run(check_upcoming_games(game_limit))
