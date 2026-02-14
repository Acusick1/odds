#!/usr/bin/env python
"""Check tier coverage for specific games to diagnose missing snapshots."""

import asyncio
import sys
from datetime import datetime

from odds_core.database import async_session_maker
from odds_core.models import Event, OddsSnapshot
from odds_lambda.fetch_tier import FetchTier
from sqlalchemy import func, select


async def check_game_tier_coverage(event_id: str | None = None, date: str | None = None):
    """
    Check tier coverage for games.

    Args:
        event_id: Specific event ID to check
        date: Date to check all games (YYYY-MM-DD format)
    """
    async with async_session_maker() as session:
        if event_id:
            # Check specific game
            event_result = await session.execute(select(Event).where(Event.id == event_id))
            event = event_result.scalar_one_or_none()

            if not event:
                print(f"Event {event_id} not found")
                return

            snapshot_result = await session.execute(
                select(
                    OddsSnapshot.fetch_tier,
                    func.count().label("count"),
                    func.min(OddsSnapshot.snapshot_time).label("first_snapshot"),
                    func.max(OddsSnapshot.snapshot_time).label("last_snapshot"),
                )
                .where(OddsSnapshot.event_id == event_id)
                .group_by(OddsSnapshot.fetch_tier)
            )

            print(f"\nGame: {event.away_team} @ {event.home_team}")
            print(f"Commence Time: {event.commence_time}")
            print(f"Status: {event.status.value}")
            print("\nTier Coverage:")
            print("=" * 90)
            print(f"{'Tier':<10} | {'Count':>6} | {'First Snapshot':<22} | {'Last Snapshot':<22}")
            print("-" * 90)

            tiers_present = set()
            for row in snapshot_result:
                tier = row.fetch_tier or "NULL"
                tiers_present.add(tier)
                print(
                    f"{tier:<10} | {row.count:>6} | {row.first_snapshot!s:<22} | {row.last_snapshot!s:<22}"
                )

            # Show missing tiers
            all_tiers = {tier.value for tier in FetchTier}
            missing_tiers = all_tiers - tiers_present
            if missing_tiers:
                print(f"\nMissing Tiers: {', '.join(sorted(missing_tiers))}")

        elif date:
            # Check all games on specific date
            target_date = datetime.strptime(date, "%Y-%m-%d").date()

            # Get events for that date
            event_result = await session.execute(
                select(Event).where(func.date(Event.commence_time) == target_date)
            )
            events = event_result.scalars().all()

            if not events:
                print(f"No games found for {date}")
                return

            print(f"\nGames on {date}: {len(events)} total")
            print("=" * 120)

            for event in events:
                # Get tier coverage for this event
                snapshot_result = await session.execute(
                    select(OddsSnapshot.fetch_tier, func.count().label("count"))
                    .where(OddsSnapshot.event_id == event.id)
                    .group_by(OddsSnapshot.fetch_tier)
                )

                tiers = {row.fetch_tier: row.count for row in snapshot_result}
                total_snapshots = sum(tiers.values())

                tier_summary = " | ".join(
                    f"{tier}: {count}" for tier, count in sorted(tiers.items())
                )

                print(f"\n{event.away_team} @ {event.home_team} ({event.commence_time})")
                print(f"  Total Snapshots: {total_snapshots}")
                print(f"  Tier Distribution: {tier_summary}")

                # Check for missing tiers
                all_tiers = {tier.value for tier in FetchTier}
                missing = all_tiers - set(tiers.keys())
                if missing:
                    print(f"  ⚠ Missing Tiers: {', '.join(sorted(missing))}")

        else:
            print("Usage:")
            print(
                "  Check specific game:  uv run python scripts/check_game_tier_coverage.py --event <event_id>"
            )
            print(
                "  Check games by date:  uv run python scripts/check_game_tier_coverage.py --date YYYY-MM-DD"
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        asyncio.run(check_game_tier_coverage())
    elif sys.argv[1] == "--event":
        asyncio.run(check_game_tier_coverage(event_id=sys.argv[2]))
    elif sys.argv[1] == "--date":
        asyncio.run(check_game_tier_coverage(date=sys.argv[2]))
    else:
        asyncio.run(check_game_tier_coverage())
