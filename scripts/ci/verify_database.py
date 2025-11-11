#!/usr/bin/env python3
"""Verify database writes after Lambda execution."""

import asyncio
import os
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def verify_database() -> bool:
    """
    Check events and snapshots exist in database.

    Returns:
        bool: True if verification passed, False otherwise
    """
    print("\n" + "=" * 60)
    print("Verifying Database Writes")
    print("=" * 60)

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("✗ DATABASE_URL environment variable not set")
        return False

    print("\n→ Connecting to database...")
    # Mask password in output
    safe_url = database_url.split("@")[1] if "@" in database_url else "..."
    print(f"  Host: {safe_url.split('/')[0]}")

    engine = create_async_engine(database_url, echo=False)

    try:
        async with engine.begin() as conn:
            # Check events table
            result = await conn.execute(text("SELECT COUNT(*) FROM events"))
            event_count = result.scalar()
            print(f"\n→ Events in database: {event_count}")

            if event_count == 0:
                print("✗ No events found in database!")
                return False

            # Check recent snapshots (last 10 minutes)
            result = await conn.execute(
                text(
                    """
                SELECT COUNT(*) FROM odds_snapshots
                WHERE created_at > NOW() - INTERVAL '10 minutes'
            """
                )
            )
            recent_snapshots = result.scalar()
            print(f"→ Recent snapshots (10min): {recent_snapshots}")

            if recent_snapshots == 0:
                print("  ⚠ No recent snapshots (may be expected if no games)")

            # Check fetch logs
            result = await conn.execute(
                text(
                    """
                SELECT COUNT(*) FROM fetch_logs
                WHERE fetch_time > NOW() - INTERVAL '10 minutes'
            """
                )
            )
            recent_fetches = result.scalar()
            print(f"→ Recent fetches (10min): {recent_fetches}")

            # Get latest fetch status
            result = await conn.execute(
                text(
                    """
                SELECT success, events_count, fetch_time
                FROM fetch_logs
                ORDER BY fetch_time DESC
                LIMIT 1
            """
                )
            )
            latest_fetch = result.fetchone()

            if latest_fetch:
                success, events, fetch_time = latest_fetch
                status = "✓ Success" if success else "✗ Failed"
                print(f"\n→ Latest fetch: {status}")
                print(f"  Events fetched: {events}")
                print(f"  Time: {fetch_time}")

                if not success:
                    print("✗ Latest fetch failed!")
                    return False

            print("\n" + "=" * 60)
            print("✓ Database verification passed")
            print("=" * 60)
            return True

    except Exception as e:
        print(f"\n✗ Database verification failed: {e}")
        return False

    finally:
        await engine.dispose()


def main():
    """Run database verification."""
    try:
        result = asyncio.run(verify_database())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
