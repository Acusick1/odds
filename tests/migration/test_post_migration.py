"""
Post-migration validation tests.

These tests validate that the database is functional after migrations are applied.
They assume the database is at the latest migration version.

Separate from integration tests because:
- Integration tests should work at any migration state
- These tests specifically validate the migrated schema
- Run after migrations in CI/CD pipelines
"""

from datetime import UTC, datetime, timedelta

import pytest
from rich.console import Console
from rich.table import Table
from sqlalchemy import text

from core.models import Event, EventStatus
from storage.readers import OddsReader
from storage.tier_validator import TierCoverageValidator
from storage.writers import OddsWriter

console = Console()


class TestPostMigration:
    """Post-migration validation - ensures database is functional after migrations."""

    @pytest.mark.asyncio
    async def test_basic_database_operations(self, test_session):
        """Test that basic database operations work after migration."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        # Create an event
        event = Event(
            id="smoke_test_event_1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC) + timedelta(hours=2),
            home_team="Test Home Team",
            away_team="Test Away Team",
            status=EventStatus.SCHEDULED,
        )

        # Test write operation
        created_event = await writer.upsert_event(event)
        await test_session.commit()

        assert created_event.id == "smoke_test_event_1"
        assert created_event.home_team == "Test Home Team"

        # Test read operation
        retrieved_event = await reader.get_event_by_id("smoke_test_event_1")
        assert retrieved_event is not None
        assert retrieved_event.home_team == "Test Home Team"

        print("\n✓ Basic database operations: PASS")

    @pytest.mark.asyncio
    async def test_status_command_functionality(self, test_session):
        """Test that 'odds status show' functionality works."""
        reader = OddsReader(test_session)

        # This is what 'odds status show' does internally
        stats = await reader.get_database_stats()

        # Verify we get expected structure
        assert "total_events" in stats
        assert "total_odds_records" in stats
        assert "total_snapshots" in stats

        print("\n✓ Status command: PASS")
        print(f"  - Total events: {stats['total_events']}")
        print(f"  - Total odds records: {stats['total_odds_records']}")
        print(f"  - Total snapshots: {stats['total_snapshots']}")

    @pytest.mark.asyncio
    async def test_validate_command_functionality(self, test_session):
        """Test that 'odds validate daily' functionality works."""
        validator = TierCoverageValidator(test_session)

        # Test validation for yesterday (should work even with no data)
        yesterday = (datetime.now(UTC) - timedelta(days=1)).date()

        # Verify it executes without crashing
        report = await validator.validate_date(yesterday)
        assert report is not None
        assert report.total_games >= 0
        assert hasattr(report, "is_valid")
        assert hasattr(report, "complete_games")

        print("\n✓ Validate command: PASS")
        print(f"  - Games validated: {report.total_games}")
        print(f"  - Complete games: {report.complete_games}")

    @pytest.mark.asyncio
    async def test_database_summary(self, test_session):
        """Print comprehensive database summary."""
        reader = OddsReader(test_session)

        # Get stats from reader
        stats = await reader.get_database_stats()

        # Get table count from information_schema
        result = await test_session.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )
        )
        table_count = result.scalar()

        # Get index count
        result = await test_session.execute(
            text("SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public'")
        )
        index_count = result.scalar()

        # Get list of tables
        result = await test_session.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' "
                "ORDER BY tablename"
            )
        )
        tables = [row[0] for row in result.fetchall()]

        # Create summary table
        summary_table = Table(
            title="Post-Migration Database State", show_header=False, title_style="bold blue"
        )
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", justify="right", style="white")

        summary_table.add_row("Tables", str(table_count))
        summary_table.add_row("Indexes", str(index_count))
        summary_table.add_row("Events", str(stats["total_events"]))
        summary_table.add_row("Odds Records", str(stats["total_odds_records"]))
        summary_table.add_row("Snapshots", str(stats["total_snapshots"]))

        # Print summary
        console.print("\n")
        console.print(summary_table)

        # Print table list
        console.print("\n[bold]Database Tables:[/bold]")
        for table_name in tables:
            console.print(f"  ✓ {table_name}")

        # Final success message
        console.print("\n[bold green]✓ All smoke tests passed![/bold green]\n")

        # Assertions to ensure test fails if something is wrong
        assert table_count >= 5  # Should have at least 5 core tables
        assert index_count > 0  # Should have indexes
        assert "events" in tables
        assert "odds" in tables
        assert "odds_snapshots" in tables
