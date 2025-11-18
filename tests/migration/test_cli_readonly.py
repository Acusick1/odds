"""
Read-only CLI tests - validates CLI commands work with migrated database.

These tests run actual CLI commands using Typer's CliRunner to ensure:
1. CLI commands execute successfully against the migrated database
2. Database query operations work through the full application stack
3. Output is valid and contains expected information

IMPORTANT: These tests are READ-ONLY and safe to run on production.
They do NOT write to the database or use API quota.
"""

import os

from odds_cli.main import app

# Get the actual database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")


class TestReadOnlyCLI:
    """Validates that read-only CLI commands work with migrated database."""

    def test_status_show(self, runner):
        """
        Test that 'odds status show' command works.

        This validates:
        - CLI can connect to database
        - Database queries work through full stack
        - ORM/SQLModel compatibility with migrated schema
        """
        result = runner.invoke(
            app,
            ["status", "show"],
            env=os.environ,
        )

        assert result.exit_code == 0, (
            f"'odds status show' command failed with exit code {result.exit_code}\n"
            f"Output: {result.stdout}\n"
            f"Error: {result.stderr if hasattr(result, 'stderr') else 'N/A'}\n"
            f"\nThis command queries multiple database tables and validates:\n"
            f"- Database connection works\n"
            f"- ORM queries succeed\n"
            f"- Schema matches SQLModel expectations"
        )

        # Verify output contains expected database statistics
        output_lower = result.stdout.lower()
        assert any(
            keyword in output_lower for keyword in ["events", "odds", "snapshots", "total"]
        ), f"Output missing expected statistics keywords:\n{result.stdout}"

    def test_quality_coverage(self, runner):
        """
        Test that 'odds quality coverage' command works.

        This validates:
        - Quality metrics queries work through full stack
        - Tier and bookmaker coverage analysis functions correctly
        - Rich formatting displays properly
        """
        result = runner.invoke(
            app,
            ["quality", "coverage", "--start", "2024-10-01", "--end", "2024-10-31"],
            env=os.environ,
        )

        assert result.exit_code == 0, (
            f"'odds quality coverage' command failed with exit code {result.exit_code}\n"
            f"Output: {result.stdout}\n"
            f"Error: {result.stderr if hasattr(result, 'stderr') else 'N/A'}\n"
            f"\nThis command queries quality metrics and validates:\n"
            f"- Quality metrics database queries work\n"
            f"- Tier coverage analysis functions correctly\n"
            f"- Bookmaker coverage analysis functions correctly"
        )

        # Verify output contains expected quality coverage keywords
        output_lower = result.stdout.lower()
        assert any(
            keyword in output_lower for keyword in ["coverage", "tier", "bookmaker"]
        ), f"Output missing expected quality coverage keywords:\n{result.stdout}"
