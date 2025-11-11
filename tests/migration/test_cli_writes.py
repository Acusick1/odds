"""
Write CLI tests - validates CLI write commands work with migrated database.

These tests run CLI commands that WRITE to the database and use API quota.

IMPORTANT: These tests are for DEV/LOCAL environments only.
- Uses API quota (~10-15 requests per test)
- Writes data to database
- Should NOT be run on production

The main migration validation is handled by test_schema_validation.py.
"""

import os

from odds_cli.main import app

# Get environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")


class TestWriteCLI:
    """Validates that write CLI commands work with migrated database."""

    def test_fetch_current(self, runner):
        """
        Test that 'odds fetch current' command works.

        This command:
        - Fetches current NBA odds from The Odds API
        - Uses API quota (~10-15 requests depending on number of games)
        - Writes odds data to database
        - Validates full write pipeline (CLI → API → DB)
        """
        result = runner.invoke(
            app,
            ["fetch", "current"],
            env=os.environ,
        )

        assert result.exit_code == 0, (
            f"'odds fetch current' command failed with exit code {result.exit_code}\n"
            f"Output: {result.stdout}\n"
            f"Error: {result.stderr if hasattr(result, 'stderr') else 'N/A'}\n"
            f"\nThis command fetches current NBA odds and writes to the database.\n"
            f"Failure indicates the write pipeline is broken after migration."
        )

        # Verify output indicates successful fetch
        output_lower = result.stdout.lower()
        assert any(
            keyword in output_lower
            for keyword in ["fetched", "success", "events", "odds", "stored"]
        ), f"Output doesn't indicate successful fetch:\n{result.stdout}"
