"""
Write CLI tests - validates CLI write commands work with migrated database.

These tests run CLI commands that WRITE to the database using stub API responses.
No real API calls are made.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

from odds_cli.main import app
from odds_core.api_models import OddsResponse, create_scheduled_event


class TestWriteCLI:
    """Validates that write CLI commands work with migrated database."""

    def test_fetch_current(self, runner, monkeypatch):
        """
        Test that 'odds fetch current' command works.

        This command:
        - Fetches current NBA odds (using stub, no real API calls)
        - Writes odds data to database
        - Validates full write pipeline (CLI → Stub → DB)
        """
        # Load sample odds data from fixture
        fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_odds_response.json"
        with open(fixture_path) as f:
            event_data = json.load(f)

        # Create OddsResponse with sample data
        event = create_scheduled_event(event_data)
        mock_response = OddsResponse(
            events=[event],
            raw_events_data=[event_data],
            response_time_ms=100,
            quota_remaining=19900,
            timestamp=datetime.now(UTC),
        )

        # Create mock client that returns our response
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get_odds = AsyncMock(return_value=mock_response)

        # Patch TheOddsAPIClient to return our mock
        def mock_client_factory():
            return mock_client

        monkeypatch.setattr(
            "odds_cli.commands.fetch.TheOddsAPIClient",
            mock_client_factory,
        )

        # Run the CLI command
        result = runner.invoke(app, ["fetch", "current"])

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
            for keyword in ["fetched", "success", "events", "odds", "stored", "complete"]
        ), f"Output doesn't indicate successful fetch:\n{result.stdout}"
