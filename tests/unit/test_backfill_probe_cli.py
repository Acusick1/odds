"""`odds backfill probe --dry-run` runs offline."""

from __future__ import annotations

from pathlib import Path

from odds_cli.main import app
from typer.testing import CliRunner

RUNNER = CliRunner(env={"COLUMNS": "250"})
MANIFEST = Path(__file__).resolve().parents[2] / "backfills" / "odds-api-mlb-2024-26.yaml"


class TestProbeDryRun:
    def test_it_lists_each_call_and_its_units_without_calling_the_api(self) -> None:
        result = RUNNER.invoke(app, ["backfill", "probe", "--manifest", str(MANIFEST), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "Estimated units: 40" in result.output
        assert "would fetch 2024-07-20T16:00:00Z  20 units" in result.output
        assert "would fetch 2026-07-18T16:00:00Z  20 units" in result.output
