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
        assert "Estimated units: 20" in result.output
        assert (
            "would fetch 2024-07-20T15:58:08Z for the 2024-07-20T15:55:38Z snapshot  10 units"
            in result.output
        )
        assert (
            "would fetch 2026-07-18T15:58:08Z for the 2026-07-18T15:55:38Z snapshot  10 units"
            in result.output
        )
