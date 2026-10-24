"""Unit tests for the candidate registry."""

from pathlib import Path

import pytest
import yaml
from odds_cli.main import app
from odds_core.candidates import (
    AgentCandidate,
    MechanicalCandidate,
    candidates_dir,
    load_candidates,
)
from typer.testing import CliRunner

MECHANICAL: dict[str, object] = {
    "name": "back-longest",
    "description": "Back the longest retail price.",
    "kind": "mechanical",
    "sport": "soccer_epl",
    "version": 1,
    "issue": 610,
    "start_date": "2026-11-01",
    "status": "active",
    "rule": "longest_price",
    "params": {"min_edge": 0.02},
}

AGENT: dict[str, object] = {
    "name": "epl-agent",
    "description": "The EPL betting agent.",
    "kind": "agent",
    "sport": "soccer_epl",
    "version": 3,
    "issue": 611,
    "start_date": "2026-11-02",
    "status": "promoted",
}


def write(directory: Path, data: dict[str, object], stem: str | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem or data['name']}.yaml").write_text(yaml.safe_dump(data))


class TestLoadCandidates:
    def test_loads_both_kinds_sorted_by_name(self, tmp_path: Path) -> None:
        write(tmp_path, MECHANICAL)
        write(tmp_path, AGENT)

        loaded = load_candidates(tmp_path)

        assert [c.name for c in loaded] == ["back-longest", "epl-agent"]
        assert isinstance(loaded[0], MechanicalCandidate)
        assert loaded[0].params == {"min_edge": 0.02}
        assert isinstance(loaded[1], AgentCandidate)

    def test_missing_directory_is_empty(self, tmp_path: Path) -> None:
        assert load_candidates(tmp_path / "absent") == []

    def test_unknown_field_rejected(self, tmp_path: Path) -> None:
        write(tmp_path, {**MECHANICAL, "surprise": 1})
        with pytest.raises(ValueError, match="back-longest.yaml"):
            load_candidates(tmp_path)

    def test_agent_with_params_rejected(self, tmp_path: Path) -> None:
        write(tmp_path, {**AGENT, "params": {}})
        with pytest.raises(ValueError):
            load_candidates(tmp_path)

    def test_bad_name_rejected(self, tmp_path: Path) -> None:
        write(tmp_path, {**MECHANICAL, "name": "Bad_Name"})
        with pytest.raises(ValueError):
            load_candidates(tmp_path)

    def test_name_must_match_file_stem(self, tmp_path: Path) -> None:
        write(tmp_path, MECHANICAL, stem="other-name")
        with pytest.raises(ValueError, match="other-name.yaml"):
            load_candidates(tmp_path)

    def test_one_agent_per_sport(self, tmp_path: Path) -> None:
        write(tmp_path, AGENT)
        write(tmp_path, {**AGENT, "name": "epl-agent-2"})
        with pytest.raises(ValueError, match="soccer_epl"):
            load_candidates(tmp_path)

    def test_agents_for_different_sports_allowed(self, tmp_path: Path) -> None:
        write(tmp_path, AGENT)
        write(tmp_path, {**AGENT, "name": "mlb-agent", "sport": "baseball_mlb"})
        assert len(load_candidates(tmp_path)) == 2

    def test_repo_registry_is_valid(self) -> None:
        load_candidates(candidates_dir())


class TestListCommand:
    def test_empty_registry(self, tmp_path: Path) -> None:
        result = CliRunner(env={"COLUMNS": "200"}).invoke(
            app, ["candidates", "list", "--dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "No candidates registered" in result.stdout

    def test_lists_and_filters_by_status(self, tmp_path: Path) -> None:
        write(tmp_path, MECHANICAL)
        write(tmp_path, AGENT)
        runner = CliRunner(env={"COLUMNS": "200"})

        everything = runner.invoke(app, ["candidates", "list", "--dir", str(tmp_path)])
        promoted = runner.invoke(
            app, ["candidates", "list", "--dir", str(tmp_path), "--status", "promoted"]
        )

        assert "back-longest" in everything.stdout and "epl-agent" in everything.stdout
        assert "#610" in everything.stdout
        assert "epl-agent" in promoted.stdout and "back-longest" not in promoted.stdout
