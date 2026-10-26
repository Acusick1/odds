"""Unit tests for the candidate registry."""

from pathlib import Path

import pytest
import yaml
from odds_cli.main import app
from odds_core.candidates import (
    AgentCandidate,
    MechanicalCandidate,
    candidates_dir,
    load_candidate_file,
    load_candidates,
)
from odds_lambda.candidate_rules import resolve_rule, rule_sha256
from typer.testing import CliRunner

MECHANICAL: dict[str, object] = {
    "name": "back-longest",
    "description": "Back the longest retail price.",
    "kind": "mechanical",
    "market": "1x2",
    "sport": "soccer_epl",
    "version": 1,
    "issue": 610,
    "start_date": "2026-11-01",
    "status": "active",
    "rule": "exchange_gap",
    "rule_sha256": rule_sha256("exchange_gap"),
    "params": {"threshold": 0.02, "minutes_before_kickoff": [60], "max_input_age_minutes": 10},
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
        assert loaded[0].params == {
            "threshold": 0.02,
            "minutes_before_kickoff": [60],
            "max_input_age_minutes": 10,
        }
        assert isinstance(loaded[1], AgentCandidate)

    def test_a_draft_file_may_lack_the_hash_but_the_registry_may_not(self, tmp_path: Path) -> None:
        draft = {k: v for k, v in MECHANICAL.items() if k != "rule_sha256"}
        write(tmp_path, draft)

        loaded = load_candidate_file(tmp_path / "back-longest.yaml")

        assert isinstance(loaded, MechanicalCandidate) and loaded.rule_sha256 is None
        with pytest.raises(ValueError, match="must record rule_sha256"):
            load_candidates(tmp_path)

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

    def test_killed_agent_does_not_block_replacement(self, tmp_path: Path) -> None:
        write(tmp_path, {**AGENT, "status": "killed"})
        write(tmp_path, {**AGENT, "name": "epl-agent-2", "status": "active"})
        assert len(load_candidates(tmp_path)) == 2

    @pytest.mark.parametrize("version", ["3", 0, 1.5, True])
    def test_version_must_be_positive_integer(self, tmp_path: Path, version: object) -> None:
        write(tmp_path, {**MECHANICAL, "version": version})
        with pytest.raises(ValueError):
            load_candidates(tmp_path)

    def test_agents_for_different_sports_allowed(self, tmp_path: Path) -> None:
        write(tmp_path, AGENT)
        write(tmp_path, {**AGENT, "name": "mlb-agent", "sport": "baseball_mlb"})
        assert len(load_candidates(tmp_path)) == 2

    def test_repo_registry_is_valid(self) -> None:
        load_candidates(candidates_dir())


class TestListCommand:
    @pytest.fixture
    def registry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: tmp_path)
        return tmp_path

    def test_empty_registry(self, registry: Path) -> None:
        result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["candidates", "list"])
        assert result.exit_code == 0
        assert "No candidates registered" in result.stdout

    def test_lists_and_filters_by_status(self, registry: Path) -> None:
        write(registry, MECHANICAL)
        write(registry, AGENT)
        runner = CliRunner(env={"COLUMNS": "200"})

        everything = runner.invoke(app, ["candidates", "list"])
        promoted = runner.invoke(app, ["candidates", "list", "--status", "promoted"])

        assert "back-longest" in everything.stdout and "epl-agent" in everything.stdout
        assert "#610" in everything.stdout
        assert "epl-agent" in promoted.stdout and "back-longest" not in promoted.stdout

    def test_invalid_file_exits_1(self, registry: Path) -> None:
        write(registry, {**MECHANICAL, "surprise": 1})
        result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["candidates", "list"])
        assert result.exit_code == 1
        assert "back-longest.yaml" in result.stdout

    def test_unknown_rule_exits_1(self, registry: Path) -> None:
        write(registry, {**MECHANICAL, "rule": "no_such_rule"})
        result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["candidates", "list"])
        assert result.exit_code == 1
        assert "back-longest" in result.stdout and "no_such_rule" in result.stdout

    def test_changed_rule_hash_exits_1(self, registry: Path) -> None:
        write(registry, {**MECHANICAL, "rule_sha256": "0" * 64})
        result = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["candidates", "list"])
        assert result.exit_code == 1
        assert "code of rule 'exchange_gap' changed" in result.stdout

    def test_rule_hash_prints_the_current_hash(self) -> None:
        result = CliRunner(env={"COLUMNS": "200"}).invoke(
            app, ["candidates", "rule-hash", "exchange_gap"]
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == rule_sha256("exchange_gap")

    def test_rule_hash_unknown_rule_exits_1(self) -> None:
        result = CliRunner(env={"COLUMNS": "200"}).invoke(
            app, ["candidates", "rule-hash", "no_such_rule"]
        )
        assert result.exit_code == 1


class TestDraftExample:
    def test_the_how_to_example_validates(self, tmp_path: Path) -> None:
        page = Path(__file__).parents[2] / "wiki/evaluation/howto-take-an-idea-to-a-verdict.md"
        block = page.read_text().split("```yaml\n", 1)[1].split("```", 1)[0]
        draft = tmp_path / "draft.yaml"
        draft.write_text("\n".join(line.removeprefix("     ") for line in block.splitlines()))

        loaded = load_candidate_file(draft)

        assert isinstance(loaded, MechanicalCandidate) and loaded.rule_sha256 is None
        resolve_rule(loaded)
