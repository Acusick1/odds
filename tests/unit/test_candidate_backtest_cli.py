"""The backtest and parity commands: argument handling, output and exit codes."""

from __future__ import annotations

import dataclasses
import json
import re
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from odds_analytics.candidate_backtest import (
    BacktestResult,
    Coverage,
    HashStatus,
    Replayed,
    _Counts,
    score,
)
from odds_analytics.candidate_model import Refit, Tuning
from odds_analytics.candidate_parity import DecisionParity, ParityReport, ParityStatus
from odds_analytics.candidate_report import DecisionStart, ScoredBet, Verdict
from odds_cli.main import app
from odds_core.candidates import MechanicalCandidate
from odds_core.vocabulary import Source
from typer.testing import CliRunner

from tests.unit.test_candidates import MECHANICAL, write

N_BOOT = 100
NOW = datetime(2026, 9, 1, tzinfo=UTC)
RUNNER = CliRunner(env={"COLUMNS": "250"})


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "candidates"
    write(directory, MECHANICAL)
    monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: directory)

    @asynccontextmanager
    async def session():
        yield None

    async def no_decisions(_session: Any) -> dict[tuple[str, int], DecisionStart]:
        return {}

    monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", session)
    monkeypatch.setattr("odds_cli.commands.candidates.load_decision_starts", no_decisions)
    return directory


def no_bets(candidate: MechanicalCandidate, counts: _Counts, events: int) -> Replayed:
    """A replay of ``candidate``'s one league that decided ``events`` and placed no bet."""
    (league,) = candidate.sports
    return Replayed([], {league: counts}, {league: events}, 0, None, (), False)


def fake_replay(monkeypatch: pytest.MonkeyPatch) -> list[MechanicalCandidate]:
    seen: list[MechanicalCandidate] = []
    only: list[object] = []

    async def replay(
        _session: Any,
        candidate: MechanicalCandidate,
        start: date,
        end: date,
        k: int,
        only_events: object = None,
    ):
        seen.append(candidate)
        only.append(only_events)
        counts = _Counts(
            decisions=10,
            with_input=9,
            with_exchange=8,
            with_polymarket=0,
            with_retail=9,
            with_book={"bet365": 9, "williamhill": 3},
        )
        result = score(candidate, start, end, no_bets(candidate, counts, 10), k, NOW, N_BOOT, 0)
        if candidate.rule != "retail_ev_model":
            return result
        return dataclasses.replace(
            result,
            range_events=12,
            tuning=Tuning(
                params={"max_depth": 3, "learning_rate": 0.05},
                cv_wmse=0.0069,
                baseline_cv_wmse=0.0072,
                folds=2,
                trials=5,
            ),
            refits=(Refit(0, NOW, 2, 40, (3, 4)),),
        )

    monkeypatch.setattr("odds_cli.commands.candidates.replay", replay)
    fake_replay.only_events = only  # type: ignore[attr-defined]
    return seen


class TestBacktestCommand:
    def test_registered_name_prints_coverage_and_writes_json(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = fake_replay(monkeypatch)
        out = tmp_path / "result.json"

        result = RUNNER.invoke(
            app,
            [
                "candidates",
                "backtest",
                "back-longest",
                "--from",
                "2025-08-01",
                "--to",
                "2025-12-31",
                "--json",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert seen[0].name == "back-longest"
        assert "With input" in result.stdout and "not_yet" in result.stdout
        assert "max input age 10 min" in result.stdout
        assert "promote" not in result.stdout
        assert "Book coverage of back-longest v1's set (uk_retail)" in result.stdout
        assert re.search(r"williamhill\s+│\s+3\s+│\s+33\.3%", result.stdout)
        assert re.search(r"bet365\s+│\s+9\s+│\s+100\.0%", result.stdout)
        summary = json.loads(out.read_text())
        assert summary["coverage"]["events"] == 10
        assert summary["coverage"]["with_book"]["williamhill"] == 3
        assert summary["books"] == "uk_retail"
        assert summary["hash_status"] == HashStatus.MATCH
        assert summary["start"] == "2025-08-01"
        assert summary["summary"]["status"] == "active"
        assert "registered" not in summary["summary"]
        assert summary["summary"]["verdict"] == "not_yet"

    def test_store_writes_the_replayed_run_and_not_the_comparison(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_replay(monkeypatch)
        write(registry, {**MECHANICAL, "name": "other"})
        stored: list[Any] = []

        async def store(_session: Any, records: list[Any]) -> None:
            stored.extend(records)

        monkeypatch.setattr("odds_analytics.backtest_store.store_runs", store)
        monkeypatch.setattr("odds_cli.commands.candidates.current_git_sha", lambda: "e" * 40)

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", "back-longest", "--from", "2025-08-01", "--to", "2025-12-31"]
            + ["--compare", "other", "--store"],
        )

        assert result.exit_code == 0, result.stdout
        (record,) = stored
        run = record.run
        assert (run.candidate_name, run.window_kind.value, run.git_sha) == (
            "back-longest",
            "development",
            "e" * 40,
        )
        assert (run.start_date, run.end_date) == (date(2025, 8, 1), date(2025, 12, 31))
        assert "Stored development run of back-longest v1" in result.stdout

    def test_nothing_is_stored_without_the_flag(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_replay(monkeypatch)
        stored: list[Any] = []

        async def store(_session: Any, records: list[Any]) -> None:
            stored.extend(records)

        monkeypatch.setattr("odds_analytics.backtest_store.store_runs", store)

        result = RUNNER.invoke(
            app,
            [
                "candidates",
                "backtest",
                "back-longest",
                "--from",
                "2025-08-01",
                "--to",
                "2025-12-31",
            ],
        )

        assert result.exit_code == 0 and stored == []

    def test_a_failed_store_exits_one_after_printing_the_replay(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_replay(monkeypatch)

        async def refuse(_session: Any, _records: list[Any]) -> None:
            raise RuntimeError("relation does not exist")

        monkeypatch.setattr("odds_analytics.backtest_store.store_runs", refuse)
        monkeypatch.setattr("odds_cli.commands.candidates.current_git_sha", lambda: "e" * 40)

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", "back-longest", "--from", "2025-08-01", "--to", "2025-12-31"]
            + ["--store"],
        )

        assert result.exit_code == 1
        assert "With input" in result.stdout and "the runs were not stored" in result.stdout

    def test_compare_replays_the_other_candidate_and_reports_both(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = fake_replay(monkeypatch)
        write(registry, {**MECHANICAL, "name": "other"})
        out = tmp_path / "result.json"

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", "back-longest", "--from", "2025-08-01", "--to", "2025-12-31"]
            + ["--compare", "other", "--json", str(out)],
        )

        assert result.exit_code == 0, result.stdout
        assert [c.name for c in seen] == ["back-longest", "other"]
        # A rule without a model decides every range event, so the comparison is not restricted.
        assert fake_replay.only_events == [None, None]  # type: ignore[attr-defined]
        assert "Compared with other" in result.stdout
        assert "Mean EV on the same decided matches" in result.stdout
        payload = json.loads(out.read_text())
        assert payload["comparison"]["name"] == "other"
        assert payload["comparison"]["summary"]["verdict"] == "not_yet"
        assert payload["comparison"]["summary"]["status"] == "active"
        assert "registered" not in payload["comparison"]["summary"]

    def _model_draft(self, tmp_path: Path) -> Path:
        draft = tmp_path / "model.yaml"
        draft.write_text(
            yaml.safe_dump(
                {
                    **{k: v for k, v in MECHANICAL.items() if k != "rule_sha256"},
                    "name": "model",
                    "rule": "retail_ev_model",
                    "params": {
                        "model": "retail_ev_price_only",
                        "threshold": 0.01,
                        "training_matches": 2,
                        "refit_every_matches": 2,
                        "tuning_trials": 5,
                        "tuning_seed": 0,
                        "tuning_min_train_matches": 1,
                        "tuning_step_matches": 1,
                        "minutes_before_kickoff": [60],
                        "max_input_age_minutes": 10,
                    },
                }
            )
        )
        return draft

    def test_compare_replays_the_other_candidate_on_the_models_scored_matches(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_replay(monkeypatch)
        draft = self._model_draft(tmp_path)

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", str(draft), "--from", "2025-08-01", "--to", "2025-12-31"]
            + ["--compare", "back-longest"],
        )

        assert result.exit_code == 0, result.stdout
        assert fake_replay.only_events == [None, (3, 4)]  # type: ignore[attr-defined]
        assert "learning_rate" in result.stdout and "0.007200" in result.stdout
        assert "12 matches in range" in result.stdout

    def test_a_model_rule_cannot_be_the_comparison(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = fake_replay(monkeypatch)
        draft = self._model_draft(tmp_path)

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", "back-longest", "--from", "2025-08-01", "--to", "2025-12-31"]
            + ["--compare", str(draft)],
        )

        assert result.exit_code == 1
        assert "is a model rule" in result.stdout
        assert seen == []

    def backtest(self, name: str = "back-longest"):
        return RUNNER.invoke(
            app, ["candidates", "backtest", name, "--from", "2025-08-01", "--to", "2025-12-31"]
        )

    def decisions(self, monkeypatch: pytest.MonkeyPatch, *keys: tuple[str, int]) -> None:
        async def recorded(_session: Any) -> dict[tuple[str, int], DecisionStart]:
            return dict.fromkeys(keys, DecisionStart(NOW, frozenset({"soccer_epl"})))

        monkeypatch.setattr("odds_cli.commands.candidates.load_decision_starts", recorded)

    def test_sibling_with_a_decision_raises_k_and_is_shown(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_replay(monkeypatch)
        write(registry, {**MECHANICAL, "name": "sibling"})
        self.decisions(monkeypatch, ("sibling", 1))
        out = tmp_path / "result.json"

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", "back-longest", "--from", "2025-08-01", "--to", "2025-12-31"]
            + ["--json", str(out)],
        )

        assert result.exit_code == 0, result.stdout
        assert "k=2" in result.stdout
        assert json.loads(out.read_text())["summary"]["family_versions"] == 2

    def test_registered_sibling_without_a_decision_does_not_count(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_replay(monkeypatch)
        write(registry, {**MECHANICAL, "name": "sibling"})
        assert "k=1" in self.backtest().stdout

    def test_registered_candidate_with_decisions_is_not_counted_twice(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_replay(monkeypatch)
        self.decisions(monkeypatch, ("back-longest", 1))
        assert "k=1" in self.backtest().stdout

    def test_draft_backtest_survives_a_rule_hash_mismatch_in_the_registry(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_replay(monkeypatch)
        write(registry, {**MECHANICAL, "name": "stale", "rule_sha256": "0" * 64})
        draft = tmp_path / "draft.yaml"
        draft.write_text(yaml.safe_dump({**MECHANICAL, "name": "draft"}))

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", str(draft), "--from", "2025-08-01", "--to", "2025-12-31"],
        )

        assert result.exit_code == 0, result.stdout
        assert "k=" in result.stdout

    def test_draft_path_without_hash_is_reported(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = fake_replay(monkeypatch)
        draft = tmp_path / "draft.yaml"
        draft.write_text(
            yaml.safe_dump({k: v for k, v in MECHANICAL.items() if k != "rule_sha256"})
        )

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", str(draft), "--from", "2025-08-01", "--to", "2025-12-31"],
        )

        assert result.exit_code == 0, result.stdout
        assert seen[0].rule_sha256 is None
        assert "rule_sha256 absent" in result.stdout

    @pytest.mark.parametrize(
        "args",
        [
            ["nope", "--from", "2025-08-01", "--to", "2025-12-31"],
            ["back-longest", "--from", "2025-12-31", "--to", "2025-08-01"],
            ["back-longest", "--from", "August", "--to", "2025-12-31"],
        ],
    )
    def test_bad_arguments_exit_1(self, registry: Path, args: list[str]) -> None:
        assert RUNNER.invoke(app, ["candidates", "backtest", *args]).exit_code == 1


def parity_report(*statuses: ParityStatus, db_replayed: tuple = (("home", "bet365", 2.2, 1.0),)):
    return ParityReport(
        "back-longest",
        1,
        [
            DecisionParity(i, i + 1, NOW, status, (), (("home", "bet365", 2.2, 1.0),), db_replayed)
            for i, status in enumerate(statuses)
        ],
    )


class TestParityCommand:
    def invoke(self, monkeypatch: pytest.MonkeyPatch, report: ParityReport):
        async def check(_session: Any, candidate: MechanicalCandidate, version: int | None):
            return report

        monkeypatch.setattr("odds_cli.commands.candidates.check_parity", check)
        return RUNNER.invoke(app, ["candidates", "parity", "back-longest"])

    def test_clean_report_exits_0(self, registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self.invoke(monkeypatch, parity_report(ParityStatus.MATCH, ParityStatus.MATCH))

        assert result.exit_code == 0, result.stdout
        assert "2 decisions checked, 2 matched, 0 mismatches, 0 data differences" in result.stdout

    def test_data_difference_is_informational(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self.invoke(monkeypatch, parity_report(ParityStatus.MATCH, db_replayed=()))

        assert result.exit_code == 0, result.stdout
        assert (
            "1 data differences" in result.stdout and "data difference decision 0" in result.stdout
        )

    def test_mismatch_exits_1_with_details(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self.invoke(monkeypatch, parity_report(ParityStatus.MISMATCH))

        assert result.exit_code == 1
        assert "mismatch decision 0" in result.stdout and "replayed" in result.stdout

    def test_no_decisions_warns_and_exits_0(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self.invoke(monkeypatch, parity_report())

        assert result.exit_code == 0, result.stdout
        assert "no decisions to check" in result.stdout

    def test_unknown_candidate_exits_1(self, registry: Path) -> None:
        assert RUNNER.invoke(app, ["candidates", "parity", "nope"]).exit_code == 1


class TestBacktestVerdictWording:
    def tables(self, verdict: Verdict) -> str:
        from io import StringIO

        from odds_cli.commands.candidates import _backtest_tables
        from rich.console import Console

        candidate = MechanicalCandidate.model_validate(MECHANICAL)
        counts = _Counts(decisions=1, with_input=1, with_exchange=1, with_retail=1)
        result = score(
            candidate,
            date(2025, 8, 1),
            date(2025, 9, 1),
            no_bets(candidate, counts, 1),
            1,
            NOW,
            N_BOOT,
            0,
        )
        result = dataclasses.replace(
            result, summary=dataclasses.replace(result.summary, verdict=verdict)
        )
        out = StringIO()
        for table in _backtest_tables(result):
            Console(file=out, width=250).print(table)
        return out.getvalue()

    def test_promote_bar_is_not_a_promotion(self) -> None:
        text = self.tables(Verdict.PROMOTE)
        assert "meets_promote_bar (development data, not a promotion)" in text
        assert "promote " not in text.replace("promote bar", "")

    @pytest.mark.parametrize("verdict", [Verdict.KILL, Verdict.CONTINUE, Verdict.NOT_YET])
    def test_other_verdicts_are_shown_as_is(self, verdict: Verdict) -> None:
        assert verdict.value in self.tables(verdict)


class TestPerLeagueTable:
    def test_a_pooled_backtest_prints_each_leagues_coverage_and_bets_for_information(
        self,
    ) -> None:
        from io import StringIO

        from odds_analytics.candidate_report import LeagueBreakdown
        from odds_cli.commands.candidates import _league_table
        from rich.console import Console

        epl = LeagueBreakdown("soccer_epl", 3, 2, 0.05, 1.5)
        cov = Coverage(4, 8, 6, 6, 0, 6, 3, 3, 2, {})
        empty = Coverage(2, 2, 0, 0, 0, 0, 0, 0, 0, {})
        table = _league_table(
            "pooled v1", (epl,), {"soccer_epl": cov, "soccer_italy_serie_a": empty}
        )
        out = StringIO()
        Console(file=out, width=250).print(table)
        text = out.getvalue()

        assert "the verdict is the pooled series'" in text
        assert "soccer_epl" in text and "+0.0500" in text and "+1.50" in text
        assert "soccer_italy_serie_a" in text


class TestCloseSources:
    def result(self) -> BacktestResult:
        candidate = MechanicalCandidate.model_validate(MECHANICAL)
        (league,) = candidate.sports

        def bet(event_id: int, source: Source | None) -> ScoredBet:
            return ScoredBet(
                candidate.name,
                candidate.version,
                event_id,
                NOW,
                1.0,
                0.5,
                2.0,
                None if source is None else 0.55,
                NOW,
                timedelta(0),
                league,
                "home",
                None,
                "bet365",
                None,
                source,
            )

        bets = [bet(1, Source.BETFAIR), bet(2, Source.ODDS_API), bet(3, Source.ODDS_API)]
        replayed = Replayed(
            [*bets, bet(4, None)], {league: _Counts()}, {league: 4}, 4, None, (), False
        )
        return score(candidate, date(2025, 8, 1), date(2025, 9, 1), replayed, 1, NOW, N_BOOT, 0)

    def test_coverage_counts_the_bets_with_a_close_by_source(self) -> None:
        assert self.result().coverage.closes == {Source.BETFAIR: 1, Source.ODDS_API: 2}

    def test_the_coverage_table_prints_them(self) -> None:
        from io import StringIO

        from odds_cli.commands.candidates import _backtest_tables
        from rich.console import Console

        out = StringIO()
        for table in _backtest_tables(self.result()):
            Console(file=out, width=250).print(table)
        text = " ".join(out.getvalue().split())
        assert "Bets with a close by close source: betfair 1, odds_api 2" in text

    def test_json_writes_them_and_each_bets_source(self) -> None:
        from odds_cli.commands.candidates import _backtest_json

        payload = json.loads(json.dumps(_backtest_json(self.result()), default=str))
        assert payload["coverage"]["closes"] == {"betfair": 1, "odds_api": 2}
        assert [b["close_source"] for b in payload["bets"]] == [
            "betfair",
            "odds_api",
            "odds_api",
            None,
        ]


class TestStoreAfterReplay:
    def test_the_replays_pooled_connections_are_dropped_before_storing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second event loop cannot reuse connections the first loop opened."""
        from odds_cli.commands import candidates

        calls: list[bool] = []

        class Engine:
            class sync_engine:  # noqa: N801
                @staticmethod
                def dispose(close: bool = True) -> None:
                    calls.append(close)

        async def store(_session: Any, _records: list[Any]) -> None:
            assert calls == [False]

        @asynccontextmanager
        async def session():
            yield None

        monkeypatch.setattr(candidates, "get_engine", lambda: Engine)
        monkeypatch.setattr(candidates, "async_session_maker", session)
        monkeypatch.setattr("odds_analytics.backtest_store.store_runs", store)

        candidates._store([])

        assert calls == [False]
