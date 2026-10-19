"""The retail-EV evaluation against a database: the window-end guard and a full run."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from odds_analytics.as_of_team_features import load_team_history
from odds_analytics.retail_ev import RetailEvFeatures, market_outcomes
from odds_analytics.retail_ev_evaluation import (
    MANIFEST_FILE,
    PREDICTIONS_FILE,
    REPORT_FILE,
    EvaluationSpec,
    IncompleteWindowError,
    OutputExistsError,
    RepeatedLookError,
    WindowEndError,
    load_evaluation_data,
    run_evaluation,
)
from odds_core.models import Event, EventStatus
from odds_lambda.storage.readers import OddsReader

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

pytestmark = pytest.mark.integration

FIRST_KICKOFF = datetime(2025, 9, 6, 14, 0, tzinfo=UTC)
TRAINING_EVENTS = 6
WINDOW_EVENTS = 4
WINDOW_START = FIRST_KICKOFF + timedelta(days=7 * TRAINING_EVENTS)
WINDOW_END = FIRST_KICKOFF + timedelta(days=7 * (TRAINING_EVENTS + WINDOW_EVENTS) - 1)
LATE_EVENT = "fduk_after_window"
NOW = datetime(2026, 10, 19, tzinfo=UTC)


def _add_event(
    session,
    event_id: str,
    kickoff: datetime,
    prices: tuple[float, ...],
    status: EventStatus = EventStatus.FINAL,
    with_close: bool = True,
) -> None:
    session.add(make_epl_event(event_id, kickoff, status=status, home_score=1, away_score=0))
    session.add(
        make_tagged_snapshot(
            event_id,
            kickoff - timedelta(hours=40),
            [
                three_way_bookmaker("bet365", (prices[0], 3.5, 4.0)),
                three_way_bookmaker("williamhill", (prices[0] - 0.05, 3.6, 4.1)),
                three_way_bookmaker("betfair_exchange", (prices[1], 3.7, 4.3)),
            ],
            "football_data_uk",
            "fduk_collection_deadline",
        )
    )
    if with_close:
        session.add(
            make_tagged_snapshot(
                event_id,
                kickoff,
                [three_way_bookmaker("betfair_exchange", (prices[2], 3.6, 4.0))],
                "football_data_uk",
                "kickoff",
            )
        )


@pytest.fixture
async def evaluation_events(pglite_async_session):
    session = pglite_async_session
    for i in range(TRAINING_EVENTS + WINDOW_EVENTS):
        _add_event(
            session,
            f"fduk_eval_{i}",
            FIRST_KICKOFF + timedelta(days=7 * i),
            (2.0 + 0.01 * i, 2.1, 2.2 - 0.02 * i),
        )
    _add_event(session, LATE_EVENT, WINDOW_END + timedelta(days=2), (2.0, 2.1, 2.2))
    await session.commit()
    return session


def _spec_fields() -> dict:
    return {
        "training_start": (FIRST_KICKOFF - timedelta(days=1)).isoformat(),
        "window": {"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat()},
        "refit_every_events": 3,
        "feature_names": RetailEvFeatures(market_outcomes("1x2")).names,
        "model": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1, "min_child_weight": 1},
        "bootstrap": {"resamples": 200, "seed": 0},
    }


class TestWindowEndGuard:
    async def test_events_after_the_window_end_are_never_loaded(self, evaluation_events) -> None:
        spec = EvaluationSpec.model_validate(_spec_fields())

        data = await load_evaluation_data(evaluation_events, spec, NOW)

        assert LATE_EVENT not in set(data.rows["event_id"])
        assert LATE_EVENT not in set(data.window_events["event_id"])
        assert data.rows["commence_time"].max() <= WINDOW_END.replace(tzinfo=None)
        assert len(data.window_events) == WINDOW_EVENTS

    async def test_refuses_an_event_after_the_window_end(
        self, evaluation_events, monkeypatch
    ) -> None:
        spec = EvaluationSpec.model_validate(_spec_fields())
        original = OddsReader.get_events_by_date_range

        async def leaky(self, start_date, end_date, **kwargs) -> list[Event]:
            return await original(self, start_date, end_date + timedelta(days=30), **kwargs)

        monkeypatch.setattr(OddsReader, "get_events_by_date_range", leaky)

        with pytest.raises(WindowEndError, match="after the window end"):
            await load_evaluation_data(evaluation_events, spec, NOW)

    async def test_refuses_a_window_that_has_not_ended(self, evaluation_events) -> None:
        spec = EvaluationSpec.model_validate(_spec_fields())

        with pytest.raises(WindowEndError, match="interim look"):
            await load_evaluation_data(evaluation_events, spec, WINDOW_END - timedelta(days=1))

    async def test_team_history_stops_at_the_window_end(self, evaluation_events) -> None:
        history = await load_team_history(
            evaluation_events, "soccer_epl", 5, with_fixtures=False, until=WINDOW_END
        )

        loaded = {e.id for events in history.season_results.values() for e in events}
        assert LATE_EVENT not in loaded
        assert "fduk_eval_0" in loaded


class TestReadinessGuard:
    @pytest.mark.parametrize(
        ("status", "with_close"),
        [(EventStatus.SCHEDULED, True), (EventStatus.FINAL, False)],
    )
    async def test_refuses_an_incomplete_window_unless_allowed(
        self, evaluation_events, status: EventStatus, with_close: bool
    ) -> None:
        _add_event(
            evaluation_events,
            "fduk_incomplete",
            WINDOW_START + timedelta(days=1),
            (2.0, 2.1, 2.2),
            status=status,
            with_close=with_close,
        )
        await evaluation_events.commit()
        spec = EvaluationSpec.model_validate(_spec_fields())

        with pytest.raises(IncompleteWindowError, match="fduk_incomplete"):
            await load_evaluation_data(evaluation_events, spec, NOW)
        with pytest.raises(IncompleteWindowError, match="fduk_incomplete"):
            await load_evaluation_data(
                evaluation_events, spec, NOW, allow_incomplete=["fduk_other"]
            )
        data = await load_evaluation_data(
            evaluation_events, spec, NOW, allow_incomplete=["fduk_incomplete"]
        )

        assert "fduk_incomplete" in set(data.window_events["event_id"])

    async def test_refuses_other_feature_columns(self, evaluation_events) -> None:
        fields = _spec_fields()
        fields["feature_names"] = fields["feature_names"][:-1]
        spec = EvaluationSpec.model_validate(fields)

        with pytest.raises(ValueError, match="not the spec's"):
            await load_evaluation_data(evaluation_events, spec, NOW)


class TestRunEvaluation:
    async def test_writes_the_outputs_once(self, evaluation_events, tmp_path: Path) -> None:
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(_spec_fields()))

        results_root = tmp_path / "results"
        results_root.mkdir()
        output_dir = results_root / "first"

        report = await run_evaluation(
            evaluation_events, spec_path, output_dir, now=NOW, results_root=results_root
        )

        manifest = json.loads((output_dir / MANIFEST_FILE).read_text())
        written = json.loads((output_dir / REPORT_FILE).read_text())
        assert (output_dir / PREDICTIONS_FILE).exists()
        assert written["verdict"] == report.verdict.value
        assert manifest["counts"]["scored_events"] == WINDOW_EVENTS
        assert manifest["counts"]["pre_window_events"] == TRAINING_EVENTS
        assert manifest["counts"]["blocks"] == 2
        assert manifest["spec_sha256"] == EvaluationSpec.from_yaml(spec_path).sha256()
        assert len(manifest["git_sha"]) == 40
        assert isinstance(manifest["git_dirty"], bool)
        assert written["coverage"]["window_event_shares"]["has_fduk_closing"] == 1.0
        assert manifest["forced"] is False
        assert manifest["git_tag"] is None or isinstance(manifest["git_tag"], str)

        refused_dir = results_root / "refused"
        with pytest.raises(RepeatedLookError):
            await run_evaluation(
                evaluation_events, spec_path, refused_dir, now=NOW, results_root=results_root
            )
        assert not refused_dir.exists()

        with pytest.raises(OutputExistsError):
            await run_evaluation(
                evaluation_events, spec_path, output_dir, now=NOW, results_root=tmp_path / "none"
            )
        with pytest.raises(RepeatedLookError):
            await run_evaluation(
                evaluation_events,
                spec_path,
                results_root / "second",
                now=NOW,
                results_root=results_root,
            )

        await run_evaluation(
            evaluation_events, spec_path, output_dir, now=NOW, force=True, results_root=results_root
        )
        forced = json.loads((output_dir / MANIFEST_FILE).read_text())
        assert forced["forced"] is True
        assert forced["overwritten_predictions_sha256"] == manifest["predictions_sha256"]
        assert forced["earlier_runs_of_spec"] == [str(output_dir / MANIFEST_FILE)]
