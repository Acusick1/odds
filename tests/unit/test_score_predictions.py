"""Tests for the score_predictions job."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.jobs.score_predictions import (
    _DEFAULT_FEATURE_CONFIG,
    _extract_features,
    score_events,
)

_EXPECTED_FEATURE_NAMES = [
    f"tab_{n}"
    for n in [
        "consensus_prob",
        "sharp_prob",
        "retail_sharp_diff",
        "num_bookmakers",
        "is_weekend",
        "day_of_week",
    ]
] + ["hours_until_event"]


def _make_event(
    event_id: str = "test_event_1",
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    hours_from_now: float = 48.0,
) -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=datetime.now(UTC) + timedelta(hours=hours_from_now),
        home_team=home_team,
        away_team=away_team,
        status=EventStatus.SCHEDULED,
    )


def _make_snapshot(
    event: Event,
    snapshot_id: int = 1,
    hours_before: float = 24.0,
) -> OddsSnapshot:
    raw_data = {
        "bookmakers": [
            {
                "key": "bet365",
                "title": "Bet365",
                "last_update": "2026-03-07T12:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": event.home_team, "price": -150},
                            {"name": "Draw", "price": 250},
                            {"name": event.away_team, "price": 200},
                        ],
                    }
                ],
            },
            {
                "key": "betway",
                "title": "Betway",
                "last_update": "2026-03-07T12:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": event.home_team, "price": -140},
                            {"name": "Draw", "price": 240},
                            {"name": event.away_team, "price": 190},
                        ],
                    }
                ],
            },
        ]
    }
    return OddsSnapshot(
        id=snapshot_id,
        event_id=event.id,
        snapshot_time=event.commence_time - timedelta(hours=hours_before),
        raw_data=raw_data,
        bookmaker_count=2,
        fetch_tier="pregame",
        hours_until_commence=hours_before,
    )


class TestExtractFeatures:
    def test_extracts_features_successfully(self) -> None:
        event = _make_event()
        snapshot = _make_snapshot(event)
        config = _DEFAULT_FEATURE_CONFIG

        result = _extract_features(event, snapshot, config, _EXPECTED_FEATURE_NAMES)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(_EXPECTED_FEATURE_NAMES),)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

    def test_returns_none_for_empty_snapshot(self) -> None:
        event = _make_event()
        snapshot = OddsSnapshot(
            id=1,
            event_id=event.id,
            snapshot_time=event.commence_time - timedelta(hours=24),
            raw_data={"bookmakers": []},
            bookmaker_count=0,
        )
        config = _DEFAULT_FEATURE_CONFIG

        result = _extract_features(event, snapshot, config, [])

        assert result is None

    def test_returns_none_on_feature_name_mismatch(self) -> None:
        event = _make_event()
        snapshot = _make_snapshot(event)
        config = _DEFAULT_FEATURE_CONFIG

        wrong_names = ["wrong_feature_1", "wrong_feature_2"]
        result = _extract_features(event, snapshot, config, wrong_names)

        assert result is None

    def test_hours_until_is_positive_for_future_event(self) -> None:
        event = _make_event(hours_from_now=48.0)
        snapshot = _make_snapshot(event, hours_before=24.0)
        config = _DEFAULT_FEATURE_CONFIG

        result = _extract_features(event, snapshot, config, _EXPECTED_FEATURE_NAMES)

        assert result is not None
        hours_until = result[-1]
        assert hours_until > 0


class TestScoreEvents:
    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions.load_model")
    @patch("odds_lambda.jobs.score_predictions.get_cached_version")
    @patch("odds_lambda.jobs.score_predictions.async_session_maker")
    async def test_scores_unscored_snapshots(
        self,
        mock_session_maker: MagicMock,
        mock_version: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.015])
        mock_load.return_value = {
            "model": mock_model,
            "feature_names": _EXPECTED_FEATURE_NAMES,
            "params": {},
        }
        mock_version.return_value = '"etag123"'

        event = _make_event()
        snapshot = _make_snapshot(event)

        mock_session = AsyncMock()

        # First execute returns events, second returns unscored snapshots,
        # third is the INSERT ON CONFLICT
        events_result = MagicMock()
        events_result.scalars.return_value.all.return_value = [event]
        snapshots_result = MagicMock()
        snapshots_result.scalars.return_value.all.return_value = [snapshot]
        insert_result = MagicMock()
        mock_session.execute = AsyncMock(
            side_effect=[events_result, snapshots_result, insert_result]
        )

        mock_session_maker.return_value.__aenter__.return_value = mock_session

        stats = await score_events(model_name="epl-clv-home", bucket="test-bucket")

        assert stats["events_checked"] == 1
        assert stats["snapshots_scored"] == 1
        assert stats["errors"] == 0

        # Verify the INSERT statement was executed (3rd call)
        assert mock_session.execute.call_count == 3
        mock_model.predict.assert_called_once()

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions.load_model")
    async def test_returns_empty_stats_on_model_load_failure(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = FileNotFoundError("Model not found")

        stats = await score_events(model_name="missing", bucket="test-bucket")

        assert stats["events_checked"] == 0
        assert stats["snapshots_scored"] == 0

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions.load_model")
    @patch("odds_lambda.jobs.score_predictions.get_cached_version")
    @patch("odds_lambda.jobs.score_predictions.async_session_maker")
    async def test_no_upcoming_events(
        self,
        mock_session_maker: MagicMock,
        mock_version: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = {
            "model": MagicMock(),
            "feature_names": [],
            "params": {},
        }
        mock_version.return_value = '"v1"'

        mock_session = AsyncMock()

        events_result = MagicMock()
        events_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=events_result)

        mock_session_maker.return_value.__aenter__.return_value = mock_session

        stats = await score_events(model_name="test", bucket="test-bucket")

        assert stats["events_checked"] == 0
        assert stats["snapshots_scored"] == 0
