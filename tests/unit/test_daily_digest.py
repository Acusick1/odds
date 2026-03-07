"""Tests for the daily_digest job."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_lambda.jobs.daily_digest import (
    build_digest_embed,
    send_digest,
)


def _make_result(
    home: str = "Arsenal",
    away: str = "Chelsea",
    home_score: int = 2,
    away_score: int = 1,
    predicted_clv: float = 0.015,
) -> dict:
    return {
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "commence_time": datetime.now(UTC) - timedelta(hours=12),
        "predicted_clv": predicted_clv,
    }


def _make_upcoming(
    home: str = "Liverpool",
    away: str = "Man City",
    predicted_clv: float = 0.025,
    hours_from_now: float = 24.0,
) -> dict:
    return {
        "home_team": home,
        "away_team": away,
        "commence_time": datetime.now(UTC) + timedelta(hours=hours_from_now),
        "predicted_clv": predicted_clv,
        "snapshot_time": datetime.now(UTC) - timedelta(hours=1),
    }


class TestBuildDigestEmbed:
    def test_both_sections(self) -> None:
        results = [_make_result()]
        upcoming = [_make_upcoming()]

        embed = build_digest_embed(results, upcoming)

        assert embed["title"] == "EPL Daily Digest"
        assert len(embed["fields"]) == 2
        assert "Post-Match" in embed["fields"][0]["name"]
        assert "Upcoming" in embed["fields"][1]["name"]

    def test_results_only(self) -> None:
        embed = build_digest_embed([_make_result()], [])

        assert len(embed["fields"]) == 1
        assert "Post-Match" in embed["fields"][0]["name"]

    def test_upcoming_only(self) -> None:
        embed = build_digest_embed([], [_make_upcoming()])

        assert len(embed["fields"]) == 1
        assert "Upcoming" in embed["fields"][0]["name"]

    def test_empty_both(self) -> None:
        embed = build_digest_embed([], [])

        assert embed["fields"] == []

    def test_clv_formatting(self) -> None:
        results = [_make_result(predicted_clv=0.032)]
        embed = build_digest_embed(results, [])

        value = embed["fields"][0]["value"]
        assert "+3.2%" in value

    def test_negative_clv(self) -> None:
        results = [_make_result(predicted_clv=-0.018)]
        embed = build_digest_embed(results, [])

        value = embed["fields"][0]["value"]
        assert "-1.8%" in value

    def test_multiple_results_mean(self) -> None:
        results = [
            _make_result(home="Arsenal", away="Chelsea", predicted_clv=0.02),
            _make_result(home="Liverpool", away="Spurs", predicted_clv=0.04),
        ]
        embed = build_digest_embed(results, [])

        value = embed["fields"][0]["value"]
        assert "2 events" in value
        assert "+3.0%" in value

    def test_upcoming_sorted_by_signal(self) -> None:
        upcoming = [
            _make_upcoming(home="Team A", away="Team B", predicted_clv=0.05),
            _make_upcoming(home="Team C", away="Team D", predicted_clv=0.01),
        ]
        embed = build_digest_embed([], upcoming)

        value = embed["fields"][0]["value"]
        assert value.index("Team A") < value.index("Team C")


class TestSendDigest:
    @pytest.mark.asyncio
    @patch("odds_cli.alerts.base.AlertManager")
    @patch("odds_lambda.jobs.daily_digest.async_session_maker")
    async def test_sends_when_data_present(
        self,
        mock_session_maker: MagicMock,
        mock_alert_cls: MagicMock,
    ) -> None:
        mock_session = AsyncMock()
        mock_session_maker.return_value.__aenter__.return_value = mock_session

        # First query: completed events with predictions
        results_result = MagicMock()
        results_result.all.return_value = []
        # Second query: upcoming events with predictions
        upcoming_result = MagicMock()

        from odds_core.models import Event, EventStatus
        from odds_core.prediction_models import Prediction

        event = Event(
            id="e1",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime.now(UTC) + timedelta(hours=24),
            home_team="Arsenal",
            away_team="Chelsea",
            status=EventStatus.SCHEDULED,
        )
        prediction = Prediction(
            id=1,
            event_id="e1",
            snapshot_id=1,
            model_name="test",
            model_version="v1",
            predicted_clv=0.02,
        )
        snapshot_time = datetime.now(UTC) - timedelta(hours=1)
        upcoming_result.all.return_value = [(event, prediction, snapshot_time)]

        mock_session.execute = AsyncMock(side_effect=[results_result, upcoming_result])

        mock_manager = AsyncMock()
        mock_alert_cls.return_value = mock_manager

        stats = await send_digest()

        assert stats["upcoming_count"] == 1
        assert stats["sent"] == 1
        mock_manager.send_embed.assert_called_once()

    @pytest.mark.asyncio
    @patch("odds_cli.alerts.base.AlertManager")
    @patch("odds_lambda.jobs.daily_digest.async_session_maker")
    async def test_skips_when_empty(
        self,
        mock_session_maker: MagicMock,
        mock_alert_cls: MagicMock,
    ) -> None:
        mock_session = AsyncMock()
        mock_session_maker.return_value.__aenter__.return_value = mock_session

        empty_result = MagicMock()
        empty_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=empty_result)

        stats = await send_digest()

        assert stats["results_count"] == 0
        assert stats["upcoming_count"] == 0
        assert stats["sent"] == 0
        mock_alert_cls.return_value.send_embed.assert_not_called()
