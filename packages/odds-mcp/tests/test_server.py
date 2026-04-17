"""Unit tests for MCP server serialization helpers and error paths."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.paper_trade_models import PaperTrade, TradeResult


class TestEventToDict:
    def _make_event(self, **overrides: object) -> Event:
        defaults = {
            "id": "evt-1",
            "sport_key": "soccer_epl",
            "sport_title": "EPL",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": datetime(2026, 4, 12, 15, 0, tzinfo=UTC),
            "status": EventStatus.SCHEDULED,
            "home_score": None,
            "away_score": None,
            "completed_at": None,
        }
        defaults.update(overrides)
        event = MagicMock(spec=Event)
        for k, v in defaults.items():
            setattr(event, k, v)
        return event

    def test_basic_fields(self) -> None:
        from odds_mcp.server import _event_to_dict

        event = self._make_event()
        result = _event_to_dict(event)
        assert result["id"] == "evt-1"
        assert result["sport_key"] == "soccer_epl"
        assert result["sport_title"] == "EPL"
        assert result["home_team"] == "Arsenal"
        assert result["away_team"] == "Chelsea"
        assert result["status"] == "scheduled"
        assert result["home_score"] is None
        assert result["away_score"] is None
        assert result["completed_at"] is None
        assert "commence_time" in result

    def test_completed_event(self) -> None:
        from odds_mcp.server import _event_to_dict

        completed = datetime(2026, 4, 12, 17, 0, tzinfo=UTC)
        event = self._make_event(
            status=EventStatus.FINAL,
            home_score=2,
            away_score=1,
            completed_at=completed,
        )
        result = _event_to_dict(event)
        assert result["status"] == "final"
        assert result["home_score"] == 2
        assert result["completed_at"] == completed.isoformat()


class TestSnapshotToDict:
    def _make_snapshot(self, **overrides: object) -> OddsSnapshot:
        defaults = {
            "id": 42,
            "event_id": "evt-1",
            "snapshot_time": datetime(2026, 4, 12, 10, 0, tzinfo=UTC),
            "created_at": datetime(2026, 4, 12, 10, 1, tzinfo=UTC),
            "bookmaker_count": 5,
            "fetch_tier": "early",
            "hours_until_commence": 5.0,
            "raw_data": {"bookmakers": [{"key": "bet365"}]},
        }
        defaults.update(overrides)
        snap = MagicMock(spec=OddsSnapshot)
        for k, v in defaults.items():
            setattr(snap, k, v)
        return snap

    def test_excludes_raw_data_by_default(self) -> None:
        from odds_mcp.server import _snapshot_to_dict

        snap = self._make_snapshot()
        result = _snapshot_to_dict(snap)
        assert "raw_data" not in result
        assert result["id"] == 42
        assert result["created_at"] is not None
        assert result["fetch_tier"] == "early"

    def test_includes_raw_data_when_requested(self) -> None:
        from odds_mcp.server import _snapshot_to_dict

        snap = self._make_snapshot()
        result = _snapshot_to_dict(snap, include_raw_data=True)
        assert "raw_data" in result
        assert result["raw_data"]["bookmakers"][0]["key"] == "bet365"

    def test_includes_extracted_odds(self) -> None:
        from odds_mcp.server import _snapshot_to_dict

        odds_obj = MagicMock()
        odds_obj.bookmaker_key = "bet365"
        odds_obj.bookmaker_title = "Bet365"
        odds_obj.market_key = "h2h"
        odds_obj.outcome_name = "Arsenal"
        odds_obj.price = -110
        odds_obj.point = None

        snap = self._make_snapshot()
        result = _snapshot_to_dict(snap, extracted_odds=[odds_obj])
        assert len(result["odds"]) == 1
        assert result["odds"][0]["bookmaker_key"] == "bet365"
        assert result["odds"][0]["price"] == -110


class TestTradeToDict:
    def test_serialization(self) -> None:
        from odds_mcp.server import _trade_to_dict

        trade = MagicMock(spec=PaperTrade)
        trade.id = 1
        trade.event_id = "evt-1"
        trade.market = "h2h"
        trade.selection = "home"
        trade.bookmaker = "bet365"
        trade.odds = -110
        trade.stake = 10.0
        trade.reasoning = "good value"
        trade.confidence = 0.8
        trade.bankroll_before = 1000.0
        trade.bankroll_after = None
        trade.placed_at = datetime(2026, 4, 12, 10, 0, tzinfo=UTC)
        trade.settled_at = None
        trade.result = None
        trade.pnl = None

        result = _trade_to_dict(trade)
        assert result["id"] == 1
        assert result["odds"] == -110
        assert result["result"] is None
        assert result["settled_at"] is None

    def test_settled_trade(self) -> None:
        from odds_mcp.server import _trade_to_dict

        trade = MagicMock(spec=PaperTrade)
        trade.id = 2
        trade.event_id = "evt-1"
        trade.market = "h2h"
        trade.selection = "home"
        trade.bookmaker = "bet365"
        trade.odds = -110
        trade.stake = 10.0
        trade.reasoning = None
        trade.confidence = None
        trade.bankroll_before = 1000.0
        trade.bankroll_after = 1009.09
        trade.placed_at = datetime(2026, 4, 12, 10, 0, tzinfo=UTC)
        trade.settled_at = datetime(2026, 4, 12, 17, 0, tzinfo=UTC)
        trade.result = TradeResult.WIN
        trade.pnl = 9.09

        result = _trade_to_dict(trade)
        assert result["result"] == "win"
        assert result["pnl"] == 9.09


class TestSafeFloat:
    def test_normal_float(self) -> None:
        from odds_mcp.server import _safe_float

        assert _safe_float(1.23456789) == 1.234568

    def test_nan(self) -> None:
        from odds_mcp.server import _safe_float

        assert _safe_float(float("nan")) is None

    def test_none(self) -> None:
        from odds_mcp.server import _safe_float

        assert _safe_float(None) is None

    def test_string(self) -> None:
        from odds_mcp.server import _safe_float

        assert _safe_float("not a number") is None


class TestRefreshScrapeUnknownLeague:
    @pytest.mark.asyncio
    async def test_unknown_league_returns_error(self) -> None:
        from odds_mcp.server import refresh_scrape

        result = await refresh_scrape(league="unknown-league", market="1x2")
        assert "error" in result
        assert "Unknown league" in result["error"]
        assert result["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_known_league_uses_correct_sport(self) -> None:
        from odds_mcp.server import refresh_scrape

        with patch(
            "odds_lambda.jobs.fetch_oddsportal.ingest_league", new_callable=AsyncMock
        ) as mock_ingest:
            mock_stats = MagicMock()
            mock_stats.league = "mlb"
            mock_stats.matches_scraped = 10
            mock_stats.matches_converted = 10
            mock_stats.events_matched = 8
            mock_stats.events_created = 2
            mock_stats.snapshots_stored = 10
            mock_stats.errors = []
            mock_ingest.return_value = mock_stats

            result = await refresh_scrape(league="mlb", market="home_away")

            assert result["league"] == "mlb"
            assert result["matches_scraped"] == 10

            # Verify the spec passed to ingest_league has sport="baseball"
            call_args = mock_ingest.call_args
            spec = call_args[0][0]
            assert spec.sport == "baseball"
            assert spec.sport_key == "baseball_mlb"


class TestGetScrapeStatus:
    @pytest.mark.asyncio
    async def test_surfaces_completed_job_result(self) -> None:
        from dataclasses import dataclass, field
        from uuid import uuid4

        from odds_mcp.server import get_scrape_status

        @dataclass
        class FakeStats:
            league: str = "england-premier-league"
            matches_scraped: int = 12
            matches_converted: int = 12
            events_matched: int = 10
            events_created: int = 2
            snapshots_stored: int = 12
            errors: list[str] = field(default_factory=list)

        job_id = uuid4()
        mock_outcome = MagicMock()
        mock_outcome.name = "success"
        mock_result = MagicMock()
        mock_result.outcome = mock_outcome
        mock_result.return_value = FakeStats()
        mock_result.started_at = datetime(2026, 4, 17, 10, 0, tzinfo=UTC)
        mock_result.finished_at = datetime(2026, 4, 17, 10, 5, tzinfo=UTC)
        mock_result.exception = None

        mock_scheduler = AsyncMock()
        mock_scheduler.get_jobs = AsyncMock(return_value=[])
        mock_scheduler.get_job_result = AsyncMock(return_value=mock_result)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_scheduler)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("odds_lambda.scheduling.backends.local.build_scheduler", return_value=mock_cm):
            response = await get_scrape_status(job_id=str(job_id))

        assert response["status"] == "ok"
        assert response["pending_scrape_jobs"] == 0
        assert response["result"]["state"] == "completed"
        assert response["result"]["outcome"] == "success"
        assert response["result"]["stats"]["matches_scraped"] == 12
        assert response["result"]["stats"]["events_matched"] == 10
        assert response["result"]["exception"] is None

    @pytest.mark.asyncio
    async def test_pending_job_reports_state(self) -> None:
        from uuid import uuid4

        from odds_mcp.server import get_scrape_status

        job_id = uuid4()
        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.task_id = "odds_lambda.jobs.fetch_oddsportal.ingest_league"
        mock_job.created_at = datetime(2026, 4, 17, 10, 0, tzinfo=UTC)
        mock_job.acquired_by = None

        mock_scheduler = AsyncMock()
        mock_scheduler.get_jobs = AsyncMock(return_value=[mock_job])
        mock_scheduler.get_job_result = AsyncMock()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_scheduler)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("odds_lambda.scheduling.backends.local.build_scheduler", return_value=mock_cm):
            response = await get_scrape_status(job_id=str(job_id))

        assert response["pending_scrape_jobs"] == 1
        assert response["jobs"][0]["state"] == "pending"
        assert "result" not in response
        mock_scheduler.get_job_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_job_id_reports_unknown(self) -> None:
        from uuid import uuid4

        from odds_mcp.server import get_scrape_status

        job_id = uuid4()
        mock_scheduler = AsyncMock()
        mock_scheduler.get_jobs = AsyncMock(return_value=[])
        mock_scheduler.get_job_result = AsyncMock(return_value=None)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_scheduler)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("odds_lambda.scheduling.backends.local.build_scheduler", return_value=mock_cm):
            response = await get_scrape_status(job_id=str(job_id))

        assert response["result"]["state"] == "unknown"


class TestGetUpcomingFixtures:
    @pytest.mark.asyncio
    async def test_returns_fixtures(self) -> None:
        from odds_mcp.server import get_upcoming_fixtures

        mock_event = MagicMock(spec=Event)
        mock_event.id = "evt-1"
        mock_event.sport_key = "soccer_epl"
        mock_event.sport_title = "EPL"
        mock_event.home_team = "Arsenal"
        mock_event.away_team = "Chelsea"
        mock_event.commence_time = datetime(2026, 4, 12, 15, 0, tzinfo=UTC)
        mock_event.status = EventStatus.SCHEDULED
        mock_event.home_score = None
        mock_event.away_score = None
        mock_event.completed_at = None

        mock_reader = AsyncMock()
        mock_reader.get_events_by_date_range = AsyncMock(return_value=[mock_event])

        with (
            patch("odds_mcp.server.async_session_maker") as mock_session_maker,
            patch("odds_mcp.server.OddsReader", return_value=mock_reader),
        ):
            mock_session_maker.return_value.__aenter__ = AsyncMock()
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            result = await get_upcoming_fixtures()
            assert len(result) == 1
            assert result[0]["id"] == "evt-1"


class TestGetCurrentOddsMissingEvent:
    @pytest.mark.asyncio
    async def test_missing_event(self) -> None:
        from odds_mcp.server import get_current_odds

        mock_reader = AsyncMock()
        mock_reader.get_event_by_id = AsyncMock(return_value=None)

        with (
            patch("odds_mcp.server.async_session_maker") as mock_session_maker,
            patch("odds_mcp.server.OddsReader", return_value=mock_reader),
        ):
            mock_session_maker.return_value.__aenter__ = AsyncMock()
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            result = await get_current_odds("nonexistent")
            assert "error" in result
            assert "nonexistent" in result["error"]


class TestGetPredictions:
    def _make_event(self) -> MagicMock:
        event = MagicMock(spec=Event)
        event.id = "evt-1"
        event.sport_key = "soccer_epl"
        event.sport_title = "EPL"
        event.home_team = "Arsenal"
        event.away_team = "Chelsea"
        event.commence_time = datetime(2026, 4, 12, 15, 0, tzinfo=UTC)
        event.status = EventStatus.SCHEDULED
        event.home_score = None
        event.away_score = None
        event.completed_at = None
        return event

    def _make_prediction(self, id: int, hours_ago: float, clv: float) -> MagicMock:
        from odds_core.prediction_models import Prediction

        pred = MagicMock(spec=Prediction)
        pred.id = id
        pred.snapshot_id = id * 100
        pred.model_name = "epl-clv-home"
        pred.model_version = "v1"
        pred.predicted_clv = clv
        pred.created_at = datetime(2026, 4, 12, 15, 0, tzinfo=UTC)
        return pred

    @pytest.mark.asyncio
    async def test_missing_event(self) -> None:
        from odds_mcp.server import get_predictions

        mock_reader = AsyncMock()
        mock_reader.get_event_by_id = AsyncMock(return_value=None)

        with (
            patch("odds_mcp.server.async_session_maker") as mock_session_maker,
            patch("odds_mcp.server.OddsReader", return_value=mock_reader),
        ):
            mock_session_maker.return_value.__aenter__ = AsyncMock()
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            result = await get_predictions("nonexistent")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_default_limit(self) -> None:
        from odds_mcp.server import get_predictions

        preds = [self._make_prediction(i, i, 0.01 * i) for i in range(1, 11)]

        mock_reader = AsyncMock()
        mock_reader.get_event_by_id = AsyncMock(return_value=self._make_event())

        # Mock session.execute to return count then predictions
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 10

        mock_pred_result = MagicMock()
        mock_pred_result.scalars.return_value.all.return_value = preds[:5]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_pred_result])

        with (
            patch("odds_mcp.server.async_session_maker") as mock_session_maker,
            patch("odds_mcp.server.OddsReader", return_value=mock_reader),
        ):
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            result = await get_predictions("evt-1")
            assert result["total_matching"] == 10
            assert result["returned"] == 5
            assert len(result["predictions"]) == 5

    @pytest.mark.asyncio
    async def test_custom_limit(self) -> None:
        from odds_mcp.server import get_predictions

        preds = [self._make_prediction(i, i, 0.01 * i) for i in range(1, 4)]

        mock_reader = AsyncMock()
        mock_reader.get_event_by_id = AsyncMock(return_value=self._make_event())

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 10

        mock_pred_result = MagicMock()
        mock_pred_result.scalars.return_value.all.return_value = preds

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_pred_result])

        with (
            patch("odds_mcp.server.async_session_maker") as mock_session_maker,
            patch("odds_mcp.server.OddsReader", return_value=mock_reader),
        ):
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            result = await get_predictions("evt-1", limit=3)
            assert result["returned"] == 3

    @pytest.mark.asyncio
    async def test_no_predictions(self) -> None:
        from odds_mcp.server import get_predictions

        mock_reader = AsyncMock()
        mock_reader.get_event_by_id = AsyncMock(return_value=self._make_event())

        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_pred_result = MagicMock()
        mock_pred_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_pred_result])

        with (
            patch("odds_mcp.server.async_session_maker") as mock_session_maker,
            patch("odds_mcp.server.OddsReader", return_value=mock_reader),
        ):
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            result = await get_predictions("evt-1")
            assert result["total_matching"] == 0
            assert result["returned"] == 0
            assert result["predictions"] == []


class TestPaperBetValidation:
    @pytest.mark.asyncio
    async def test_rejects_non_scheduled_event(self) -> None:
        from odds_mcp.server import paper_bet

        mock_event = MagicMock(spec=Event)
        mock_event.id = "evt-1"
        mock_event.status = EventStatus.FINAL

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_event

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("odds_mcp.server.async_session_maker") as mock_session_maker:
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            result = await paper_bet(
                event_id="evt-1",
                market="h2h",
                selection="home",
                odds=-110,
                stake=10.0,
            )
            assert "error" in result
            assert "final" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_event(self) -> None:
        from odds_mcp.server import paper_bet

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("odds_mcp.server.async_session_maker") as mock_session_maker:
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            result = await paper_bet(
                event_id="nonexistent",
                market="h2h",
                selection="home",
                odds=-110,
                stake=10.0,
            )
            assert "error" in result
            assert "nonexistent" in result["error"]
