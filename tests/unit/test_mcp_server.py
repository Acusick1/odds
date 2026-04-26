"""Unit tests for MCP server serialization helpers and error paths."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.paper_trade_models import PaperTrade, TradeResult


class TestCoerceBookmakerList:
    def test_none_passthrough(self) -> None:
        from odds_mcp.server import _coerce_bookmaker_list

        assert _coerce_bookmaker_list(None) is None

    def test_list_passthrough(self) -> None:
        from odds_mcp.server import _coerce_bookmaker_list

        assert _coerce_bookmaker_list(["bet365", "betway"]) == ["bet365", "betway"]

    def test_empty_string_becomes_none(self) -> None:
        from odds_mcp.server import _coerce_bookmaker_list

        assert _coerce_bookmaker_list("") is None
        assert _coerce_bookmaker_list("   ") is None

    def test_json_array_string(self) -> None:
        from odds_mcp.server import _coerce_bookmaker_list

        assert _coerce_bookmaker_list('["bet365", "betway"]') == ["bet365", "betway"]

    def test_json_array_with_whitespace(self) -> None:
        from odds_mcp.server import _coerce_bookmaker_list

        assert _coerce_bookmaker_list('  ["pinnacle", "betfair_exchange"]  ') == [
            "pinnacle",
            "betfair_exchange",
        ]

    def test_comma_separated_string(self) -> None:
        from odds_mcp.server import _coerce_bookmaker_list

        assert _coerce_bookmaker_list("bet365,betway,betfred") == ["bet365", "betway", "betfred"]

    def test_comma_separated_with_whitespace(self) -> None:
        from odds_mcp.server import _coerce_bookmaker_list

        assert _coerce_bookmaker_list("bet365, betway , betfred") == [
            "bet365",
            "betway",
            "betfred",
        ]

    def test_single_value_string(self) -> None:
        from odds_mcp.server import _coerce_bookmaker_list

        assert _coerce_bookmaker_list("pinnacle") == ["pinnacle"]

    def test_malformed_json_raises(self) -> None:
        from odds_mcp.server import _coerce_bookmaker_list

        with pytest.raises(ValueError, match="Malformed JSON array"):
            _coerce_bookmaker_list("[bet365, betway")


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
        from uuid import uuid4

        from odds_mcp.server import refresh_scrape

        job_id = uuid4()
        mock_scheduler = AsyncMock()
        mock_scheduler.add_job = AsyncMock(return_value=job_id)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_scheduler)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("odds_lambda.scheduling.backends.local.build_scheduler", return_value=mock_cm):
            result = await refresh_scrape(league="mlb", market="home_away")

        assert result["league"] == "mlb"
        assert result["market"] == "home_away"
        assert result["job_id"] == str(job_id)

        # Verify the LeagueSpec passed to the scheduler has the MLB sport identifiers
        call_kwargs = mock_scheduler.add_job.call_args.kwargs
        spec = call_kwargs["args"][0]
        assert spec.sport == "baseball"
        assert spec.sport_key == "baseball_mlb"
        assert spec.markets == ["home_away"]


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

            result = await get_current_odds("nonexistent", market="h2h")
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


class TestFindRetailEdges:
    """Unit tests for the ``find_retail_edges`` MCP tool."""

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

    def _make_odds(
        self,
        bookmaker: str,
        outcome: str,
        price: int,
        *,
        market: str = "h2h",
        point: float | None = None,
    ) -> MagicMock:
        o = MagicMock()
        o.bookmaker_key = bookmaker
        o.bookmaker_title = bookmaker
        o.market_key = market
        o.outcome_name = outcome
        o.price = price
        o.point = point
        return o

    def _make_snapshot(self) -> MagicMock:
        snap = MagicMock()
        snap.id = 1
        snap.snapshot_time = datetime(2026, 4, 12, 10, 0, tzinfo=UTC)
        return snap

    def _make_sharp_result(
        self,
        prices: dict[str, dict[str, object]] | None = None,
    ) -> MagicMock:
        from odds_core.match_brief_models import SharpPriceMeta

        default_prices = {
            "Arsenal": {"bookmaker": "pinnacle", "price": -110, "implied_prob": 0.524},
            "Chelsea": {"bookmaker": "pinnacle", "price": -110, "implied_prob": 0.524},
        }
        prices = prices if prices is not None else default_prices

        result = MagicMock()
        result.prices = prices
        result.meta = {
            outcome: SharpPriceMeta(
                snapshot_id=1,
                snapshot_time=datetime(2026, 4, 12, 10, 0, tzinfo=UTC),
                age_seconds=0.0,
            )
            for outcome in prices
        }
        return result

    def _make_book_result(
        self,
        odds: list[MagicMock],
        snapshot_time: datetime | None = None,
    ) -> MagicMock:
        """Build a BookPriceResult-like mock from a flat list of Odds mocks."""
        from odds_lambda.storage.readers import BookPriceEntry, BookPriceMeta, BookPriceResult

        snap_time = snapshot_time or datetime(2026, 4, 12, 10, 0, tzinfo=UTC)
        entries: dict[tuple[str, str, float | None], BookPriceEntry] = {}
        for o in odds:
            key = (o.bookmaker_key, o.outcome_name, o.point)
            # First-wins: matches OddsReader.get_latest_book_prices, which walks
            # snapshots newest-first and skips a key once already present
            # (readers.py: ``if key in book_result.entries: continue``).
            if key in entries:
                continue
            entries[key] = BookPriceEntry(
                odds=o,
                meta=BookPriceMeta(
                    snapshot_id=1,
                    snapshot_time=snap_time,
                    age_seconds=0.0,
                ),
            )
        return BookPriceResult(entries=entries)

    def _mock_reader(
        self,
        sharp_result: MagicMock | None = None,
        book_result: object | None = None,
    ) -> AsyncMock:
        reader = AsyncMock()
        reader.get_event_by_id = AsyncMock(return_value=self._make_event())
        reader.get_latest_snapshot = AsyncMock(return_value=self._make_snapshot())
        reader.get_sharp_prices = AsyncMock(return_value=sharp_result or self._make_sharp_result())
        if book_result is not None:
            reader.get_latest_book_prices = AsyncMock(return_value=book_result)
        return reader

    async def _call(self, odds: list, reader: AsyncMock, market: str = "h2h") -> dict:
        from odds_mcp.server import find_retail_edges

        # Build a BookPriceResult from the supplied odds list to feed the
        # reader's get_latest_book_prices mock — replaces the old
        # extract_odds_from_snapshot patch.
        reader.get_latest_book_prices = AsyncMock(return_value=self._make_book_result(odds))

        with (
            patch("odds_mcp.server.async_session_maker") as mock_session_maker,
            patch("odds_mcp.server.OddsReader", return_value=reader),
        ):
            mock_session_maker.return_value.__aenter__ = AsyncMock()
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            return await find_retail_edges(event_id="evt-1", market=market)

    @pytest.mark.asyncio
    async def test_response_shape(self) -> None:
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("pinnacle", "Chelsea", -110),
            self._make_odds("bet365", "Arsenal", -105),
            self._make_odds("bet365", "Chelsea", -115),
            self._make_odds("betway", "Arsenal", -115),
            self._make_odds("betway", "Chelsea", -105),
        ]
        result = await self._call(odds, self._mock_reader())

        assert "event" in result
        assert "snapshot_time" in result
        assert result["sharp_bookmakers"] == ["pinnacle", "betfair_exchange"]
        assert isinstance(result["per_outcome"], list)
        assert isinstance(result["retail_edges"], list)

        # One entry per (outcome, point) — both with point=None
        assert {(e["outcome"], e["point"]) for e in result["per_outcome"]} == {
            ("Arsenal", None),
            ("Chelsea", None),
        }
        for entry in result["per_outcome"]:
            assert set(entry.keys()) == {
                "outcome",
                "point",
                "sharp_implied_prob",
                "sharp_snapshot_time",
                "sharp_age_seconds",
                "best_retail",
                "worst_retail",
                "n_books",
                "median_divergence",
                "dispersion_stddev",
            }
            assert entry["n_books"] == 2
            for side in ("best_retail", "worst_retail"):
                row = entry[side]
                assert set(row.keys()) == {
                    "book",
                    "price",
                    "implied_prob",
                    "divergence",
                    "z_score",
                    "market_hold",
                    "snapshot_time",
                    "book_age_seconds",
                }

    @pytest.mark.asyncio
    async def test_sign_convention_negative_divergence_longer_than_sharp(self) -> None:
        # midnite prices Arsenal *longer* than sharp → negative divergence, lands in retail_edges.
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("pinnacle", "Chelsea", -110),
            self._make_odds("bet365", "Arsenal", -115),  # shorter than sharp
            self._make_odds("bet365", "Chelsea", -105),
            self._make_odds("betway", "Arsenal", -120),  # shorter than sharp
            self._make_odds("betway", "Chelsea", -100),
            self._make_odds("midnite", "Arsenal", +110),  # longer than sharp
            self._make_odds("midnite", "Chelsea", -140),
        ]
        result = await self._call(odds, self._mock_reader())

        edges = result["retail_edges"]
        assert len(edges) >= 1
        # midnite on Arsenal should be in edges; divergence should be negative
        midnite_arsenal = [e for e in edges if e["book"] == "midnite" and e["outcome"] == "Arsenal"]
        assert len(midnite_arsenal) == 1
        assert midnite_arsenal[0]["divergence"] < 0

    @pytest.mark.asyncio
    async def test_retail_edges_contains_only_negatives(self) -> None:
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("pinnacle", "Chelsea", -110),
            self._make_odds("bet365", "Arsenal", -115),
            self._make_odds("bet365", "Chelsea", -105),
            self._make_odds("betway", "Arsenal", -120),
            self._make_odds("betway", "Chelsea", -100),
            self._make_odds("midnite", "Arsenal", +110),
            self._make_odds("midnite", "Chelsea", -140),
        ]
        result = await self._call(odds, self._mock_reader())

        for edge in result["retail_edges"]:
            assert edge["divergence"] < 0

    @pytest.mark.asyncio
    async def test_empty_retail_edges_when_all_retail_shorter(self) -> None:
        # Every retail book is shorter than sharp on every outcome.
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("pinnacle", "Chelsea", -110),
            self._make_odds("bet365", "Arsenal", -130),
            self._make_odds("bet365", "Chelsea", -130),
            self._make_odds("betway", "Arsenal", -125),
            self._make_odds("betway", "Chelsea", -125),
        ]
        result = await self._call(odds, self._mock_reader())

        assert result["retail_edges"] == []

    @pytest.mark.asyncio
    async def test_tie_break_determinism_on_equal_divergence(self) -> None:
        # Two books with identical divergence on the same outcome — tie-break
        # must sort alphabetically by book name.
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("pinnacle", "Chelsea", -110),
            # zeta_book and alpha_book: identical longer-than-sharp price on Arsenal
            self._make_odds("zeta_book", "Arsenal", +110),
            self._make_odds("zeta_book", "Chelsea", -140),
            self._make_odds("alpha_book", "Arsenal", +110),
            self._make_odds("alpha_book", "Chelsea", -140),
            self._make_odds("bet365", "Arsenal", -115),
            self._make_odds("bet365", "Chelsea", -105),
        ]
        result = await self._call(odds, self._mock_reader())

        # Two Arsenal edges: alpha_book must come before zeta_book
        arsenal_edges = [e for e in result["retail_edges"] if e["outcome"] == "Arsenal"]
        arsenal_books = [e["book"] for e in arsenal_edges]
        assert arsenal_books.index("alpha_book") < arsenal_books.index("zeta_book")

        # best_retail tie-break on Arsenal: alphabetical
        arsenal_bucket = next(e for e in result["per_outcome"] if e["outcome"] == "Arsenal")
        assert arsenal_bucket["best_retail"]["book"] == "alpha_book"

    @pytest.mark.asyncio
    async def test_missing_sharp_nulls_divergence_and_zscore(self) -> None:
        # Sharp has only Arsenal — Chelsea sharp is missing.
        sharp_result = self._make_sharp_result(
            {"Arsenal": {"bookmaker": "pinnacle", "price": -110, "implied_prob": 0.524}}
        )
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("bet365", "Arsenal", -105),
            self._make_odds("bet365", "Chelsea", -115),
            self._make_odds("betway", "Arsenal", -120),
            self._make_odds("betway", "Chelsea", -105),
        ]
        result = await self._call(odds, self._mock_reader(sharp_result=sharp_result))

        chelsea = next(e for e in result["per_outcome"] if e["outcome"] == "Chelsea")
        assert chelsea["sharp_implied_prob"] is None
        # best_retail still populated, but divergence/z_score null
        assert chelsea["best_retail"] is not None
        assert chelsea["best_retail"]["divergence"] is None
        assert chelsea["best_retail"]["z_score"] is None
        assert chelsea["worst_retail"]["divergence"] is None
        assert chelsea["worst_retail"]["z_score"] is None

        # Chelsea produces no entries in retail_edges (no divergence to rank on)
        assert not any(e["outcome"] == "Chelsea" for e in result["retail_edges"])

    @pytest.mark.asyncio
    async def test_market_hold_populated_on_h2h(self) -> None:
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("pinnacle", "Chelsea", -110),
            self._make_odds("bet365", "Arsenal", -105),
            self._make_odds("bet365", "Chelsea", -115),
        ]
        result = await self._call(odds, self._mock_reader())

        for entry in result["per_outcome"]:
            assert entry["best_retail"]["market_hold"] is not None
            assert isinstance(entry["best_retail"]["market_hold"], float)

    @pytest.mark.asyncio
    async def test_market_hold_populated_on_1x2(self) -> None:
        sharp_result = self._make_sharp_result(
            {
                "Arsenal": {"bookmaker": "pinnacle", "price": -120, "implied_prob": 0.545},
                "Draw": {"bookmaker": "pinnacle", "price": 280, "implied_prob": 0.263},
                "Chelsea": {"bookmaker": "pinnacle", "price": 310, "implied_prob": 0.244},
            }
        )
        odds = [
            self._make_odds("pinnacle", "Arsenal", -120, market="1x2"),
            self._make_odds("pinnacle", "Draw", 280, market="1x2"),
            self._make_odds("pinnacle", "Chelsea", 310, market="1x2"),
            self._make_odds("bet365", "Arsenal", -130, market="1x2"),
            self._make_odds("bet365", "Draw", 260, market="1x2"),
            self._make_odds("bet365", "Chelsea", 290, market="1x2"),
        ]
        result = await self._call(odds, self._mock_reader(sharp_result=sharp_result), market="1x2")

        for entry in result["per_outcome"]:
            assert entry["best_retail"]["market_hold"] is not None

    @pytest.mark.asyncio
    async def test_market_hold_null_on_totals(self) -> None:
        sharp_result = self._make_sharp_result(
            {
                "Over": {"bookmaker": "pinnacle", "price": -110, "implied_prob": 0.524},
                "Under": {"bookmaker": "pinnacle", "price": -110, "implied_prob": 0.524},
            }
        )
        odds = [
            self._make_odds("pinnacle", "Over", -110, market="totals", point=2.5),
            self._make_odds("pinnacle", "Under", -110, market="totals", point=2.5),
            self._make_odds("bet365", "Over", -105, market="totals", point=2.5),
            self._make_odds("bet365", "Under", -115, market="totals", point=2.5),
        ]
        result = await self._call(
            odds, self._mock_reader(sharp_result=sharp_result), market="totals"
        )

        for entry in result["per_outcome"]:
            assert entry["best_retail"]["market_hold"] is None
            assert entry["worst_retail"]["market_hold"] is None

    @pytest.mark.asyncio
    async def test_market_hold_null_on_spreads(self) -> None:
        sharp_result = self._make_sharp_result(
            {
                "Arsenal": {"bookmaker": "pinnacle", "price": -110, "implied_prob": 0.524},
                "Chelsea": {"bookmaker": "pinnacle", "price": -110, "implied_prob": 0.524},
            }
        )
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110, market="spreads", point=-1.5),
            self._make_odds("pinnacle", "Chelsea", -110, market="spreads", point=1.5),
            self._make_odds("bet365", "Arsenal", -105, market="spreads", point=-1.5),
            self._make_odds("bet365", "Chelsea", -115, market="spreads", point=1.5),
        ]
        result = await self._call(
            odds, self._mock_reader(sharp_result=sharp_result), market="spreads"
        )

        for entry in result["per_outcome"]:
            assert entry["best_retail"]["market_hold"] is None

    @pytest.mark.asyncio
    async def test_partial_book_market_hold_none(self) -> None:
        # partial_book only has Arsenal, priced LONGER than sharp so it wins
        # best_retail on that outcome. It is dropped from per_book_market_holds
        # (missing Chelsea), so its row must carry market_hold=None — and that
        # null propagates into best_retail.
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("pinnacle", "Chelsea", -110),
            self._make_odds("bet365", "Arsenal", -105),
            self._make_odds("bet365", "Chelsea", -115),
            self._make_odds("betway", "Arsenal", -115),
            self._make_odds("betway", "Chelsea", -105),
            # partial_book Arsenal at +120 → implied 0.4545 (longest on Arsenal)
            self._make_odds("partial_book", "Arsenal", +120),
        ]
        result = await self._call(odds, self._mock_reader())

        arsenal_bucket = next(e for e in result["per_outcome"] if e["outcome"] == "Arsenal")

        # partial_book wins best_retail (lowest implied_prob / longest price)
        assert arsenal_bucket["best_retail"]["book"] == "partial_book"
        # and its market_hold must be None because partial_book is dropped
        # from per_book_market_holds (it is missing the Chelsea outcome).
        assert arsenal_bucket["best_retail"]["market_hold"] is None

        # Its divergence is negative, so it surfaces in retail_edges — same null hold.
        edges = [
            e
            for e in result["retail_edges"]
            if e["book"] == "partial_book" and e["outcome"] == "Arsenal"
        ]
        assert len(edges) == 1
        assert edges[0]["market_hold"] is None

        # Full-book retail rows still report a float hold (sanity: bet365 on Chelsea).
        chelsea_bucket = next(e for e in result["per_outcome"] if e["outcome"] == "Chelsea")
        assert isinstance(chelsea_bucket["best_retail"]["market_hold"], float)

    @pytest.mark.asyncio
    async def test_zscore_hand_checked_fixture(self) -> None:
        # Sharp implied = 0.524 on Arsenal. Retail divergences chosen so median and
        # stddev are hand-computable.
        # Sharp prob for -110 = 0.5238095... We use 0.524 for sharp_implied_prob.
        #
        # Retail divergences (d_i = retail_prob - 0.524) for 4 books:
        #   A: retail_prob = 0.524  → d = 0.000
        #   B: retail_prob = 0.540  → d = 0.016
        #   C: retail_prob = 0.508  → d = -0.016
        #   D: retail_prob = 0.532  → d = 0.008
        # median = (0.000 + 0.008) / 2 = 0.004
        # pstdev = sqrt(mean((d - mean)^2)); mean = 0.002
        # devs^2 = (-.002)^2 + (.014)^2 + (-.018)^2 + (.006)^2
        #        = .000004 + .000196 + .000324 + .000036 = .000560
        # var = .000560 / 4 = .000140; stddev = sqrt(.000140) ~ 0.011832
        # z for book C (-0.016): (-0.016 - 0.004) / 0.011832 ~ -1.6903

        # To produce those retail_probs exactly we pick American odds that yield them.
        # implied_prob = 100 / (price + 100) for positive; -price / (-price + 100) for negative.
        # Solve for each:
        #   0.524 → price -110.08 (use -110 → 0.523809 ≈ 0.524)
        #   0.540 → price -117.39 (use -117 → 0.539171)
        #   0.508 → price -103.25 (use -103 → 0.507389)
        #   0.532 → price -113.67 (use -114 → 0.532710)
        # These are approximations; we compute expected values programmatically below.
        sharp_prob = round(1 / (1 + 100 / 110), 6)  # 0.523810
        sharp_result = self._make_sharp_result(
            {
                "Arsenal": {"bookmaker": "pinnacle", "price": -110, "implied_prob": sharp_prob},
                "Chelsea": {"bookmaker": "pinnacle", "price": -110, "implied_prob": sharp_prob},
            }
        )

        # Four retail books with hand-picked Arsenal prices.
        retail_prices = {"book_a": -110, "book_b": -130, "book_c": -104, "book_d": -117}
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("pinnacle", "Chelsea", -110),
        ]
        for book, price in retail_prices.items():
            odds.append(self._make_odds(book, "Arsenal", price))
            # Symmetric Chelsea price so per-book market hold is sensible
            odds.append(self._make_odds(book, "Chelsea", -110))

        result = await self._call(odds, self._mock_reader(sharp_result=sharp_result))

        arsenal_bucket = next(e for e in result["per_outcome"] if e["outcome"] == "Arsenal")
        assert arsenal_bucket["n_books"] == 4

        # Expected divergences
        from odds_core.odds_math import calculate_implied_probability

        expected_divs = {
            book: round(round(calculate_implied_probability(price), 6) - sharp_prob, 6)
            for book, price in retail_prices.items()
        }
        divs_sorted = sorted(expected_divs.values())
        import statistics as pystats

        expected_median = round(pystats.median(divs_sorted), 6)
        expected_stddev = round(pystats.pstdev(divs_sorted), 6)

        assert arsenal_bucket["median_divergence"] == expected_median
        assert arsenal_bucket["dispersion_stddev"] == expected_stddev

        # z_score on the best_retail row should match formula
        best = arsenal_bucket["best_retail"]
        expected_best_z = round((best["divergence"] - expected_median) / expected_stddev, 6)
        assert best["z_score"] == expected_best_z

        # Check that at least one edge row has the same z_score (the one for best_retail
        # if its divergence is negative)
        if best["divergence"] < 0:
            matching = [
                e
                for e in result["retail_edges"]
                if e["outcome"] == "Arsenal" and e["book"] == best["book"]
            ]
            assert len(matching) == 1
            assert matching[0]["z_score"] == expected_best_z

    @pytest.mark.asyncio
    async def test_no_retail_books_when_all_books_are_sharp(self) -> None:
        # Every book in the snapshot is in the configured sharp set, so there
        # is no retail row on any outcome. per_outcome / retail_edges are both
        # empty, and the top-level response is still well-formed.
        odds = [
            self._make_odds("pinnacle", "Arsenal", -110),
            self._make_odds("pinnacle", "Chelsea", -110),
            self._make_odds("betfair_exchange", "Arsenal", -108),
            self._make_odds("betfair_exchange", "Chelsea", -112),
        ]
        result = await self._call(odds, self._mock_reader())

        assert "error" not in result
        assert result["sharp_bookmakers"] == ["pinnacle", "betfair_exchange"]
        assert result["snapshot_time"] is not None
        assert result["per_outcome"] == []
        assert result["retail_edges"] == []
