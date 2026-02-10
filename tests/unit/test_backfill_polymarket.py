"""Tests for Polymarket backfill job."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.polymarket_models import PolymarketMarket, PolymarketMarketType
from odds_lambda.jobs import backfill_polymarket


class TestBackfillMarketHistory:
    """Tests for _backfill_market_history function."""

    @pytest.mark.asyncio
    async def test_skips_already_backfilled_market(
        self,
        mock_polymarket_client,
        mock_polymarket_writer,
        sample_polymarket_market,
    ):
        """Market whose id is in backfilled_market_ids returns status=skipped."""
        backfilled_ids = {1}  # Market id 1 is already backfilled
        commence_time = datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC)

        result = await backfill_polymarket._backfill_market_history(
            client=mock_polymarket_client,
            writer=mock_polymarket_writer,
            market=sample_polymarket_market,
            commence_time=commence_time,
            backfilled_market_ids=backfilled_ids,
            dry_run=False,
        )

        assert result["status"] == "skipped"
        assert result["points"] == 0
        mock_polymarket_client.get_price_history.assert_not_called()
        mock_polymarket_writer.bulk_store_price_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_error_when_no_token_ids(
        self,
        mock_polymarket_client,
        mock_polymarket_writer,
        sample_polymarket_market,
    ):
        """Market with empty clob_token_ids returns status=error."""
        # Create market with no token IDs
        market_no_tokens = PolymarketMarket(
            id=2,
            polymarket_event_id=1,
            pm_market_id="test-market-no-tokens",
            condition_id="test-condition",
            question="Test Market",
            clob_token_ids=[],  # Empty token IDs
            outcomes=["A", "B"],
            market_type=PolymarketMarketType.MONEYLINE,
            active=False,
            closed=True,
        )
        commence_time = datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC)

        result = await backfill_polymarket._backfill_market_history(
            client=mock_polymarket_client,
            writer=mock_polymarket_writer,
            market=market_no_tokens,
            commence_time=commence_time,
            backfilled_market_ids=set(),
            dry_run=False,
        )

        assert result["status"] == "error"
        assert result["points"] == 0
        mock_polymarket_client.get_price_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_no_data_when_history_empty(
        self,
        mock_polymarket_client,
        mock_polymarket_writer,
        sample_polymarket_market,
    ):
        """Returns status=no_data when price history is empty."""
        mock_polymarket_client.get_price_history.return_value = []
        commence_time = datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC)

        result = await backfill_polymarket._backfill_market_history(
            client=mock_polymarket_client,
            writer=mock_polymarket_writer,
            market=sample_polymarket_market,
            commence_time=commence_time,
            backfilled_market_ids=set(),
            dry_run=False,
        )

        assert result["status"] == "no_data"
        assert result["points"] == 0
        mock_polymarket_writer.bulk_store_price_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_backfill_stores_data(
        self,
        mock_polymarket_client,
        mock_polymarket_writer,
        sample_polymarket_market,
        sample_price_history,
    ):
        """Successful backfill returns status=success with inserted count."""
        mock_polymarket_client.get_price_history.return_value = sample_price_history
        mock_polymarket_writer.bulk_store_price_history.return_value = len(sample_price_history)
        commence_time = datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC)

        with patch("odds_lambda.jobs.backfill_polymarket.asyncio.sleep") as mock_sleep:
            result = await backfill_polymarket._backfill_market_history(
                client=mock_polymarket_client,
                writer=mock_polymarket_writer,
                market=sample_polymarket_market,
                commence_time=commence_time,
                backfilled_market_ids=set(),
                dry_run=False,
            )

        assert result["status"] == "success"
        assert result["points"] == len(sample_price_history)
        mock_polymarket_writer.bulk_store_price_history.assert_called_once_with(
            market=sample_polymarket_market,
            history=sample_price_history,
            commence_time=commence_time,
        )
        # Verify rate limit sleep was called
        mock_sleep.assert_called_once_with(0.1)

    @pytest.mark.asyncio
    async def test_dry_run_skips_storage_but_throttles(
        self,
        mock_polymarket_client,
        mock_polymarket_writer,
        sample_polymarket_market,
        sample_price_history,
    ):
        """Dry run skips storage but still throttles API calls."""
        mock_polymarket_client.get_price_history.return_value = sample_price_history
        commence_time = datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC)

        with patch("odds_lambda.jobs.backfill_polymarket.asyncio.sleep") as mock_sleep:
            result = await backfill_polymarket._backfill_market_history(
                client=mock_polymarket_client,
                writer=mock_polymarket_writer,
                market=sample_polymarket_market,
                commence_time=commence_time,
                backfilled_market_ids=set(),
                dry_run=True,
            )

        assert result["status"] == "success"
        assert result["points"] == len(sample_price_history)
        mock_polymarket_writer.bulk_store_price_history.assert_not_called()
        # Dry run DOES call sleep after API call (rate limiting applies to both paths)
        mock_sleep.assert_called_once_with(0.1)

    @pytest.mark.asyncio
    async def test_rate_limit_sleep_called_after_store(
        self,
        mock_polymarket_client,
        mock_polymarket_writer,
        sample_polymarket_market,
        sample_price_history,
    ):
        """Verifies rate-limit delay is applied after successful store."""
        mock_polymarket_client.get_price_history.return_value = sample_price_history
        mock_polymarket_writer.bulk_store_price_history.return_value = 5
        commence_time = datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC)

        with patch("odds_lambda.jobs.backfill_polymarket.asyncio.sleep") as mock_sleep:
            await backfill_polymarket._backfill_market_history(
                client=mock_polymarket_client,
                writer=mock_polymarket_writer,
                market=sample_polymarket_market,
                commence_time=commence_time,
                backfilled_market_ids=set(),
                dry_run=False,
            )

        # Verify sleep called with 100ms delay
        mock_sleep.assert_called_once_with(0.1)


class TestFetchAllClosedEvents:
    """Tests for _fetch_all_closed_events pagination logic."""

    @pytest.mark.asyncio
    async def test_single_page_returns_all_events(self, mock_polymarket_client, sample_event_data):
        """Single page with fewer results than limit returns all events."""
        # Return fewer than 100 events (stops pagination)
        mock_polymarket_client.get_nba_events.return_value = [sample_event_data]

        events = await backfill_polymarket._fetch_all_closed_events(mock_polymarket_client)

        assert len(events) == 1
        assert events[0]["id"] == "event-123"
        mock_polymarket_client.get_nba_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_pages_accumulates_events(self, mock_polymarket_client):
        """Multiple pages accumulate events until empty response."""
        # Create 100 events for first page, 50 for second page
        first_page = [{"id": f"event-{i}", "markets": []} for i in range(100)]
        second_page = [{"id": f"event-{i}", "markets": []} for i in range(100, 150)]

        mock_polymarket_client.get_nba_events.side_effect = [
            first_page,  # First page (full)
            second_page,  # Second page (partial - stops pagination)
        ]

        events = await backfill_polymarket._fetch_all_closed_events(mock_polymarket_client)

        assert len(events) == 150
        assert mock_polymarket_client.get_nba_events.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_first_response_returns_empty(self, mock_polymarket_client):
        """Empty first response returns empty list immediately."""
        mock_polymarket_client.get_nba_events.return_value = []

        events = await backfill_polymarket._fetch_all_closed_events(mock_polymarket_client)

        assert events == []
        mock_polymarket_client.get_nba_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_pagination_stops_on_empty_response(self, mock_polymarket_client):
        """Pagination stops when empty list is returned."""
        first_page = [{"id": f"event-{i}", "markets": []} for i in range(100)]
        mock_polymarket_client.get_nba_events.side_effect = [
            first_page,
            [],  # Empty response - stops pagination
        ]

        events = await backfill_polymarket._fetch_all_closed_events(mock_polymarket_client)

        assert len(events) == 100
        assert mock_polymarket_client.get_nba_events.call_count == 2


class TestMainIntegration:
    """Integration tests for main backfill job function."""

    @pytest.mark.asyncio
    async def test_early_return_when_polymarket_disabled(self, mock_polymarket_settings):
        """Returns early when polymarket.enabled=False."""
        mock_polymarket_settings.polymarket.enabled = False

        with patch(
            "odds_lambda.jobs.backfill_polymarket.get_settings",
            return_value=mock_polymarket_settings,
        ):
            await backfill_polymarket.main(dry_run=True)

        # No assertions needed - just verifies no exceptions raised and early return

    @pytest.mark.asyncio
    async def test_processes_only_moneyline_by_default(
        self,
        mock_polymarket_client,
        mock_async_session,
        sample_event_data,
        sample_polymarket_market,
    ):
        """Only moneyline markets processed by default."""
        # Create markets of different types
        moneyline_market = PolymarketMarket(
            id=1,
            polymarket_event_id=1,
            pm_market_id="ml-market",
            condition_id="cond-ml",
            question="Lakers vs Celtics",
            clob_token_ids=["token-ml"],
            outcomes=["Lakers", "Celtics"],
            market_type=PolymarketMarketType.MONEYLINE,
            active=False,
            closed=True,
        )

        spread_market = PolymarketMarket(
            id=2,
            polymarket_event_id=1,
            pm_market_id="spread-market",
            condition_id="cond-spread",
            question="Lakers Spread: -6.5",
            clob_token_ids=["token-spread"],
            outcomes=["Lakers -6.5", "Celtics +6.5"],
            market_type=PolymarketMarketType.SPREAD,
            active=False,
            closed=True,
        )

        mock_event = MagicMock(
            id=1,
            pm_event_id="event-123",
            start_date=datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC),
        )

        with (
            patch("odds_lambda.jobs.backfill_polymarket.PolymarketClient") as mock_client_class,
            patch("odds_lambda.jobs.backfill_polymarket.async_session_maker") as mock_session_ctx,
            patch("odds_lambda.jobs.backfill_polymarket.get_settings") as mock_get_settings,
        ):
            mock_client_class.return_value = mock_polymarket_client
            mock_polymarket_client.get_nba_events.return_value = [sample_event_data]
            mock_polymarket_client.get_price_history.return_value = [{"t": 1705339200, "p": "0.5"}]

            mock_session_ctx.return_value = mock_async_session

            mock_reader = AsyncMock()
            mock_reader.get_backfilled_market_ids.return_value = set()

            mock_writer = AsyncMock()
            mock_writer.upsert_event.return_value = mock_event
            mock_writer.upsert_markets.return_value = [
                moneyline_market,
                spread_market,
            ]  # Both types
            mock_writer.bulk_store_price_history.return_value = 1

            mock_settings = MagicMock()
            mock_settings.polymarket.enabled = True
            mock_get_settings.return_value = mock_settings

            with (
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketReader",
                    return_value=mock_reader,
                ),
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketWriter",
                    return_value=mock_writer,
                ),
                patch("odds_lambda.jobs.backfill_polymarket.asyncio.sleep"),
            ):
                await backfill_polymarket.main(
                    include_spreads=False,  # Spreads NOT included
                    include_totals=False,
                    dry_run=False,
                )

            # Only 1 call for moneyline market, not 2
            assert mock_polymarket_client.get_price_history.call_count == 1

    @pytest.mark.asyncio
    async def test_includes_spreads_when_flag_set(
        self,
        mock_polymarket_client,
        mock_async_session,
        sample_event_data,
    ):
        """Spread markets processed when include_spreads=True."""
        moneyline_market = MagicMock(
            id=1,
            market_type=PolymarketMarketType.MONEYLINE,
            clob_token_ids=["token-ml"],
        )
        spread_market = MagicMock(
            id=2,
            market_type=PolymarketMarketType.SPREAD,
            clob_token_ids=["token-spread"],
        )

        mock_event = MagicMock(
            id=1,
            pm_event_id="event-123",
            start_date=datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC),
        )

        with (
            patch("odds_lambda.jobs.backfill_polymarket.PolymarketClient") as mock_client_class,
            patch("odds_lambda.jobs.backfill_polymarket.async_session_maker") as mock_session_ctx,
            patch("odds_lambda.jobs.backfill_polymarket.get_settings") as mock_get_settings,
        ):
            mock_client_class.return_value = mock_polymarket_client
            mock_polymarket_client.get_nba_events.return_value = [sample_event_data]
            mock_polymarket_client.get_price_history.return_value = [{"t": 1705339200, "p": "0.5"}]

            mock_session_ctx.return_value = mock_async_session

            mock_reader = AsyncMock()
            mock_reader.get_backfilled_market_ids.return_value = set()

            mock_writer = AsyncMock()
            mock_writer.upsert_event.return_value = mock_event
            mock_writer.upsert_markets.return_value = [moneyline_market, spread_market]
            mock_writer.bulk_store_price_history.return_value = 1

            mock_settings = MagicMock()
            mock_settings.polymarket.enabled = True
            mock_get_settings.return_value = mock_settings

            with (
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketReader",
                    return_value=mock_reader,
                ),
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketWriter",
                    return_value=mock_writer,
                ),
                patch("odds_lambda.jobs.backfill_polymarket.asyncio.sleep"),
            ):
                await backfill_polymarket.main(
                    include_spreads=True,  # Spreads INCLUDED
                    include_totals=False,
                    dry_run=False,
                )

            # 2 calls: one for moneyline, one for spread
            assert mock_polymarket_client.get_price_history.call_count == 2

    @pytest.mark.asyncio
    async def test_per_event_error_resilience(
        self,
        mock_polymarket_client,
        mock_async_session,
    ):
        """One event failing doesn't prevent processing next event."""
        event1 = {
            "id": "good-event",
            "ticker": "nba-good",
            "title": "Good Game",
            "slug": "good",
            "startDate": "2024-01-15T19:00:00Z",
            "endDate": "2024-01-15T22:00:00Z",
            "active": False,
            "closed": True,
            "markets": [],
        }

        event2 = {
            "id": "bad-event",
            "ticker": "nba-bad",
            "title": "Bad Game",
            "slug": "bad",
            "startDate": "2024-01-15T20:00:00Z",
            "endDate": "2024-01-15T23:00:00Z",
            "active": False,
            "closed": True,
            "markets": [],
        }

        with (
            patch("odds_lambda.jobs.backfill_polymarket.PolymarketClient") as mock_client_class,
            patch("odds_lambda.jobs.backfill_polymarket.async_session_maker") as mock_session_ctx,
            patch("odds_lambda.jobs.backfill_polymarket.get_settings") as mock_get_settings,
        ):
            mock_client_class.return_value = mock_polymarket_client
            mock_polymarket_client.get_nba_events.return_value = [event1, event2]

            mock_session_ctx.return_value = mock_async_session

            mock_reader = AsyncMock()
            mock_reader.get_backfilled_market_ids.return_value = set()

            mock_writer = AsyncMock()
            # First event succeeds, second fails
            mock_writer.upsert_event.side_effect = [
                MagicMock(id=1, pm_event_id="good-event"),
                Exception("Database error"),
            ]

            mock_settings = MagicMock()
            mock_settings.polymarket.enabled = True
            mock_get_settings.return_value = mock_settings

            with (
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketReader",
                    return_value=mock_reader,
                ),
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketWriter",
                    return_value=mock_writer,
                ),
            ):
                # Should not raise exception
                await backfill_polymarket.main(dry_run=True)

            # Both events attempted
            assert mock_writer.upsert_event.call_count == 2

    @pytest.mark.asyncio
    async def test_dry_run_skips_commit(
        self,
        mock_polymarket_client,
        mock_async_session,
        sample_event_data,
    ):
        """Dry run does not call session.commit."""
        with (
            patch("odds_lambda.jobs.backfill_polymarket.PolymarketClient") as mock_client_class,
            patch("odds_lambda.jobs.backfill_polymarket.async_session_maker") as mock_session_ctx,
            patch("odds_lambda.jobs.backfill_polymarket.get_settings") as mock_get_settings,
        ):
            mock_client_class.return_value = mock_polymarket_client
            mock_polymarket_client.get_nba_events.return_value = [sample_event_data]
            mock_polymarket_client.get_price_history.return_value = [{"t": 1705339200, "p": "0.5"}]

            mock_session_ctx.return_value = mock_async_session

            mock_reader = AsyncMock()
            mock_reader.get_backfilled_market_ids.return_value = set()

            mock_writer = AsyncMock()
            mock_writer.upsert_event.return_value = MagicMock(
                id=1,
                pm_event_id="event-123",
                start_date=datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC),
            )
            mock_writer.upsert_markets.return_value = [
                MagicMock(
                    id=1,
                    market_type=PolymarketMarketType.MONEYLINE,
                    clob_token_ids=["token"],
                )
            ]

            mock_settings = MagicMock()
            mock_settings.polymarket.enabled = True
            mock_get_settings.return_value = mock_settings

            with (
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketReader",
                    return_value=mock_reader,
                ),
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketWriter",
                    return_value=mock_writer,
                ),
                patch("odds_lambda.jobs.backfill_polymarket.asyncio.sleep"),
            ):
                await backfill_polymarket.main(dry_run=True)

            # Verify commit NOT called in dry run
            mock_async_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_dry_run_commits_changes(
        self,
        mock_polymarket_client,
        mock_async_session,
        sample_event_data,
    ):
        """Non-dry run calls session.commit."""
        with (
            patch("odds_lambda.jobs.backfill_polymarket.PolymarketClient") as mock_client_class,
            patch("odds_lambda.jobs.backfill_polymarket.async_session_maker") as mock_session_ctx,
            patch("odds_lambda.jobs.backfill_polymarket.get_settings") as mock_get_settings,
        ):
            mock_client_class.return_value = mock_polymarket_client
            mock_polymarket_client.get_nba_events.return_value = [sample_event_data]
            mock_polymarket_client.get_price_history.return_value = [{"t": 1705339200, "p": "0.5"}]

            mock_session_ctx.return_value = mock_async_session

            mock_reader = AsyncMock()
            mock_reader.get_backfilled_market_ids.return_value = set()

            mock_writer = AsyncMock()
            mock_writer.upsert_event.return_value = MagicMock(
                id=1,
                pm_event_id="event-123",
                start_date=datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC),
            )
            mock_writer.upsert_markets.return_value = [
                MagicMock(
                    id=1,
                    market_type=PolymarketMarketType.MONEYLINE,
                    clob_token_ids=["token"],
                )
            ]
            mock_writer.bulk_store_price_history.return_value = 1

            mock_settings = MagicMock()
            mock_settings.polymarket.enabled = True
            mock_get_settings.return_value = mock_settings

            with (
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketReader",
                    return_value=mock_reader,
                ),
                patch(
                    "odds_lambda.jobs.backfill_polymarket.PolymarketWriter",
                    return_value=mock_writer,
                ),
                patch("odds_lambda.jobs.backfill_polymarket.asyncio.sleep"),
            ):
                await backfill_polymarket.main(dry_run=False)

            # Verify commit WAS called
            mock_async_session.commit.assert_called()
