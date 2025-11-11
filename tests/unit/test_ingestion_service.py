"""Unit tests for the odds ingestion service."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.api_models import OddsResponse, api_dict_to_event
from odds_core.models import Event, FetchLog, Odds, OddsSnapshot
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.ingestion import (
    EventIngestionFailure,
    OddsIngestionCallbacks,
    OddsIngestionService,
)
from sqlalchemy import select


class _StubClient:
    """Async context manager returning a prebuilt odds response."""

    def __init__(self, response: OddsResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get_odds(self, *args, **kwargs) -> OddsResponse:
        return self._response


class TestOddsIngestionService:
    """Validate ingestion behaviour against the database."""

    @pytest.mark.asyncio
    async def test_ingest_sport_persists_data_and_invokes_callbacks(
        self,
        mock_settings,
        mock_session_factory,
        test_session,
    ):
        """Successful ingestion stores events, snapshots, odds, and logs callbacks."""

        event_payload = {
            "id": "ingest_event_success",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": datetime.now(UTC).isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Los Angeles Lakers", "price": -120},
                                {"name": "Boston Celtics", "price": 110},
                            ],
                        }
                    ],
                }
            ],
        }
        event = api_dict_to_event(event_payload)
        response = OddsResponse(
            events=[event],
            raw_events_data=[event_payload],
            response_time_ms=150,
            quota_remaining=12345,
            timestamp=datetime.now(UTC),
        )

        callbacks_record: dict[str, list] = {
            "fetch_complete": [],
            "processed": [],
            "failed": [],
        }

        callbacks = OddsIngestionCallbacks(
            on_fetch_complete=lambda resp: callbacks_record["fetch_complete"].append(resp),
            on_event_processed=lambda event_id: callbacks_record["processed"].append(event_id),
            on_event_failed=lambda event_id, exc: callbacks_record["failed"].append(
                (event_id, exc)
            ),
        )

        service = OddsIngestionService(
            settings=mock_settings,
            session_factory=mock_session_factory,
            client_factory=lambda: _StubClient(response),
        )

        result = await service.ingest_sport(
            "basketball_nba",
            fetch_tier=FetchTier.CLOSING,
            callbacks=callbacks,
        )

        assert result.sport_key == "basketball_nba"
        assert result.total_events == 1
        assert result.processed_events == 1
        assert result.quota_remaining == 12345
        assert result.failures == []

        assert callbacks_record["fetch_complete"] == [response]
        assert callbacks_record["processed"] == ["ingest_event_success"]
        assert callbacks_record["failed"] == []

        await test_session.rollback()

        db_event = (
            await test_session.execute(select(Event).where(Event.id == "ingest_event_success"))
        ).scalar_one()
        assert db_event.home_team == "Los Angeles Lakers"

        snapshot = (
            await test_session.execute(
                select(OddsSnapshot).where(OddsSnapshot.event_id == "ingest_event_success")
            )
        ).scalar_one()
        assert snapshot.bookmaker_count == 1
        assert snapshot.fetch_tier == FetchTier.CLOSING.value
        assert snapshot.hours_until_commence is not None

        odds_records = (
            (
                await test_session.execute(
                    select(Odds).where(Odds.event_id == "ingest_event_success")
                )
            )
            .scalars()
            .all()
        )
        assert len(odds_records) == 2

        fetch_log = (
            await test_session.execute(
                select(FetchLog).where(FetchLog.sport_key == "basketball_nba")
            )
        ).scalar_one()
        assert fetch_log.events_count == 1
        assert fetch_log.success is True
        assert fetch_log.api_quota_remaining == 12345

    @pytest.mark.asyncio
    async def test_ingest_sport_records_failures_when_snapshot_storage_fails(
        self,
        mock_settings,
        mock_session_factory,
        test_session,
        monkeypatch,
    ):
        """Errors during snapshot storage should be captured as failures without aborting."""

        event_payload = {
            "id": "ingest_event_failure",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": (datetime.now(UTC) + timedelta(hours=4)).isoformat(),
            "home_team": "Golden State Warriors",
            "away_team": "Miami Heat",
            "bookmakers": [],
        }
        event = api_dict_to_event(event_payload)
        response = OddsResponse(
            events=[event],
            raw_events_data=[event_payload],
            response_time_ms=75,
            quota_remaining=9999,
            timestamp=datetime.now(UTC),
        )

        async def _failing_store(*args, **kwargs):
            raise RuntimeError("snapshot_failure")

        monkeypatch.setattr("odds_lambda.ingestion.OddsWriter.store_odds_snapshot", _failing_store)

        callbacks_record: dict[str, list] = {"failed": []}
        callbacks = OddsIngestionCallbacks(
            on_event_failed=lambda event_id, exc: callbacks_record["failed"].append((event_id, exc))
        )

        service = OddsIngestionService(
            settings=mock_settings,
            session_factory=mock_session_factory,
            client_factory=lambda: _StubClient(response),
        )

        result = await service.ingest_sport(
            "basketball_nba",
            fetch_tier=FetchTier.PREGAME,
            callbacks=callbacks,
        )

        assert result.total_events == 1
        assert result.processed_events == 0
        assert result.error_count == 1

        failure = result.failures[0]
        assert isinstance(failure, EventIngestionFailure)
        assert failure.event_id == "ingest_event_failure"
        assert failure.error == "snapshot_failure"

        await test_session.rollback()

        fetch_log = (
            await test_session.execute(
                select(FetchLog).where(FetchLog.sport_key == "basketball_nba")
            )
        ).scalar_one()
        assert fetch_log.events_count == 1
        assert fetch_log.success is False

        failed_event_id, exc = callbacks_record["failed"][0]
        assert failed_event_id == "ingest_event_failure"
        assert isinstance(exc, RuntimeError)
