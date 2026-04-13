"""Tests for fetch_oddsportal_results job."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.models import Event, EventStatus
from odds_lambda.jobs.fetch_oddsportal_results import (
    DEFAULT_SPORT_KEY,
    _db_market_for_spec,
    _market_key_for_spec,
    _match_record_to_event,
    _parse_score,
    get_pending_events,
    process_results,
)


def _make_event(
    *,
    event_id: str = "test_event_1",
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    commence_time: datetime | None = None,
    status: EventStatus = EventStatus.SCHEDULED,
) -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=commence_time or datetime(2026, 3, 1, 15, 0, 0, tzinfo=UTC),
        home_team=home_team,
        away_team=away_team,
        status=status,
    )


def _make_record(
    *,
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    match_date: str = "2026-03-01 15:00:00 UTC",
    home_score: str = "2",
    away_score: str = "1",
    with_odds: bool = True,
) -> dict:
    record: dict = {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "home_score": home_score,
        "away_score": away_score,
    }
    if with_odds:
        record["1x2_market"] = [
            {
                "bookmaker_name": "bet365",
                "odds_history_data": [
                    {
                        "opening_odds": {"timestamp": "2026-02-28T10:00:00", "odds": 2.0},
                        "odds_history": [{"timestamp": "2026-03-01T14:50:00", "odds": 1.9}],
                    },
                    {
                        "opening_odds": {"timestamp": "2026-02-28T10:00:00", "odds": 3.5},
                        "odds_history": [{"timestamp": "2026-03-01T14:50:00", "odds": 3.4}],
                    },
                    {
                        "opening_odds": {"timestamp": "2026-02-28T10:00:00", "odds": 4.0},
                        "odds_history": [{"timestamp": "2026-03-01T14:50:00", "odds": 4.2}],
                    },
                ],
            }
        ]
    return record


class TestParseScore:
    def test_valid_scores(self) -> None:
        assert _parse_score({"home_score": "2", "away_score": "1"}) == (2, 1)

    def test_zero_scores(self) -> None:
        assert _parse_score({"home_score": "0", "away_score": "0"}) == (0, 0)

    def test_missing_home_score(self) -> None:
        assert _parse_score({"away_score": "1"}) is None

    def test_missing_away_score(self) -> None:
        assert _parse_score({"home_score": "2"}) is None

    def test_non_digit_score(self) -> None:
        assert _parse_score({"home_score": "N/A", "away_score": "1"}) is None

    def test_empty_string_score(self) -> None:
        assert _parse_score({"home_score": "", "away_score": "1"}) is None


class TestMatchRecordToEvent:
    def test_matches_by_team_and_date(self) -> None:
        event = _make_event()
        record = _make_record()
        assert _match_record_to_event(record, [event]) is event

    def test_no_match_different_teams(self) -> None:
        event = _make_event(home_team="Liverpool", away_team="Everton")
        record = _make_record()
        assert _match_record_to_event(record, [event]) is None

    def test_no_match_outside_24h_window(self) -> None:
        event = _make_event(
            commence_time=datetime(2026, 3, 5, 15, 0, 0, tzinfo=UTC),
        )
        record = _make_record(match_date="2026-03-01 15:00:00 UTC")
        assert _match_record_to_event(record, [event]) is None

    def test_matches_within_24h_window(self) -> None:
        event = _make_event(
            commence_time=datetime(2026, 3, 1, 20, 0, 0, tzinfo=UTC),
        )
        record = _make_record(match_date="2026-03-01 15:00:00 UTC")
        assert _match_record_to_event(record, [event]) is event

    def test_empty_pending_events(self) -> None:
        record = _make_record()
        assert _match_record_to_event(record, []) is None

    def test_missing_team_in_record(self) -> None:
        event = _make_event()
        record = {"home_team": "", "away_team": "Chelsea", "match_date": "2026-03-01 15:00:00 UTC"}
        assert _match_record_to_event(record, [event]) is None

    def test_missing_match_date(self) -> None:
        event = _make_event()
        record = {"home_team": "Arsenal", "away_team": "Chelsea", "match_date": ""}
        assert _match_record_to_event(record, [event]) is None


class TestGetPendingEvents:
    @pytest.mark.asyncio
    async def test_includes_scheduled_events(self, test_session) -> None:
        event = _make_event(status=EventStatus.SCHEDULED)
        test_session.add(event)
        await test_session.commit()

        result = await get_pending_events(test_session)
        assert len(result) == 1
        assert result[0].id == event.id

    @pytest.mark.asyncio
    async def test_includes_live_events(self, test_session) -> None:
        event = _make_event(status=EventStatus.LIVE)
        test_session.add(event)
        await test_session.commit()

        result = await get_pending_events(test_session)
        assert len(result) == 1
        assert result[0].id == event.id

    @pytest.mark.asyncio
    async def test_excludes_final_events(self, test_session) -> None:
        event = _make_event(status=EventStatus.FINAL)
        test_session.add(event)
        await test_session.commit()

        result = await get_pending_events(test_session)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_excludes_future_events(self, test_session) -> None:
        event = _make_event(
            status=EventStatus.SCHEDULED,
            commence_time=datetime(2099, 1, 1, 12, 0, 0, tzinfo=UTC),
        )
        test_session.add(event)
        await test_session.commit()

        result = await get_pending_events(test_session)
        assert len(result) == 0


class TestProcessResults:
    @pytest.mark.asyncio
    async def test_early_exit_no_pending_events(self) -> None:
        from unittest.mock import MagicMock

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            return_value=mock_session,
        ):
            stats = await process_results(raw_matches=[])
            assert stats.matches_scraped == 0
            assert stats.events_updated == 0

    @pytest.mark.asyncio
    async def test_updates_event_and_stores_snapshot(self, test_session) -> None:
        event = _make_event()
        test_session.add(event)
        await test_session.commit()

        record = _make_record()

        mock_session_maker = AsyncMock()
        mock_session_maker.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_maker.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            return_value=mock_session_maker,
        ):
            stats = await process_results(raw_matches=[record])

        assert stats.events_updated == 1
        assert stats.snapshots_stored == 1

        await test_session.refresh(event)
        assert event.status == EventStatus.FINAL
        assert event.home_score == 2
        assert event.away_score == 1

    @pytest.mark.asyncio
    async def test_idempotency_final_events_skipped(self, test_session) -> None:
        event = _make_event(status=EventStatus.FINAL)
        event.home_score = 2
        event.away_score = 1
        test_session.add(event)
        await test_session.commit()

        record = _make_record()

        mock_session_maker = AsyncMock()
        mock_session_maker.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_maker.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            return_value=mock_session_maker,
        ):
            stats = await process_results(raw_matches=[record])

        # Already FINAL → not in pending list → not matched
        assert stats.events_updated == 0

    @pytest.mark.asyncio
    async def test_unmatched_records_counted(self, test_session) -> None:
        event = _make_event()
        test_session.add(event)
        await test_session.commit()

        unmatched_record = _make_record(home_team="Liverpool", away_team="Everton")

        mock_session_maker = AsyncMock()
        mock_session_maker.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_maker.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            return_value=mock_session_maker,
        ):
            stats = await process_results(raw_matches=[unmatched_record])

        assert stats.events_not_matched == 1
        assert stats.events_updated == 0

    @pytest.mark.asyncio
    async def test_record_without_scores_skipped(self, test_session) -> None:
        event = _make_event()
        test_session.add(event)
        await test_session.commit()

        record = _make_record(home_score="", away_score="")

        mock_session_maker = AsyncMock()
        mock_session_maker.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_maker.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            return_value=mock_session_maker,
        ):
            stats = await process_results(raw_matches=[record])

        assert stats.events_updated == 0
        await test_session.refresh(event)
        assert event.status == EventStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_snapshot_has_correct_metadata(self, test_session) -> None:
        event = _make_event()
        test_session.add(event)
        await test_session.commit()

        record = _make_record()

        mock_session_maker = AsyncMock()
        mock_session_maker.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_maker.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            return_value=mock_session_maker,
        ):
            await process_results(raw_matches=[record])

        from odds_core.models import OddsSnapshot
        from sqlalchemy import select

        result = await test_session.execute(
            select(OddsSnapshot).where(OddsSnapshot.event_id == event.id)
        )
        snapshots = list(result.scalars().all())
        assert len(snapshots) == 1

        snap = snapshots[0]
        assert snap.api_request_id == "oddsportal_closing"
        assert snap.fetch_tier == "closing"
        assert snap.bookmaker_count == 1
        assert "bookmakers" in snap.raw_data


class TestSportResolution:
    """Tests for sport-aware spec lookup in the results job."""

    def test_default_sport_key_is_epl(self) -> None:
        assert DEFAULT_SPORT_KEY == "soccer_epl"

    def test_market_key_epl(self) -> None:
        from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT

        spec = _LEAGUE_SPEC_BY_SPORT["soccer_epl"]
        assert _market_key_for_spec(spec) == "1x2_market"

    def test_market_key_mlb(self) -> None:
        from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT

        spec = _LEAGUE_SPEC_BY_SPORT["baseball_mlb"]
        assert _market_key_for_spec(spec) == "home_away_market"

    def test_db_market_epl(self) -> None:
        from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT

        spec = _LEAGUE_SPEC_BY_SPORT["soccer_epl"]
        assert _db_market_for_spec(spec) == "h2h"

    def test_db_market_mlb(self) -> None:
        from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT

        spec = _LEAGUE_SPEC_BY_SPORT["baseball_mlb"]
        assert _db_market_for_spec(spec) == "h2h"

    def test_num_outcomes_epl(self) -> None:
        from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT

        spec = _LEAGUE_SPEC_BY_SPORT["soccer_epl"]
        assert spec.num_outcomes == 3

    def test_num_outcomes_mlb(self) -> None:
        from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT

        spec = _LEAGUE_SPEC_BY_SPORT["baseball_mlb"]
        assert spec.num_outcomes == 2

    @pytest.mark.asyncio
    async def test_unknown_sport_returns_empty_stats(self) -> None:
        stats = await process_results(raw_matches=[], sport="unknown_sport")
        assert stats.matches_scraped == 0
        assert stats.events_updated == 0
