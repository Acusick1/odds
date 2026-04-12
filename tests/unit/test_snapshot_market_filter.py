"""Tests for get_latest_snapshot market filtering."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.storage.readers import OddsReader


def _make_event(event_id: str = "test_event_1") -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=datetime(2026, 4, 13, 15, 0, tzinfo=UTC),
        home_team="Man Utd",
        away_team="Leeds",
        status=EventStatus.SCHEDULED,
    )


def _make_snapshot(
    event_id: str,
    snapshot_time: datetime,
    market_key: str,
) -> OddsSnapshot:
    return OddsSnapshot(
        event_id=event_id,
        snapshot_time=snapshot_time,
        raw_data={
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": market_key,
                            "outcomes": [
                                {"name": "Man Utd", "price": 1.80},
                                {"name": "Draw", "price": 3.50},
                                {"name": "Leeds", "price": 4.20},
                            ],
                        }
                    ],
                }
            ],
            "source": "oddsportal_live",
        },
        bookmaker_count=1,
    )


class TestGetLatestSnapshotMarketFilter:
    """get_latest_snapshot with market parameter filters by raw_data market key."""

    @pytest.mark.asyncio
    async def test_returns_h2h_when_newer_totals_exists(self, test_session):
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        t_base = datetime(2026, 4, 13, 10, 0, tzinfo=UTC)
        h2h_snap = _make_snapshot(event.id, t_base, "h2h")
        totals_snap = _make_snapshot(event.id, t_base + timedelta(minutes=30), "totals")

        test_session.add_all([h2h_snap, totals_snap])
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_snapshot(event.id, market="h2h")

        assert result is not None
        assert result.id == h2h_snap.id

    @pytest.mark.asyncio
    async def test_without_market_returns_newest(self, test_session):
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        t_base = datetime(2026, 4, 13, 10, 0, tzinfo=UTC)
        h2h_snap = _make_snapshot(event.id, t_base, "h2h")
        totals_snap = _make_snapshot(event.id, t_base + timedelta(minutes=30), "totals")

        test_session.add_all([h2h_snap, totals_snap])
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_snapshot(event.id)

        assert result is not None
        assert result.id == totals_snap.id

    @pytest.mark.asyncio
    async def test_returns_none_when_market_not_present(self, test_session):
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        t_base = datetime(2026, 4, 13, 10, 0, tzinfo=UTC)
        totals_snap = _make_snapshot(event.id, t_base, "totals")

        test_session.add(totals_snap)
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_snapshot(event.id, market="h2h")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_most_recent_matching_market(self, test_session):
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        t_base = datetime(2026, 4, 13, 10, 0, tzinfo=UTC)
        old_h2h = _make_snapshot(event.id, t_base, "h2h")
        new_h2h = _make_snapshot(event.id, t_base + timedelta(hours=1), "h2h")
        totals_snap = _make_snapshot(event.id, t_base + timedelta(hours=2), "totals")

        test_session.add_all([old_h2h, new_h2h, totals_snap])
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_snapshot(event.id, market="h2h")

        assert result is not None
        assert result.id == new_h2h.id
