"""Unit tests for OddsReader.get_snapshot_coverage."""

from datetime import UTC, datetime

import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.storage.readers import OddsReader, _season_from_date


def _make_event(event_id: str, commence_time: datetime) -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=commence_time,
        home_team="Arsenal",
        away_team="Chelsea",
        status=EventStatus.FINAL,
    )


def _make_snapshot(
    event_id: str,
    hours_until_commence: float,
    api_request_id: str | None = None,
    snapshot_time: datetime | None = None,
) -> OddsSnapshot:
    if snapshot_time is None:
        snapshot_time = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    return OddsSnapshot(
        event_id=event_id,
        snapshot_time=snapshot_time,
        raw_data={},
        bookmaker_count=1,
        api_request_id=api_request_id,
        hours_until_commence=hours_until_commence,
    )


class TestSeasonFromDate:
    def test_august_is_new_season(self):
        assert _season_from_date(2024, 8) == "2024-25"

    def test_july_is_prior_season(self):
        assert _season_from_date(2025, 7) == "2024-25"

    def test_january_is_current_year_minus_one(self):
        assert _season_from_date(2024, 1) == "2023-24"

    def test_boundary_month_7_vs_8(self):
        assert _season_from_date(2023, 7) == "2022-23"
        assert _season_from_date(2023, 8) == "2023-24"


class TestGetSnapshotCoverage:
    @pytest.mark.asyncio
    async def test_returns_empty_for_no_data(self, pglite_async_session):
        reader = OddsReader(pglite_async_session)
        coverage, season_totals = await reader.get_snapshot_coverage(sport_key="soccer_epl")
        assert coverage == []
        assert season_totals == {}

    @pytest.mark.asyncio
    async def test_source_normalisation_oddsportal(self, pglite_async_session):
        event = _make_event("e1", datetime(2025, 1, 15, 15, 0, tzinfo=UTC))
        snapshot = _make_snapshot(
            "e1", hours_until_commence=10.0, api_request_id="oddsportal_live_epl"
        )
        pglite_async_session.add(event)
        pglite_async_session.add(snapshot)
        await pglite_async_session.commit()

        reader = OddsReader(pglite_async_session)
        coverage, _ = await reader.get_snapshot_coverage(sport_key="soccer_epl")

        assert len(coverage) == 1
        _, source, _, _ = coverage[0]
        assert source == "oddsportal"

    @pytest.mark.asyncio
    async def test_source_normalisation_football_data_uk(self, pglite_async_session):
        event = _make_event("e1", datetime(2025, 1, 15, 15, 0, tzinfo=UTC))
        snapshot = _make_snapshot(
            "e1", hours_until_commence=10.0, api_request_id="football_data_uk"
        )
        pglite_async_session.add(event)
        pglite_async_session.add(snapshot)
        await pglite_async_session.commit()

        reader = OddsReader(pglite_async_session)
        coverage, _ = await reader.get_snapshot_coverage(sport_key="soccer_epl")

        _, source, _, _ = coverage[0]
        assert source == "football_data_uk"

    @pytest.mark.asyncio
    async def test_source_normalisation_unknown_falls_back_to_odds_api(self, pglite_async_session):
        event = _make_event("e1", datetime(2025, 1, 15, 15, 0, tzinfo=UTC))
        snapshot = _make_snapshot(
            "e1", hours_until_commence=10.0, api_request_id="some_other_source"
        )
        pglite_async_session.add(event)
        pglite_async_session.add(snapshot)
        await pglite_async_session.commit()

        reader = OddsReader(pglite_async_session)
        coverage, _ = await reader.get_snapshot_coverage(sport_key="soccer_epl")

        _, source, _, _ = coverage[0]
        assert source == "odds_api"

    @pytest.mark.asyncio
    async def test_source_normalisation_none_falls_back_to_odds_api(self, pglite_async_session):
        event = _make_event("e1", datetime(2025, 1, 15, 15, 0, tzinfo=UTC))
        snapshot = _make_snapshot("e1", hours_until_commence=10.0, api_request_id=None)
        pglite_async_session.add(event)
        pglite_async_session.add(snapshot)
        await pglite_async_session.commit()

        reader = OddsReader(pglite_async_session)
        coverage, _ = await reader.get_snapshot_coverage(sport_key="soccer_epl")

        _, source, _, _ = coverage[0]
        assert source == "odds_api"

    @pytest.mark.asyncio
    async def test_tier_bucket_boundaries(self, pglite_async_session):
        event = _make_event("e1", datetime(2025, 1, 15, 15, 0, tzinfo=UTC))
        pglite_async_session.add(event)

        snapshots = [
            _make_snapshot("e1", 72.0, snapshot_time=datetime(2025, 1, 10, 12, 0, tzinfo=UTC)),
            _make_snapshot("e1", 24.0, snapshot_time=datetime(2025, 1, 13, 12, 0, tzinfo=UTC)),
            _make_snapshot("e1", 12.0, snapshot_time=datetime(2025, 1, 14, 12, 0, tzinfo=UTC)),
            _make_snapshot("e1", 3.0, snapshot_time=datetime(2025, 1, 15, 8, 0, tzinfo=UTC)),
            _make_snapshot("e1", 0.5, snapshot_time=datetime(2025, 1, 15, 14, 0, tzinfo=UTC)),
        ]
        for s in snapshots:
            pglite_async_session.add(s)
        await pglite_async_session.commit()

        reader = OddsReader(pglite_async_session)
        coverage, _ = await reader.get_snapshot_coverage(sport_key="soccer_epl")

        buckets = {bucket for _, _, bucket, _ in coverage}
        assert buckets == {"72h+", "24-72h", "12-24h", "3-12h", "<3h"}

    @pytest.mark.asyncio
    async def test_events_column_counts_distinct_events_not_snapshots(self, pglite_async_session):
        event = _make_event("e1", datetime(2025, 1, 15, 15, 0, tzinfo=UTC))
        pglite_async_session.add(event)

        # Two snapshots for the same event in the same bucket
        for i in range(3):
            pglite_async_session.add(
                _make_snapshot(
                    "e1",
                    hours_until_commence=10.0,
                    snapshot_time=datetime(2025, 1, 15, i, 0, tzinfo=UTC),
                )
            )
        await pglite_async_session.commit()

        reader = OddsReader(pglite_async_session)
        coverage, season_totals = await reader.get_snapshot_coverage(sport_key="soccer_epl")

        # Only 1 distinct event, regardless of snapshot count
        assert len(coverage) == 1
        _, _, _, event_count = coverage[0]
        assert event_count == 1
        assert season_totals["2024-25"] == 1

    @pytest.mark.asyncio
    async def test_season_totals_count_distinct_events_across_sources(self, pglite_async_session):
        event = _make_event("e1", datetime(2025, 1, 15, 15, 0, tzinfo=UTC))
        pglite_async_session.add(event)

        # Same event covered by two different sources
        pglite_async_session.add(
            _make_snapshot("e1", hours_until_commence=10.0, api_request_id="oddsportal_live_epl")
        )
        pglite_async_session.add(
            _make_snapshot(
                "e1",
                hours_until_commence=10.0,
                api_request_id="football_data_uk",
                snapshot_time=datetime(2025, 1, 15, 13, 0, tzinfo=UTC),
            )
        )
        await pglite_async_session.commit()

        reader = OddsReader(pglite_async_session)
        coverage, season_totals = await reader.get_snapshot_coverage(sport_key="soccer_epl")

        # Two source×bucket cells but only 1 distinct event for the season
        assert len(coverage) == 2
        assert season_totals["2024-25"] == 1

    @pytest.mark.asyncio
    async def test_season_boundary_august_starts_new_season(self, pglite_async_session):
        jul_event = _make_event("e_jul", datetime(2024, 7, 31, 15, 0, tzinfo=UTC))
        aug_event = _make_event("e_aug", datetime(2024, 8, 1, 15, 0, tzinfo=UTC))
        pglite_async_session.add(jul_event)
        pglite_async_session.add(aug_event)
        pglite_async_session.add(_make_snapshot("e_jul", hours_until_commence=10.0))
        pglite_async_session.add(
            _make_snapshot(
                "e_aug",
                hours_until_commence=10.0,
                snapshot_time=datetime(2024, 8, 1, 12, 0, tzinfo=UTC),
            )
        )
        await pglite_async_session.commit()

        reader = OddsReader(pglite_async_session)
        _, season_totals = await reader.get_snapshot_coverage(sport_key="soccer_epl")

        assert "2023-24" in season_totals  # July game
        assert "2024-25" in season_totals  # August game
        assert season_totals["2023-24"] == 1
        assert season_totals["2024-25"] == 1

    @pytest.mark.asyncio
    async def test_ignores_events_without_hours_until_commence(self, pglite_async_session):
        event = _make_event("e1", datetime(2025, 1, 15, 15, 0, tzinfo=UTC))
        pglite_async_session.add(event)
        # Snapshot with no hours_until_commence
        pglite_async_session.add(
            OddsSnapshot(
                event_id="e1",
                snapshot_time=datetime(2025, 1, 15, 12, 0, tzinfo=UTC),
                raw_data={},
                bookmaker_count=1,
                api_request_id=None,
                hours_until_commence=None,
            )
        )
        await pglite_async_session.commit()

        reader = OddsReader(pglite_async_session)
        coverage, season_totals = await reader.get_snapshot_coverage(sport_key="soccer_epl")

        assert coverage == []
        assert season_totals == {}
