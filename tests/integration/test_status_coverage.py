"""Integration tests for `odds status coverage`: events per season, source and tier covered."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from io import StringIO
from unittest.mock import patch

import pytest
from odds_cli.commands import status
from odds_cli.commands.status import coverage_table
from odds_core.models import Event, OddsSnapshot
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.storage.readers import OddsReader, SnapshotCoverage
from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from tests.test_helpers import make_epl_event


def _snapshot(
    event: Event,
    hours_before: float,
    source: Source,
    time_basis: TimeBasis = TimeBasis.OBSERVED,
    market: Market = Market.ONE_X_TWO,
) -> OddsSnapshot:
    return OddsSnapshot(
        event_id=event.id,
        snapshot_time=event.commence_time - timedelta(hours=hours_before),
        source=source,
        market=market,
        time_basis=time_basis,
    )


def _render(coverage: SnapshotCoverage, sport_key: str = "soccer_epl") -> list[list[str]]:
    """The table's body rows, each as its cells."""
    buffer = StringIO()
    Console(file=buffer, width=200).print(coverage_table(coverage, sport_key))
    return [
        [cell.strip() for cell in line.strip("│").split("│")]
        for line in buffer.getvalue().splitlines()
        if line.startswith("│") and "Season" not in line
    ]


class TestSnapshotCoverage:
    @pytest.mark.asyncio
    async def test_counts_distinct_events_per_season_source_and_tier(
        self, pglite_async_session: AsyncSession
    ) -> None:
        autumn = make_epl_event(1, datetime(2024, 9, 14, 14, tzinfo=UTC))
        spring = make_epl_event(2, datetime(2025, 5, 3, 14, tzinfo=UTC))
        unpriced = make_epl_event(3, datetime(2025, 1, 4, 15, tzinfo=UTC))
        pglite_async_session.add_all([autumn, spring, unpriced])
        await pglite_async_session.flush()
        pglite_async_session.add_all(
            [
                _snapshot(autumn, 2.0, Source.ODDSPORTAL),
                _snapshot(autumn, 1.0, Source.ODDSPORTAL),
                _snapshot(spring, 1.5, Source.ODDSPORTAL),
                _snapshot(spring, 100.0, Source.BETFAIR),
            ]
        )
        await pglite_async_session.commit()

        coverage = await OddsReader(pglite_async_session).get_snapshot_coverage("soccer_epl")

        assert coverage.events_by_season == {2024: 3}
        assert coverage.events_by_cell == {
            (2024, Source.ODDSPORTAL, FetchTier.CLOSING): 2,
            (2024, Source.BETFAIR, FetchTier.OPENING): 1,
        }

    @pytest.mark.asyncio
    async def test_fduk_opening_and_close_take_their_time_basis_tiers(
        self, pglite_async_session: AsyncSession
    ) -> None:
        event = make_epl_event(1, datetime(2024, 9, 14, 14, tzinfo=UTC))
        pglite_async_session.add(event)
        await pglite_async_session.flush()
        pglite_async_session.add_all(
            [
                _snapshot(event, 45.0, Source.FOOTBALL_DATA_UK, TimeBasis.UPPER_BOUND),
                _snapshot(event, 0.0, Source.FOOTBALL_DATA_UK, TimeBasis.CLOSE),
            ]
        )
        await pglite_async_session.commit()

        coverage = await OddsReader(pglite_async_session).get_snapshot_coverage("soccer_epl")

        assert coverage.events_by_cell == {
            (2024, Source.FOOTBALL_DATA_UK, FetchTier.EARLY): 1,
            (2024, Source.FOOTBALL_DATA_UK, FetchTier.CLOSING): 1,
        }

    @pytest.mark.asyncio
    async def test_reads_only_the_named_sport_with_its_season_calendar(
        self, pglite_async_session: AsyncSession
    ) -> None:
        epl = make_epl_event(1, datetime(2025, 5, 3, 14, tzinfo=UTC))
        mlb = Event(
            id=2,
            sport_key="baseball_mlb",
            sport_title="MLB",
            commence_time=datetime(2025, 5, 3, 23, tzinfo=UTC),
            home_team="New York Yankees",
            away_team="Boston Red Sox",
        )
        pglite_async_session.add_all([epl, mlb])
        await pglite_async_session.flush()
        pglite_async_session.add_all(
            [
                _snapshot(epl, 1.0, Source.ODDSPORTAL),
                _snapshot(mlb, 1.0, Source.ODDS_API, market=Market.H2H),
            ]
        )
        await pglite_async_session.commit()

        coverage = await OddsReader(pglite_async_session).get_snapshot_coverage("baseball_mlb")

        assert coverage.events_by_season == {2025: 1}
        assert coverage.events_by_cell == {(2025, Source.ODDS_API, FetchTier.CLOSING): 1}


class TestCoverageTable:
    COVERAGE = SnapshotCoverage(
        events_by_season={2015: 380, 2024: 1200},
        events_by_cell={
            (2015, Source.FOOTBALL_DATA_UK, FetchTier.EARLY): 380,
            (2015, Source.FOOTBALL_DATA_UK, FetchTier.CLOSING): 380,
            (2024, Source.FOOTBALL_DATA_UK, FetchTier.EARLY): 200,
            (2024, Source.FOOTBALL_DATA_UK, FetchTier.CLOSING): 380,
            (2024, Source.ODDSPORTAL, FetchTier.OPENING): 12,
            (2024, Source.ODDSPORTAL, FetchTier.CLOSING): 1000,
        },
    )

    def test_each_season_lists_every_source_with_dashes_for_empty_cells(self) -> None:
        rows = _render(self.COVERAGE)

        # Columns: season, source, events, opening, early, day_before, pregame, closing, in_play.
        assert rows[:4] == [
            ["2015-16", "oddsportal", "380", "-", "-", "-", "-", "-", "-"],
            ["", "football_data_uk", "", "-", "380", "-", "-", "380", "-"],
            ["2024-25", "oddsportal", "1,200", "12", "-", "-", "-", "1,000", "-"],
            ["", "football_data_uk", "", "-", "200", "-", "-", "380", "-"],
        ]

    def test_total_sums_each_source_over_seasons(self) -> None:
        rows = _render(self.COVERAGE)

        assert rows[4:] == [
            ["Total", "oddsportal", "1,580", "12", "-", "-", "-", "1,000", "-"],
            ["", "football_data_uk", "", "-", "580", "-", "-", "760", "-"],
        ]

    def test_a_season_without_snapshots_is_a_row_of_dashes(self) -> None:
        rows = _render(SnapshotCoverage(events_by_season={2026: 162}, events_by_cell={}), "x")

        assert rows[0] == ["2026-27", "-", "162", "-", "-", "-", "-", "-", "-"]

    def test_a_calendar_year_sport_labels_seasons_by_year(self) -> None:
        coverage = SnapshotCoverage(
            events_by_season={2025: 1},
            events_by_cell={(2025, Source.ODDS_API, FetchTier.CLOSING): 1},
        )

        assert _render(coverage, "baseball_mlb")[0][0] == "2025"


async def _run_show_coverage(session: AsyncSession, sport_key: str) -> str:
    @asynccontextmanager
    async def maker():
        yield session

    buffer = StringIO()
    with (
        patch.object(status, "async_session_maker", maker),
        patch.object(status, "console", Console(file=buffer, width=200)),
    ):
        await status._show_coverage(sport_key)
    return buffer.getvalue()


class TestShowCoverage:
    @pytest.mark.asyncio
    async def test_prints_the_named_sports_table(self, pglite_async_session: AsyncSession) -> None:
        event = make_epl_event(1, datetime(2024, 9, 14, 14, tzinfo=UTC))
        pglite_async_session.add(event)
        await pglite_async_session.flush()
        pglite_async_session.add(_snapshot(event, 1.0, Source.BETFAIR))
        await pglite_async_session.commit()

        output = await _run_show_coverage(pglite_async_session, "soccer_epl")

        assert "Snapshot coverage: soccer_epl" in output
        assert "betfair" in output

    @pytest.mark.asyncio
    async def test_a_sport_without_events_says_so(self, pglite_async_session: AsyncSession) -> None:
        output = await _run_show_coverage(pglite_async_session, "baseball_mlb")

        assert "No baseball_mlb events" in output

    def test_sport_defaults_to_epl(self) -> None:
        sports: list[str] = []

        async def record(sport_key: str) -> None:
            sports.append(sport_key)

        with patch.object(status, "_show_coverage", record):
            result = CliRunner().invoke(status.app, ["coverage"])

        assert result.exit_code == 0
        assert sports == ["soccer_epl"]
