"""Unit tests for the ``get_probable_pitchers`` MCP tool and its writer/reader."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from odds_core.mlb_data_models import MlbProbablePitchers, MlbProbablePitchersRecord
from odds_lambda.storage.mlb_pitcher_reader import MlbPitcherReader
from odds_lambda.storage.mlb_pitcher_writer import MlbPitcherWriter
from sqlalchemy import select


def _record(
    *,
    game_pk: int,
    fetched_at: datetime,
    commence_time: datetime,
    home_pitcher: str | None = "Home Ace",
    away_pitcher: str | None = "Away Ace",
    home_pitcher_id: int | None = 1001,
    away_pitcher_id: int | None = 1002,
) -> MlbProbablePitchersRecord:
    return MlbProbablePitchersRecord(
        game_pk=game_pk,
        commence_time=commence_time,
        fetched_at=fetched_at,
        home_team="Boston Red Sox",
        away_team="New York Yankees",
        game_type="R",
        home_pitcher_name=home_pitcher,
        home_pitcher_id=home_pitcher_id if home_pitcher else None,
        away_pitcher_name=away_pitcher,
        away_pitcher_id=away_pitcher_id if away_pitcher else None,
    )


class TestMlbPitcherWriter:
    @pytest.mark.asyncio
    async def test_insert_persists_rows(self, pglite_async_session) -> None:
        writer = MlbPitcherWriter(pglite_async_session)
        commence = datetime(2026, 4, 26, 23, 5, tzinfo=UTC)
        fetched = datetime(2026, 4, 26, 6, 0, tzinfo=UTC)
        records = [_record(game_pk=1, fetched_at=fetched, commence_time=commence)]

        await writer.insert_snapshots(records)
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(MlbProbablePitchers))
        rows = list(result.scalars().all())
        assert len(rows) == 1
        assert rows[0].game_pk == 1
        assert rows[0].home_pitcher_name == "Home Ace"

    @pytest.mark.asyncio
    async def test_idempotent_on_same_fetched_at(self, pglite_async_session) -> None:
        writer = MlbPitcherWriter(pglite_async_session)
        commence = datetime(2026, 4, 26, 23, 5, tzinfo=UTC)
        fetched = datetime(2026, 4, 26, 6, 0, tzinfo=UTC)
        records = [_record(game_pk=1, fetched_at=fetched, commence_time=commence)]

        await writer.insert_snapshots(records)
        await pglite_async_session.commit()
        await writer.insert_snapshots(records)
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(MlbProbablePitchers))
        rows = list(result.scalars().all())
        assert len(rows) == 1, "Re-inserting the same (game_pk, fetched_at) must be a no-op"

    @pytest.mark.asyncio
    async def test_new_fetched_at_appends_new_row(self, pglite_async_session) -> None:
        writer = MlbPitcherWriter(pglite_async_session)
        commence = datetime(2026, 4, 26, 23, 5, tzinfo=UTC)
        first_fetch = datetime(2026, 4, 26, 6, 0, tzinfo=UTC)
        second_fetch = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)

        # First snapshot: pitcher not yet announced.
        await writer.insert_snapshots(
            [
                _record(
                    game_pk=1,
                    fetched_at=first_fetch,
                    commence_time=commence,
                    home_pitcher=None,
                    home_pitcher_id=None,
                )
            ]
        )
        await pglite_async_session.commit()

        # Second snapshot: pitcher now announced.
        await writer.insert_snapshots(
            [_record(game_pk=1, fetched_at=second_fetch, commence_time=commence)]
        )
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(
            select(MlbProbablePitchers).order_by(MlbProbablePitchers.fetched_at)
        )
        rows = list(result.scalars().all())
        assert len(rows) == 2
        assert rows[0].home_pitcher_name is None
        assert rows[1].home_pitcher_name == "Home Ace"


class TestMlbPitcherReader:
    @pytest.mark.asyncio
    async def test_returns_latest_per_game_pk(self, pglite_async_session) -> None:
        writer = MlbPitcherWriter(pglite_async_session)
        commence = datetime(2026, 4, 26, 23, 5, tzinfo=UTC)
        first_fetch = datetime(2026, 4, 26, 6, 0, tzinfo=UTC)
        second_fetch = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)

        await writer.insert_snapshots(
            [
                _record(
                    game_pk=1,
                    fetched_at=first_fetch,
                    commence_time=commence,
                    home_pitcher=None,
                    home_pitcher_id=None,
                )
            ]
        )
        await writer.insert_snapshots(
            [_record(game_pk=1, fetched_at=second_fetch, commence_time=commence)]
        )
        await pglite_async_session.commit()

        reader = MlbPitcherReader(pglite_async_session)
        rows = await reader.get_latest_in_window(
            commence - timedelta(hours=24),
            commence + timedelta(hours=24),
        )

        assert len(rows) == 1
        # Latest is the second_fetch row (pitcher announced).
        assert rows[0].fetched_at == second_fetch
        assert rows[0].home_pitcher_name == "Home Ace"

    @pytest.mark.asyncio
    async def test_orders_by_commence_time(self, pglite_async_session) -> None:
        writer = MlbPitcherWriter(pglite_async_session)
        fetched = datetime(2026, 4, 26, 6, 0, tzinfo=UTC)
        late = datetime(2026, 4, 27, 1, 0, tzinfo=UTC)
        early = datetime(2026, 4, 26, 23, 5, tzinfo=UTC)

        # Insert out of commence-time order.
        await writer.insert_snapshots(
            [
                _record(game_pk=2, fetched_at=fetched, commence_time=late),
                _record(game_pk=1, fetched_at=fetched, commence_time=early),
            ]
        )
        await pglite_async_session.commit()

        reader = MlbPitcherReader(pglite_async_session)
        rows = await reader.get_latest_in_window(
            datetime(2026, 4, 26, 0, 0, tzinfo=UTC),
            datetime(2026, 4, 28, 0, 0, tzinfo=UTC),
        )

        assert [r.game_pk for r in rows] == [1, 2]

    @pytest.mark.asyncio
    async def test_window_filter(self, pglite_async_session) -> None:
        writer = MlbPitcherWriter(pglite_async_session)
        fetched = datetime(2026, 4, 26, 6, 0, tzinfo=UTC)

        in_window = datetime(2026, 4, 26, 23, 5, tzinfo=UTC)
        out_of_window = datetime(2026, 5, 5, 23, 5, tzinfo=UTC)

        await writer.insert_snapshots(
            [
                _record(game_pk=1, fetched_at=fetched, commence_time=in_window),
                _record(game_pk=2, fetched_at=fetched, commence_time=out_of_window),
            ]
        )
        await pglite_async_session.commit()

        reader = MlbPitcherReader(pglite_async_session)
        rows = await reader.get_latest_in_window(
            datetime(2026, 4, 26, 0, 0, tzinfo=UTC),
            datetime(2026, 4, 28, 0, 0, tzinfo=UTC),
        )

        assert [r.game_pk for r in rows] == [1]


# ---------------------------------------------------------------------------
# get_probable_pitchers MCP tool — full write-through path with mocked fetcher.
# ---------------------------------------------------------------------------


class _FakeFetcher:
    """Minimal stand-in for MlbStatsFetcher that records calls."""

    def __init__(self, records_by_call: list[list[MlbProbablePitchersRecord]]) -> None:
        self._records_by_call = records_by_call
        self.calls: list[tuple[Any, ...]] = []

    async def __aenter__(self) -> _FakeFetcher:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def fetch_dates(
        self,
        target_dates: list[Any],
        *,
        fetched_at: datetime | None = None,
    ) -> list[MlbProbablePitchersRecord]:
        idx = len(self.calls)
        self.calls.append((target_dates, fetched_at))
        if idx >= len(self._records_by_call):
            return []
        records = self._records_by_call[idx]
        if fetched_at is not None:
            # Re-stamp to mirror the real fetcher's behaviour: every record
            # from a single ``fetch_dates`` call shares the same ``fetched_at``.
            return [
                MlbProbablePitchersRecord(
                    game_pk=r.game_pk,
                    commence_time=r.commence_time,
                    fetched_at=fetched_at,
                    home_team=r.home_team,
                    away_team=r.away_team,
                    game_type=r.game_type,
                    home_pitcher_name=r.home_pitcher_name,
                    home_pitcher_id=r.home_pitcher_id,
                    away_pitcher_name=r.away_pitcher_name,
                    away_pitcher_id=r.away_pitcher_id,
                )
                for r in records
            ]
        return records


class TestGetProbablePitchersTool:
    @pytest.mark.asyncio
    async def test_write_through_returns_persisted_rows(self, pglite_async_session) -> None:
        from odds_mcp.tools import mlb as mlb_module

        in_window = datetime(2026, 4, 26, 23, 5, tzinfo=UTC)
        out_of_window = datetime(2026, 5, 10, 23, 5, tzinfo=UTC)

        fake_fetcher = _FakeFetcher(
            [
                [
                    _record(
                        game_pk=1,
                        fetched_at=datetime(1970, 1, 1, tzinfo=UTC),  # rewritten by tool
                        commence_time=in_window,
                    ),
                    _record(
                        game_pk=2,
                        fetched_at=datetime(1970, 1, 1, tzinfo=UTC),
                        commence_time=out_of_window,
                    ),
                ]
            ]
        )

        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=pglite_async_session)
        session_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(mlb_module, "MlbStatsFetcher", return_value=fake_fetcher),
            patch.object(mlb_module, "async_session_maker", return_value=session_cm),
        ):
            result = await mlb_module.get_probable_pitchers(lookahead_hours=48)

        # Both rows persisted, but only the in-window game is returned.
        result_db = await pglite_async_session.execute(select(MlbProbablePitchers))
        all_rows = list(result_db.scalars().all())
        assert {r.game_pk for r in all_rows} == {1, 2}

        assert result["lookahead_hours"] == 48
        assert result["fetch_status"] == "live"
        assert result["game_count"] == 1
        game = result["games"][0]
        assert game["game_pk"] == 1
        assert game["home_pitcher_name"] == "Home Ace"
        assert game["away_pitcher_name"] == "Away Ace"
        assert "hours_until_commence" in game
        assert "fetched_at" in game

    @pytest.mark.asyncio
    async def test_idempotent_recall_appends_new_fetched_at_row(self, pglite_async_session) -> None:
        from odds_mcp.tools import mlb as mlb_module

        in_window = datetime.now(UTC) + timedelta(hours=12)

        # First call: pitcher not yet announced. Second call: pitcher announced.
        first_records = [
            _record(
                game_pk=42,
                fetched_at=datetime(1970, 1, 1, tzinfo=UTC),
                commence_time=in_window,
                home_pitcher=None,
                home_pitcher_id=None,
            )
        ]
        second_records = [
            _record(
                game_pk=42,
                fetched_at=datetime(1970, 1, 1, tzinfo=UTC),
                commence_time=in_window,
            )
        ]
        fake_fetcher_first = _FakeFetcher([first_records])
        fake_fetcher_second = _FakeFetcher([second_records])

        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=pglite_async_session)
        session_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(mlb_module, "MlbStatsFetcher", return_value=fake_fetcher_first),
            patch.object(mlb_module, "async_session_maker", return_value=session_cm),
        ):
            first = await mlb_module.get_probable_pitchers(lookahead_hours=48)

        with (
            patch.object(mlb_module, "MlbStatsFetcher", return_value=fake_fetcher_second),
            patch.object(mlb_module, "async_session_maker", return_value=session_cm),
        ):
            second = await mlb_module.get_probable_pitchers(lookahead_hours=48)

        # Both calls returned a row for the same game.
        assert first["game_count"] == 1
        assert second["game_count"] == 1
        assert first["games"][0]["home_pitcher_name"] is None
        assert second["games"][0]["home_pitcher_name"] == "Home Ace"

        # Two snapshot rows were persisted (different fetched_at values).
        result_db = await pglite_async_session.execute(
            select(MlbProbablePitchers).order_by(MlbProbablePitchers.fetched_at)
        )
        rows = list(result_db.scalars().all())
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_refresh_false_skips_fetch_and_reads_db(self, pglite_async_session) -> None:
        """``refresh=False`` returns cached rows and never invokes the fetcher."""
        from odds_mcp.tools import mlb as mlb_module

        in_window = datetime.now(UTC) + timedelta(hours=12)
        prior_fetch = datetime.now(UTC) - timedelta(hours=6)
        writer = MlbPitcherWriter(pglite_async_session)
        await writer.insert_snapshots(
            [_record(game_pk=77, fetched_at=prior_fetch, commence_time=in_window)]
        )
        await pglite_async_session.commit()

        fetcher_factory = MagicMock(
            side_effect=AssertionError("fetcher must not be constructed when refresh=False")
        )

        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=pglite_async_session)
        session_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(mlb_module, "MlbStatsFetcher", fetcher_factory),
            patch.object(mlb_module, "async_session_maker", return_value=session_cm),
        ):
            result = await mlb_module.get_probable_pitchers(lookahead_hours=48, refresh=False)

        assert result["fetch_status"] == "db_only"
        assert result["game_count"] == 1
        assert result["games"][0]["game_pk"] == 77
        assert result["games"][0]["fetched_at"] == prior_fetch.isoformat()
        fetcher_factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_db_when_mlbam_unreachable(self, pglite_async_session) -> None:
        """When MLBAM raises, return cached DB rows with fetch_status=stale_db_only."""
        from odds_mcp.tools import mlb as mlb_module

        # Pre-populate the DB with a prior snapshot.
        in_window = datetime.now(UTC) + timedelta(hours=12)
        prior_fetch = datetime.now(UTC) - timedelta(hours=6)
        writer = MlbPitcherWriter(pglite_async_session)
        await writer.insert_snapshots(
            [_record(game_pk=99, fetched_at=prior_fetch, commence_time=in_window)]
        )
        await pglite_async_session.commit()

        class _FailingFetcher:
            async def __aenter__(self) -> _FailingFetcher:
                return self

            async def __aexit__(self, *exc_info: Any) -> None:
                return None

            async def fetch_dates(self, *args: Any, **kwargs: Any) -> Any:
                raise httpx.ConnectError("mlbam unreachable")

        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=pglite_async_session)
        session_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(mlb_module, "MlbStatsFetcher", return_value=_FailingFetcher()),
            patch.object(mlb_module, "async_session_maker", return_value=session_cm),
        ):
            result = await mlb_module.get_probable_pitchers(lookahead_hours=48)

        assert result["fetch_status"] == "stale_db_only"
        assert result["game_count"] == 1
        assert result["games"][0]["game_pk"] == 99
        # The returned row carries the prior fetch's timestamp, not "now".
        assert result["games"][0]["fetched_at"] == prior_fetch.isoformat()

        # No new snapshot row written on the failed fetch.
        rows_db = await pglite_async_session.execute(select(MlbProbablePitchers))
        assert len(list(rows_db.scalars().all())) == 1

    @pytest.mark.asyncio
    async def test_orders_returned_games_by_commence_time(self, pglite_async_session) -> None:
        from odds_mcp.tools import mlb as mlb_module

        now = datetime.now(UTC)
        late = now + timedelta(hours=20)
        early = now + timedelta(hours=4)

        records = [
            _record(
                game_pk=2,
                fetched_at=datetime(1970, 1, 1, tzinfo=UTC),
                commence_time=late,
            ),
            _record(
                game_pk=1,
                fetched_at=datetime(1970, 1, 1, tzinfo=UTC),
                commence_time=early,
            ),
        ]
        fake_fetcher = _FakeFetcher([records])

        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=pglite_async_session)
        session_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(mlb_module, "MlbStatsFetcher", return_value=fake_fetcher),
            patch.object(mlb_module, "async_session_maker", return_value=session_cm),
        ):
            result = await mlb_module.get_probable_pitchers(lookahead_hours=48)

        assert [g["game_pk"] for g in result["games"]] == [1, 2]
