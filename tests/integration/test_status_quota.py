"""Integration tests for `odds status quota` against mixed-source fetch logs."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from io import StringIO
from unittest.mock import patch

import pytest
from odds_cli.commands import status
from odds_core.config import APIConfig
from odds_core.models import FetchLog
from odds_lambda.storage.readers import KeyQuota, OddsReader
from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession


def _log(
    minutes_ago: int,
    quota: int | None,
    sport_key: str = "soccer_epl",
    plan: int | None = 500,
) -> FetchLog:
    """A fetch log whose key has a ``plan``-credit plan; ``plan=None`` is a pre-migration row."""
    return FetchLog(
        fetch_time=datetime.now(UTC) - timedelta(minutes=minutes_ago),
        sport_key=sport_key,
        events_count=8,
        bookmakers_count=1,
        success=True,
        api_quota_remaining=quota,
        api_quota_used=None if quota is None or plan is None else plan - quota,
    )


async def _run_show_quota(session: AsyncSession) -> str:
    @asynccontextmanager
    async def maker():
        yield session

    buffer = StringIO()
    with (
        patch.object(status, "async_session_maker", maker),
        patch.object(status, "console", Console(file=buffer, width=120)),
    ):
        await status._show_quota()
    return buffer.getvalue()


class TestFetchLogsWithQuota:
    @pytest.mark.asyncio
    async def test_skips_newer_rows_without_quota(self, pglite_async_session):
        pglite_async_session.add_all([_log(60, 300), _log(30, 290), _log(1, None)])
        await pglite_async_session.commit()

        logs = await OddsReader(pglite_async_session).get_fetch_logs(limit=10, with_quota=True)

        assert [log.api_quota_remaining for log in logs] == [290, 300]


class TestLatestQuotaByKey:
    API = APIConfig(keys="a,b", key_index_by_sport={"basketball_nba": 1})

    @pytest.mark.asyncio
    async def test_each_key_reports_the_newest_quota_of_its_sports(self, pglite_async_session):
        pglite_async_session.add_all(
            [
                _log(60, 80),
                _log(30, 70),
                _log(1, None),
                _log(45, 500, "basketball_nba"),
                _log(15, 497, "basketball_nba"),
            ]
        )
        await pglite_async_session.commit()

        quotas = await OddsReader(pglite_async_session).get_latest_quota_by_key(self.API)

        assert quotas == {
            0: KeyQuota(("soccer_epl",), 70, 430),
            1: KeyQuota(("basketball_nba",), 497, 3),
        }

    @pytest.mark.asyncio
    async def test_sports_sharing_a_key_give_one_reading_from_the_newest_row(
        self, pglite_async_session
    ):
        pglite_async_session.add_all([_log(30, 70), _log(10, 60, "baseball_mlb")])
        await pglite_async_session.commit()

        quotas = await OddsReader(pglite_async_session).get_latest_quota_by_key(self.API)

        assert quotas == {0: KeyQuota(("baseball_mlb", "soccer_epl"), 60, 440)}

    @pytest.mark.asyncio
    async def test_a_reading_over_a_month_old_is_ignored(self, pglite_async_session):
        pglite_async_session.add_all([_log(60 * 24 * 32, 19900, "basketball_nba"), _log(5, 70)])
        await pglite_async_session.commit()

        quotas = await OddsReader(pglite_async_session).get_latest_quota_by_key(self.API)

        assert quotas == {0: KeyQuota(("soccer_epl",), 70, 430)}

    @pytest.mark.asyncio
    async def test_the_plan_is_used_plus_remaining_from_the_newest_reading(
        self, pglite_async_session
    ):
        pglite_async_session.add_all([_log(30, 18000, plan=20000), _log(5, 17900, plan=20000)])
        await pglite_async_session.commit()

        quotas = await OddsReader(pglite_async_session).get_latest_quota_by_key(self.API)

        quota = quotas[0]
        assert (quota.remaining, quota.used, quota.limit) == (17900, 2100, 20000)
        assert quota.fraction == pytest.approx(0.895)

    @pytest.mark.asyncio
    async def test_rows_without_a_used_count_are_skipped(self, pglite_async_session):
        pglite_async_session.add_all(
            [_log(30, 70), _log(5, 60, plan=None), _log(5, 400, "basketball_nba", plan=None)]
        )
        await pglite_async_session.commit()

        quotas = await OddsReader(pglite_async_session).get_latest_quota_by_key(self.API)

        assert quotas == {0: KeyQuota(("soccer_epl",), 70, 430)}

    def test_a_zero_plan_has_a_zero_fraction(self):
        assert KeyQuota(("soccer_epl",), 0, 0).fraction == 0.0


class TestShowQuota:
    @pytest.mark.asyncio
    async def test_prints_each_key_separately(self, pglite_async_session):
        pglite_async_session.add_all([_log(30, 70), _log(15, 497, "basketball_nba")])
        await pglite_async_session.commit()

        output = await _run_show_quota(pglite_async_session)

        assert output.index("key 0: soccer_epl") < output.index("key 1: basketball_nba")
        assert "497 (99.4%)" in output
        assert "70 (14.0%)" in output

    @pytest.mark.asyncio
    async def test_the_total_is_the_keys_plan_not_a_setting(self, pglite_async_session):
        pglite_async_session.add(_log(5, 17900, plan=20000))
        await pglite_async_session.commit()

        output = await _run_show_quota(pglite_async_session)

        assert "20,000" in output
        assert "17,900 (89.5%)" in output

    @pytest.mark.asyncio
    async def test_reads_newest_row_with_quota(self, pglite_async_session):
        pglite_async_session.add_all([_log(60, 300), _log(30, 290), _log(1, None)])
        await pglite_async_session.commit()

        output = await _run_show_quota(pglite_async_session)

        assert "Remaining" in output
        assert "290" in output

    @pytest.mark.asyncio
    async def test_explains_when_no_row_has_quota(self, pglite_async_session):
        pglite_async_session.add(_log(1, None))
        await pglite_async_session.commit()

        output = await _run_show_quota(pglite_async_session)

        assert "No fetch log records an API quota" in output

    @pytest.mark.asyncio
    async def test_used_column_is_per_fetch_consumption(self, pglite_async_session):
        pglite_async_session.add_all(
            [_log(40, 500), _log(30, 480), _log(20, 470), _log(10, 490), _log(1, None)]
        )
        await pglite_async_session.commit()

        output = await _run_show_quota(pglite_async_session)

        rows = [line.split("│")[1:-1] for line in output.splitlines() if "m ago" in line]
        assert [[cell.strip() for cell in row[1:]] for row in rows] == [
            ["490", "reset"],
            ["470", "10"],
            ["480", "20"],
            ["500", ""],
        ]
