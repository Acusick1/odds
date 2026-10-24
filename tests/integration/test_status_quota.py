"""Integration tests for `odds status quota` against mixed-source fetch logs."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from io import StringIO
from unittest.mock import patch

import pytest
from odds_cli.commands import status
from odds_core.models import FetchLog
from odds_lambda.storage.readers import OddsReader
from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession


def _log(minutes_ago: int, quota: int | None) -> FetchLog:
    return FetchLog(
        fetch_time=datetime.now(UTC) - timedelta(minutes=minutes_ago),
        sport_key="soccer_epl",
        events_count=8,
        bookmakers_count=1,
        success=True,
        api_quota_remaining=quota,
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


class TestShowQuota:
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
