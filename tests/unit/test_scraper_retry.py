"""Tests for the run_scraper_with_retry helper in oddsportal_common."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.oddsportal_common import (
    MAX_SCRAPER_RETRIES,
    run_scraper_with_retry,
)


@dataclass
class FakeScrapeStats:
    total_urls: int = 0
    successful: int = 0
    failed: int = 0
    partial: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_urls": self.total_urls,
            "successful": self.successful,
            "failed": self.failed,
            "partial": self.partial,
        }


@dataclass
class FakeScrapeResult:
    success: list[dict[str, Any]] = field(default_factory=list)
    failed: list[Any] = field(default_factory=list)
    stats: FakeScrapeStats = field(default_factory=FakeScrapeStats)

    def get_error_breakdown(self) -> dict[str, list[str]]:
        return {}


def _make_result(
    success: list[dict[str, Any]] | None = None,
) -> FakeScrapeResult:
    return FakeScrapeResult(
        success=success or [],
        stats=FakeScrapeStats(successful=len(success) if success else 0),
    )


SCRAPER_PATCH = "oddsharvester.core.scraper_app.run_scraper"
SLEEP_PATCH = "odds_lambda.oddsportal_common.asyncio.sleep"


class TestRunScraperWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        matches = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        mock_run = AsyncMock(return_value=_make_result(success=matches))

        with patch(SCRAPER_PATCH, mock_run):
            result = await run_scraper_with_retry(
                command="upcoming", sport="football", headless=True
            )

        assert result == matches
        assert mock_run.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_empty_then_succeeds(self) -> None:
        matches = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        empty = _make_result(success=[])
        success = _make_result(success=matches)

        mock_run = AsyncMock(side_effect=[empty, success])

        with (
            patch(SCRAPER_PATCH, mock_run),
            patch(SLEEP_PATCH, new_callable=AsyncMock),
        ):
            result = await run_scraper_with_retry(
                command="upcoming", sport="football", headless=True
            )

        assert result == matches
        assert mock_run.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_empty_after_all_retries_exhausted(self) -> None:
        empty = _make_result(success=[])
        mock_run = AsyncMock(return_value=empty)

        with (
            patch(SCRAPER_PATCH, mock_run),
            patch(SLEEP_PATCH, new_callable=AsyncMock),
        ):
            result = await run_scraper_with_retry(
                command="upcoming", sport="football", headless=True
            )

        assert result == []
        assert mock_run.await_count == MAX_SCRAPER_RETRIES

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_none(self) -> None:
        mock_run = AsyncMock(return_value=None)

        with patch(SCRAPER_PATCH, mock_run):
            with pytest.raises(RuntimeError, match="fatal init error"):
                await run_scraper_with_retry(command="upcoming", sport="football", headless=True)

        assert mock_run.await_count == 1
