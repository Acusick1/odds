"""Tests for the run_scraper_with_retry helper in oddsportal_common."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.oddsportal_common import (
    MAX_SCRAPER_RETRIES,
    _retry_failed_urls,
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
class FakeFailedUrl:
    url: str
    is_retryable: bool = True
    error_type: str = "timeout"
    error_message: str = "timed out"


@dataclass
class FakeScrapeResult:
    success: list[dict[str, Any]] = field(default_factory=list)
    failed: list[Any] = field(default_factory=list)
    partial: list[Any] = field(default_factory=list)
    stats: FakeScrapeStats = field(default_factory=FakeScrapeStats)

    def get_error_breakdown(self) -> dict[str, list[str]]:
        return {}

    def get_retryable_urls(self) -> list[str]:
        return [f.url for f in self.failed if f.is_retryable]


def _make_result(
    success: list[dict[str, Any]] | None = None,
    failed: list[FakeFailedUrl] | None = None,
) -> FakeScrapeResult:
    s = success or []
    f = failed or []
    return FakeScrapeResult(
        success=s,
        failed=f,
        stats=FakeScrapeStats(
            successful=len(s),
            failed=len(f),
            total_urls=len(s) + len(f),
        ),
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

        assert result.success == matches
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

        assert result.success == matches
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

        assert result.success == []
        assert mock_run.await_count == MAX_SCRAPER_RETRIES

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_none(self) -> None:
        mock_run = AsyncMock(return_value=None)

        with patch(SCRAPER_PATCH, mock_run):
            with pytest.raises(RuntimeError, match="fatal init error"):
                await run_scraper_with_retry(command="upcoming", sport="football", headless=True)

        assert mock_run.await_count == 1


class TestFailedUrlRetry:
    @pytest.mark.asyncio
    async def test_no_retry_when_no_failures(self) -> None:
        """All matches succeed — no retry call made."""
        result = _make_result(success=[{"home_team": "Arsenal"}])
        mock_run = AsyncMock()

        with patch(SCRAPER_PATCH, mock_run):
            out = await _retry_failed_urls(result, {"sport": "football"})

        assert out.success == [{"home_team": "Arsenal"}]
        mock_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_failure_triggers_retry_and_recovers(self) -> None:
        """7 of 18 fail, retry recovers 5 of those 7."""
        original_success = [{"match": i} for i in range(11)]
        failed_urls = [FakeFailedUrl(url=f"https://oddsportal.com/match/{i}") for i in range(7)]
        result = _make_result(success=original_success, failed=failed_urls)

        # Retry recovers 5, still fails 2
        retry_success = [{"match": f"recovered_{i}"} for i in range(5)]
        retry_failed = [FakeFailedUrl(url=f"https://oddsportal.com/match/{i}") for i in range(2)]
        retry_result = _make_result(success=retry_success, failed=retry_failed)

        mock_run = AsyncMock(return_value=retry_result)

        with patch(SCRAPER_PATCH, mock_run):
            out = await _retry_failed_urls(
                result,
                {"sport": "football", "markets": ["1x2"], "headless": True},
            )

        assert len(out.success) == 16  # 11 original + 5 recovered
        assert len(out.failed) == 2  # only the still-failed from retry
        assert out.stats.successful == 16
        assert out.stats.failed == 2

        # Verify retry was called with match_links
        call_kwargs = mock_run.call_args.kwargs
        assert len(call_kwargs["match_links"]) == 7
        assert call_kwargs["sport"] == "football"
        assert call_kwargs["markets"] == ["1x2"]
        assert "command" not in call_kwargs
        assert "leagues" not in call_kwargs

    @pytest.mark.asyncio
    async def test_retry_still_fails_all(self) -> None:
        """Retry runs but recovers nothing."""
        original_success = [{"match": 0}]
        failed_urls = [FakeFailedUrl(url="https://oddsportal.com/match/1")]
        result = _make_result(success=original_success, failed=failed_urls)

        retry_result = _make_result(
            success=[],
            failed=[FakeFailedUrl(url="https://oddsportal.com/match/1")],
        )
        mock_run = AsyncMock(return_value=retry_result)

        with patch(SCRAPER_PATCH, mock_run):
            out = await _retry_failed_urls(result, {"sport": "football"})

        assert len(out.success) == 1  # original only
        assert len(out.failed) == 1  # still failed
        mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_returns_none_gracefully(self) -> None:
        """If the retry scraper returns None (init error), keep original result."""
        original_success = [{"match": 0}]
        failed_urls = [FakeFailedUrl(url="https://oddsportal.com/match/1")]
        result = _make_result(success=original_success, failed=failed_urls)

        mock_run = AsyncMock(return_value=None)

        with patch(SCRAPER_PATCH, mock_run):
            out = await _retry_failed_urls(result, {"sport": "football"})

        assert len(out.success) == 1
        assert len(out.failed) == 1  # unchanged

    @pytest.mark.asyncio
    async def test_non_retryable_urls_skipped(self) -> None:
        """Only retryable URLs are retried."""
        original_success = [{"match": 0}]
        failed_urls = [
            FakeFailedUrl(url="https://oddsportal.com/match/1", is_retryable=False),
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        mock_run = AsyncMock()

        with patch(SCRAPER_PATCH, mock_run):
            out = await _retry_failed_urls(result, {"sport": "football"})

        # No retryable URLs → no retry call
        mock_run.assert_not_awaited()
        assert len(out.success) == 1

    @pytest.mark.asyncio
    async def test_no_sport_skips_retry(self) -> None:
        """If sport is missing from kwargs, skip retry gracefully."""
        failed_urls = [FakeFailedUrl(url="https://oddsportal.com/match/1")]
        result = _make_result(success=[{"match": 0}], failed=failed_urls)

        mock_run = AsyncMock()

        with patch(SCRAPER_PATCH, mock_run):
            out = await _retry_failed_urls(result, {"command": "upcoming"})

        mock_run.assert_not_awaited()
        assert len(out.failed) == 1


class TestEndToEndWithFailedUrlRetry:
    @pytest.mark.asyncio
    async def test_partial_scrape_triggers_failed_url_retry(self) -> None:
        """Full flow: initial scrape partially fails, retry recovers some."""
        initial_success = [{"match": i} for i in range(11)]
        initial_failed = [FakeFailedUrl(url=f"https://oddsportal.com/match/{i}") for i in range(7)]
        initial_result = _make_result(success=initial_success, failed=initial_failed)

        retry_success = [{"match": "recovered_0"}]
        retry_failed = [FakeFailedUrl(url=f"https://oddsportal.com/match/{i}") for i in range(6)]
        retry_result = _make_result(success=retry_success, failed=retry_failed)

        mock_run = AsyncMock(side_effect=[initial_result, retry_result])

        with patch(SCRAPER_PATCH, mock_run):
            result = await run_scraper_with_retry(
                command="upcoming", sport="football", markets=["1x2"], headless=True
            )

        assert len(result.success) == 12  # 11 + 1 recovered
        assert len(result.failed) == 6  # still failed from retry
        assert mock_run.await_count == 2
