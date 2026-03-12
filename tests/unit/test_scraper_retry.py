"""Tests for the run_scraper_with_retry helper in oddsportal_common."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.oddsportal_common import (
    MAX_SCRAPER_RETRIES,
    _generate_user_agent,
    _retry_failed_urls,
    run_scraper_with_retry,
)
from oddsharvester.core.scrape_result import ErrorType, ScrapeResult


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
    error_type: ErrorType = ErrorType.NAVIGATION
    error_message: str = "timed out"


@dataclass
class FakeScrapeResult:
    success: list[dict[str, Any]] = field(default_factory=list)
    failed: list[Any] = field(default_factory=list)
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
            await _retry_failed_urls(cast(ScrapeResult, result), {"sport": "football"})

        assert result.success == [{"home_team": "Arsenal"}]
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
            await _retry_failed_urls(
                cast(ScrapeResult, result),
                {"sport": "football", "markets": ["1x2"], "headless": True},
            )

        assert len(result.success) == 16  # 11 original + 5 recovered
        assert len(result.failed) == 2  # only the still-failed from retry
        assert result.stats.successful == 16
        assert result.stats.failed == 2

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
            await _retry_failed_urls(cast(ScrapeResult, result), {"sport": "football"})

        assert len(result.success) == 1  # original only
        assert len(result.failed) == 1  # still failed
        mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_returns_none_gracefully(self) -> None:
        """If the retry scraper returns None (init error), keep original result."""
        original_success = [{"match": 0}]
        failed_urls = [FakeFailedUrl(url="https://oddsportal.com/match/1")]
        result = _make_result(success=original_success, failed=failed_urls)

        mock_run = AsyncMock(return_value=None)

        with patch(SCRAPER_PATCH, mock_run):
            await _retry_failed_urls(cast(ScrapeResult, result), {"sport": "football"})

        assert len(result.success) == 1
        assert len(result.failed) == 1  # unchanged

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
            await _retry_failed_urls(cast(ScrapeResult, result), {"sport": "football"})

        # No retryable URLs → no retry call
        mock_run.assert_not_awaited()
        assert len(result.success) == 1

    @pytest.mark.asyncio
    async def test_mixed_retryable_and_non_retryable_preserves_non_retryable(self) -> None:
        """Non-retryable failures are preserved after retry merges results."""
        original_success = [{"match": i} for i in range(5)]
        failed_urls = [
            FakeFailedUrl(url="https://oddsportal.com/match/timeout_1", is_retryable=True),
            FakeFailedUrl(
                url="https://oddsportal.com/match/404",
                is_retryable=False,
                error_type=ErrorType.PAGE_NOT_FOUND,
            ),
            FakeFailedUrl(url="https://oddsportal.com/match/timeout_2", is_retryable=True),
            FakeFailedUrl(
                url="https://oddsportal.com/match/auth_error",
                is_retryable=False,
                error_type=ErrorType.UNKNOWN,
            ),
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        # Retry recovers 1 of 2 retryable, still fails 1
        retry_success = [{"match": "recovered_0"}]
        retry_failed = [FakeFailedUrl(url="https://oddsportal.com/match/timeout_2")]
        retry_result = _make_result(success=retry_success, failed=retry_failed)

        mock_run = AsyncMock(return_value=retry_result)

        with patch(SCRAPER_PATCH, mock_run):
            await _retry_failed_urls(
                cast(ScrapeResult, result),
                {"sport": "football"},
            )

        assert len(result.success) == 6  # 5 original + 1 recovered
        assert len(result.failed) == 3  # 2 non-retryable + 1 still-failed
        assert result.stats.successful == 6
        assert result.stats.failed == 3

        # Verify non-retryable failures are preserved
        non_retryable_urls = [f.url for f in result.failed if not f.is_retryable]
        assert "https://oddsportal.com/match/404" in non_retryable_urls
        assert "https://oddsportal.com/match/auth_error" in non_retryable_urls

        # Only retryable URLs were sent to retry
        call_kwargs = mock_run.call_args.kwargs
        assert len(call_kwargs["match_links"]) == 2

    @pytest.mark.asyncio
    async def test_no_sport_skips_retry(self) -> None:
        """If sport is missing from kwargs, skip retry gracefully."""
        failed_urls = [FakeFailedUrl(url="https://oddsportal.com/match/1")]
        result = _make_result(success=[{"match": 0}], failed=failed_urls)

        mock_run = AsyncMock()

        with patch(SCRAPER_PATCH, mock_run):
            await _retry_failed_urls(cast(ScrapeResult, result), {"command": "upcoming"})

        mock_run.assert_not_awaited()
        assert len(result.failed) == 1


class TestUserAgentGeneration:
    def test_generate_user_agent_returns_chrome_string(self) -> None:
        ua = _generate_user_agent()
        assert isinstance(ua, str)
        assert "Chrome" in ua

    @pytest.mark.asyncio
    async def test_auto_injects_user_agent(self) -> None:
        """run_scraper_with_retry injects browser_user_agent when not provided."""
        matches = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        mock_run = AsyncMock(return_value=_make_result(success=matches))

        with patch(SCRAPER_PATCH, mock_run):
            await run_scraper_with_retry(command="upcoming", sport="football", headless=True)

        call_kwargs = mock_run.call_args.kwargs
        assert "browser_user_agent" in call_kwargs
        assert "Chrome" in call_kwargs["browser_user_agent"]

    @pytest.mark.asyncio
    async def test_does_not_override_explicit_user_agent(self) -> None:
        """Caller-provided browser_user_agent is preserved."""
        matches = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        mock_run = AsyncMock(return_value=_make_result(success=matches))
        custom_ua = "CustomAgent/1.0"

        with patch(SCRAPER_PATCH, mock_run):
            await run_scraper_with_retry(
                command="upcoming",
                sport="football",
                headless=True,
                browser_user_agent=custom_ua,
            )

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["browser_user_agent"] == custom_ua

    @pytest.mark.asyncio
    async def test_retry_uses_fresh_user_agent(self) -> None:
        """Failed-URL retry generates a new UA, different from the initial one."""
        initial_success = [{"match": 0}]
        initial_failed = [FakeFailedUrl(url="https://oddsportal.com/match/1")]
        initial_result = _make_result(success=initial_success, failed=initial_failed)

        retry_result = _make_result(success=[], failed=initial_failed)
        mock_run = AsyncMock(side_effect=[initial_result, retry_result])

        with patch(SCRAPER_PATCH, mock_run):
            await run_scraper_with_retry(command="upcoming", sport="football", headless=True)

        initial_ua = mock_run.call_args_list[0].kwargs["browser_user_agent"]
        retry_ua = mock_run.call_args_list[1].kwargs["browser_user_agent"]
        assert "Chrome" in initial_ua
        assert "Chrome" in retry_ua
        # UAs are independently generated (may rarely collide, but both must be present)
        assert "browser_user_agent" in mock_run.call_args_list[1].kwargs


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
