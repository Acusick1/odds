"""Tests for the run_scraper_with_retry helper in oddsportal_common."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from odds_lambda.oddsportal_common import (
    MAX_FAILED_URL_RETRY_PASSES,
    MAX_SCRAPER_RETRIES,
    _generate_user_agent,
    _get_retriable_failed_urls,
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
    error_message: str | None = "timed out"


@dataclass
class FakeScrapeResult:
    success: list[dict[str, Any]] = field(default_factory=list)
    failed: list[Any] = field(default_factory=list)
    stats: FakeScrapeStats = field(default_factory=FakeScrapeStats)

    def get_error_breakdown(self) -> dict[str, list[str]]:
        return {}


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
USERAGENT_PATCH = "fake_useragent.UserAgent"


def _mock_useragent() -> MagicMock:
    """Create a mock UserAgent that returns a realistic Chrome UA string."""
    mock_cls = MagicMock()
    type(mock_cls.return_value).random = PropertyMock(
        return_value=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    )
    return mock_cls


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
    async def test_empty_result_returned_with_single_attempt(self) -> None:
        """With MAX_SCRAPER_RETRIES=1, empty result is returned immediately (no retry)."""
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
        assert mock_run.await_count == 1

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


class TestGetRetriableFailedUrls:
    def test_excludes_page_not_found(self) -> None:
        failed = [
            FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION),
            FakeFailedUrl(
                url="https://oddsportal.com/match/2", error_type=ErrorType.PAGE_NOT_FOUND
            ),
            FakeFailedUrl(
                url="https://oddsportal.com/match/3", error_type=ErrorType.HEADER_NOT_FOUND
            ),
        ]
        urls = _get_retriable_failed_urls(failed)
        assert urls == [
            "https://oddsportal.com/match/1",
            "https://oddsportal.com/match/3",
        ]

    def test_retries_all_non_404_error_types(self) -> None:
        """Every ErrorType except PAGE_NOT_FOUND should be retried."""
        non_404_types = [
            ErrorType.NAVIGATION,
            ErrorType.PARSING,
            ErrorType.MARKET_EXTRACTION,
            ErrorType.HEADER_NOT_FOUND,
            ErrorType.RATE_LIMITED,
            ErrorType.UNKNOWN,
        ]
        failed = [
            FakeFailedUrl(url=f"https://oddsportal.com/match/{i}", error_type=et)
            for i, et in enumerate(non_404_types)
        ]
        urls = _get_retriable_failed_urls(failed)
        assert len(urls) == len(non_404_types)

    def test_retries_urls_with_none_error_message(self) -> None:
        """The upstream bug: error_message=None, is_retryable=False. Should still retry."""
        failed = [
            FakeFailedUrl(
                url="https://oddsportal.com/match/1",
                is_retryable=False,
                error_type=ErrorType.UNKNOWN,
                error_message=None,
            ),
        ]
        urls = _get_retriable_failed_urls(failed)
        assert urls == ["https://oddsportal.com/match/1"]

    def test_retries_header_not_found_even_when_not_retryable(self) -> None:
        """HEADER_NOT_FOUND with is_retryable=False (upstream bug) should still retry."""
        failed = [
            FakeFailedUrl(
                url="https://oddsportal.com/match/1",
                is_retryable=False,
                error_type=ErrorType.HEADER_NOT_FOUND,
                error_message=None,
            ),
        ]
        urls = _get_retriable_failed_urls(failed)
        assert urls == ["https://oddsportal.com/match/1"]

    def test_empty_list(self) -> None:
        assert _get_retriable_failed_urls([]) == []


class TestFailedUrlRetry:
    @pytest.mark.asyncio
    async def test_no_retry_when_no_failures(self) -> None:
        result = _make_result(success=[{"home_team": "Arsenal"}])
        mock_run = AsyncMock()

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(
                cast(ScrapeResult, result), {"command": "upcoming", "sport": "football"}
            )

        assert result.success == [{"home_team": "Arsenal"}]
        mock_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_failure_triggers_retry_and_recovers(self) -> None:
        """7 of 18 fail, single retry pass recovers 5 of those 7."""
        original_success = [{"match": i} for i in range(11)]
        failed_urls = [
            FakeFailedUrl(
                url=f"https://oddsportal.com/match/{i}", error_type=ErrorType.HEADER_NOT_FOUND
            )
            for i in range(7)
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        # Single retry pass recovers 5, still fails 2
        retry1_success = [{"match": f"recovered_{i}"} for i in range(5)]
        retry1_failed = [
            FakeFailedUrl(
                url=f"https://oddsportal.com/match/{i}", error_type=ErrorType.HEADER_NOT_FOUND
            )
            for i in range(2)
        ]
        retry1_result = _make_result(success=retry1_success, failed=retry1_failed)

        mock_run = AsyncMock(return_value=retry1_result)

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(
                cast(ScrapeResult, result),
                {"command": "upcoming", "sport": "football", "markets": ["1x2"], "headless": True},
            )

        assert len(result.success) == 16  # 11 original + 5 recovered
        assert len(result.failed) == 2  # still failed after single pass
        assert result.stats.successful == 16
        assert result.stats.failed == 2
        assert mock_run.await_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausts_all_passes(self) -> None:
        """Retry runs 3 passes but never fully recovers."""
        original_success = [{"match": 0}]
        failed_urls = [
            FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION)
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        still_failed = _make_result(
            success=[],
            failed=[
                FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION)
            ],
        )
        mock_run = AsyncMock(return_value=still_failed)

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(
                cast(ScrapeResult, result), {"command": "upcoming", "sport": "football"}
            )

        assert len(result.success) == 1  # original only
        assert len(result.failed) == 1  # still failed
        assert mock_run.await_count == MAX_FAILED_URL_RETRY_PASSES

    @pytest.mark.asyncio
    async def test_retry_returns_none_gracefully(self) -> None:
        """If the retry scraper returns None (init error), stop retrying."""
        original_success = [{"match": 0}]
        failed_urls = [
            FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION)
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        mock_run = AsyncMock(return_value=None)

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(
                cast(ScrapeResult, result), {"command": "upcoming", "sport": "football"}
            )

        assert len(result.success) == 1
        assert len(result.failed) == 1  # unchanged
        mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_page_not_found_excluded_from_retry(self) -> None:
        """PAGE_NOT_FOUND (404) failures are never retried."""
        original_success = [{"match": 0}]
        failed_urls = [
            FakeFailedUrl(
                url="https://oddsportal.com/match/404",
                is_retryable=False,
                error_type=ErrorType.PAGE_NOT_FOUND,
            ),
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        mock_run = AsyncMock()

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(
                cast(ScrapeResult, result), {"command": "upcoming", "sport": "football"}
            )

        # No retriable URLs → no retry call
        mock_run.assert_not_awaited()
        assert len(result.success) == 1
        assert len(result.failed) == 1

    @pytest.mark.asyncio
    async def test_non_retryable_flag_ignored_for_non_404(self) -> None:
        """URLs with is_retryable=False but non-404 error types ARE retried."""
        original_success = [{"match": 0}]
        failed_urls = [
            FakeFailedUrl(
                url="https://oddsportal.com/match/1",
                is_retryable=False,
                error_type=ErrorType.UNKNOWN,
                error_message=None,
            ),
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        retry_result = _make_result(
            success=[{"match": "recovered"}],
            failed=[],
        )
        mock_run = AsyncMock(return_value=retry_result)

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(
                cast(ScrapeResult, result), {"command": "upcoming", "sport": "football"}
            )

        assert len(result.success) == 2  # original + recovered
        assert len(result.failed) == 0
        mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mixed_404_and_retriable_preserves_404(self) -> None:
        """404s are preserved in failed list; other error types are retried."""
        original_success = [{"match": i} for i in range(5)]
        failed_urls = [
            FakeFailedUrl(
                url="https://oddsportal.com/match/timeout_1", error_type=ErrorType.NAVIGATION
            ),
            FakeFailedUrl(
                url="https://oddsportal.com/match/404", error_type=ErrorType.PAGE_NOT_FOUND
            ),
            FakeFailedUrl(
                url="https://oddsportal.com/match/header", error_type=ErrorType.HEADER_NOT_FOUND
            ),
            FakeFailedUrl(
                url="https://oddsportal.com/match/none_error",
                is_retryable=False,
                error_type=ErrorType.UNKNOWN,
                error_message=None,
            ),
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        # Single pass recovers 2 of 3 retriable, still fails 1
        retry_success = [{"match": "recovered_0"}, {"match": "recovered_1"}]
        retry_failed = [
            FakeFailedUrl(
                url="https://oddsportal.com/match/timeout_1", error_type=ErrorType.NAVIGATION
            )
        ]
        retry_result = _make_result(success=retry_success, failed=retry_failed)

        mock_run = AsyncMock(return_value=retry_result)

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(
                cast(ScrapeResult, result),
                {"command": "upcoming", "sport": "football"},
            )

        assert len(result.success) == 7  # 5 original + 2 recovered
        assert len(result.failed) == 2  # 404 + 1 still-failing retriable
        assert any(f.error_type == ErrorType.PAGE_NOT_FOUND for f in result.failed)
        assert result.stats.successful == 7
        assert result.stats.failed == 2

        # Only non-404 URLs were sent to retry
        first_call_kwargs = mock_run.call_args_list[0].kwargs
        assert len(first_call_kwargs["match_links"]) == 3

    @pytest.mark.asyncio
    async def test_command_forwarded_to_retry_calls(self) -> None:
        """The command kwarg must be passed through to retry run_scraper calls."""
        original_success = [{"match": 0}]
        failed_urls = [
            FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION)
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        retry_result = _make_result(success=[{"match": "recovered"}], failed=[])
        mock_run = AsyncMock(return_value=retry_result)

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(
                cast(ScrapeResult, result),
                {"command": "historic", "sport": "football", "headless": True},
            )

        retry_call_kwargs = mock_run.call_args.kwargs
        assert retry_call_kwargs["command"] == "historic"
        assert retry_call_kwargs["sport"] == "football"

    @pytest.mark.asyncio
    async def test_missing_command_raises_key_error(self) -> None:
        """Missing command in original_kwargs surfaces a clear KeyError."""
        failed_urls = [
            FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION)
        ]
        result = _make_result(success=[{"match": 0}], failed=failed_urls)

        mock_run = AsyncMock()

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            with pytest.raises(KeyError, match="command"):
                await _retry_failed_urls(
                    cast(ScrapeResult, result),
                    {"sport": "football"},
                )

    @pytest.mark.asyncio
    async def test_no_sport_skips_retry(self) -> None:
        failed_urls = [
            FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION)
        ]
        result = _make_result(success=[{"match": 0}], failed=failed_urls)

        mock_run = AsyncMock()

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(cast(ScrapeResult, result), {"command": "upcoming"})

        mock_run.assert_not_awaited()
        assert len(result.failed) == 1

    @pytest.mark.asyncio
    async def test_early_break_when_all_recovered(self) -> None:
        """Stops retrying once all failures are recovered."""
        original_success = [{"match": 0}]
        failed_urls = [
            FakeFailedUrl(
                url="https://oddsportal.com/match/1", error_type=ErrorType.HEADER_NOT_FOUND
            ),
            FakeFailedUrl(url="https://oddsportal.com/match/2", error_type=ErrorType.UNKNOWN),
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        retry_result = _make_result(
            success=[{"match": "r1"}, {"match": "r2"}],
            failed=[],
        )
        mock_run = AsyncMock(return_value=retry_result)

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            await _retry_failed_urls(
                cast(ScrapeResult, result), {"command": "upcoming", "sport": "football"}
            )

        assert len(result.success) == 3
        assert len(result.failed) == 0
        mock_run.assert_awaited_once()  # only 1 pass needed

    @pytest.mark.asyncio
    async def test_delay_called_before_each_pass(self) -> None:
        """Each retry pass sleeps before launching the browser."""
        original_success = [{"match": 0}]
        failed_urls = [
            FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION)
        ]
        result = _make_result(success=original_success, failed=failed_urls)

        still_failed = _make_result(
            success=[],
            failed=[
                FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION)
            ],
        )
        mock_run = AsyncMock(return_value=still_failed)
        mock_sleep = AsyncMock()

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, mock_sleep):
            await _retry_failed_urls(
                cast(ScrapeResult, result), {"command": "upcoming", "sport": "football"}
            )

        assert mock_sleep.await_count == MAX_FAILED_URL_RETRY_PASSES


class TestUserAgentGeneration:
    def test_generate_user_agent_returns_chrome_string(self) -> None:
        with patch(USERAGENT_PATCH, _mock_useragent()):
            ua = _generate_user_agent()
        assert isinstance(ua, str)
        assert "Chrome" in ua

    @pytest.mark.asyncio
    async def test_auto_injects_user_agent(self) -> None:
        matches = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        mock_run = AsyncMock(return_value=_make_result(success=matches))

        with patch(SCRAPER_PATCH, mock_run), patch(USERAGENT_PATCH, _mock_useragent()):
            await run_scraper_with_retry(command="upcoming", sport="football", headless=True)

        call_kwargs = mock_run.call_args.kwargs
        assert "browser_user_agent" in call_kwargs
        assert "Chrome" in call_kwargs["browser_user_agent"]

    @pytest.mark.asyncio
    async def test_does_not_override_explicit_user_agent(self) -> None:
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
    async def test_single_attempt_generates_user_agent(self) -> None:
        """With MAX_SCRAPER_RETRIES=1, a single UA is generated for the one attempt."""
        matches = [{"home_team": "Arsenal", "away_team": "Chelsea"}]
        success = _make_result(success=matches)

        mock_run = AsyncMock(return_value=success)
        ua_gen = MagicMock(return_value="UA-1")

        with (
            patch(SCRAPER_PATCH, mock_run),
            patch("odds_lambda.oddsportal_common._generate_user_agent", ua_gen),
        ):
            result = await run_scraper_with_retry(
                command="upcoming", sport="football", headless=True
            )

        assert result.success == matches
        assert mock_run.await_count == 1
        assert mock_run.call_args.kwargs["browser_user_agent"] == "UA-1"
        ua_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_uses_fresh_user_agent(self) -> None:
        """Failed-URL retry generates a new UA, different from the initial one."""
        initial_success = [{"match": 0}]
        initial_failed = [
            FakeFailedUrl(url="https://oddsportal.com/match/1", error_type=ErrorType.NAVIGATION)
        ]
        initial_result = _make_result(success=initial_success, failed=initial_failed)

        retry_result = _make_result(success=[], failed=initial_failed)
        mock_run = AsyncMock(side_effect=[initial_result, retry_result, retry_result, retry_result])

        with (
            patch(SCRAPER_PATCH, mock_run),
            patch(SLEEP_PATCH, new_callable=AsyncMock),
            patch(USERAGENT_PATCH, _mock_useragent()),
        ):
            await run_scraper_with_retry(command="upcoming", sport="football", headless=True)

        initial_ua = mock_run.call_args_list[0].kwargs["browser_user_agent"]
        retry_ua = mock_run.call_args_list[1].kwargs["browser_user_agent"]
        assert "Chrome" in initial_ua
        assert "Chrome" in retry_ua
        assert "browser_user_agent" in mock_run.call_args_list[1].kwargs


class TestEndToEndWithFailedUrlRetry:
    @pytest.mark.asyncio
    async def test_partial_scrape_triggers_failed_url_retry(self) -> None:
        """Full flow: initial scrape partially fails, single retry pass recovers some."""
        initial_success = [{"match": i} for i in range(11)]
        initial_failed = [
            FakeFailedUrl(
                url=f"https://oddsportal.com/match/{i}", error_type=ErrorType.HEADER_NOT_FOUND
            )
            for i in range(7)
        ]
        initial_result = _make_result(success=initial_success, failed=initial_failed)

        retry_success = [{"match": "recovered_0"}]
        retry_failed = [
            FakeFailedUrl(
                url=f"https://oddsportal.com/match/{i}", error_type=ErrorType.HEADER_NOT_FOUND
            )
            for i in range(6)
        ]
        retry_result = _make_result(success=retry_success, failed=retry_failed)

        mock_run = AsyncMock(side_effect=[initial_result, retry_result])

        with patch(SCRAPER_PATCH, mock_run), patch(SLEEP_PATCH, new_callable=AsyncMock):
            result = await run_scraper_with_retry(
                command="upcoming", sport="football", markets=["1x2"], headless=True
            )

        assert len(result.success) == 12  # 11 + 1 recovered
        assert len(result.failed) == 6  # still failed after single pass
        # 1 initial + 1 retry pass
        assert mock_run.await_count == 2

        # Verify command was forwarded to retry calls
        retry_call_kwargs = mock_run.call_args_list[1].kwargs
        assert retry_call_kwargs["command"] == "upcoming"
