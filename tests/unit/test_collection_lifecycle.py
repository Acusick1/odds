from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_lambda.scheduling.collection import CollectionResult, PostbookTiming, run_collection
from odds_lambda.scheduling.decision import ScheduleDecision


class TestCollectionLifecycle:
    async def exercise(
        self,
        *,
        active: bool = True,
        postbook: bool = True,
        deferred_failure: Exception | None = None,
        failure: BaseException | None = None,
        book_failure: int | None = None,
        inside: bool = False,
        prepare_failure: bool = False,
    ) -> list[str]:
        events: list[str] = []
        bookings = 0

        async def decide() -> ScheduleDecision:
            events.append("decide")
            return ScheduleDecision(active, "test", datetime.now(UTC) + timedelta(hours=1), None)

        async def book(**kwargs: object) -> None:
            nonlocal bookings
            bookings += 1
            events.append("book")
            if bookings == book_failure:
                raise OSError("booking failed")

        @asynccontextmanager
        async def alerts(name: str) -> AsyncIterator[None]:
            events.append("enter")
            try:
                yield
            except BaseException:
                events.append("failed")
                raise
            else:
                events.append("heartbeat")

        def prepare() -> None:
            events.append("prepare")
            if prepare_failure:
                raise ValueError("cert failed")

        async def collect() -> CollectionResult:
            events.append("collect")
            if failure is not None:
                raise failure
            return CollectionResult(postbook=postbook, failure=deferred_failure)

        with (
            patch("odds_lambda.scheduling.collection.self_schedule", side_effect=book),
            patch("odds_lambda.scheduling.collection.job_alert_context", alerts),
        ):
            try:
                await run_collection(
                    job_name="fetch-oddsportal-epl",
                    sport="soccer_epl",
                    dry_run=True,
                    decide=decide,
                    collect=collect,
                    prepare=prepare,
                    post_label="post-fetch",
                    postbook_timing=(
                        PostbookTiming.BEFORE_HEARTBEAT
                        if inside
                        else PostbookTiming.AFTER_HEARTBEAT
                    ),
                )
            except BaseException as exc:
                exc.add_note(",".join(events))
                raise
        return events

    async def test_prebooks_before_work_and_requeries_after_heartbeat(self) -> None:
        assert await self.exercise() == [
            "decide",
            "book",
            "prepare",
            "enter",
            "collect",
            "heartbeat",
            "decide",
            "book",
        ]

    async def test_dormancy_books_and_heartbeats_without_preparing_or_collecting(self) -> None:
        assert await self.exercise(active=False) == ["decide", "book", "enter", "heartbeat"]

    async def test_empty_success_can_keep_prebooking_without_postbooking(self) -> None:
        assert await self.exercise(postbook=False) == [
            "decide",
            "book",
            "prepare",
            "enter",
            "collect",
            "heartbeat",
        ]

    async def test_prebooking_failure_prevents_work_and_heartbeat(self) -> None:
        with pytest.raises(OSError) as error:
            await self.exercise(book_failure=1)
        assert error.value.__notes__ == ["decide,book"]

    async def test_postbooking_failure_preserves_success(self) -> None:
        assert (await self.exercise(book_failure=2))[-3:] == ["heartbeat", "decide", "book"]

    async def test_preparation_failure_stays_outside_alert_context(self) -> None:
        with pytest.raises(ValueError) as error:
            await self.exercise(prepare_failure=True)
        assert error.value.__notes__ == ["decide,book,prepare"]

    @pytest.mark.parametrize("failure", [OSError("source failed"), asyncio.CancelledError()])
    async def test_source_failure_or_cancellation_never_postbooks_or_heartbeats(
        self, failure: BaseException
    ) -> None:
        with pytest.raises(type(failure)) as error:
            await self.exercise(failure=failure)
        assert error.value.__notes__ == ["decide,book,prepare,enter,collect,failed"]

    async def test_polymarket_books_before_heartbeat(self) -> None:
        assert (await self.exercise(inside=True))[-3:] == ["decide", "book", "heartbeat"]

    async def test_polymarket_deferred_failure_books_then_fails_without_heartbeat(self) -> None:
        with pytest.raises(RuntimeError) as error:
            await self.exercise(deferred_failure=RuntimeError("gamma failed"), inside=True)
        assert error.value.__notes__ == ["decide,book,prepare,enter,collect,decide,book,failed"]

    async def test_booking_cancellation_propagates(self) -> None:
        collect = AsyncMock()
        decision = ScheduleDecision(True, "test", datetime.now(UTC) + timedelta(hours=1), None)
        with (
            patch(
                "odds_lambda.scheduling.collection.self_schedule",
                side_effect=asyncio.CancelledError,
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await run_collection(
                job_name="fetch-oddsportal-epl",
                sport="soccer_epl",
                dry_run=True,
                decide=AsyncMock(return_value=decision),
                collect=collect,
                post_label="post-fetch",
            )
        collect.assert_not_awaited()
