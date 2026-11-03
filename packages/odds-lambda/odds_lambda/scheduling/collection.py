from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto

import structlog
from odds_core.alerts import job_alert_context

from odds_lambda.scheduling.decision import ScheduleDecision
from odds_lambda.scheduling.helpers import self_schedule

logger = structlog.get_logger()


class PostbookTiming(Enum):
    AFTER_HEARTBEAT = auto()
    BEFORE_HEARTBEAT = auto()


@dataclass(frozen=True, slots=True)
class CollectionResult:
    postbook: bool = True
    failure: Exception | None = None


async def run_collection(
    *,
    job_name: str,
    sport: str,
    dry_run: bool,
    decide: Callable[[], Awaitable[ScheduleDecision]],
    collect: Callable[[], Awaitable[CollectionResult]],
    post_label: str,
    postbook_timing: PostbookTiming = PostbookTiming.AFTER_HEARTBEAT,
    prepare: Callable[[], None] | None = None,
) -> None:
    async def book(label: str, *, required: bool = False) -> ScheduleDecision:
        decision = await decide()
        assert decision.next_execution is not None  # noqa: S101
        try:
            await self_schedule(
                job_name=job_name,
                next_time=decision.next_execution,
                dry_run=dry_run,
                sport=sport,
                reason=f"{label}: {decision.reason}",
            )
        except Exception:
            logger.error("collection_scheduling_failed", job_name=job_name, exc_info=True)
            if required:
                raise
        return decision

    decision = await book("pre-schedule", required=True)
    if not decision.should_execute:
        async with job_alert_context(job_name):
            logger.info(
                "collection_dormant", job_name=job_name, sport=sport, reason=decision.reason
            )
        return

    if prepare is not None:
        prepare()
    async with job_alert_context(job_name):
        result = await collect()
        if postbook_timing is PostbookTiming.BEFORE_HEARTBEAT and result.postbook:
            await book(post_label)
        if result.failure is not None:
            raise result.failure
    if postbook_timing is PostbookTiming.AFTER_HEARTBEAT and result.postbook:
        await book(post_label)
