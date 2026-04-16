"""Local APScheduler 4 backend with Postgres-backed data store and event broker."""

from __future__ import annotations

from datetime import datetime
from types import TracebackType

import structlog
from apscheduler import AsyncScheduler, ConflictPolicy, SchedulerRole
from apscheduler.datastores.sqlalchemy import SQLAlchemyDataStore
from apscheduler.eventbrokers.asyncpg import AsyncpgEventBroker
from apscheduler.triggers.date import DateTrigger
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from odds_lambda.scheduling.backends.base import (
    BackendHealth,
    JobStatus,
    RetryConfig,
    ScheduledJob,
    SchedulerBackend,
    ValidationResult,
)
from odds_lambda.scheduling.exceptions import (
    BackendUnavailableError,
    CancellationFailedError,
    JobNotFoundError,
    SchedulingFailedError,
)

logger = structlog.get_logger()


_shared_engine: AsyncEngine | None = None


def _get_shared_engine() -> AsyncEngine:
    """Return a module-level engine, created once on first use."""
    global _shared_engine
    if _shared_engine is None:
        from odds_core.config import get_settings

        settings = get_settings()
        _shared_engine = create_async_engine(settings.database.url)
    return _shared_engine


def build_scheduler(
    *,
    role: SchedulerRole = SchedulerRole.both,
    engine: AsyncEngine | None = None,
) -> AsyncScheduler:
    """Construct an ``AsyncScheduler`` backed by the project Postgres database."""
    if engine is None:
        engine = _get_shared_engine()

    data_store = SQLAlchemyDataStore(engine)
    event_broker = AsyncpgEventBroker.from_async_sqla_engine(engine)

    return AsyncScheduler(
        data_store=data_store,
        event_broker=event_broker,
        role=role,
    )


class LocalSchedulerBackend(SchedulerBackend):
    """APScheduler 4 backend using Postgres-backed data store and event broker.

    Uses ``SQLAlchemyDataStore`` + ``AsyncpgEventBroker`` so that any process
    sharing the same Postgres database can add jobs that the scheduler picks up
    immediately via LISTEN/NOTIFY.

    Usage::

        async with LocalSchedulerBackend() as backend:
            # Bootstrap jobs, then run forever
            await asyncio.Event().wait()
    """

    def __init__(self, dry_run: bool = False, retry_config: RetryConfig | None = None) -> None:
        super().__init__(dry_run=dry_run, retry_config=retry_config)
        self._scheduler: AsyncScheduler | None = None
        self._configured_tasks: set[str] = set()
        logger.info("local_scheduler_initialized", dry_run=self.dry_run)

    def validate_configuration(self) -> ValidationResult:
        from odds_lambda.scheduling.health_check import ValidationBuilder

        builder = ValidationBuilder()

        try:
            from apscheduler import AsyncScheduler  # noqa: F401
        except ImportError:
            builder.add_error("APScheduler 4 not installed")

        try:
            from odds_lambda.scheduling.jobs import list_available_jobs

            jobs = list_available_jobs()
            if not jobs:
                builder.add_warning("Job registry is empty")
        except ImportError as e:
            builder.add_warning(f"Job registry import warning: {e}")

        return builder.build()

    async def health_check(self) -> BackendHealth:
        from odds_lambda.scheduling.health_check import HealthCheckBuilder

        builder = HealthCheckBuilder(self.get_backend_name())

        validation = self.validate_configuration()
        builder.check_condition(
            validation.is_valid,
            "Configuration valid",
            f"Configuration invalid: {', '.join(validation.errors)}",
        )

        if self._scheduler is not None:
            builder.pass_check("Scheduler running")
            builder.add_detail("scheduler_state", "running")

            schedules = await self._scheduler.get_schedules()
            builder.add_detail("scheduled_jobs_count", len(schedules))
            builder.pass_check(f"{len(schedules)} schedules registered")
        else:
            builder.pass_check("Scheduler initialized (not started)")
            builder.add_detail("scheduler_state", "stopped")

        return builder.build()

    async def get_scheduled_jobs(self) -> list[ScheduledJob]:
        if self.dry_run:
            logger.info("dry_run_get_scheduled_jobs")
            return []

        if self._scheduler is None:
            logger.warning("local_scheduler_not_started")
            return []

        schedules = await self._scheduler.get_schedules()
        jobs: list[ScheduledJob] = []
        for sched in schedules:
            jobs.append(
                ScheduledJob(
                    job_name=sched.id,
                    next_run_time=sched.next_fire_time,
                    status=JobStatus.SCHEDULED,
                )
            )
        return jobs

    async def get_job_status(self, job_name: str) -> ScheduledJob | None:
        if self.dry_run:
            logger.info("dry_run_get_job_status", job=job_name)
            return None

        if self._scheduler is None:
            logger.warning("local_scheduler_not_started")
            return None

        try:
            from apscheduler import ScheduleLookupError

            sched = await self._scheduler.get_schedule(job_name)
        except ScheduleLookupError:
            return None

        return ScheduledJob(
            job_name=job_name,
            next_run_time=sched.next_fire_time,
            status=JobStatus.SCHEDULED,
        )

    async def schedule_next_execution(
        self,
        job_name: str,
        next_time: datetime,
        payload: dict[str, object] | None = None,
    ) -> None:
        if self.dry_run:
            logger.info(
                "dry_run_schedule",
                job=job_name,
                next_time=next_time.isoformat(),
                backend="local_apscheduler",
            )
            return

        try:
            from odds_lambda.scheduling.jobs import JobContext, get_job_function, resolve_job_name

            base_name, resolved_sport = resolve_job_name(job_name)
            job_func = get_job_function(base_name)
            ctx_payload: dict[str, object] = {}
            if resolved_sport:
                ctx_payload["sport"] = resolved_sport
            if payload:
                ctx_payload.update(payload)
            ctx = JobContext.from_payload(ctx_payload)

            await self._add_schedule(job_func, job_name, next_time, ctx)

            logger.info(
                "local_job_scheduled",
                job=job_name,
                next_time=next_time.isoformat(),
            )

        except KeyError as e:
            raise SchedulingFailedError(str(e)) from e
        except Exception as e:
            logger.error(
                "local_scheduling_failed",
                job=job_name,
                next_time=next_time.isoformat(),
                error=str(e),
                exc_info=True,
            )
            raise SchedulingFailedError(f"Failed to schedule {job_name}: {e}") from e

    async def _add_schedule(
        self,
        job_func: object,
        job_name: str,
        next_time: datetime,
        ctx: object,
    ) -> None:
        """Write a schedule to the data store.

        If this backend is running as a context manager the existing
        scheduler is reused.  Otherwise a short-lived scheduler is
        created to write to the shared Postgres data store — the
        running scheduler process picks it up via LISTEN/NOTIFY.
        """
        if self._scheduler is not None:
            task_key = f"{job_func.__module__}.{job_func.__qualname__}"  # type: ignore[union-attr]
            if task_key not in self._configured_tasks:
                await self._scheduler.configure_task(job_func)
                self._configured_tasks.add(task_key)

            await self._scheduler.add_schedule(
                job_func,
                trigger=DateTrigger(run_time=next_time),
                id=job_name,
                args=[ctx],
                conflict_policy=ConflictPolicy.replace,
            )
        else:
            async with build_scheduler(role=SchedulerRole.scheduler) as scheduler:
                await scheduler.configure_task(job_func)
                await scheduler.add_schedule(
                    job_func,
                    trigger=DateTrigger(run_time=next_time),
                    id=job_name,
                    args=[ctx],
                    conflict_policy=ConflictPolicy.replace,
                )

    async def cancel_scheduled_execution(self, job_name: str) -> None:
        if self.dry_run:
            logger.info("dry_run_cancel", job=job_name, backend="local_apscheduler")
            return

        if self._scheduler is None:
            raise BackendUnavailableError("Scheduler not started")

        try:
            from apscheduler import ScheduleLookupError

            # Verify schedule exists before removing (remove silently ignores missing IDs)
            try:
                await self._scheduler.get_schedule(job_name)
            except ScheduleLookupError as e:
                raise JobNotFoundError(f"Job {job_name} not found") from e

            await self._scheduler.remove_schedule(job_name)
            logger.info("local_job_cancelled", job=job_name)
        except (JobNotFoundError, BackendUnavailableError):
            raise
        except Exception as e:
            logger.error(
                "local_cancel_failed",
                job=job_name,
                error=str(e),
                exc_info=True,
            )
            raise CancellationFailedError(f"Failed to cancel {job_name}: {e}") from e

    def get_backend_name(self) -> str:
        return "local_apscheduler"

    async def __aenter__(self) -> LocalSchedulerBackend:
        self._scheduler = build_scheduler()
        await self._scheduler.__aenter__()
        await self._scheduler.start_in_background()
        logger.info("local_scheduler_started")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if self._scheduler is not None:
            await self._scheduler.__aexit__(exc_type, exc_val, exc_tb)
            self._scheduler = None
            self._configured_tasks.clear()
        logger.info("local_scheduler_shutdown")
        return False
