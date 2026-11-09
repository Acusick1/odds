"""A TTL cache whose refresh is single-flight and whose failures leave the last value in place."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import structlog
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

logger = structlog.get_logger()

TTL_SECONDS = 60.0
# A failed refresh is not retried for this long, so a down database sees one query per window.
FAILURE_BACKOFF_SECONDS = 10.0
# What the database raises when it cannot answer: refused, unreachable, timed out, out of pool.
DATABASE_ERRORS: tuple[type[BaseException], ...] = (
    DBAPIError,
    PoolTimeoutError,
    OSError,
    TimeoutError,
)
# Driver errors the query itself causes, such as bad SQL or a table the role has no grant on.
QUERY_ERRORS: tuple[type[BaseException], ...] = (ProgrammingError,)


class FailureKind(StrEnum):
    # The database could not answer.
    UNAVAILABLE = "unavailable"
    # Anything else: a defect in odds-web or the code it calls.
    FAULT = "fault"


@dataclass(frozen=True)
class Failure:
    kind: FailureKind
    exception: str
    message: str


@dataclass(frozen=True)
class Cached[T]:
    """``value`` is the last good value and ``as_of`` when it was loaded, both None before one
    exists; ``failure`` is set when the refresh behind this result failed."""

    value: T | None
    as_of: datetime | None
    failure: Failure | None


@dataclass
class _Entry[T]:
    value: T | None = None
    as_of: datetime | None = None
    loaded_at: float | None = None
    failure: Failure | None = None
    failed_at: float | None = None


class SingleFlightCache:
    """At most one loader runs per key; concurrent readers wait for it and share its result."""

    def __init__(
        self,
        ttl: float = TTL_SECONDS,
        failure_backoff: float = FAILURE_BACKOFF_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._failure_backoff = failure_backoff
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get[T](self, key: str, loader: Callable[[], Awaitable[T]]) -> Cached[T]:
        """The key's value, loading it when it is older than the TTL.

        A failed load is logged with its traceback and keeps the last good value. A database
        error other than a query error is ``FailureKind.UNAVAILABLE``; any other exception is
        ``FailureKind.FAULT``, still caught so one broken section leaves the rest of the page
        standing.
        """
        entry: _Entry[T] = self._entries.setdefault(key, _Entry())
        if self._settled(entry):
            return self._result(entry)
        async with self._locks.setdefault(key, asyncio.Lock()):
            if self._settled(entry):
                return self._result(entry)
            try:
                value = await loader()
            except Exception as exc:
                if isinstance(exc, DATABASE_ERRORS) and not isinstance(exc, QUERY_ERRORS):
                    logger.warning("web_section_unavailable", section=key, exc_info=True)
                    kind = FailureKind.UNAVAILABLE
                else:
                    logger.error("web_section_failed", section=key, exc_info=True)
                    kind = FailureKind.FAULT
                entry.failure = Failure(kind, type(exc).__name__, str(exc))
                entry.failed_at = self._clock()
            else:
                entry.value, entry.as_of = value, datetime.now(UTC)
                entry.loaded_at, entry.failure, entry.failed_at = self._clock(), None, None
            return self._result(entry)

    def _settled(self, entry: _Entry) -> bool:
        """Whether the entry answers without a refresh: loaded within the TTL, or failed within
        the backoff."""
        now = self._clock()
        if entry.failed_at is not None:
            return now - entry.failed_at < self._failure_backoff
        return entry.loaded_at is not None and now - entry.loaded_at < self._ttl

    @staticmethod
    def _result[T](entry: _Entry[T]) -> Cached[T]:
        return Cached(entry.value, entry.as_of, entry.failure)
