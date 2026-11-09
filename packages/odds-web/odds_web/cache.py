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
# SQLSTATE classes of a request the server rejected: feature not supported, data exception,
# integrity violation, invalid transaction state, invalid authorization (a wrong password), no
# such database, syntax error or access rule violation (a missing grant). asyncpg's errors reach SQLAlchemy mostly as a bare DBAPIError, so the class of
# the exception cannot tell these apart from a statement timeout (57014); the SQLSTATE can.
QUERY_SQLSTATE_CLASSES: frozenset[str] = frozenset({"0A", "22", "23", "25", "28", "3D", "42"})
# OSErrors from reading files, such as the candidate registry, rather than from the network.
FILE_ERRORS: tuple[type[OSError], ...] = (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
)


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


def _sqlstate(exc: BaseException) -> str | None:
    for error in (exc, getattr(exc, "orig", None)):
        code = getattr(error, "sqlstate", None)
        if isinstance(code, str):
            return code
    return None


def failure_kind(exc: Exception) -> FailureKind:
    """``UNAVAILABLE`` when the database could not answer, ``FAULT`` for anything else.

    An error with a SQLSTATE is a fault when its class is one of ``QUERY_SQLSTATE_CLASSES`` and
    unavailable otherwise (a cancelled statement, a server starting up or out of resources). A
    driver error without one, the pool running dry, and a network error are unavailable.
    """
    sqlstate = _sqlstate(exc)
    if sqlstate is not None:
        return (
            FailureKind.FAULT if sqlstate[:2] in QUERY_SQLSTATE_CLASSES else FailureKind.UNAVAILABLE
        )
    if isinstance(exc, ProgrammingError):
        return FailureKind.FAULT
    if isinstance(exc, DBAPIError | PoolTimeoutError):
        return FailureKind.UNAVAILABLE
    if isinstance(exc, OSError) and not isinstance(exc, FILE_ERRORS):
        return FailureKind.UNAVAILABLE
    return FailureKind.FAULT


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

        A failed load is logged with its traceback and keeps the last good value. Its kind is
        ``failure_kind``'s; a fault is still caught, so one broken section leaves the rest of the
        page standing.
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
                kind = failure_kind(exc)
                if kind is FailureKind.UNAVAILABLE:
                    logger.warning("web_section_unavailable", section=key, exc_info=True)
                else:
                    logger.error("web_section_failed", section=key, exc_info=True)
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
