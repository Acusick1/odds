"""A TTL cache whose refresh is single-flight and whose failures leave the last value in place."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

TTL_SECONDS = 60.0
# A failed refresh is not retried for this long, so a down database sees one query per window.
FAILURE_BACKOFF_SECONDS = 10.0


@dataclass(frozen=True)
class Cached[T]:
    """``value`` is the last good value, None before one exists; ``error`` is set when the
    refresh behind this result failed, and ``as_of`` is then when ``value`` was loaded."""

    value: T | None
    as_of: datetime | None
    error: str | None


@dataclass
class _Entry[T]:
    value: T | None = None
    as_of: datetime | None = None
    loaded_at: float | None = None
    error: str | None = None
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
        entry: _Entry[T] = self._entries.setdefault(key, _Entry())
        if self._settled(entry):
            return self._result(entry)
        async with self._locks.setdefault(key, asyncio.Lock()):
            if self._settled(entry):
                return self._result(entry)
            try:
                value = await loader()
            except Exception as exc:
                entry.error = f"{type(exc).__name__}: {exc}"
                entry.failed_at = self._clock()
            else:
                entry.value, entry.as_of = value, datetime.now(UTC)
                entry.loaded_at, entry.error, entry.failed_at = self._clock(), None, None
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
        return Cached(entry.value, entry.as_of, entry.error)
