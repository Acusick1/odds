from __future__ import annotations

import asyncio

import pytest
from odds_web.cache import SingleFlightCache


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Loader:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self) -> object:
        self.calls += 1
        await asyncio.sleep(0)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def cache(clock: Clock) -> SingleFlightCache:
    return SingleFlightCache(ttl=60, failure_backoff=10, clock=clock)


class TestSingleFlightCache:
    async def test_a_value_is_reused_within_the_ttl_and_reloaded_after(
        self, cache: SingleFlightCache, clock: Clock
    ) -> None:
        loader = Loader("a", "b")

        assert (await cache.get("k", loader)).value == "a"
        clock.now += 59
        assert (await cache.get("k", loader)).value == "a"
        clock.now += 2
        assert (await cache.get("k", loader)).value == "b"
        assert loader.calls == 2

    async def test_concurrent_readers_share_one_load(self, cache: SingleFlightCache) -> None:
        loader = Loader("a")

        results = await asyncio.gather(*(cache.get("k", loader) for _ in range(5)))

        assert loader.calls == 1
        assert {r.value for r in results} == {"a"}

    async def test_keys_load_independently(self, cache: SingleFlightCache) -> None:
        assert (await cache.get("x", Loader(1))).value == 1
        assert (await cache.get("y", Loader(2))).value == 2

    async def test_a_failure_before_any_value_is_an_error_without_a_time(
        self, cache: SingleFlightCache
    ) -> None:
        result = await cache.get("k", Loader(RuntimeError("down")))

        assert result.value is None and result.as_of is None
        assert result.error == "RuntimeError: down"

    async def test_a_failed_refresh_keeps_the_last_value_and_its_time(
        self, cache: SingleFlightCache, clock: Clock
    ) -> None:
        loaded = await cache.get("k", Loader("a"))
        clock.now += 61

        result = await cache.get("k", Loader(RuntimeError("down")))

        assert result.value == "a" and result.as_of == loaded.as_of
        assert result.error is not None

    async def test_a_failure_is_not_retried_within_the_backoff(
        self, cache: SingleFlightCache, clock: Clock
    ) -> None:
        loader = Loader(RuntimeError("down"), "a")
        await cache.get("k", loader)
        clock.now += 9

        assert (await cache.get("k", loader)).error is not None
        assert loader.calls == 1

        clock.now += 2
        recovered = await cache.get("k", loader)
        assert recovered.value == "a" and recovered.error is None
