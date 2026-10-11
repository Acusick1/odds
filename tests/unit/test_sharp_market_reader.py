"""OddsReader.get_sharp_market: newest snapshot where one sharp book quotes every outcome."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, OddsSnapshot
from odds_lambda.storage.readers import OddsReader

CUTOFF = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)
OUTCOMES = ["Arsenal", "Chelsea", "Draw"]
SHARP = ["pinnacle", "betfair_exchange"]


def _book(key: str, market: str = "h2h", names: tuple[str, ...] = tuple(OUTCOMES)) -> dict:
    prices = {"Arsenal": -110, "Chelsea": 300, "Draw": 300}
    return {
        "key": key,
        "title": key,
        "markets": [
            {
                "key": market,
                "outcomes": [{"name": n, "price": prices[n]} for n in names],
            }
        ],
    }


async def _add(session, minutes_before_cutoff: int, *books: dict) -> int:
    snapshot = OddsSnapshot(
        event_id="evt-1",
        snapshot_time=CUTOFF - timedelta(minutes=minutes_before_cutoff),
        raw_data={"bookmakers": list(books)},
        bookmaker_count=len(books),
    )
    session.add(snapshot)
    await session.flush()
    assert snapshot.id is not None
    return snapshot.id


@pytest.fixture
async def reader(test_session) -> OddsReader:
    test_session.add(
        Event(
            id="evt-1",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=CUTOFF,
            home_team="Arsenal",
            away_team="Chelsea",
        )
    )
    await test_session.flush()
    return OddsReader(test_session)


async def _lookup(reader: OddsReader, lookback_hours: float = 24.0):
    return await reader.get_sharp_market(
        "evt-1",
        markets=["h2h", "1x2"],
        outcomes=OUTCOMES,
        sharp_bookmakers=SHARP,
        lookback_hours=lookback_hours,
        now=CUTOFF,
    )


class TestGetSharpMarket:
    async def test_older_full_market_beats_newer_partial_one(
        self, reader: OddsReader, test_session
    ) -> None:
        await _add(test_session, 120, _book("pinnacle"))
        await _add(test_session, 5, _book("pinnacle", names=("Arsenal",)))

        market = await _lookup(reader)

        assert market is not None
        assert market.bookmaker == "pinnacle"
        assert market.snapshot_time == CUTOFF - timedelta(minutes=120)
        assert market.implied_probs["Arsenal"] == pytest.approx(0.52381, abs=1e-5)
        assert set(market.implied_probs) == set(OUTCOMES)

    async def test_newest_complete_snapshot_wins(self, reader: OddsReader, test_session) -> None:
        await _add(test_session, 120, _book("pinnacle"))
        await _add(test_session, 5, _book("pinnacle"))

        market = await _lookup(reader)

        assert market is not None
        assert market.snapshot_time == CUTOFF - timedelta(minutes=5)

    async def test_fallback_book_when_primary_never_complete(
        self, reader: OddsReader, test_session
    ) -> None:
        await _add(
            test_session,
            5,
            _book("pinnacle", names=("Arsenal",)),
            _book("betfair_exchange"),
        )

        market = await _lookup(reader)

        assert market is not None
        assert market.bookmaker == "betfair_exchange"

    async def test_primary_book_wins_even_if_older_than_fallback(
        self, reader: OddsReader, test_session
    ) -> None:
        await _add(test_session, 120, _book("pinnacle"))
        await _add(test_session, 5, _book("betfair_exchange"))

        market = await _lookup(reader)

        assert market is not None
        assert market.bookmaker == "pinnacle"

    async def test_1x2_key_is_searched(self, reader: OddsReader, test_session) -> None:
        await _add(test_session, 5, _book("betfair_exchange", market="1x2"))

        market = await _lookup(reader)

        assert market is not None
        assert market.bookmaker == "betfair_exchange"

    async def test_none_when_no_sharp_book_is_complete(
        self, reader: OddsReader, test_session
    ) -> None:
        await _add(test_session, 5, _book("bet365"), _book("pinnacle", names=("Arsenal",)))

        assert await _lookup(reader) is None

    async def test_none_without_snapshots(self, reader: OddsReader) -> None:
        assert await _lookup(reader) is None

    async def test_snapshots_after_cutoff_and_outside_lookback_are_ignored(
        self, reader: OddsReader, test_session
    ) -> None:
        await _add(test_session, -10, _book("pinnacle"))
        await _add(test_session, 60 * 25, _book("pinnacle"))

        assert await _lookup(reader) is None
