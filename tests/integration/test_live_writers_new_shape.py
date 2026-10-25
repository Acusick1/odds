"""OddsPortal live and Betfair write snapshots, prices and payloads in the new shape (#672)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from odds_core.models import (
    DataQualityLog,
    Event,
    EventExternalId,
    Odds,
    OddsSnapshot,
    Price,
    SnapshotPayload,
)
from odds_core.vocabulary import Market, MarketStatus, Source, TimeBasis
from odds_lambda.betfair.client import BetfairBook, BetfairRunner
from odds_lambda.betfair.constants import SPORT_CONFIG
from odds_lambda.jobs import fetch_betfair_exchange
from odds_lambda.jobs.fetch_oddsportal import LEAGUE_SPEC_BY_NAME, LeagueScrape, ingest_league
from odds_lambda.oddsportal_adapter import MatchOdds
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import FakeBetfairClient

EPL = "soccer_epl"
HOME, AWAY = "Arsenal", "Chelsea"
KICKOFF = (datetime.now(UTC) + timedelta(days=2)).replace(
    hour=15, minute=0, second=0, microsecond=0
)


def _book_entry(key: str, prices: tuple[float, ...], **extra: Any) -> dict[str, Any]:
    names = [HOME, "Draw", AWAY] if len(prices) == 3 else ["Over", "Under"]
    market = "1x2" if len(prices) == 3 else "totals"
    outcomes = [
        {"name": n, "price_decimal": p, **({"point": 2.5} if market == "totals" else {})}
        for n, p in zip(names, prices, strict=True)
    ]
    return {
        "key": key,
        "title": key,
        "last_update": "2026-10-24T12:00:00Z",
        "markets": [{"key": market, "outcomes": outcomes}],
        **extra,
    }


def _op_match(
    market: str, books: list[dict[str, Any]], scraped: datetime, start: datetime = KICKOFF
) -> MatchOdds:
    return MatchOdds(
        market=market,
        home_team=HOME,
        away_team=AWAY,
        match_date=start,
        match_link="https://www.oddsportal.com/football/england/premier-league/x-Kl1AOZN9/",
        scraped_date=scraped,
        raw_data={
            "bookmakers": books,
            "source": "oddsportal_live",
            "oddsportal_event_id": "Kl1AOZN9",
        },
        bookmaker_count=len(books),
    )


def _scrape(scraped: datetime, start: datetime = KICKOFF) -> LeagueScrape:
    return LeagueScrape(
        matches=[
            _op_match(
                "1x2",
                [
                    _book_entry(
                        "betfair_exchange",
                        (2.1, 3.4, 3.6),
                        betfair_matched={"home": 5117, "draw": 16, "away": 900},
                    ),
                    _book_entry("bet365", (2.0, 3.3, 3.5)),
                ],
                scraped,
                start,
            ),
            _op_match(
                "over_under_2_5",
                [_book_entry("bet365", (1.9, 1.95))],
                scraped + timedelta(seconds=1),
                start,
            ),
        ],
        events_listed=1,
        upcoming_listed=1,
    )


async def _all(session: AsyncSession, model: type) -> list[Any]:
    return list((await session.execute(select(model))).scalars().all())


async def _count(session: AsyncSession, model: type) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def _external_ids(session: AsyncSession) -> set[tuple[str, str]]:
    return {(row.source.value, row.external_id) for row in await _all(session, EventExternalId)}


class TestOddsPortalLive:
    async def test_writes_each_market_snapshot_with_prices_and_one_payload(
        self, job_session: AsyncSession
    ) -> None:
        scraped = datetime.now(UTC).replace(microsecond=0)

        stats = await ingest_league(
            LEAGUE_SPEC_BY_NAME["england-premier-league"], scraped=_scrape(scraped)
        )

        assert (stats.events_created, stats.events_matched, stats.snapshots_stored) == (1, 1, 2)
        snapshots = sorted(await _all(job_session, OddsSnapshot), key=lambda s: s.snapshot_time)
        assert [(s.source, s.market, s.time_basis) for s in snapshots] == [
            (Source.ODDSPORTAL, Market.ONE_X_TWO, TimeBasis.OBSERVED),
            (Source.ODDSPORTAL, Market.TOTALS, TimeBasis.OBSERVED),
        ]
        [payload] = await _all(job_session, SnapshotPayload)
        assert {s.payload_id for s in snapshots} == {payload.id}
        assert [m["source"] for m in payload.payload] == ["oddsportal_live"] * 2
        assert all(s.api_request_id.startswith("oddsportal_live_") for s in snapshots)
        assert snapshots[0].raw_data["oddsportal_event_id"] == "Kl1AOZN9"

        prices = await _all(job_session, Price)
        rows = {
            (p.snapshot_id, p.bookmaker, p.outcome.value, p.line, p.side.value): (
                p.price,
                p.matched,
            )
            for p in prices
        }
        assert len(rows) == 8
        assert rows[(snapshots[0].id, "betfair_exchange", "home", None, "back")] == (
            Decimal("2.1"),
            Decimal("5117"),
        )
        assert rows[(snapshots[0].id, "bet365", "draw", None, "back")] == (Decimal("3.3"), None)
        assert rows[(snapshots[1].id, "bet365", "over", Decimal("2.5"), "back")] == (
            Decimal("1.9"),
            None,
        )
        assert await _count(job_session, Odds) == 8

    async def test_records_the_oddsportal_event_id(self, job_session: AsyncSession) -> None:
        await ingest_league(
            LEAGUE_SPEC_BY_NAME["england-premier-league"], scraped=_scrape(datetime.now(UTC))
        )

        [event] = await _all(job_session, Event)
        assert await _external_ids(job_session) == {("oddsportal", "Kl1AOZN9")}
        assert event.id.startswith("op_live_ARS_CHE_")

    async def test_a_repeated_scrape_at_the_same_time_writes_nothing_twice(
        self, job_session: AsyncSession
    ) -> None:
        scraped = datetime.now(UTC)
        spec = LEAGUE_SPEC_BY_NAME["england-premier-league"]
        await ingest_league(spec, scraped=_scrape(scraped))

        stats = await ingest_league(spec, scraped=_scrape(scraped))

        assert stats.errors == []
        assert await _count(job_session, OddsSnapshot) == 2
        assert await _count(job_session, Price) == 8
        assert await _count(job_session, Odds) == 8

    async def test_a_moved_future_start_moves_the_event(self, job_session: AsyncSession) -> None:
        spec = LEAGUE_SPEC_BY_NAME["england-premier-league"]
        await ingest_league(spec, scraped=_scrape(datetime.now(UTC)))

        moved = KICKOFF + timedelta(minutes=90)
        await ingest_league(spec, scraped=_scrape(datetime.now(UTC), start=moved))

        [event] = await _all(job_session, Event)
        assert event.commence_time == moved

    async def test_alias_collision_keeps_the_first_book_and_logs_the_second(
        self, job_session: AsyncSession
    ) -> None:
        scrape = LeagueScrape(
            matches=[
                _op_match(
                    "1x2",
                    [_book_entry("7bet", (2.1, 3.4, 3.6)), _book_entry("7betuk", (2.2, 3.5, 3.7))],
                    datetime.now(UTC),
                )
            ],
            events_listed=1,
            upcoming_listed=1,
        )

        await ingest_league(LEAGUE_SPEC_BY_NAME["england-premier-league"], scraped=scrape)

        prices = await _all(job_session, Price)
        assert {(p.bookmaker, p.price) for p in prices if p.outcome.value == "home"} == {
            ("7bet", Decimal("2.1"))
        }
        [log] = [
            row
            for row in await _all(job_session, DataQualityLog)
            if row.issue_type == "bookmaker_alias_collision"
        ]
        assert log.raw_data["dropped_key"] == "7betuk"
        # The old shape is written as before: both entries reach the odds table.
        assert await _count(job_session, Odds) == 6


def _betfair_book(
    start: datetime, *, status: str = "OPEN", inplay: bool = False, event_id: str = "35001"
) -> BetfairBook:
    return BetfairBook(
        market_id="1.234",
        betfair_event_id=event_id,
        betfair_event_name=f"{HOME} v {AWAY}",
        market_start_time=start,
        market_status=status,
        inplay=inplay,
        total_matched=385702.5,
        runners=[
            BetfairRunner(1, HOME, 2.1, 2.12, 292.0, 314.0, 2.1),
            BetfairRunner(2, AWAY, 3.6, 3.7, 50.0, 40.0, None),
            BetfairRunner(3, "The Draw", 3.4, 3.45, 80.0, 70.0, 3.4),
        ],
    )


async def _ingest_betfair(*books: BetfairBook, snapshot_time: datetime) -> Any:
    return await fetch_betfair_exchange.ingest_sport(
        SPORT_CONFIG[EPL],
        client=FakeBetfairClient(list(books)),  # type: ignore[arg-type]
        lookahead_hours=72,
        snapshot_time=snapshot_time,
    )


class TestBetfair:
    async def test_writes_back_lay_and_traded_rows_and_market_metadata(
        self, job_session: AsyncSession
    ) -> None:
        now = datetime.now(UTC)

        stats = await _ingest_betfair(_betfair_book(KICKOFF), snapshot_time=now)

        assert (stats.events_created, stats.snapshots_stored) == (1, 1)
        [snapshot] = await _all(job_session, OddsSnapshot)
        assert (snapshot.source, snapshot.market, snapshot.time_basis) == (
            Source.BETFAIR,
            Market.ONE_X_TWO,
            TimeBasis.OBSERVED,
        )
        assert snapshot.in_play is False
        assert snapshot.market_status is MarketStatus.OPEN
        assert snapshot.total_matched == Decimal("385702.5")
        assert snapshot.market_start_time == KICKOFF
        assert snapshot.raw_data["bookmakers"][0]["betfair_meta"]["market_status"] == "OPEN"
        [payload] = await _all(job_session, SnapshotPayload)
        assert snapshot.payload_id == payload.id

        rows = {
            (p.outcome.value, p.side.value): (p.price, p.size)
            for p in await _all(job_session, Price)
        }
        assert rows == {
            ("home", "back"): (Decimal("2.1"), Decimal("292.0")),
            ("home", "lay"): (Decimal("2.12"), Decimal("314.0")),
            ("home", "traded"): (Decimal("2.1"), None),
            ("away", "back"): (Decimal("3.6"), Decimal("50.0")),
            ("away", "lay"): (Decimal("3.7"), Decimal("40.0")),
            ("draw", "back"): (Decimal("3.4"), Decimal("80.0")),
            ("draw", "lay"): (Decimal("3.45"), Decimal("70.0")),
            ("draw", "traded"): (Decimal("3.4"), None),
        }
        assert await _count(job_session, Odds) == 3
        assert await _external_ids(job_session) == {("betfair", "35001")}

    async def test_betfair_and_oddsportal_land_on_one_event_with_both_ids(
        self, job_session: AsyncSession
    ) -> None:
        await ingest_league(
            LEAGUE_SPEC_BY_NAME["england-premier-league"], scraped=_scrape(datetime.now(UTC))
        )

        await _ingest_betfair(
            _betfair_book(KICKOFF + timedelta(minutes=30)), snapshot_time=datetime.now(UTC)
        )

        [event] = await _all(job_session, Event)
        assert event.commence_time == KICKOFF
        assert await _external_ids(job_session) == {
            ("oddsportal", "Kl1AOZN9"),
            ("betfair", "35001"),
        }

    async def test_suspended_market_status_is_lower_cased(self, job_session: AsyncSession) -> None:
        await _ingest_betfair(
            _betfair_book(KICKOFF, status="SUSPENDED"), snapshot_time=datetime.now(UTC)
        )

        [snapshot] = await _all(job_session, OddsSnapshot)
        assert snapshot.market_status is MarketStatus.SUSPENDED

    async def test_invalid_lay_price_is_logged_and_not_stored(
        self, job_session: AsyncSession
    ) -> None:
        book = _betfair_book(KICKOFF)
        book.runners[0] = BetfairRunner(1, HOME, 2.1, 1.0, 292.0, 314.0, 2.1)

        await _ingest_betfair(book, snapshot_time=datetime.now(UTC))

        sides = {(p.outcome.value, p.side.value) for p in await _all(job_session, Price)}
        assert ("home", "back") in sides
        assert ("home", "lay") not in sides
        assert "invalid_price" in {r.issue_type for r in await _all(job_session, DataQualityLog)}
