"""Tests for OddsReader.get_latest_book_prices per-bookmaker lookback resolution."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.storage.readers import BookPriceResult, OddsReader


def _make_event(event_id: str = "evt-1") -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=datetime(2026, 4, 25, 15, 0, tzinfo=UTC),
        home_team="Brighton",
        away_team="Wolves",
        status=EventStatus.SCHEDULED,
    )


def _make_raw_data(
    bookmakers: list[dict[str, object]],
    snapshot_time: datetime,
    market_key: str = "1x2",
) -> dict:
    """Build a raw_data blob from a compact bookmaker spec.

    Each entry in *bookmakers* is ``{"key": str, "outcomes": list[dict]}``.
    """
    return {
        "bookmakers": [
            {
                "key": bm["key"],
                "title": str(bm["key"]).title(),
                "last_update": snapshot_time.isoformat(),
                "markets": [{"key": market_key, "outcomes": bm["outcomes"]}],
            }
            for bm in bookmakers
        ]
    }


def _make_snapshot(
    event_id: str,
    snapshot_time: datetime,
    bookmakers: list[dict[str, object]],
) -> OddsSnapshot:
    return OddsSnapshot(
        event_id=event_id,
        snapshot_time=snapshot_time,
        raw_data=_make_raw_data(bookmakers, snapshot_time),
        bookmaker_count=len(bookmakers),
        fetch_tier="pregame",
        hours_until_commence=2.0,
    )


# Standard 17-book retail panel mirroring the OddsPortal scrape coverage
# documented in CLAUDE.md (~30 bookmakers per snapshot, including these 17).
_RETAIL_BOOKS_17 = [
    "bet365",
    "betway",
    "betfred",
    "betvictor",
    "bwin",
    "paddypower",
    "skybet",
    "williamhill",
    "10bet",
    "betmgm",
    "888sport",
    "midnite",
    "betuk",
    "spreadex",
    "betano",
    "unibet_uk",
    "allbritishcasino",
]


def _retail_panel(prices: dict[str, int]) -> list[dict[str, object]]:
    """Build the 17-retail-book bookmaker spec sharing a common price set.

    *prices* maps outcome name to American odds.
    """
    return [
        {
            "key": book,
            "outcomes": [{"name": name, "price": price} for name, price in prices.items()],
        }
        for book in _RETAIL_BOOKS_17
    ]


class TestGetLatestBookPrices:
    """Per-bookmaker latest-within-lookback resolution."""

    @pytest.mark.asyncio
    async def test_bfe_row_newer_than_op_row_surfaces_17_retail(self, test_session) -> None:
        """When a Betfair-Exchange-only snapshot wins the latest-snapshot race,
        the older OP snapshot's 17 retail books are still surfaced.
        """
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        anchor = datetime(2026, 4, 25, 13, 0, tzinfo=UTC)

        # Older OP snapshot: 17 retail books + pinnacle
        op_time = anchor - timedelta(minutes=30)
        op_snapshot = _make_snapshot(
            event.id,
            op_time,
            [
                {
                    "key": "pinnacle",
                    "outcomes": [
                        {"name": "Brighton", "price": -140},
                        {"name": "Draw", "price": 270},
                        {"name": "Wolves", "price": 350},
                    ],
                },
                *_retail_panel({"Brighton": -130, "Draw": 280, "Wolves": 360}),
            ],
        )

        # Newer BFE-only snapshot
        bfe_time = anchor - timedelta(minutes=5)
        bfe_snapshot = _make_snapshot(
            event.id,
            bfe_time,
            [
                {
                    "key": "betfair_exchange",
                    "outcomes": [
                        {"name": "Brighton", "price": -135},
                        {"name": "Draw", "price": 275},
                        {"name": "Wolves", "price": 355},
                    ],
                }
            ],
        )

        test_session.add_all([op_snapshot, bfe_snapshot])
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_book_prices(
            event.id, market="1x2", lookback_hours=2.0, now=anchor
        )

        assert isinstance(result, BookPriceResult)

        # All 17 retail books surface, each with all 3 outcomes
        retail_books_surfaced = {
            bm
            for (bm, _outcome, _point) in result.entries.keys()
            if bm not in {"pinnacle", "betfair_exchange"}
        }
        assert retail_books_surfaced == set(_RETAIL_BOOKS_17)
        assert len(retail_books_surfaced) == 17

        # bet365 has all three outcomes from the older OP snapshot
        for outcome in ("Brighton", "Draw", "Wolves"):
            entry = result.entries[("bet365", outcome, None)]
            assert entry.meta.snapshot_time == op_time

        # BFE entries come from the newer snapshot
        bfe_entry = result.entries[("betfair_exchange", "Brighton", None)]
        assert bfe_entry.meta.snapshot_time == bfe_time

    @pytest.mark.asyncio
    async def test_book_a_only_in_older_book_b_only_in_newer_both_surfaced(
        self, test_session
    ) -> None:
        """Book A only appears in the N-2 snapshot, book B only in N. Both
        surface with provenance pointing to their respective snapshots.
        """
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        anchor = datetime(2026, 4, 25, 13, 0, tzinfo=UTC)
        old_time = anchor - timedelta(minutes=40)
        new_time = anchor - timedelta(minutes=5)

        old_snapshot = _make_snapshot(
            event.id,
            old_time,
            [
                {
                    "key": "book_a",
                    "outcomes": [
                        {"name": "Brighton", "price": -150},
                        {"name": "Draw", "price": 250},
                        {"name": "Wolves", "price": 320},
                    ],
                }
            ],
        )

        new_snapshot = _make_snapshot(
            event.id,
            new_time,
            [
                {
                    "key": "book_b",
                    "outcomes": [
                        {"name": "Brighton", "price": -140},
                        {"name": "Draw", "price": 260},
                        {"name": "Wolves", "price": 340},
                    ],
                }
            ],
        )

        test_session.add_all([old_snapshot, new_snapshot])
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_book_prices(
            event.id, market="1x2", lookback_hours=2.0, now=anchor
        )

        # Both books present, with provenance to their respective snapshots
        for outcome in ("Brighton", "Draw", "Wolves"):
            assert result.entries[("book_a", outcome, None)].meta.snapshot_time == old_time
            assert result.entries[("book_b", outcome, None)].meta.snapshot_time == new_time

    @pytest.mark.asyncio
    async def test_book_in_both_snapshots_only_newest_surfaced(self, test_session) -> None:
        """When the same bookmaker appears in both N-2 and N, only the newer
        price is surfaced (one row per (book, outcome, point))."""
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        anchor = datetime(2026, 4, 25, 13, 0, tzinfo=UTC)
        old_time = anchor - timedelta(minutes=40)
        new_time = anchor - timedelta(minutes=5)

        # Same book, different prices in old vs new
        old_snapshot = _make_snapshot(
            event.id,
            old_time,
            [
                {
                    "key": "bet365",
                    "outcomes": [
                        {"name": "Brighton", "price": -150},
                        {"name": "Draw", "price": 250},
                        {"name": "Wolves", "price": 320},
                    ],
                }
            ],
        )

        new_snapshot = _make_snapshot(
            event.id,
            new_time,
            [
                {
                    "key": "bet365",
                    "outcomes": [
                        {"name": "Brighton", "price": -130},
                        {"name": "Draw", "price": 280},
                        {"name": "Wolves", "price": 360},
                    ],
                }
            ],
        )

        test_session.add_all([old_snapshot, new_snapshot])
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_book_prices(
            event.id, market="1x2", lookback_hours=2.0, now=anchor
        )

        # Only the newer price surfaces, with provenance to the newer snapshot
        brighton = result.entries[("bet365", "Brighton", None)]
        assert brighton.odds.price == -130
        assert brighton.meta.snapshot_time == new_time

        draw = result.entries[("bet365", "Draw", None)]
        assert draw.odds.price == 280
        assert draw.meta.snapshot_time == new_time

        wolves = result.entries[("bet365", "Wolves", None)]
        assert wolves.odds.price == 360
        assert wolves.meta.snapshot_time == new_time

        # Exactly 3 entries — one per outcome, no duplicates from the older snapshot
        bet365_keys = [k for k in result.entries.keys() if k[0] == "bet365"]
        assert len(bet365_keys) == 3

    @pytest.mark.asyncio
    async def test_book_absent_from_window_not_surfaced(self, test_session) -> None:
        """A snapshot outside [now - lookback_hours, now] does not contribute."""
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        anchor = datetime(2026, 4, 25, 13, 0, tzinfo=UTC)

        # In-window snapshot has bet365
        in_window_time = anchor - timedelta(minutes=30)
        in_window_snapshot = _make_snapshot(
            event.id,
            in_window_time,
            [
                {
                    "key": "bet365",
                    "outcomes": [
                        {"name": "Brighton", "price": -130},
                        {"name": "Draw", "price": 280},
                        {"name": "Wolves", "price": 360},
                    ],
                }
            ],
        )

        # Out-of-window snapshot (5 hours before anchor) has betway
        out_of_window_time = anchor - timedelta(hours=5)
        out_of_window_snapshot = _make_snapshot(
            event.id,
            out_of_window_time,
            [
                {
                    "key": "betway",
                    "outcomes": [
                        {"name": "Brighton", "price": -120},
                        {"name": "Draw", "price": 290},
                        {"name": "Wolves", "price": 380},
                    ],
                }
            ],
        )

        test_session.add_all([in_window_snapshot, out_of_window_snapshot])
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_book_prices(
            event.id, market="1x2", lookback_hours=2.0, now=anchor
        )

        # bet365 (in-window) surfaces; betway (out-of-window) does not
        books_surfaced = {bm for (bm, _outcome, _point) in result.entries.keys()}
        assert "bet365" in books_surfaced
        assert "betway" not in books_surfaced

    @pytest.mark.asyncio
    async def test_market_filter_skips_other_markets(self, test_session) -> None:
        """Snapshots not containing the requested market key are skipped."""
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        anchor = datetime(2026, 4, 25, 13, 0, tzinfo=UTC)
        snap_time = anchor - timedelta(minutes=30)

        # Only totals data — no 1x2 market
        snap = OddsSnapshot(
            event_id=event.id,
            snapshot_time=snap_time,
            raw_data=_make_raw_data(
                [
                    {
                        "key": "bet365",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 2.5},
                            {"name": "Under", "price": -110, "point": 2.5},
                        ],
                    }
                ],
                snap_time,
                market_key="totals",
            ),
            bookmaker_count=1,
            fetch_tier="pregame",
            hours_until_commence=2.0,
        )
        test_session.add(snap)
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_book_prices(
            event.id, market="1x2", lookback_hours=2.0, now=anchor
        )

        assert result.entries == {}

    @pytest.mark.asyncio
    async def test_age_seconds_anchored_on_now(self, test_session) -> None:
        """age_seconds is computed from the supplied ``now`` argument."""
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        anchor = datetime(2026, 4, 25, 13, 0, tzinfo=UTC)
        snap_time = anchor - timedelta(minutes=10)

        snap = _make_snapshot(
            event.id,
            snap_time,
            [
                {
                    "key": "bet365",
                    "outcomes": [
                        {"name": "Brighton", "price": -130},
                    ],
                }
            ],
        )
        test_session.add(snap)
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_book_prices(
            event.id, market="1x2", lookback_hours=2.0, now=anchor
        )

        entry = result.entries[("bet365", "Brighton", None)]
        # 10 minutes = 600 seconds
        assert entry.meta.age_seconds == pytest.approx(600.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_totals_keyed_per_point(self, test_session) -> None:
        """Same bookmaker on the same outcome but different points produces
        separate entries."""
        event = _make_event()
        test_session.add(event)
        await test_session.flush()

        anchor = datetime(2026, 4, 25, 13, 0, tzinfo=UTC)
        snap_time = anchor - timedelta(minutes=15)

        snap = OddsSnapshot(
            event_id=event.id,
            snapshot_time=snap_time,
            raw_data=_make_raw_data(
                [
                    {
                        "key": "bet365",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 2.5},
                            {"name": "Over", "price": +120, "point": 3.5},
                            {"name": "Under", "price": -110, "point": 2.5},
                            {"name": "Under", "price": -140, "point": 3.5},
                        ],
                    }
                ],
                snap_time,
                market_key="totals",
            ),
            bookmaker_count=1,
            fetch_tier="pregame",
            hours_until_commence=2.0,
        )
        test_session.add(snap)
        await test_session.flush()

        reader = OddsReader(test_session)
        result = await reader.get_latest_book_prices(
            event.id, market="totals", lookback_hours=2.0, now=anchor
        )

        # Distinct keys for each (bookmaker, outcome, point) tuple
        assert ("bet365", "Over", 2.5) in result.entries
        assert ("bet365", "Over", 3.5) in result.entries
        assert ("bet365", "Under", 2.5) in result.entries
        assert ("bet365", "Under", 3.5) in result.entries
        assert len(result.entries) == 4
