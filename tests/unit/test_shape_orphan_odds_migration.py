"""Tests for the a6f3c1e9d2b4_shape_orphan_odds_rows migration's envelope rebuild."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from odds_core.vocabulary import Market, PriceOutcome, Side, Source
from odds_lambda.price_rows import expand_envelope

TAKEN = datetime(2026, 3, 24, 13, 23, 6, tzinfo=UTC)


def _load_migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "a6f3c1e9d2b4_shape_orphan_odds_rows.py"
    )
    spec = importlib.util.spec_from_file_location("a6f3c1e9d2b4_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    return _load_migration()


def _row(book: str, market: str, name: str, price: float, point: float | None = None) -> Any:
    return SimpleNamespace(
        bookmaker_key=book,
        bookmaker_title=book.title(),
        market_key=market,
        outcome_name=name,
        price_decimal=price,
        point=point,
        last_update=TAKEN,
    )


class TestEnvelope:
    def test_rows_read_back_as_one_price_row_each(self, migration: ModuleType) -> None:
        rows = [
            _row("bet365", "1x2", "Brentford", 2.2),
            _row("bet365", "1x2", "Draw", 3.3),
            _row("bet365", "1x2", "Everton", 3.2),
            _row("betfair", "1x2", "Brentford", 2.25),
        ]

        expanded = expand_envelope(
            migration._envelope(rows), Source.ODDSPORTAL, "Brentford", "Everton"
        )

        assert expanded.issues == ()
        [market] = expanded.markets
        assert market.market is Market.ONE_X_TWO
        assert {(p.bookmaker, p.outcome, p.side, p.price) for p in market.prices} == {
            ("bet365", PriceOutcome.HOME, Side.BACK, Decimal("2.2")),
            ("bet365", PriceOutcome.DRAW, Side.BACK, Decimal("3.3")),
            ("bet365", PriceOutcome.AWAY, Side.BACK, Decimal("3.2")),
            ("betfair_sportsbook", PriceOutcome.HOME, Side.BACK, Decimal("2.25")),
        }
        assert {p.changed_at for p in market.prices} == {TAKEN}

    def test_totals_keep_their_line(self, migration: ModuleType) -> None:
        rows = [
            _row("unibet", "totals", "Over", 1.9, 2.5),
            _row("unibet", "totals", "Under", 1.9, 2.5),
        ]

        [market] = expand_envelope(
            migration._envelope(rows), Source.ODDSPORTAL, "Brentford", "Everton"
        ).markets

        assert {(p.outcome, p.line) for p in market.prices} == {
            (PriceOutcome.OVER, Decimal("2.5")),
            (PriceOutcome.UNDER, Decimal("2.5")),
        }

    def test_the_envelope_is_marked_as_rebuilt(self, migration: ModuleType) -> None:
        envelope = migration._envelope([_row("bet365", "1x2", "Draw", 3.3)])

        assert envelope[migration.REBUILT_MARKER] == "odds"
        assert envelope["source"] == "oddsportal_live"
