"""A snapshot envelope (``raw_data``) read as one snapshot per market, with its price rows.

Every writer stores the envelope it built as ``raw_data`` and the rows read from it here, so
the backfill can expand stored envelopes with the same reading.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from odds_core.bookmakers import canonical_bookmaker
from odds_core.odds_math import outcome_price_decimal
from odds_core.team import normalize_team_name
from odds_core.time import parse_api_datetime
from odds_core.vocabulary import Market, MarketStatus, PriceOutcome, Side, Source

from odds_lambda.betfair.adapter import strip_pitcher_annotation, totals_runner

DRAW_NAME = "Draw"
BETFAIR_DRAW_NAMES: frozenset[str] = frozenset({"the draw", "draw"})
# football-data.co.uk's best and average price across books, which no one offers.
FDUK_AGGREGATE_BOOKMAKERS: frozenset[str] = frozenset({"market_max", "market_avg"})
# The Odds API's Betfair lay prices, written as lay rows of the match-odds snapshot.
ODDS_API_LAY_MARKET = "h2h_lay"
# One envelope's markets in the order its snapshots are stored; the first is the row the
# envelope itself is stored on.
MARKET_ORDER: tuple[Market, ...] = (Market.ONE_X_TWO, Market.H2H, Market.SPREADS, Market.TOTALS)
# Sources whose bookmaker ``last_update`` is the book's own last change, not our fetch time.
CHANGE_STAMPED_SOURCES: frozenset[Source] = frozenset({Source.ODDS_API, Source.ODDSPORTAL})

_LINED_MARKETS: frozenset[Market] = frozenset({Market.SPREADS, Market.TOTALS})
_OUTCOME_LABELS: Mapping[str, PriceOutcome] = {o.value: o for o in PriceOutcome}


@dataclass(frozen=True)
class PriceSpec:
    bookmaker: str
    outcome: PriceOutcome
    line: Decimal | None
    side: Side
    price: Decimal
    size: Decimal | None = None
    matched: Decimal | None = None
    changed_at: datetime | None = None

    @property
    def key(self) -> tuple[str, PriceOutcome, Decimal | None, Side]:
        return (self.bookmaker, self.outcome, self.line, self.side)


@dataclass(frozen=True)
class MarketMeta:
    """An exchange market's state, which goes on its snapshot (Betfair only)."""

    in_play: bool | None = None
    market_status: MarketStatus | None = None
    total_matched: Decimal | None = None
    market_start_time: datetime | None = None


@dataclass(frozen=True)
class MarketSnapshot:
    """One market of an envelope: the snapshot's market, market metadata and prices."""

    market: Market
    prices: tuple[PriceSpec, ...]
    meta: MarketMeta = MarketMeta()


class RowIssueType(StrEnum):
    """The ``data_quality_logs.issue_type`` of something an envelope held but no row took."""

    BOOKMAKER_WITHOUT_KEY = "bookmaker_without_key"
    BOOKMAKER_ALIAS_COLLISION = "bookmaker_alias_collision"
    DUPLICATE_PRICE = "duplicate_price"
    INVALID_PRICE = "invalid_price"
    UNMATCHED_OUTCOME = "unmatched_outcome"
    UNKNOWN_MARKET = "unknown_market"
    UNKNOWN_MARKET_STATUS = "unknown_market_status"
    # A market whose snapshot key another snapshot already holds.
    SNAPSHOT_KEY_CONFLICT = "snapshot_key_conflict"


@dataclass(frozen=True)
class RowIssue:
    """Something in the envelope that became no row; written to ``data_quality_logs``."""

    issue_type: RowIssueType
    description: str
    context: dict[str, Any]


@dataclass(frozen=True)
class ExpandedEnvelope:
    """The envelope's markets in ``MARKET_ORDER``, and what was left out of them."""

    markets: tuple[MarketSnapshot, ...]
    issues: tuple[RowIssue, ...]


@dataclass
class _MarketRows:
    prices: dict[tuple[str, PriceOutcome, Decimal | None, Side], PriceSpec] = field(
        default_factory=dict
    )
    # The raw book key each canonical book was first read from, per market.
    first_keys: dict[str, str] = field(default_factory=dict)


class _Reader:
    def __init__(self, source: Source, home_team: str, away_team: str) -> None:
        self.source = source
        self.home_team = normalize_team_name(home_team)
        self.away_team = normalize_team_name(away_team)
        self.markets: dict[Market, _MarketRows] = {}
        self.meta: dict[Market, MarketMeta] = {}
        self.issues: list[RowIssue] = []

    def log_issue(self, issue_type: RowIssueType, description: str, **context: Any) -> None:
        self.issues.append(
            RowIssue(issue_type, description, {"source": self.source.value, **context})
        )

    def outcome_of(self, name: str, market: Market) -> PriceOutcome | None:
        if market is Market.TOTALS:
            return {"over": PriceOutcome.OVER, "under": PriceOutcome.UNDER}.get(name.lower())
        if market is Market.ONE_X_TWO and name == DRAW_NAME:
            return PriceOutcome.DRAW
        team = normalize_team_name(name)
        if team == self.home_team:
            return PriceOutcome.HOME
        if team == self.away_team:
            return PriceOutcome.AWAY
        return None

    def claim_book(self, market: Market, raw_key: str) -> str | None:
        """The canonical key a book's ``market`` rows go under, or ``None`` when an earlier
        entry already gave that market's rows for the book (the first entry wins)."""
        rows = self.markets.setdefault(market, _MarketRows())
        book = canonical_bookmaker(self.source, raw_key)
        first = rows.first_keys.get(book)
        if first is None:
            rows.first_keys[book] = raw_key
            return book
        self.log_issue(
            RowIssueType.BOOKMAKER_ALIAS_COLLISION,
            f"{self.source} {market} book {raw_key!r} is {book!r}, already read from {first!r}; "
            "the first entry is kept",
            market=market.value,
            bookmaker=book,
            kept_key=first,
            dropped_key=raw_key,
        )
        return None

    def add(self, market: Market, spec: PriceSpec) -> None:
        rows = self.markets.setdefault(market, _MarketRows())
        if spec.key in rows.prices:
            self.log_issue(
                RowIssueType.DUPLICATE_PRICE,
                f"{self.source} {market} {spec.bookmaker} {spec.outcome} {spec.side} is quoted "
                "twice; the first is kept",
                market=market.value,
                bookmaker=spec.bookmaker,
                outcome=spec.outcome.value,
                side=spec.side.value,
                line=None if spec.line is None else str(spec.line),
            )
            return
        rows.prices[spec.key] = spec

    def decimal_price(
        self, value: Any, market: Market, bookmaker: str, outcome: str, side: Side
    ) -> Decimal | None:
        """``value`` as a decimal price above 1.0; anything else is logged and dropped."""
        number = _finite(value)
        if number is not None and number > 1.0:
            return _decimal(number)
        self.log_issue(
            RowIssueType.INVALID_PRICE,
            f"{self.source} {market} {bookmaker} {outcome} {side} price {value!r} is not above 1.0",
            market=market.value,
            bookmaker=bookmaker,
            outcome=outcome,
            side=side.value,
            price=repr(value),
        )
        return None

    def probability_price(
        self, value: Any, market: Market, bookmaker: str, outcome: str, side: Side
    ) -> Decimal | None:
        """Decimal odds ``1/p`` of a probability strictly between 0 and 1."""
        number = _finite(value)
        if number is not None and 0.0 < number < 1.0:
            return _decimal(1 / number)
        self.log_issue(
            RowIssueType.INVALID_PRICE,
            f"{self.source} {market} {bookmaker} {outcome} {side} probability {value!r} is not "
            "between 0 and 1",
            market=market.value,
            bookmaker=bookmaker,
            outcome=outcome,
            side=side.value,
            probability=repr(value),
        )
        return None

    def log_unmatched(self, market: Market, bookmaker: str, name: object) -> None:
        self.log_issue(
            RowIssueType.UNMATCHED_OUTCOME,
            f"{self.source} {market} {bookmaker} outcome {name!r} is neither "
            f"{self.home_team!r}, {self.away_team!r} nor a draw, over or under",
            market=market.value,
            bookmaker=bookmaker,
            outcome_name=repr(name),
            home_team=self.home_team,
            away_team=self.away_team,
        )


def expand_envelope(
    raw_data: Mapping[str, Any], source: Source, home_team: str, away_team: str
) -> ExpandedEnvelope:
    """Read an envelope ``source`` wrote for the event ``home_team`` v ``away_team``.

    Outcomes are matched once, here: team names (normalized) to home and away, ``Draw`` on
    ``1x2``, ``Over``/``Under`` on ``totals``. Book keys go through the source's alias map
    (``odds_core.bookmakers.BOOKMAKER_ALIASES_BY_SOURCE``); when two entries name one book, the
    first is kept. Unmatched outcomes, prices not above 1.0, alias collisions and unknown markets
    become ``issues`` rather than rows.
    """
    reader = _Reader(source, home_team, away_team)
    for entry in raw_data.get("bookmakers") or []:
        raw_key = str(entry.get("key") or "").strip()
        if not raw_key:
            reader.log_issue(
                RowIssueType.BOOKMAKER_WITHOUT_KEY, f"{source} entry has no bookmaker key"
            )
            continue
        if source is Source.FOOTBALL_DATA_UK and raw_key in FDUK_AGGREGATE_BOOKMAKERS:
            continue
        if source is Source.BETFAIR:
            _read_betfair(reader, entry, raw_key)
        elif source is Source.POLYMARKET:
            _read_polymarket(reader, entry, raw_key)
        else:
            _read_retail(reader, entry, raw_key)
    markets = tuple(
        MarketSnapshot(
            market=market,
            prices=tuple(reader.markets[market].prices.values()),
            meta=reader.meta.get(market, MarketMeta()),
        )
        for market in MARKET_ORDER
        if market in reader.markets
    )
    return ExpandedEnvelope(markets, tuple(reader.issues))


def stored_market(raw_market: Mapping[str, Any]) -> tuple[Market, Side] | None:
    """The market and side an envelope market's prices are stored under, ``None`` for a market
    no snapshot takes. ``h2h_lay`` is the lay side of the match odds: 1x2 when it quotes a draw,
    else h2h."""
    key = raw_market.get("key")
    if key == ODDS_API_LAY_MARKET:
        draw = any(o.get("name") == DRAW_NAME for o in raw_market.get("outcomes") or [])
        return (Market.ONE_X_TWO if draw else Market.H2H), Side.LAY
    try:
        return Market(key), Side.BACK
    except ValueError:
        return None


def _market_of(reader: _Reader, raw_market: Mapping[str, Any]) -> tuple[Market, Side] | None:
    read = stored_market(raw_market)
    if read is None:
        key = raw_market.get("key")
        reader.log_issue(
            RowIssueType.UNKNOWN_MARKET, f"{reader.source} market {key!r} is not stored", market=key
        )
    return read


def _read_retail(reader: _Reader, entry: Mapping[str, Any], raw_key: str) -> None:
    matched = entry.get("betfair_matched") or {}
    entry_changed = _changed_at(reader.source, entry.get("last_update"))
    claimed: dict[Market, str | None] = {}
    for raw_market in entry.get("markets") or []:
        read = _market_of(reader, raw_market)
        if read is None:
            continue
        market, side = read
        if market not in claimed:
            claimed[market] = reader.claim_book(market, raw_key)
        book = claimed[market]
        if book is None:
            continue
        changed = _changed_at(reader.source, raw_market.get("last_update")) or entry_changed
        for raw_outcome in raw_market.get("outcomes") or []:
            name = str(raw_outcome.get("name") or "")
            outcome = reader.outcome_of(name, market)
            if outcome is None:
                reader.log_unmatched(market, book, raw_outcome.get("name"))
                continue
            try:
                value: Any = outcome_price_decimal(raw_outcome)
            except (KeyError, TypeError, ValueError):
                value = None
            price = reader.decimal_price(value, market, book, outcome.value, side)
            if price is None:
                continue
            reader.add(
                market,
                PriceSpec(
                    bookmaker=book,
                    outcome=outcome,
                    line=_line(market, raw_outcome.get("point")),
                    side=side,
                    price=price,
                    matched=_decimal(_finite(matched.get(outcome.value))) if matched else None,
                    changed_at=changed,
                ),
            )


def _read_betfair(reader: _Reader, entry: Mapping[str, Any], raw_key: str) -> None:
    meta = entry.get("betfair_meta") or {}
    for raw_market in entry.get("markets") or []:
        read = _market_of(reader, raw_market)
        if read is None:
            continue
        market = read[0]
        book = reader.claim_book(market, raw_key)
        if book is None:
            continue
        reader.meta[market] = MarketMeta(
            in_play=meta.get("inplay"),
            market_status=_market_status(reader, meta.get("market_status")),
            total_matched=_decimal(_finite(meta.get("total_matched"))),
            market_start_time=_datetime(meta.get("market_start_time")),
        )
        for runner in meta.get("runners") or []:
            name = str(runner.get("runner_name") or "")
            outcome, line = _betfair_outcome(reader, name, market)
            if outcome is None:
                reader.log_unmatched(market, book, runner.get("runner_name"))
                continue
            for side, price_field, size_field in (
                (Side.BACK, "best_back", "back_size"),
                (Side.LAY, "best_lay", "lay_size"),
                (Side.TRADED, "last_price_traded", None),
            ):
                value = runner.get(price_field)
                if value is None:
                    continue
                price = reader.decimal_price(value, market, book, outcome.value, side)
                if price is None:
                    continue
                size = _decimal(_finite(runner.get(size_field))) if size_field else None
                reader.add(market, PriceSpec(book, outcome, line, side, price, size=size))


def _betfair_outcome(
    reader: _Reader, runner_name: str, market: Market
) -> tuple[PriceOutcome | None, Decimal | None]:
    """A Betfair runner's outcome, and its line on ``totals`` ("Over 2.5 Goals")."""
    if market is Market.TOTALS:
        read = totals_runner(runner_name)
        return (None, None) if read is None else read
    if market is Market.ONE_X_TWO and runner_name.strip().lower() in BETFAIR_DRAW_NAMES:
        return PriceOutcome.DRAW, None
    return reader.outcome_of(strip_pitcher_annotation(runner_name), market), None


def _read_polymarket(reader: _Reader, entry: Mapping[str, Any], raw_key: str) -> None:
    meta = entry.get("polymarket_meta") or {}
    for raw_market in entry.get("markets") or []:
        read = _market_of(reader, raw_market)
        if read is None:
            continue
        market = read[0]
        book = reader.claim_book(market, raw_key)
        if book is None:
            continue
        for side, probabilities in (
            (Side.MID, meta.get("yes_probabilities")),
            (Side.BACK, meta.get("best_ask")),
            (Side.LAY, meta.get("best_bid")),
        ):
            for label, probability in (probabilities or {}).items():
                outcome = _OUTCOME_LABELS.get(label)
                if outcome is None:
                    reader.log_unmatched(market, book, label)
                    continue
                price = reader.probability_price(probability, market, book, label, side)
                if price is not None:
                    reader.add(market, PriceSpec(book, outcome, None, side, price))


def _market_status(reader: _Reader, value: object) -> MarketStatus | None:
    if value is None:
        return None
    try:
        return MarketStatus(str(value).lower())
    except ValueError:
        reader.log_issue(
            RowIssueType.UNKNOWN_MARKET_STATUS,
            f"{reader.source} market status {value!r} is not stored",
            market_status=repr(value),
        )
        return None


def _changed_at(source: Source, value: object) -> datetime | None:
    if source not in CHANGE_STAMPED_SOURCES:
        return None
    return _datetime(value)


def _datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return parse_api_datetime(value)
    except ValueError:
        return None


def _line(market: Market, point: object) -> Decimal | None:
    return _decimal(_finite(point)) if market in _LINED_MARKETS else None


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _decimal(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def market_envelope(raw_data: Mapping[str, Any], market: Market) -> dict[str, Any]:
    """``raw_data`` with each book's markets cut to those stored under ``market``
    (``stored_market``), dropping books left without one."""
    books = []
    for entry in raw_data.get("bookmakers") or []:
        kept = [
            m
            for m in entry.get("markets") or []
            if (read := stored_market(m)) is not None and read[0] is market
        ]
        if kept:
            books.append({**entry, "markets": kept})
    return {**raw_data, "bookmakers": books}
