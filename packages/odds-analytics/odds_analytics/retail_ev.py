"""Decision rows for the retail-EV target of ADR-0012.

One row per event, outcome, UK retail book and retail snapshot time t. The target is
the bet's EV against the devigged Betfair Exchange close; the baseline treats the
exchange price at or before t as the close.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS, canonical_bookmaker_key
from odds_core.models import Event, OddsSnapshot
from odds_core.odds_math import devig_probabilities, outcome_price_decimal
from odds_core.snapshot_sources import SnapshotKind, snapshot_kind

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from odds_analytics.feature_groups import PreparedFeatureData
    from odds_analytics.training.config import FeatureConfig

logger = structlog.get_logger()

DECISION_WINDOW = timedelta(days=7)
DECISION_CUTOFF = timedelta(minutes=5)
LIVE_CLOSE_WINDOW = timedelta(minutes=30)
MAX_EXCHANGE_AGE = timedelta(hours=2)
# A wider back book is a thin market whose best backs are not a price (ADR-0014).
MAX_EXCHANGE_OVERROUND = 0.05
EXCHANGE_BOOKMAKER = "betfair_exchange"
DRAW_OUTCOME_NAME = "Draw"
THREE_WAY_MARKET = "1x2"
# Lower edges of the reporting horizon buckets, in hours to kick-off.
HORIZON_BUCKET_EDGES_HOURS: tuple[float, ...] = (0.0, 1.0, 6.0, 24.0, 72.0)
HORIZON_BUCKET_LABELS: tuple[str, ...] = ("0-1h", "1-6h", "6-24h", "1-3d", "3-7d")
SEASON_START_MONTH = 7
TWO_WAY_MARKET = "h2h"

RETAIL_KINDS: frozenset[SnapshotKind] = frozenset(
    {
        SnapshotKind.ODDS_API,
        SnapshotKind.ODDSPORTAL,
        SnapshotKind.ODDSPORTAL_LIVE,
        SnapshotKind.FDUK_OPENING,
    }
)
EXCHANGE_AT_T_KINDS: frozenset[SnapshotKind] = frozenset(
    {SnapshotKind.BETFAIR_LIVE, SnapshotKind.FDUK_OPENING}
)


class Outcome(StrEnum):
    HOME = "home"
    DRAW = "draw"
    AWAY = "away"


def market_outcomes(market: str) -> tuple[Outcome, ...]:
    if market == THREE_WAY_MARKET:
        return (Outcome.HOME, Outcome.DRAW, Outcome.AWAY)
    if market == TWO_WAY_MARKET:
        return (Outcome.HOME, Outcome.AWAY)
    raise ValueError(f"retail_ev has no outcomes for market {market!r}")


@dataclass(frozen=True)
class ExchangePrice:
    """Devigged Betfair Exchange back prices from one snapshot."""

    time: datetime
    kind: SnapshotKind
    fair: dict[Outcome, float]
    overround: float


@dataclass(frozen=True)
class RetailEvRow:
    event_id: str
    commence_time: datetime
    outcome: Outcome
    book: str
    retail_kind: SnapshotKind
    decision_time: datetime
    retail_decimal: float
    book_fair_prob: float | None
    book_overround: float | None
    consensus_prob: float
    num_bookmakers: int
    exchange: ExchangePrice
    close: ExchangePrice

    @property
    def hours_to_kickoff(self) -> float:
        return (self.commence_time - self.decision_time).total_seconds() / 3600

    @property
    def horizon_bucket(self) -> str:
        hours = self.hours_to_kickoff
        if not 0 <= hours <= DECISION_WINDOW.total_seconds() / 3600:
            raise ValueError(f"hours to kick-off {hours} outside the decision window")
        idx = bisect.bisect_right(HORIZON_BUCKET_EDGES_HOURS, hours) - 1
        return HORIZON_BUCKET_LABELS[idx]

    @property
    def season(self) -> str:
        year = self.commence_time.year
        start = year if self.commence_time.month >= SEASON_START_MONTH else year - 1
        return f"{start}-{(start + 1) % 100:02d}"

    @property
    def exchange_age_hours(self) -> float:
        return (self.decision_time - self.exchange.time).total_seconds() / 3600

    @property
    def target(self) -> float:
        return self.close.fair[self.outcome] * self.retail_decimal - 1

    @property
    def baseline(self) -> float:
        return self.exchange.fair[self.outcome] * self.retail_decimal - 1


@dataclass
class DropCounts:
    events_without_close: int = 0
    rows_without_exchange: int = 0
    rows_stale_exchange: int = 0
    wide_exchange_snapshots: int = 0
    prices_outside_roster: Counter[str] = field(default_factory=Counter)

    def add(self, other: DropCounts) -> None:
        self.events_without_close += other.events_without_close
        self.rows_without_exchange += other.rows_without_exchange
        self.rows_stale_exchange += other.rows_stale_exchange
        self.wide_exchange_snapshots += other.wide_exchange_snapshots
        self.prices_outside_roster.update(other.prices_outside_roster)


@dataclass
class RetailEvDataset:
    rows: list[RetailEvRow] = field(default_factory=list)
    drops: DropCounts = field(default_factory=DropCounts)

    @property
    def event_ids(self) -> np.ndarray:
        return np.array([r.event_id for r in self.rows])

    @property
    def targets(self) -> np.ndarray:
        return np.array([r.target for r in self.rows], dtype=np.float64)

    @property
    def baselines(self) -> np.ndarray:
        return np.array([r.baseline for r in self.rows], dtype=np.float64)

    @property
    def commence_times(self) -> np.ndarray:
        return np.array(
            [r.commence_time.astimezone(UTC).replace(tzinfo=None) for r in self.rows],
            dtype="datetime64[us]",
        )

    @property
    def event_weights(self) -> np.ndarray:
        return event_weights(self.event_ids)

    @property
    def loss_weights(self) -> np.ndarray:
        odds = np.array([r.retail_decimal for r in self.rows], dtype=np.float64)
        return self.event_weights / odds**2


def event_weights(event_ids: np.ndarray) -> np.ndarray:
    """``w_event = 1 / (rows in the event)`` for each row."""
    _, inverse, counts = np.unique(event_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _outcome_names(event: Event, market: str) -> dict[str, Outcome]:
    names = {event.home_team: Outcome.HOME, event.away_team: Outcome.AWAY}
    if Outcome.DRAW in market_outcomes(market):
        names[DRAW_OUTCOME_NAME] = Outcome.DRAW
    return names


def _book_prices(
    raw_data: dict[str, Any], kind: SnapshotKind, market: str, names: dict[str, Outcome]
) -> dict[str, dict[Outcome, float]]:
    """Valid decimal prices per canonical book key; the first entry wins for an aliased book."""
    books: dict[str, dict[Outcome, float]] = {}
    for bookmaker in raw_data.get("bookmakers", []):
        key = canonical_bookmaker_key(bookmaker.get("key", ""), kind)
        if key in books:
            continue
        for mkt in bookmaker.get("markets", []):
            if mkt.get("key") != market:
                continue
            prices: dict[Outcome, float] = {}
            for outcome in mkt.get("outcomes", []):
                side = names.get(outcome.get("name"))
                if side is None:
                    continue
                price = outcome_price_decimal(outcome)
                if price > 1.0:
                    prices[side] = price
            if prices:
                books[key] = prices
            break
    return books


def _devig(prices: dict[Outcome, float], outcomes: tuple[Outcome, ...]) -> dict[Outcome, float]:
    fair = devig_probabilities(*(1 / prices[o] for o in outcomes))
    return dict(zip(outcomes, fair, strict=True))


def _overround(prices: dict[Outcome, float], outcomes: tuple[Outcome, ...]) -> float:
    return sum(1 / prices[o] for o in outcomes) - 1


def _is_in_play(raw_data: dict[str, Any]) -> bool:
    return any(
        bool((bm.get("betfair_meta") or {}).get("inplay"))
        for bm in raw_data.get("bookmakers", [])
        if canonical_bookmaker_key(bm.get("key", ""), SnapshotKind.BETFAIR_LIVE)
        == EXCHANGE_BOOKMAKER
    )


def _exchange_price(
    snapshot: OddsSnapshot,
    kind: SnapshotKind,
    market: str,
    names: dict[str, Outcome],
    outcomes: tuple[Outcome, ...],
) -> ExchangePrice | None:
    prices = _book_prices(snapshot.raw_data, kind, market, names).get(EXCHANGE_BOOKMAKER)
    if prices is None or set(prices) != set(outcomes):
        return None
    return ExchangePrice(
        time=snapshot.snapshot_time,
        kind=kind,
        fair=_devig(prices, outcomes),
        overround=_overround(prices, outcomes),
    )


def build_event_rows(
    event: Event,
    snapshots: Sequence[OddsSnapshot],
    market: str,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
) -> tuple[list[RetailEvRow], DropCounts]:
    """Build every decision row for one event from its snapshots, in time order.

    The close is the last live Betfair snapshot that is not in-play within
    ``LIVE_CLOSE_WINDOW`` before kick-off, else the FDUK closing price (stamped at
    kick-off). The exchange price at t comes only from live Betfair or FDUK opening
    snapshots at or before t, so nothing after t enters a row's inputs. Exchange
    prices whose back overround exceeds ``max_exchange_overround`` are not prices.
    """
    drops = DropCounts()
    outcomes = market_outcomes(market)
    names = _outcome_names(event, market)
    kickoff = event.commence_time
    ordered = sorted(snapshots, key=lambda s: s.snapshot_time)
    kinds = [snapshot_kind(s.raw_data) for s in ordered]

    exchange_at_t: list[ExchangePrice] = []
    live_closes: list[ExchangePrice] = []
    fduk_close: ExchangePrice | None = None
    for snapshot, kind in zip(ordered, kinds, strict=True):
        if kind not in EXCHANGE_AT_T_KINDS and kind is not SnapshotKind.FDUK_CLOSING:
            continue
        if kind is SnapshotKind.BETFAIR_LIVE and _is_in_play(snapshot.raw_data):
            continue
        price = _exchange_price(snapshot, kind, market, names, outcomes)
        if price is None:
            continue
        if price.overround > max_exchange_overround:
            drops.wide_exchange_snapshots += 1
            continue
        if kind is SnapshotKind.FDUK_CLOSING:
            fduk_close = price
            continue
        if price.time < kickoff:
            exchange_at_t.append(price)
        if (
            kind is SnapshotKind.BETFAIR_LIVE
            and kickoff - LIVE_CLOSE_WINDOW <= price.time <= kickoff
        ):
            live_closes.append(price)

    close = live_closes[-1] if live_closes else fduk_close
    if close is None:
        drops.events_without_close = 1
        return [], drops

    exchange_times = [p.time for p in exchange_at_t]
    window_start = kickoff - DECISION_WINDOW
    window_end = kickoff - DECISION_CUTOFF
    rows: list[RetailEvRow] = []
    for snapshot, kind in zip(ordered, kinds, strict=True):
        t = snapshot.snapshot_time
        if kind not in RETAIL_KINDS or not window_start <= t <= window_end:
            continue
        retail: dict[str, dict[Outcome, float]] = {}
        for book, prices in _book_prices(snapshot.raw_data, kind, market, names).items():
            if book in UK_RETAIL_BOOKMAKERS:
                retail[book] = prices
            else:
                drops.prices_outside_roster[book] += len(prices)
        n_candidates = sum(len(p) for p in retail.values())
        if n_candidates == 0:
            continue
        idx = bisect.bisect_right(exchange_times, t) - 1
        if idx < 0:
            drops.rows_without_exchange += n_candidates
            continue
        exchange = exchange_at_t[idx]
        if t - exchange.time > MAX_EXCHANGE_AGE:
            drops.rows_stale_exchange += n_candidates
            continue

        implied_by_outcome: dict[Outcome, list[float]] = {o: [] for o in outcomes}
        for prices in retail.values():
            for outcome, price in prices.items():
                implied_by_outcome[outcome].append(1 / price)

        for book, prices in retail.items():
            complete = set(prices) == set(outcomes)
            book_fair = _devig(prices, outcomes) if complete else None
            book_overround = _overround(prices, outcomes) if complete else None
            for outcome, price in prices.items():
                implied = implied_by_outcome[outcome]
                rows.append(
                    RetailEvRow(
                        event_id=event.id,
                        commence_time=kickoff,
                        outcome=outcome,
                        book=book,
                        retail_kind=kind,
                        decision_time=t,
                        retail_decimal=price,
                        book_fair_prob=book_fair[outcome] if book_fair is not None else None,
                        book_overround=book_overround,
                        consensus_prob=float(np.mean(implied)),
                        num_bookmakers=len(implied),
                        exchange=exchange,
                        close=close,
                    )
                )
    return rows, drops


RETAIL_BOOK_COLUMNS: tuple[str, ...] = tuple(sorted(UK_RETAIL_BOOKMAKERS))

_PRICE_FEATURE_NAMES: tuple[str, ...] = (
    "hours_to_kickoff",
    "retail_implied_prob",
    "book_fair_prob",
    "book_overround",
    "consensus_prob",
    "retail_consensus_diff",
    "num_bookmakers",
    "sharp_prob",
    "retail_sharp_diff",
    "exchange_overround",
    "exchange_age_hours",
    "baseline_ev",
    "is_weekend",
    "day_of_week",
)


def retail_ev_feature_names(outcomes: tuple[Outcome, ...]) -> list[str]:
    return (
        list(_PRICE_FEATURE_NAMES)
        + [f"outcome_{o.value}" for o in outcomes]
        + [f"book_{b}" for b in RETAIL_BOOK_COLUMNS]
    )


def _nan_if_none(value: float | None) -> float:
    return np.nan if value is None else value


def retail_ev_features(row: RetailEvRow, outcomes: tuple[Outcome, ...]) -> np.ndarray:
    implied = 1 / row.retail_decimal
    sharp = row.exchange.fair[row.outcome]
    weekday = row.commence_time.weekday()
    price_block = [
        row.hours_to_kickoff,
        implied,
        _nan_if_none(row.book_fair_prob),
        _nan_if_none(row.book_overround),
        row.consensus_prob,
        implied - row.consensus_prob,
        float(row.num_bookmakers),
        sharp,
        implied - sharp,
        row.exchange.overround,
        row.exchange_age_hours,
        row.baseline,
        float(weekday >= 5),
        float(weekday),
    ]
    outcome_block = [float(row.outcome is o) for o in outcomes]
    book_block = [float(row.book == b) for b in RETAIL_BOOK_COLUMNS]
    return np.array(price_block + outcome_block + book_block, dtype=np.float64)


async def load_retail_ev_dataset(
    events: Sequence[Event],
    session: AsyncSession,
    market: str,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
) -> RetailEvDataset:
    """Decision rows for every event, in the events' order, from all its snapshots."""
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)
    dataset = RetailEvDataset()
    for event in events:
        snapshots = await reader.get_snapshots_for_event(event.id)
        rows, drops = build_event_rows(event, snapshots, market, max_exchange_overround)
        dataset.rows.extend(rows)
        dataset.drops.add(drops)
    logger.info(
        "retail_ev_rows_built",
        events=len(events),
        rows=len(dataset.rows),
        events_without_close=dataset.drops.events_without_close,
        rows_without_exchange=dataset.drops.rows_without_exchange,
        rows_stale_exchange=dataset.drops.rows_stale_exchange,
        wide_exchange_snapshots=dataset.drops.wide_exchange_snapshots,
        prices_outside_roster=dict(dataset.drops.prices_outside_roster.most_common()),
    )
    return dataset


async def prepare_retail_ev_training_data(
    events: list[Event], session: AsyncSession, config: FeatureConfig
) -> PreparedFeatureData:
    """Feature matrix, targets, loss weights and kick-off times for ``retail_ev``."""
    from odds_analytics.feature_groups import PreparedFeatureData, filter_completed_events
    from odds_analytics.training.feature_selection import apply_variance_filter

    valid_events = filter_completed_events(events)
    if not valid_events:
        raise ValueError(f"No valid events found in {len(events)} total events")

    market = config.primary_market
    outcomes = market_outcomes(market)
    dataset = await load_retail_ev_dataset(valid_events, session, market)
    if not dataset.rows:
        raise ValueError(f"No retail_ev rows after processing {len(valid_events)} events")

    X = np.array([retail_ev_features(r, outcomes) for r in dataset.rows], dtype=np.float32)
    X, feature_names, _ = apply_variance_filter(
        X, retail_ev_feature_names(outcomes), config.variance_threshold
    )
    y = dataset.targets.astype(np.float32)
    # Mean-one scaling keeps min_child_weight and reg_lambda on their unweighted scale.
    weights = dataset.loss_weights / dataset.loss_weights.mean()

    logger.info(
        "prepared_retail_ev_training_data",
        num_samples=len(y),
        num_events=len(set(dataset.event_ids.tolist())),
        num_features=len(feature_names),
        target_mean=float(np.mean(y)),
        target_std=float(np.std(y)),
    )

    return PreparedFeatureData(
        X=X,
        y=y,
        feature_names=feature_names,
        event_ids=dataset.event_ids,
        sample_weights=weights,
        commence_times=dataset.commence_times,
    )
