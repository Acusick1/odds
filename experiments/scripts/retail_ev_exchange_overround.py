"""Evidence for ignoring wide Betfair Exchange books in the retail-EV target (ADR-0014).

Builds the ADR-0012 decision rows for finished EPL events with no overround cap, then
prints the gap between the exchange price at t and the close by the exchange
snapshot's back overround, and compares three policies for wide books: no cap, drop
the row, and fall back to the latest narrow price (the builder's default).

Usage:
    uv run python experiments/scripts/retail_ev_exchange_overround.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import structlog
from odds_analytics.retail_ev import (
    MAX_EXCHANGE_OVERROUND,
    RetailEvDataset,
    load_retail_ev_dataset,
)
from odds_analytics.retail_ev_metrics import weighted_mse
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

SPORT_KEY = "soccer_epl"
MARKET = "1x2"
BANDS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.10, math.inf)
HORIZONS: tuple[float, ...] = (0.0, 1.0, 6.0, 24.0, 72.0, 168.0 + 1e-9)
HORIZON_LABELS: tuple[str, ...] = ("0-1h", "1-6h", "6-24h", "1-3d", "3-7d")


def gaps(dataset: RetailEvDataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": dataset.event_ids,
            "exchange_time": [r.exchange.time for r in dataset.rows],
            "overround": [r.exchange.overround for r in dataset.rows],
            "horizon": pd.cut(
                [r.hours_to_kickoff for r in dataset.rows],
                list(HORIZONS),
                labels=list(HORIZON_LABELS),
                right=False,
            ),
            "gap": [r.close.fair[r.outcome] - r.exchange.fair[r.outcome] for r in dataset.rows],
        }
    )


def band_table(df: pd.DataFrame) -> str:
    df = df.assign(band=pd.cut(df["overround"], list(BANDS), right=True, include_lowest=True))
    lines = [
        "| horizon | back overround | rows | exchange snapshots | gap std (pp) |",
        "|---|---|---|---|---|",
    ]
    for horizon in (*HORIZON_LABELS, "all"):
        part = df if horizon == "all" else df[df["horizon"] == horizon]
        for band, group in part.groupby("band", observed=True):
            snapshots = group[["event_id", "exchange_time"]].drop_duplicates()
            lines.append(
                f"| {horizon} | {band} | {len(group):,} | {len(snapshots):,} "
                f"| {group['gap'].std() * 100:.2f} |"
            )
    return "\n".join(lines)


def policy_row(name: str, dataset: RetailEvDataset, keep: np.ndarray | None = None) -> str:
    y, base, ids = dataset.targets, dataset.baselines, dataset.event_ids
    odds = np.array([r.retail_decimal for r in dataset.rows])
    if keep is not None:
        y, base, ids, odds = y[keep], base[keep], ids[keep], odds[keep]
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse] / odds**2
    return f"| {name} | {len(y):,} | {len(counts):,} | {weighted_mse(y, base, w):.5f} |"


async def main(start: date, end: date) -> None:
    async with async_session_maker() as session:
        events = await filter_events_by_date_range(
            session,
            start_date=datetime.combine(start, datetime.min.time(), tzinfo=UTC),
            end_date=datetime.combine(end, datetime.max.time(), tzinfo=UTC),
            status=None,
            sport_key=SPORT_KEY,
        )
        uncapped = await load_retail_ev_dataset(events, session, MARKET, math.inf)
        capped = await load_retail_ev_dataset(events, session, MARKET)

    df = gaps(uncapped)
    print(f"Finished {SPORT_KEY} events {start} to {end}: {len(events)}\n")
    print("### Exchange price at t minus close, by horizon and back overround\n")
    print(band_table(df))
    narrow = np.array([r.exchange.overround <= MAX_EXCHANGE_OVERROUND for r in uncapped.rows])
    print(f"\n### Policies for books over {MAX_EXCHANGE_OVERROUND:.0%}\n")
    print("| policy | rows | events | baseline wMSE |")
    print("|---|---|---|---|")
    print(policy_row("no cap", uncapped))
    print(policy_row("drop the row", uncapped, narrow))
    print(policy_row("fall back to the latest narrow price", capped))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 8, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date())
    args = parser.parse_args()
    asyncio.run(main(args.start, args.end))
