"""Row counts and baseline error of the ADR-0012 retail-EV target (#522).

Builds the decision rows for finished EPL events and prints Markdown tables of row
counts by season, retail source, close source, horizon bucket and book, with the
baseline's weighted MSE and the target's weighted standard deviation per group. No
model is trained.

Usage:
    uv run python experiments/scripts/retail_ev_target_report.py
    uv run python experiments/scripts/retail_ev_target_report.py --start 2025-08-01
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import structlog
from odds_analytics.retail_ev import (
    MAX_EXCHANGE_OVERROUND,
    RetailEvDataset,
    RetailEvRow,
    load_retail_ev_dataset,
)
from odds_analytics.retail_ev_metrics import weighted_mse
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker
from odds_core.models import EventStatus

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

SPORT_KEY = "soccer_epl"
MARKET = "1x2"
HORIZON_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 1.0, "0-1h"),
    (1.0, 6.0, "1-6h"),
    (6.0, 24.0, "6-24h"),
    (24.0, 72.0, "1-3d"),
    (72.0, 168.0 + 1e-9, "3-7d"),
)


def season_of(row: RetailEvRow) -> str:
    start = row.commence_time.year if row.commence_time.month >= 7 else row.commence_time.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def horizon_bucket(row: RetailEvRow) -> str:
    hours = row.hours_to_kickoff
    for low, high, label in HORIZON_BUCKETS:
        if low <= hours < high:
            return label
    raise ValueError(f"hours to kick-off {hours} outside the decision window")


def frame(dataset: RetailEvDataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": dataset.event_ids,
            "season": [season_of(r) for r in dataset.rows],
            "retail_source": [r.retail_kind.value for r in dataset.rows],
            "close_source": [r.close.kind.value for r in dataset.rows],
            "horizon": [horizon_bucket(r) for r in dataset.rows],
            "book": [r.book for r in dataset.rows],
            "target": dataset.targets,
            "baseline": dataset.baselines,
            "w_event": dataset.event_weights,
            "w_loss": dataset.loss_weights,
        }
    )


def summarise(df: pd.DataFrame, key: str, order: Callable[[pd.Index], pd.Index] | None) -> str:
    records = []
    for value, group in df.groupby(key, sort=False):
        mean = np.average(group["target"], weights=group["w_loss"])
        records.append(
            {
                key: value,
                "rows": len(group),
                "events": group["event_id"].nunique(),
                "sum w_event": group["w_event"].sum(),
                "baseline wMSE": weighted_mse(
                    group["target"].to_numpy(), group["baseline"].to_numpy(), group["w_loss"]
                ),
                "target wstd": float(
                    np.sqrt(np.average((group["target"] - mean) ** 2, weights=group["w_loss"]))
                ),
            }
        )
    table = pd.DataFrame(records).set_index(key)
    table = table.loc[order(table.index)] if order else table.sort_values("rows", ascending=False)
    return to_markdown(table, key)


def to_markdown(table: pd.DataFrame, key: str) -> str:
    header = [key, *table.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in table.iterrows():
        cells = [str(idx)]
        for col, value in row.items():
            if col in ("rows", "events"):
                cells.append(f"{int(value):,}")
            elif col == "sum w_event":
                cells.append(f"{value:,.1f}")
            else:
                cells.append(f"{value:.5f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def main(start: date, end: date) -> None:
    async with async_session_maker() as session:
        events = await filter_events_by_date_range(
            session,
            start_date=datetime.combine(start, datetime.min.time(), tzinfo=UTC),
            end_date=datetime.combine(end, datetime.max.time(), tzinfo=UTC),
            status=EventStatus.FINAL,
            sport_key=SPORT_KEY,
        )
        dataset = await load_retail_ev_dataset(events, session, MARKET)

    df = frame(dataset)
    drops = dataset.drops
    print(f"Finished {SPORT_KEY} events {start} to {end}: {len(events)}")
    print(f"Events with rows: {df['event_id'].nunique()}; rows: {len(df):,}")
    print(
        f"Dropped: {drops.events_without_close} events without a close; "
        f"{drops.rows_without_exchange:,} rows with no exchange price at or before t; "
        f"{drops.rows_stale_exchange:,} rows whose exchange price is over 2 h old; "
        f"{drops.wide_exchange_snapshots:,} exchange snapshots with back overround over "
        f"{MAX_EXCHANGE_OVERROUND:.0%} ignored"
    )
    print(f"All rows: baseline wMSE {weighted_mse(df['target'], df['baseline'], df['w_loss']):.5f}")
    bucket_order = [label for _, _, label in HORIZON_BUCKETS]
    for key, order in (
        ("season", lambda idx: idx.sort_values()),
        ("retail_source", None),
        ("close_source", None),
        ("horizon", lambda idx: pd.Index([b for b in bucket_order if b in idx])),
        ("book", None),
    ):
        print(f"\n### By {key.replace('_', ' ')}\n")
        print(summarise(df, key, order))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 8, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date())
    args = parser.parse_args()
    asyncio.run(main(args.start, args.end))
