"""Betfair Historical NBA Data Analysis.

Parses Betfair historical data files (bz2-compressed JSON) to assess NBA market
liquidity, coverage, and price behaviour. Works with any data tier:

  - Basic (free): 1-min LTP only — market coverage, price evolution, activity
  - Advanced (paid): 1-sec, top 3 back/lay with volume — full liquidity analysis
  - Pro (paid): 50ms, full depth — overkill for this purpose

Download data from https://historicdata.betfair.com/ selecting:
  Sport: Basketball
  Plan: Basic (free) or Advanced
  Market: Match Odds (and optionally Asian Handicap, Over/Under)
  Region/Competition: NBA

Outputs saved to experiments/results/betfair_historical/

Usage:
    uv run python experiments/scripts/betfair_historical_analysis.py <data_dir>

    data_dir: directory containing .bz2 or .tar files from Betfair historical data
"""

from __future__ import annotations

import argparse
import bz2
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "betfair_historical"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["figure.dpi"] = 120


def find_data_files(data_dir: Path) -> list[Path]:
    """Find all .bz2 market data files, including inside extracted tar directories."""
    files = sorted(data_dir.rglob("*.bz2"))
    print(f"Found {len(files)} .bz2 files in {data_dir}")
    return files


def parse_market_file(filepath: Path) -> dict | None:
    """Parse a single Betfair historical market file.

    Each file contains newline-delimited JSON objects representing market snapshots.
    First object is the market definition, subsequent objects are price updates.
    """
    try:
        with bz2.open(filepath, "rt", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}")
        return None

    if not lines:
        return None

    snapshots = []
    market_def = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Extract market definition from first snapshot that has it
        if market_def is None and "mc" in data:
            for mc in data["mc"]:
                if "marketDefinition" in mc:
                    market_def = mc["marketDefinition"]
                    break

        # Extract price data from each snapshot
        if "mc" in data:
            timestamp = data.get("pt")  # published time (epoch ms)
            for mc in data["mc"]:
                if "rc" in mc:  # runner changes
                    snap = {
                        "timestamp_ms": timestamp,
                        "runners": {},
                    }
                    for rc in mc["rc"]:
                        runner_id = rc.get("id")
                        runner_data: dict = {}

                        # Last traded price (Basic tier)
                        if "ltp" in rc:
                            runner_data["ltp"] = rc["ltp"]

                        # Traded volume (Basic tier)
                        if "tv" in rc:
                            runner_data["traded_volume"] = rc["tv"]

                        # Best back/lay (Advanced tier)
                        if "atb" in rc:  # available to back
                            runner_data["back"] = sorted(rc["atb"], key=lambda x: -x[0])
                        if "atl" in rc:  # available to lay
                            runner_data["lay"] = sorted(rc["atl"], key=lambda x: x[0])

                        # Traded price-volume ladder (Advanced tier)
                        if "trd" in rc:
                            runner_data["traded"] = rc["trd"]

                        if runner_data:
                            snap["runners"][runner_id] = runner_data

                    if snap["runners"]:
                        snapshots.append(snap)

    if market_def is None:
        return None

    # Extract runner names from market definition
    runner_names = {}
    for runner in market_def.get("runners", []):
        runner_names[runner.get("id")] = runner.get("name", f"Runner {runner.get('id')}")

    return {
        "event_name": market_def.get("eventName", "Unknown"),
        "market_type": market_def.get("marketType", "Unknown"),
        "market_time": market_def.get("marketTime"),
        "country_code": market_def.get("countryCode"),
        "competition": market_def.get("name", ""),
        "runners": runner_names,
        "n_snapshots": len(snapshots),
        "snapshots": snapshots,
        "status": market_def.get("status"),
    }


def analyse_market(market: dict) -> dict | None:
    """Compute summary statistics for a single market."""
    snapshots = market["snapshots"]
    if not snapshots:
        return None

    runner_ids = list(market["runners"].keys())
    if len(runner_ids) < 2:
        return None

    # Time range
    timestamps = [s["timestamp_ms"] for s in snapshots if s["timestamp_ms"]]
    if not timestamps:
        return None

    first_ts = datetime.fromtimestamp(min(timestamps) / 1000, tz=UTC)
    last_ts = datetime.fromtimestamp(max(timestamps) / 1000, tz=UTC)

    # Parse market start time
    market_time = market.get("market_time")
    if market_time and isinstance(market_time, str):
        try:
            market_start = datetime.fromisoformat(market_time.replace("Z", "+00:00"))
        except ValueError:
            market_start = None
    else:
        market_start = None

    hours_before_first = None
    if market_start:
        hours_before_first = (market_start - first_ts).total_seconds() / 3600

    # Collect LTPs and spreads over time
    ltps: dict[int, list[float]] = {rid: [] for rid in runner_ids}
    spreads: list[float] = []
    back_overrounds: list[float] = []

    for snap in snapshots:
        for rid in runner_ids:
            if rid in snap["runners"] and "ltp" in snap["runners"][rid]:
                ltps[rid].append(snap["runners"][rid]["ltp"])

        # Back-lay spread (Advanced tier only)
        back_prices = {}
        lay_prices = {}
        for rid in runner_ids[:2]:  # Two-runner market
            if rid in snap["runners"]:
                rd = snap["runners"][rid]
                if "back" in rd and rd["back"]:
                    back_prices[rid] = rd["back"][0][0]  # best back price
                if "lay" in rd and rd["lay"]:
                    lay_prices[rid] = rd["lay"][0][0]  # best lay price

        # Per-runner spreads
        for rid in runner_ids[:2]:
            if rid in back_prices and rid in lay_prices:
                back = back_prices[rid]
                lay = lay_prices[rid]
                mid = (back + lay) / 2
                if mid > 0:
                    spreads.append((lay - back) / mid * 100)

        # Back overround
        if len(back_prices) == 2:
            prices = list(back_prices.values())
            overround = sum(1 / p for p in prices) - 1
            back_overrounds.append(overround)

    # Final LTP for each runner (closing price proxy)
    closing_ltps = {}
    for rid in runner_ids:
        if ltps[rid]:
            closing_ltps[rid] = ltps[rid][-1]

    # Total traded volume (if available)
    total_volume = 0.0
    last_snap = snapshots[-1]
    for rid in runner_ids:
        if rid in last_snap["runners"] and "traded_volume" in last_snap["runners"][rid]:
            total_volume += last_snap["runners"][rid]["traded_volume"]

    return {
        "event_name": market["event_name"],
        "market_type": market["market_type"],
        "market_time": market_time,
        "n_snapshots": len(snapshots),
        "first_snapshot": first_ts.isoformat(),
        "last_snapshot": last_ts.isoformat(),
        "hours_before_first_snapshot": round(hours_before_first, 1) if hours_before_first else None,
        "duration_hours": round((last_ts - first_ts).total_seconds() / 3600, 1),
        "n_ltp_updates_r1": len(ltps[runner_ids[0]]),
        "n_ltp_updates_r2": len(ltps[runner_ids[1]]) if len(runner_ids) > 1 else 0,
        "closing_ltp_r1": closing_ltps.get(runner_ids[0]),
        "closing_ltp_r2": closing_ltps.get(runner_ids[1]),
        "runner_1": market["runners"].get(runner_ids[0], "?"),
        "runner_2": market["runners"].get(runner_ids[1], "?") if len(runner_ids) > 1 else "?",
        "total_volume": total_volume if total_volume > 0 else None,
        "median_spread_pct": round(np.median(spreads), 2) if spreads else None,
        "mean_spread_pct": round(np.mean(spreads), 2) if spreads else None,
        "median_overround": round(np.median(back_overrounds), 4) if back_overrounds else None,
        "has_depth_data": len(spreads) > 0,
    }


def plot_liquidity_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary plots from analysed markets."""
    match_odds = df[df["market_type"] == "MATCH_ODDS"].copy()
    if match_odds.empty:
        print("No MATCH_ODDS markets to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Snapshot count distribution
    ax = axes[0, 0]
    ax.hist(match_odds["n_snapshots"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of snapshots per market")
    ax.set_ylabel("Count")
    ax.set_title("Market Data Density")
    ax.axvline(
        match_odds["n_snapshots"].median(),
        color="red",
        linestyle="--",
        label=f"Median: {match_odds['n_snapshots'].median():.0f}",
    )
    ax.legend()

    # 2. Hours before first snapshot
    col = "hours_before_first_snapshot"
    valid = match_odds[match_odds[col].notna()]
    ax = axes[0, 1]
    if not valid.empty:
        ax.hist(valid[col], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Hours before game start")
        ax.set_ylabel("Count")
        ax.set_title("When Markets Open (first snapshot)")
        ax.axvline(
            valid[col].median(),
            color="red",
            linestyle="--",
            label=f"Median: {valid[col].median():.1f}h",
        )
        ax.legend()

    # 3. Spread distribution (Advanced tier only)
    ax = axes[1, 0]
    spreads = match_odds[match_odds["median_spread_pct"].notna()]
    if not spreads.empty:
        ax.hist(spreads["median_spread_pct"], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Median back-lay spread (%)")
        ax.set_ylabel("Count")
        ax.set_title("Back-Lay Spread Distribution")
        ax.axvline(
            spreads["median_spread_pct"].median(),
            color="red",
            linestyle="--",
            label=f"Median: {spreads['median_spread_pct'].median():.2f}%",
        )
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No spread data\n(Basic tier — LTP only)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )

    # 4. Volume distribution (if available)
    ax = axes[1, 1]
    vols = match_odds[match_odds["total_volume"].notna() & (match_odds["total_volume"] > 0)]
    if not vols.empty:
        ax.hist(np.log10(vols["total_volume"]), bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("log10(Total matched volume £)")
        ax.set_ylabel("Count")
        ax.set_title("Matched Volume Distribution")
        ax.axvline(
            np.log10(vols["total_volume"].median()),
            color="red",
            linestyle="--",
            label=f"Median: £{vols['total_volume'].median():,.0f}",
        )
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No volume data\n(requires Live key or Advanced tier)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )

    plt.suptitle("Betfair NBA Market Liquidity Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "liquidity_summary.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Betfair historical NBA data")
    parser.add_argument("data_dir", type=Path, help="Directory containing .bz2 data files")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to process (0 = all)")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir} does not exist")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_data_files(args.data_dir)
    if not files:
        print("No .bz2 files found")
        sys.exit(1)

    if args.max_files > 0:
        files = files[: args.max_files]
        print(f"Processing first {args.max_files} files")

    # Parse all market files
    print(f"\nParsing {len(files)} market files...")
    markets = []
    for i, filepath in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}...")
        market = parse_market_file(filepath)
        if market:
            markets.append(market)

    print(f"Parsed {len(markets)} markets from {len(files)} files")

    # Analyse each market
    print("\nAnalysing markets...")
    results = []
    for market in markets:
        stats = analyse_market(market)
        if stats:
            results.append(stats)

    if not results:
        print("No analysable markets found")
        sys.exit(1)

    df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'=' * 70}")
    print("  BETFAIR NBA HISTORICAL DATA SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total markets analysed: {len(df)}")
    print(f"  Market types: {df['market_type'].value_counts().to_dict()}")

    match_odds = df[df["market_type"] == "MATCH_ODDS"]
    if not match_odds.empty:
        print(f"\n  MATCH ODDS ({len(match_odds)} markets):")
        print(
            f"    Snapshots per market: {match_odds['n_snapshots'].median():.0f} median, "
            f"{match_odds['n_snapshots'].mean():.0f} mean"
        )

        col = "hours_before_first_snapshot"
        valid = match_odds[match_odds[col].notna()]
        if not valid.empty:
            print(f"    Market opens: {valid[col].median():.1f}h median before game")

        if match_odds["median_spread_pct"].notna().any():
            spreads = match_odds["median_spread_pct"].dropna()
            print(
                f"    Back-lay spread: {spreads.median():.2f}% median, {spreads.mean():.2f}% mean"
            )

        if match_odds["median_overround"].notna().any():
            ors = match_odds["median_overround"].dropna()
            print(
                f"    Back overround: {ors.median() * 100:.2f}% median, "
                f"{ors.mean() * 100:.2f}% mean"
            )

        vols = match_odds[match_odds["total_volume"].notna() & (match_odds["total_volume"] > 0)]
        if not vols.empty:
            print(
                f"    Matched volume: £{vols['total_volume'].median():,.0f} median, "
                f"£{vols['total_volume'].sum():,.0f} total"
            )
            print(
                f"    Volume range: £{vols['total_volume'].min():,.0f} — "
                f"£{vols['total_volume'].max():,.0f}"
            )

        n_with_ltp = match_odds[match_odds["n_ltp_updates_r1"] > 0].shape[0]
        print(f"    Markets with LTP updates: {n_with_ltp}/{len(match_odds)}")

    # Save CSV
    csv_path = OUTPUT_DIR / "market_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")

    # Plot
    plot_liquidity_summary(df, OUTPUT_DIR)

    print(f"\nAll outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
