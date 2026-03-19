"""Compare Pinnacle vs Betfair Exchange sharp features on overlapping events.

For events where both Pinnacle and Betfair Exchange closing odds exist,
compute the tabular features with each as sharp reference and measure
correlation. High correlation means BFE is a viable Pinnacle replacement.

Usage:
    uv run python experiments/scripts/sharp_feature_correlation.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from odds_analytics.training.config import MLTrainingConfig
from odds_analytics.training.data_preparation import prepare_training_data_from_config
from odds_core.database import async_session_maker

logger = structlog.get_logger()

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


async def main() -> None:
    # Load data with Pinnacle sharp
    pin_config = MLTrainingConfig.from_yaml(
        str(CONFIGS_DIR / "xgboost_epl_pinnacle_sharp_2024.yaml")
    )
    logger.info("loading_pinnacle_data")
    async with async_session_maker() as session:
        pin_data = await prepare_training_data_from_config(pin_config, session)

    # Load data with Betfair Exchange sharp
    bfe_config = MLTrainingConfig.from_yaml(str(CONFIGS_DIR / "xgboost_epl_betfair_sharp.yaml"))
    logger.info("loading_betfair_data")
    async with async_session_maker() as session:
        bfe_data = await prepare_training_data_from_config(bfe_config, session)

    # Build DataFrames with event IDs
    pin_events = pin_data.event_ids_train
    bfe_events = bfe_data.event_ids_train

    pin_df = pd.DataFrame(pin_data.X_train, columns=pin_data.feature_names)
    pin_df["event_id"] = pin_events
    pin_df["y"] = pin_data.y_train

    bfe_df = pd.DataFrame(bfe_data.X_train, columns=bfe_data.feature_names)
    bfe_df["event_id"] = bfe_events
    bfe_df["y"] = bfe_data.y_train

    # Find overlapping events
    pin_event_set = set(pin_events)
    bfe_event_set = set(bfe_events)
    overlap = pin_event_set & bfe_event_set

    print(f"\nPinnacle events: {len(pin_event_set)}")
    print(f"Betfair Exchange events: {len(bfe_event_set)}")
    print(f"Overlapping events: {len(overlap)}")
    print(f"BFE-only events (no Pinnacle): {len(bfe_event_set - pin_event_set)}")
    print(f"Pinnacle-only events (no BFE): {len(pin_event_set - bfe_event_set)}")

    # Filter to overlapping events, aligned by event_id
    pin_overlap = pin_df[pin_df["event_id"].isin(overlap)].set_index("event_id").sort_index()
    bfe_overlap = bfe_df[bfe_df["event_id"].isin(overlap)].set_index("event_id").sort_index()

    # Verify targets match (same events should have same bet365 CLV target)
    target_corr = np.corrcoef(pin_overlap["y"], bfe_overlap["y"])[0, 1]
    target_diff = (pin_overlap["y"] - bfe_overlap["y"]).abs().max()
    print(f"\nTarget correlation: {target_corr:.6f}")
    print(f"Max target difference: {target_diff:.8f}")

    # Compare features
    print(f"\n{'Feature':<40} {'Correlation':>12} {'Pin mean':>10} {'BFE mean':>10} {'Diff':>10}")
    print("-" * 85)

    sharp_features = []
    non_sharp_features = []

    for feat in pin_data.feature_names:
        pin_vals = pin_overlap[feat].values
        bfe_vals = bfe_overlap[feat].values

        # Skip if all NaN
        valid = ~(np.isnan(pin_vals) | np.isnan(bfe_vals))
        if valid.sum() < 10:
            print(f"{feat:<40} {'insufficient data':>12}")
            continue

        corr = np.corrcoef(pin_vals[valid], bfe_vals[valid])[0, 1]
        pin_mean = np.nanmean(pin_vals)
        bfe_mean = np.nanmean(bfe_vals)
        diff = abs(pin_mean - bfe_mean)

        marker = ""
        if "sharp" in feat:
            marker = " ← sharp-derived"
            sharp_features.append((feat, corr, pin_mean, bfe_mean))
        else:
            non_sharp_features.append((feat, corr, pin_mean, bfe_mean))

        print(f"{feat:<40} {corr:>12.4f} {pin_mean:>10.4f} {bfe_mean:>10.4f} {diff:>10.4f}{marker}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    if sharp_features:
        sharp_corrs = [c for _, c, _, _ in sharp_features]
        print(f"Sharp-derived features ({len(sharp_features)}):")
        print(f"  Mean correlation: {np.mean(sharp_corrs):.4f}")
        print(f"  Min correlation:  {min(sharp_corrs):.4f}")
    if non_sharp_features:
        non_sharp_corrs = [c for _, c, _, _ in non_sharp_features]
        print(f"Non-sharp features ({len(non_sharp_features)}):")
        print(f"  Mean correlation: {np.mean(non_sharp_corrs):.4f}")
        print(f"  Min correlation:  {min(non_sharp_corrs):.4f}")

    # Check: do BFE-only events have different characteristics?
    bfe_only_events = bfe_event_set - pin_event_set
    if bfe_only_events:
        bfe_only = bfe_df[bfe_df["event_id"].isin(bfe_only_events)]
        print(f"\nBFE-only events (missing Pinnacle, n={len(bfe_only_events)}):")
        print(f"  Target mean: {bfe_only['y'].mean():.6f}")
        print(f"  Target std:  {bfe_only['y'].std():.6f}")
        print(f"  vs overlap target mean: {bfe_overlap['y'].mean():.6f}")


if __name__ == "__main__":
    asyncio.run(main())
