---
type: concept
title: Decision time
summary: The as-of instant a bet or prediction is made (hours before the game), how backtests and snapshot samplers choose it, and the default windows.
sources:
  - packages/odds-analytics/odds_analytics/backtesting/config.py::BacktestConfig
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
  - packages/odds-analytics/odds_analytics/feature_groups.py::TimeRangeSampler
  - packages/odds-analytics/odds_analytics/training/config.py::SamplingConfig
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - docs/BACKTESTING_GUIDE.md#data-requirements
  - docs/MODELING.md#hours-to-game-effect
  - docs/MODELING.md#current-conclusions
verified_at: 7887039
---

# Decision time

Decision time is the instant a prediction or bet is assumed to be made. Everything after it is off limits ([look-ahead-bias](look-ahead-bias.md)); the target is measured from it to the close ([clv](clv.md)). It is set separately for backtests and for training.

## Backtests

`packages/odds-analytics/odds_analytics/backtesting/config.py::BacktestConfig` defaults to 1.0. The engine subtracts it from `commence_time` and asks the reader for odds around that instant ([backtest-engine](backtest-engine.md)). The CLI exposes no flag for it.

## Training

`packages/odds-analytics/odds_analytics/training/config.py::SamplingConfig` selects the snapshot sampler:

- `strategy: time_range` (the default) uses `TimeRangeSampler`: snapshots between `min_hours` 3.0 and `max_hours` 12.0 before kickoff, at most `max_samples_per_event` 5, picked from equal-width time bins.
- `strategy: tier` uses `TierSampler`: one row per event, the latest snapshot whose tier is the `decision_tier` (default `pregame`) or any tier further from kickoff. `IN_PLAY` is excluded ([fetch-tier](fetch-tier.md)).

The pregame window (3-12 h) is the documented default because sharp-retail divergence peaks 3-6 h before NBA games ([exp-hours-to-game](exp-hours-to-game.md)); timing on EPL is untested ([clv-model-findings](clv-model-findings.md)).

## Hazards

- **Production uses the sharp tier.** `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml` sets `sampling.strategy: tier` with `decision_tier: sharp` (12-24 h), not the documented pregame window. For events that only have opening and closing snapshots, the candidate set is just the opening snapshot, so the effective decision time is wherever that snapshot was taken (read from code, not run).
- **Tier fallback dilutes the window.** A `pregame` request also accepts sharp, early and opening snapshots, so about 93% of OddsPortal events fell back to the same sharp-tier snapshot as a `sharp` request (Exp 6, see [exp-nba-target-and-data-volume](exp-nba-target-and-data-volume.md)).
- **Backtest and training differ.** Backtests default to 1 h before kickoff; the model is trained at 3-24 h. A model-driven strategy evaluated at 1 h sees inputs unlike its training rows (inferred) ([line-movement-models](line-movement-models.md)).
- **No tolerance control.** The backtest tolerance (30 minutes) is hard-coded.

## Related

- [Feature pipeline](feature-pipeline.md) — where the samplers run
- [Howto: run a backtest](howto-run-backtest.md) — data prerequisites
