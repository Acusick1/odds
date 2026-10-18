---
type: reference
title: Target functions
description: The retail-EV row builder, its features, weights and keep-bar metrics, the devigged and raw delta functions, the `_compute_target` dispatch, closing-snapshot selection, and the target fields of FeatureConfig.
sources:
  - odds_analytics.retail_ev
  - odds_analytics.retail_ev_metrics
  - odds_core.snapshot_sources
  - odds_core.bookmakers.UK_RETAIL_BOOKMAKERS
  - odds_core.bookmakers.canonical_bookmaker_key
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
  - odds_analytics.sequence_loader.calculate_devigged_totals_target
  - odds_analytics.sequence_loader.calculate_regression_target
  - odds_analytics.sequence_loader.extract_devigged_h2h_probs
  - odds_analytics.feature_groups._compute_target
  - odds_analytics.feature_groups._select_closing_snapshot
  - odds_analytics.training.config.FeatureConfig
status: stable
---

# Target functions

`odds_analytics.training.config.FeatureConfig.target_type` selects the target: `retail_ev` is built by `odds_analytics.retail_ev`; `devigged_bookmaker` and `raw` by functions in `odds_analytics.sequence_loader`, which holds the delta maths for every model although its name suggests only LSTM sequence loading. Its `odds_analytics.sequence_loader.TargetType` enum is referenced nowhere else in the repo (#473). Why each target is shaped as it is is on [CLV and the training target](clv-target.md).

## Retail-EV rows

`odds_analytics.feature_groups.prepare_training_data` hands a `retail_ev` config to `odds_analytics.retail_ev.prepare_retail_ev_training_data` and raises if a sampler is passed. That function loads each FINAL event with `odds_analytics.feature_groups.collect_event_data` (snapshots holding the primary market, without prediction-market-only snapshots) and calls `odds_analytics.retail_ev.build_event_rows`.

**Snapshot kinds.** `odds_core.snapshot_sources.snapshot_kind` reads a snapshot's `raw_data` tags into an `odds_core.snapshot_sources.SnapshotKind`: no `source` is the Odds API, and an FDUK snapshot is opening or closing by its `timing.basis` (`odds_core.snapshot_sources.FdukTimeBasis`), never by `fetch_tier`. An unknown source tag raises `ValueError`.

**Rows.** For each snapshot of a kind in `odds_analytics.retail_ev.RETAIL_KINDS` (Odds API, OddsPortal, OddsPortal live, FDUK opening) whose time t lies between `odds_analytics.retail_ev.DECISION_WINDOW` and `odds_analytics.retail_ev.DECISION_CUTOFF` before kick-off, every outcome priced by a book in `odds_core.bookmakers.UK_RETAIL_BOOKMAKERS` is a row. Book keys pass through `odds_core.bookmakers.canonical_bookmaker_key` first, which maps aliases such as `betfred_uk` to one key and the Odds API's `betmgm` (US BetMGM) to `betmgm_us`, outside the UK roster. FDUK closing snapshots, at kick-off, never give rows. Outcomes are matched by name to the event's home and away teams and `Draw` (`odds_analytics.retail_ev.market_outcomes`: three for `1x2`, two for `h2h`).

**Close.** The last `betfair_exchange` price from a live Betfair snapshot (`odds_core.snapshot_sources.SnapshotKind.BETFAIR_LIVE`) with all outcomes, not in-play (`betfair_meta.inplay`) and within `odds_analytics.retail_ev.LIVE_CLOSE_WINDOW` before kick-off; otherwise the FDUK closing snapshot's `betfair_exchange` price. With neither, the event gives no rows and counts in `odds_analytics.retail_ev.DropCounts.events_without_close`.

An exchange price, at the close or at t, counts only when its back overround is at most `odds_analytics.retail_ev.MAX_EXCHANGE_OVERROUND`; wider ones are skipped and counted in `odds_analytics.retail_ev.DropCounts.wide_exchange_snapshots`.

**Exchange price at t.** The latest complete `betfair_exchange` price at or before t from live Betfair or FDUK opening snapshots (`odds_analytics.retail_ev.EXCHANGE_AT_T_KINDS`). Rows without one, or with one older than `odds_analytics.retail_ev.MAX_EXCHANGE_AGE`, are dropped and counted in `odds_analytics.retail_ev.DropCounts`. Exchange prices inside OddsPortal or Odds API snapshots are not used.

**Devig, target and baseline.** Exchange and book prices are devigged proportionally with `odds_core.odds_math.devig_probabilities` on `price_decimal`. `odds_analytics.retail_ev.RetailEvRow.target` is `close × retail_decimal − 1` and `odds_analytics.retail_ev.RetailEvRow.baseline` is `exchange-at-t × retail_decimal − 1`, both for the row's outcome.

**Weights.** `odds_analytics.retail_ev.RetailEvDataset.loss_weights` is `event_weights / retail_decimal²`, with `odds_analytics.retail_ev_metrics.event_weights` giving `1 / (rows in the event)`. The training matrix scales them to mean 1, which leaves weighted metrics unchanged and keeps `min_child_weight` and `reg_lambda` on their unweighted scale; they reach XGBoost through [the training pipeline](training-pipeline.md#sample-weights).

## Retail-EV features

`odds_analytics.retail_ev.retail_ev_features` returns, for one row, values that use only the row's own snapshot at t, the exchange price at or before t, and the event's kick-off time (names from `odds_analytics.retail_ev.retail_ev_feature_names`):

- `hours_to_kickoff`, `is_weekend`, `day_of_week`;
- the book's `retail_implied_prob` (`1 / retail_decimal`), `book_fair_prob` and `book_overround` (NaN when the book lacks an outcome);
- `consensus_prob` (mean implied probability for the outcome across the snapshot's UK books), `retail_consensus_diff` and `num_bookmakers`, the analogues of the tabular group's `consensus_prob` and `num_bookmakers`;
- `sharp_prob` (the devigged exchange price at t), `retail_sharp_diff` (`retail_implied_prob − sharp_prob`, against a devigged sharp price unlike the tabular group's), `exchange_overround`, `exchange_age_hours` and `baseline_ev`;
- one-hot `outcome_*` and `book_*` columns over `odds_analytics.retail_ev.RETAIL_BOOK_COLUMNS`.

`feature_groups` may add only the event-level groups in `odds_analytics.training.config.EVENT_CONTEXT_GROUPS`, appended per row by `odds_analytics.feature_groups._extract_event_context_parts`; the variance filter then drops constant columns, such as books absent from the data ([feature pipeline](feature-pipeline.md)).

## Keep-bar metrics

`odds_analytics.retail_ev_metrics` implements ADR-0012's two tests, each returning an `odds_analytics.retail_ev_metrics.BootstrapEstimate` (point, 95% percentile CI, `excludes_zero`) from `odds_analytics.retail_ev_metrics.DEFAULT_BOOTSTRAP_SAMPLES` resamples of whole events:

- `odds_analytics.retail_ev_metrics.mse_gain`: baseline weighted MSE minus the model's, with the loss weights; positive means the model is better.
- `odds_analytics.retail_ev_metrics.pick_ev_gain`: the total `w_event × target` over rows the model predicts EV > 0, minus the same total over rows the baseline predicts EV > 0. An empty pick set totals zero.

## Devigged delta

- `odds_analytics.sequence_loader.calculate_devigged_bookmaker_target` returns the closing minus snapshot devigged probability of the configured outcome (`home`, `away` or `draw`) for one bookmaker, or `None` if either side lacks that bookmaker or the outcome. It handles 2-way and 3-way markets; `draw` exists only in 3-way markets.
- `odds_analytics.sequence_loader.extract_devigged_h2h_probs` converts prices with `odds_core.models.Odds.implied_probability` (1 over the stored decimal price) and normalises with `odds_core.odds_math.devig_probabilities`. Devigging is proportional only, and the 3-way (`Draw`) probability is devigged together with home and away ([odds arithmetic](../foundations/odds-math.md), [devigging](../foundations/devigging.md)).
- `odds_analytics.sequence_loader.calculate_devigged_totals_target` does the same for Over/Under.

## Dispatch

`odds_analytics.feature_groups._compute_target` serves the delta targets:

- `"devigged_bookmaker"` uses the functions above.
- `"raw"` uses `odds_analytics.sequence_loader.calculate_regression_target`, the mean vig-inclusive implied-probability delta. Devigging therefore applies to a delta target only when the YAML sets `target_type: devigged_bookmaker`.

`odds_analytics.training.config.FeatureConfig` requires `target_type`. `target_bookmaker` is required for `devigged_bookmaker` and must be omitted otherwise; `devigged_bookmaker` is valid only for `h2h`, `1x2` and `totals`, and `retail_ev` only for `h2h` and `1x2` with the `xgboost` adapter; `outcome` must be `over` or `under` exactly when the market is `totals`, and `retail_ev` ignores it. A config that breaks these rules fails validation.

`odds_analytics.feature_groups._compute_target` passes `outcome` to every branch, so the target and the tabular features describe the same side.

## Closing snapshot of a delta target

`odds_analytics.feature_groups._select_closing_snapshot` takes the snapshots in `closing_tier`. Under `devigged_bookmaker` it prefers those with the target bookmaker's data for the primary market, then the first match in `closing_source_priority`; otherwise, and under `raw`, it takes the latest. `closing_source_priority` is matched against `odds_core.models.OddsSnapshot.api_request_id`, whose values are provenance tags such as `football_data_uk` and `oddsportal_closing` ([provenance tags](../storage/odds-tables.md#provenance)). Because selection is by tier, the FDUK loader clamps opening snapshots out of the closing tier (`scripts/ingest_football_data_uk.py`). A devigged config reads its target bookmaker through a guard that raises if it is unset. `odds_analytics.feature_groups.prepare_training_data` drops events with no closing snapshot and, under `devigged_bookmaker`, events whose closing snapshot lacks the target bookmaker, whichever sampler picks the decision snapshots ([feature pipeline](feature-pipeline.md)). `retail_ev` selects no closing snapshot here.

## Config fields

`odds_analytics.training.config.FeatureConfig` carries `target_type`, `target_bookmaker`, `closing_tier` and `closing_source_priority`; all fields and defaults are on [Training config](training-config.md). The tabular group's `retail_sharp_diff` is the strongest predictor of the delta target ([sharp benchmark](sharp-benchmark.md)).
