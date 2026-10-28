---
type: reference
title: Target functions
description: The retail-EV row builder, its features, weights and keep-bar metrics, the raw delta function, the `_compute_target` dispatch, closing-snapshot selection, and the target fields of FeatureConfig.
sources:
  - odds_analytics.retail_ev
  - odds_analytics.retail_ev_metrics
  - odds_core.betfair_close.pick_close
  - odds_core.exchange_price.ExchangePrice
  - odds_core.exchange_price.MAX_EXCHANGE_OVERROUND
  - odds_analytics.as_of_team_features
  - odds_core.snapshot_sources
  - odds_core.bookmakers.UK_RETAIL_BOOKMAKERS
  - odds_core.bookmakers.canonical_bookmaker_key
  - odds_analytics.targets.calculate_regression_target
  - odds_analytics.feature_groups._compute_target
  - odds_analytics.training.config.FeatureConfig
status: stable
---

# Target functions

`odds_analytics.training.config.FeatureConfig.target_type` selects the target: `retail_ev` is built by `odds_analytics.retail_ev` and `raw` by `odds_analytics.targets.calculate_regression_target`. Why each target is shaped as it is is on [CLV and the training target](clv-target.md).

## Retail-EV rows

`odds_analytics.feature_groups.prepare_training_data` passes a `retail_ev` config to `odds_analytics.retail_ev.prepare_retail_ev_training_data` and raises if a sampler is passed. That function and the report scripts both go through `odds_analytics.retail_ev.load_retail_ev_dataset`, which reads every snapshot of each event (`odds_lambda.storage.readers.OddsReader.get_snapshots_for_event`, ties in snapshot time ordered by snapshot id), calls `odds_analytics.retail_ev.build_event_rows`, and logs the drop counts once per build. It skips events that kick off after `now` (the current time unless the caller passes one). Events are selected by having a close, not by status or scores: an event with no close yields no rows, and an unsettled event with a close yields them.

**Snapshot kinds.** `odds_core.snapshot_sources.snapshot_kind` reads a snapshot's `raw_data` tags into an `odds_core.snapshot_sources.SnapshotKind`: no `source` is the Odds API, and an FDUK snapshot is opening or closing by its `timing.basis` (`odds_core.snapshot_sources.FdukTimeBasis`), never by `fetch_tier`. An unknown source tag raises `ValueError`.

The builder reads each snapshot's `raw_data`, not its price rows: `odds_core.market_prices.book_prices` gives each book's prices and `odds_core.snapshot_sources.snapshot_kind` its kind.

**Rows.** For each snapshot of a kind in `odds_analytics.retail_ev.RETAIL_KINDS` (Odds API, OddsPortal, OddsPortal live, FDUK opening) whose time t lies between `odds_analytics.retail_ev.DECISION_WINDOW` and `odds_analytics.retail_ev.DECISION_CUTOFF` before kick-off, every outcome priced by a book in `odds_core.bookmakers.UK_RETAIL_BOOKMAKERS` is a row. Book keys pass through `odds_core.bookmakers.canonical_bookmaker_key` first, which maps aliases such as `betfred_uk` to one key and the Odds API's `betmgm` (US BetMGM) to `betmgm_us`, outside the UK roster. FDUK closing snapshots, at kick-off, never give rows. Prices from books outside the roster in a decision-window snapshot are counted per key in `odds_analytics.retail_ev.DropCounts.prices_outside_roster`. Outcomes are matched by name to the event's home and away teams and `Draw` (`odds_core.market_prices.market_outcomes`: three for `1x2`, two for `h2h`; over and under for `totals`, which the builder does not read). Each row is an `odds_analytics.retail_ev.RetailEvRow`: the `odds_analytics.retail_ev.RetailEvInput` known at t plus the close. `odds_analytics.retail_ev.quote_inputs` builds the inputs of one retail snapshot, one per book and outcome, and the [retail-EV model rule](../evaluation/candidate-rules.md#the-retail-ev-model-rule) builds its inputs from a market state with the same function. For reporting, each input labels its horizon bucket from hours to kick-off (lower edges `odds_analytics.retail_ev.HORIZON_BUCKET_EDGES_HOURS`, labels `odds_analytics.retail_ev.HORIZON_BUCKET_LABELS`; a row outside the 7-day decision window raises `ValueError`) and its season, which starts in `odds_analytics.retail_ev.SEASON_START_MONTH`.

**Close.** `odds_analytics.retail_ev.build_event_rows` reads an `odds_core.exchange_price.ExchangePrice` (time, the source and time basis its snapshot is stored under, whether it is in-play, devigged probabilities, overround) from each live Betfair and FDUK snapshot that prices every outcome on `betfair_exchange` (`odds_analytics.retail_ev.EXCHANGE_KINDS`; in-play from `betfair_meta.inplay`), and takes the event's close from `odds_core.betfair_close.pick_close`, the rule `odds_core.betfair_close.event_close` applies to stored quotes. It is the last live Betfair price that is not in-play and within `odds_core.betfair_close.LIVE_CLOSE_WINDOW` before kick-off; otherwise the FDUK closing snapshot's `betfair_exchange` price. Kick-off is the event's `commence_time`. Betfair's own start (`betfair_meta.market_start_time`) plays no part, so a pre-play snapshot taken after it but before kick-off can be the close ([Betfair job](../collection/betfair-exchange.md#flow)). With neither, the event gives no rows and counts in `odds_analytics.retail_ev.DropCounts.events_without_close`.

An exchange price, at the close or at t, counts only when its back overround is at most `odds_core.exchange_price.MAX_EXCHANGE_OVERROUND`; wider ones are skipped and counted in `odds_analytics.retail_ev.DropCounts.wide_exchange_snapshots` (counted by the at-t pass of `odds_analytics.retail_ev.build_event_rows`, which passes every price it read to `odds_core.betfair_close.pick_close` with the same cap).

**Exchange price at t.** The latest complete `betfair_exchange` price at or before t from live Betfair or FDUK opening snapshots that is not in-play. Rows without one, or with one older than `odds_analytics.retail_ev.MAX_EXCHANGE_AGE`, are dropped and counted in `odds_analytics.retail_ev.DropCounts`. An FDUK opening row takes only the price in its own snapshot and is dropped without one (`odds_analytics.retail_ev.DropCounts.rows_without_own_exchange`), because its t is only an upper bound ([CLV target](clv-target.md#the-retail-ev-target)). Exchange prices inside OddsPortal or Odds API snapshots are not used. [Candidate rules](../evaluation/candidate-rules.md#market-state) take both prices from one snapshot instead.

**Devig, target and baseline.** Exchange and book prices are devigged proportionally with `odds_core.odds_math.devig_probabilities` on `price_decimal`. `odds_analytics.retail_ev.RetailEvRow.target` is `close × retail_decimal − 1` and `odds_analytics.retail_ev.RetailEvInput.baseline` is `exchange-at-t × retail_decimal − 1`, both for the row's outcome.

**Weights.** `odds_analytics.retail_ev.RetailEvDataset.loss_weights` is `odds_analytics.retail_ev.LOSS_WEIGHT_SCALE × event_weights / retail_decimal²`, with `odds_analytics.retail_ev.event_weights` giving `1 / (rows in the event)` and the scale fixed at 1,013. An event's weights therefore sum to 1,013 times its mean `1 / retail_decimal²`: 125 to 291, mean 156, on development data, whatever other events the training rows hold. 1,013 makes that mean equal the scale the frozen 2024-25 residual trial was tuned at: mean-one weights over 112,455 rows and 720 events, 156.2 per event. The weighted metrics do not depend on the scale; `min_child_weight` and `reg_lambda` do, so a fixed scale keeps a tuned value worth the same number of events at every refit ([weight scale](clv-target.md#the-retail-ev-target)). The training matrix passes the weights unchanged; they reach XGBoost through [the training pipeline](training-pipeline.md#sample-weights), with each row's kick-off time (`odds_analytics.retail_ev.RetailEvDataset.commence_times`) for the [walk-forward gap](cross-validation.md#how-folds-are-built). Outside that pipeline, `odds_analytics.retail_ev.fit_retail_ev` fits the model wrapper on a matrix, targets and loss weights, passing the baseline as the base margin for the `residual` arm; the [retail-EV evaluation](../evaluation/retail-ev-evaluation.md#refit-blocks) and the model rule fit through it.

## Retail-EV features

`odds_analytics.retail_ev.RetailEvFeatures` turns rows into the feature matrix and names. Its price features use only the row's own snapshot at t, the exchange price at or before t, and the event's kick-off time:

- `hours_to_kickoff`, `is_weekend`, `day_of_week`;
- the book's `retail_implied_prob` (`1 / retail_decimal`), `book_fair_prob` and `book_overround` (NaN when the book lacks an outcome);
- `consensus_prob` (mean implied probability for the outcome across the snapshot's UK books), `retail_consensus_diff` and `num_bookmakers`, the analogues of the tabular group's `consensus_prob` and `num_bookmakers`;
- `sharp_prob` (the devigged exchange price at t), `retail_sharp_diff` (`retail_implied_prob − sharp_prob`, against a devigged sharp price unlike the tabular group's), `exchange_overround`, `exchange_age_hours` and `baseline_ev`;
- one-hot `outcome_*` and `book_*` columns over `odds_analytics.retail_ev.RETAIL_BOOK_COLUMNS`.

`feature_groups` adds optional groups from `odds_analytics.training.config.RetailEvFeatureGroupName`, appended in that literal's order. Each reads only data at or before t, and a missing value is NaN:

| Group | Columns | Source at or before t |
|---|---|---|
| `polymarket_at_t` | `pm_prob`, `pm_exchange_diff`, `pm_consensus_diff`, `pm_baseline_ev`, `pm_spread`, `pm_age_hours` | the latest complete `polymarket` snapshot before kick-off and no older than `odds_analytics.retail_ev.MAX_POLYMARKET_AGE` (2 hours), devigged proportionally (`odds_analytics.retail_ev.PolymarketPrice`). The diffs are against `sharp_prob` and `consensus_prob`, `pm_baseline_ev` is `pm_prob × retail_decimal − 1`, and `pm_spread` is the outcome's Yes ask minus bid, which only live snapshots store ([Polymarket snapshots](../collection/polymarket-odds-snapshots.md#snapshot-shape)) |
| `dynamics` | `exchange_move_{1,3,6}h`, `consensus_move_{1,3,6}h`, `retail_dispersion` | for each of `odds_analytics.retail_ev.MOVE_WINDOWS` Δ, a move compares t with the latest price of the same source at or before t − Δ and no older than `odds_analytics.retail_ev.MAX_LAGGED_PRICE_AGE` (2 hours) before it. `exchange_move` is `sharp_prob` minus that exchange price (live Betfair against live Betfair, FDUK opening against FDUK opening), NaN when it is the same snapshot as the price at t. `consensus_move` is the mean change in implied probability over the books priced in both retail snapshots. `retail_dispersion` is the standard deviation of the outcome's implied probability across the snapshot's UK books, NaN for one book (`odds_analytics.retail_ev.Dynamics`) |
| `schedule_as_of_t` | the 8 `eplsched_*` columns of the [EPL schedule group](epl-context-features.md) | `odds_analytics.as_of_team_features.TeamHistory.schedule_at` |
| `standings_as_of_t` | the 11 `stnd_*` columns of the [standings group](epl-context-features.md) | `odds_analytics.as_of_team_features.TeamHistory.standings_at`, with `form_window` |

Moves compare one source with itself because each source carries different books and a different price basis, so a change of source is never read as a move. An FDUK opening row, its event's only FDUK opening snapshot, therefore has no moves. An input built from one market state has no history (`dynamics` is `None`), and the group raises `ValueError` on it.

FDUK opening rows (`odds_analytics.retail_ev.UPPER_BOUND_TIME_KINDS`, `odds_analytics.retail_ev.RetailEvInput.decision_time_is_exact`) have a decision time that is only an upper bound, so `odds_analytics.retail_ev.build_event_rows` gives them no Polymarket price and no lagged exchange price, and `odds_analytics.retail_ev.RetailEvFeatures` gives them NaN schedule and standings ([CLV target](clv-target.md#the-retail-ev-target)).

The two team groups count a match only once its result is known: it kicked off before `odds_analytics.as_of_team_features.results_known_before`, t minus `odds_analytics.as_of_team_features.RESULT_LAG` (3 hours, which covers a match to full time `(inferred)`). Rest days and the 14- and 7-day congestion windows are still measured back from the event's kick-off, so a match between t and kick-off is missing from the features rather than leaking into them. ESPN fixture rows count only with a played status (`odds_analytics.as_of_team_features.FINISHED_FIXTURE_STATUSES`), because a rescheduled match leaves a `Scheduled` or `Postponed` row at its old date, and a team with no played fixture row before then has NaN congestion columns. `odds_analytics.as_of_team_features.load_team_history` loads the EPL results (`odds_analytics.standings_features.load_season_events_cache`) and, for the schedule group, the ESPN fixtures, once per build; the events must share one sport. Its `until` bound stops both at a kick-off or fixture date; the training pipeline loads every match, and the [retail-EV evaluation](../evaluation/retail-ev-evaluation.md#what-it-reads) stops at its window end.

The event-level groups (`tabular`, `polymarket`, `injuries` and `rest`) are rejected under `retail_ev`, and the `retail_ev` groups under every other target: the event-level ones are computed as of kick-off, not as of t ([look-ahead](look-ahead-bias.md#no-future-data-layer-by-layer)). The variance filter drops constant columns, such as books absent from the data ([feature pipeline](feature-pipeline.md)).

## Keep-bar metrics

`odds_analytics.retail_ev_metrics` implements ADR-0012's two tests, each returning an `odds_analytics.retail_ev_metrics.BootstrapEstimate` (point, 95% percentile CI, `excludes_zero`) from `odds_analytics.retail_ev_metrics.DEFAULT_BOOTSTRAP_SAMPLES` resamples of whole events:

- `odds_analytics.retail_ev_metrics.mse_gain`: baseline weighted MSE minus the model's, with the loss weights; positive means the model is better.
- `odds_analytics.retail_ev_metrics.pick_ev_gain`: the total `w_event × target` over rows the model predicts EV > 0, minus the same total over rows the baseline predicts EV > 0. An empty pick set totals zero.
- `odds_analytics.retail_ev_metrics.event_clustered_ratio`: a sum of numerators over a sum of denominators with percentile CIs at chosen levels from whole-event resamples; the [candidate report](../evaluation/candidate-registry.md#odds-candidates-report) uses it for its stake-weighted mean EV.
- `odds_analytics.retail_ev_metrics.mse_gain_of_event_totals` and `odds_analytics.retail_ev_metrics.pick_ev_gain_of_event_totals` run the same bootstraps on per-event totals (weighted squared errors and loss-weight totals, or pick-EV differences), which the row-level functions compute and pass to them. The [keep-bar power simulation](../evaluation/exp-retail-ev-power.md) calls them directly.

## Raw delta and dispatch

`odds_analytics.feature_groups._compute_target` serves the `raw` target: it extracts the configured `outcome`'s odds from the decision and closing snapshots for the primary market and returns `odds_analytics.targets.calculate_regression_target`, the closing minus snapshot mean implied-probability delta for `h2h` and point delta for `spreads` and `totals`; any other market, `1x2` included, returns `None` and the row is dropped. The implied probabilities are `odds_core.models.Odds.implied_probability` (1 over the stored decimal price), not devigged ([odds arithmetic](../foundations/odds-math.md), [devigging](../foundations/devigging.md)).

`odds_analytics.training.config.FeatureConfig` requires `target_type` (`raw` or `retail_ev`). `retail_ev_arm` (`odds_analytics.training.config.RetailEvArm`) is `direct` unless `target_type` is `retail_ev`; its `residual` value makes the model fit the target minus the baseline ([residual arm](training-config.md#retail-ev-arm)). `retail_ev` is valid only for `h2h` and `1x2` with the `xgboost` adapter, and ignores `outcome`. A config that breaks these rules fails validation.

## Closing snapshot of a raw target

`odds_analytics.feature_groups.collect_event_data` takes the latest snapshot in `closing_tier` as the closing snapshot. Because selection is by tier, the FDUK loader (`scripts/ingest_football_data_uk.py`) stamps its close `close`, which is CLOSING, and its opening `upper_bound`; `odds_lambda.tier_utils.snapshot_tiers` never gives an opening a closing tier while the event has a `close` snapshot, so an opening under 3 h out is PREGAME, not CLOSING ([the tier a snapshot covers](../storage/fetch-tier.md#the-tier-a-snapshot-covers)). A re-ingest leaves an event's snapshots in place when a candidate decision used one as an input. `odds_analytics.feature_groups.prepare_training_data` drops events with no closing snapshot, and events whose closing snapshot has none of the `sharp_bookmakers`, whichever sampler picks the decision snapshots ([feature pipeline](feature-pipeline.md)). `retail_ev` selects no closing snapshot here.

## Config fields

`odds_analytics.training.config.FeatureConfig` carries `target_type`, `retail_ev_arm` and `closing_tier`; all fields and defaults are on [Training config](training-config.md). The tabular group's `retail_sharp_diff` is built from the same prices as the `raw` target ([sharp benchmark](sharp-benchmark.md)).
