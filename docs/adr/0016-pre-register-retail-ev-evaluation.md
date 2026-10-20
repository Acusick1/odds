---
status: accepted
---

# Pre-register the fresh-season evaluation of the retail-EV model

The [ADR-0012](0012-retail-ev-against-betfair-exchange-close.md) retail-EV model is judged once, on the 2026-27 EPL events that kick off on or after a freeze date, by ADR-0012's two keep-bar conditions. This ADR names the season, the freeze date, what is frozen, the single look, the collection rules and the statistics that decide.

## Context

ADR-0012 keeps a model only if it passes both keep-bar conditions "on a fresh season", and leaves the season and its freeze date to pre-registration (#539, under the map #506).

- **Development results.** On 2024-25 and 2025-26, the residual arm (XGBoost on the target minus the baseline, with the baseline added back) gains 1.0 to 1.4% of the baseline's weighted MSE, with CIs that include zero. Its picks are level with the baseline's. The direct arm is worse than the baseline ([price-only](../../wiki/evaluation/exp-epl-retail-ev-price-only.md)). No as-of-t group beats price-only ([as-of-t groups](../../wiki/evaluation/exp-epl-retail-ev-as-of-t-groups.md)), and re-tuning inside 2025-26 does worse than the settings tuned on 2024-25 ([nested tuning](../../wiki/evaluation/exp-epl-retail-ev-nested-tuning.md)).
- **Power.** One season of 380 events gives the MSE condition 80% power at a gain of about 4.5 to 6%. The pick condition needs about 16 to 21% for a diffuse edge, or exact knowledge of the move on about 2 to 2.6% of events. Pick power below about 0.15 on FDUK-only rows is not evidence of an edge ([keep-bar power](../../wiki/evaluation/exp-retail-ev-power.md)).
- **2026-27 so far.** No model has been scored on it. The target report (#537) and an early version of the power analysis counted its rows descriptively. Its events before 2026-10-19 have FDUK opening and closing prices and live Betfair prices, but no live OddsPortal rows with bet365. Production has collected OddsPortal live with bet365 for EPL since 2026-10-19.
- **Pipeline and evaluation command.** The residual arm is a pipeline option, `FeatureConfig.retail_ev_arm: residual`, which passes `baseline_ev` to XGBoost as its base margin (#558). `odds evaluate retail-ev --spec` runs a one-look evaluation from a YAML spec and writes a manifest (#559; [retail-EV evaluation](../../wiki/evaluation/retail-ev-evaluation.md)). [ADR-0014](0014-ignore-wide-exchange-prices.md) (the 5% exchange-overround cap), [ADR-0017](0017-fduk-opening-rows-use-own-exchange-price.md) (FDUK opening rows pair only with their own exchange price) and [ADR-0013](0013-collect-polymarket-epl-as-sharp-reference.md) (Polymarket collection) are accepted.

## Decision

### Evaluation window and freeze

- **Freeze date:** 2026-11-02 00:00 UTC, or the first Monday after this ADR is merged, whichever is later. The freeze is the annotated git tag `retail-ev-eval-2026-27`, pushed before the window starts; if it is pushed later, `window.start` moves to the first Monday after the tag before it is pushed.
- **Evaluation events:** EPL 2026-27 events whose kick-off is at or after the freeze date, through the last 2026-27 match. That is about 280 events if the freeze is on 2026-11-02 (inferred from the fixture count to mid-October). The spec's window ends 2027-06-30, after the last scheduled match, so a match rescheduled late in the season stays in it. Kick-off means the event's actual kick-off, so a postponed match counts if it is played in the window.
- **2026-27 events before the freeze** are training data only. They are never scored.

### Prerequisites before the freeze

All are met.

1. **The residual arm is a pipeline option (#558).** It fits with no intercept, where the experiment script's transform fitted one, so a model that learns nothing predicts the baseline exactly.
2. **The evaluation command exists and is reviewed (#559).** Its dry run on development data with 50-event refits and the frozen trial gains 32.0 × 10⁻⁶ [−25.7, 88.9] in weighted MSE, against 32.4 [−24.2, 87.7] for the #558 rerun of the price-only protocol. That is within seed noise (seed 42 minus seed 7 is about 20), not exact; dropping the all-zero `book_boylesports` column reproduces the rerun to every digit. `experiments/configs/retail_ev_evaluation_2026_27_dry_run.yaml`, the frozen spec with its window moved to development data, scores 134 events of February to July 2026 in 14 blocks and gives RETIRE: MSE gain 6.9 × 10⁻⁶ [−75.9, 86.2], pick EV gain 0.023 [−0.007, 0.060].
3. **The loss-weight scale is fixed (#558)** (see [what is frozen](#what-is-frozen)).
4. **The row rules are fixed** by ADR-0014 and ADR-0017.

### What is frozen

Everything the evaluation runs is the code and spec at the freeze tag. The look runs that tag, not a later commit. The mechanism is an `odds_analytics.retail_ev_evaluation.EvaluationSpec` YAML, `experiments/configs/retail_ev_evaluation_2026_27.yaml`, whose window is the one [above](#evaluation-window-and-freeze). `experiments/configs/retail_ev_evaluation_2026_27_dry_run.yaml` is the same spec with a development window. The spec holds every choice in this section and rejects unknown fields.

- **Target, rows and baseline:** ADR-0012 as built in `odds_analytics.retail_ev`, with ADR-0014's 5% overround cap and ADR-0017's FDUK opening-row rule.
- **Model:** `arm: residual` on the 39 price-only columns of `odds_analytics.retail_ev.RetailEvFeatures`, pinned in the spec's `feature_names`. They include the book one-hot list at the tag, with `book_boylesports`, which has no development row and so stays an all-zero column; the evaluation runs no variance filter. A book outside the UK retail roster those columns come from has no rows; it is only counted in the report's `prices_outside_roster`.
- **Hyperparameters:** the residual seed-42 best trial from 2024-25 tuning, as recorded in `experiments/results/retail_ev_as_of_t_groups/summary.json` (spec `trial`: variant `price_only`, run `residual_s42`): 100 trees, depth 4, learning rate 0.0101, `min_child_weight` 1982, subsample 0.6, colsample 0.7, λ 1.95. No further tuning.
- **Seed:** 42 for XGBoost's `random_state`. Identical runs give identical predictions (#559's tests).
- **Weight scale:** each row's loss weight is `odds_analytics.retail_ev.LOSS_WEIGHT_SCALE × w_event / odds²`, with the constant 1013, so an event weighs about 156 in every training set. 1013 is the scale the frozen trial was tuned at: 156.19 weight units per event (112,455 rows over 720 events) divided by the mean per-event `1/odds²` of 0.15413 (#558). Under mean-one row weights, the frozen `min_child_weight` of 1,982 stands for 195 events per leaf on 2024-25 alone and 6.7 on 2025-26 alone; under the fixed scale it stands for about 12.5 to 13.
- **Training window and refits:** expanding walk-forward from 2024-08-01 (`training_start`). Evaluation events are scored in blocks of 10 in kick-off order (`refit_every_events`), about one matchday each when events have rows. Each block's model trains on every event that kicks off at least 7 days before the block's first event, because decision times reach 7 days before kick-off. Its newest training data is therefore 7 to 14 days old. The last block may be partial and is scored.

### One look

- The evaluation is `odds evaluate retail-ev --spec experiments/configs/retail_ev_evaluation_2026_27.yaml`, with `--output-dir experiments/results/retail_ev_evaluation_2026_27`, run once from a checkout of the tag, after 2027-06-30 and once FDUK's final 2026-27 file is loaded ([how-to](../../wiki/evaluation/howto-run-retail-ev-evaluation.md)).
- It runs on a local copy of production (`odds copy from-prod`), never against production. Its manifest records the git SHA and tag, `git_dirty`, the spec, trial and data hashes, the predictions hash and the coverage counts. The outputs are committed.
- The tag, the committed spec and the committed manifest record the look. The command also refuses a window that has not ended, and refuses a spec whose hash is already in a manifest under the checkout's `experiments/results` unless `--force`, which it records. That guard does not see a run in another checkout, so a second run anywhere voids the look unless it only repeats an identical run.
- No interim result is computed. Collection health is watched through the existing alerts and heartbeats, which show no model output.

### Collection requirements and gaps

- **Expected collection:** live OddsPortal (with bet365) and live Betfair Exchange for every EPL event, at the cadences on [fetch-oddsportal](../../wiki/collection/fetch-oddsportal-job.md) and [Betfair Exchange](../../wiki/collection/betfair-exchange.md), and the FDUK 2026-27 file.
- **No exclusions for gaps.** Every evaluation event with a close and at least one row under the frozen rules is scored. Model and baseline are scored on the same rows, so a gap changes which rows exist, and with it power, but favours neither (inferred). The close falls back to FDUK as ADR-0012 sets out. An event with no row is not scored, and the report lists those events with their status and source flags.
- **Readiness and named exceptions.** The command refuses a window in which any event is not FINAL or has no FDUK closing snapshot, unless that event is named with `--allow-incomplete <event ID>`; the manifest records each named event. An event may be named only if it was not played as listed: a SCHEDULED ghost of a rescheduled fixture (such as `op_live_ASTVIL_LIV_2026-05-17T1130` in development, #440), or a match abandoned or postponed past the window. Each named event and its reason go in the result. A played event whose FDUK file is late waits for the file instead.
- **The Betfair close is a collection risk (#560).** FDUK can leave Betfair Exchange closing prices blank for whole matchdays; its 2025-26 file has none for kick-offs from 2026-04-20 to 2026-05-04. The live Betfair job is then the only close, and 11 development events in that span had neither and got no rows. The live Betfair job runs 5 minutes before each kick-off (#562), and the health check alerts when an EPL event has no live Betfair close (#560). Gaps of this kind reduce the scored events, not the test's validity.
- **Coverage is reported, not gated:** the share of evaluation events with OddsPortal live rows, with a live Betfair close, and FDUK-only, plus outage runs, all in the command's report.
- **Collection fixes are allowed** during the season. A change to the target, row, feature or model code is not. If the tagged code cannot read season data, a fix limited to reading may be applied on a branch from the tag, reviewed, and named in the report.

### Statistics that decide

Over all scored rows of the evaluation events (`odds_analytics.retail_ev_evaluation.keep_bar`):

1. **MSE condition:** `odds_analytics.retail_ev_metrics.mse_gain`, baseline weighted MSE minus the model's.
2. **Pick condition:** `odds_analytics.retail_ev_metrics.pick_ev_gain`, the model's total `w_event`-weighted realised EV over rows with predicted EV > 0, minus the baseline's.

Each has an event-clustered percentile bootstrap 95% CI with 10,000 resamples and seed 0 (spec `bootstrap`). The model is **kept** if both CI lower bounds lie above zero, and **retired** otherwise. Each condition is then a one-sided test at 2.5%, and requiring both keeps the overall false-keep rate at about 2.5% or less whatever the dependence between them, so no multiplicity correction applies (intersection–union test; approximate, because the CIs are percentile bootstraps). Slices by horizon, retail source, event type, outcome and block are reported for description only and decide nothing.

## Considered options

- **Season length.** Taken: post-freeze 2026-27 only, about 280 events, with minimum detectable gains of about 7% for the MSE condition and 22 to 25% for a diffuse pick edge (inferred, interpolated between 190 and 380 events on [keep-bar power](../../wiki/evaluation/exp-retail-ev-power.md#minimum-detectable-effect-at-80-power)). Rejected: adding all of 2027-28, about 660 events, with about 3% and 13 to 15% (inferred, extrapolated). The pick condition binds in both. The development gain of 1.0 to 1.4% is below what either design detects reliably, and two seasons would block any new model's fresh season for 19 months. A challenger developed meanwhile can be pre-registered on 2027-28 in its own ADR.
- **Looks.** Taken: one, at season end. Rejected: an interim look with a 99.9% bar (Haybittle–Peto), considered only with the two-season design.
- **Window start.** Taken: the freeze. Rejected: 2026-10-19, the start of live bet365 collection, because events would be scored before anything was frozen; and all of 2026-27, whose early events were counted descriptively (#537) and have no live OddsPortal rows.
- **Pick condition.** Taken: ADR-0012's condition 2 as written, which the power analysis tested. Rejected: weighting each pick's realised EV by `1/odds`, whose power is untested and which would be a change made after seeing development data.
- **Hyperparameters.** Taken: the frozen 2024-25 trial. Rejected: one fresh tuning study on all development events, and nested re-tuning at every refit, which did worse on development data ([nested tuning](../../wiki/evaluation/exp-epl-retail-ev-nested-tuning.md)).
- **Refits.** Taken: every 10 events. Rejected: every 50, the development CV step, which leaves the model up to five matchdays behind the dense live rows; and one fit at the freeze, which never trains on dense live 2026-27 rows before scoring them.
- **Features.** Taken: price-only. Rejected: adding as-of-t groups, none of which beat price-only on development data, and evaluating several candidates, each of whose CIs would widen to 1 − 0.05/k.
- **Prospective logging of predictions at decision time.** It proves no look-ahead, but needs the model in a scheduled job before the freeze. The tagged command, run once on a copy, uses only data at or before each row's t by construction ([look-ahead](../../wiki/modelling/look-ahead-bias.md)), and ADR-0017 removes the one known leak, from FDUK's upper-bound times.
- **Excluding events with collection gaps.** It would make the scored events depend on outages, and gives the model no advantage to remove (inferred).

## Consequences

- Until the look, no change to the retail-EV target, rows, features or model code can affect this evaluation, because the look runs the tag. Development on 2026-27 data for another model is allowed, but that model cannot then use 2026-27 as its fresh season.
- A fail retires this model. It does not show the absence of an edge smaller than the minimum detectable effect.
- If this ADR merges after 2026-11-02, the freeze and window start move to the next Monday, the spec's `window.start` changes with them before the tag, and the window loses those events.
- The no-intercept residual arm differs slightly from the script transform the development results used, so the development numbers are a guide to the frozen model, not its exact record.

## Evidence

- Issues #506, #537, #538, #539, #540, #545, #549, #560, #562; PRs #558 and #559.
- Specs: `experiments/configs/retail_ev_evaluation_2026_27.yaml` (the look) and `experiments/configs/retail_ev_evaluation_2026_27_dry_run.yaml`, with its outputs in `experiments/results/retail_ev_evaluation_2026_27_dry_run/`.
- Wiki: [retail-EV evaluation](../../wiki/evaluation/retail-ev-evaluation.md), [keep-bar power](../../wiki/evaluation/exp-retail-ev-power.md), [price-only](../../wiki/evaluation/exp-epl-retail-ev-price-only.md), [as-of-t groups](../../wiki/evaluation/exp-epl-retail-ev-as-of-t-groups.md), [nested tuning](../../wiki/evaluation/exp-epl-retail-ev-nested-tuning.md), [CLV target](../../wiki/modelling/clv-target.md#the-retail-ev-target).
- `experiments/results/retail_ev_as_of_t_groups/summary.json` for the frozen hyperparameters.
