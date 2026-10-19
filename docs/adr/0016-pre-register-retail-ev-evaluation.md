---
status: proposed
---

# Pre-register the fresh-season evaluation of the retail-EV model

The [ADR-0012](0012-retail-ev-against-betfair-exchange-close.md) retail-EV model is judged once, on the 2026-27 EPL events that kick off on or after a freeze date, by ADR-0012's two keep-bar conditions. This ADR names the season, the freeze date, what is frozen, the single look, the collection rules and the statistics that decide. Its choices are drafts: the [open choices](#open-choices-for-the-user) list what the user decides before it is accepted.

## Context

ADR-0012 keeps a model only if it passes both keep-bar conditions "on a fresh season", and leaves the season and its freeze date to pre-registration (#539, under the map #506).

- **Development results.** On 2024-25 and 2025-26, the residual arm (XGBoost on the target minus the baseline, with the baseline added back) gains 1.0 to 1.4% of the baseline's weighted MSE, with CIs that include zero. Its picks are level with the baseline's. The direct arm is worse than the baseline ([price-only](../../wiki/evaluation/exp-epl-retail-ev-price-only.md)). No as-of-t group beats price-only ([as-of-t groups](../../wiki/evaluation/exp-epl-retail-ev-as-of-t-groups.md)), and re-tuning inside 2025-26 does worse than the settings tuned on 2024-25 ([nested tuning](../../wiki/evaluation/exp-epl-retail-ev-nested-tuning.md)).
- **Power.** One season of 380 events gives the MSE condition 80% power at a gain of about 4.5 to 6%. The pick condition needs about 16 to 21% for a diffuse edge, or exact knowledge of the move on about 2 to 2.6% of events. Pick power below about 0.15 on FDUK-only rows is not evidence of an edge ([keep-bar power](../../wiki/evaluation/exp-retail-ev-power.md)).
- **2026-27 so far.** No model has been scored on it. The target report (#537) and an early version of the power analysis counted its rows descriptively. Its events before 2026-10-19 have FDUK opening and closing prices and live Betfair prices, but no live OddsPortal rows with bet365. Production has collected OddsPortal live with bet365 for EPL since 2026-10-19.
- **Code gaps.** The residual arm is a script transform in `experiments/scripts/retail_ev_price_only.py`; the training pipeline fits only the direct arm. [ADR-0014](0014-ignore-wide-exchange-prices.md) (the 5% exchange-overround cap) and [ADR-0013](0013-collect-polymarket-epl-as-sharp-reference.md) (Polymarket collection) are proposed.

## Decision

### Evaluation window and freeze

- **Freeze date:** 2026-11-02 00:00 UTC, or the first Monday after the prerequisites below are merged, whichever is later. The freeze is the git tag `retail-ev-eval-2026-27`.
- **Evaluation events:** EPL 2026-27 events whose kick-off is at or after the freeze date, through the last 2026-27 match. That is about 280 events if the freeze is on 2026-11-02 (inferred from the fixture count to mid-October). Kick-off means the event's actual kick-off, so a postponed match counts if it is played in the window.
- **2026-27 events before the freeze** are training data only. They are never scored.

### Prerequisites before the freeze

1. The residual arm is a pipeline option, so the frozen config can express it.
2. The evaluation script exists and is reviewed. It is run once on development data (events up to 2026-07-31) as a dry run and must reproduce the price-only residual result on [price-only](../../wiki/evaluation/exp-epl-retail-ev-price-only.md) within seed noise.
3. ADR-0014 is accepted or rejected, so the row rules are fixed.
4. The loss-weight scale is fixed (see [what is frozen](#what-is-frozen)).
5. This ADR is accepted.

### What is frozen

Everything the evaluation runs is the code and config at the freeze tag. The look runs that tag, not a later commit.

- **Target, rows and baseline:** ADR-0012 as built in `odds_analytics.retail_ev`, with ADR-0014's cap if accepted, and the FDUK opening-row rule on [CLV target](../../wiki/modelling/clv-target.md#the-retail-ev-target).
- **Model:** the residual arm on the 38 price-only columns of `odds_analytics.retail_ev.RetailEvFeatures`, including the book one-hot list at the tag. A book not in that list is scored with every book column at zero.
- **Hyperparameters:** the residual seed-42 best trial from 2024-25 tuning, as recorded in `experiments/results/retail_ev_as_of_t_groups/summary.json` (variant `price_only`): 100 trees, depth 4, learning rate 0.0101, `min_child_weight` 1982, subsample 0.6, colsample 0.7, λ 1.95. No further tuning.
- **Seed:** 42 for XGBoost's `random_state`. Two runs of the tagged script on the same data must give identical predictions.
- **Weight scale:** the loss weights are scaled so that the mean weight per event equals its development value, not so that the mean row weight is one. Today's scaling ties `min_child_weight` to rows per event, so dense 2026-27 events would loosen the regularisation the tuner chose (inferred). The value is recorded at the tag.
- **Training window and refits:** expanding walk-forward from 2024-08-01. Evaluation events are scored in blocks of 50 in kick-off order. Each block's model trains on every event that kicks off at least 7 days before the block's first event. The last block may be partial and is scored.

### One look

- The evaluation is run once, after the last 2026-27 match and once FDUK's final 2026-27 file is loaded.
- It runs on a local copy of production (`odds copy from-prod`), never against production. The report records the tag, `git_dirty`, the data hash of the scored rows and the count of rows and events by retail source.
- No interim result is computed. Collection health is watched through the existing alerts and heartbeats, which show no model output.

### Collection requirements and gaps

- **Expected collection:** live OddsPortal (with bet365) and live Betfair Exchange for every EPL event, at the cadences on [fetch-oddsportal](../../wiki/collection/fetch-oddsportal-job.md) and [Betfair Exchange](../../wiki/collection/betfair-exchange.md), and the FDUK 2026-27 file.
- **No exclusions for gaps.** Every evaluation event with a close and at least one row under the frozen rules is scored. Model and baseline are scored on the same rows, so a gap changes which rows exist, and with it power, but favours neither (inferred). The close falls back to FDUK as ADR-0012 sets out. An event with no row is not scored, and the report counts those events.
- **Coverage is reported, not gated:** the share of evaluation events with OddsPortal live rows, with a live Betfair close, and FDUK-only, plus the outages longer than a matchday.
- **Collection fixes are allowed** during the season. A change to the target, row, feature or model code is not. If the tagged code cannot read season data, a fix limited to reading may be applied on a branch from the tag, reviewed, and named in the report.

### Statistics that decide

Over all scored rows of the evaluation events:

1. **MSE condition:** `odds_analytics.retail_ev_metrics.mse_gain`, baseline weighted MSE minus the model's.
2. **Pick condition:** `odds_analytics.retail_ev_metrics.pick_ev_gain`, the model's total `w_event`-weighted realised EV over rows with predicted EV > 0, minus the baseline's.

Each has an event-clustered percentile bootstrap 95% CI with 10,000 resamples and seed 0. The model is **kept** if both CIs lie above zero, and **retired** otherwise. Requiring both conditions at 95% each keeps the overall false-keep rate at 5% or less, so no multiplicity correction applies (intersection–union test). Slices by horizon, retail source, event type, outcome and block are reported for description only and decide nothing.

## Open choices for the user

Each choice lists the options; the recommendation is the one drafted above.

1. **Season length.**
   - (a) Post-freeze 2026-27 only, about 280 events. *Recommended.* Minimum detectable MSE gain about 7% (inferred, interpolated between 190 and 380 events on [keep-bar power](../../wiki/evaluation/exp-retail-ev-power.md#minimum-detectable-effect-at-80-power)).
   - (b) Post-freeze 2026-27 and all of 2027-28, about 660 events. Minimum detectable MSE gain about 3%, pick about 13%, but the model stays frozen until mid-2028.
   - Why (a): the development gain is 1.0 to 1.4%, which neither design detects reliably, and (b) blocks any new model for 19 months. A challenger developed meanwhile can be pre-registered on 2027-28 in its own ADR. A fail under (a) rules out a gain above about 7%, not an edge.
   - Option (b) changes ADR-0012's "on a fresh season" and would supersede that line.
2. **Number of looks.**
   - (a) Once, at season end. *Recommended.*
   - (b) With two seasons, an interim look at the end of 2026-27 that keeps only if both 99.9% CIs exclude zero (Haybittle–Peto), and the final look at 95%.
3. **Freeze date and window start.**
   - (a) Freeze 2026-11-02 or after the prerequisites, scoring from the freeze. *Recommended.*
   - (b) Score from 2026-10-19, the start of live bet365 collection. It needs the prerequisites done before any scored event, which is not possible now.
   - (c) Score all of 2026-27. Its early events were counted descriptively (#537), have no live OddsPortal rows, and kicked off before anything was frozen.
4. **Pick condition.**
   - (a) Keep ADR-0012's condition 2 as written. *Recommended:* it is what the power analysis tested, and a change now is a choice made after seeing development data.
   - (b) Weight each pick's realised EV by `1/odds`, which puts the pick total in probability units and reduces the weight of longshots. Its power is untested, and it supersedes ADR-0012's condition 2 through a new ADR.
5. **Hyperparameters.**
   - (a) The frozen 2024-25 residual seed-42 trial. *Recommended:* no new choice, and nested re-tuning did worse on development data.
   - (b) One fresh tuning study on all development events before the freeze, with the #540 search space.
   - (c) Nested re-tuning at every refit.
6. **Refits.**
   - (a) Expanding refits every 50 evaluation events. *Recommended:* the model learns from dense 2026-27 rows as production would.
   - (b) One fit at the freeze, never refitted. Simpler, but no model has trained on dense live rows before scoring them.
7. **Feature set.**
   - (a) Price-only. *Recommended:* no group beat it on development data.
   - (b) Add a group, such as all four as-of-t groups. A Polymarket group needs ADR-0013 accepted and the live job running before the freeze.
   - (c) Several candidates at once. Each condition's CI then widens to 1 − 0.05/k for k candidates.
8. **Weight scale.** (a) Fix the per-event weight scale before the freeze. *Recommended.* (b) Keep mean-one row weights and accept that `min_child_weight` loosens as dense events enter training.
9. **ADR-0014.** (a) Accept it before the freeze. *Recommended:* the development baseline improves with the cap. (b) Reject it and evaluate uncapped. Either way it is settled before the freeze.

## Considered options

The alternatives are the options above that are not recommended. Two further ones:

- **Prospective logging of predictions at decision time.** It proves no look-ahead, but needs the model in a scheduled job before the freeze. The tagged script, run once on a copy, uses only data at or before each row's t by construction ([look-ahead](../../wiki/modelling/look-ahead-bias.md)), and FDUK opening rows already pair only with their own snapshot.
- **Excluding events with collection gaps.** It would make the scored events depend on outages, and gives the model no advantage to remove (inferred).

## Consequences

- Until the look, no change to the retail-EV target, rows, features or model code can affect this evaluation, because the look runs the tag. Development on 2026-27 data for another model is allowed, but that model cannot then use 2026-27 as its fresh season.
- A fail retires this model. It does not show the absence of an edge smaller than the minimum detectable effect.
- The prerequisites are code work that must be merged before the freeze date. If they slip, the freeze and window start move with them.
- If ADR-0014 is rejected, its constant must be removed or disabled at the tag.

## Evidence

- Issues #506, #537, #538, #539, #540, #545, #549.
- Wiki: [keep-bar power](../../wiki/evaluation/exp-retail-ev-power.md), [price-only](../../wiki/evaluation/exp-epl-retail-ev-price-only.md), [as-of-t groups](../../wiki/evaluation/exp-epl-retail-ev-as-of-t-groups.md), [nested tuning](../../wiki/evaluation/exp-epl-retail-ev-nested-tuning.md), [CLV target](../../wiki/modelling/clv-target.md#the-retail-ev-target).
- `experiments/results/retail_ev_as_of_t_groups/summary.json` for the frozen hyperparameters.
