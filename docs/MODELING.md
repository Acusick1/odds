# CLV Model Findings

This page explains what the CLV prediction model can and cannot do, and the experimental
evidence behind its current configuration. The target, feature groups, CV settings and
config fields are defined in [DATA_MODELS.md](DATA_MODELS.md); this page covers why they
are set the way they are. The betting agent uses the model as a supplementary signal only
(see [BETTING_AGENT.md](BETTING_AGENT.md#role-of-the-xgboost-model)).

## What the model predicts

The model predicts line movement: the devigged bet365 CLV delta between a snapshot and
close (see [Training Target](DATA_MODELS.md#training-target)). A positive signal means
a mispriced line can be identified before it corrects.

The signal is weak but real. On EPL the best configuration reaches CV R² ≈ 0.058
([principled search re-test](#principled-search-space-and-protocol-re-test), 2026-04-23);
on NBA it plateaued at R² ≈ 0.036 ([bet365 at scale](#xgboost-bet365-at-scale), 2026-02-27).
It is not strong enough to bet on alone: on NBA, flat betting at bet365's ~4.5% vig loses
money at every threshold that produces a meaningful number of bets, even with line
shopping across four UK bookmakers ([Exp 7, 7b](#betting-simulation-and-line-shopping)).

## Deployed model

`epl-clv-home` version `v308058c`, published 2026-04-23: XGBoost on tabular + match_stats
features, sliding-380:50 walk-forward CV, the principled search space below, hybrid sharp
reference. CV val_R² = +0.058, val_MSE = 0.000716, replacing a tabular-only model (CV val_R² =
+0.035). Source config: `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`
(#362); the production scorer handles multi-feature-group models, including the standings,
match_stats and epl_schedule cache preload (#359). The live S3 version
is the authority; `odds model publish` writes it (see [CLI.md](CLI.md#publish-model-to-s3)).

## Current conclusions

Each conclusion links to the experiment that supports it in [Evidence](#evidence).

- **bet365 is the target, not Pinnacle.** Pinnacle CLV was unpredictable with public
  features for both XGBoost (CV R² = −0.017) and LSTM (CV R² = −0.075); Pinnacle lines move
  less, leaving 6.7× less target variance than bet365
  ([Pinnacle tuning](#pinnacle-as-target)).
- **The sharp reference is Pinnacle with Betfair Exchange fallback.** Betfair Exchange
  `sharp_prob` correlates 0.9994 with Pinnacle, and the fallback recovers the events
  Pinnacle stopped covering in January 2026 ([hybrid sharp](#hybrid-sharp-reference)).
- **Sharp-vs-retail divergence is the core feature.** It was the strongest raw correlate
  from the first experiment onwards ([Exp 1](#early-nba-experiments)), and the control with
  more data but no sharp reference showed no signal
  ([EPL with Pinnacle sharp](#epl-with-pinnacle-sharp-reference)).
- **For EPL, match_stats helps; other groups do not.** Adding match_stats gives the best
  individual improvement; standings is neutral, epl_schedule hurts slightly, rest produces
  no EPL features, and FPL availability degrades results
  ([EPL feature ablation](#epl-feature-group-ablation)).
- **Sliding-380:50 is the CV protocol.** The one-season sliding window gave the lowest MSE
  at validation steps of 10 and 50; a step of 50 was chosen so the tuner can tell feature
  sets apart. Once the search space stopped collapsing to constant predictors, it also edged
  expanding windows ([CV protocol](#cv-protocol-grid), [principled search](#principled-search-space-and-protocol-re-test)).
- **Injury features add nothing (NBA).** Once both variants are tuned the same way, the
  apparent injury signal disappears at every decision tier ([injuries](#injury-features-nba)).
  This has not been tested on EPL. See [INJURIES.md](INJURIES.md).
- **Sequence models add nothing.** LSTM variants never beat predicting the mean
  ([LSTM](#lstm-sequence-models)).
- **More historical data alone will not help (NBA).** The learning curve plateaus at ~1.5K
  events ([Exp 6](#data-volume-learning-curve)). EPL has ~1.8K events; a similar plateau is
  likely but not independently confirmed.
- **Pregame (3–12h) is the default decision window.** Sharp-retail divergence peaks 3–6h
  before the game on NBA ([Exp 4](#hours-to-game-effect)). Timing on EPL has not been tested.
- **Polymarket features are untested at scale.** They were only tried on 230 events, with
  no measurable effect ([Exp 2](#early-nba-experiments)). See [POLYMARKET.md](POLYMARKET.md).

## Guards against known failure modes

Three bugs invalidated earlier results. The pipeline now guards against each:

- **In-play snapshots as decision points.** `TierSampler` once offered `IN_PLAY` snapshots
  as pregame candidates, which leaks post-close information into features. They are
  excluded; the contaminated LSTM result (R² ≈ +0.02) became R² ≈ −0.015 once fixed
  ([Exp 2](#early-nba-experiments)).
- **Cross-source snapshot contamination (#231).** With combined OddsPortal + FDUK data,
  `TierSampler` picked the latest snapshot by wall-clock time, so OddsPortal snapshots
  without Pinnacle beat FDUK snapshots with Pinnacle for ~37 of 150 events in the final
  fold, and `np.nan_to_num` turned the missing Pinnacle probabilities into 0.0. `TierSampler`
  now takes `required_bookmakers` to filter candidates, and features keep NaN (XGBoost
  handles it natively). Results from 2026-03-10 are invalid.
- **Over-regularised search space.** The original Optuna search space let sliding-380:50
  converge on constant predictors. See
  [principled search](#principled-search-space-and-protocol-re-test) for the space in use.

## Evidence

Results as recorded at the time of each experiment. Where an experiment has a
`FINDINGS.md`, that file holds the full results.

### Data used

| | OddsPortal | Odds API | football-data.co.uk |
|--|-----------|----------|---------------------|
| **Events** | ~5K NBA (5 seasons) + ~1.8K EPL (5 seasons) | ~1K NBA (Mar–Apr 2025 + Oct 2025–Feb 2026) | ~4K EPL (11 seasons, 2015–2026) |
| **Snapshots/event** | 2 (opening + closing) | 15+ avg (all tiers) | 2 (opening + closing) |
| **Overlap** | 1,810 EPL events matched to FDUK | 141 NBA events matched to OddsPortal | 1,810 EPL events matched to OddsPortal |

football-data.co.uk is compiled manually (twice weekly, post-match). Its Pinnacle odds
come from Pinnacle's public API, which became unreliable after July 2025: 81 events in
2025-26 lack Pinnacle closing data. Betfair Exchange closing odds are available from
2024-25 onwards. Bookmaker sets and event ID patterns per source are in
[DATA_MODELS.md](DATA_MODELS.md#sportsbook-odds).

### Early NBA experiments

All February 2026, devigged Pinnacle target, 193–230 events.

- **XGBoost v1** (2026-02-14, 193 events, 75 features): R² ≈ 0. Likely causes at the time:
  too little data, 21 tabular features zeroed by a bug in `XGBoostAdapter`, order book gaps.
- **Exp 1, feature-target correlations** (2026-02-17, 229 events): max |r| = 0.12, from
  sharp-retail divergence; 0 of 75 features significant after Benjamini-Hochberg correction.
  [FINDINGS.md](../experiments/results/exp1_feature_correlations/FINDINGS.md)
- **Exp 2, feature group isolation** (2026-02-18, 230 events): every group, architecture
  and decision time gave R² < 0; adding Polymarket features made no difference. Found the
  `IN_PLAY` look-ahead bug.
  [FINDINGS.md](../experiments/results/exp2_feature_group_isolation/FINDINGS.md)

- **Polymarket vs sportsbook cross-source validation** (Feb 2026, 228 events): PM-SB
  probability correlation 0.9978, so both track the same market; PM prices sit ~2.4pp below
  SB (liquidity premium / vig difference); only 92 of 228 events had time-matched snapshots
  within 30 minutes, which limits cross-source features.

### Injury features (NBA)

- **XGBoost with injury + rest features** (2026-02-20, 800 events, Pinnacle target): the
  first positive result, CV R² = 0.020 ± 0.025, with injuries at 55% of feature importance.
- **bet365 at scale** (2026-02-27, ~5K OddsPortal events): tabular + injuries CV R² =
  0.036 ± 0.033 against tabular-only 0.036 ± 0.028 once both were tuned. The earlier
  injury signal came from comparing a tuned model with an untuned baseline.
- **Exp 6b, closing tier** (2026-03-01, 479 Odds API events): injuries add exactly zero at
  the closing tier; the Exp 4 GTD correlation (r = 0.28, Pinnacle) does not transfer to the
  bet365 target (r = −0.01).
  [FINDINGS.md](../experiments/results/exp6b_injury_closing_tier/FINDINGS.md)

### LSTM sequence models

| Date | Variant | Data | CV R² |
|------|---------|------|-------|
| 2026-02-21 | Sequence only (15 features × 24 timesteps) | 800 events, Pinnacle | −0.010 ± 0.039 |
| 2026-02-21 | Sequence + 17 static features | 800 events, Pinnacle | −0.122 ± 0.115 |
| 2026-02-28 | Sequence only, packed sequences, 50 trials | ~1K Odds API events, Pinnacle | −0.075 ± 0.113 |

Config: `experiments/configs/lstm_pinnacle_tuning.yaml`.

### XGBoost bet365 at scale

2026-02-27, ~5K OddsPortal NBA events, 11-fold walk-forward CV, 100 Optuna trials, devigged
bet365 target. Tabular-only CV R² = 0.036 ± 0.028; adding injuries gave the same result.
Configs: `experiments/configs/xgboost_bet365_tuning.yaml`,
`experiments/configs/xgboost_bet365_baseline_tuning.yaml`.

### Pinnacle as target

2026-03-01, ~800 Odds API NBA events, 4-fold walk-forward CV, 100 trials: CV R² =
−0.017 ± 0.015, with the tuner converging to maximum regularisation. Pinnacle MSE
(0.000431) is 6.7× lower than bet365 MSE (0.002883). Config:
`experiments/configs/xgboost_pinnacle_tuning.yaml`.

### Hours-to-game effect

Exp 4, 2026-02-28, 477 Odds API NBA events, 1,593 samples across 1–48h before the game.
Mean |CLV delta| grows from 1.4pp at 0–3h to 3.0pp at 24–36h. Sharp-retail divergence
peaks at 3–6h (r = +0.097). Per-bin models were all R² < 0 because the dataset is too small.
[FINDINGS.md](../experiments/results/exp4_hours_to_game/FINDINGS.md)

### Data volume learning curve

Exp 6, 2026-03-01, 500–4,524 OddsPortal NBA events, tuned tabular-only bet365 model. R²
plateaus at ~1.5K events and oscillates between 0 and +0.035 afterwards; dR²/dN =
6.7e-06 at N = 4,524.
[FINDINGS.md](../experiments/results/exp6_learning_curve/FINDINGS.md)

### Betting simulation and line shopping

- **Exp 7** (2026-03-01, ~4,500 OddsPortal NBA events, 12-fold expanding walk-forward,
  flat $100 bets on bet365): thresholds 0.005–0.03 all lose (−2.4% to −11.8% ROI). Threshold
  0.05 shows +3.36% ROI on 56 bets, not significant (p = 0.260). Average CLV captured is
  positive at every threshold, so the model finds direction but the vig consumes it.
  [FINDINGS.md](../experiments/results/exp7_backtest_sim/FINDINGS.md)
- **Exp 7b** (2026-03-01, same simulation, best price across bet365, Betway, Betfred, bwin):
  effective overround drops from 4.13% to 1.59% and ROI improves 0.3–1.3pp per threshold,
  but thresholds ≤ 0.03 still lose. bet365 already offers the best price 67% of the time.
  [FINDINGS.md](../experiments/results/exp7b_line_shopping/FINDINGS.md)

### EPL with Pinnacle sharp reference

March 2026, ~1.7K combined OddsPortal + FDUK EPL events, devigged bet365 target, Pinnacle
as sharp reference, 100 trials per run.

- Control with combined data but bet365 as its own sharp reference: CV R² = −0.005 ± 0.008.
  More data alone adds no signal.
- After the #231 fix (2026-03-15), expanding:150 protocol: tabular CV R² = 0.019 ± 0.043;
  tabular + standings + match_stats CV R² = 0.052 ± 0.045.
- The MSE drop against the OddsPortal-only baseline (0.0024 → 0.0007) reflects lower target
  variance with Pinnacle closing, not necessarily better prediction.

Sliding-380 results (val_step = 50, ~26 folds, 100 trials each, 2026-03-18):

| Experiment | Sharp Ref | Data | Features | CV MSE | CV R² |
|---|---|---|---|---|---|
| OddsPortal-only baseline | bet365 | OP only | 7 tabular | 0.002410 | — |
| Combined + Pinnacle sharp | pinnacle | OP + FDUK | 7 tabular | **0.000732** ± 0.000244 | **0.026** ± 0.060 |
| Combined + Pinnacle + standings | pinnacle | OP + FDUK | 18 (7 tab + 11 stnd) | **0.000726** ± 0.000210 | 0.023 ± 0.069 |
| Combined + Pinnacle + epl_schedule | pinnacle | OP + FDUK | 11 (7 tab + 4 sched) | 0.000736 ± 0.000244 | 0.023 ± 0.057 |
| Combined + Pinnacle + match_stats | pinnacle | OP + FDUK | 21 (7 tab + 14 mstat) | 0.000740 ± 0.000229 | 0.010 ± 0.063 |
| Combined + Pinnacle + standings + epl_schedule | pinnacle | OP + FDUK | 22 (7 tab + 11 stnd + 4 sched) | 0.000741 ± 0.000245 | 0.015 ± 0.064 |
| Combined + Pinnacle + standings + match_stats | pinnacle | OP + FDUK | 32 (7 tab + 11 stnd + 14 mstat) | 0.000759 ± 0.000243 | 0.004 ± 0.050 |
| Combined + Pinnacle + stnd + mstat + fpl_availability | pinnacle | OP + FDUK | 41 (32 + 9 fplav) | 0.000745 ± 0.000228 | 0.004 ± 0.050 |

Expanding-window results (val_step = 150, ~7 folds, 100 trials each):

| Experiment | Features | CV MSE | CV R² |
|---|---|---|---|
| Tabular only | 7 tabular | 0.000793 ± 0.000118 | 0.019 ± 0.043 |
| Tabular + standings + match_stats | 32 (7 tab + 11 stnd + 14 mstat) | **0.000767** ± 0.000113 | **0.052** ± 0.045 |

At the time, the disagreement was attributed to training-set size: sliding-380 gives ~12
events per feature with 32 features, while expanding windows grow past 1,000 events
(inferred). The [principled search re-test](#principled-search-space-and-protocol-re-test)
showed it was a search-space artefact instead, and these sliding-380 numbers do not
reproduce under current data and the hybrid sharp reference. Feature counts here are as
recorded; the current tabular group has 6 features.

Configs: `experiments/configs/xgboost_epl_combined_*_tuning.yaml`.

### CV protocol grid

2026-03-16, ~1.7K EPL events, tabular features, Pinnacle sharp. Script:
`experiments/scripts/cv_protocol_grid.py`; results:
`experiments/results/cv_protocol_grid/grid_results.csv`. CV MSE by validation step and
training window:

| val_step | expanding | sliding-95 | sliding-190 | sliding-380 | sliding-760 |
|----------|-----------|------------|-------------|-------------|-------------|
| **10** | 0.000728 | 0.000752 | 0.000749 | **0.000706** | 0.000789 |
| **50** | 0.000757 | 0.000748 | 0.000756 | **0.000732** | 0.000794 |
| **150** | 0.000793 | 0.000758 | 0.000742 | 0.000752 | 0.000758 |

- Sliding-380 (one season) is the best window: shorter windows starve the model and
  longer ones include stale data (inferred).
- val_step = 10 gives the lowest MSE but leads the tuner to pick identical hyperparameters
  for every feature set, so it cannot tell feature sets apart. val_step = 50 keeps the
  sliding-380 advantage while giving the tuner enough per-fold signal.
- A rerun with 32 features (2026-03-19) gave the same lowest-MSE cell (sliding-380:10,
  0.000693), but R² favoured expanding windows: expanding:50 R² = 0.061, expanding:150
  R² = 0.052, sliding-380:50 R² = −0.013.

### Hybrid sharp reference

2026-03-19, 1.8K EPL events. Pinnacle's API shutdown (January 2026) excluded 71 events.
Betfair Exchange `sharp_prob` correlates 0.9994 with Pinnacle on 589 overlapping events,
with a ~1.1pp level shift that XGBoost absorbs through split points (inferred).
`sharp_bookmakers: [pinnacle, betfair_exchange]` uses the first available bookmaker in the
list, recovering all 71 events; folds 0–19 match Pinnacle-only and the two new folds show
R² of +7.4% and +2.7%. Config: `experiments/configs/xgboost_epl_hybrid_sharp.yaml`.

### EPL feature group ablation

- **FPL availability** (2026-03-20, 41 features with hybrid sharp): CV R² = −0.017 ± 0.031
  under sliding-380:50 and 0.022 ± 0.037 under expanding:150, both below the baselines
  without it. The FPL tables and reader remain, but there is no `fpl_availability` feature
  group in the current training code, so this run cannot be reproduced as recorded.
- **Per-group ablation** (2026-04-23, 1,813 events through 2026-04-20, expanding:50, hybrid
  sharp, 100 trials each, original search space):

| Config | Features | val_R² | val_MSE | Δ R² vs tabular |
|---|---|---|---|---|
| tabular only | 7 | +0.0354 ± 0.0643 | 0.000765 | — |
| + rest | 7 (+0) | +0.0354 ± 0.0643 | 0.000765 | 0.000 (no EPL features) |
| + epl_schedule | 15 (+8) | +0.0277 ± 0.0564 | 0.000770 | −0.008 |
| + standings | 18 (+11) | +0.0367 ± 0.0657 | 0.000764 | +0.001 |
| **+ match_stats** | **21 (+14)** | **+0.0514 ± 0.0791** | **0.000751** | **+0.016** |
| all 5 groups | 40 | +0.0429 ± 0.0781 | 0.000757 | +0.008 |

Effect sizes are small relative to fold-to-fold standard deviation (±0.06–0.08).

An offline comparison on 2026-04-23 scored 17 upcoming EPL events with the tabular-only
and tabular + match_stats models (1,088 shared event-snapshot pairs). The models agreed on
direction for 86.9% of pairs; match_stats was on average 0.4pp more bearish on the home
side (mean Δ −0.0040, max |Δ| 0.0220). Without realised closing lines this shows the models
differ, not which is right.

### Principled search space and protocol re-test

2026-04-23. Under sliding-380:50 the original search space produced byte-identical results
for every feature set: every trial was a constant predictor (one unique prediction across 50
validation examples, early-stopped at iteration 9). A manual low-regularisation config
(`min_child_weight` = 5, `gamma` = 0) did not degenerate and beat the tuner's best by ~12%
MSE, so the search space, not the protocol, was at fault. The revised space,
drawn from published XGBoost tuning guidance for small-N, low-SNR regression:

```yaml
n_estimators:       [100, 600]  step 50
max_depth:          [2, 4]
learning_rate:      [0.01, 0.1]  log
min_child_weight:   [1, 15]
subsample:          [0.5, 1.0]  step 0.1
colsample_bytree:   [0.5, 1.0]  step 0.1
reg_lambda:         [0.01, 5.0]  log
gamma:              0.0 (locked)
reg_alpha:          0.0 (locked)
early_stopping_rounds: 50
```

The original space differed in `n_estimators` [50, 400], `max_depth` [2, 6],
`learning_rate` [0.01, 0.3], `min_child_weight` [5, 50], linear `reg_lambda` [0.5, 5.0],
tuned `gamma` [0, 1] and `reg_alpha` [0, 2], and `early_stopping_rounds` 15. Reasons given
for each change (inferred):

- `min_child_weight` of up to 50 forces each leaf to hold ≥13% of 380 training rows, which
  guarantees collapse; a heuristic of ≤ √N/10 suggests ≈2.
- `gamma` and `min_child_weight` are redundant regularisers at low depth, so locking
  `gamma` at 0 removes a joint over-regularisation trap.
- Tuning both L1 and L2 at once is avoided, so `reg_alpha` is locked.
- Log-scale `reg_lambda` from 0.01 lets the tuner find lighter L2.
- With learning rates ≤ 0.05 and noisy 50-example validation sets, improvement rarely
  shows before tree 20, so early stopping at 15 aborted before any tree could help. This
  was the largest single fix.

Results, 100 trials each, 1,813 events, hybrid sharp:

| Config | Original search val_R² | Principled val_R² | Principled val_MSE |
|---|---|---|---|
| expanding:50 + tabular | +0.0354 | +0.0472 | 0.000753 |
| expanding:50 + tabular + match_stats | +0.0514 | +0.0579 | 0.000743 |
| sliding-380:50 + tabular | constant predictor | +0.0517 | 0.000721 |
| sliding-380:50 + tabular + match_stats | constant predictor | **+0.0581** | **0.000716** |

Under the principled search, sliding-380:50 produces 48–50 unique predictions on 50
validation examples (best iteration 31–33, against 9 under the original space) and slightly beats expanding:50 on MSE. The match_stats over
tabular-only ranking holds under both protocols.

## Experiment log

| Date | Experiment | Features | Target | Samples | Result | Decision | Notes |
|------|-----------|----------|--------|---------|--------|----------|-------|
| 2026-02-14 | XGBoost v1 | tabular + trajectory + PM + cross-source | devigged pinnacle | 656 (193 events) | R² ≈ 0 | — | Multi-horizon, group CV; 21 tab features zeroed (bug) |
| 2026-02-17 | Exp 1: correlations | 60 testable / 75 total | devigged pinnacle | 719 (229 events) | max \|r\|=0.12 | Proceed to Exp 2 | Sharp-retail diff strongest; 12/60 uncorrected, 0/60 BH |
| 2026-02-18 | Exp 2: feature groups | 47 features, 6 groups | devigged pinnacle | 538–719 (230 events) | All R²<0 | No signal at 230 events | 2×2 (arch × time): all cells R²≈0; TierSampler IN_PLAY bug fixed |
| 2026-02-20 | XGBoost + injuries/rest | tabular 4 + injury 6 + rest 5 + timing 3 | devigged pinnacle | 800 events | val R²=0.050, CV R²=0.020±0.025 | First positive signal | 100-trial Optuna; injuries 55% importance; 6 dead features |
| 2026-02-21 | LSTM seq-only | 15 seq features × 24 timesteps | devigged pinnacle | 800 events | CV R²=-0.010±0.039 | No signal | 100-trial Optuna; best: hidden=48, layers=2 |
| 2026-02-21 | LSTM + static branch | 15 seq + 17 static (tab+inj+rest) | devigged pinnacle | 800 events | CV R²=-0.122±0.115 | Worse than seq-only | Static features increase overfitting; XGBoost remains best |
| 2026-02-27 | XGBoost bet365 tuned | tabular 4 + injury 6 | devigged bet365 | ~5K events (OddsPortal) | CV R²=0.036±0.033 | Plateau at ~3.6% | 11-fold walk-forward; injuries add zero over tabular-only |
| 2026-02-27 | XGBoost bet365 baseline tuned | tabular 4 | devigged bet365 | ~5K events (OddsPortal) | CV R²=0.036±0.028 | Same as +injuries | Confirms injuries are noise; public features plateau |
| 2026-02-27 | LSTM mask fix | 15 seq features × 8 timesteps | devigged pinnacle | ~1K events (Odds API) | — | Bug fix | Packed sequences for correct mask application (#162) |
| 2026-02-28 | LSTM Pinnacle tuned | 15 seq × 8 timesteps | devigged pinnacle | ~1K events (Odds API) | CV R²=-0.075±0.113 | LSTM ruled out | 50-trial Optuna, 5-fold walk-forward; packed sequences; worse than constant predictor |
| 2026-02-28 | Exp 4: hours-to-game | tabular 6 + injury 6 | devigged pinnacle | 1,593 (477 events, Odds API) | All bins R²<0; sharp-retail peaks 3-6h | Pregame window confirmed | CLV delta grows with hours; GTD injury r=0.28 at 0-3h; dataset too small for per-bin models |
| 2026-03-01 | XGBoost Pinnacle tuned | tabular 4 + injury 6 | devigged pinnacle | ~800 events (Odds API) | CV R²=-0.017±0.015 | No signal | 100-trial walk-forward; max regularization; Pinnacle CLV unpredictable with public features |
| 2026-03-01 | Exp 6: learning curve | tabular 4 | devigged bet365 | 500–4,524 events (OddsPortal) | Plateau at ~1.5K; R²≈0.02 | More data won't help | Log-fit dR²/dN=6.7e-06; injuries add nothing at sharp; pregame tier diluted (93% fallback to sharp) |
| 2026-03-01 | Exp 6b: injury closing tier | tabular 4 ± injury 6 | devigged bet365 | 479 (closing) / 826 (sharp) events (Odds API) | Closing R²=0.596 (artifact); sharp R²=-0.02 | GTD hypothesis falsified | Closing-tier target near-zero (std=0.0016); injuries add zero at both tiers; inj_impact_gtd_away r=-0.01 (not r=0.28 from Exp 4) |
| 2026-03-01 | Exp 7: backtest sim | tabular 4 | devigged bet365 | 2,088 bets (4,524 events OddsPortal) | Best ROI +3.36% (p=0.260) | Not profitable | Flat $100, 12-fold walk-forward; all thresholds ≤0.03 negative ROI; CLV signal too weak to overcome ~4.5% vig |
| 2026-03-01 | Exp 7b: line shopping | tabular 4 | devigged bet365 | 2,088 bets (4 bookmakers) | Best ROI +3.68% (p=0.334) | Still not profitable | Vig drops 4.13%→1.59% (2.55pp); ROI +0.3–1.3pp at all thresholds; bet365 best price 67% of time |
| 2026-03-10 | EPL combined + Pinnacle sharp | tabular 7, pinnacle sharp | devigged bet365 | ~1.8K EPL (OP+FDUK) | CV R²=0.031±0.014 (invalid) | Superseded | Inflated by the #231 bug (snapshot contamination + nan_to_num); see the 2026-03-15 rows |
| 2026-03-10 | EPL combined + bet365 sharp (control) | tabular 7, bet365 sharp | devigged bet365 | ~1.8K EPL (OP+FDUK) | CV R²=-0.005±0.008 | More data ≠ signal | Control proves improvement from Pinnacle features, not data volume |
| 2026-03-15 | #231 bug fix | — | — | — | — | Bug fix | Cross-source snapshot contamination: TierSampler picked OP (no Pinnacle) over FDUK; nan_to_num filled NaN→0.0. Fix: required_bookmakers filter + remove nan_to_num |
| 2026-03-15 | EPL combined + Pinnacle (post-fix) | tabular 7, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | CV R²=0.019±0.043 | Trustworthy baseline | 6-fold walk-forward (expanding:150), 100 trials; post-fix clean results |
| 2026-03-15 | EPL combined + standings | tabular 7 + standings 11, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | CV R²=0.030±0.040 | +1% R² from standings | 6-fold walk-forward (expanding:150), 100 trials |
| 2026-03-16 | EPL combined + match_stats | tabular 7 + match_stats 14, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | CV R²=0.025±0.042 | +0.6% R² over tabular | 5-fold walk-forward (expanding:150), 100 trials |
| 2026-03-16 | EPL combined + standings + match_stats | tabular 7 + standings 11 + match_stats 14, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | CV R²=0.052±0.045 | Best under expanding protocol | 5-fold walk-forward (expanding:150), 100 trials; reproduced 2026-03-19 (R²=0.052) |
| 2026-03-16 | CV protocol grid search | tabular 7, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | Sliding-380 best window; val_step=50 balances signal | Adopt sliding-380:50 | 15-cell grid (5 windows × 3 val steps); sliding-380:10 lowest MSE (0.000706) but val_step=10 forces identical hyperparams across feature sets; val_step=50 (MSE=0.000732) preserves discrimination |
| 2026-03-18 | EPL feature group re-run (all groups) | 7 configs, pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | Tabular R²=0.026±0.060 (best under sliding-380) | Best under sliding-380 | Sliding-380, val_step=50, ~26 folds, 100 trials each. Extra features degrade R² under this protocol. Verification artifacts saved for all runs. |
| 2026-03-19 | CV protocol grid: all-features | 32 features (tab+stnd+mstat), pinnacle sharp | devigged bet365 | ~1.7K EPL (OP+FDUK) | expanding:50 R²=0.061; expanding:150 R²=0.052; sliding-380:50 R²=-0.013 | Protocol determines feature value | 15-cell grid with 32 features. Expanding windows favour more features (R²=5-6%); sliding-380 penalises them. Lowest MSE still sliding-380:10 (0.000693). Production retraining strategy is the deciding factor. |
| 2026-03-19 | Hybrid sharp: Pinnacle + BFE | 32 features, hybrid sharp [pinnacle, betfair_exchange] | devigged bet365 | 1.8K EPL (OP+FDUK) | R²=0.060±0.080 (fixed params, expanding:50) | Adopt hybrid sharp | BFE sharp_prob correlates 0.9994 with Pinnacle. Hybrid recovers 71 post-shutdown events (1800 vs 1729). Folds 0-19 identical to Pinnacle-only; extra folds 20-21 show R²=+7.4%, +2.7%. Code change: sharp_bookmakers uses priority-ordered fallback (ANY, not ALL). |
| 2026-03-20 | FPL availability (sliding-380:50) | tabular 7 + standings 11 + match_stats 14 + fpl_availability 9, hybrid sharp [pinnacle, betfair_exchange] | devigged bet365 | 1.8K EPL (OP+FDUK) | CV R²=-0.017±0.031 | FPL adds noise | 28-fold walk-forward, 100 trials. Adding FPL availability (41 features total) degrades from tabular-only baseline (R²=0.026). Market already prices in squad availability. |
| 2026-03-20 | FPL availability (expanding:150) | tabular 7 + standings 11 + match_stats 14 + fpl_availability 9, hybrid sharp [pinnacle, betfair_exchange] | devigged bet365 | 1.8K EPL (OP+FDUK) | CV R²=0.022±0.037 | FPL degrades expanding baseline | 9-fold walk-forward, 100 trials. Positive R² but below expanding:150 all-features baseline (R²=0.052 with 32 features). FPL's 9 extra features (41 total) halve the R². |
| 2026-04-23 | EPL ablation re-run (expanding:50, hybrid sharp) | per-group: tabular, +rest, +epl_schedule, +standings, +match_stats, all-5 | devigged bet365 | 1.8K EPL (OP+FDUK) | tabular R²=0.0354; +match_stats R²=0.0514 (best); +standings R²=0.0367 (≈neutral); +epl_schedule R²=0.0277; +rest R²=0.0354 (no-op); all-5 R²=0.0429 | match_stats is best individual; rest no-op for EPL; all-5 worse than tabular+match_stats | 22-fold walk-forward (expanding:50, val_step=50), 100 trials each, 1813 events through 2026-04-20. `rest` extractor produces 0 features for EPL. Effect sizes small vs std (±0.06–0.08). |
| 2026-04-23 | sliding-380:50 ablation (degenerate) | same per-group set | devigged bet365 | 1.8K EPL | All configs converged to same trial 86, identical val_MSE to 16 decimals (constant predictor verified directly: 50 val examples → 1 unique prediction) | Sliding-380 + standard Optuna search degenerates | Standard search space (mcw∈[5,50], gamma∈[0,1], reg_λ∈[0.5,5]) + early_stopping=15 + 380 train events makes "predict-near-mean" the lowest-MSE model. Manual low-reg config (mcw=5, gamma=0) does not degenerate and beats Optuna best by ~12% MSE. The protocol isn't broken; the standard tuning protocol applied to it is. |
| 2026-04-23 | Principled search re-test | tabular ± match_stats, hybrid sharp | devigged bet365 | 1,813 EPL | sliding-380:50 + match_stats val_R²=0.0581, MSE=0.000716 (best) | Deploy as `epl-clv-home v308058c` | Revised search space; sliding-380:50 not degenerate under it; match_stats > tabular under both protocols |
