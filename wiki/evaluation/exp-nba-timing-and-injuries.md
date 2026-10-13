---
type: explanation
title: NBA timing and injury experiments
description: When NBA books post lines, how CLV and divergence vary with hours to the game (Exp 4), and why injury features add nothing once tuned equally (Exp 6, 6b).
sources:
  - experiments/results/exp4_hours_to_game/FINDINGS.md
  - experiments/results/exp6_learning_curve/FINDINGS.md
  - experiments/results/exp6b_injury_closing_tier/FINDINGS.md
  - odds_cli.commands.injuries.fetch
  - experiments/scripts/exp4_hours_to_game.py
  - experiments/configs/xgboost_bet365_injuries.yaml
  - experiments/configs/xgboost_bet365_tuning.yaml
  - experiments/configs/xgboost_bet365_baseline_tuning.yaml
  - odds_analytics.injury_features.extract_injury_features
  - odds_lambda.fetch_tier.FetchTier
  - odds_lambda.tier_utils.calculate_tier
  - odds_lambda.oddsportal_common.hours_to_tier
status: stable
---

# NBA timing and injury experiments

Three NBA studies about when to decide and what information is available then. All are NBA-only; EPL timing and injuries are untested ([decision time](../modelling/look-ahead-bias.md), [tiers](../storage/fetch-tier.md)).

- The sharp book posts lines 12-24 h before the game; soft books post days earlier ([line release](#bookmaker-line-release-timing)).
- The pregame window stays the default decision window ([sampling defaults](../modelling/training-config.md)); no evidence supports changing it ([Exp 4](#hours-to-game-exp-4)).
- Injury features add no predictive signal at any tier once both arms are tuned equally ([Exp 6, 6b](#nba-injury-features-exp-6-6b)).

## Bookmaker line release timing

Measured from Odds API NBA snapshots: when each bookmaker first appears relative to game time. The analysis date and the query behind it are not recorded and no results file holds the tables (they were added directly in 0cc694e), so the numbers cannot be regenerated from the repo and are unverified.

**Key finding.** The sharp book (Pinnacle) releases lines 12-24 hours before game time; soft books (DraftKings, FanDuel) release them 2-3+ days earlier.

First capture, hours before game:

| Bookmaker | Average | Median | Max |
|---|---|---|---|
| DraftKings | 64.4 | 24.0 | 2293 |
| Caesars | 41.2 | 23.1 | 1624 |
| BetMGM | 37.6 | 19.6 | 1696 |
| BetRivers | 35.1 | 13.0 | 1696 |
| Bovada | 35.0 | 12.0 | 1696 |
| FanDuel | 25.3 | 23.8 | 1696 |
| Pinnacle | 16.1 | 12.0 | 134 |

Bookmakers available by window:

| Hours before game | Average books | Meaning |
|---|---|---|
| 48+ | 2.7 | only soft books (DK, FD) |
| 24-48 | 3.5 | more books coming online |
| 12-24 | 4.3 | Pinnacle typically opens |
| under 12 | 6.9 | all books available |

Implications:

1. "Opening line" varies by bookmaker: DraftKings opens days out, Pinnacle about 12-24 h out.
2. The sharp opening line is Pinnacle's first line ([sharp benchmark](../modelling/sharp-benchmark.md)).
3. Most line movement occurs 12-24 h out, when sharp money enters via Pinnacle `(inferred)`.
4. Early soft-book lines are less predictive because they are adjusted when the sharp book opens `(inferred)`.
5. Opening-tier snapshots (72+ h) capture only soft-book lines; the sharp tier (12-24 h) is where Pinnacle first appears `(inferred)`; the measurement is an unrecorded NBA Odds API snapshot analysis. These are the code's tier bounds: `odds_lambda.fetch_tier.FetchTier.max_hours` gives OPENING above 72 h, EARLY up to 72 h, SHARP up to 24 h, PREGAME up to 12 h, CLOSING up to 3 h (`odds_lambda.fetch_tier.FetchTier`).

## Hours to game (Exp 4)

Question: when before the game should a decision be made? Exp 4 (2026-02-28, git `afab058`, #166) used 1,593 samples from 477 Odds API NBA events, 1-48 h before the game, Pinnacle target, 13 features (tabular 6 + injury 6 + `hours_until_event`). Script: `experiments/scripts/exp4_hours_to_game.py`.

| Hours before game | 0-3 | 3-6 | 6-9 | 9-12 | 12-18 | 18-24 | 24-36 |
|---|---|---|---|---|---|---|---|
| Samples | 165 | 454 | 419 | 275 | 40 | 200 | 39 |
| Target std | 0.029 | 0.022 | 0.035 | 0.040 | 0.027 | 0.035 | 0.038 |
| Mean abs CLV delta | 0.014 | 0.015 | 0.023 | 0.026 | 0.021 | 0.027 | 0.030 |

- Mean absolute CLV delta grows from 1.4 pp at 0-3 h to 3.0 pp at 24-36 h, so there is more room the earlier the decision; the 3-6 h bin has the lowest variance.
- `tab_retail_sharp_diff` peaks at 3-6 h (r = +0.097) and is weakest at 0-3 h (r = +0.030). Divergence peaks 3-6 h before NBA games.
- `inj_impact_gtd_away` is the strongest single correlation in the project at 0-3 h (r = +0.28, p = 0.0003), flipping negative at 18-24 h; it did not survive on bet365 (Exp 6b below).
- Probability level correlates negatively with CLV close to the game (reversion) and positively far from it.
- Per-bin XGBoost models were all R² < 0 (27-317 training samples per bin); the pooled model is -0.009, with 12-18 h and 18-24 h the only bins with MSE ratio below 1.

**Conclusion.** The default sampler window is the pregame tier, with no evidence to change it. The dataset is too small to name an optimal hour, and the 5K-event OddsPortal data (opening and closing snapshots only) cannot answer timing questions. The production model samples the `sharp` tier instead, and `odds_analytics.feature_groups.TierSampler` falls back to farther tiers when the requested one is absent: about 93% of "pregame" OddsPortal events used the sharp snapshot ([decision time](../modelling/look-ahead-bias.md)).

## NBA injury features (Exp 6, 6b)

Conclusion: injury features add no predictive signal for CLV at any decision tier once models are tuned the same way. The pipeline still collects the data (manual CLI only, no scheduled job: `odds_cli.commands.injuries.fetch`) and the feature group still exists, but the deployed model does not use it ([NBA injuries and stats](../collection/nba-injuries-and-stats.md), [feature groups](../modelling/feature-groups.md)).

- **First positive result** (2026-02-20, #160, 800 events, Pinnacle target, tabular 4 + injury 6 + rest 5 + timing 3 features, 100-trial Optuna): val R² 0.050, CV R² = 0.020 ± 0.025, injuries 55% of feature importance, 6 dead features.
- **bet365 at scale** (2026-02-27, #166, about 5K OddsPortal events, 11-fold walk-forward): tabular + injuries CV R² 0.036 ± 0.033 against tabular-only 0.036 ± 0.028 once both were tuned. The earlier signal came from comparing a tuned model with an untuned baseline. Configs: `experiments/configs/xgboost_bet365_tuning.yaml`, `experiments/configs/xgboost_bet365_baseline_tuning.yaml`, `experiments/configs/xgboost_bet365_injuries.yaml`.
- **Exp 6, timing diagnostic** (2026-03-01, #168, `experiments/results/exp6_learning_curve/FINDINGS.md`): sharp tier tabular R² +0.018 vs +0.006 with injuries; pregame +0.001 vs +0.0015. About 93% of "pregame" events fell back to the sharp snapshot, and OddsPortal has no 0-3 h snapshots ([target bookmaker and data volume](exp-target-and-data-volume.md)).
- **Exp 6b, closing tier** (2026-03-01, #169, Odds API, bet365 target, 479 closing and 826 sharp events): at closing both arms give R² 0.5959 identical to 16 decimals, a measurement artefact because the target is almost zero (std 0.0016 against 0.054 at the sharp tier; MSE 0.000005). At the sharp tier injuries slightly hurt (-0.031 vs -0.020). The Exp 4 game-time-decision correlation (`inj_impact_gtd_away` r = 0.28, Pinnacle target) does not transfer: r = -0.01 (p = 0.81) on bet365. At the sharp tier `tab_retail_sharp_diff` dominates (r = 0.356). Source: `experiments/results/exp6b_injury_closing_tier/FINDINGS.md`.

By the time game-time injury designations are public (0-3 h) the closing line has priced them, so the closing-tier target is near zero: a catch-22 for testing the hypothesis. The GTD hypothesis is falsified for bet365; the signal in this data is cross-sectional market state ([sharp benchmark](../modelling/sharp-benchmark.md)). The injury pipeline can be deprioritised.

## Caveats

- **Closing-tier R² near 0.6 is an artefact** and must not be quoted as skill.
- The injury weights come from season-level player stats (`odds_analytics.injury_features.extract_injury_features`) and the group is NBA-only; EPL has no equivalent.
- Snapshot coverage is bimodal (peaks at 3-4 h and 8-10 h, secondary cluster 22-24 h), so the 12-18 h and 24-36 h Exp 4 bins have only 40 and 39 samples.
- Exp 4 used 13 features including the injury group, which later runs found uninformative.
- **Tier boundaries are encoded more than once and disagree** at exactly 3, 12, 24 and 72 h: `odds_lambda.tier_utils.calculate_tier` uses `<=` while `odds_lambda.oddsportal_common.hours_to_tier` uses `<` ([fetch tier](../storage/fetch-tier.md#boundary-encodings), #446).
- Line-release numbers are NBA and Odds API only.

## Related

- [CLV model findings](clv-model-findings.md), [Early NBA signal search and LSTM](exp-early-nba-signal-and-lstm.md)
