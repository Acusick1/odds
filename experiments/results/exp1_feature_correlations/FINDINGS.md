# Experiment 1: Feature-Target Correlation Analysis

## Setup

- **Date**: 2026-02-17
- **Dataset**: 719 samples, 229 events, 75 features
- **Sampling**: multi-horizon time_range (3-12h, max 5/event), avg 3.1 rows/event
- **Target**: devigged Pinnacle CLV delta (mean=-0.001, std=0.038)
- **Method**: Pearson and Spearman correlations, BH and Bonferroni correction

## Key Results

Of 75 features, 15 are constant (zero variance — orderbook features not yet collected, plus `is_home/away_team` trivially constant for `outcome=home`). Of the 60 testable features, 12 pass uncorrected p < 0.05, but 0/60 survive BH (FDR) or Bonferroni correction. Correlations are weak but structured — they align with theoretical priors about sharp-retail divergence.

### Top 10 Features (by average |correlation|)

| Rank | Feature | Pearson r | Spearman ρ | Group |
|------|---------|-----------|------------|-------|
| 1 | tab_retail_sharp_diff_home | +0.120 | +0.120 | tabular |
| 2 | traj_max_prob_decrease | +0.101 | +0.131 | trajectory |
| 3 | tab_retail_sharp_diff_away | -0.103 | -0.119 | tabular |
| 4 | traj_sharp_retail_divergence_trend | +0.055 | +0.134 | trajectory |
| 5 | traj_prob_range | -0.068 | -0.116 | trajectory |
| 6 | traj_prob_volatility | -0.066 | -0.108 | trajectory |
| 7 | pm_pm_away_prob | +0.074 | +0.074 | polymarket |
| 8 | traj_trend_slope | +0.089 | +0.056 | trajectory |
| 9 | xsrc_pm_sb_divergence_direction | -0.079 | -0.050 | cross_source |
| 10 | tab_best_away_odds | -0.077 | -0.051 | tabular |

### By Feature Group

| Group | N | Mean |r| | Max |r| | Significant (p<0.05) |
|-------|---|---------|---------|----------------------|
| cross_source | 5 | 0.060 | 0.092 | 3 |
| tabular | 27 | 0.059 | 0.120 | 5 |
| trajectory | 23 | 0.037 | 0.101 | 3 |
| polymarket | 4 | 0.034 | 0.074 | 1 |
| timing | 1 | 0.014 | 0.014 | 0 |

### Target Distribution

- Skewness: 1.78, kurtosis: 12.8 (heavy right tail)
- Target std decreases closer to game: far=0.049, mid=0.045, close=0.025
- Markets become ~2x more efficient in the last few hours

### Correlation by Time Bin (top features)

| Feature | Far (>8h) | Mid (5-8h) | Close (3-5h) |
|---------|-----------|------------|---------------|
| tab_retail_sharp_diff_home | +0.056 | +0.152 | +0.088 |
| traj_max_prob_decrease | +0.175 | +0.108 | +0.069 |
| traj_sharp_retail_divergence_trend | +0.051 | +0.095 | +0.005 |

Sharp-retail diff peaks in mid window. Trajectory momentum strongest far from game.

### Multicollinearity

140 feature pairs with |r| > 0.8. Primary sources:
- Perfect duplicates from home/away + generic pattern (consensus_prob == home_consensus_prob when outcome=home)
- Line shopping features nearly redundant (best_home_odds ↔ best_available_odds: r=1.0)
- Trajectory volatility cluster (prob_volatility ↔ prob_range: r=0.97)

### Sparse Features

15/75 features are >50% zero:
- 11 PM orderbook features (100% zero) — orderbook data requires live polling, not yet deployed
- 3 cross-source features that depend on orderbook data (100% zero)
- traj_sharp_leads_retail (60% zero) — binary feature, sparsity expected

## Bug Found and Fixed

**21 tabular features were 100% zero due to a bug in `XGBoostAdapter.transform()`.** The adapter pre-filtered odds snapshots to only the target outcome (`outcome=home`) before passing them to `TabularFeatureExtractor`, which needs bilateral (home + away) data to compute consensus, sharp/retail divergence, and market hold features. Fix: remove the `outcome` parameter from `extract_odds_from_snapshot` calls for tabular and cross-source feature extraction.

Impact: sparse features dropped from 40/75 to 15/75. `tab_retail_sharp_diff_home` (previously zero) became the #1 correlated feature. Cross-source features partially activated (5/8 now non-zero).

### Outlier Robustness

Target winsorized at 1st/99th percentile (16 values clipped to [-0.089, 0.115]):

| Feature | Raw r | Winsorized r | Delta |
|---------|-------|-------------|-------|
| tab_retail_sharp_diff_home | +0.120 | +0.148 | +0.027 |
| traj_max_prob_decrease | +0.101 | +0.106 | +0.005 |
| tab_retail_sharp_diff_away | -0.103 | -0.135 | -0.032 |
| traj_sharp_retail_divergence_trend | +0.055 | +0.067 | +0.013 |

Correlations strengthen after removing outliers — the signal is not driven by extreme target values. The heavy tails (skewness 1.78) add noise rather than signal.

## Interpretation

The strongest signal is in **sharp-retail divergence** — both the point-in-time tabular version (r=0.12) and the trajectory trend version (ρ=0.13). This aligns with the theoretical prior: when retail books deviate from Pinnacle, the line corrects toward Pinnacle's price. The effect explains ~1.4% of target variance individually.

No feature individually explains enough to build a useful model. The question for subsequent experiments is whether the top 5-8 features combine constructively (they capture different aspects: current divergence, divergence trend, momentum) or are just noisy reflections of the same weak signal.

## Implications for Next Experiments

1. **Feature pruning required** — 140 collinear pairs mean ~40 features are redundant duplicates. Any model should use a deduplicated set.
2. **Sharp-retail feature group** (tab_retail_sharp_diff_home/away, traj_sharp_retail_divergence_trend, traj_max_prob_decrease) is the strongest candidate for a minimal model.
3. **PM features are underrepresented** — only pm_away_prob and 5 cross-source features have data. Orderbook features await live polling deployment.
4. **Mid time window** (5-8h) shows strongest sharp-retail correlation — may be the optimal decision window.

## Artifacts

- `correlations.csv` — full correlation table for all 75 features
- `correlations_bar.png` — bar chart of all Pearson/Spearman correlations
- `scatter_top6.png` — scatter plots for top 6 features vs target
- `target_distribution.png` — target histogram and QQ plot
- `intercorrelation_heatmap.png` — top 20 features inter-correlation matrix
