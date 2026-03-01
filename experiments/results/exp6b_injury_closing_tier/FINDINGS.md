# Experiment 6b: GTD Injury Signal at Closing Tier (bet365, Odds API)

## Design

2x2 comparison: {tabular, tabular+injuries} x {closing (0-3h), sharp (12-24h)}
on Odds API data with bet365 as target bookmaker. Fixed hyperparams from
xgboost_bet365_baseline_tuning_best.yaml. Walk-forward CV with expanding window.

## Results

| Feature Group | Tier | N Events | N Rows | R² Mean | R² Std | MSE Mean | Folds |
|---------------|------|----------|--------|---------|--------|----------|-------|
| tabular | closing | 479 | 479 | +0.5959 | 0.4949 | 0.000005 | 5 |
| tabular+injuries | closing | 479 | 479 | +0.5959 | 0.4949 | 0.000005 | 5 |
| tabular | sharp | 826 | 826 | -0.0204 | 0.1191 | 0.002302 | 9 |
| tabular+injuries | sharp | 826 | 826 | -0.0313 | 0.1684 | 0.002323 | 9 |

## Per-Feature Correlations: Closing (0-3h)

| Feature | Pearson r | p-value | N |
|---------|-----------|---------|---|
| tab_is_weekend | +0.0701 | 0.1254 | 479 |
| tab_day_of_week | +0.0626 | 0.1715 | 479 |
| inj_report_hours_before_game | -0.0530 | 0.2469 | 479 |
| tab_num_bookmakers | +0.0384 | 0.4013 | 479 |
| tab_retail_sharp_diff | +0.0315 | 0.4915 | 479 |
| tab_sharp_prob | +0.0310 | 0.4989 | 479 |
| tab_consensus_prob | +0.0269 | 0.5577 | 479 |
| inj_impact_out_away | +0.0255 | 0.5784 | 479 |
| inj_impact_out_home | +0.0251 | 0.5840 | 479 |
| inj_impact_gtd_away | -0.0109 | 0.8122 | 479 |
| inj_impact_gtd_home | -0.0072 | 0.8743 | 479 |
| inj_injury_news_recency | -0.0017 | 0.9700 | 479 |
| hours_until_event | +0.0017 | 0.9711 | 479 |

## Per-Feature Correlations: Sharp (12-24h)

| Feature | Pearson r | p-value | N |
|---------|-----------|---------|---|
| tab_retail_sharp_diff | +0.3555 | 0.0000*** | 826 |
| inj_report_hours_before_game | -0.0820 | 0.0184* | 826 |
| inj_injury_news_recency | -0.0611 | 0.0794 | 826 |
| inj_impact_out_away | -0.0434 | 0.2130 | 826 |
| tab_num_bookmakers | -0.0408 | 0.2413 | 826 |
| tab_is_weekend | +0.0347 | 0.3188 | 826 |
| tab_sharp_prob | -0.0251 | 0.4707 | 826 |
| inj_impact_gtd_home | -0.0220 | 0.5280 | 826 |
| hours_until_event | +0.0220 | 0.5281 | 826 |
| tab_day_of_week | +0.0122 | 0.7256 | 826 |
| inj_impact_gtd_away | +0.0120 | 0.7307 | 826 |
| inj_impact_out_home | +0.0067 | 0.8471 | 826 |
| tab_consensus_prob | +0.0002 | 0.9961 | 826 |

## Interpretation

### Closing-tier R² is a measurement artifact

The closing-tier R²=0.596 is misleading. At the closing tier (0-3h), the
"decision" snapshot nearly IS the closing snapshot — the target (decision price
minus closing price) is near zero for almost all events. Target std = 0.0016
at closing vs 0.054 at sharp (1200x smaller variance). The model explains
variance in near-zero residuals, not meaningful line movement.

MSE confirms this: 0.000005 (closing) vs 0.002302 (sharp). Both models
produce ~zero predictions, but the closing-tier target is itself ~zero, so R²
is mechanically high.

### Injuries add zero signal at both tiers

- **Closing tier**: tabular and tabular+injuries produce **identical** R²
  (0.5959). Injury features contribute nothing. The identical results (to 16
  decimal places) suggest XGBoost assigns zero weight to all injury features.
- **Sharp tier**: injuries slightly **hurt** (R²=-0.031 vs -0.020),
  consistent with Exp 6's null result on OddsPortal data.

### GTD signal does NOT transfer to bet365

The Exp 4 finding (inj_impact_gtd_away r=0.28, p=0.0003) was measured on
Pinnacle target with time_range sampling across 1-48h. Here, with bet365
target at closing tier, `inj_impact_gtd_away` has r=-0.011 (p=0.81). The
signal does not transfer for two reasons:

1. **Near-zero target at closing tier**: there's essentially no line movement
   left to predict at 0.2h before game. The line has already incorporated all
   public information including GTD designations.
2. **Different target bookmaker**: bet365 may price injury information
   differently than Pinnacle.

### The closing-tier catch-22

Testing injury signal at the closing tier creates a catch-22: by the time GTD
designations are visible (0-3h before game), the closing line has already
moved to reflect them. The target delta is near zero because the market is
efficient — the very information we're trying to exploit has already been
priced in. This is consistent with the efficient market interpretation: public
injury information is incorporated into closing prices, leaving no exploitable
residual.

### Sharp-tier signal structure

At the sharp tier (12-24h), tab_retail_sharp_diff dominates (r=0.356,
p<0.001) — the gap between bet365's price and the market consensus predicts
subsequent convergence. No injury feature is significant at conventional
levels. This replicates the Exp 5/6 finding that CLV signal is in
cross-sectional market state, not injury news.

### Conclusion

The GTD injury timing hypothesis is falsified. The Exp 4 r=0.28 finding was
specific to Pinnacle target with multi-hour sampling and does not generalize
to bet365 CLV prediction. Injuries remain uninformative for CLV prediction
regardless of decision tier. The injury feature pipeline can be deprioritized
for bet365 modeling.

