---
type: concept
title: Sharp benchmark and retail-sharp divergence
description: The sharp reference price and sharp_bookmakers priority semantics, the retail-minus-sharp divergence that is the model's strongest feature and the agent's screener, and the three code copies whose defaults differ.
sources:
  - packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatureExtractor
  - packages/odds-analytics/odds_analytics/feature_extraction.py::calculate_sharp_retail_metrics
  - packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatureExtractor
  - packages/odds-analytics/odds_analytics/feature_extraction.py::DEFAULT_RETAIL_BOOKMAKERS
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
  - packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-mcp/odds_mcp/server.py::find_retail_edges
  - packages/odds-mcp/agents/common.md
  - packages/odds-core/odds_core/odds_math.py::calculate_implied_probability
status: stable
---

# Sharp benchmark and retail-sharp divergence

The sharp benchmark is the price treated as closest to fair: Pinnacle, with Betfair Exchange as the fallback after Pinnacle's API stopped covering EPL in January 2026 (71 events lost; Betfair `sharp_prob` correlates 0.9994 with Pinnacle, about a 1.1 pp level shift). Rationale and evidence: [hybrid sharp reference](decision-hybrid-sharp-reference.md). FDUK's Pinnacle odds are also unreliable after July 2025 ([sources and keys](../collection/sources-and-keys.md)).

It anchors the model's strongest feature, the retail-sharp divergence, and the agent's retail screener ([MCP market tools](../agent/mcp-market-tools.md)). Because the agent sees the same quantity directly through `packages/odds-mcp/odds_mcp/server.py::find_retail_edges`, the model adds little for it ([betting agent](../agent/betting-agent.md)).

## Priority semantics

`sharp_bookmakers` is an ordered list and the first bookmaker that has odds in the market wins (`packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatureExtractor`):

- Features: `packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatureExtractor` tries each sharp bookmaker in order and stops at the first with odds in the market; `packages/odds-analytics/odds_analytics/feature_extraction.py::calculate_sharp_retail_metrics` does the same.
- Sampling: `packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler` keeps only snapshots containing any (not all) of `required_bookmakers`, and `packages/odds-analytics/odds_analytics/feature_groups.py::prepare_training_data` drops events lacking every sharp bookmaker at close. This filter applies only when `set(sharp_bookmakers) != {target_bookmaker}` ([feature pipeline](feature-pipeline.md)).
- MCP: `packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader.get_sharp_prices` scans snapshots newest-first within a 2 h lookback and resolves the priority order per outcome ([reader](../storage/odds-reader-and-storage-layer.md)); `packages/odds-mcp/odds_mcp/server.py::find_retail_edges` defaults to `[pinnacle, betfair_exchange]`.

## Why a sharp reference

- The model's signal needs it: combined EPL data with bet365 as its own sharp reference gave no signal (CV R² -0.005; PR #217) ([hybrid sharp reference](decision-hybrid-sharp-reference.md)).
- Pinnacle first posts NBA lines 12-24 h before the game `(inferred)`; the measurement behind it is an Odds API snapshot analysis whose date and query were not recorded, so re-check against `odds_snapshots`. Opening-tier snapshots (72+ h) hold only soft-book lines ([NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md), [fetch tier](../storage/fetch-tier.md)).
- The agent prompt treats sharp prices as strong but imperfect, and the model as supplementary sanity checks (`packages/odds-mcp/agents/common.md`); why the prompt says so is `(inferred)`.

## Retail-sharp divergence

**Definition.** `divergence = retail_implied_prob - sharp_implied_prob` for one outcome. Negative means the retail book prices that outcome longer than the sharp reference. Implied probabilities come from `packages/odds-core/odds_core/odds_math.py::calculate_implied_probability` and are not devigged ([odds arithmetic and devigging](../foundations/odds-math-and-devigging.md)); the target is devigged ([CLV target](clv-target.md)).

It is the strongest raw correlate of CLV from the first experiment onward (r = 0.12 in Exp 1, `experiments/results/exp1_feature_correlations/FINDINGS.md`; r = 0.356 at the sharp tier in Exp 6b, `experiments/results/exp6b_injury_closing_tier/FINDINGS.md`) and the core model feature `retail_sharp_diff` ([early NBA signal](../evaluation/exp-early-nba-signal-and-lstm.md), [NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md)). The production YAML also lists bet365, the target bookmaker, in `retail_bookmakers`, so the target bookmaker's snapshot price is an input to `retail_sharp_diff` ([bet365 target](decision-bet365-target.md)).

### Code copies

| | Feature, tabular path | Feature, helper path | MCP screener |
|---|---|---|---|
| Where | `packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatureExtractor` | `packages/odds-analytics/odds_analytics/feature_extraction.py::calculate_sharp_retail_metrics`, called by `packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatureExtractor` | `packages/odds-mcp/odds_mcp/server.py::find_retail_edges`, sharp from `OddsReader.get_sharp_prices` |
| Retail side | Mean of per-book implied probabilities for the outcome, over configured `retail_bookmakers` | Implied probability of the mean American price | One row per book, every non-sharp book in the window |
| Sharp side | First bookmaker in the list with any odds in the market | Same | First bookmaker per outcome, so outcomes can resolve to different books |
| Defaults | `["pinnacle"]`; retail `fanduel`, `draftkings`, `betmgm` (`DEFAULT_RETAIL_BOOKMAKERS`) | Same | `["pinnacle", "betfair_exchange"]`; no retail list |
| Time | Snapshot at decision time | Same | Newest price per book within a lookback window (default 2 h) |

Two features with the same name can therefore differ: averaging probabilities and taking the probability of the averaged price differ for books on opposite sides of even money. The training defaults are Pinnacle-only; hybrid production configs set `sharp_bookmakers` explicitly ([training pipeline](training-pipeline.md)).

## Hazards

- **The sharp roster is defined in several places.** `FeatureConfig.sharp_bookmakers` and `DEFAULT_SHARP_BOOKMAKERS` default to `["pinnacle"]`, the strategies default to `pinnacle`, and `_DEFAULT_SHARP_BOOKMAKERS` in `packages/odds-mcp/odds_mcp/server.py` is `["pinnacle", "betfair_exchange"]` with a comment saying to keep it in sync; the two lists differ on purpose (AUD T3/A13, re-verified). `packages/odds-mcp/odds_mcp/server.py::save_match_brief` stamps the hybrid list literal ([match briefs](../agent/match-briefs-and-paper-trades.md)). A training YAML that omits `betfair_exchange` silently loses the fallback.
- **No fall-through within a market.** The tabular extractor picks the first sharp bookmaker that has the market; if that bookmaker lacks the requested outcome, `sharp_prob` stays NaN rather than trying the next bookmaker.
- **Vig is included on both sides.** `sharp_prob` and `retail_sharp_diff` use raw implied probabilities. A retail book with a larger margin shows a more positive divergence on every outcome, so compare with `market_hold` in the screener output to separate margin from divergence.
- **MCP can mix bookmakers.** Per-outcome resolution across snapshots can return Pinnacle for one outcome and Betfair for another.
- **Point-bucket mismatch.** The screener buckets by `(outcome, point)` but looks up the sharp price by outcome name only, so on totals and spreads a sharp price for one line can be compared with retail prices at another.
- **Not co-temporal.** Sharp and retail prices can come from different snapshots.

## Related

- [Feature pipeline](feature-pipeline.md) — the tabular group that consumes it
- [CLV model findings](../evaluation/clv-model-findings.md)
- [Decision time and look-ahead bias](decision-time-and-look-ahead.md) — the feature uses decision-time snapshots
