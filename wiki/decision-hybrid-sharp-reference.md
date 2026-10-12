---
type: decision
title: Pinnacle with Betfair Exchange fallback as sharp reference
summary: Why sharp_bookmakers is [pinnacle, betfair_exchange] with first-available semantics.
sources:
  - docs/MODELING.md#hybrid-sharp-reference
  - docs/MODELING.md#current-conclusions
  - docs/DATA_MODELS.md#choosing-data-for-an-experiment
  - packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatureExtractor
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
  - experiments/configs/xgboost_epl_hybrid_sharp.yaml
verified_at: 7887039
---

# Pinnacle with Betfair Exchange fallback as sharp reference

**Decision (2026-03-19).** EPL training uses `sharp_bookmakers: [pinnacle, betfair_exchange]`; the first bookmaker in the list that has odds wins. No ADR exists; rationale is from `docs/MODELING.md`.

## Context

Pinnacle's public API stopped covering EPL events in January 2026, so 71 of about 1.8K events had no Pinnacle closing odds and were excluded ([sharp-benchmark](sharp-benchmark.md)). Betfair Exchange closing odds exist in football-data.co.uk from 2024-25 and as live snapshots from the Betfair job ([betfair-exchange-client](betfair-exchange-client.md)).

## Evidence

On 589 overlapping events Betfair Exchange `sharp_prob` correlates 0.9994 with Pinnacle, with a roughly 1.1 pp level shift that XGBoost absorbs through split points `(inferred)`. The hybrid recovers all 71 events (1,800 vs 1,729); folds 0-19 match Pinnacle-only and the two new folds score R² +7.4% and +2.7%; fixed-parameter R² 0.060 ± 0.080 ([exp-epl-sharp-reference](exp-epl-sharp-reference.md)).

## Consequences

- The semantics changed from all-of to any-of: `TierSampler.required_bookmakers`, the event filter in `prepare_training_data` and `TabularFeatureExtractor` all try bookmakers in priority order ([feature-pipeline](feature-pipeline.md)).
- The EPL production setup is `data_source: null`, `sharp_bookmakers: [pinnacle, betfair_exchange]`, `closing_source_priority: [football_data_uk]` ([training-config](training-config.md)).
- The agent's MCP screener defaults to the same list ([mcp-find-retail-edges](mcp-find-retail-edges.md)).

## Hazards

- A single training set mixes Pinnacle-based and Betfair-based `sharp_prob` values (and the model cannot tell which row is which except via the level shift).
- The code default is `["pinnacle"]`; the hybrid exists only where a config or MCP default sets it.
- The experiment config `experiments/configs/xgboost_epl_hybrid_sharp.yaml` predates the principled search space ([exp-principled-search-space](exp-principled-search-space.md)).

## Related

- [CLV model findings](clv-model-findings.md), [decision-bet365-target](decision-bet365-target.md)
