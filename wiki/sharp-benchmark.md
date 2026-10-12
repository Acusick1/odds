---
type: concept
title: Sharp benchmark
summary: The sharp reference price (Pinnacle, with Betfair Exchange as fallback), sharp_bookmakers priority semantics, and how features, the agent and MCP use it.
sources:
  - packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatureExtractor
  - packages/odds-analytics/odds_analytics/feature_extraction.py::calculate_sharp_retail_metrics
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
  - packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-mcp/odds_mcp/server.py::find_retail_edges
  - docs/DATA_MODELS.md#target-and-features-trainingfeatures
  - docs/MODELING.md#current-conclusions
  - docs/BETTING_AGENT.md#design-principles
  - docs/BOOKMAKER_LINE_RELEASE.md#implications
verified_at: 7887039
---

# Sharp benchmark

The sharp benchmark is the price treated as closest to fair: Pinnacle, with Betfair Exchange as the fallback after Pinnacle's API shutdown ([decision-hybrid-sharp-reference](decision-hybrid-sharp-reference.md)). It anchors the model's strongest feature ([retail-sharp-divergence](retail-sharp-divergence.md)) and the agent's retail screener ([mcp-find-retail-edges](mcp-find-retail-edges.md)).

## Priority semantics

`sharp_bookmakers` is an ordered list and the first bookmaker that has odds in the market wins ([docs/DATA_MODELS.md](../docs/DATA_MODELS.md#target-and-features-trainingfeatures)):

- Features: `packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatureExtractor` tries each sharp bookmaker in order and stops at the first with odds in the market; `calculate_sharp_retail_metrics` does the same.
- Sampling: `packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler` keeps only snapshots containing any (not all) of `required_bookmakers`, and `prepare_training_data` drops events lacking every sharp bookmaker at close. This filter applies only when `set(sharp_bookmakers) != {target_bookmaker}`.
- MCP: `packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader.get_sharp_prices` scans snapshots newest-first within a 2 h lookback and resolves the priority order per outcome; `find_retail_edges` defaults to `[pinnacle, betfair_exchange]`.

## Why a sharp reference

- The model's signal needs it: combined EPL data with bet365 as its own sharp reference gave no signal ([exp-epl-sharp-reference](exp-epl-sharp-reference.md)).
- Pinnacle first posts NBA lines 12-24 h before the game, so opening-tier snapshots (72+ h) hold only soft-book lines ([exp-bookmaker-line-release](exp-bookmaker-line-release.md), [fetch-tier](fetch-tier.md)).
- The agent treats the sharp price as "a strong reference, not gospel"; the design principles carry no decision record, so that rationale is `(inferred)` ([docs/BETTING_AGENT.md](../docs/BETTING_AGENT.md#design-principles)).

## Hazards

- **The roster is defined in several places.** `FeatureConfig.sharp_bookmakers` and `DEFAULT_SHARP_BOOKMAKERS` default to `["pinnacle"]`, the strategies default to `pinnacle`, and `_DEFAULT_SHARP_BOOKMAKERS` in `packages/odds-mcp/odds_mcp/server.py` is `["pinnacle", "betfair_exchange"]` with a comment saying to keep it in sync (AUD T3/A13, re-verified). A training YAML that omits `betfair_exchange` silently loses the fallback.
- **No fall-through within a market.** The tabular extractor picks the first sharp bookmaker that has the market; if that bookmaker lacks the requested outcome, `sharp_prob` stays NaN rather than trying the next bookmaker.
- **Feature prices are vig-inclusive.** `sharp_prob` and `retail_sharp_diff` use raw implied probabilities, not devigged ones; the target is devigged ([clv](clv.md), [devigging](devigging.md)).
- **MCP can mix bookmakers.** Per-outcome resolution across snapshots can return Pinnacle for one outcome and Betfair for another.

## Related

- [Feature groups](feature-groups.md) — the tabular group that consumes it
- [Betting agent](betting-agent.md) — the agent that uses sharp prices
