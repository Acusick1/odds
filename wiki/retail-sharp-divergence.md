---
type: concept
title: Retail-sharp divergence
summary: "Retail implied probability minus sharp implied probability, the model's strongest feature and the basis of find_retail_edges, with every code copy and how they differ."
sources:
  - packages/odds-analytics/odds_analytics/feature_extraction.py::calculate_sharp_retail_metrics
  - packages/odds-analytics/odds_analytics/feature_extraction.py::TabularFeatureExtractor
  - packages/odds-analytics/odds_analytics/feature_extraction.py::SequenceFeatureExtractor
  - packages/odds-analytics/odds_analytics/feature_extraction.py::DEFAULT_RETAIL_BOOKMAKERS
  - packages/odds-mcp/odds_mcp/server.py::find_retail_edges
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-core/odds_core/odds_math.py::calculate_implied_probability
  - docs/MODELING.md#current-conclusions
  - docs/BETTING_AGENT.md#role-of-the-xgboost-model
  - docs/DATA_MODELS.md#feature-groups
verified_at: 7887039
---

# Retail-sharp divergence

**Definition.** `divergence = retail_implied_prob - sharp_implied_prob` for one outcome. Negative means the retail book prices that outcome longer than the sharp reference. Implied probabilities come from `packages/odds-core/odds_core/odds_math.py::calculate_implied_probability` and are not devigged ([Devigging](devigging.md), [Odds math](odds-math.md)).

It is the strongest raw correlate of CLV from the first experiment onward and the core model feature `retail_sharp_diff` ([MODELING.md](../docs/MODELING.md#current-conclusions)). The agent sees the same quantity directly through [find_retail_edges](mcp-find-retail-edges.md), which is why the model adds little for it ([Betting agent](betting-agent.md)). The sharp side is defined in [Sharp benchmark](sharp-benchmark.md).

## The three code copies

| | Feature, tabular path | Feature, helper path | MCP screener |
|---|---|---|---|
| Where | `TabularFeatureExtractor` in `feature_extraction.py` | `calculate_sharp_retail_metrics`, called by `SequenceFeatureExtractor` | `find_retail_edges` in `server.py`, sharp from `OddsReader.get_sharp_prices` |
| Retail side | Mean of per-book implied probabilities for the outcome, over configured `retail_bookmakers` | Implied probability of the mean American price | One row per book, every non-sharp book in the window |
| Sharp side | First bookmaker in the list with any odds in the market | Same | First bookmaker per outcome, so outcomes can resolve to different books |
| Defaults | `["pinnacle"]`; retail `fanduel`, `draftkings`, `betmgm` (`DEFAULT_RETAIL_BOOKMAKERS`) | Same | `["pinnacle", "betfair_exchange"]`; no retail list |
| Time | Snapshot at decision time | Same | Newest price per book within a lookback window (default 2 h) |

Two features with the same name can therefore differ: averaging probabilities and taking the probability of the averaged price are different for books on opposite sides of even money. The training defaults are Pinnacle-only; hybrid production configs set `sharp_bookmakers` explicitly ([Decision: hybrid sharp reference](decision-hybrid-sharp-reference.md), [Training config](training-config.md)).

## Hazards

- Vig is included on both sides. A retail book with a larger margin shows a more positive divergence on every outcome, so compare with `market_hold` in the screener output.
- The screener buckets by `(outcome, point)` but looks up the sharp price by outcome name only, so on totals and spreads a sharp price for one line can be compared with retail prices at another ([find_retail_edges](mcp-find-retail-edges.md#hazards)).
- Sharp and retail prices can come from different snapshots, so the divergence is not co-temporal.
- The MCP default sharp list is a literal in `server.py` whose comment says it duplicates the feature default "keep in sync"; the two lists differ on purpose.

## Related

- [Feature groups](feature-groups.md)
- [CLV model findings](clv-model-findings.md)
- [Look-ahead bias](look-ahead-bias.md) — the feature uses decision-time snapshots
