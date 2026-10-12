---
type: concept
title: Project pivots and NBA-era fossils
summary: "The three pivots (NBA to EPL/MLB, model edge to information edge, structural consolidations) and the NBA-era assumptions still live in code."
sources:
  - packages/odds-cli/odds_cli/commands/backfill.py
  - packages/odds-analytics/odds_analytics/backfill_executor.py::BackfillExecutor
  - packages/odds-lambda/odds_lambda/storage/tier_validator.py::TierCoverageValidator
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-analytics/odds_analytics/strategies.py::BasicEVStrategy
  - packages/odds-lambda/odds_lambda/polymarket_fetcher.py::PolymarketClient
  - packages/odds-core/odds_core/config.py::PolymarketConfig
  - packages/odds-cli/odds_cli/main.py::app
  - docs/audits/comprehension-audit-2026-06-09.md#t1-stratified-doc-rot-from-three-pivots
  - docs/audits/comprehension-audit-2026-06-09.md#t2-pivot-fossils-in-live-code-paths
  - docs/BETTING_AGENT.md
  - docs/INJURIES.md
verified_at: 7887039
---

# Project pivots and NBA-era fossils

The repo carries the residue of three pivots ([audit T1](../docs/audits/comprehension-audit-2026-06-09.md#t1-stratified-doc-rot-from-three-pivots)). Knowing which era a module comes from tells you whether its defaults match the active sports ([Project overview](project-overview.md)).

## The pivots

1. **NBA to EPL and MLB.** The first experiments are NBA (February 2026); EPL runs start in March ([MODELING.md](../docs/MODELING.md#experiment-log)). EPL and MLB are now the active sports; NBA code remains (injuries, game logs, Polymarket targeting).
2. **Model edge to information edge**, recorded as 2026-04-12 in the audit. The agent's synthesis of many sources, not the CLV model, is the hypothesised edge; the model is supplementary ([BETTING_AGENT.md](../docs/BETTING_AGENT.md)). See [Pivot to information edge](decision-pivot-to-information-edge.md). The NBA injury pipeline's own header says it collects data the deployed model does not use ([INJURIES.md](../docs/INJURIES.md)); see [NBA injury reports](nba-injury-reports.md).
3. **Structural consolidations**: alerts moved from odds-cli to odds-core, self-scheduling unified into one decision module, scoring inlined into the OddsPortal job. See [Alerts module](alerts-module.md), [Self-scheduling](self-scheduling.md), [Inline CLV scoring](job-score-predictions.md).

## NBA assumptions still in live code

Each fails quietly on EPL and MLB data:

- `fetch`, `discover` and `quality` commands default `--sport` to `basketball_nba` ([odds CLI](cli-overview.md#hazards)).
- `backfill plan` and `backfill scores` hardcode `sport_key="basketball_nba"` (`packages/odds-cli/odds_cli/commands/backfill.py`), as does `BackfillExecutor` (`packages/odds-analytics/odds_analytics/backfill_executor.py`). See [Historical backfill](howto-historical-backfill.md).
- `TierCoverageValidator` checks the API fixture list with `sport="basketball_nba"` only (`packages/odds-lambda/odds_lambda/storage/tier_validator.py`); `OddsReader.get_games_by_date` buckets days noon-to-noon UTC "to capture a full NBA game day" and `TierCoverageValidator` uses it for every sport ([Completeness validation](completeness-validation.md), [OddsReader](odds-reader.md)).
- `BasicEVStrategy` defaults to markets `h2h`, `spreads`, `totals` and US retail books, which match nothing in OddsPortal EPL data (`1x2`, UK books) ([Backtest strategies](backtest-strategies.md), [market keys](sport-and-market-keys.md)).
- Polymarket code targets NBA (`PolymarketClient.get_nba_events`, `PolymarketConfig.nba_series_id`) ([Polymarket](polymarket-pipeline.md)).
- The CLI banner and root project description say "NBA".

## Hazards

- The mismatches surface as empty results (no events, zero opportunities), not errors, so an NBA default and an empty EPL window look alike.
- `.env.example` sets `SPORTS=basketball_nba` ([Configuration](configuration.md#hazards)). Verify any doc claim about defaults against code ([Documentation map](documentation-map.md)).

## Related

- [Sport and market keys](sport-and-market-keys.md)
- [Betting agent](betting-agent.md)
