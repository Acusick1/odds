---
status: superseded by ADR-0013
---

# Deprioritise Polymarket as a trading venue

## Context

The pipeline was built for NBA. It has a client, five `polymarket_*` tables, ingestion, fetch and backfill jobs, manual event linking and a 16-column NBA-only feature group ([polymarket](../../wiki/collection/polymarket.md)). The AWS rules for its jobs were put behind `enable_polymarket` on 2026-02-15 (9591f7b, "disable polymarket jobs", no reason given).

The deprioritisation was first recorded on 2026-03-09 in `docs/POLYMARKET.md` (1ebb99b): "Deprioritized indefinitely. Not accessible from UK, data likely collinear with sportsbook odds, 30-day CLOB retention creates ongoing maintenance burden. Pipeline code exists but is inactive." The same day, the strategic goal in `CLAUDE.md` replaced Polymarket with Betfair Exchange as the alternative execution venue (19851af). The record behind each reason:

- **Collinearity.** A February 2026 NBA cross-source check over 228 events found a 0.9978 correlation between Polymarket and sportsbook probabilities, but only 92 of the events had snapshots matched within 30 minutes (`docs/MODELING.md`, "Cross-source validation", added in #134). The other feature evidence is just as thin:
  - In Exp 1 the 11 Polymarket order-book features were all zero, because live polling had not been deployed (#134).
  - Exp 2 (2026-02-18, 230 events) found "no evidence of incremental signal from Polymarket data at this sample size" (#136, `experiments/results/exp2_feature_group_isolation/FINDINGS.md`).
- **UK access.** UK access is an external fact the repo cannot verify. It was recorded on 2026-03-09 (1ebb99b) and widened on 2026-04-17 to trading "blocked across all sports" (dff040d).
- **Retention.** CLOB price history expires on a rolling basis of about 30 days, so the backfill has to run every few days (`docs/POLYMARKET.md`).

Later records added market observations, which are external facts as of their dates:

- **EPL.** On 2026-04-13 the EPL books were recorded as thin ($10K-$100K per match) and AMM-driven, so the order book reflects market-maker parameters rather than public sentiment (3f74022). No probe script or results file supports this.
- **MLB.** On 2026-04-17 the MLB books were recorded as deep: median $152K per game, 1 cent spreads and 32-44 levels per side. They are also collinear with devigged bet365 through arbitrage, which makes MLB a sharp-reference candidate rather than a mispricing venue (`docs/POLYMARKET.md`, dff040d). The probe scripts are committed, but no results file exists. These observations are on [polymarket-market-research](../../wiki/collection/polymarket-market-research.md).

## Decision

Do not trade on Polymarket, and stop extending the Polymarket pipeline.

- The NBA-targeted code and tables stay in place ("Pipeline code exists if liquidity improves", 3f74022).
- On AWS, Terraform `enable_polymarket` defaults to false (`deployment/aws/terraform/variables.tf`).

Using Polymarket as an MLB sharp reference is outside this decision and is still open (#456).

## Consequences

- No Polymarket job runs by default anywhere:
  - The AWS gate defaults to false.
  - The local default `odds_core.config.SchedulerConfig` `bootstrap_jobs` omits `fetch-polymarket` and `backfill-polymarket`.
  - `odds_core.config.PolymarketConfig.enabled` defaults to true, so the jobs still execute when they are invoked directly or added to the bootstrap list.
  - The repo does not record the deployed AWS state.
- Price history for resolved markets expires after about 31 days, and order books cannot be backfilled, so data that was not collected live is lost ([polymarket-market-research](../../wiki/collection/polymarket-market-research.md#data-availability)).
- The deployed model config uses only the tabular and match_stats feature groups (`experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`).
- The pipeline stays NBA-only. The tickers and the configured series and tag ids are NBA, and events are linked only by `odds polymarket link` (#456).
- The live pipeline never populates `best_bid` and `best_ask` on price snapshots (#451). `PolymarketWriter.store_price_snapshot` computes `spread` and `midpoint` only when both are present, so those stay null too. Order-book snapshots compute all four from the book.
- Extending the pipeline to another sport needs that sport's series and tag ids and a ticker parser. Apart from `get_nba_events`, the client is sport-agnostic ([polymarket](../../wiki/collection/polymarket.md#components)).

## Evidence

- PRs and issues: #134, #136, #451, #456.
- Commits: 9591f7b (AWS gate), 1ebb99b and 19851af (deprioritisation recorded, 2026-03-09), 3f74022 (EPL liquidity), dff040d (MLB probes).
- Prior text:
  - `git show 1ebb99b:docs/POLYMARKET.md` ("Status") and `git show 7887039:docs/POLYMARKET.md` ("Status", "Backfill Job").
  - `git show 7887039:docs/MODELING.md` ("Polymarket vs sportsbook cross-source validation").
  - The `AGENTS.md` "Polymarket Integration" section at `7887039`, which was removed in #458.
- Results: `experiments/results/exp2_feature_group_isolation/FINDINGS.md`.
- Scripts: `experiments/scripts/polymarket_mlb_discover.py`, `experiments/scripts/polymarket_mlb_orderbooks.py`, `experiments/scripts/polymarket_mlb_volume.py`, `experiments/scripts/polymarket_mlb_sentiment.py`.
- Code: `deployment/aws/terraform/variables.tf`, `odds_core.config.SchedulerConfig`, `odds_core.config.PolymarketConfig`, `odds_lambda.storage.polymarket_writer.PolymarketWriter.store_price_snapshot`.
