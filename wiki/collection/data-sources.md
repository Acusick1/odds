---
type: explanation
title: Data sources
description: Every odds feed and what each one supplies, bookmaker roles, the Pinnacle coverage gap, and where each feed runs.
sources:
  - odds_core.config.DataCollectionConfig
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_core.event_ids.LIVE_EVENT_ID_PREFIX
  - scripts/ingest_football_data_uk.py
  - odds_lambda.jobs.fetch_polymarket_odds
status: stable
---

# Data sources

Five feeds write `raw_data` snapshots in The Odds API shape ([odds tables](../storage/odds-tables.md)). The Odds API's outcomes carry its integer American `price`; the others carry the source decimal as `price_decimal`; older stored snapshots can still hold an American `price` ([price encodings](../foundations/odds-math.md#price-encodings)). Each is tagged by `odds_core.models.Event.id` prefix or `odds_core.models.OddsSnapshot.api_request_id` ([events](../storage/events.md)); the live jobs' tags (`oddsportal_live_`, `betfair_api_`, `odds_api_`, `polymarket_live_`) are what the per-source [stale-data check](../operations/health-monitor.md#stale-data-check) keys on. The keys and defaults each feed uses are in [sport-and-market-keys](sport-and-market-keys.md).

| Source | Supplies | Collected by | Where it runs | Event ids |
|---|---|---|---|---|
| The Odds API | Pinnacle, Betfair Exchange and UK retail books for EPL; US books for MLB and NBA; dense snapshots | `fetch-odds` job ([fetch-odds-job](fetch-odds-job.md), [client](odds-api-client.md)) | mini-PC, EPL only | API hex ids |
| OddsPortal | UK books, upcoming and closing odds | `fetch-oddsportal` job ([fetch-oddsportal-job](fetch-oddsportal-job.md)) | mini-PC | `op_live_*` (`odds_core.event_ids.LIVE_EVENT_ID_PREFIX`) |
| Betfair Exchange | Best back price per outcome | `fetch-betfair-exchange` job ([betfair-exchange](betfair-exchange.md)) | mini-PC | joins existing events |
| Polymarket | EPL three-way match midpoints, history from 2025-08 | `odds polymarket backfill-odds` (manual) and the opt-in `fetch-polymarket-odds` job, which is not bootstrapped and so not running ([polymarket-odds-snapshots](polymarket-odds-snapshots.md)) | manual; the job is opt-in | joins existing events |
| football-data.co.uk (FDUK) | Historical EPL opening and closing odds from Pinnacle, the exchange and UK retail books, with snapshot times stamped from FDUK's documented collection times (`odds_core.snapshot_sources.FdukTimeBasis`) and the full-time score for events that have none | manual script ([historical-ingest-scripts](historical-ingest-scripts.md)) | manual | `fduk_*` / `op_<season>_*` |

ESPN and MLB are non-odds feeds: [epl-context-data](epl-context-data.md), [mlb-probable-pitchers](mlb-probable-pitchers.md). The NBA Polymarket price and order-book pipeline was built for NBA and is deprioritised ([ADR-0007](../../docs/adr/0007-deprioritise-polymarket.md), [polymarket](polymarket.md), [market research](polymarket-market-research.md)). The ESPN and MLB collectors run as a daily cron. Every scheduled collector runs on the mini-PC ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). The agent-facing view of the same feeds, including sources it rejected, is [agent-data-sources](../agent/agent-data-sources.md).

## Bookmaker roles

(Inferred.) Pinnacle and Circa Sports are sharp, low-margin books (Circa is Vegas-based). DraftKings, FanDuel, BetMGM and Caesars (`williamhill_us`) are major US retail books, BetRivers is regional retail and Bovada is offshore. UK books come from OddsPortal and, for EPL, from the Odds API (William Hill, Paddy Power, Sky Bet, Ladbrokes, Coral, Betfred, BetVictor, 888sport). bet365 is the suspended model's target, and only OddsPortal and FDUK carry it ([ADR-0002](../../docs/adr/0002-bet365-clv-target.md)); Betway, Betfred and bwin are retail. The per-sport Odds API bookmaker lists are `odds_core.config.DataCollectionConfig.bookmakers` ([sport-and-market-keys](sport-and-market-keys.md#odds-api-scope)).

One book has one key whatever the source: the Odds API keys are renamed to the OddsPortal and FDUK keys (`betfair_exchange`, `betfred`, `ladbrokes`, `888sport`), and the Odds API's EPL three-way market is stored as `1x2` with canonical team names, so the same book from two sources shares one slot ([sport-and-market-keys](sport-and-market-keys.md#odds-api-scope)). EPL `1x2` reads therefore see Odds API rows: readers that take the newest price per book, such as `odds_mcp.server.find_retail_edges`, take it from whichever source wrote last, and `odds_analytics.feature_groups.TierSampler` with `required_bookmakers` can pick an Odds API snapshot, which carries no bet365.

## Pinnacle coverage gap

Pinnacle's Odds API coverage of EPL stopped in January 2026 (71 events lost; a request on 2026-10-18 returned Pinnacle for all 21 upcoming events), and FDUK's Pinnacle odds are unreliable after July 2025. Betfair Exchange is the sharp fallback ([ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md); the experiments are in [exp-epl-sharp-reference](../evaluation/exp-epl-sharp-reference.md)). Event counts and season coverage are data facts the repo cannot confirm; query the database.

## Deployed state

The deployed schedule of each collector is not in the repo. `odds scheduler list-jobs --backend aws` lists what runs on AWS, and [job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs) gives the design placement of every job.

## Related

- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Odds API quota accounting](odds-api-quota.md)
- [CLV model findings](../evaluation/clv-model-findings.md), for which source each experiment used
