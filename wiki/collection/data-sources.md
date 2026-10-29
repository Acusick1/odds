---
type: explanation
title: Data sources
description: Every odds feed and what each one supplies, bookmaker roles, the Pinnacle coverage gap, and where each feed runs.
sources:
  - odds_core.config.DataCollectionConfig
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - scripts/ingest_football_data_uk.py
  - odds_lambda.jobs.fetch_polymarket_odds
status: stable
---

# Data sources

Five feeds write snapshots from envelopes in The Odds API shape ([odds tables](../storage/odds-tables.md#envelope-format)). The Odds API's outcomes carry its integer American `price`; the others carry the source decimal as `price_decimal` ([price encodings](../foundations/odds-math.md#price-encodings)). A snapshot's feed is its `odds_core.models.OddsSnapshot.source`, and an event's id is numbered by the database whichever feed created it, with each feed's own id for it held as an external id ([events](../storage/events.md#external-ids)); the per-source [stale-data check](../operations/health-monitor.md#stale-data-check) keys on `source`. The keys and defaults each feed uses are in [sport-and-market-keys](sport-and-market-keys.md).

| Source | Supplies | Collected by | Where it runs | Events |
|---|---|---|---|---|
| The Odds API | Pinnacle, Betfair Exchange and UK retail books for EPL; US books for MLB and NBA; dense snapshots | `fetch-odds` job ([fetch-odds-job](fetch-odds-job.md), [client](odds-api-client.md)) | production host, EPL only | creates events, recording the API's id |
| OddsPortal | UK books, upcoming and closing odds | `fetch-oddsportal` job ([fetch-oddsportal-job](fetch-oddsportal-job.md)) | production host | creates events, recording the OddsPortal id |
| Betfair Exchange | Best back price per outcome | `fetch-betfair-exchange` job ([betfair-exchange](betfair-exchange.md)) | production host | creates events, recording the Betfair id |
| Polymarket | EPL three-way match midpoints, history from 2025-08 | `odds polymarket backfill-odds` (manual) and the bootstrapped per-sport `fetch-polymarket-odds-epl` job ([polymarket-odds-snapshots](polymarket-odds-snapshots.md)) | backfill manual; the job self-schedules and is dormant with no stored event in 28 days | attaches to existing events, recording the Polymarket id, and creates none; snapshots stored as source `polymarket` |
| football-data.co.uk (FDUK) | Historical EPL opening and closing odds from Pinnacle, the exchange and UK retail books, with snapshot times stamped from FDUK's documented collection times (`scripts/ingest_football_data_uk.py`; time basis `upper_bound` for the opening and `close` for the closing, from which a snapshot's tier is derived, [fetch tier](../storage/fetch-tier.md)) and the full-time score for events that have none; snapshots and price rows, and the kick-off of events no better source holds an id on | manual script ([historical-ingest-scripts](historical-ingest-scripts.md)) | manual | creates events of matches no stored event has, and skips a row matching several events ([historical-ingest-scripts](historical-ingest-scripts.md)); FDUK issues no id |

ESPN and MLB are non-odds feeds: [epl-context-data](epl-context-data.md), [mlb-probable-pitchers](mlb-probable-pitchers.md). Polymarket is collected for EPL only ([ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md), [market research](polymarket-market-research.md)). The ESPN and MLB collectors run as a daily cron. Every scheduled collector runs on the production host ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). The agent-facing view of the same feeds, including sources it rejected, is [agent-data-sources](../agent/agent-data-sources.md).

## Bookmaker roles

(Inferred.) Pinnacle and Circa Sports are sharp, low-margin books (Circa is Vegas-based). DraftKings, FanDuel, BetMGM and Caesars (`williamhill_us`) are major US retail books, BetRivers is regional retail and Bovada is offshore. UK books come from OddsPortal and, for EPL, from the Odds API (William Hill, Paddy Power, Sky Bet, Ladbrokes, Coral, Betfred, BetVictor, 888sport). Only OddsPortal and FDUK carry bet365; Betway, Betfred and bwin are retail. The per-sport Odds API bookmaker lists are `odds_core.config.DataCollectionConfig.bookmakers` ([sport-and-market-keys](sport-and-market-keys.md#odds-api-scope)).

One book has one key whatever the source: the Odds API keys are renamed to the OddsPortal and FDUK keys (`betfair_exchange`, `betfred`, `ladbrokes`, `888sport`), and the Odds API's EPL three-way market is stored as `1x2` with canonical team names, so the same book from two sources shares one slot ([sport-and-market-keys](sport-and-market-keys.md#odds-api-scope)). EPL `1x2` reads therefore see Odds API rows: readers that take the newest price per book, such as `odds_mcp.server.find_retail_edges` (over `prices` rows, [market tools](../agent/mcp-market-tools.md)), take it from whichever source wrote last, and `odds_analytics.feature_groups.TierSampler` with `required_bookmakers` can pick an Odds API snapshot, which carries no bet365, when the Odds API's close has a sharp book (it derives each snapshot's tier with `odds_analytics.feature_groups.tiers_of`, whatever its source).

## Pinnacle coverage gap

Pinnacle's Odds API coverage of EPL stopped in January 2026 (71 events lost; a request on 2026-10-18 returned Pinnacle for all 21 upcoming events), and FDUK's Pinnacle odds are unreliable after July 2025. Betfair Exchange is the sharp fallback ([ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md)). Event counts and season coverage are data facts the repo cannot confirm; query the database.

## Deployed state

The deployed schedule of each collector is not in the repo. `odds scheduler list-jobs --backend aws` lists what runs on AWS, and [job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs) gives the design placement of every job.

## Related

- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Odds API quota accounting](odds-api-quota.md)
