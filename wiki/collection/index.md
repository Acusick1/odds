---
title: Data collection
description: "Every external feed and the job that collects it: The Odds API, OddsPortal, Betfair Exchange, results and status jobs, historical loads, and the EPL, MLB, NBA and Polymarket context feeds."
---

# Data collection

Each external source has a fetcher, a job (or manual script) that runs it, and a destination table. [data-sources](data-sources.md) is the inventory and the place to start; the other pages cover one feed each, with a reference page for the code and an explanation page where the reasoning needs one.

<!-- pages:start -->
### Reference

- [Betfair Exchange client and job](betfair-exchange.md) — The Betfair client, cert loading and adapter, the fetch-betfair-exchange job's cadence and where it runs, and what price is actually stored as betfair_exchange.
- [EPL context data (ESPN, FPL, FBref)](epl-context-data.md) — ESPN fixtures (the fetch-espn-fixtures job) and lineups, FPL availability and FBref stats, covering fetchers, scripts, tables and readers, and which of them any feature or tool actually consumes.
- [Historical ingest scripts](historical-ingest-scripts.md) — scripts/scrape_oddsportal.py, ingest_oddsportal.py, ingest_football_data_uk.py: inputs, matching, ids, snapshot shape, idempotency.
- [MLB probable pitchers](mlb-probable-pitchers.md) — The MLB Stats API fetcher, the fetch-mlb-probables job, the append-only mlb_probable_pitchers table and its reader, and how the MCP tool reads it.
- [NBA injury reports, game logs and player stats](nba-injuries-and-stats.md) — The manual NBA feeds (injury PDFs, stats.nba.com game logs, PBPStats player stats), their manual CLIs, writers and event matching.
- [Odds API historical backfill](odds-api-backfill.md) — odds discover, odds backfill and the GameSelector, GapBackfillPlanner, BackfillExecutor modules.
- [Odds API ingestion service](odds-ingestion-service.md) — OddsIngestionService steps and callbacks, the fetch_logs table, and odds fetch current / odds discover upcoming.
- [OddsPortal scraper](oddsportal-scraper.md) — The OddsPortal HTTP feed client, the adapter that converts its prices, the shared oddsportal_common helpers, the results page archive reader, and odds scrape upcoming.
- [Polymarket match prices as odds snapshots](polymarket-odds-snapshots.md) — The Polymarket client and settings, how Polymarket's match markets (EPL three-way, NBA two-way) become polymarket odds snapshots, the history grid and its drop rules, attaching matches to events, the backfill command and the fetch-polymarket-odds job.
- [Results and status jobs](results-jobs.md) — update-status, fetch-scores, fetch-oddsportal-results, fetch-mlb-results and odds fetch scores: cadence, flow, heartbeats, failure handling.
- [Sport, market and bookmaker keys](sport-and-market-keys.md) — SportKey, job-name suffixes, the market-key dialects per writer, bookmaker key maps, and DataCollectionConfig defaults.
- [The fetch-odds job](fetch-odds-job.md) — Event sync, gate, pre/post-schedule and log events of odds_lambda.jobs.fetch_odds.
- [The fetch-oddsportal job](fetch-oddsportal-job.md) — LEAGUE_SPECS, flow, cadence, runs before candidate decision times, alerts, heartbeats of odds_lambda.jobs.fetch_oddsportal.
- [The Odds API client and key rotation](odds-api-client.md) — TheOddsAPIClient methods, retry and key rotation with the SSM index, and unit cost per endpoint.

### Explanation

- [Data sources](data-sources.md) — Every odds feed and what each one supplies, bookmaker roles, the Pinnacle coverage gap, and where each feed runs.
- [Event status lifecycle](event-status-lifecycle.md) — The six statuses, which job or command sets each, and what each status gates downstream.
- [Odds API quota accounting](odds-api-quota.md) — How quota is spent, recorded in fetch_logs, alerted by the health monitor, and checked with odds status quota, and where those disagree.
- [Polymarket market character and data availability](polymarket-market-research.md) — Observed market depth by sport, series and tag ids from research, market richness, and how long the API keeps data.

### How-to guides

- [Add a sport](howto-add-sport.md) — Steps to add a sport: SportKey, suffix, overnight window, league spec, Betfair config, team aliases, job sport allowlists, Terraform rules, heartbeat keys.
- [Backfill Odds API history (NBA)](howto-backfill-odds-api-history.md) — Steps for discover, plan, execute, scores and status, and the tier-gap route.
- [Load historical EPL odds](howto-load-historical-epl-odds.md) — Steps to scrape OddsPortal, ingest it, then ingest football-data.co.uk.
<!-- pages:end -->
