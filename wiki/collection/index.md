---
title: Data collection
description: "Every external feed and the job that collects it: The Odds API, OddsPortal, Betfair Exchange, results and status jobs, historical loads, and the EPL, MLB, NBA and Polymarket context feeds."
---

# Data collection

Each external source has a fetcher, a job (or manual script) that runs it, and a destination table. [sources-and-keys](sources-and-keys.md) is the inventory and the place to start; the other pages cover one feed each.

<!-- pages:start -->
### Concepts

- [Data sources, sport keys and market keys](sources-and-keys.md) — Every odds feed and what it supplies, the bookmaker coverage per source, SportKey and job-name suffixes, the h2h/1x2/OddsHarvester market-key dialects, and what adding a sport, bookmaker or market touches.

### Processes

- [Event status lifecycle and results jobs](event-lifecycle.md) — The five event statuses, which job or command sets each (update-status to LIVE; fetch-scores, fetch-oddsportal-results and CLI to FINAL), each job's cadence and failure modes, and what each status gates.
- [Odds API ingestion and the fetch-odds job](odds-api-ingestion.md) — How one Odds API fetch becomes stored rows through OddsIngestionService, what the fetch-odds job adds (event sync, gate, scheduling, quota alerts), and what fetch_logs records and omits.

### Components

- [Betfair Exchange client and job](betfair-exchange.md) — The Betfair client, cert loading and adapter, the fetch-betfair-exchange job's cadence on AWS and locally, and what price is actually stored as betfair_exchange.
- [EPL context data (ESPN, FPL, FBref)](epl-context-data.md) — ESPN fixtures (the fetch-espn-fixtures job) and lineups, FPL availability and FBref stats, covering fetchers, scripts, tables and readers, and which of them any feature or tool actually consumes.
- [MLB probable pitchers](mlb-probable-pitchers.md) — The MLB Stats API fetcher, the fetch-mlb-probables job, the append-only mlb_probable_pitchers table and its reader, and how the MCP tool reads it.
- [NBA injury reports, game logs and player stats](nba-injuries-and-stats.md) — The deprioritized NBA feeds (injury PDFs, stats.nba.com game logs, PBPStats player stats), their manual CLIs and event matching, and the injury and rest features built on them, which carry no CLV value.
- [OddsPortal scraper and fetch-oddsportal job](oddsportal.md) — OddsHarvester/Playwright scraping, LEAGUE_SPECS, the raw_data adapter and retries, the fetch-oddsportal job's cadence and alerts, and why its inline CLV scoring only works on a local scheduler.
- [Polymarket pipeline and tables](polymarket.md) — The deprioritized, NBA-targeted Polymarket collection: client, ingestion, fetch and backfill jobs, manual event linking, the five polymarket_* tables and order-book metrics.
- [The Odds API client, key rotation and quota](odds-api-client-and-quota.md) — TheOddsAPIClient endpoints, retry and key rotation with the SSM index, unit cost per endpoint, and where quota is logged, alerted and checked (and where those checks mislead).

### How-tos

- [Load historical odds and scores](howto-load-historical-odds.md) — How to load history two ways, OddsPortal/football-data.co.uk scripts (snapshots only, EPL) and the Odds API historical backfill (NBA-only, paid), and what each produces and breaks.
<!-- pages:end -->
