---
title: Data collection
description: "Every external feed and the job that collects it: The Odds API, OddsPortal, Betfair Exchange, results and status jobs, historical loads, and the EPL, MLB, NBA and Polymarket context feeds."
---

# Data collection

Each external source has a fetcher, a job (or manual script) that runs it, and a destination table. [data-sources](data-sources.md) is the inventory and the place to start; the other pages cover one feed each, with a reference page for the code and an explanation page where the reasoning needs one.

<!-- pages:start -->
### Reference

- [Betfair Exchange client and job](betfair-exchange.md) — The Betfair client, cert loading and adapter, the fetch-betfair-exchange job's cadence and where it runs, and what price is actually stored as betfair_exchange.
- [EPL context data (ESPN, FPL, FBref)](epl-context-data.md) — ESPN events (the fetch-espn-fixtures job) and lineups, FPL availability and FBref stats, covering fetchers, scripts, tables and readers, and which of them any feature or tool actually consumes.
- [ESPN football schedule](espn-football-schedule.md) — How the fetch-espn-fixtures job creates the upcoming events of the Championship, La Liga, Serie A, Bundesliga and Ligue 1 from ESPN's scoreboards, which ESPN team names it drops, and the dry-run command.
- [Historical ingest scripts](historical-ingest-scripts.md) — scripts/scrape_oddsportal.py, ingest_oddsportal.py, ingest_football_data_uk.py: inputs, matching, ids, snapshot shape, idempotency.
- [MLB probable pitchers](mlb-probable-pitchers.md) — The MLB Stats API fetcher, the fetch-mlb-probables job, the append-only mlb_probable_pitchers table and its reader, and how the MCP tool reads it.
- [MLB season load](mlb-season-load.md) — odds events sync-mlb-seasons: which MLB Stats API games become FINAL events, where postponed, resumed and doubleheader games are placed, how existing events are reused, and what the report lists.
- [NBA schedule, injury reports, game logs and player stats](nba-injuries-and-stats.md) — The ESPN NBA schedule and settlement, the injury-report job and the manual feeds (stats.nba.com game logs, PBPStats player stats): sources, CLIs, writers and event matching.
- [Odds API historical backfill](odds-api-backfill.md) — Acquisition manifests, the causal call planner, the executor that runs its calls, odds discover and odds backfill.
- [Odds API ingestion service](odds-ingestion-service.md) — OddsIngestionService steps and callbacks, the fetch_logs table, and odds fetch current / odds discover upcoming.
- [Odds API vendor facts](odds-api-facts.md) — What The Odds API bills, when its historical snapshots fall, how it rate-limits, which books and events its historical responses carry per league, and how long a backfill call takes, each with its source.
- [OddsPortal scraper](oddsportal-scraper.md) — The OddsPortal HTTP feed client, the adapter that converts its prices, the shared oddsportal_common helpers, the results page archive reader, and odds scrape upcoming.
- [Polymarket event prices as odds snapshots](polymarket-odds-snapshots.md) — The Polymarket client and settings, how Polymarket's event markets (EPL three-way, NBA two-way) become polymarket odds snapshots, the history grid and its drop rules, attaching games to events, the backfill command and the fetch-polymarket-odds job.
- [Results and status jobs](results-jobs.md) — update-status, fetch-scores, fetch-oddsportal-results, fetch-mlb-results and odds fetch scores: cadence, flow, heartbeats, failure handling.
- [Sport, market and book keys](sport-and-market-keys.md) — SportKey, job-name suffixes, the market-key dialects per writer, book key maps, and DataCollectionConfig defaults.
- [The fetch-odds job](fetch-odds-job.md) — Event sync, gate, pre/post-schedule and log events of odds_lambda.jobs.fetch_odds.
- [The fetch-oddsportal job](fetch-oddsportal-job.md) — LEAGUE_SPECS, flow, cadence, runs before candidate decision times and lookback points, alerts, heartbeats of odds_lambda.jobs.fetch_oddsportal.
- [The Odds API client and per-league keys](odds-api-client.md) — TheOddsAPIClient methods, retry, the key each league spends, and unit cost per endpoint.

### Explanation

- [Data sources](data-sources.md) — Every odds feed and what each one supplies, book roles, Pinnacle coverage by source, and where each feed runs.
- [Event status lifecycle](event-status-lifecycle.md) — The six statuses, which job or command sets each, and what each status gates downstream.
- [Odds API quota accounting](odds-api-quota.md) — How quota is spent, recorded in fetch_logs, alerted by the health monitor, and checked with odds status quota, and where those disagree.
- [Polymarket market character and data availability](polymarket-market-research.md) — Observed market depth by league, series and tag ids from research, market richness, and how long the API keeps data.

### How-to guides

- [Add a league](howto-add-sport.md) — Steps to add a league: SportKey, timezone, kick-off authority, suffix, overnight window, league spec and Betfair config with their event horizons, Odds API fetch plan, candidate markets, close and fallback, team aliases, job league allowlists, Terraform rules, heartbeat keys, the configured leagues, and what odds-web then shows.
- [Backfill Odds API history](howto-backfill-odds-api-history.md) — Steps to cost an acquisition manifest, execute its calls, audit what they stored, and fill scores and check status.
- [Load historical EPL odds](howto-load-historical-epl-odds.md) — Steps to scrape OddsPortal, ingest it, then ingest football-data.co.uk.
<!-- pages:end -->
