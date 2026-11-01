---
type: explanation
title: Agent data sources and rejected sources
description: Which sources are collected on a schedule and exposed through MCP versus researched live each wake-up, why lineups come from web search, and why every evaluated source was rejected.
sources:
  - odds_lambda.jobs.fetch_oddsportal.CADENCE
  - odds_lambda.jobs.fetch_betfair_exchange.CADENCE
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_lambda.espn_fixture_fetcher.EspnFixtureFetcher
  - scripts/ingest_football_data_uk.py
  - scripts/ingest_espn_lineups.py
  - packages/odds-mcp/agents/epl/base.md
  - packages/odds-mcp/agents/mlb/base.md
status: stable
---

# Agent data sources and rejected sources

The [betting agent](betting-agent.md) draws on two kinds of source. The split decides where a new source belongs: stable structured data is collected and queried through MCP, while qualitative and fast-changing context is researched live. The reasons below are external facts the repo cannot verify, as of the dates given (April 2026); prices and tier names in the rejected-source table are likewise external. The reasons were recorded in the agent data-sources doc this wiki replaced (added in 3f74022 on 2026-04-13, ESPN row revised in c0ecdf5 on 2026-04-14); only FPL (#255) has a repo experiment behind it. #295 and #296 record the attempted API-Football lineup tool (#296 closed unmerged), not the reasons it was rejected. The lineup decision is [ADR-0006](../../docs/adr/0006-lineups-from-web-research.md) and the Polymarket decision is [ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md).

## Scheduled collection

Collected by jobs or ingestion scripts into the database and available through [MCP tools](mcp-server.md). Where each job runs is in the [job deployment matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs).

| Source | Provides | Collection | Page |
|---|---|---|---|
| OddsPortal | UK multi-bookmaker odds (`odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`), upcoming and closing | Self-scheduling by kickoff proximity, plus a run 5 min before each mechanical candidate's decision time, dormant with no fixture in 28 days (`odds_lambda.jobs.fetch_oddsportal.CADENCE`, `odds_lambda.jobs.fetch_oddsportal.SCRAPE_LEAD`) | [OddsPortal](../collection/oddsportal-scraper.md) |
| Betfair Exchange | Best back/lay Match Odds, the sharp reference | Self-scheduling by kickoff proximity, plus a run 5 min before each kickoff (taken late up to kickoff), dormant with no fixture in 28 days (`odds_lambda.jobs.fetch_betfair_exchange.CADENCE`, `odds_lambda.jobs.fetch_betfair_exchange.CLOSE_LEAD`) | [Betfair](../collection/betfair-exchange.md) |
| The Odds API | US bookmaker odds | Self-scheduling by fetch tier, EPL and NBA (its `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlist; the NBA injury-report job spends none), each sport on its own key; 500 units per month per free key, whether one `ODDS_API_KEY` or each of `ODDS_API_KEYS`; each key's plan size is read from the API's `x-requests-used` and `x-requests-remaining` headers, so a paid key needs no setting | [Odds API quota](../collection/odds-api-quota.md) |
| football-data.co.uk | Historical EPL opening and closing `1x2` odds from Pinnacle, Betfair Exchange and UK retail books (stored as the CSV decimal) | Manual script (`scripts/ingest_football_data_uk.py`; snapshots and price rows with time basis `upper_bound` (opening) or `close`, tiered by that basis, an opening never closing beside its close ([fetch tier](../storage/fetch-tier.md#the-tier-a-snapshot-covers)); also fills missing results on existing events and sets the kick-off of events no better source holds an id on (read as UK local time, stored as UTC); it issues no id of its own, so it finds an event by sport, teams and kick-off, and skips a row whose match is ambiguous ([event resolver](../storage/event-resolver.md))) | [historical loads](../collection/historical-ingest-scripts.md) |
| ESPN fixtures | All-competition EPL schedule | Daily job (cron in `odds_lambda.scheduling.jobs._JOB_CRON_MAP`) | [EPL context](../collection/epl-context-data.md) |
| ESPN lineups | Starting XIs | Manual script | [EPL context](../collection/epl-context-data.md) |
| MLB Stats API | Probable pitchers; MLB results by `gamePk` | Daily jobs (`fetch-mlb-probables`, `fetch-mlb-results`) | [MLB pitchers](../collection/mlb-probable-pitchers.md), [results jobs](../collection/results-jobs.md#fetch-mlb-results) |
| Polymarket | EPL 1X2 midpoints and best bid and ask as `polymarket` snapshots at 5 min to 1 h spacing | Bootstrapped job, self-scheduling by kickoff proximity (`odds_lambda.jobs.fetch_polymarket_odds.CADENCE`) | [Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md) |

The cadence column summarises code; the `odds_lambda.jobs.fetch_oddsportal.CADENCE` and `odds_lambda.scheduling.jobs._JOB_CRON_MAP` values are authoritative. Every scheduled collector runs on the production host; the ESPN and MLB collectors are daily crons with heartbeat expectations and failing loudly on empty fetches, and every other collector self-schedules ([local scheduler](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)). ESPN is still used in the pipeline for past and upcoming fixtures (`odds_lambda.espn_fixture_fetcher.EspnFixtureFetcher`, [how it requests them](../collection/epl-context-data.md#espn-fixtures)) and a manual lineup ingestion script exists (`scripts/ingest_espn_lineups.py`); the lineup rejection below applies to lineup timing for the agent, not to those feeds. The [pipeline-side view](../collection/data-sources.md) of the same feeds lists every bookmaker and key.

## On-demand research

Not collected on a schedule; the agent searches and browses these each wake-up with web search and Playwright (the list is in `packages/odds-mcp/agents/epl/base.md`; MLB sources are in `packages/odds-mcp/agents/mlb/base.md`). Nothing in the repo fetches these sites: what the agent does with them is defined only by the prompt files ([prompts](betting-agent.md#prompts)). Odds for a match come through the MCP tools under the one integer event id of the fixture, whichever source stored them ([market tools](mcp-market-tools.md)).

| Source | Provides | Use |
|---|---|---|
| BBC Sport | Confirmed lineups, previews, injury updates | Team news, lineup confirmation near kickoff |
| Club websites and Twitter | Official lineups, press conference quotes | Primary lineup confirmation |
| RotoWire | Predicted and confirmed lineups | Cross-reference |
| Understat | xG, shot maps, form | Match preview context |
| Reddit r/soccer | Match threads, qualitative news | Supplementary context, sentiment |
| OddsShark | Consensus picks | Crude public-sentiment proxy |

## Rejected sources

| Source | Reason recorded |
|---|---|
| Betfair Exchange volume | Needs a £499 Live App Key and active betting; the free tier gives prices only |
| Twitter/X API | $100-5,000 per month under 2023 pricing, economically unviable |
| Action Network, BetQL | US sportsbook data; the "public" they track does not move EPL lines |
| Accumulator popularity | No source publishes most-backed acca legs with volume; tipster output is not crowd data |
| Tipster consensus (Predictz, Forebet) | Model-driven, lags sharp money, no signal for anticipating line movement |
| FPL API ownership and transfers | Tested as XGBoost features (#255): R² below the baseline without them; transfer spikes lag the news sources the agent searches directly |
| API-Football | Free tier blocks current-season fixture lookup (2022-2024 only) and the lineups endpoint needs the gated fixture ids; Pro plan about $20 per month |
| football-data.org | A different site from football-data.co.uk; lineups cost €29 per month and timing is undocumented |
| ESPN API (for lineups) | Lineups appear around KO-60 but non-monotonically (present at KO-60, gone by KO-45 in a 2026-04-14 Champions League test; nothing at KO-59 in a 2026-04-13 EPL test); 403 from cloud and datacenter IPs |
| Polymarket (as a sentiment source) | The EPL book tracks Betfair Exchange within 0.35-0.5 pp ([#518](https://github.com/Acusick1/odds/issues/518)), so it adds a price reference, not independent sentiment; UK trading is blocked. See [Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md) and [ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md) |

### Lineup sourcing (2026-04-13)

The record is the agent data-sources doc this wiki replaced, at 3f74022 (2026-04-13) and c0ecdf5 (the ESPN timing tests of 2026-04-13/14), plus the base prompt's "Lineup drops land roughly 60 minutes before kick-off" (`packages/odds-mcp/agents/epl/base.md`). #296, an API-Football lineup tool, was closed unmerged. No probe script or results file records the API or ESPN tests.

Every structured lineup API investigated is paywalled, blocks current-season data on its free tier, or has unconfirmed pre-kickoff timing. The agent's web search and Playwright are the lineup source: BBC Sport publishes confirmed lineups within minutes of the official team sheet (KO-60), and the agent reads them directly ([agent prompts](agent-prompts.md)). The record concludes "No new infrastructure needed", and no scheduled lineup job exists; the stored historical lineups are manually ingested and feed no feature group ([EPL context data](../collection/epl-context-data.md)). The decision is recorded in [ADR-0006](../../docs/adr/0006-lineups-from-web-research.md).

## Related

- [Goal and strategy](../foundations/goal-and-strategy.md)
- [The betting agent](betting-agent.md)
- [About the system and its pivots](../foundations/project-overview.md)
