---
type: component
title: MCP scrape and scheduling tools
summary: "refresh_scrape, get_scrape_status, get_scheduled_jobs and schedule_next_wakeup: what each does to the scheduler store and what it does not guarantee."
sources:
  - packages/odds-mcp/odds_mcp/server.py::refresh_scrape
  - packages/odds-mcp/odds_mcp/server.py::get_scrape_status
  - packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs
  - packages/odds-mcp/odds_mcp/server.py::schedule_next_wakeup
  - packages/odds-lambda/odds_lambda/scheduling/backends/local.py::build_scheduler
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LEAGUE_SPEC_BY_NAME
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::sport_key_to_suffix
  - packages/odds-core/odds_core/agent_wakeup_models.py::AgentWakeup
  - docs/BETTING_AGENT.md#mcp-tools
  - docs/REPORTING.md#6-mcp-reporting-tools--packagesodds-mcpodds_mcpserverpy
verified_at: 7887039
---

# MCP scrape and scheduling tools

Four tools in `packages/odds-mcp/odds_mcp/server.py` ([odds-mcp server](mcp-server.md)). The first three talk to the local APScheduler store through `build_scheduler(role=SchedulerRole.scheduler)`, the Postgres-backed scheduler shared with `odds scheduler start` ([Local scheduler](local-scheduler.md)); the last writes the `agent_wakeups` table.

## `refresh_scrape(league="england-premier-league", market="1x2")`

Queues one on-demand OddsPortal scrape. The league must be a key of `LEAGUE_SPEC_BY_NAME` (OddsHarvester league slugs from `LEAGUE_SPECS`); otherwise the tool returns an error listing the known leagues. It replaces the spec's markets with `[market]` and calls `scheduler.add_job(ingest_league, args=[spec])`, returning a `job_id`. The job runs in the scheduler process, not the MCP process; a running scheduler picks it up immediately via Postgres LISTEN/NOTIFY, and with none running the job waits in the store ([OddsPortal scraper](oddsportal-scraper.md)). `market` is not validated here: it must be an OddsHarvester name (`1x2`, `over_under_2_5`, `home_away`).

## `get_scrape_status(job_id=None)`

Lists pending and running jobs whose task id contains `ingest_league` (`state` is `running` when acquired, else `pending`). With a `job_id` that is no longer pending it fetches the result: `completed` (with `outcome`, `started_at`, `finished_at`, `stats` from `IngestionStats`, `exception`), `expired` when APScheduler has dropped the result, or `unknown`. Any scheduler error returns `status: "unavailable"` with an empty list. After `expired`, verify ingestion with `get_current_odds` ([market data tools](mcp-market-data-tools.md)).

## `get_scheduled_jobs(sport=None)`

Lists APScheduler schedules as `job_name`, `next_run_time`, `status` (always `"scheduled"`). The sport filter keeps names ending in `-{suffix}`, where the suffix comes from `sport_key_to_suffix` (`epl`, `mlb`) or falls back to the raw key ([Sport and market keys](sport-and-market-keys.md)). A scheduler failure returns an empty list with a message.

## `schedule_next_wakeup(sport, delay_hours, reason)`

Upserts the single `agent_wakeups` row for the sport (`uq_agent_wakeup_sport_key`) with `requested_time = now + delay_hours`, clamping the delay to 0.5-168 hours, and resets `consumed_at`. Returns `sport`, `requested_time`, the clamped `delay_hours` and `reason`. The [agent-run job](job-agent-run.md) reads and consumes it after the agent exits.

## Hazards

- `get_scheduled_jobs` reads the local APScheduler store only; it does not see AWS EventBridge rules (use `odds scheduler list-jobs --backend aws`). `docs/REPORTING.md` calls its filter a substring match; the code is a suffix match, so NBA keys (no suffix) match nothing ([Reporting surfaces](reporting-surfaces.md)).
- `schedule_next_wakeup`'s docstring says the job reschedules "if the requested time is sooner"; the job applies any future time, including later ones.
- `refresh_scrape` and `get_scrape_status` need write access to the APScheduler tables, which the restricted `odds_agent` role is granted explicitly ([agent-run job](job-agent-run.md#database-role)).
- A run is visible to `get_scrape_status` only while its result is retained.

## Related

- [Betting agent](betting-agent.md)
- [Self-scheduling](self-scheduling.md)
