---
type: reference
title: odds-mcp server
description: How the FastMCP server is launched and mounted, the 17-tool catalogue and owning pages, sport sub-servers, shared vocabulary, and the scrape and scheduling tools (refresh_scrape, get_scrape_status, get_scheduled_jobs, schedule_next_wakeup).
sources:
  - odds_mcp.server.mcp
  - odds_mcp.server._assert_unique_tool_names
  - odds_mcp.server._coerce_bookmaker_list
  - odds_mcp.server.MarketKey
  - odds_mcp.server.refresh_scrape
  - odds_mcp.server.get_scrape_status
  - odds_mcp.server.get_scheduled_jobs
  - odds_mcp.server.schedule_next_wakeup
  - odds_mcp.tools.epl.epl_mcp
  - odds_mcp.tools.mlb.mlb_mcp
  - packages/odds-mcp/pyproject.toml
  - odds_lambda.scheduling.backends.local.build_scheduler
  - odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME
  - odds_lambda.scheduling.jobs.sport_key_to_suffix
  - odds_core.agent_wakeup_models.AgentWakeup
  - odds_mcp.server.get_current_odds
  - tests/unit/test_mcp_server.py
  - .mcp.json
mentions:
  - odds_lambda.storage.readers.OddsReader
status: stable
---

# odds-mcp server

`odds_mcp.server` defines `mcp = FastMCP("odds-mcp", ...)`, the tool interface the [betting agent](betting-agent.md) uses. Tools are thin wrappers around `odds_lambda.storage.readers.OddsReader` queries, job code and the paper-trading service; they return JSON-safe dicts, open their own sessions with `odds_core.database.async_session_maker()` and report many failures as `{"error": ...}` dicts. All times are UTC. The `@mcp.tool()` definitions are the source of truth for the toolset (`odds_mcp.server.mcp`).

## Launch

`packages/odds-mcp/pyproject.toml` declares fastmcp, httpx, apscheduler, structlog, sqlalchemy and pydantic besides the workspace packages, each because `odds_mcp` imports it directly.

`.mcp.json` registers a stdio server named `odds` that runs `uv run fastmcp run odds_mcp.server`. Running the file directly (`python server.py`) first awaits `odds_mcp.server._assert_unique_tool_names`, then `mcp.run()`. The `fastmcp run` path never calls `odds_core.logging_setup.configure_logging` (nothing in `odds_mcp.server` does) and skips `odds_mcp.server._assert_unique_tool_names`, which is only invoked under `if __name__ == "__main__"` ([code conventions](../foundations/code-conventions.md)). Unprefixed mounts let a child tool silently shadow a parent tool with the same name; under `fastmcp run` the uniqueness assertion does not run, and `tests/unit/test_mcp_server.py` checks uniqueness.

## Tool catalogue (17)

| Tool | Owning page |
|---|---|
| `odds_mcp.server.get_upcoming_fixtures`, `odds_mcp.server.get_current_odds`, `odds_mcp.server.get_odds_history`, `odds_mcp.server.get_predictions` | [market tools](mcp-market-tools.md) |
| `odds_mcp.server.find_retail_edges` | [market tools](mcp-market-tools.md#retail-edges) |
| `odds_mcp.server.save_match_brief`, `odds_mcp.server.get_match_brief`, `odds_mcp.server.get_slate_briefs` | [briefs](match-briefs.md) |
| `odds_mcp.server.paper_bet` (decimal price, optional candidate), `odds_mcp.server.get_portfolio`, `odds_mcp.server.settle_bets` | [paper trading](paper-trading.md) |
| `odds_mcp.server.refresh_scrape`, `odds_mcp.server.get_scrape_status`, `odds_mcp.server.get_scheduled_jobs`, `odds_mcp.server.schedule_next_wakeup` | [below](#scrape-and-scheduling-tools) |
| `odds_mcp.tools.epl.get_team_context` (EPL), `odds_mcp.tools.mlb.get_probable_pitchers` (MLB) | [sport context tools](mcp-sport-context-tools.md) |

15 tools are defined in `odds_mcp.server` (`odds_mcp.server.find_retail_edges`, which excludes the sharp list and `odds_core.bookmakers.NON_BETTABLE_BOOKMAKERS`, has its own test in `tests/unit/test_mcp_server.py`) and 2 in `odds_mcp/tools/`. `odds_mcp.server` has `_DEFAULT_SHARP_BOOKMAKERS = ["pinnacle", "betfair_exchange"]` and a comment saying it duplicates the feature default and must be kept in sync; the values differ deliberately (#421, [code copies](../modelling/sharp-benchmark.md#code-copies)).

## Sport sub-servers

`odds_mcp.tools.epl.epl_mcp` and `odds_mcp.tools.mlb.mlb_mcp` are separate `FastMCP` instances mounted with `mcp.mount(...)` and no namespace, so tool names stay verbatim. Adding a sport tool means a new module mounted the same way.

## Shared vocabulary

`MarketKey = Literal["h2h", "1x2", "totals", "spreads"]` types the market argument; `league` and `sport` arguments take a `SportKey` ([sport and market keys](../collection/sport-and-market-keys.md#market-key-dialects)). The feed market keys that `odds_mcp.server.refresh_scrape` takes (`1x2`, `over_under_2_5`, `home_away`) are a different vocabulary from the stored `MarketKey`. `odds_mcp.server._coerce_bookmaker_list` accepts a JSON-array string or comma-separated string for bookmaker lists because Claude Code's MCP client stringifies array parameters (anthropics/claude-code#22394, per the code comment).

## Dependencies

`odds-mcp` declares odds-core, odds-lambda, odds-analytics and fastmcp. The only odds-analytics import is `odds_analytics.utils.per_book_market_holds` (it takes decimal prices), used by `odds_mcp.server.find_retail_edges` ([workspace packages](../foundations/workspace-packages.md)). `odds_mcp.server` also imports `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME` from `odds_lambda.jobs.fetch_oddsportal` and `odds_lambda.oddsportal_client.FEED_MARKETS`.

## Scrape and scheduling tools

Four tools in `odds_mcp.server`. The first three talk to the local APScheduler store through `build_scheduler(role=SchedulerRole.scheduler)`, the Postgres-backed scheduler shared with `odds scheduler start` ([local scheduler](../scheduling/local-scheduler.md)); the last writes the `agent_wakeups` table. The restricted `odds_agent` role is granted write access to the APScheduler tables explicitly, which `odds_mcp.server.refresh_scrape` and `odds_mcp.server.get_scrape_status` need ([database role](agent-run-job.md#database-role)).

**`odds_mcp.server.refresh_scrape(league="england-premier-league", market="1x2")`** queues one on-demand OddsPortal scrape. The league must be a key of `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME` (each spec's `league` in `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`); otherwise the tool returns an error listing the known leagues. `market` must be a key of `odds_lambda.oddsportal_client.FEED_MARKETS`; otherwise the tool returns an error listing the known markets. It replaces the spec's markets with `[market]` and calls `scheduler.add_job(ingest_league, args=[spec])`, returning a `job_id`. The job runs in the scheduler process, not the MCP process; a running scheduler picks it up immediately via Postgres LISTEN/NOTIFY, and with none running the job waits in the store, so the tool needs a running local scheduler ([OddsPortal job](../collection/fetch-oddsportal-job.md)).

**`odds_mcp.server.get_scrape_status(job_id=None)`** lists pending and running jobs whose task id contains `odds_lambda.jobs.fetch_oddsportal.ingest_league` (`state` is `running` when acquired, else `pending`). With a `job_id` that is not pending it fetches the result: `completed` (with `outcome`, `started_at`, `finished_at`, `stats` from `odds_lambda.jobs.fetch_oddsportal.IngestionStats` (including `events_listed` and `upcoming_listed`), `exception`), `expired` when APScheduler has dropped the result, or `unknown`. Any scheduler error returns `status: "unavailable"` with an empty list. A run is visible to the tool only while its result is retained; after `expired`, verify ingestion with `odds_mcp.server.get_current_odds`. These runs bypass `odds_lambda.jobs.fetch_oddsportal.main` (which takes one sport per run), so a run that stores nothing neither fails nor writes a heartbeat, and they ignore the job's dormancy.

**`odds_mcp.server.get_scheduled_jobs(sport=None)`** lists APScheduler schedules as `job_name`, `next_run_time` and a status field (always `"scheduled"`). The sport filter keeps names ending in `-{suffix}`, where the suffix comes from `odds_lambda.scheduling.jobs.sport_key_to_suffix` (`epl`, `mlb`) or falls back to the raw key. A scheduler failure returns an empty list with a message. The tool reads the local APScheduler store whatever `SCHEDULER_BACKEND` says, so it does not see AWS EventBridge rules (use `odds scheduler list-jobs --backend aws`), and its suffix filter matches nothing for NBA, which has no suffix (#449).

**`odds_mcp.server.schedule_next_wakeup(sport, delay_hours, reason)`** upserts the single `agent_wakeups` row for the sport (`uq_agent_wakeup_sport_key`) with `requested_time = now + delay_hours`, clamping the delay to 0.5-168 hours, and resets `consumed_at`. It returns `sport`, `requested_time`, the clamped `delay_hours` and `reason`. The [agent-run job](agent-run-job.md#wake-up-requests) reads and consumes it after the agent exits; any future time, later as well as sooner, overrides the proximity schedule. The tool's docstring says the job reschedules only if the requested time is sooner than the default, which the code does not check (#449).

## Related

- [The betting agent](betting-agent.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
