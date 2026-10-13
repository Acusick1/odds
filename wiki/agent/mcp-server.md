---
type: component
title: odds-mcp server and scheduling tools
description: How the FastMCP server is launched and mounted, the 17-tool catalogue and owning pages, shared vocabulary, and the scrape and scheduling tools (refresh_scrape, get_scrape_status, get_scheduled_jobs, schedule_next_wakeup) and what they do not guarantee.
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
  - tests/unit/test_mcp_server.py
  - .mcp.json
status: stable
---

# odds-mcp server and scheduling tools

`odds_mcp.server` defines `mcp = FastMCP("odds-mcp", ...)`, the tool interface the [betting agent](betting-agent.md) uses. Tools are thin wrappers over `odds_lambda.storage.readers.OddsReader`, jobs and the paper-trading service; they return JSON-safe dicts, open their own sessions with `async_session_maker()` and report many failures as `{"error": ...}` dicts. All times are UTC. The `@mcp.tool()` definitions are the source of truth for the toolset (`odds_mcp.server.mcp`).

## Launch

`.mcp.json` registers a stdio server named `odds` that runs `uv run fastmcp run odds_mcp.server`. Running the file directly (`python server.py`) first awaits `odds_mcp.server._assert_unique_tool_names`, then `mcp.run()`. The `fastmcp run` path never calls `odds_core.logging_setup.configure_logging` (nothing in `odds_mcp.server` does) and skips `odds_mcp.server._assert_unique_tool_names`, which is only invoked under `if __name__ == "__main__"` ([conventions](../foundations/conventions.md)).

## Tool catalogue (17)

| Tool | Owning page |
|---|---|
| `odds_mcp.server.get_upcoming_fixtures`, `odds_mcp.server.get_current_odds`, `odds_mcp.server.get_odds_history`, `odds_mcp.server.get_predictions` | [market tools](mcp-market-tools.md) |
| `odds_mcp.server.find_retail_edges` | [market tools](mcp-market-tools.md#retail-edges) |
| `odds_mcp.server.save_match_brief`, `odds_mcp.server.get_match_brief`, `odds_mcp.server.get_slate_briefs` | [briefs](match-briefs-and-paper-trades.md#match-briefs) |
| `odds_mcp.server.paper_bet`, `odds_mcp.server.get_portfolio`, `odds_mcp.server.settle_bets` | [paper trades](match-briefs-and-paper-trades.md#paper-trades) |
| `odds_mcp.server.refresh_scrape`, `odds_mcp.server.get_scrape_status`, `odds_mcp.server.get_scheduled_jobs`, `odds_mcp.server.schedule_next_wakeup` | [below](#scrape-and-scheduling-tools) |
| `odds_mcp.tools.epl.get_team_context` (EPL), `odds_mcp.tools.mlb.get_probable_pitchers` (MLB) | [sport context tools](mcp-sport-context-tools.md) |

15 tools are defined in `odds_mcp.server` and 2 in `odds_mcp/tools/`.

## Sport sub-servers

`odds_mcp.tools.epl.epl_mcp` and `odds_mcp.tools.mlb.mlb_mcp` are separate `FastMCP` instances mounted with `mcp.mount(...)` and no namespace, so tool names stay verbatim. Adding a sport tool means a new module mounted the same way.

## Shared vocabulary

`MarketKey = Literal["h2h", "1x2", "totals", "spreads"]` types the market argument; `league` and `sport` arguments take a `SportKey` ([sport and market keys](../collection/sources-and-keys.md)). The OddsHarvester market names that `odds_mcp.server.refresh_scrape` takes (`1x2`, `over_under_2_5`, `home_away`) are a different vocabulary from the stored `MarketKey`. `odds_mcp.server._coerce_bookmaker_list` accepts a JSON-array string or comma-separated string for bookmaker lists because Claude Code's MCP client stringifies array parameters (anthropics/claude-code#22394, per the code comment).

## Dependencies

`odds-mcp` declares odds-core, odds-lambda, odds-analytics and fastmcp. The only odds-analytics import is `odds_analytics.utils.per_book_market_holds`, used by `odds_mcp.server.find_retail_edges` ([packages](../foundations/packages-and-cli.md)). `odds_mcp.server` also imports `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME` from the job module `odds_lambda.jobs.fetch_oddsportal`.

## Scrape and scheduling tools

Four tools in `odds_mcp.server`. The first three talk to the local APScheduler store through `build_scheduler(role=SchedulerRole.scheduler)`, the Postgres-backed scheduler shared with `odds scheduler start` ([local scheduler](../scheduling/local-scheduler.md)); the last writes the `agent_wakeups` table.

**`refresh_scrape(league="england-premier-league", market="1x2")`** queues one on-demand OddsPortal scrape. The league must be a key of `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME` (OddsHarvester league slugs from `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`); otherwise the tool returns an error listing the known leagues. It replaces the spec's markets with `[market]` and calls `scheduler.add_job(ingest_league, args=[spec])`, returning a `job_id`. The job runs in the scheduler process, not the MCP process; a running scheduler picks it up immediately via Postgres LISTEN/NOTIFY, and with none running the job waits in the store ([OddsPortal](../collection/oddsportal.md)). `market` is not validated here.

**`get_scrape_status(job_id=None)`** lists pending and running jobs whose task id contains `odds_lambda.jobs.fetch_oddsportal.ingest_league` (`state` is `running` when acquired, else `pending`). With a `job_id` that is not pending it fetches the result: `completed` (with `outcome`, `started_at`, `finished_at`, `stats` from `odds_lambda.jobs.fetch_oddsportal.IngestionStats`, `exception`), `expired` when APScheduler has dropped the result, or `unknown`. Any scheduler error returns `status: "unavailable"` with an empty list. After `expired`, verify ingestion with `odds_mcp.server.get_current_odds`.

**`get_scheduled_jobs(sport=None)`** lists APScheduler schedules as `job_name`, `next_run_time` and a status field (always `"scheduled"`). The sport filter keeps names ending in `-{suffix}`, where the suffix comes from `odds_lambda.scheduling.jobs.sport_key_to_suffix` (`epl`, `mlb`) or falls back to the raw key. A scheduler failure returns an empty list with a message.

**`schedule_next_wakeup(sport, delay_hours, reason)`** upserts the single `agent_wakeups` row for the sport (`uq_agent_wakeup_sport_key`) with `requested_time = now + delay_hours`, clamping the delay to 0.5-168 hours, and resets `consumed_at`. It returns `sport`, `requested_time`, the clamped `delay_hours` and `reason`. The [agent-run job](betting-agent.md#agent_wakeups) reads and consumes it after the agent exits; any future time overrides the proximity schedule.

## Hazards

- **Name shadowing.** Unprefixed mounts let a child tool silently shadow a parent tool with the same name. `odds_mcp.server._assert_unique_tool_names` runs only under `if __name__ == "__main__"`, and the configured launch is `fastmcp run`, so the "fail loudly at startup" comment does not describe that launch path. The protection in practice is `tests/unit/test_mcp_server.py`.
- 15 of 17 tools live in `odds_mcp.server` (about 1,300 lines); `tools/` holds only the sport-specific ones.
- `odds_mcp.server` has `_DEFAULT_SHARP_BOOKMAKERS = ["pinnacle", "betfair_exchange"]` and a comment saying it duplicates the feature default and must be kept in sync; the values differ deliberately ([sharp benchmark](../modelling/sharp-benchmark.md#code-copies)).
- **`odds_mcp.server.get_scheduled_jobs` is local-only.** It reads the local APScheduler store; it does not see AWS EventBridge rules (use `odds scheduler list-jobs --backend aws`). Its filter is a suffix match (`job_name.endswith(f"-{suffix}")` with `odds_lambda.scheduling.jobs.sport_key_to_suffix`), so NBA keys (no suffix) match nothing ([reporting](../operations/reporting-and-daily-digest.md)).
- **`odds_mcp.server.refresh_scrape` needs a running local scheduler** or the job just waits; and `odds_mcp.server.refresh_scrape` and `odds_mcp.server.get_scrape_status` need write access to the APScheduler tables, which the restricted `odds_agent` role is granted explicitly ([database role](betting-agent.md#database-role)).
- `odds_mcp.server.schedule_next_wakeup`'s docstring says the job reschedules "if the requested time is sooner"; the job applies any future time, including later ones.
- A run is visible to `odds_mcp.server.get_scrape_status` only while its result is retained.
- The standings logic in `odds_mcp.tools.epl` derives a points table from `espn_fixtures` rows with the same tiebreak chain as `odds_analytics.standings_features`, which derives it from `odds_core.models.Event` rows ([sport context tools](mcp-sport-context-tools.md)).

## Related

- [Betting agent](betting-agent.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Project overview](../foundations/project-overview.md)
