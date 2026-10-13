---
type: component
title: odds-mcp server and scheduling tools
description: How the FastMCP server is launched and mounted, the 17-tool catalogue and owning pages, shared vocabulary, and the scrape and scheduling tools (refresh_scrape, get_scrape_status, get_scheduled_jobs, schedule_next_wakeup) and what they do not guarantee.
sources:
  - packages/odds-mcp/odds_mcp/server.py::mcp
  - packages/odds-mcp/odds_mcp/server.py::_assert_unique_tool_names
  - packages/odds-mcp/odds_mcp/server.py::_coerce_bookmaker_list
  - packages/odds-mcp/odds_mcp/server.py::MarketKey
  - packages/odds-mcp/odds_mcp/server.py::refresh_scrape
  - packages/odds-mcp/odds_mcp/server.py::get_scrape_status
  - packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs
  - packages/odds-mcp/odds_mcp/server.py::schedule_next_wakeup
  - packages/odds-mcp/odds_mcp/tools/epl.py::epl_mcp
  - packages/odds-mcp/odds_mcp/tools/mlb.py::mlb_mcp
  - packages/odds-mcp/pyproject.toml
  - packages/odds-lambda/odds_lambda/scheduling/backends/local.py::build_scheduler
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LEAGUE_SPEC_BY_NAME
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::sport_key_to_suffix
  - packages/odds-core/odds_core/agent_wakeup_models.py::AgentWakeup
  - tests/unit/test_mcp_server.py
  - .mcp.json
status: stable
---

# odds-mcp server and scheduling tools

`packages/odds-mcp/odds_mcp/server.py` defines `mcp = FastMCP("odds-mcp", ...)`, the tool interface the [betting agent](betting-agent.md) uses. Tools are thin wrappers over `packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader`, jobs and the paper-trading service; they return JSON-safe dicts, open their own sessions with `async_session_maker()` and report many failures as `{"error": ...}` dicts. All times are UTC. The `@mcp.tool()` definitions are the source of truth for the toolset (`packages/odds-mcp/odds_mcp/server.py::mcp`).

## Launch

`.mcp.json` registers a stdio server named `odds` that runs `uv run fastmcp run packages/odds-mcp/odds_mcp/server.py`. Running the file directly (`python server.py`) first awaits `packages/odds-mcp/odds_mcp/server.py::_assert_unique_tool_names`, then `mcp.run()`. The `fastmcp run` path never calls `packages/odds-core/odds_core/logging_setup.py::configure_logging` (nothing in `packages/odds-mcp/odds_mcp/server.py` does) and skips `packages/odds-mcp/odds_mcp/server.py::_assert_unique_tool_names`, which is only invoked under `if __name__ == "__main__"` ([conventions](../foundations/conventions.md)).

## Tool catalogue (17)

| Tool | Owning page |
|---|---|
| `packages/odds-mcp/odds_mcp/server.py::get_upcoming_fixtures`, `packages/odds-mcp/odds_mcp/server.py::get_current_odds`, `packages/odds-mcp/odds_mcp/server.py::get_odds_history`, `packages/odds-mcp/odds_mcp/server.py::get_predictions` | [market tools](mcp-market-tools.md) |
| `packages/odds-mcp/odds_mcp/server.py::find_retail_edges` | [market tools](mcp-market-tools.md#retail-edges) |
| `packages/odds-mcp/odds_mcp/server.py::save_match_brief`, `packages/odds-mcp/odds_mcp/server.py::get_match_brief`, `packages/odds-mcp/odds_mcp/server.py::get_slate_briefs` | [briefs](match-briefs-and-paper-trades.md#match-briefs) |
| `packages/odds-mcp/odds_mcp/server.py::paper_bet`, `packages/odds-mcp/odds_mcp/server.py::get_portfolio`, `packages/odds-mcp/odds_mcp/server.py::settle_bets` | [paper trades](match-briefs-and-paper-trades.md#paper-trades) |
| `packages/odds-mcp/odds_mcp/server.py::refresh_scrape`, `packages/odds-mcp/odds_mcp/server.py::get_scrape_status`, `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs`, `packages/odds-mcp/odds_mcp/server.py::schedule_next_wakeup` | [below](#scrape-and-scheduling-tools) |
| `packages/odds-mcp/odds_mcp/tools/epl.py::get_team_context` (EPL), `packages/odds-mcp/odds_mcp/tools/mlb.py::get_probable_pitchers` (MLB) | [sport context tools](mcp-sport-context-tools.md) |

15 tools are defined in `packages/odds-mcp/odds_mcp/server.py` and 2 in `odds_mcp/tools/`.

## Sport sub-servers

`packages/odds-mcp/odds_mcp/tools/epl.py::epl_mcp` and `packages/odds-mcp/odds_mcp/tools/mlb.py::mlb_mcp` are separate `FastMCP` instances mounted with `mcp.mount(...)` and no namespace, so tool names stay verbatim. Adding a sport tool means a new module mounted the same way.

## Shared vocabulary

`MarketKey = Literal["h2h", "1x2", "totals", "spreads"]` types the market argument; `league` and `sport` arguments take a `SportKey` ([sport and market keys](../collection/sources-and-keys.md)). The OddsHarvester market names that `packages/odds-mcp/odds_mcp/server.py::refresh_scrape` takes (`1x2`, `over_under_2_5`, `home_away`) are a different vocabulary from the stored `MarketKey`. `packages/odds-mcp/odds_mcp/server.py::_coerce_bookmaker_list` accepts a JSON-array string or comma-separated string for bookmaker lists because Claude Code's MCP client stringifies array parameters (anthropics/claude-code#22394, per the code comment).

## Dependencies

`odds-mcp` declares odds-core, odds-lambda, odds-analytics and fastmcp. The only odds-analytics import is `odds_analytics.utils.per_book_market_holds`, used by `packages/odds-mcp/odds_mcp/server.py::find_retail_edges` ([packages](../foundations/packages-and-cli.md)). `packages/odds-mcp/odds_mcp/server.py` also imports `LEAGUE_SPEC_BY_NAME` from the job module `odds_lambda.jobs.fetch_oddsportal`.

## Scrape and scheduling tools

Four tools in `packages/odds-mcp/odds_mcp/server.py`. The first three talk to the local APScheduler store through `build_scheduler(role=SchedulerRole.scheduler)`, the Postgres-backed scheduler shared with `odds scheduler start` ([local scheduler](../scheduling/local-scheduler.md)); the last writes the `agent_wakeups` table.

**`refresh_scrape(league="england-premier-league", market="1x2")`** queues one on-demand OddsPortal scrape. The league must be a key of `LEAGUE_SPEC_BY_NAME` (OddsHarvester league slugs from `LEAGUE_SPECS`); otherwise the tool returns an error listing the known leagues. It replaces the spec's markets with `[market]` and calls `scheduler.add_job(ingest_league, args=[spec])`, returning a `job_id`. The job runs in the scheduler process, not the MCP process; a running scheduler picks it up immediately via Postgres LISTEN/NOTIFY, and with none running the job waits in the store ([OddsPortal](../collection/oddsportal.md)). `market` is not validated here.

**`get_scrape_status(job_id=None)`** lists pending and running jobs whose task id contains `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::ingest_league` (`state` is `running` when acquired, else `pending`). With a `job_id` that is not pending it fetches the result: `completed` (with `outcome`, `started_at`, `finished_at`, `stats` from `IngestionStats`, `exception`), `expired` when APScheduler has dropped the result, or `unknown`. Any scheduler error returns `status: "unavailable"` with an empty list. After `expired`, verify ingestion with `packages/odds-mcp/odds_mcp/server.py::get_current_odds`.

**`get_scheduled_jobs(sport=None)`** lists APScheduler schedules as `job_name`, `next_run_time`, `status` (always `"scheduled"`). The sport filter keeps names ending in `-{suffix}`, where the suffix comes from `packages/odds-lambda/odds_lambda/scheduling/jobs.py::sport_key_to_suffix` (`epl`, `mlb`) or falls back to the raw key. A scheduler failure returns an empty list with a message.

**`schedule_next_wakeup(sport, delay_hours, reason)`** upserts the single `agent_wakeups` row for the sport (`uq_agent_wakeup_sport_key`) with `requested_time = now + delay_hours`, clamping the delay to 0.5-168 hours, and resets `consumed_at`. It returns `sport`, `requested_time`, the clamped `delay_hours` and `reason`. The [agent-run job](betting-agent.md#agent_wakeups) reads and consumes it after the agent exits; any future time overrides the proximity schedule.

## Hazards

- **Name shadowing.** Unprefixed mounts let a child tool silently shadow a parent tool with the same name. `packages/odds-mcp/odds_mcp/server.py::_assert_unique_tool_names` runs only under `if __name__ == "__main__"`, and the configured launch is `fastmcp run`, so the "fail loudly at startup" comment does not describe that launch path. The protection in practice is `tests/unit/test_mcp_server.py`.
- 15 of 17 tools live in `packages/odds-mcp/odds_mcp/server.py` (about 1,300 lines); `tools/` holds only the sport-specific ones.
- `packages/odds-mcp/odds_mcp/server.py` has `_DEFAULT_SHARP_BOOKMAKERS = ["pinnacle", "betfair_exchange"]` and a comment saying it duplicates the feature default and must be kept in sync; the values differ deliberately ([sharp benchmark](../modelling/sharp-benchmark.md#code-copies)).
- **`packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` is local-only.** It reads the local APScheduler store; it does not see AWS EventBridge rules (use `odds scheduler list-jobs --backend aws`). Its filter is a suffix match (`job_name.endswith(f"-{suffix}")` with `packages/odds-lambda/odds_lambda/scheduling/jobs.py::sport_key_to_suffix`), so NBA keys (no suffix) match nothing ([reporting](../operations/reporting-and-daily-digest.md)).
- **`packages/odds-mcp/odds_mcp/server.py::refresh_scrape` needs a running local scheduler** or the job just waits; and `packages/odds-mcp/odds_mcp/server.py::refresh_scrape` and `packages/odds-mcp/odds_mcp/server.py::get_scrape_status` need write access to the APScheduler tables, which the restricted `odds_agent` role is granted explicitly ([database role](betting-agent.md#database-role)).
- `packages/odds-mcp/odds_mcp/server.py::schedule_next_wakeup`'s docstring says the job reschedules "if the requested time is sooner"; the job applies any future time, including later ones.
- A run is visible to `packages/odds-mcp/odds_mcp/server.py::get_scrape_status` only while its result is retained.
- The standings logic in `packages/odds-mcp/odds_mcp/tools/epl.py` derives a points table from `espn_fixtures` rows with the same tiebreak chain as `packages/odds-analytics/odds_analytics/standings_features.py`, which derives it from `packages/odds-core/odds_core/models.py::Event` rows ([sport context tools](mcp-sport-context-tools.md)).

## Related

- [Betting agent](betting-agent.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [Project overview](../foundations/project-overview.md)
