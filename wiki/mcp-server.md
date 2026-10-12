---
type: component
title: odds-mcp server
summary: "The FastMCP server package: launch, the tool catalogue with the page that owns each tool, sport sub-server mounting, the tool-name uniqueness check and the dependency on odds-analytics."
sources:
  - packages/odds-mcp/odds_mcp/server.py::mcp
  - packages/odds-mcp/odds_mcp/server.py::_assert_unique_tool_names
  - packages/odds-mcp/odds_mcp/server.py::_coerce_bookmaker_list
  - packages/odds-mcp/odds_mcp/server.py::MarketKey
  - packages/odds-mcp/odds_mcp/tools/epl.py::epl_mcp
  - packages/odds-mcp/odds_mcp/tools/mlb.py::mlb_mcp
  - packages/odds-mcp/pyproject.toml
  - tests/unit/test_mcp_server.py
  - .mcp.json
  - docs/BETTING_AGENT.md#mcp-tools
  - docs/REPORTING.md#6-mcp-reporting-tools--packagesodds-mcpodds_mcpserverpy
verified_at: 7887039
---

# odds-mcp server

`packages/odds-mcp/odds_mcp/server.py` defines `mcp = FastMCP("odds-mcp", ...)`, the tool interface the [betting agent](betting-agent.md) uses. Tools are thin wrappers over `OddsReader`, jobs and the paper-trading service; they return JSON-safe dicts, open their own sessions with `async_session_maker()` and report many failures as `{"error": ...}` dicts. All times are UTC. The `@mcp.tool()` definitions are the source of truth for the toolset ([BETTING_AGENT.md](../docs/BETTING_AGENT.md#mcp-tools)).

## Launch

`.mcp.json` registers a stdio server named `odds` that runs `uv run fastmcp run packages/odds-mcp/odds_mcp/server.py`. Running the file directly (`python server.py`) first awaits `_assert_unique_tool_names()`, then `mcp.run()`.

## Tool catalogue (17)

| Tool | Owning page |
|---|---|
| `get_upcoming_fixtures`, `get_current_odds`, `get_odds_history`, `get_predictions` | [Market data tools](mcp-market-data-tools.md) |
| `find_retail_edges` | [find_retail_edges](mcp-find-retail-edges.md) |
| `save_match_brief`, `get_match_brief`, `get_slate_briefs` | [Match briefs](match-briefs.md) |
| `paper_bet`, `get_portfolio`, `settle_bets` | [Paper trading](paper-trading.md) |
| `refresh_scrape`, `get_scrape_status`, `get_scheduled_jobs`, `schedule_next_wakeup` | [Scrape and scheduling tools](mcp-scrape-and-schedule-tools.md) |
| `get_team_context` (EPL), `get_probable_pitchers` (MLB) | [Sport context tools](mcp-sport-context-tools.md) |

15 tools are defined in `server.py` and 2 in `odds_mcp/tools/`.

## Sport sub-servers

`packages/odds-mcp/odds_mcp/tools/epl.py::epl_mcp` and `packages/odds-mcp/odds_mcp/tools/mlb.py::mlb_mcp` are separate `FastMCP` instances mounted with `mcp.mount(...)` and no namespace, so tool names stay verbatim. Adding a sport tool means a new module mounted the same way.

## Shared vocabulary

`MarketKey = Literal["h2h", "1x2", "totals", "spreads"]` types the market argument; `league` and `sport` arguments take a `SportKey` ([Sport and market keys](sport-and-market-keys.md)). `_coerce_bookmaker_list` accepts a JSON-array string or comma-separated string for bookmaker lists because Claude Code's MCP client stringifies array parameters (anthropics/claude-code#22394, per the code comment).

## Dependencies

`odds-mcp` declares odds-core, odds-lambda, odds-analytics and fastmcp. The only odds-analytics import is `odds_analytics.utils.per_book_market_holds`, used by `find_retail_edges` ([odds-analytics](package-odds-analytics.md)). `server.py` also imports `LEAGUE_SPEC_BY_NAME` from the job module `odds_lambda.jobs.fetch_oddsportal`.

## Hazards

- Unprefixed mounts let a child tool silently shadow a parent tool with the same name. `_assert_unique_tool_names` runs only under `if __name__ == "__main__"`, and the configured launch is `fastmcp run`, so the "fail loudly at startup" comment does not describe that launch path. The protection in practice is `tests/unit/test_mcp_server.py`.
- 15 of 17 tools live in `server.py` (about 1,300 lines); `tools/` holds only the sport-specific ones.
- `server.py` carries a `_DEFAULT_SHARP_BOOKMAKERS` comment saying it duplicates the feature default and must be kept in sync; the values differ deliberately ([Sharp benchmark](sharp-benchmark.md)).
- The standings logic in `packages/odds-mcp/odds_mcp/tools/epl.py` derives a points table from `espn_fixtures` rows with the same tiebreak chain as `standings_features.py`, which derives it from `Event` rows ([Sport context tools](mcp-sport-context-tools.md)).

## Related

- [Reporting surfaces](reporting-surfaces.md) — MCP as a reporting channel
- [Project overview](project-overview.md)
