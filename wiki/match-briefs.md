---
type: component
title: Match briefs
summary: "The append-only match_briefs table and the save, get and slate MCP tools: schema, sharp-price snapshot, decision enum and triage use."
sources:
  - packages/odds-core/odds_core/match_brief_models.py::MatchBrief
  - packages/odds-core/odds_core/match_brief_models.py::BriefDecision
  - packages/odds-core/odds_core/match_brief_models.py::SharpPriceMap
  - packages/odds-mcp/odds_mcp/server.py::save_match_brief
  - packages/odds-mcp/odds_mcp/server.py::get_match_brief
  - packages/odds-mcp/odds_mcp/server.py::get_slate_briefs
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - docs/BETTING_AGENT.md#match_briefs-table
  - docs/BETTING_AGENT.md#wake-up-workflow
  - docs/REPORTING.md#6-mcp-reporting-tools--packagesodds-mcpodds_mcpserverpy
  - docs/REPORTING.md#8-storage-backing-reports
verified_at: 7887039
---

# Match briefs

`match_briefs` (`packages/odds-core/odds_core/match_brief_models.py::MatchBrief`) is the agent's cross-session memory: one row per researched match per wake-up, never updated. Each [wake-up](matchday-wakeup.md) appends a new brief; the agent reads the latest per match for triage and the full history only for matches it researches ([BETTING_AGENT.md](../docs/BETTING_AGENT.md#match_briefs-table)).

## Schema

| Column | Notes |
|---|---|
| `id`, `event_id` | `event_id` is a foreign key to `events.id`, indexed ([events](table-events.md)) |
| `decision` | `BriefDecision`: `watching`, `bet` or `skip` |
| `summary` | One line for triage views (the tool docstring suggests under about 100 characters; not enforced) |
| `brief_text` | Free-form; structure is set by the agent prompt |
| `sharp_price_at_brief` | JSON `SharpPriceMap`, keyed by outcome name: `{bookmaker, price, implied_prob}`; nullable |
| `created_at` | UTC, timezone-aware |

## MCP tools (`packages/odds-mcp/odds_mcp/server.py`)

- `save_match_brief(event_id, market, decision, summary, brief_text)` inserts a row. It anchors on the event's latest snapshot containing `market`, then calls `OddsReader.get_sharp_prices` with `["pinnacle", "betfair_exchange"]` and a 24-hour lookback to stamp `sharp_price_at_brief`; no snapshot gives `null`. Returns the saved row.
- `get_match_brief(event_id, limit=None)` returns briefs newest first, all of them unless `limit` is set, plus the event.
- `get_slate_briefs(league="soccer_epl", days_ahead=7)` lists scheduled fixtures in the window (`days_ahead` clamped to 1-30), each with its latest brief's `decision`, `summary` and `created_at`, or `latest_brief: null`. It uses `DISTINCT ON (event_id)` ordered by newest.

The sharp stamp lets a later wake-up compare the price at brief time with the current price ([Sharp benchmark](sharp-benchmark.md), [Retail-sharp divergence](retail-sharp-divergence.md)).

## Hazards

- `market` selects which snapshot to read sharp prices from but is not stored: `MatchBrief` has no market column, so a brief cannot be tied to the market it priced, and `sharp_price_at_brief` is keyed by outcome name only.
- Sharp prices are resolved per outcome, so the stamp can mix Pinnacle and Betfair prices across outcomes ([find_retail_edges](mcp-find-retail-edges.md#hazards)).
- The default sharp list is a literal in `server.py`, not shared with the feature code, which defaults to Pinnacle only ([Sharp benchmark](sharp-benchmark.md)).
- The agent database role needs `INSERT` on `match_briefs`; other tables are read-only for it ([agent-run job](job-agent-run.md)).

## Related

- [odds-mcp server](mcp-server.md)
- [Paper trading](paper-trading.md) — bets the briefs justify
