---
type: reference
title: Match briefs
description: The match_briefs table, the sharp-price stamp, and the save, get and slate MCP tools.
sources:
  - odds_core.match_brief_models.MatchBrief
  - odds_core.match_brief_models.BriefDecision
  - odds_core.match_brief_models.SharpPriceMap
  - odds_lambda.storage.readers.OddsReader.get_sharp_prices
  - odds_mcp.server.save_match_brief
  - odds_mcp.server.get_match_brief
  - odds_mcp.server.get_slate_briefs
  - odds_mcp.server._DEFAULT_SHARP_BOOKMAKERS
  - packages/odds-mcp/agents/common.md
status: stable
---

# Match briefs

Briefs are what the [betting agent](betting-agent.md) concluded about a match, written through MCP tools ([odds-mcp server](mcp-server.md)). The restricted agent role needs `INSERT` on `match_briefs` and is read-only elsewhere except for the tables listed under [database role](agent-run-job.md#database-role). The agent's other record is [paper trades](paper-trading.md).

## The match_briefs table

`match_briefs` (`odds_core.match_brief_models.MatchBrief`) is the agent's cross-session memory: one row per researched match per wake-up, never updated. Each [wake-up](agent-prompts.md) appends a new brief; the agent reads the latest per match for triage and the full history only for matches it researches (Triage and Existing Briefs in `packages/odds-mcp/agents/common.md`).

| Column | Notes |
|---|---|
| `id`, `event_id` | `event_id` is a foreign key to `events.id`, indexed ([events](../storage/events.md)) |
| `decision` | `odds_core.match_brief_models.BriefDecision`: `watching`, `bet` or `skip` |
| summary | One line for triage views (the tool docstring suggests under about 100 characters; not enforced) |
| `brief_text` | Free-form; structure is set by the agent prompt |
| `sharp_price_at_brief` | JSON `SharpPriceMap`, keyed by outcome name: `{bookmaker, price, price_decimal, implied_prob}` (`price` is the derived American quote, `implied_prob` comes from `price_decimal`); nullable |
| `created_at` | UTC, timezone-aware |

`odds_mcp.server.save_match_brief` takes `market` to choose the snapshot but the row has no market column, so a brief cannot be tied to the market it priced, and `sharp_price_at_brief` is keyed by outcome only (#480).

## MCP tools

Three tools defined in `odds_mcp.server` write and read briefs:

- `odds_mcp.server.save_match_brief(event_id, market, decision, summary, brief_text)` inserts a row. It anchors on the event's latest snapshot containing `market`, then calls `odds_lambda.storage.readers.OddsReader.get_sharp_prices` with `["pinnacle", "betfair_exchange"]` and a 24-hour lookback to stamp `sharp_price_at_brief`; no snapshot gives `null`. Returns the saved row.
- `get_match_brief(event_id, limit=None)` returns briefs newest first, all of them unless `limit` is set, plus the event.
- `get_slate_briefs(league="soccer_epl", days_ahead=7)` lists scheduled fixtures in the window (`days_ahead` clamped to 1-30), each with its latest brief's `decision`, summary and `created_at`, or `latest_brief: null`. It uses `DISTINCT ON (event_id)` ordered by newest.

## Sharp-price stamp

The stamp lets a later wake-up compare the price at brief time with the current price ([sharp benchmark](../modelling/sharp-benchmark.md)). Sharp prices are resolved per outcome, so the stamp can mix Pinnacle and Betfair prices across outcomes ([retail edges](mcp-market-tools.md#retail-edges)). The default sharp list is the literal `odds_mcp.server._DEFAULT_SHARP_BOOKMAKERS` (`pinnacle`, `betfair_exchange`), not shared with the feature code, which defaults to Pinnacle only ([code copies](../modelling/sharp-benchmark.md#code-copies)).

## Related

- [Paper trading](paper-trading.md)
- [Agent prompts](agent-prompts.md)
- [odds-mcp server](mcp-server.md)
