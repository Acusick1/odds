---
type: reference
title: Briefs
description: The briefs table, the sharp-price stamp, and the save, get and slate MCP tools.
sources:
  - odds_core.brief_models.Brief
  - odds_core.brief_models.BriefDecision
  - odds_core.brief_models.SharpPriceMap
  - odds_mcp.prices.resolve_sharp_prices
  - odds_mcp.server.save_brief
  - odds_mcp.server.get_brief
  - odds_mcp.server.get_slate_briefs
  - odds_core.bookmakers.SHARP_BOOKMAKERS
  - packages/odds-mcp/agents/common.md
status: stable
---

# Briefs

The [betting agent](betting-agent.md) writes [briefs](../foundations/glossary.md#brief) through MCP tools ([odds-mcp server](mcp-server.md)). The restricted agent role needs `INSERT` on `briefs` and is read-only elsewhere except for the tables listed under [database role](agent-run-job.md#database-role). The agent's other record is its [bets](bets.md).

## The briefs table

`briefs` (`odds_core.brief_models.Brief`) is the agent's cross-session memory: one row per researched event per [wake-up](../foundations/glossary.md#wake-up), never updated afterwards. Each wake-up ([steps](agent-prompts.md)) appends a new brief; the agent reads the latest per event for triage and the full history only for events it researches (Triage and Existing Briefs in `packages/odds-mcp/agents/common.md`).

| Column | Notes |
|---|---|
| `id`, `event_id` | `event_id` is an integer foreign key to `events.id`, indexed ([events](../storage/events.md)) |
| `decision` | `odds_core.brief_models.BriefDecision`: `watching`, `bet` or `skip` |
| summary | One line for triage views (the tool docstring suggests under about 100 characters; not enforced) |
| `brief_text` | Free-form; structure is set by the agent prompt |
| `sharp_price_at_brief` | JSON `SharpPriceMap`, keyed by outcome name: `{bookmaker, price, price_decimal, implied_prob}` (`price` is the derived American price, `implied_prob` comes from `price_decimal`); nullable |
| `created_at` | UTC, timezone-aware |

`odds_mcp.server.save_brief` takes `market` to choose the snapshot but the row has no market column, so a brief cannot be tied to the market it priced, and `sharp_price_at_brief` is keyed by outcome only (#480).

## MCP tools

Three MCP tools write and read briefs (none touches bets):

- `odds_mcp.server.save_brief(event_id, market, decision, summary, brief_text)` inserts a row. It anchors on the event's latest snapshot of `market` holding a `back` or `mid` price row (`odds_mcp.prices.latest_priced_snapshot`), then calls `odds_mcp.prices.resolve_sharp_prices` (the shared market-window loader and sharp projection; [resolution](mcp-market-tools.md#retail-edges)) with `odds_core.bookmakers.SHARP_BOOKMAKERS` (`pinnacle`, `betfair_exchange`) and a 24-hour lookback to stamp `sharp_price_at_brief`, one price per outcome from its lowest line (`odds_mcp.prices.SharpLines.by_outcome`); no snapshot gives `null`. Returns the saved row.
- `get_brief(event_id, limit=None)` returns briefs newest first, all of them unless `limit` is set, plus the event.
- `get_slate_briefs(league="soccer_epl", days_ahead=7)` lists scheduled events in the window (`days_ahead` clamped to 1-30), each with its latest brief's `decision`, summary and `created_at`, or `latest_brief: null`. It uses `DISTINCT ON (event_id)` ordered by newest.

## Sharp-price stamp

The stamp lets a later wake-up compare the price at brief time with the current price ([sharp references](../modelling/sharp-benchmark.md)). Sharp prices are resolved per outcome, so the stamp can mix Pinnacle and Betfair prices across outcomes ([retail edges](mcp-market-tools.md#retail-edges)). The sharp list is the agent's `odds_core.bookmakers.SHARP_BOOKMAKERS`, which differs on purpose from the feature code's Pinnacle-only default ([code copies](../modelling/sharp-benchmark.md#code-copies)).

## Related

- [Bets](bets.md)
- [Agent prompts](agent-prompts.md)
- [odds-mcp server](mcp-server.md)
