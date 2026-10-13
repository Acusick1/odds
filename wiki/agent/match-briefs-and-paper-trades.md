---
type: component
title: Match briefs and paper trades
description: "The agent's two persistent records: append-only match_briefs with their sharp-price stamp and the save/get/slate tools, and paper_trades with place, settle and portfolio logic via MCP and odds paper."
sources:
  - odds_core.match_brief_models.MatchBrief
  - odds_core.match_brief_models.BriefDecision
  - odds_core.match_brief_models.SharpPriceMap
  - odds_core.paper_trade_models.PaperTrade
  - odds_lambda.paper_trading.place_trade
  - odds_lambda.paper_trading.settle_trades
  - odds_lambda.paper_trading.get_portfolio_summary
  - odds_lambda.storage.readers.OddsReader
  - odds_mcp.server.save_match_brief
  - odds_mcp.server.get_match_brief
  - odds_mcp.server.get_slate_briefs
  - odds_mcp.server.paper_bet
  - odds_mcp.server.get_portfolio
  - odds_mcp.server.settle_bets
  - odds_cli.commands.paper
  - packages/odds-mcp/agents/common.md
  - odds_cli.commands.paper.list_trades
  - odds_cli.commands.paper.settle
  - odds_cli.commands.paper.summary
  - odds_lambda.jobs.daily_digest
status: stable
---

# Match briefs and paper trades

The [betting agent](betting-agent.md) leaves two durable records in the database: briefs (what it concluded about a match) and paper trades (the simulated bets it placed). Both are written through MCP tools ([odds-mcp server](mcp-server.md)); the restricted agent role needs `INSERT` on `match_briefs` and `INSERT, UPDATE` on `paper_trades`, and is read-only elsewhere ([database role](betting-agent.md#database-role)). Paper trades are the only place agent results surface: they are read via MCP `odds_mcp.server.get_portfolio` and `odds paper`, and the [daily digest](../operations/reporting-and-daily-digest.md) never reads them (`odds_lambda.jobs.daily_digest` has no paper-trade reference).

## Match briefs

`match_briefs` (`odds_core.match_brief_models.MatchBrief`) is the agent's cross-session memory: one row per researched match per wake-up, never updated. Each [wake-up](agent-wakeup-workflow.md) appends a new brief; the agent reads the latest per match for triage and the full history only for matches it researches (Triage and Existing Briefs in `packages/odds-mcp/agents/common.md`).

| Column | Notes |
|---|---|
| `id`, `event_id` | `event_id` is a foreign key to `events.id`, indexed ([events](../storage/events.md)) |
| `decision` | `odds_core.match_brief_models.BriefDecision`: `watching`, `bet` or `skip` |
| summary | One line for triage views (the tool docstring suggests under about 100 characters; not enforced) |
| `brief_text` | Free-form; structure is set by the agent prompt |
| `sharp_price_at_brief` | JSON `SharpPriceMap`, keyed by outcome name: `{bookmaker, price, implied_prob}`; nullable |
| `created_at` | UTC, timezone-aware |

MCP tools in `odds_mcp.server`:

- `save_match_brief(event_id, market, decision, summary, brief_text)` inserts a row. It anchors on the event's latest snapshot containing `market`, then calls `odds_lambda.storage.readers.OddsReader.get_sharp_prices` with `["pinnacle", "betfair_exchange"]` and a 24-hour lookback to stamp `sharp_price_at_brief`; no snapshot gives `null`. Returns the saved row.
- `get_match_brief(event_id, limit=None)` returns briefs newest first, all of them unless `limit` is set, plus the event.
- `get_slate_briefs(league="soccer_epl", days_ahead=7)` lists scheduled fixtures in the window (`days_ahead` clamped to 1-30), each with its latest brief's `decision`, summary and `created_at`, or `latest_brief: null`. It uses `DISTINCT ON (event_id)` ordered by newest.

The sharp stamp lets a later wake-up compare the price at brief time with the current price ([sharp benchmark](../modelling/sharp-benchmark.md)).

## Paper trades

Table `paper_trades` (`odds_core.paper_trade_models.PaperTrade`); service `odds_lambda.paper_trading`.

Columns: `event_id` (FK), `market`, `selection` (`home`/`draw`/`away`), `bookmaker` (default `best_available` from the MCP tool), `odds` (American int), `stake`, `reasoning`, `confidence` (0-1, optional), `bankroll_before`, `bankroll_after`, `placed_at`, `settled_at`, `result` (`odds_core.paper_trade_models.TradeResult`: win, loss, push, void), `pnl`. A partial index `ix_paper_trades_unsettled` covers `settled_at IS NULL`. Bankroll columns are per-trade snapshots: `bankroll_after` is `bankroll_before + pnl`, not a running balance. `odds_core.paper_trade_models.PaperTrade.to_bet_record` converts a trade for backtest analysis tools by reusing `odds_analytics.backtesting.models.BetRecord` ([backtesting](../evaluation/backtesting.md)).

Service functions:

- `odds_lambda.paper_trading.place_trade` validates stake greater than 0, stake not above `bankroll`, and selection in home/draw/away, then flushes (callers commit).
- `get_portfolio_summary(initial_bankroll=1000.0)`: `current_bankroll = initial + total_pnl - open_exposure`; `roi` is `total_pnl / total_staked * 100`.

Entry points:

| Surface | Command | Notes |
|---|---|---|
| MCP `odds_mcp.server.paper_bet` | place | Event must be `scheduled`; `bankroll` defaults to the portfolio's current bankroll; errors come back as dicts |
| MCP `odds_mcp.server.get_portfolio` | read | Summary plus open trade details |
| MCP `odds_mcp.server.settle_bets` | settle | Settled count, total P&L, trades |
| CLI `odds paper list [--settled] [-n 50]` (`odds_cli.commands.paper.list_trades`), `odds paper settle` (`odds_cli.commands.paper.settle`), `odds paper summary [--bankroll 1000]` (`odds_cli.commands.paper.summary`) | mirror | the summary command adds open exposure per event ([odds CLI](../foundations/packages-and-cli.md)) |

### Settlement

`odds_lambda.paper_trading.settle_trades` settles unsettled trades whose event is `FINAL` with both scores, using `odds_core.odds_math.determine_h2h_winner`; it is idempotent. P&L comes from `odds_core.odds_math.calculate_profit_from_odds`. `FINAL` and the scores are set by the fetch-scores job (AWS) or the OddsPortal results job, so stalled results jobs leave bets unsettled ([event lifecycle](../collection/event-lifecycle.md)).

## Hazards

- **Brief market is not stored.** `market` selects which snapshot to read sharp prices from but is not stored: `odds_core.match_brief_models.MatchBrief` has no market column, so a brief cannot be tied to the market it priced, and `sharp_price_at_brief` is keyed by outcome name only.
- **Mixed sharp books.** Sharp prices are resolved per outcome, so the stamp can mix Pinnacle and Betfair prices across outcomes ([retail edges tool](mcp-market-tools.md#retail-edges)).
- The default sharp list is a literal in `odds_mcp.server`, not shared with the feature code, which defaults to Pinnacle only ([sharp benchmark](../modelling/sharp-benchmark.md#code-copies)).
- **Settlement ignores `market`.** Every trade is settled as a head-to-head result from the final score, so a non-`h2h` bet would be graded wrongly; `odds_lambda.paper_trading.place_trade` does not restrict `market`.
- Draws are only meaningful for 3-way markets; a `home`/`away` bet on a drawn football match loses, and no trade is ever settled as `push` or `void` by this code.
- **Staking rules are prompt-only.** The 0-3% conviction tiers and the edge basis are not enforced here ([staking rules](agent-wakeup-workflow.md#what-code-enforces)).
- **Reverse dependency.** `odds_core.paper_trade_models.PaperTrade` imports `odds_analytics.backtesting.models` for `odds_analytics.backtesting.models.BetRecord`, so odds-core depends on odds-analytics ([packages](../foundations/packages-and-cli.md)).

## Related

- [odds-mcp server](mcp-server.md)
- [Backtest engine and metrics](../evaluation/backtesting.md) — `odds_analytics.backtesting.models.BetRecord` consumers
