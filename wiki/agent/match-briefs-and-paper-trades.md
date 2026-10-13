---
type: component
title: Match briefs and paper trades
description: "The agent's two persistent records: append-only match_briefs with their sharp-price stamp and the save/get/slate tools, and paper_trades with place, settle and portfolio logic via MCP and odds paper."
sources:
  - packages/odds-core/odds_core/match_brief_models.py::MatchBrief
  - packages/odds-core/odds_core/match_brief_models.py::BriefDecision
  - packages/odds-core/odds_core/match_brief_models.py::SharpPriceMap
  - packages/odds-core/odds_core/paper_trade_models.py::PaperTrade
  - packages/odds-lambda/odds_lambda/paper_trading.py::place_trade
  - packages/odds-lambda/odds_lambda/paper_trading.py::settle_trades
  - packages/odds-lambda/odds_lambda/paper_trading.py::get_portfolio_summary
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-mcp/odds_mcp/server.py::save_match_brief
  - packages/odds-mcp/odds_mcp/server.py::get_match_brief
  - packages/odds-mcp/odds_mcp/server.py::get_slate_briefs
  - packages/odds-mcp/odds_mcp/server.py::paper_bet
  - packages/odds-mcp/odds_mcp/server.py::get_portfolio
  - packages/odds-mcp/odds_mcp/server.py::settle_bets
  - packages/odds-cli/odds_cli/commands/paper.py
  - packages/odds-mcp/agents/common.md
  - packages/odds-cli/odds_cli/commands/paper.py::list_trades
  - packages/odds-cli/odds_cli/commands/paper.py::settle
  - packages/odds-cli/odds_cli/commands/paper.py::summary
  - packages/odds-lambda/odds_lambda/jobs/daily_digest.py
status: stable
---

# Match briefs and paper trades

The [betting agent](betting-agent.md) leaves two durable records in the database: briefs (what it concluded about a match) and paper trades (the simulated bets it placed). Both are written through MCP tools ([odds-mcp server](mcp-server.md)); the restricted agent role needs `INSERT` on `match_briefs` and `INSERT, UPDATE` on `paper_trades`, and is read-only elsewhere ([database role](betting-agent.md#database-role)). Paper trades are the only place agent results surface: they are read via MCP `packages/odds-mcp/odds_mcp/server.py::get_portfolio` and `odds paper`, and the [daily digest](../operations/reporting-and-daily-digest.md) never reads them (`packages/odds-lambda/odds_lambda/jobs/daily_digest.py` has no paper-trade reference).

## Match briefs

`match_briefs` (`packages/odds-core/odds_core/match_brief_models.py::MatchBrief`) is the agent's cross-session memory: one row per researched match per wake-up, never updated. Each [wake-up](agent-wakeup-workflow.md) appends a new brief; the agent reads the latest per match for triage and the full history only for matches it researches (Triage and Existing Briefs in `packages/odds-mcp/agents/common.md`).

| Column | Notes |
|---|---|
| `id`, `event_id` | `event_id` is a foreign key to `events.id`, indexed ([events](../storage/events.md)) |
| `decision` | `packages/odds-core/odds_core/match_brief_models.py::BriefDecision`: `watching`, `bet` or `skip` |
| summary | One line for triage views (the tool docstring suggests under about 100 characters; not enforced) |
| `brief_text` | Free-form; structure is set by the agent prompt |
| `sharp_price_at_brief` | JSON `SharpPriceMap`, keyed by outcome name: `{bookmaker, price, implied_prob}`; nullable |
| `created_at` | UTC, timezone-aware |

MCP tools in `packages/odds-mcp/odds_mcp/server.py`:

- `save_match_brief(event_id, market, decision, summary, brief_text)` inserts a row. It anchors on the event's latest snapshot containing `market`, then calls `OddsReader.get_sharp_prices` with `["pinnacle", "betfair_exchange"]` and a 24-hour lookback to stamp `sharp_price_at_brief`; no snapshot gives `null`. Returns the saved row.
- `get_match_brief(event_id, limit=None)` returns briefs newest first, all of them unless `limit` is set, plus the event.
- `get_slate_briefs(league="soccer_epl", days_ahead=7)` lists scheduled fixtures in the window (`days_ahead` clamped to 1-30), each with its latest brief's `decision`, summary and `created_at`, or `latest_brief: null`. It uses `DISTINCT ON (event_id)` ordered by newest.

The sharp stamp lets a later wake-up compare the price at brief time with the current price ([sharp benchmark](../modelling/sharp-benchmark.md)).

## Paper trades

Table `paper_trades` (`packages/odds-core/odds_core/paper_trade_models.py::PaperTrade`); service `packages/odds-lambda/odds_lambda/paper_trading.py`.

Columns: `event_id` (FK), `market`, `selection` (`home`/`draw`/`away`), `bookmaker` (default `best_available` from the MCP tool), `odds` (American int), `stake`, `reasoning`, `confidence` (0-1, optional), `bankroll_before`, `bankroll_after`, `placed_at`, `settled_at`, `result` (`packages/odds-core/odds_core/paper_trade_models.py::TradeResult`: win, loss, push, void), `pnl`. A partial index `ix_paper_trades_unsettled` covers `settled_at IS NULL`. Bankroll columns are per-trade snapshots: `bankroll_after` is `bankroll_before + pnl`, not a running balance. `PaperTrade.to_bet_record` converts a trade for backtest analysis tools by reusing `packages/odds-analytics/odds_analytics/backtesting/models.py::BetRecord` ([backtesting](../evaluation/backtesting.md)).

Service functions:

- `packages/odds-lambda/odds_lambda/paper_trading.py::place_trade` validates stake greater than 0, stake not above `bankroll`, and selection in home/draw/away, then flushes (callers commit).
- `get_portfolio_summary(initial_bankroll=1000.0)`: `current_bankroll = initial + total_pnl - open_exposure`; `roi` is `total_pnl / total_staked * 100`.

Entry points:

| Surface | Command | Notes |
|---|---|---|
| MCP `packages/odds-mcp/odds_mcp/server.py::paper_bet` | place | Event must be `scheduled`; `bankroll` defaults to the portfolio's current bankroll; errors come back as dicts |
| MCP `packages/odds-mcp/odds_mcp/server.py::get_portfolio` | read | Summary plus open trade details |
| MCP `packages/odds-mcp/odds_mcp/server.py::settle_bets` | settle | Settled count, total P&L, trades |
| CLI `odds paper list [--settled] [-n 50]` (`packages/odds-cli/odds_cli/commands/paper.py::list_trades`), `odds paper settle` (`packages/odds-cli/odds_cli/commands/paper.py::settle`), `odds paper summary [--bankroll 1000]` (`packages/odds-cli/odds_cli/commands/paper.py::summary`) | mirror | the summary command adds open exposure per event ([odds CLI](../foundations/packages-and-cli.md)) |

### Settlement

`packages/odds-lambda/odds_lambda/paper_trading.py::settle_trades` settles unsettled trades whose event is `FINAL` with both scores, using `packages/odds-core/odds_core/odds_math.py::determine_h2h_winner`; it is idempotent. P&L comes from `packages/odds-core/odds_core/odds_math.py::calculate_profit_from_odds`. `FINAL` and the scores are set by the fetch-scores job (AWS) or the OddsPortal results job, so stalled results jobs leave bets unsettled ([event lifecycle](../collection/event-lifecycle.md)).

## Hazards

- **Brief market is not stored.** `market` selects which snapshot to read sharp prices from but is not stored: `packages/odds-core/odds_core/match_brief_models.py::MatchBrief` has no market column, so a brief cannot be tied to the market it priced, and `sharp_price_at_brief` is keyed by outcome name only.
- **Mixed sharp books.** Sharp prices are resolved per outcome, so the stamp can mix Pinnacle and Betfair prices across outcomes ([retail edges tool](mcp-market-tools.md#retail-edges)).
- The default sharp list is a literal in `packages/odds-mcp/odds_mcp/server.py`, not shared with the feature code, which defaults to Pinnacle only ([sharp benchmark](../modelling/sharp-benchmark.md#code-copies)).
- **Settlement ignores `market`.** Every trade is settled as a head-to-head result from the final score, so a non-`h2h` bet would be graded wrongly; `packages/odds-lambda/odds_lambda/paper_trading.py::place_trade` does not restrict `market`.
- Draws are only meaningful for 3-way markets; a `home`/`away` bet on a drawn football match loses, and no trade is ever settled as `push` or `void` by this code.
- **Staking rules are prompt-only.** The 0-3% conviction tiers and the edge basis are not enforced here ([staking rules](agent-wakeup-workflow.md#what-code-enforces)).
- **Reverse dependency.** `packages/odds-core/odds_core/paper_trade_models.py::PaperTrade` imports `odds_analytics.backtesting.models` for `packages/odds-analytics/odds_analytics/backtesting/models.py::BetRecord`, so odds-core depends on odds-analytics ([packages](../foundations/packages-and-cli.md)).

## Related

- [odds-mcp server](mcp-server.md)
- [Backtest engine and metrics](../evaluation/backtesting.md) — `packages/odds-analytics/odds_analytics/backtesting/models.py::BetRecord` consumers
