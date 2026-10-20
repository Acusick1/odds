---
type: reference
title: Paper trading
description: The paper_trades table, the paper-trading service, MCP and CLI entry points, portfolio maths and settlement.
sources:
  - odds_core.paper_trade_models.PaperTrade
  - odds_core.paper_trade_models.TradeResult
  - odds_lambda.paper_trading.place_trade
  - odds_lambda.paper_trading.settle_trades
  - odds_lambda.paper_trading.SETTLEABLE_MARKETS
  - odds_lambda.paper_trading.VOIDING_STATUSES
  - odds_lambda.paper_trading.get_portfolio_summary
  - odds_core.odds_math.determine_h2h_winner
  - odds_core.odds_math.calculate_profit_from_odds
  - odds_mcp.server.paper_bet
  - odds_mcp.server.get_portfolio
  - odds_mcp.server.settle_bets
  - odds_cli.commands.paper.list_trades
  - odds_cli.commands.paper.settle
  - odds_cli.commands.paper.summary
  - odds_lambda.jobs.daily_digest
status: stable
---

# Paper trading

Paper trades are the simulated bets the [betting agent](betting-agent.md) places, written through MCP tools ([odds-mcp server](mcp-server.md)). The restricted agent role needs `INSERT, UPDATE` on `paper_trades` ([database role](agent-run-job.md#database-role)). Paper trades are the only place agent results surface: they are read via MCP `odds_mcp.server.get_portfolio` and `odds paper`, and the [daily digest](../operations/daily-digest.md) never reads them (`odds_lambda.jobs.daily_digest` has no paper-trade reference).

## The paper_trades table

Table `paper_trades` (`odds_core.paper_trade_models.PaperTrade`); service `odds_lambda.paper_trading`.

Columns: `event_id` (FK), `market`, `selection` (`home`/`draw`/`away`), `bookmaker` (default `best_available` from the MCP tool), `odds` (American int), `stake`, `reasoning`, `confidence` (0-1, optional), `bankroll_before`, `bankroll_after`, `placed_at`, `settled_at`, `result` (`odds_core.paper_trade_models.TradeResult`: win, loss, push, void), `pnl`. A partial index `ix_paper_trades_unsettled` covers `settled_at IS NULL`. Bankroll columns are per-trade snapshots: `bankroll_after` is `bankroll_before + pnl`, not a running balance. `odds_core.paper_trade_models.PaperTrade.to_bet_record` converts a trade for backtest analysis tools by reusing `odds_analytics.backtesting.models.BetRecord` ([backtest engine](../evaluation/backtest-engine.md)); this makes odds-core depend on odds-analytics ([workspace packages](../foundations/workspace-packages.md)).

## Service functions

- `odds_lambda.paper_trading.place_trade` validates stake greater than 0, stake not above `bankroll`, market in `odds_lambda.paper_trading.SETTLEABLE_MARKETS` (`h2h`, `1x2`) and selection in home/draw/away (`draw` on an `h2h` market only for an event of a three-way sport, `odds_core.utils.is_three_way_sport`), then flushes (callers commit). The stake caps and the requirement to state an edge are prompt-level ([agent prompts](agent-prompts.md#what-code-enforces)).
- `get_portfolio_summary(initial_bankroll=1000.0)`: `current_bankroll = initial + total_pnl - open_exposure`; `roi` is `total_pnl / total_staked * 100`. The `bankroll` that `odds_mcp.server.paper_bet` uses for sizing defaults to the portfolio's current bankroll, unless the agent passes its own.

## Entry points

| Surface | Command | Notes |
|---|---|---|
| MCP `odds_mcp.server.paper_bet` | place | Event must be `scheduled`; `bankroll` defaults to the portfolio's current bankroll; errors come back as dicts |
| MCP `odds_mcp.server.get_portfolio` | read | Summary plus open trade details |
| MCP `odds_mcp.server.settle_bets` | settle | Settled count, total P&L, trades |
| CLI `odds paper list [--settled] [-n 50]` (`odds_cli.commands.paper.list_trades`), `odds paper settle` (`odds_cli.commands.paper.settle`), `odds paper summary [--bankroll 1000]` (`odds_cli.commands.paper.summary`) | mirror | the summary command adds open exposure per event ([odds CLI](../foundations/odds-cli.md)) |

## Settlement

`odds_lambda.paper_trading.settle_trades` is idempotent and handles two cases. A trade on a `FINAL` event with both scores is graded with `odds_core.odds_math.determine_h2h_winner` for the markets in `odds_lambda.paper_trading.SETTLEABLE_MARKETS`, so a `home`/`away` bet on a drawn match loses and a `draw` bet on a 1x2 market wins; a trade on any other market is logged (`paper_trade_unsupported_market`) and left open. A trade on a `CANCELLED` event (`odds_lambda.paper_trading.VOIDING_STATUSES`) settles as `void` with zero P&L and `bankroll_after` equal to `bankroll_before`. A trade on a `POSTPONED` event stays open, because a re-dated event returns to SCHEDULED and settles when played; nothing settles as `push`. P&L comes from `odds_core.odds_math.calculate_profit_from_odds`. `FINAL` and the scores come from the fetch-scores job (AWS) or the OddsPortal results job, `CANCELLED` from the OddsPortal results job, so stalled jobs leave bets unsettled ([event status lifecycle](../collection/event-status-lifecycle.md), [results jobs](../collection/results-jobs.md)).

## Related

- [Match briefs](match-briefs.md)
- [Agent prompts](agent-prompts.md)
- [Backtest engine](../evaluation/backtest-engine.md), where `odds_analytics.backtesting.models.BetRecord` is consumed
