---
type: component
title: Paper trading
summary: "The paper_trades table, the place, settle and portfolio service, the MCP tools paper_bet, get_portfolio and settle_bets, and the odds paper CLI."
sources:
  - packages/odds-lambda/odds_lambda/paper_trading.py::place_trade
  - packages/odds-lambda/odds_lambda/paper_trading.py::settle_trades
  - packages/odds-lambda/odds_lambda/paper_trading.py::get_portfolio_summary
  - packages/odds-core/odds_core/paper_trade_models.py::PaperTrade
  - packages/odds-mcp/odds_mcp/server.py::paper_bet
  - packages/odds-mcp/odds_mcp/server.py::get_portfolio
  - packages/odds-mcp/odds_mcp/server.py::settle_bets
  - packages/odds-cli/odds_cli/commands/paper.py
  - docs/CLI.md#paper-trading-commands
  - docs/REPORTING.md#6-mcp-reporting-tools--packagesodds-mcpodds_mcpserverpy
  - docs/REPORTING.md#8-storage-backing-reports
verified_at: 7887039
---

# Paper trading

Simulated bets by the [betting agent](betting-agent.md), recorded before the event and settled against final scores. Table: `paper_trades` (`packages/odds-core/odds_core/paper_trade_models.py::PaperTrade`). Service: `packages/odds-lambda/odds_lambda/paper_trading.py`. The portfolio path is the only place agent results surface; the [daily digest](job-daily-digest.md) does not read paper trades ([REPORTING.md](../docs/REPORTING.md#9-known-limitations)).

## Table

`event_id` (FK), `market`, `selection` (`home`/`draw`/`away`), `bookmaker` (default `best_available` from the MCP tool), `odds` (American int), `stake`, `reasoning`, `confidence` (0-1, optional), `bankroll_before`, `bankroll_after`, `placed_at`, `settled_at`, `result` (`TradeResult`: win, loss, push, void), `pnl`. A partial index `ix_paper_trades_unsettled` covers `settled_at IS NULL`. `PaperTrade.to_bet_record` converts a trade for backtest analysis tools. Bankroll columns are per-trade snapshots: `bankroll_after` is `bankroll_before + pnl`, not a running balance.

## Service

- `place_trade` validates stake greater than 0, stake not above `bankroll`, and selection in home/draw/away, then flushes (callers commit).
- `settle_trades` settles unsettled trades whose event is `FINAL` with both scores, using `determine_h2h_winner`; idempotent. P&L comes from `calculate_profit_from_odds`.
- `get_portfolio_summary(initial_bankroll=1000.0)`: `current_bankroll = initial + total_pnl - open_exposure`; `roi` is `total_pnl / total_staked * 100`.

## Entry points

| Surface | Command | Notes |
|---|---|---|
| MCP `paper_bet` | place | Event must be `scheduled`; `bankroll` defaults to the portfolio's current bankroll; errors come back as dicts |
| MCP `get_portfolio` | read | Summary plus open trade details |
| MCP `settle_bets` | settle | Settled count, total P&L, trades |
| CLI `odds paper list [--settled] [-n 50]`, `settle`, `summary [--bankroll 1000]` | mirror | `summary` adds open exposure per event ([odds CLI](cli-overview.md)) |

The agent database role needs `INSERT, UPDATE` on `paper_trades` ([agent-run job](job-agent-run.md)).

## Hazards

- Settlement ignores `market`: every trade is settled as a head-to-head result from the final score, so a non-`h2h` bet would be graded wrongly. `place_trade` does not restrict `market`.
- Draws are only meaningful for 3-way markets; a `home`/`away` bet on a drawn football match loses, and no trade is ever settled as `push` or `void` by this code.
- Stake sizing rules are not enforced here ([Conviction tiers](conviction-tiers.md)).

## Related

- [Match briefs](match-briefs.md)
- [odds-mcp server](mcp-server.md)
- [Backtest metrics](backtest-metrics.md) — `BetRecord` consumers
