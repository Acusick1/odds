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
  - odds_core.odds_math.selection_won
  - odds_core.odds_math.calculate_profit_from_decimal
  - odds_core.git_info.current_git_sha
  - odds_mcp.server.paper_bet
  - odds_mcp.server.get_portfolio
  - odds_mcp.server.settle_bets
  - odds_cli.commands.paper.list_trades
  - odds_cli.commands.paper.settle
  - odds_cli.commands.paper.summary
  - odds_core.paper_trade_models.CandidateDecision
  - odds_lambda.paper_trading.BANKROLL_SIZED
status: stable
---

# Paper trading

Paper trades are the simulated bets the [betting agent](betting-agent.md) places, written through MCP tools ([odds-mcp server](mcp-server.md)), and the flat-stake bets the [run-candidates job](../evaluation/run-candidates-job.md) places for mechanical candidates. The restricted agent role needs `INSERT, UPDATE` on `paper_trades` ([database role](agent-run-job.md#database-role)). Paper trades are the only place results surface: they are read via MCP `odds_mcp.server.get_portfolio` and `odds paper`.

## The paper_trades table

Table `paper_trades` (`odds_core.paper_trade_models.PaperTrade`); service `odds_lambda.paper_trading`.

Columns: `event_id` (FK), `market` (`1x2`, `h2h` or `totals`), `selection` (`home`/`draw`/`away`, or `over`/`under` on totals), `line` (numeric, the selection's goal line such as 2.5 on totals, null in a market without lines), `bookmaker` (default `best_available` from the MCP tool), `price_decimal` (float above 1.0, the decimal price taken, exact for a UK decimal quote), `stake`, `reasoning`, `confidence` (0-1, optional), `bankroll_before`, `bankroll_after` (both null for a flat-stake candidate trade), `placed_at`, `settled_at`, `result` (`odds_core.paper_trade_models.TradeResult`: win, loss, push, void), `pnl`, and the candidate fields below. A partial index `ix_paper_trades_unsettled` covers `settled_at IS NULL`. Bankroll columns are per-trade snapshots: `bankroll_after` is `bankroll_before + pnl`, not a running balance. A check constraint requires a bankroll unless the trade has a candidate.

### Candidate fields

`candidate_name` (indexed) and `candidate_version` are both set or both null (a check constraint); null means the trade was placed outside the [candidate registry](../evaluation/candidate-registry.md), and such a trade is never scored. `decision_id` references the `odds_core.paper_trade_models.CandidateDecision` that placed the trade (one of a candidate version's decisions on the match, one per decision time), and is null for the agent's trades (the decision also records the id of the snapshot its rule decided from; [decision record](../evaluation/run-candidates-job.md#the-decision-record)). Trades with a decision are unique on candidate, version, event, selection and line, with null lines not distinct, because a candidate version bets each selection, an outcome on a line, of a match once ([once per selection](../evaluation/candidate-rules.md#once-per-selection)); the agent's trades may repeat a bet. `git_sha` is the commit of the checkout the trade was placed from, resolved once at process start by `odds_core.git_info.resolve_git_sha`, with a `-dirty` suffix when the tree has uncommitted changes, and null when git or the repository is unavailable.

## Service functions

- `odds_lambda.paper_trading.place_trade` validates a decimal price above 1.0, stake greater than 0, stake not above `bankroll` (`bankroll=None` places a flat-stake candidate trade, skips that check and records no bankroll; without a candidate it raises), market in `odds_lambda.paper_trading.SETTLEABLE_MARKETS` (`h2h`, `1x2`, `totals`), selection among the market's outcomes (`draw` on an `h2h` market only for an event of a three-way sport, `odds_core.utils.is_three_way_sport`) and the line by `odds_lambda.candidate_rules.is_bettable_line` (a half line on totals, none otherwise), then flushes (callers commit). When `candidate_name` and `candidate_version` are given, it also checks them against the registry (`odds_core.candidates.load_candidates` over `odds_core.candidates.candidates_dir`) and raises `ValueError` unless the candidate exists, its status is active, its `start_date` is not after today (UTC), its version is the file's and its `sport` is the event's sport key (the event must exist). A candidate trade on a three-way sport must use `1x2`, not `h2h`. Giving one of the two raises too. The stake caps and the requirement to state an edge are prompt-level ([agent prompts](agent-prompts.md#what-code-enforces)).
- `get_portfolio_summary(initial_bankroll=1000.0)`: `current_bankroll = initial + total_pnl - open_exposure`; `roi` is `total_pnl / total_staked * 100`. Every figure counts only trades with a bankroll (`odds_lambda.paper_trading.BANKROLL_SIZED`), so flat-stake candidate trades stay out of the agent's bankroll; so do the open trades of `odds_mcp.server.get_portfolio` and the open exposure of `odds paper summary`. Candidate trades are summarised by the candidate report. The `bankroll` that `odds_mcp.server.paper_bet` uses for sizing defaults to the portfolio's current bankroll, unless the agent passes its own.

## Entry points

| Surface | Command | Notes |
|---|---|---|
| MCP `odds_mcp.server.paper_bet` | place | Takes `price_decimal` and the optional `candidate_name` and `candidate_version`, and no line, so it places no totals trade. Event must be `scheduled`; `bankroll` defaults to the portfolio's current bankroll; errors come back as dicts |
| MCP `odds_mcp.server.get_portfolio` | read | Summary, open trade details and the per-candidate checkpoint summaries of the [candidate report](../evaluation/candidate-registry.md#odds-candidates-report), or an error message there when the registry is invalid |
| MCP `odds_mcp.server.settle_bets` | settle | Settled count, total P&L, trades |
| CLI `odds paper list [--settled] [-n 50]` (`odds_cli.commands.paper.list_trades`), `odds paper settle` (`odds_cli.commands.paper.settle`), `odds paper summary [--bankroll 1000]` (`odds_cli.commands.paper.summary`) | mirror | a selection on a line shows with it, such as `over 2.5`; the summary command adds open exposure per event ([odds CLI](../foundations/odds-cli.md)) |

## Settlement

`odds_lambda.paper_trading.settle_trades` is idempotent and handles two cases. The [run-candidates job](../evaluation/run-candidates-job.md#one-run) runs it on every run, and `odds paper settle` and `odds_mcp.server.settle_bets` run it by hand. A trade on a `FINAL` event with both scores is graded with `odds_core.odds_math.selection_won` for the markets in `odds_lambda.paper_trading.SETTLEABLE_MARKETS`: a `home`/`away` bet on a drawn match loses and a `draw` bet on a 1x2 market wins, and a totals bet wins over when the total score (home plus away) exceeds its line and under when it falls short. A totals trade without a half line, which could push, is logged (`paper_trade_unsettleable_line`) and left open, and so is a trade on any other market (`paper_trade_unsupported_market`). A trade on a `CANCELLED` event (`odds_lambda.paper_trading.VOIDING_STATUSES`) settles as `void` with zero P&L and `bankroll_after` equal to `bankroll_before`. A flat-stake trade keeps both bankroll columns null. A trade on a `POSTPONED` event stays open, because a re-dated event returns to SCHEDULED and settles when played; nothing settles as `push`. P&L comes from `odds_core.odds_math.calculate_profit_from_decimal` on the trade's decimal price. `FINAL`, the scores and `CANCELLED` come from the results jobs (OddsPortal for EPL, the MLB Stats API for MLB) or the EPL fetch-scores job, so stalled jobs leave bets unsettled ([event status lifecycle](../collection/event-status-lifecycle.md), [results jobs](../collection/results-jobs.md)).

## Related

- [Match briefs](match-briefs.md)
- [Agent prompts](agent-prompts.md)
- [Parity check](../evaluation/candidate-parity.md), which compares a candidate decision's trades with a replay
