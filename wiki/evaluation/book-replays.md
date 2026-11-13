---
type: reference
title: Book replays
description: How the candidate report replays a version's recorded decisions under other book sets and with each book left out, what each replay scores and what it is allowed to affect.
sources:
  - odds_analytics.book_replays
mentions:
  - odds_analytics.candidate_report.CandidateReport
  - odds_analytics.candidate_report.candidate_report
  - odds_core.bookmakers.BOOK_SETS
status: stable
---

# Book replays

A book replay answers, for information only, what a registered mechanical version would have done on the decisions it took if it had priced from other books ([ADR-0025](../../docs/adr/0025-book-set-is-part-of-a-candidate.md)). It never feeds a checkpoint, verdict, family count, `odds candidates report` or `odds_mcp.server.get_portfolio`. To act on one, register a new version with that [book set](candidate-rules.md#book-set) and give it its own forward test. The [candidate page](../operations/odds-web.md#candidate-page) shows the replays.

## What is replayed

`odds_analytics.book_replays.book_replays` replays each mechanical entry of the registry that the report holds, by name and version, so a version absent from the registry has none. `odds_analytics.candidate_report.candidate_report` calls it when asked for `replays`, after the summaries are built, and puts the results in `odds_analytics.candidate_report.CandidateReport.book_replays`. A [model rule](candidate-rules.md#model-rules) has no recorded decisions and no replay. A version whose replay raises is logged as `book_replay_failed` and left out, so a fault in this information cannot take down the report that holds the verdicts.

`odds_analytics.book_replays.replay_variants` takes the version's non-missed decisions in the order the [parity check](candidate-parity.md#replay-of-a-decision) takes them (`odds_analytics.candidate_parity.load_decisions`). For each it reads the quotes of its recorded input snapshots once (`odds_analytics.candidate_parity.recorded_quotes`) and rebuilds the state under every variant in memory (`odds_analytics.candidate_parity.recorded_intents`, with the decision's recorded fixture values and kick-off), so the replay is the parity check's, with another set of books. The variants are:

- each named book set (`odds_core.bookmakers.BookSet`), the version's own among them, which reproduces its live bets because it is the parity check's replay;
- the version's own set with each of its books left out in turn.

Each variant chains [once per selection](candidate-rules.md#once-per-selection) across a match's decisions on its own bets, as the backtest does. A replay covers only decisions the version took: a decision missed for want of inputs recorded no snapshot, so a larger set cannot recover it.

## What is scored

`odds_analytics.book_replays.replay_variants` turns each variant's bets into `odds_analytics.candidate_report.ScoredBet` placed at the decision's `known_at`, with the close of the report (`odds_analytics.candidate_report.selection_close_probability` over `odds_core.close.close_quotes`, never the backtest's substitute). A bet is settled when its event is final with both scores, by `odds_core.odds_math.selection_won` and `odds_core.odds_math.calculate_profit_from_decimal`, as `odds_lambda.paper_trading.settle_trades` settles trades, and is unsettled otherwise.

`odds_analytics.book_replays.score_bets` summarises a series as an `odds_analytics.book_replays.BookScore`: its bets, the settled ones, the bets with a close in the scoreable prefix (`odds_analytics.candidate_report.scoreable_prefix` at the report's `now`), their stake-weighted mean EV and the P&L of the settled bets.

`odds_analytics.book_replays.BookReplay` holds, per version:

- `sets`: one `odds_analytics.book_replays.BookSetReplay` per book set, with its score, whether the version registers it and its running mean EV after each bet with a close (`odds_analytics.candidate_report.RunningScore.mean_ev`, the same definition as the version's own running score, so the registered set's series is the live one);
- `books`: one `odds_analytics.book_replays.BookRow` per book of the registered set that a paper trade was placed at or whose removal changes the replay's bet count or mean EV, in key order. `placed` scores the version's paper trades at the book, over the same scoreable prefix as the per-league breakdown (`odds_analytics.candidate_report.league_breakdown`), and `without` scores the replay with the book left out.
