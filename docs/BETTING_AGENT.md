# Betting Agent

An LLM betting agent that synthesizes live odds, sharp market pricing and unstructured
match context to find mispriced bets, and records them as paper trades. There is an agent
per sport (EPL and MLB); both are in interactive evaluation, not autonomous production.

The thesis being tested: an LLM agent with the right tools can process more information,
from more sources, more consistently than any individual bettor. It can monitor bookmaker
odds, exchange prices, lineup announcements, press conferences and fixture context, then
reason across all of it to find structural mispricings. The hypothesized edge is
**breadth and speed of information synthesis**, not prediction modeling.

## Design Principles

These principles have no recorded decision record; their stated reasons are (inferred).

1. **Sharp price is a strong reference, not gospel.** Pinnacle / Betfair Exchange price is
   a useful anchor, but the market is not perfectly efficient. Structural biases
   (bookmaker liability management, accumulator distortion, public money on popular teams)
   create mispricings that are not just "the sharp market hasn't caught up yet".
2. **Breadth of information is the edge.** The agent's value comes from synthesizing more
   sources than an individual bettor. The XGBoost CLV model is a supplementary signal.
3. **MCP is the tool interface.** Tools are exposed through an MCP server, so the agent
   runtime can be swapped.
4. **Tools return structured data**: dicts/JSON the agent can reason over, not log messages.
5. **The database is shared state.** Odds, events, paper trades and match briefs all live
   in PostgreSQL.
6. **Local-first.** The agent runs on the local scheduler, never on AWS (see the
   [Job Deployment Matrix](DEPLOYMENT.md#job-deployment-matrix)).

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Agent Runtime                      │
│                                                 │
│  claude -p "/agent {sport}" subprocess          │
│  started by the agent-run job / odds agent run  │
│                                                 │
│  Consumes tools via MCP client                  │
│  Researches via web search + Playwright         │
└────────┬───────────────┬───────────────┬────────┘
         │               │               │
   ┌─────▼─────┐   ┌────▼──────┐   ┌────▼──────────┐
   │ odds-mcp  │   │ Web       │   │ Playwright     │
   │           │   │ Search    │   │ MCP            │
   │ fixtures  │   │           │   │                │
   │ odds      │   │ Press     │   │ Club sites     │
   │ edges     │   │ conferences│  │ Lineup pages   │
   │ briefs    │   │ Injury    │   │ BBC/ESPN       │
   │ paper_bet │   │ news      │   │                │
   │ portfolio │   │           │   │                │
   └─────┬─────┘   └───────────┘   └────────────────┘
         │
   ┌─────▼──────────────────────────┐
   │ PostgreSQL                     │
   │ events, odds_snapshots,        │
   │ paper_trades, match_briefs,    │
   │ predictions, agent_wakeups     │
   └────────────────────────────────┘
```

The agent's instructions are the prompt files in `packages/odds-mcp/agents/`:
`common.md` (cross-sport strategy and the wake-up workflow) plus a sport file
(`epl/base.md`, `mlb/base.md`). Those files are authoritative for how the agent behaves;
this page summarizes them.

The `agent-run` job (`packages/odds-lambda/odds_lambda/jobs/agent_run.py`) runs one wake-up
and self-schedules the next by fixture proximity: every 24h more than 72h before kickoff,
12h at 24–72h, 4h at 3–24h, 1h inside 3h, and 12h when there are no fixtures. The agent can
override the next wake-up with the `schedule_next_wakeup` tool. `odds agent run` runs a
single wake-up by hand (see [CLI.md](CLI.md#run-the-betting-agent)). Run traces are
written to `logs/agent_runs/` (see [REPORTING.md](REPORTING.md#7-agent-run-logs--packagesodds-lambdaodds_lambdajobsagent_runpy)).

## Matchday Workflow

### The timeline for an EPL match (Saturday 3pm KO example)

| Window | Market activity | Agent action |
|--------|----------------|--------------|
| Mon-Thu | Lines open, early sharp money | Light research, pre-market read |
| Fri evening | Press conferences, market firms up | Research, build briefs |
| Sat morning | Team news leaks, odds move | Update briefs, early bets if edge found |
| Sat ~14:00 | Confirmed lineups drop (~KO-60) | Deep research, final decisions |
| Sat 14:00-15:00 | Final adjustments, late money | Bets placed (if any) |
| Sat evening | Matches complete | Settle bets, report P&L |

### Wake-up workflow

Every wake-up follows the same steps, defined in `packages/odds-mcp/agents/common.md`:
orient, settle, triage, pre-market read (new fixtures only), research, brief, decide.
Depth of research scales with kickoff proximity, and bets can be placed at any time if
there is a clear edge.

Each researched match gets a new brief in the `match_briefs` table. A brief records the
agent's pre-market read, the current sharp price, where and by how much the agent disagrees
with it, its assessment, watch-for items for the next wake-up, and a decision: BET, SKIP or
WATCHING.

### Conviction tiers

The agent does not bet on qualitative team assessment. Every bet needs a specific basis:
an information gap, retail or cross-venue dispersion, a structural bias, or fundamental
value. Stakes are 0% (no identifiable edge, the default), 1%, 2% or 3% of bankroll by
conviction; the criteria for each tier are in `packages/odds-mcp/agents/common.md`.

## MCP Tools

The agent consumes the pipeline through the MCP server. The `@mcp.tool()` definitions in
`packages/odds-mcp/odds_mcp/server.py` are the source of truth for the toolset (some
sport-specific tools are registered from `odds_mcp/tools/`). By purpose:

- **Odds & fixtures**: `get_upcoming_fixtures`, `get_current_odds`, `get_odds_history`,
  `find_retail_edges` (ranked retail-vs-sharp divergence screener with per-outcome
  dispersion stats)
- **Predictions**: `get_predictions` (CLV model inference, supplementary signal)
- **Briefs**: `save_match_brief` (append-only), `get_match_brief`, `get_slate_briefs`
  (per-event triage)
- **Paper trading**: `paper_bet`, `get_portfolio`, `settle_bets`
- **Scraping & scheduling**: `refresh_scrape`, `get_scrape_status`, `get_scheduled_jobs`,
  `schedule_next_wakeup`
- **Sport-specific context**: `get_team_context` (EPL), `get_probable_pitchers` (MLB)

## match_briefs table

```
id:                  int PK
event_id:            str FK -> events.id
decision:            enum (watching | bet | skip)
summary:             str (one-line for triage views)
brief_text:          str
sharp_price_at_brief: JSON  (sharp odds snapshot for later comparison)
created_at:          datetime (UTC)
```

Briefs are append-only: each wake-up creates a new row. The agent loads the latest brief
per match for triage and the full brief only for matches it researches.

## paper_trades sharp context

Each `PaperTrade` stores the sharp reference at placement and at close, so the agent's bets
can be compared with a mechanical sharp-soft rule without re-deriving prices from snapshots.
All fields are nullable; missing sharp data never blocks placement or settlement. Field
definitions are in `packages/odds-core/odds_core/paper_trade_models.py`.

- **Placement** (`paper_bet`): `sharp_prob_at_bet` (devigged sharp probability of the
  selection), `sharp_bookmaker_at_bet`, `sharp_snapshot_time_at_bet` and
  `sharp_spread_at_bet` (bet implied probability minus the sharp probability; negative means
  the bet beats the sharp price).
- **Settlement** (`settle_bets`): `sharp_prob_at_close`, `sharp_bookmaker_at_close`,
  `sharp_snapshot_time_at_close` (last pre-kickoff snapshot holding a sharp price) and `clv`
  (sharp closing probability minus bet implied probability; positive means the line moved in
  the bet's favour).

The sharp price comes from `OddsReader.get_sharp_prices` with the priority list
`AGENT_SHARP_BOOKMAKERS` (Pinnacle, then Betfair Exchange) in
`packages/odds-core/odds_core/sharp_reference.py`, shared with the MCP tools. The XGBoost
model's Pinnacle-only default is a separate setting. A reading is stored only when every
outcome of the market has a sharp price, because devigging a partial market overstates the
selection's probability. Existing trades are not backfilled.

## Role of the XGBoost Model

The CLV prediction model is a supplementary signal. Its signal is weak and not profitable
on its own (see [MODELING.md](MODELING.md#what-the-model-predicts)), and its strongest
feature (`retail_sharp_diff`) is the sharp-retail spread, which the agent sees directly via
`find_retail_edges`. The agent prompt treats predictive models as sanity checks, never as a
trading signal on their own.
