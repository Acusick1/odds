# Betting Agent

Autonomous EPL betting agent that synthesizes live odds, sharp market pricing, cross-venue volume data, and unstructured game context to identify and execute (paper then real) bets.

The thesis: an LLM agent with the right tools can process more information, from more sources, more consistently than any individual bettor. It can simultaneously monitor bookmaker odds, exchange orderbooks, public sentiment signals, lineup announcements, press conferences, and fixture context — then reason across all of it to identify structural mispricings. The edge is **breadth and speed of information synthesis**, not prediction modeling.

## Design Principles

1. **Sharp price is a strong reference, not gospel** — Pinnacle / Betfair Exchange closing price is a useful anchor but the market is not perfectly efficient. Structural biases (bookmaker liability management, accumulator distortion, public money flow on popular teams) create exploitable mispricings that aren't just "the sharp market hasn't caught up yet."
2. **Breadth of information is the edge** — the agent's value comes from synthesizing more sources than any individual bettor: cross-venue volume data, public sentiment signals, lineup/team news, and structural market context. The XGBoost CLV model is a supplementary signal, not the primary edge.
3. **MCP as canonical tool interface** — tools are MCP servers consumed by any runtime. Keeps agent runtime swappable.
4. **Tools return structured data** — dicts/JSON the agent can reason over, not log messages.
5. **Database is shared state** — odds, events, paper trades, match briefs, evaluation results all live in PostgreSQL.
6. **Local-first** — scraper and agent run locally (residential IP, no Cloudflare issues). AWS handles non-scraping jobs (settlement, alerts, scoring).

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Agent Runtime                      │
│                                                 │
│  Phase 2-3: Claude Code / claude -p             │
│  Phase 5+:  Pydantic AI (if warranted)          │
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
   │ spreads   │   │ conferences│  │ Lineup pages   │
   │ briefs    │   │ Injury    │   │ BBC/ESPN       │
   │ paper_bet │   │ news      │   │                │
   │ portfolio │   │           │   │                │
   └─────┬─────┘   └───────────┘   └────────────────┘
         │
   ┌─────▼──────────────────────────┐
   │ PostgreSQL                     │
   │ events, odds_snapshots,        │
   │ paper_trades, match_briefs,    │
   │ predictions                    │
   └────────────────────────────────┘
```

## Matchday Workflow

### The timeline for an EPL match (Saturday 3pm KO example)

| Window | Market activity | Agent action |
|--------|----------------|--------------|
| Mon-Thu | Lines open, early sharp money | — |
| Fri evening | Press conferences, market firms up | **Checkpoint 1**: context building |
| Sat morning | Team news leaks, odds move | Scraper increases frequency |
| Sat ~13:30 | Confirmed lineups drop (T-90) | **Checkpoint 2**: decision |
| Sat 13:30-15:00 | Final adjustments, late money | Bets placed (if any) |

### Checkpoint 1: Context Building (day before)

Triggered by scheduler. Agent reads fixtures and odds from DB, researches press conferences and injury news via web search, writes a structured brief per match.

**Output per match** (saved to `match_briefs` table):
- Current sharp price (Pinnacle/Betfair Exchange)
- Sharp-soft spread per bookmaker
- Key news items found (injuries, suspensions, manager quotes)
- Preliminary view: interesting / not interesting
- Watch-for items for Checkpoint 2 (e.g. "Saka fitness test tomorrow")

**No bets placed.** This checkpoint is context-building only.

### Checkpoint 2: Decision (KO -90 minutes)

Triggered by scheduler after lineup announcements. Agent loads its Checkpoint 1 brief, checks for confirmed lineups (web search/Playwright), compares current sharp price to brief-time price.

**Decision tree (draft — will evolve during interactive evaluation):**
1. Load Checkpoint 1 brief — what did I flag yesterday?
2. Get current prices across bookmakers and exchanges — what moved, what didn't?
3. Search for confirmed lineups — any surprises vs. expected XI?
4. Assess the landscape: is there an information gap, a structural bias, or a liability-driven mispricing?
5. If yes: which venue offers the best price to exploit it?
6. Decide: bet / skip, with full reasoning
7. If betting: place via `paper_bet` with conviction tier and reasoning

### Conviction Framework

The agent does not bet on "vibes" or qualitative team assessment. Every bet requires a specific basis — but that basis is not limited to information gaps vs. the sharp market. Structural mispricing (bookmaker liability shading, accumulator distortion, public bias) is equally valid.

| Tier | Stake (% bankroll) | Criteria |
|------|-------------------|----------|
| No bet | 0% | No identifiable edge. **Default.** |
| Low | 1% | Plausible edge but uncertain magnitude |
| Medium | 2% | Clear edge with supporting evidence from multiple sources |
| High | 3% | Strong edge with convergent signals (volume, news, price action) |

The exact criteria for each tier and what constitutes a valid "edge" will be refined during Phase 2 interactive evaluation. The initial framework is deliberately broad — we want to learn what works, not prematurely constrain the agent's reasoning.

### Reactive Triggers (Phase 4)

Scraper detects significant odds movement (>3% implied probability shift within 1 hour). Launches an agent session to investigate why. The agent researches the cause and decides whether to act.

## Local Scraper

The existing OddsPortal scraper runs locally instead of on AWS Lambda. Residential IP avoids Cloudflare issues. Frequency adapts to match proximity:

| Time to nearest KO | Scrape interval |
|--------------------|----------------|
| > 24 hours | Every 4 hours |
| 4-24 hours | Every 1 hour |
| 1-4 hours | Every 20 minutes |
| < 1 hour | Every 10 minutes |

Implemented via the existing `LocalSchedulerBackend` with proximity-aware scheduling in `_calculate_next_execution`.

## MCP Tools

### Existing (Phase 1 — complete)

| Tool | Purpose |
|------|---------|
| `get_upcoming_fixtures` | Events from DB |
| `get_current_odds` | Latest snapshots per bookmaker |
| `get_odds_history` | Odds movement timeline |
| `get_event_features` | Feature vector (standings, schedule, stats) |
| `get_predictions` | CLV model inference (supplementary signal) |
| `paper_bet` | Record a paper trade with reasoning |
| `get_portfolio` | Open bets, P&L summary |
| `settle_bets` | Settle completed events |
| `refresh_scrape` | Trigger OddsPortal scrape on-demand |

### New (Phase 2)

| Tool | Purpose |
|------|---------|
| `save_match_brief` | Persist checkpoint analysis to DB |
| `get_match_brief` | Load prior checkpoint brief (cross-session memory) |
| `get_sharp_soft_spread` | Dedicated sharp vs soft divergence view |

## Data Model Additions

### match_briefs table

```
id:                  int PK
event_id:            str FK -> events.id
checkpoint:          str ("context" | "decision")
brief_text:          str
sharp_price_at_brief: JSON  (sharp odds snapshot for later comparison)
created_at:          datetime (UTC)
```

## Phased Rollout

### Phase 1: odds-mcp server — COMPLETE

9 MCP tools wrapping existing jobs and DB queries. Paper trading infrastructure in place.

### Phase 2: Agent prompting and interactive evaluation

Build the two-checkpoint workflow and iterate interactively.

- Rewrite AGENT.md system prompt with information-edge thesis and conviction framework
- Add `match_briefs` table + migration
- Add `save_match_brief`, `get_match_brief`, `get_sharp_soft_spread` MCP tools
- Run Checkpoint 1 + 2 interactively for 2-3 matchdays
- Iterate on prompt: which research patterns surface actionable info? Where does the agent waste time?
- Identify reliable lineup/team news data sources (API-Football free tier insufficient — see `docs/AGENT_DATA_SOURCES.md`)

### Phase 3: Autonomous scheduled evaluation

Automate the checkpoint workflow and run it without manual intervention.

- Build `agent_scheduler.py` — queries DB for next matchday, computes checkpoint times, invokes `claude -p`
- Set up local scraper on adaptive frequency schedule
- Discord notifications for checkpoint completion and bets placed
- Daily P&L digest after settlement
- Run autonomously for multiple full matchday slates
- Monitor: token costs, scraper reliability, reasoning quality

### Phase 4: Reactive triggers and evaluation framework

Add event-driven agent triggers and prove (or disprove) edge.

- Scraper-triggered agent wake-ups on significant line movement
- Evaluation framework comparing agent vs baselines:
  - **Mechanical spread**: bet whenever retail_sharp_diff > threshold (no agent needed)
  - **Closing line value**: did the line move in the agent's favor after the bet?
  - **Random baseline**: same frequency/sizing, random direction
- New CLI command: `odds agent evaluate`
- Statistical significance testing (binomial test on CLV rate, target p < 0.05)

### Phase 5: Live execution

Gate: statistically significant CLV over 100+ paper bets.

- Betfair Exchange API integration (programmatic, legal, ~2% commission)
- Hard bankroll limits enforced at tool level, not prompt level
- Kill switch: max daily loss circuit breaker
- Agent prompt unchanged — only execution tool changes from `paper_bet` to `live_bet`

## Evaluation Criteria

The agent must demonstrate it adds value beyond a simple mechanical rule. Track per bet:

- Sharp price at time of bet placement
- Sharp price at close (filled by settlement)
- Whether the line moved in the agent's favor (CLV achieved)
- The mechanical sharp-soft spread at time of bet (baseline comparison)
- Agent's stated reasoning (for qualitative review)

**Phase 5 gate**: CLV rate significantly above 50% (p < 0.05) over 100+ bets, AND positive ROI after commission.

## Role of the XGBoost Model

The CLV prediction model (R² = 3-6%, backtest ROI +3.4% at p = 0.26) is demoted to a supplementary signal. Its strongest feature (`retail_sharp_diff`) is the sharp-soft spread, which the agent can observe directly via `get_sharp_soft_spread`.

`get_predictions` remains available as a sanity check. The agent prompt says: "Model predictions are weakly predictive. Do not bet based on model output alone."

If the information-edge approach proves viable, the model may be retrained to incorporate agent-discovered features. Until then, no further model engineering investment.

## Open Questions

### Strategic (resolve during Phase 2 interactive evaluation)

- **What types of edge actually work?** Information gaps (news before the market adjusts), structural biases (liability shading, acca distortion, big-team public loading), or both? We don't know yet — the agent should explore all angles and we evaluate what produces CLV.
- **How should the agent use volume data?** Polymarket orderbook imbalance, FPL transfer spikes — these are proxies for public sentiment, but the mapping from "public is piling on X" to "bet against X" is not straightforward. Bookmakers already shade for this. The question is whether they shade *enough*.
- **What does the conviction framework actually look like?** The tiers above are placeholders. During interactive evaluation we'll learn what signals reliably precede profitable bets and refine accordingly.
- **Execution venue strategy**: Soft bookmakers (better prices but account-limiting risk) vs Betfair Exchange (durable but efficient pricing). May depend on edge type — information edges may need exchange speed, structural edges may tolerate slower sportsbook execution.

### Tactical

- **Lineup data source reliability**: BBC Sport, ESPN, club Twitter — which is fastest and most parseable for the agent?
- **Agent memory depth**: Is one brief per checkpoint per match enough, or does the agent need to reference briefs from previous matchdays (e.g. "last time this team played after CL midweek")?
- **Cost at scale**: 10 matches × 2 checkpoints × 38 matchdays = ~760 agent sessions/season. Estimate $0.50-1.00/session = $380-760/season. Monitor actual usage.
- **Reactive trigger threshold**: 3% implied probability is a starting point. Too sensitive = noise, too conservative = missed moves. Calibrate from data.
- **Betfair API tier**: Basic free API (delayed) vs Exchange Streaming (real-time). Delayed is likely sufficient for pre-match betting.
