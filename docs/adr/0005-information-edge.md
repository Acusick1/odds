---
status: deprecated
---

# Treat the agent's information synthesis as the primary edge

Deprecated: strategy lives in [Goal and strategy](../../wiki/foundations/goal-and-strategy.md).

## Context

When this decision was recorded in April 2026, `docs/BETTING_AGENT.md` described the CLV model as "R² = 3-6%, backtest ROI +3.4% at p = 0.26".

- On NBA the CV R² was about 0.036 (#170; the sport is given in `docs/MODELING.md` at 7887039, "What the model predicts"). Flat betting at bet365's roughly 4.5% vig lost money at every threshold from 0.005 to 0.03 (#170, `experiments/results/exp7_backtest_sim/FINDINGS.md`). The only profitable threshold, 0.05, made +3.36% ROI on 56 bets at p = 0.26, which is not significant.
- Shopping across four UK bookmakers cut the effective vig from 4.13% to 1.59%, but every threshold up to 0.03 stayed unprofitable (#171, `experiments/results/exp7b_line_shopping/FINDINGS.md`).
- The model's strongest feature, `retail_sharp_diff`, is the sharp-retail spread, and the agent can read that spread directly (`docs/BETTING_AGENT.md`, "Role of the XGBoost Model"). Today it does so through `odds_mcp.server.find_retail_edges`.

## Decision

The hypothesised edge is the breadth and speed of an LLM agent's information synthesis, "not prediction modeling". The agent monitors bookmaker odds, exchange order books, public sentiment signals, lineup announcements, press conferences and fixture context, and reasons across all of them. This thesis is set out in `docs/BETTING_AGENT.md`, which #287 cites (2026-04-12) and which was first committed in 3f74022 (2026-04-13). #290 (merged 2026-04-12, closing #287) wrote the agent prompt for it. That prompt names information gaps, structural biases and cross-venue divergence as edge types to explore. The CLV model is "demoted to a supplementary signal", and `get_predictions` stays available as a sanity check (`docs/BETTING_AGENT.md`; #290).

## Consequences

- The record does not say why the edge moved from the model to the agent. `docs/BETTING_AGENT.md` gives the model's figures in the section that demotes it but states no reason, and no discussion behind #287 or #290 is recorded.
- The decision is in force:
  - The Thesis in `packages/odds-mcp/agents/common.md` treats predictive models as "supplementary sanity checks, never a trading signal on their own".
  - `odds_lambda.jobs.fetch_oddsportal` scores predictions inline through `odds_lambda.jobs.score_predictions.score_events`.
  - `odds_mcp.server.get_predictions` reads the stored predictions.
- The recorded plan was to stop investing in model engineering until the information edge proved viable, and then possibly retrain the model on features the agent discovered (`docs/BETTING_AGENT.md` at 7887039). Model work continued anyway. On 2026-04-23 an EPL re-test reached CV R² about 0.058 (#361), and a tabular + match_stats model was deployed (`docs/MODELING.md` at 7887039, "Deployed model").
- Effort moved to agent tooling. Phase 2 of the recorded rollout planned match briefs and MCP tools (`docs/BETTING_AGENT.md`). The code now has the MCP server, match briefs, paper trades and the `agent-run` job (`odds_lambda.jobs.agent_run`, #322).
- No evaluation of the thesis is recorded. The recorded test compares the agent with mechanical-spread, CLV and random baselines. It passes only if the CLV rate is above 50% at p < 0.05 over 100 or more bets, with positive ROI after commission (`docs/BETTING_AGENT.md`, "Evaluation Criteria"). No code runs that comparison yet (#422). Results exist only as paper trades, read through `odds_mcp.server.get_portfolio` and `odds paper`.
- Since 250733a (2026-06-01, "update local default jobs", no reason given), `agent-run` is commented out of the default `odds_core.config.SchedulerConfig` `bootstrap_jobs`. The agent runs only when that job is added back or when `odds agent run` is called.

## Evidence

- PRs and issues: #170, #171, #287, #290, #322, #361, #422.
- Commits: 3f74022 (`docs/BETTING_AGENT.md` first committed) and 250733a (`agent-run` removed from the default bootstrap).
- Results: `experiments/results/exp7_backtest_sim/FINDINGS.md`, `experiments/results/exp7b_line_shopping/FINDINGS.md`.
- Prior text: `git show 3f74022:docs/BETTING_AGENT.md` ("Role of the XGBoost Model", "Phased Rollout", "Evaluation Criteria"); `git show 7887039:docs/MODELING.md` ("Deployed model"); `AGENTS.md` at `7887039` ("Strategic Goal").
- Code and prompts: `packages/odds-mcp/agents/common.md`, `odds_lambda.jobs.score_predictions.score_events`, `odds_mcp.server.get_predictions`, `odds_core.config.SchedulerConfig`.
- Wiki: [The betting agent](../../wiki/agent/betting-agent.md), [About the system and its pivots](../../wiki/foundations/project-overview.md), [CLV model findings](../../wiki/evaluation/clv-model-findings.md), [Betting simulation](../../wiki/evaluation/exp-betting-simulation.md).
