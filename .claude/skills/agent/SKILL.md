---
description: Run the betting agent's wake-up workflow for one sport, or review its recent performance.
disable-model-invocation: true
---

Determine the sport from the first word of the arguments, then strip that word:

- `mlb` or `baseball_mlb`: the sport is `mlb`.
- `epl` or `soccer_epl`: the sport is `epl`.
- Any other sport key (one containing `_`, such as `basketball_nba`): stop and report that no agent exists for that sport.
- Anything else, or no arguments: the sport is `epl`, and the first word is not stripped.

Read both of these files and follow their instructions:
1. `packages/odds-mcp/agents/common.md` — cross-sport workflow rules (apply to every run)
2. `packages/odds-mcp/agents/{sport}/base.md` — sport context, betting rules, and wake-up workflow

Run the betting agent workflow autonomously — do not ask for confirmation, just execute. Follow the Wake-Up Workflow section in the base file: orient, settle, triage, research, brief, decide.

If arguments remain after the sport, interpret them as special actions:
- `review` — Load recent match briefs and paper trades. Summarise performance and reasoning quality. No research or bets.

Examples:
- `/agent` — EPL wake-up workflow
- `/agent mlb` or `/agent baseball_mlb` — MLB wake-up workflow
- `/agent review` — EPL performance review
- `/agent mlb review` — MLB performance review

Arguments: $ARGUMENTS
