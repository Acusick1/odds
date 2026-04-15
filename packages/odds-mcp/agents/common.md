# Common Agent Rules

Cross-sport workflow rules that apply to every checkpoint and every sport. Loaded alongside the sport-specific base and checkpoint files.

## Existing Briefs

Before triaging or researching any fixture, call `get_match_brief` for every event on the slate. Skip games that already have a recent brief for the current checkpoint. This avoids duplicate research and wasted web searches.

## Parallelism

Maximise parallel tool calls and sub-agent research. Once games are selected for deep research, research them concurrently — do not work through games sequentially.

## Early Games (Checkpoint 1 only)

After identifying the slate, skip any game that has already started. Partition the remaining selected games by start time relative to now. If any game starts within 6 hours, there will be no CP2 opportunity before kickoff/start. Complete the full CP1 workflow on all other games first, then read `checkpoint2.md` for the current sport and run its decision flow on the early games.

## Scrape Freshness

`refresh_scrape` enqueues a background job that takes up to 5 minutes. After kicking off a scrape, start a Monitor on `uv run odds scrape job-status <id> --wait` to be notified on completion, then continue with other work. Do not pull spread data for decision-making until the monitor reports completion.

## Observations Log

Each sport has an `observations.md` file in its agent directory (e.g. `agents/mlb/observations.md`). Read it at the start of every session. At the end of every session, append any new observations — bookmaker patterns, edge-type performance, market structure notes, tool gaps, or anything else that would help future sessions. Each entry should include the date, sample size, and confidence level. Update or graduate existing entries as evidence accumulates.

## Do Not Write Memories

Never create or update memory files during agent workflows. Report issues and observations back to the user so they can fix the agent system (prompts, checkpoint files, tool behavior) directly.
