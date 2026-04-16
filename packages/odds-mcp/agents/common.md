# Common Agent Rules

Cross-sport workflow rules that apply to every wake-up and every sport. Loaded alongside the sport-specific base file.

## Existing Briefs

Before triaging or researching any fixture, call `get_slate_briefs` to see the latest decision and summary for every event on the slate. Only call `get_match_brief` for specific events you decide to research further. This avoids loading full brief text for the entire slate.

## Parallelism

Maximise parallel tool calls and sub-agent research. Once games are selected for deep research, research them concurrently — do not work through games sequentially.

## Scrape Freshness

`refresh_scrape` enqueues a background job that takes up to 5 minutes. After kicking off a scrape, start a Monitor on `uv run odds scrape job-status <id> --wait` to be notified on completion, then continue with other work. Do not pull spread data for decision-making until the monitor reports completion.

## Scheduling

After completing your workflow, decide when you should next wake up. If there is developing news, a watch-for item to check, or a match approaching KO, call `schedule_next_wakeup` with an appropriate delay and reason. If there is nothing to check back on, skip the call — the scheduler will wake you at the default fixture-proximity interval.

## Observations Log

Each sport has an `observations.md` file in its agent directory (e.g. `agents/mlb/observations.md`). Read it at the start of every session. At the end of every session, append any new observations — bookmaker patterns, edge-type performance, market structure notes, tool gaps, or anything else that would help future sessions. Each entry should include the date, sample size, and confidence level. Update or graduate existing entries as evidence accumulates.

## Do Not Write Memories

Never create or update memory files during agent workflows. Report issues and observations back to the user so they can fix the agent system (prompts, tool behavior) directly.
