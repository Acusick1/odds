# Checkpoint 1: Context Building (day before match)

## Goal

Build a structured brief for each upcoming match. Identify what could create an edge by matchday and flag items to revisit at Checkpoint 2.

## Constraints

- No bets at this checkpoint.
- 1-3 web searches per match. If you can't find anything interesting with targeted searches, move on.
- Skip matches that already have a recent context brief (check with `get_match_brief`).

## Brief Format

Save with `save_match_brief(checkpoint="context")`:

```
SHARP PRICE: [home/draw/away implied probs from sharp bookmaker]
SHARP-SOFT SPREAD: [notable divergences, which bookmaker, which direction]
TEAM NEWS: [key findings — injuries, suspensions, rotation risk, manager quotes]
PRELIMINARY VIEW: [interesting / not interesting / watching]
WATCH-FOR AT CHECKPOINT 2: [specific items — e.g. "Saka fitness test tomorrow", "Check if rotation for CL"]
```
