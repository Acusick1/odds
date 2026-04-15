# Checkpoint 2: Pre-Game Decision (~22:00 UTC / 6 PM ET)

## Goal

Load your Checkpoint 1 briefs, verify starters are confirmed, check for late changes and price movements, and make bet/skip decisions. Lineups typically drop 1-3 hours before first pitch.

## Constraints

- Load the Checkpoint 1 brief first — don't repeat research you've already done.
- Check your "watch-for" items from Checkpoint 1.
- Check `get_portfolio` before sizing any bet.
- Skip games that already have a recent decision brief.

## Brief Format

Save with `save_match_brief(checkpoint="decision")`:

```
CHECKPOINT 1 RECAP: [one-line summary of what you flagged]
PRICE MOVEMENT SINCE CHECKPOINT 1: [sharp price then vs now]
STARTER CONFIRMED: [yes/no, any late changes]
LINEUP NEWS: [notable lineup changes or absences]
WEATHER UPDATE: [if relevant]
EDGE ASSESSMENT: [specific edge identified, or "no edge"]
DECISION: [BET / SKIP]
If BET: [selection, odds, bookmaker, stake, conviction tier, full reasoning]
If SKIP: [one-line reason]
```
