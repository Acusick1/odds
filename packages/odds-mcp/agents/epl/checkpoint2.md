# Checkpoint 2: Decision (KO minus 90 minutes)

## Goal

Load your Checkpoint 1 briefs, check for new information (especially confirmed lineups and price movements), and make a bet/skip decision for each match.

Confirmed lineups typically drop at KO-60 to KO-75. Check for them but don't block on them — proceed with other analysis and circle back.

## Constraints

- Load the Checkpoint 1 brief first — don't repeat research you've already done.
- Check `get_portfolio` before sizing any bet.
- Skip matches that already have a recent decision brief.

## Brief Format

Save with `save_match_brief(checkpoint="decision")`:

```
CHECKPOINT 1 RECAP: [one-line summary of what you flagged]
PRICE MOVEMENT SINCE CHECKPOINT 1: [sharp price then vs now]
LINEUP NEWS: [confirmed XI or notable absentees]
EDGE ASSESSMENT: [specific edge identified, or "no edge"]
DECISION: [BET / SKIP]
If BET: [selection, odds, bookmaker, stake, conviction tier, full reasoning]
If SKIP: [one-line reason]
```
