# Checkpoint 1: Morning Research (~14:00 UTC / 10 AM ET)

## Goal

Triage today's slate, select 3-5 games for deep research, and build context briefs. Probable pitchers are typically confirmed by this time.

## Early-Game Rule

After identifying the slate, partition selected games by start time relative to now:

- **Early games** (starting within 6 hours): Run the full decision flow inline — there will be no CP2 opportunity before first pitch. Save both a context and decision brief.
- **Evening games** (starting 6+ hours out): Build context briefs only. Decisions deferred to CP2.

## Triage Criteria

Select games based on: pitching mismatches, team streaks, bullpen fatigue patterns, weather at outdoor parks, public betting tendencies, or anything that suggests a potential market inefficiency. Skip the rest.

## Constraints

- No bets on evening games at this checkpoint.
- 2-4 web searches per game. If you can't find anything interesting, move on.
- Skip games that already have a recent context brief.

## Context Brief Format (evening games)

Save with `save_match_brief(checkpoint="context")`:

```
TRIAGE REASON: [why this game was selected for deep research]
SHARP PRICE: [home/away implied probs from Betfair Exchange]
SHARP-SOFT SPREAD: [notable divergences, which bookmaker, which direction]
PITCHING MATCHUP: [SP1 vs SP2, recent form, key splits]
TEAM NEWS: [injuries, bullpen availability, lineup changes]
WEATHER: [if outdoor park — wind, temp, relevance to totals]
PRELIMINARY VIEW: [interesting / not interesting / watching]
WATCH-FOR AT CHECKPOINT 2: [specific items — e.g. "Confirm SP not scratched", "Check if bullpen arm available"]
```

## Decision Brief Format (early games)

Save with `save_match_brief(checkpoint="decision")`:

```
CONTEXT RECAP: [one-line summary of what you found above]
STARTER CONFIRMED: [yes/no, any late changes]
LINEUP NEWS: [notable lineup changes, absences, or "not yet posted"]
WEATHER UPDATE: [if relevant]
EDGE ASSESSMENT: [specific edge identified, or "no edge"]
DECISION: [BET / SKIP]
If BET: [selection, odds, bookmaker, stake, conviction tier, full reasoning]
If SKIP: [one-line reason]
NOTE: Early game — full analysis at CP1 (no CP2 before first pitch).
```
