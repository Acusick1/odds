---
status: proposed
---

# Make a candidate's book set part of its definition

## Context

A mechanical candidate backs the longest price among the UK retail roster (`odds_core.bookmakers.UK_RETAIL_BOOKMAKERS`), one set shared by every rule. Its backtest reads Odds API snapshots and its forward test reads OddsPortal snapshots, and the two sources quote different parts of the roster. The Odds API's `uk` region carries 12 roster books. OddsPortal's UK view carries about 20, including bet365, midnite, betmgm, betano, spreadex and bwin, none of which the Odds API carries in that region. No request parameter adds them (#900).

#900 paired each OddsPortal snapshot with the Odds API snapshot within 3 minutes of it, from KO−6h to kick-off, and ran the rules on each source's state alone (143 EPL pairs over about 40 matches, 1,232 MLB pairs):

- On the full roster, `exchange_gap` at a 1% threshold fired on 21.7% of EPL OddsPortal states and 2.8% of Odds API states. 26 of the 33 bets that fired only on OddsPortal were at midnite.
- Restricted to the books both sources quoted, the two agreed on 97–99% of EPL decisions for every rule tested.
- The EPL exchange prices of the two sources agreed within 1 pp in 81–95% of comparisons, so the difference is the roster, not the sharp price.
- 7 of the first 13 `polymarket-gap-epl` paper trades were at books the Odds API backfill does not quote.

A forward test on the full roster therefore places most of its bets at books its backtest never saw, so the backtest says little about its bet volume or EV.

## Decision

- **Each mechanical candidate declares its book set**, `books`, by name from `odds_core.bookmakers.BookSet`: `uk_retail`, the full roster, or `odds_api_uk`, the roster books the Odds API's `uk` region carries. A rule takes retail prices only from the candidate's set. The Betfair Exchange is read as the sharp price whatever the set.
- **The set is part of the version.** Changing it is a new version, which the version check enforces like any other key. Adding `books: uk_retail` to a file that predates the field is not a change, because that is the set every candidate already used, so the live candidates keep their versions and forward tests.
- **One set everywhere.** The live runner, the replay backtest and the parity check read the candidate's set. The backtest reports how often its snapshots quoted each book of the set.
- **Other sets are information only.** The candidate report replays a version's recorded decisions with other book sets, and with each of its books left out, and the web UI shows the results beside the candidate's own. No checkpoint, verdict or family count reads them. To act on one, register a new version with that set and give it its own forward test.
- **The family is unchanged.** Versions that differ only in book set share a family under [ADR-0022](0022-pooled-league-candidates.md), so each raises the other's k.

## Considered options

- **One global roster (the status quo).** Keeps every candidate on the books actually bet with, but backtest and forward test keep diverging, as #900 measured.
- **Restrict every live candidate to the books the backfill covers.** Restores parity, but never measures what the other books are worth, though they are books the user bets with.
- **A live Odds API feed for the forward test.** Costs quota every month and still lacks the books OddsPortal adds.
- **The exchange price from another source.** #900 found the EPL exchange prices of the two sources already agree, so this would not change EPL decisions.
- **A free-form book list in each candidate file.** More flexible, but a copied list can drift from the roster. Named sets are defined once in code.

## Consequences

- A candidate on `uk_retail` cannot be fully backtested: its backtest covers only the books the Odds API carries, and the coverage report shows how much. The evidence for the other books comes from its forward test alone.
- A candidate on `odds_api_uk` reproduces its backtest's decisions live, apart from the price differences between sources that #900 measured, but bets at fewer books than the user can use.
- A replay with another set covers only the decisions the version took. A decision missed for want of inputs recorded no snapshot, so a larger set cannot recover it.
- Running both sets of one idea as separate versions raises both versions' k.

## Evidence

- Issues: #900 (source comparison), #901 (decision and implementation).
- [Odds API vendor facts](../../wiki/collection/odds-api-facts.md): the books the `uk` region returns.
