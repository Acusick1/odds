---
status: accepted
---

# Make EPL 2022-24 development data and seal only the five leagues' 2025-26

## Context

[ADR-0021](0021-sealed-holdout-and-one-shot-gate.md) sealed EPL 2022-23 and 2023-24 as holdout and named EPL 2020-22 as development data, beside EPL 2024-26 and the five leagues' 2024-25. Round 1's cell selection made EPL 2020-22 one of its three development blocks ([#821](https://github.com/Acusick1/odds/issues/821#issuecomment-6769611091)).

EPL 2020-22 was never bought. The Odds API's EPL Betfair Exchange `h2h` starts on 2023-04-01 ([#729](https://github.com/Acusick1/odds/issues/729)), so the EPL manifest `backfills/odds-api-epl-2022-26.yaml` starts there ([#859](https://github.com/Acusick1/odds/issues/859)). FDUK has a Betfair Exchange close only from 2024-25. EPL 2020-22 therefore has no exchange price at any decision time and no close, and its block could never hold a bet that counts.

When the decision was made, no cell had been selected and no holdout had run: `candidates/selections/` and `candidates/holdout.jsonl` did not exist. The EPL 2022-24 sections of the [EPL backfill audit](../../backfills/odds-api-epl-2022-26.audit.md) report counts, coverage and ages only, and the substitute-close comparison ([#882](https://github.com/Acusick1/odds/pull/882)) ran on EPL 2024-26, so no EPL 2022-24 price has been examined.

## Decision

Made with the user on 2026-11-13. It supersedes ADR-0021's EPL sealing only.

- **EPL 2022-23 and 2023-24 are development data.** These are the days from 2022-08-01 to 2024-07-31. The data starts on 2023-04-01.
- **The sealed holdout is the five leagues' 2025-26 only:** La Liga, Serie A, the Bundesliga, Ligue 1 and the Championship, from 2025-08-01 to 2026-07-31. EPL has no sealed window. A pre-registration that lists EPL runs its holdout on its other leagues and records EPL as a league without a window. A pre-registration that lists only EPL has no holdout run.
- **The switch `odds_analytics.holdout.EPL_2022_24_SEALED` is removed.**
- **Round 1's cell selection.** A development block "EPL 2022-24", from 2022-08-01 to 2024-07-31, replaces "EPL 2020-22". The same condition applies to it: a positive mean EV when it has at least 20 bets with a close. The blocks are EPL 2022-24, EPL 2024-26 and the five leagues' 2024-25. The grids are unchanged.
- The rest of ADR-0021 stands: the seal in the replay library, the frozen grid and later selection, the one-shot ledger and the gate.

## Consequences

- EPL 2022-24 bets are scored only through the Odds API substitute close that ADR-0023 proposes ([#882](https://github.com/Acusick1/odds/pull/882), [#730](https://github.com/Acusick1/odds/issues/730)), once it is accepted and built. Until then those bets have no close and don't count. The EPL 2022-24 block then has no bets with a close, so it never blocks a cell's eligibility, and it adds nothing to the 50-bet and 30-event minimums. EPL 2022-23 before 2023-04-01 stays without a close either way.
- The holdout is smaller. Every round-1 idea pools EPL with the five leagues, so each still has a holdout run, on the five leagues alone.
- An EPL-only idea goes from development data straight to the forward test.
- A development replay, the substitute-close comparison and the backfill audit can now read EPL 2022-24 prices. The [EPL backfill audit](../../backfills/odds-api-epl-2022-26.audit.md) reported those seasons as sealed. A rerun would add their price-derived checks.
- The round-1 comment that the pre-registrations link to as their selection rule still lists "EPL 2020-22". This ADR amends the block list, and `odds_analytics.cell_selection.DEVELOPMENT_BLOCKS` implements the amended list.

## Evidence

- Issues: [#884](https://github.com/Acusick1/odds/issues/884) (this change), #820 (ADR-0021's decision), #821 (round 1's selection rule), #729 (probes: EPL Betfair Exchange `h2h` from 2023-04-01), #859 (the Odds API month plan), #730 and #882 (the substitute close).
- Data: `backfills/odds-api-epl-2022-26.yaml` and the [EPL backfill audit](../../backfills/odds-api-epl-2022-26.audit.md#summary).
- Code: `odds_analytics.holdout.holdout_windows`, `odds_analytics.cell_selection.DEVELOPMENT_BLOCKS`.
