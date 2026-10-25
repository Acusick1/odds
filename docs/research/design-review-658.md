# Review of the #658 data-model decisions (#659, #660, #662, #663, #664)

Code read at `b8591833` (main, includes #649). Paths are relative to `/home/andrew/code/odds/worktrees/review-658`. "Confirmed" means read in code; "plausible" means inferred and not verified against data or docs.

Findings are ranked most severe first.

---

## 1. #649's Betfair close scheduling reads two fields that #662 drops (confirmed)

**Finding.** `next_betfair_kickoff` finds an event's Betfair start from `raw_data->bookmakers[0]->betfair_meta->market_start_time` on snapshots selected by `api_request_id LIKE 'betfair_api_%'`. #662 drops `api_request_id`, and its snapshot metadata columns (`in_play`, `market_status`, `total_matched`) leave out `market_start_time`.

**Evidence.** `packages/odds-lambda/odds_lambda/betfair/kickoff.py:70-84` (JSON path plus the `api_request_id` filter), `betfair/adapter.py:226-228` (writes `market_start_time`), `collection_calendar.py:223-250` (`close_kickoff_lookup`). #662's resolution comment was posted at 11:22:40Z, two minutes after #649 merged at 11:20:27Z, so it did not see the new field.

**Consequence.** After the cutover, `parse_betfair_start` gets `None` for every event and falls back to `commence_time`. The early in-play switch then gets no close run again, so #640 regresses silently and nothing errors.

**Fix.** Add a nullable `market_start_time` snapshot column for the Betfair source (or store it on the Betfair external id in `event_external_ids`), and select on `source = 'betfair'`. Add it to the #665 port list for `betfair/kickoff.py`.

## 2. `quotes()` cannot serve the live runner, parity or the data replay as specified (confirmed)

**Finding.** The specified signature `quotes(session, event_ids, market, *, sources, time_bases, bookmakers, sides, since, until)` takes neither a `known_at` (stored-by) bound nor snapshot ids. #664 itself says "Parity replays load quotes by those ids". #662 also does not say whether `created_at` survives on snapshots.

**Evidence.**
- `candidate_rules/market_state.py:151-158`: `load_market_state` filters `OddsSnapshot.created_at <= known_at`. The live runner passes its run time, and the data comparison passes the decision's `known_at`.
- `odds_analytics/candidate_parity.py:100-107`: the recorded-input replay selects by `input_snapshot_ids`.

**Consequence.** As specified, the live runner would see rows stored after its run, and parity cannot be built.

**Fix.** Keep `odds_snapshots.created_at` and copy it from the old snapshot in the backfill. Split Odds API snapshots inherit their parent's `created_at`, and `snapshot_payloads.received_at` is not a substitute. Add `known_at: datetime | None` and `snapshot_ids: Sequence[int] | None` to `quotes`.

## 3. "Latest quote per book" changes exchange-gap-epl's retail semantics and conflicts with retail-EV's (confirmed)

**Finding.** #664 says "a pure builder picks the latest quote per book at or before t" and that the live runner and replay share it. Today `MarketState` takes retail by whole snapshot: per kind, only the latest snapshot that quotes the market counts, and a book absent from it is absent from the state.

**Evidence.**
- `market_state.py:74-80, 95-122`.
- `wiki/evaluation/candidate-rules.md:37`: "A book absent from that snapshot is absent from the state, however recently an earlier snapshot quoted it".
- #641/#642, merged today, tightened this same rule.
- `retail_ev.py:341-385` deliberately differs. It takes retail per book, includes FDUK opening (`UPPER_BOUND_TIME_KINDS`) rows, and takes their exchange baseline from the same FDUK snapshot. `market_state.py:81-85` documents the difference.

**Consequence.**
- Under per-book selection, a book that OddsPortal stopped showing (pulled or struck lines; see `agents/common.md:123`) keeps its older price inside `MAX_RETAIL_AGE`, and the rule may bet it. That is a rule change beyond the v2 hash bump.
- If retail-EV is moved onto the `MarketState` builder, its FDUK-opening rows disappear, which is most of its history (ADR-0012:75). If it isn't, two builders exist, which contradicts "the same builder".

**Fix.** State the builder as "latest snapshot per (source, time_basis) that quotes the market, then that snapshot's books". This is derivable from `Quote.snapshot_id` and time. Name retail-EV's builder as a separate, documented one, or decide explicitly to change it.

## 4. Write-time canonicalisation and outcome mapping have no collision or failure policy, and lose recoverable data (confirmed)

**Finding.**

(a) Two raw keys in one snapshot can collapse to one canonical key. Examples: Odds API `betfair_ex_uk` and `betfair_ex_eu` both become `betfair_exchange`; OddsPortal `7bet`/`7betuk` and `betmgm`/`betmgmuk` collapse too. The `prices` unique key (snapshot, bookmaker, outcome, line, side) then rejects the second row. Today `book_prices` silently takes the first.

(b) Outcomes are mapped to home/away by name equality with the event's team strings. A misnamed event therefore loses its prices permanently at write. Today they stay in `raw_data` and become readable after `odds events repair-names`.

(c) `repair-names` itself reads Betfair `runner_name` from `betfair_meta.runners`, and nothing in #663 keeps it.

**Evidence.**
- `data_fetcher.py:31-43`.
- `bookmakers.py:19-28` (`BOOKMAKER_ALIASES`).
- `market_prices.py:42-47` (first entry wins).
- `market_prices.py:32-36, 53-55` (name match, unmatched names skipped).
- `storage/event_name_repair.py:47-63`. `odds-cli/commands/events.py:136` exposes it.

**Consequence.** Writer exceptions or lost rows on the first collision. `events repair-names` breaks, and the archive would have to be read outside its "re-parse only" rule. Outcome identity comes from a mutable field (event team names).

**Fix.** Map outcomes from the source's own home/away position or role at write, not by string equality with `events.home_team`. Specify first-wins with a data-quality log for alias collisions. Decide whether to keep runner names (for example in a per-source participant mapping) or to delete or port `repair-names` against the new resolver.

## 5. Deriving `fetch_tier` on read reclassifies FDUK openings as "closing" (confirmed)

**Finding.** FDUK opening snapshots are stored with a tier that "never [is] the closing tier, which selects closing prices". A generic derivation from (snapshot_time, commence_time) gives an opening stamped less than 3 h before kick-off the closing tier.

**Evidence.**
- `scripts/ingest_football_data_uk.py:207-224` (`opening_snapshot_time`, `opening_fetch_tier`).
- `feature_groups.py:152-154`: the closing snapshot is the last one whose `fetch_tier` equals the closing tier.

A Friday 19:30 UK kick-off with the Friday 17:00 deadline is 2.5 h out.

**Consequence.** The closing snapshot selection in `feature_groups` changes, and so does tier coverage. Separately, every SQL tier filter needs the tier thresholds re-expressed in SQL, which duplicates `FetchTier`:
- `storage/readers.py:490-615`
- `odds_analytics/quality_metrics.py:397-454`
- `scripts/check_tier_distribution.py:31-47`
- `scripts/check_game_tier_coverage.py:34-90`
- `gap_backfill_planner.py:446`

**Fix.** Select closes by `time_basis` and `source`, never by tier. Derive tier in one place: either a Python function only, or a SQL function or view generated from `FetchTier`. Delete or port the tier scripts and readers explicitly.

## 6. The unique snapshot key will hit existing rows, and decision ids dangle (confirmed / plausible)

**Finding.** Several existing data shapes violate (event_id, source, market, snapshot_time), and #662 states no write-conflict policy.

**Evidence.**
- **Same-time groups are left in the data on purpose (confirmed).** `same_time_pairs.py` keeps `REFUSED` groups, those referenced by `CandidateDecision.input_snapshot_ids` (`:66-80, :219`), and `CONFLICTING_LEFT` groups with "no established cause" (`:1-10`).
- **OddsPortal historical opening and closing can share a stamp (plausible).** Each is the max over books' first-outcome timestamps (`oddsportal_common.py:196-200`). When no book moved after the latest opening, the two times are equal.
- **Twin merges in the migration move snapshots onto one event.**
- **FDUK re-ingest replaces snapshots by `api_request_id`** (`ingest_football_data_uk.py:548, 563`). A re-ingest after `commence_time` moves stamps the close at the new kick-off, so a key-based upsert leaves two `close` snapshots. `event_close` then takes the later one.
- **`input_snapshot_ids` is an `ARRAY(Integer)`, not an FK** (`paper_trade_models.py:61-65`). The per-market split and re-keying change every snapshot id, and exchange-gap-epl v1 starts recording on 2026-10-26 (`candidates/exchange-gap-epl.yaml`).

**Consequence.** The migration fails, or arbitrarily drops snapshots that decisions used. v1 parity reports every input as missing.

**Questions for #665.**
- Run a pre-migration count grouped by the new key, including source merges such as `oddsportal` + `oddsportal_live`.
- Choose a writer conflict policy (`ON CONFLICT DO NOTHING` plus a log, or raise).
- Keep an old→new snapshot id map and rewrite `input_snapshot_ids`, or record that v1 parity is retired.
- Make the FDUK writer replace by (event, source, time_basis), not by the unique key.

## 7. The close is defined in four places, and #663's statement omits the overround filter (confirmed)

**Finding.** #663 states the Betfair close as source `betfair`, observed, not in-play, a back row for every outcome, last within 30 min, else FDUK `close`. The code also drops any price whose back overround is above `MAX_EXCHANGE_OVERROUND` (0.05, ADR-0014).

The health check re-implements the close without that filter, and the candidate report pre-filters with its own SQL. #664 ports neither.

**Evidence.**
- `betfair_close.py:113-114, 205-207`
- `health_monitor.py:300-395` (`check_missing_betfair_closes`)
- `candidate_report.py:223-241` (`close_candidate_snapshots`, filtering on raw_data `source`/`timing.basis`)
- `candidate_backtest.py:103-108`

**Consequence.** An implementation that follows #663's text changes the close. The health alert and the report can drift from the close they check.

**Fix.** One close function over quotes, called by the scorer, report, backtest, health monitor and retail-EV. Put the overround cap in the stated definition.

## 8. Event-id writers and readers the decisions miss (confirmed)

| Code | Dependency | Consequence |
|---|---|---|
| `jobs/fetch_scores.py:139-151`; `odds-cli/commands/fetch.py:153` | Odds API id passed as `Event.id` to `update_event_status` | Every score update logs `event_not_found` and is lost silently (fetch-scores is a default bootstrap job, `config.py:117`) |
| `ingestion.py:131-140` (fetch-odds, default sports EPL+MLB) | Creates or upserts MLB events with the Odds API id and no gamePk | Bypasses the "MLB matches by gamePk" resolver unless ported |
| `training/data_preparation.py:345-353`, `storage/readers.py:86-117` | Picks the data source by `op_%`/`fduk_%` id pattern | Breaks, and after twin merges "an OddsPortal event" means something else |
| `polymarket_odds.py:192-195` (`preferred_event`) | `is_odds_api_event_id` | Prefix deleted |
| `storage/event_name_repair.py:88,120`; `storage/game_pk_backfill.py:12,218` | `EventSource`, `canonical_event_id` | Not in #660's delete list beyond run_candidates/backtest |
| `storage/event_merge.py:183-268` (`EventMerger`) | Keeps the best-ranked event's `commence_time` | The migration's twin merge needs a kick-off authority order, which #660 defers for EPL/NBA |
| `packages/odds-mcp/agents/epl/base.md:11`, `agents/mlb/base.md:21` | Tell the agent to prefer `op_live_*` ids | Prompt instructions become wrong |
| `copy_from_production.py:136-176` | Copies by `Event.id`, with `skip_existing` by id | Surrogate ids from prod collide with locally created ids |
| `health_monitor.py:403-415` | Alert de-dup by event ids in `AlertHistory.context` JSON | Re-alerts once across the cutover (minor) |
| `results_run.py:77`, `jobs/fetch_mlb_results.py:104-165`, `storage/mlb_score_repair.py` | `Event.mlb_game_pk` | Need a join to `event_external_ids` |
| `jobs/fetch_oddsportal_results.py:139` | Matches results by teams ±2 h | Should use the OddsPortal external id |

Other raw_data, `Odds` or column readers that are neither ported nor deleted by name:
- `collection_calendar.py:253-262` and `health_monitor.py:269` (staleness by `raw_data->source`). This needs `(source, time_basis)` because `oddsportal` merges historical and live.
- `odds_analytics/quality_metrics.py` (uses `bookmaker_title`, which #663 drops).
- `storage/readers.py:450-470` (`get_database_stats` counts `Odds`, used by `odds status`).
- `odds-cli/commands/validate.py` and `storage/tier_validator.py`.
- `gap_backfill_planner.py` and `game_selector.py`.
- `polymarket_odds_backfill.py:130` (de-dups on `api_request_id`).
- `retail_ev_evaluation.py:64,256` (`"fduk_opening"` kind string in saved outputs).
- MCP `get_current_odds(include_raw_data)` (`mcp/server.py:143`), which would read the archive.

## 9. `no_game_pk_reason` is settlement state, not only a health-check acknowledgement (confirmed)

**Finding.** #660 moves it to a "health-check acknowledgement record". It is also read as "this event cannot be settled; stop trying".

**Evidence.**
- `storage/unsettled_events.py:27-37`: `find_unsettled_events` excludes these events.
- `storage/game_pk_backfill.py:247-251`: planning skips them.
- `storage/game_pk_backfill.py:514-520`: `mark_no_game_pk`.

**Consequence.** If only the health check reads the new record, settlement and backfill planning lose the exclusion.

**Fix.** Name the record for what it is, for example "event cannot be keyed to gamePk", and have settlement and planning read it. Alternatively, state that the health check owns the exclusion for all three callers.

## 10. The resolver's lock key on UTC date splits evening games in the Americas (plausible)

**Finding.** The advisory lock is on (sport, home, away, date). MLB and NBA evening games start around 23:00–03:00 UTC, and Betfair's start can lead `commence_time` by up to 3 h (`betfair/kickoff.py:15-17`, 2026-10-24 World Series). Two writers can take different date keys for one game, both miss each other's uncommitted event and both create one.

**Consequence.** For MLB, the unique `gamePk` external id catches the race through an IntegrityError path, which `writers.py:174-197` already handles. NBA has no such backstop, so it gets duplicate events, and #660 deletes the twin collapse.

**Fix.** Lock on (sport, home, away) only, or on both adjacent dates.

## 11. Legacy Polymarket NBA migration has unrepresentable rows (confirmed from schema; data volumes not checked)

- **Market types.** `PolymarketMarket.market_type` includes `player_prop` and `other` (inventory, `polymarket_models.py`). These don't fit the `home/draw/away/over/under` outcome enum. Decide whether to drop them.
- **Invalid prices.** Prices are probabilities, so p = 0 or 1 on settled markets gives division by zero or price = 1.0, and the `price > 1.0` CHECK rejects the row. Decide whether to clamp, drop or log.
- **Unmatched events.** `polymarket_events.event_id` is nullable (`polymarket_models.py:35`). Unmatched games have no event to attach a snapshot to. Count them before the migration, since the 255 games are the reason for migrating.
- **Order-book depth.** `PolymarketOrderBookSnapshot.raw_book` levels beyond the top of book have no `prices` home and no `snapshot_payloads` source. Decide whether to archive them as payloads.

## 12. `time_basis` classification gaps (plausible)

- **Polymarket backfill.** These are not source stamps: they are as-of resamples at our grid times, taking the latest observation at or before each point within `MAX_STALENESS` (`polymarket_odds.py:265-296`). They behave like `observed`. The label is harmless only if `as_of` treats `source_stamped` and `observed` alike. Specify that, or classify them as `observed`.
- **Odds API historical.** These are stamped at the requested time (`backfill_executor.py:267-279`). Per the API docs (from memory, not verified), the snapshot returned is at or before that time, so the stamp is an upper bound of up to about 5–10 min.
- **Odds API live vs historical in stored data.** They carry no tag and cannot be told apart. #665 needs a heuristic such as the `created_at − snapshot_time` gap.
- **OddsPortal historical opening.** The stamp is the max over books (`oddsportal_common.py:196-200`), so some books' opening prices were already superseded at the stamp. Per-row `changed_at` preserves the truth, and readers should prefer it.
- **`api_request_id` is still needed during migration.** It is the only way to tell Polymarket history (`"polymarket_history"`) from live (`polymarket_odds_backfill.py:79`). Drop it after the backfill, not before.

## 13. Smaller inaccuracies in the decisions (confirmed)

- **The OddsPortal market is in the payload.** #659 and #662 say it lives only in `api_request_id`, but historical OddsPortal `raw_data` carries it in `bookmakers[].markets[].key = db_market` (`oddsportal_common.py:203-212`, `ingest_oddsportal.py:380`).
- **Odds API `h2h_lay` is soccer.** It is not relabelled to `1x2` (`utils.py:27-46` relabels only `h2h`), and Betfair-via-Odds-API is UK-region, so it is almost certainly EPL. Its lay rows belong to the `1x2` snapshot and include Draw. #663 says the "h2h snapshot".
- **Exact-time uniqueness dedupes less than #662 implies.** "Same-time pairs become impossible" holds only for exact timestamps. OddsPortal live stamps each market with its own `datetime.now(UTC)` (`jobs/fetch_oddsportal.py:192`). Two scrapes (scheduler plus MCP `refresh_scrape`) produce near-identical, non-conflicting snapshots, and the key does not dedupe them. That is fine, but the claim that the key "does the dedupe job `api_request_id` did" holds only for backfills.

## 14. Hashing `odds_core.quotes` ties MCP and analytics changes to candidate versions (confirmed; a trade-off, not a break)

**Finding.** `quotes` is also the MCP and retail-EV reader. Any change to it fails `test_registered_candidates_are_valid` in CI, and at runtime `run_candidates.py:102-106` skips the candidate with an alert until a new version is registered.

**Consequence.** Every reader tweak made for MCP restarts the forward test.

**Fix.** This is the user's choice. A narrower option is to hash the pure builder plus a small loader module dedicated to rules.

## 15. Feasibility and anti-patterns (confirmed unless marked)

**Feasibility.**
- **NULLS NOT DISTINCT is supported everywhere it runs.** Production and local Postgres are `postgres:15` (`docker-compose.yml:3`; `wiki/foundations/databases-and-environments.md:40-42`), and NULLS NOT DISTINCT exists from 15.0.
- **Tests.** They use py-pglite 0.5.3 bundling PGlite 0.3.3, which is PG 17 (plausible).
- **SQLAlchemy.** 2.0.44 should accept `postgresql_nulls_not_distinct=True` on the constraint or index (plausible; verify alembic autogenerate renders it).
- **Re-keying FKs.** It touches exactly the nine FK tables #660 lists (`models.py:136,177,237`, `match_brief_models.py:58`, `paper_trade_models.py:48,90`, `game_log_models.py:52`, `injury_models.py:54`, `polymarket_models.py:35`). `input_snapshot_ids` and alert-context ids are non-FK references that also need rewriting (see items 6 and 8).

**Anti-patterns.**
- **A duplicated definition:** the close in four places (item 7).
- **Another duplicated definition:** tier thresholds in Python and SQL (item 5).
- **Identity from mutable fields:** outcome identity, FDUK matching and the lock key all use team-name strings (items 4 and 10), and `events repair-names` renames them.
- **Side conflates two concepts.** `side` mixes book side (`back`, `lay`) with price kind (`mid`, `traded`). It doesn't break anything, but every reader must filter `sides`, or "latest quote per book" sees several prices per outcome.
- **`market` has no stated type.** It stays a free string while `outcome` is an enum. Consider a closed set.
