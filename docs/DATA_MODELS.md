# Training Data Reference

Reference for the CLV model's training data: sources, target, feature groups,
cross-validation settings and config fields. Why the model is configured this way, and the
experimental evidence, is in [MODELING.md](MODELING.md).

## Data Sources

### Sportsbook Odds

Three sources with complementary bookmaker coverage. Each fetch stores an `OddsSnapshot`
tagged with the `FetchTier` it was collected in (see
[Fetch Tier System](ARCHITECTURE.md#fetch-tier-system)).

| Source | Bookmakers | Events | Key value |
|--------|-----------|--------|-----------|
| Odds API | Pinnacle, FanDuel, DraftKings, BetMGM, Bovada (US) | ~1K NBA (dense snapshots) | Dense snapshot sequences, US bookmakers |
| OddsPortal | bet365, Betway, Betfred, bwin (UK) | ~5K NBA + ~1.8K EPL (opening + closing only) | UK retail bookmakers |
| football-data.co.uk | Pinnacle, bet365, bwin, William Hill, Betvictor, Interwetten, Ladbrokes | ~4K EPL (11 seasons, 2015–2026) | Pinnacle + Betfair Exchange closing odds (sharp reference) |

Event IDs identify the source by prefix (see [Core Tables](DATABASE.md#core-tables));
`data_source` filters on it. 1,810 EPL events are matched between OddsPortal and FDUK.

The `betfair_exchange` bookmaker key has two sources: football-data.co.uk Betfair Exchange
closing odds (columns `BFE*`/`BFEC*`, from 2024-25 onwards) and live snapshots from the
`fetch-betfair-exchange` job (see [ARCHITECTURE.md](ARCHITECTURE.md#betfair-exchange)).

Each snapshot contains one `Odds` row per bookmaker per market per outcome. American odds are
converted to implied probabilities internally.

### Polymarket Prices

The Polymarket pipeline is deprioritized; see [POLYMARKET.md](POLYMARKET.md).

## Training Target

The regression target is the devigged bookmaker CLV delta at each snapshot:

```
target = devigged_fair_close - devigged_fair_at_snapshot
```

It measures how far, and which way, the line moves from the snapshot to close. A model
that predicts it can flag games where the current price is likely to shift, which drives
the bet-or-pass decision.

The target bookmaker is configurable (`target_bookmaker`). bet365 is the target for EPL;
see [MODELING.md](MODELING.md#current-conclusions) for why not Pinnacle.

Why a delta rather than the absolute closing price (inferred):

- The decision needs how much the line will move, not where it will end up.
- A delta is stationary and mean-zero, so it is easier to learn than absolute levels.
- It maps directly to CLV: a positive delta means the current price is too low and the
  line will move up.

## Feature Groups

`feature_groups` in the config selects which groups to compose. Available groups:

| Group | Features | Sports | Contents |
|-------|----------|--------|----------|
| `tabular` | 6 | all | `consensus_prob`, `sharp_prob`, `retail_sharp_diff`, `num_bookmakers`, `is_weekend`, `day_of_week` (`TabularFeatures`), outcome-relative at decision time |
| `standings` | 11 | EPL | League position, points gap, goal difference, form over the last `form_window` matches (default 5), goals scored and conceded rates |
| `match_stats` | 14 | EPL | Rolling averages over the last `match_stats_window` matches (default 5) of shots, shots on target, corners, fouls, yellow and red cards, half-time goals; only matches completed before `commence_time` |
| `epl_schedule` | 8 | EPL | Rest days, rest advantage, midweek flag, matches in the trailing 14 days, European matches in the trailing 7 days, from ESPN all-competition fixtures |
| `epl_lineup` | 6 | EPL | Starting XI changes against the previous match and cumulative starts lost, per team and as a difference |
| `rest` | 5 | NBA | Days rest, rest advantage, back-to-back flags, from NBA game logs; produces no features for EPL |
| `injuries` | 6 | NBA | Impact-weighted injury burden (see [INJURIES.md](INJURIES.md#features-6)) |
| `polymarket` | 14 | NBA | Polymarket prices and order book microstructure (deprioritized) |

The cross-source divergence features (`pm_sb_prob_divergence`, `pm_sharp_divergence`) are
computed alongside the Polymarket group. The LSTM adapter uses per-snapshot sequence
features instead (15 per timestep: odds, implied probability, hours to game, time of day,
changes from previous and opening, odds spread, and sharp odds and divergence).

Injury, Polymarket and sequence features showed no predictive value; see
[MODELING.md](MODELING.md#current-conclusions).

When optional data is missing, features are NaN-filled and the row is kept.

## Cross-Validation Protocol

Walk-forward CV with a sliding one-season window and ~5-matchday validation steps:

```yaml
cv_method: walk_forward
window_type: sliding
min_train_events: 380    # ~1 EPL season
max_train_events: 380
val_step_events: 50      # ~5 matchdays
test_split: 0.0
```

This yields ~26 folds for ~1.8K EPL events. There is no held-out test set; walk-forward CV
simulates the production retrain cycle. The protocol was chosen by grid search; see
[CV protocol grid](MODELING.md#cv-protocol-grid).

## Configuration

Training uses YAML config files in `experiments/configs/`, parsed by
`packages/odds-analytics/odds_analytics/training/config.py` (authoritative for fields and
defaults). Fields sit under `training.data`, `training.model` and `training.features`; for
a complete example see `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`.

### Data selection (`training.data`)

- `data_source`: `"oddsportal"`, `"oddsapi"`, `"football_data_uk"`, `"all"`, or `null` (no
  filter). Filters by event ID prefix.
- `sport_key`: sport filter (e.g. `"soccer_epl"`, `"basketball_nba"`)
- `start_date` / `end_date`: date range filter (always required)
- `min_snapshots`: minimum snapshots per event (e.g. `5` for dense sequence data)

### Cross-validation (`training.data`)

- `cv_method`: `"walk_forward"`, `"timeseries"`, or `"kfold"`
- `window_type`: `"sliding"` or `"expanding"` (walk-forward only)
- `min_train_events` / `max_train_events`: training window size bounds
- `val_step_events`: number of events per validation fold
- `validation_split`: set to `0.0` when using CV
- `test_split`: set to `0.0` for walk-forward CV

### Target and features (`training.features`)

- `target_type`: `"devigged_bookmaker"`
- `target_bookmaker`: e.g. `"bet365"`, `"pinnacle"`
- `sharp_bookmakers`: sharp references in priority order; the first available bookmaker
  wins (e.g. `[pinnacle, betfair_exchange]`)
- `closing_source_priority`: ordered list of preferred sources for closing snapshot
  selection, matched on `api_request_id` (e.g. `[football_data_uk]`). When several sources
  provide a closing snapshot for an event, the first in the list wins.
- `feature_groups`: groups to include (e.g. `[tabular, match_stats]`)
- `outcome`: `"home"` or `"away"` for h2h, `"over"` or `"under"` for totals

### Choosing data for an experiment

`target_bookmaker` must belong to the selected source's bookmaker set.

- **EPL, bet365 target, hybrid sharp** (the production setup): `data_source: null`
  (OddsPortal + FDUK combined), `sharp_bookmakers: [pinnacle, betfair_exchange]`,
  `closing_source_priority: [football_data_uk]`.
- **NBA, bet365 target**: `data_source: oddsportal`; bet365 is only available from
  OddsPortal.
- **NBA, Pinnacle target**: `data_source: oddsapi`; Pinnacle is only available from the
  Odds API for NBA.
- **Sequence (LSTM) experiments**: also set `min_snapshots: 5` or higher.
