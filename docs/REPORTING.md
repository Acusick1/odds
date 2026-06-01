# Reporting System

This document maps the reporting surfaces of the pipeline — Discord alerts, digests, health monitoring, CLI status, MCP state queries, and agent-run traces — and the storage tables that back them. It also lists sharp edges and possible redesign directions.

"Reporting" here means: everything the system emits *about itself* (operationally) or *about its outputs* (predictions, paper trades, agent briefs) so that a human or the betting agent can know what is happening.

---

## 1. System map

```
                       ┌───────────────────────────────────────────────┐
                       │             Producers (jobs, services)        │
                       │                                               │
                       │ fetch_odds, fetch_*, score_predictions,       │
                       │ agent_run, daily_digest, check_health, …      │
                       └──────────┬────────────────────────────┬───────┘
                                  │                            │
                wrap with         │                            │ writes to
       async with job_alert_      │                            │ FetchLog, DataQualityLog,
       context(name)              │                            │ Prediction, MatchBrief,
                                  │                            │ PaperTrade, AgentWakeup
                                  ▼                            ▼
                       ┌─────────────────────┐        ┌────────────────────┐
                       │   alerts.py         │        │   Storage tables    │
                       │   (odds-core)       │        │   (Postgres)        │
                       │                     │        │                     │
                       │  AlertManager       │◄──────►│  AlertHistory       │
                       │  DiscordAlert       │ rate   │  FetchLog           │
                       │  job_alert_context  │ limit  │  DataQualityLog     │
                       │  send_job_warning   │ heart- │  Prediction         │
                       │  send_critical/…    │ beat   │  MatchBrief         │
                       │                     │        │  PaperTrade         │
                       └─────────┬───────────┘        │  AgentWakeup        │
                                 │ POST embed         └─────────┬──────────┘
                                 ▼                              │
                       ┌────────────────────┐   reads           │
                       │  Discord webhook   │                   │
                       └────────────────────┘                   │
                                                                ▼
            ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
            │  HealthMonitor   │  │  daily_digest    │  │  CLI / MCP       │
            │  (poll-based)    │  │  (poll-based)    │  │  (on-demand)     │
            │                  │  │                  │  │                  │
            │  stale data,     │  │  results +       │  │  odds status,    │
            │  failure streaks,│  │  upcoming preds  │  │  odds scheduler, │
            │  quota, quality, │  │  → embed         │  │  MCP tools       │
            │  heartbeats      │  │                  │  │  (get_portfolio, │
            │  → alerts        │  │                  │  │   get_scrape_… ) │
            └──────────────────┘  └──────────────────┘  └──────────────────┘
```

Three reporting *modes* coexist:

1. **Event-driven**: jobs emit alerts at the moment something happens (job crash, low quota, soft scrape failure). Output: Discord. Storage: `AlertHistory` (used for rate-limit + heartbeat).
2. **Poll-driven**: scheduled jobs (`check_health`, `daily_digest`) walk recent state and emit a summary alert. Output: Discord. Storage: reads from `FetchLog`, `OddsSnapshot`, `DataQualityLog`, `Prediction`, `AlertHistory`.
3. **On-demand**: CLI commands and MCP tools serve current state to a human/agent. Output: terminal table or JSON. Storage: reads from everything.

---

## 2. Alert primitives — `packages/odds-core/odds_core/alerts.py`

Everything that ends up in Discord goes through this module.

### Class hierarchy

| Class / function | Role |
|---|---|
| `AlertBase` | Abstract channel: `send(message, severity)`, `send_embed(embed)` |
| `DiscordAlert` | Webhook-backed implementation; colors by severity (info=blue, warning=yellow, error=red, critical=dark red) |
| `AlertManager` | Fan-out router. Reads `settings.alerts`, populates `self.channels` from configured channels, broadcasts to all. Disabled when `alert_enabled=False` or no channels configured. |
| `alert_manager` | Module-level singleton |

Only one channel class exists today (Discord). The interface is set up for additional channels (Slack, email) but there are no other implementations.

### Module-level helpers

- `send_info`, `send_warning`, `send_error`, `send_critical`: one-shot delivery, *not* rate-limited, *not* recorded.
- `check_rate_limit(alert_type, rate_limit_minutes=30)`: returns `True` if no `AlertHistory` row with that `alert_type` exists inside the window. **Severity is not part of the key** — a `warning` and a `critical` with the same alert_type block each other.
- `record_to_alert_history(alert_type, severity, message)`: inserts row with `sent_at = now(UTC)`.
- `job_alert_context(job_name)`: async context manager every job wraps its body in. On exception → rate-limited `critical` alert + raise. On clean exit → writes a `heartbeat:{job_name}` row to `AlertHistory` (no Discord — heartbeats are silent unless missing).
- `send_job_warning(alert_type, message)`: rate-limited warning, returns `True` if sent.

### Alert-type taxonomy

`alert_type` strings double as rate-limit keys and as keys in heartbeat expectations. The current vocabulary:

| Type | Source | Severity | Notes |
|---|---|---|---|
| `job_failure:{job_name}` | `job_alert_context` | critical | Fired on exception |
| `heartbeat:{job_name}` | `job_alert_context` | info | DB-only; no Discord |
| `quota_low`, `quota_critical` | `fetch_odds`, `HealthMonitor` | warning/critical | Tiered |
| `consecutive_failures` | `HealthMonitor` | error | ≥3 consecutive fetch failures |
| `stale_data` | `HealthMonitor` | warning | No new snapshots within `stale_data_hours` |
| `data_quality_errors` | `HealthMonitor` | warning | error+critical count > threshold |
| `missing_heartbeat:{job_name}` | `HealthMonitor` | warning | Job did not complete inside expected window |
| `health_check_failure` | `HealthMonitor` | critical | Health check itself crashed |
| Ad-hoc job warnings (e.g. empty scrape) | various jobs | warning | Free-form via `send_job_warning` |

---

## 3. Health monitoring — `packages/odds-lambda/odds_lambda/health_monitor.py`

Polled by the `check_health` job. Single class: `HealthMonitor(session, settings)`. Entry point: `check_system_health()` runs five checks and returns a `HealthStatus`.

### Checks

1. **`check_stale_data`**: hours since `max(OddsSnapshot.snapshot_time)`. Threshold: `settings.alerts.stale_data_hours`.
2. **`check_consecutive_failures`**: counts the leading run of `success=False` rows in the last 10 `FetchLog` entries. Threshold: `consecutive_failures_threshold`.
3. **`check_api_quota`**: pulls `api_quota_remaining` from latest fetch log via `OddsReader.get_database_stats()`. Tiered: `quota_critical_threshold` (10%) → critical, `quota_warning_threshold` (20%) → warning.
4. **`check_data_quality`**: counts `error` + `critical` rows in `DataQualityLog` over last 24h. Threshold: `data_quality_error_threshold`.
5. **`check_job_heartbeats`**: for each `(job_name, max_hours)` in `settings.alerts.heartbeat_expectations`, looks up `max(sent_at)` where `alert_type = heartbeat:{job_name}`. If older than `max_hours`, fires `missing_heartbeat:{job_name}` warning.

After the checks, `purge_old_heartbeats()` deletes heartbeat rows older than `heartbeat_retention_days` (default 7). This is the only retention sweep in the system.

### Internals

`HealthMonitor` has its own `_should_send_alert` / `_record_alert` / `_send_alert` triplet that duplicates the `check_rate_limit` / `record_to_alert_history` flow in `alerts.py`. The duplication exists because `HealthMonitor` carries an `AsyncSession` and an explicit `Settings`; the alerts.py helpers create their own session per call.

### Metrics object

`HealthMetrics` is a Pydantic model bundling the numbers that the health check computed: fetch success rate 24h, hours since last fetch, quota, consecutive failures, data quality errors, scheduled/live/final event counts. It is returned inside `HealthStatus` but **never emitted to Discord** — it exists only as a return value for the job invoker. No human consumer reads it.

---

## 4. Daily digest — `packages/odds-lambda/odds_lambda/jobs/daily_digest.py`

The only "human-facing summary of model outputs" report. Run on a schedule per sport.

### Flow

1. `main(ctx)` wraps in `job_alert_context("daily-digest-{sport}")`.
2. `send_digest()` does two queries:
   - **Completed** events (`status=FINAL`, `completed_at >= now - lookback_hours`) joined with latest `Prediction` per event for the configured `model_name`.
   - **Upcoming** events (`status=SCHEDULED`, `commence_time in (now, now+lookahead_hours]`) joined with latest `Prediction`, ordered by `|predicted_clv|` DESC.
3. `build_digest_embed()` assembles a green Discord embed with two fields:
   - `Post-Match Results (last {window})` — each line: ✅/❌ icon, teams, score, predicted side + CLV%, and a footer `X events | Y/X correct side`.
   - `Upcoming Predictions (next {window})` — teams, kickoff, predicted side, CLV%.
4. Sends via `alert_manager.send_embed(embed)`. Skips silently if both sections are empty.

### Domain logic

- `_value_side(predicted_clv)`: positive CLV → home undervalued, negative → away. This is the home-outcome predicted-CLV convention — the digest assumes a home-side model. There is no support for a draw side or a multi-model digest.
- `_result_hit(...)`: hit if `home_score > away_score` matches the predicted side. Draws count as a miss either way.
- `MAX_FIELD_CHARS = 1024`: Discord field-value limit. Long fields are truncated with `...`.

### Coupling

- Hard-coded `model_name` default: `settings.model.name or "epl-clv-home"`.
- Hard-coded sport display names (`_SPORT_DISPLAY_NAMES`): EPL, NBA, MLB.
- No paper-trade integration. The digest reports model predictions but not actual paper bets placed or P&L — those are MCP-side only.

---

## 5. CLI status surfaces

### `packages/odds-cli/odds_cli/commands/status.py`

| Command | Reads | Displays |
|---|---|---|
| `odds status show` | `OddsReader.get_database_stats()`, latest FetchLog | Single table: last fetch (mins ago, ✓/✗), event counts by status, API quota (color-coded), 24h success rate (color-coded) |
| `odds status show --verbose` | + `get_data_quality_logs(last 24h, limit 10)` | + event status breakdown + recent quality issues, severity-colored |
| `odds status quota` | Last 10 FetchLog rows | Totals (used/remaining, %) + trend table with delta |
| `odds status events --days N --team T` | Events table | List of events, status-colored, scores |

These are *Rich*-printed terminal tables — no machine-readable output. The same data is also surfaced via MCP for the agent.

### `packages/odds-cli/odds_cli/commands/scheduler.py`

- `odds scheduler list`: calls the active scheduler backend's job listing; renders a job table. Catches `BackendUnavailableError` for backends that don't support listing (Railway).
- `odds scheduler health`: calls `backend.health_check()` → `HealthCheckResult`; renders a green/red overall + per-check breakdown.

---

## 6. MCP reporting tools — `packages/odds-mcp/odds_mcp/server.py`

The agent gets state through tools. Reporting-relevant ones (others fetch raw odds/fixtures):

| Tool | Returns | Notes |
|---|---|---|
| `get_scrape_status(job_id?)` | Pending+running scrape jobs; per-job state, outcome, stats, exception | Pulls from APScheduler backend |
| `get_scheduled_jobs(sport?)` | All scheduled jobs, next_run_time, status | Substring sport filter on job name |
| `get_predictions(event_id, limit=5, since_hours?)` | Most-recent predictions for an event | |
| `get_portfolio(initial_bankroll=1000)` | Portfolio summary: bankroll, P&L, ROI, W/L/P record, open trades | Pure derivation from `PaperTrade` table |
| `settle_bets()` | Settles open trades against `FINAL` events; returns settlements | Writes to `PaperTrade` |
| `save_match_brief(event_id, market, decision, summary, brief_text)` | Creates `MatchBrief` row; snapshots sharp prices at write time | |
| `get_slate_briefs(league, days_ahead)` | Latest brief per upcoming event | Truncates to most recent brief per event |
| `schedule_next_wakeup(sport, delay_hours, reason)` | Upserts `AgentWakeup` row (one active per sport) | Read+consumed by `agent_run` |

The portfolio path is the only place "results from agent bets" surface. The daily digest doesn't read paper trades; the CLI has its own `odds paper` group that mirrors the MCP helpers.

---

## 7. Agent-run logs — `packages/odds-lambda/odds_lambda/jobs/agent_run.py`

Not a "report" in the alerting sense — the agent's own output is captured as a JSONL trace and consumed offline.

### Producer

`_run_claude_agent(sport)` spawns:

```
claude -p "/agent {sport}" --model claude-sonnet-4-6 \
       --output-format stream-json --verbose \
       --dangerously-skip-permissions
```

Stream lines are:

1. Always written to `logs/agent_runs/{sport}_{timestamp}_{pid}.jsonl`, with an 8 MiB per-line limit to avoid `LimitOverrunError` from `asyncio.StreamReader`.
2. Selectively tee'd to structlog via `_log_stream_message`:
   - `type=assistant` blocks with `tool_use` → `agent_tool_use` (tool name + input preview).
   - `type=result` → `agent_run_summary` (result text, num_turns, duration_ms, cost_usd).

Trace files older than the newest-per-sport cap (`AGENT_RUN_LOG_KEEP`) are pruned each run. The subprocess runs under `AGENT_DATABASE_URL` (read-only role) if set, otherwise falls back to the parent DSN with a warning.

### Consumer

`experiments/scripts/show_agent_summaries.py` reads the JSONL files, filters by `--sport`, `--since`, `--limit`, and prints the `result` summaries. The filename encodes `sport`, `timestamp`, `pid`; one file can contain multiple sessions (the script tracks `session_idx`).

### Coupling

- Filename format (`SPORT_YYYYMMDDTHHMMSSZ_PID.jsonl`) is the only schema connecting producer (`agent_run`) and consumer (`show_agent_summaries`). No DB.
- The agent's *own* memory of its runs is `MatchBrief` rows it writes via `save_match_brief()`. The JSONL trace is for humans.

---

## 8. Storage backing reports

All Postgres. Models live in `packages/odds-core/odds_core/models.py` unless noted.

| Table | Purpose | Writers | Readers |
|---|---|---|---|
| `alert_history` | Alert dedup + heartbeat ledger | `record_to_alert_history`, `HealthMonitor._record_alert` | `check_rate_limit`, `HealthMonitor.check_job_heartbeats`, `purge_old_heartbeats` |
| `fetch_logs` | Per-fetch result + quota snapshot | `OddsIngestionService` | `odds status` CLI, `HealthMonitor.check_consecutive_failures`, `HealthMonitor.check_api_quota` |
| `data_quality_logs` | Ingestion warnings/errors | `OddsIngestionService` | `odds status show --verbose`, `HealthMonitor.check_data_quality` |
| `predictions` (`prediction_models.py`) | Scored predictions | `score_predictions` job | `daily_digest`, MCP `get_predictions` |
| `match_briefs` (`match_brief_models.py`) | Agent brief log (append-only) | MCP `save_match_brief` | MCP `get_slate_briefs`, `get_match_brief` |
| `agent_wakeups` (`agent_wakeup_models.py`) | Agent → scheduler override | MCP `schedule_next_wakeup` | `agent_run._check_agent_requested_wakeup` |
| `paper_trades` (`paper_trade_models.py`) | Paper bets + settlements | MCP `paper_bet`, `settle_bets`; CLI mirrors | MCP `get_portfolio`, CLI `odds paper *` |

Index notes:
- `alert_history.ix_alert_type_sent_at` — composite, supports rate-limit lookup and heartbeat lookup in one shape.
- `fetch_logs.fetch_time` — used by status CLI and HealthMonitor.
- `paper_trades.ix_paper_trades_unsettled` — partial on `settled_at IS NULL`.

---

## 9. Sharp edges and quirks

These are the friction points to know about before touching this code.

1. **Two rate-limit implementations.** `alerts.py::check_rate_limit` (opens its own session) and `HealthMonitor._should_send_alert` (uses injected session) duplicate logic and config reading. Drifting them is easy.
2. **Severity-blind rate limiting.** `alert_type` is the entire rate-limit key. A `warning` followed by a `critical` of the same alert_type is suppressed for `rate_limit_minutes`. For the quota chain (`quota_low` then `quota_critical`) this works only because the alert types differ — anyone adding tiered alerts must remember to pick distinct alert_types.
3. **Heartbeat history conflated with alert history.** Heartbeats live in the same table as Discord-sent alerts, distinguished only by `alert_type` prefix (`heartbeat:`). `purge_old_heartbeats` filters on this string prefix. The schema invariant "heartbeat rows are silent, alert rows were sent" is not enforced anywhere.
4. **Heartbeat expectations are config-only.** `settings.alerts.heartbeat_expectations` is a dict of `{job_name: max_hours}`. If a job is renamed, its heartbeat key drifts silently and the missing-heartbeat alert never fires. The same string also has to match what `job_alert_context` is called with — there is no central registry.
5. **`HealthMetrics` is collected but never displayed.** Health-check runs gather a snapshot of the system, return it to the job, and it disappears. Discord only receives the issue strings, not the metrics object.
6. **Daily digest is single-model and single-side.** Hard-coded home-side CLV convention; one `model_name` per run. No support for multi-model contrast, draw side, or aggregating multiple sports in one embed.
7. **Daily digest ignores paper trades.** "What did the model say" is reported; "what did we actually bet, and how did it do" lives in a separate MCP-only surface. No Discord-side P&L summary exists.
8. **Discord-only.** `AlertBase` is set up for plug-in channels but only `DiscordAlert` exists. Single point of failure: if the webhook URL is rotated or the Discord webhook breaks, all alerting silently no-ops (errors are logged but not surfaced anywhere else).
9. **Webhook failure is not itself an alert.** `DiscordAlert._post` logs `discord_alert_error` on exception. There is no secondary channel and no DB-side "alert delivery failed" marker — `record_to_alert_history` is called regardless of whether the Discord POST succeeded, so a failed delivery still counts toward rate-limiting.
10. **Rate-limit query is read-then-write without a transaction.** Two concurrent jobs can both observe "no recent alert", both send, both record. In practice jobs don't overlap on alert-type but it isn't enforced.
11. **`alert_manager` is module-level**, instantiated at import time from global settings. Re-reading settings (e.g. flipping `alert_enabled` at runtime) requires re-importing or constructing a new manager. Tests have to monkey-patch it.
12. **Agent-run reporting has no DB component.** The only durable record is the JSONL file. If the disk is wiped (Lambda ephemeral storage, container restart on Railway), traces are lost — `MatchBrief` is the only persistent agent record.
13. **Sport-aware naming is ad-hoc.** Daily digest uses `make_compound_job_name("daily-digest", sport)` → `daily-digest-epl`. Other places use the sport key directly (`soccer_epl`). Heartbeat keys, CloudWatch metrics, Discord embed titles, and CLI display names each have their own translation table.

---

## 10. Alternative directions (bigger shape changes)

These are not "fixes" — they're different ways the reporting layer could have been built. Useful for sketching where the system could evolve.

### Bus-based instead of point-to-point

Today every job pushes alerts directly through `AlertManager`. An alternative: jobs publish typed events (`JobFailed`, `QuotaLow`, `HeartbeatRecorded`, `BriefSaved`) to an internal bus; alert formatting, persistence, and channel fan-out are subscribers. Pros: decouples job logic from channel choice; trivially adds a "log all events" sink for audit. Cons: more indirection for a system with one channel and a handful of alert types.

### Pull-based reporting (Prometheus-shaped)

Instead of jobs pushing alerts to Discord, jobs would write metrics to a metrics store (CloudWatch, Prometheus). Discord becomes a destination for *alarm rules* over those metrics. Pros: standard tooling, dashboards for free, alerting becomes declarative ("alert when fetch_success_rate_24h < 0.95"). Cons: requires a metrics backend; loses the rich per-event context that the current digests carry.

### Single "control plane" surface

Replace Discord webhook + CLI + MCP with one HTTP service that:
- Serves status, alerts, predictions, briefs, paper trades as JSON.
- Hosts a tiny web UI for humans.
- Discord becomes a thin push-notification channel that links back to the UI.

Today the same information is rendered three ways (terminal table, Discord embed, MCP dict). A unified API would centralize the "what is the current state" query.

### Briefs as event-sourced agent log

Right now `MatchBrief` is append-only but each brief is a snapshot of the agent's whole reasoning. An event-sourced log of `(decision_event, prior_observations, prior_pre_market_read, …)` would let downstream tooling diff agent reasoning across briefs, score calibration over time, and replay agent state without re-running the agent. The current schema treats every brief as a fresh dump; events would expose the *delta*.

### CloudWatch-native alerting

The codebase already uses CloudWatch. The hand-rolled `HealthMonitor` checks (stale data, quota, consecutive failures, data quality) are exactly the kind of thing CloudWatch metric alarms handle natively. The pipeline could:
- Emit one metric per check via PutMetricData.
- Define CloudWatch alarms (with SNS → Discord webhook lambda).
- Delete `HealthMonitor`.

The cost: AWS-only, harder to test locally. The benefit: no custom rate-limit code, no custom heartbeat table, alarms get history and graphs for free.

### Separation of "tooling alerts" and "trading alerts"

Two separate Discord destinations:
- **#odds-ops** (job failures, quota, stale data, heartbeats) — operational, mostly silent.
- **#odds-trades** (paper trades placed, daily digest, agent run summaries) — substantive, daily traffic.

Today they're collapsed into one webhook and one channel. The signal-to-noise tradeoff differs sharply between the two; they probably want different rate limits, retention, and audiences.

---

## 11. Reference: where each piece lives

- Alert core: `packages/odds-core/odds_core/alerts.py`
- Alert / fetch / data-quality tables: `packages/odds-core/odds_core/models.py` (`AlertHistory`, `FetchLog`, `DataQualityLog`)
- Health monitor: `packages/odds-lambda/odds_lambda/health_monitor.py`
- Health-check job: `packages/odds-lambda/odds_lambda/jobs/check_health.py`
- Daily digest job: `packages/odds-lambda/odds_lambda/jobs/daily_digest.py`
- Agent-run job: `packages/odds-lambda/odds_lambda/jobs/agent_run.py`
- Agent-run viewer: `experiments/scripts/show_agent_summaries.py`
- Status CLI: `packages/odds-cli/odds_cli/commands/status.py`
- Scheduler CLI: `packages/odds-cli/odds_cli/commands/scheduler.py`
- Paper CLI: `packages/odds-cli/odds_cli/commands/paper.py`
- MCP server: `packages/odds-mcp/odds_mcp/server.py`
- Paper trading core: `packages/odds-core/odds_core/paper_trading.py` (helpers `place_trade`, `settle_trades`, `get_portfolio_summary`)
- Match brief models: `packages/odds-core/odds_core/match_brief_models.py`
- Agent wakeup models: `packages/odds-core/odds_core/agent_wakeup_models.py`
- Paper trade models: `packages/odds-core/odds_core/paper_trade_models.py`
- Alert config: `packages/odds-core/odds_core/config.py::AlertConfig`
