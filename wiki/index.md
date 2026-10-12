# Wiki index

Start here. Each line is a page and its summary; open the pages whose summaries match the question and follow their links. Page rules are in [SCHEMA.md](SCHEMA.md).

## Concepts

- [Agent data sources](agent-data-sources.md) — Sources collected on a schedule and exposed through MCP, versus sources the agent researches on demand at each wake-up.
- [Backtest metrics and results](backtest-metrics.md) — ROI, win rate, Sharpe, Sortino, drawdown, profit factor and Calmar as the code computes them, and the BacktestResult model with its JSON and CSV forms.
- [CLV model findings](clv-model-findings.md) — Hub - what the CLV model can and cannot do, the deployed model, current conclusions and data used, with links to every experiment and decision.
- [Closing line value (CLV) and the training target](clv.md) — What CLV means here, the devigged CLV-delta regression target, target_bookmaker, and where the target maths lives.
- [Code conventions and quality gates](code-conventions.md) — Type hints, async-only DB access, model-type choice, the uv workspace, which gates are mechanical (ruff, Vale, lychee) and which are convention, and structlog logging.
- [Conviction tiers and edge types](conviction-tiers.md) — Stakes of 0-3% of bankroll by conviction and the four edge bases every bet must name, plus what code does and does not enforce.
- [Data sources and bookmaker coverage](data-sources.md) — Every external odds feed the pipeline ingests, what each supplies, how it is collected, and which bookmaker and market keys come from where.
- [Database connections and environments](database-connections.md) — How code obtains DB sessions (async_session_maker), NullPool in Lambda, pool sizing and the import-time engine, plus the test/local/dev/prod databases and their env vars.
- [Decision time](decision-time.md) — The as-of instant a bet or prediction is made (hours before the game), how backtests and snapshot samplers choose it, and the default windows.
- [Devigging](devigging.md) — Proportional removal of bookmaker margin to get fair probabilities, where it is implemented, and which consumers deliberately stay vig-inclusive.
- [Documentation map and authority](documentation-map.md) — Which repo doc is authoritative for what, the doc-truth lints (Vale, lychee), the glossary and ADR convention, and how the wiki relates to the docs.
- [Event lifecycle](event-lifecycle.md) — The five EventStatus values, which job or command sets each (only LIVE and FINAL are ever written), and what each status gates.
- [Event matching and team names](event-matching.md) — How each source attaches rows to an events row (five matcher implementations with different windows and ambiguity policies) and the two team-name normalizers.
- [Feature groups](feature-groups.md) — The catalogue of composable feature groups (tabular, standings, match_stats, epl_schedule, epl_lineup, rest, injuries, polymarket, plus LSTM sequence features), their counts, the NaN-fill policy, and how to add one.
- [Fetch tier](fetch-tier.md) — The opening/early/sharp/pregame/closing/in-play tiers by hours to kickoff, their intervals, where tiers are stored and used, and the four boundary encodings that disagree.
- [Job deployment matrix](job-deployment-matrix.md) — Where each job can run (scheduler Lambda, scraper Lambda, local) and on what schedule, which gates decide it, and why the repo does not record the deployed state.
- [Look-ahead bias](look-ahead-bias.md) — The no-future-data rule across backtesting and features, how each layer enforces it, and the symmetric get_odds_at_time window that breaks it.
- [Odds API quota and cost](odds-api-quota.md) — Unit cost per Odds API endpoint, free and paid tiers, how key rotation multiplies quota, and where usage is logged, alerted and checked.
- [Project overview](project-overview.md) — What the system is, which sports are active, where the edge is meant to come from, what runs where, the top-level repo layout and the tech stack.
- [Project pivots and NBA-era fossils](project-pivots.md) — The three pivots (NBA to EPL/MLB, model edge to information edge, structural consolidations) and the NBA-era assumptions still live in code.
- [Retail-sharp divergence](retail-sharp-divergence.md) — Retail implied probability minus sharp implied probability, the model's strongest feature and the basis of find_retail_edges, with every code copy and how they differ.
- [Sharp benchmark](sharp-benchmark.md) — The sharp reference price (Pinnacle, with Betfair Exchange as fallback), sharp_bookmakers priority semantics, and how features, the agent and MCP use it.
- [Sport and market keys](sport-and-market-keys.md) — SportKey values, short sport suffixes in job names, the market-key dialects (h2h vs 1x2, OddsHarvester names) and what touching a new sport, bookmaker or market involves.
- [Timezone handling](timezone-handling.md) — UTC everywhere with datetime.now(UTC), what ruff enforces, the odds_core.time helpers, and the naive-datetime sites that remain.

## Processes

- [Matchday wake-up workflow](matchday-wakeup.md) — The steps of each agent wake-up (orient, settle, triage, pre-market read, research, brief, decide) and the EPL matchday timeline.
- [Odds ingestion flow](odds-ingestion-flow.md) — How one Odds API fetch becomes stored data - client, validator, OddsWriter (raw plus normalized), FetchLog - via the OddsIngestionService shared by the job and `odds fetch current`.
- [Reporting surfaces](reporting-surfaces.md) — Event-driven, polled and on-demand reporting: what the system reports about itself and its outputs, through Discord, CLI status, MCP and traces.
- [Self-scheduling](self-scheduling.md) — Each job invocation computes and books its own next run (an EventBridge rule or an APScheduler entry), pre-schedules so the chain survives a crash, and follows a per-job failure policy.
- [Training pipeline](training-pipeline.md) — From YAML config to a trained model - odds train run and tune, data preparation, CV, tuning, MLflow, verification artifacts and saved models.

## Components

- [AWS EventBridge backend](aws-eventbridge-backend.md) — How the AWS backend reschedules each job's EventBridge rule: rule naming, placeholder rules, ignore_changes, and the backend's env-var contract.
- [AWS infrastructure (Lambdas and Terraform)](aws-infrastructure.md) — The two Lambda functions and their images, Terraform layout, variables and bootstrap, plus SSM, CloudWatch, constraints, costs and usage analysis.
- [Agent run traces](agent-run-traces.md) — JSONL stream traces under logs/agent_runs, the structlog tee, pruning, and the show_agent_summaries viewer.
- [Alerts module](alerts-module.md) — alerts.py: AlertManager and DiscordAlert, the send_* helpers, job_alert_context, send_job_warning, configuration, and what happens when delivery fails.
- [Backtest engine](backtest-engine.md) — BacktestEngine and BacktestConfig - the event loop, odds at decision time through OddsReader, bet sizing (Kelly, flat, percentage), constraints and settlement.
- [Backtest strategies](backtest-strategies.md) — The flat, basic EV, arbitrage and model-driven strategies, their defaults, and the STRATEGIES registry.
- [Betfair Exchange client](betfair-exchange-client.md) — BetfairExchangeClient, certificate loading and the adapter that turns best back Match Odds into betfair_exchange snapshots.
- [Betting agent](betting-agent.md) — The per-sport LLM agent run as a claude subprocess over odds-mcp, web search and Playwright: runtime, prompt composition, design principles and status.
- [CI/CD workflows](ci-cd-workflows.md) — GitHub Actions: tests, lint, ephemeral dev deploy, manual prod deploy, daily validation, and live-API and schedule health checks.
- [Completeness and coverage validation](completeness-validation.md) — odds validate, odds quality coverage, TierCoverageValidator, the scripts/check_* tools and the daily-validation workflow, which together check that collected data is complete.
- [Configuration (Settings)](configuration.md) — The Pydantic Settings classes, env-var naming (prefixed vs bare), what is frozen at import, which variables bypass Settings, and the rules for changing config.
- [Copy from production](copy-from-production.md) — odds copy from-prod: what it copies, its options, and the fallback that treats DATABASE_URL as production.
- [EPL context data (ESPN, FPL, FBref)](epl-context-data.md) — ESPN fixtures (with the fetch-espn-fixtures job) and lineups, FPL availability and FBref stats - fetchers, ingestion scripts, tables, readers and writers.
- [EPL context feature groups](features-epl-context.md) — The standings, match_stats, epl_schedule and epl_lineup extractors - inputs, windows and point-in-time rules - plus the undocumented lineup signal check.
- [Feature pipeline](feature-pipeline.md) — The five layers of feature_groups.py - EventDataBundle collection, TierSampler/TimeRangeSampler sampling, targets, XGBoost/LSTM adapters, the orchestrator - and the sampler guards.
- [Health monitor and check-health job](health-monitor.md) — The five health checks (stale data, failure streaks, quota, quality, heartbeats), their thresholds, and the duplicated alert plumbing.
- [Historical odds ingestion scripts](historical-odds-ingestion.md) — One-off scripts that scrape OddsPortal history and load it plus football-data.co.uk CSVs as Event and OddsSnapshot rows in the live raw_data format.
- [Hyperparameter tuner (Optuna)](hyperparameter-tuner.md) — OptunaTuner and its objective, search spaces (including the principled space), study storage, feature selection, and the tuner's error-swallowing behaviour.
- [Inline CLV scoring and predictions table](job-score-predictions.md) — score_events() at the end of fetch-oddsportal: S3 model load, feature extraction and idempotent writes to the `predictions` table.
- [Job registry and JobContext](job-registry.md) — scheduling/jobs.py: the job map, lazy import, per-sport jobs, compound job names, bootstrap overrides and the JobContext passed to every job.
- [Lambda handler](lambda-handler.md) — The single Lambda entry point: EventBridge payload format, job and sport routing, and its extra failure alert that skips rate limiting.
- [Line-movement model strategies (XGBoost, LSTM)](line-movement-models.md) — XGBoostLineMovementStrategy and LSTMLineMovementStrategy - training, prediction and evaluate_opportunity, Kelly-confidence parameters, LSTM status, and ml_strategy_example.
- [Local scheduler (APScheduler)](local-scheduler.md) — The `odds scheduler` start/run-once/smoke/list-jobs commands on APScheduler 4 with a Postgres store: bootstrap_jobs, cron floors, dry-run and `--db` sandboxing.
- [MCP odds, fixture and prediction tools](mcp-market-data-tools.md) — get_upcoming_fixtures, get_current_odds, get_odds_history and get_predictions: inputs, outputs, window semantics and the market vocabulary they need.
- [MCP scrape and scheduling tools](mcp-scrape-and-schedule-tools.md) — refresh_scrape, get_scrape_status, get_scheduled_jobs and schedule_next_wakeup: what each does to the scheduler store and what it does not guarantee.
- [MCP sport context tools](mcp-sport-context-tools.md) — EPL get_team_context (standings, form and fixtures derived from ESPN fixtures) and MLB get_probable_pitchers (live read-through with a cached fallback).
- [MLB probable pitchers](mlb-probable-pitchers.md) — The MLB Stats API fetcher, the fetch-mlb-probables job, the mlb_probable_pitchers snapshot table and its domain-record reader.
- [Match briefs](match-briefs.md) — The append-only match_briefs table and the save, get and slate MCP tools: schema, sharp-price snapshot, decision enum and triage use.
- [NBA game logs and player stats](nba-stats-data.md) — stats.nba.com game logs (via Playwright) and PBPStats player season stats - tables, writers and CLI, and what uses them (rest features, score backfill, injury weighting).
- [NBA injury reports and features](nba-injury-reports.md) — Official NBA injury PDFs into nba_injury_reports - event matching, event-driven backfill and the six injury features, which carry no CLV value.
- [Odds arithmetic (odds_math)](odds-math.md) — Shared American/decimal conversion, implied probability, proportional devig and profit helpers in odds-core, plus the diverging converter copies elsewhere.
- [Odds validation and data_quality_logs](data-quality-validation.md) — OddsValidator checks, the log-only behaviour, the unused REJECT_INVALID_ODDS flag, and the data_quality_logs table they write.
- [OddsPortal scraper integration](oddsportal-scraper.md) — OddsHarvester/Playwright scraping, bookmaker mapping, the raw_data adapter, retry logic, browser lock and LEAGUE_SPECS, shared by the OddsPortal jobs and `odds scrape`.
- [OddsReader](odds-reader.md) — Time-aware odds queries (odds at time, line movement, best odds, sharp prices, database stats), and which ones read normalized rows and which re-parse raw JSON.
- [OddsWriter](odds-writer.md) — Writes OddsSnapshot and Odds rows, assigns the fetch tier per event, finds or creates events, and bulk-inserts for backfill.
- [Paper trading](paper-trading.md) — The paper_trades table, the place, settle and portfolio service, the MCP tools paper_bet, get_portfolio and settle_bets, and the odds paper CLI.
- [Polymarket pipeline](polymarket-pipeline.md) — Deprioritized, NBA-targeted Polymarket collection - Gamma/CLOB client, ingestion service, fetch-polymarket and backfill-polymarket jobs, ticker-based event linking, config and market status per sport.
- [Polymarket tables](table-polymarket.md) — The five polymarket_* tables - relationships, indexes, order-book derived metrics and storage estimates.
- [S3 model store and loader](s3-model-store.md) — The versioned model bucket layout, `odds model publish`, and ETag-cached model loading for scoring.
- [Scheduler backend interface](scheduler-backends.md) — The SchedulerBackend ABC, backend selection with SCHEDULER_BACKEND, the Railway no-op shim, BackendUnavailableError and backend health checks.
- [Scheduling decision engine](scheduling-decision-engine.md) — decision.py: decide_forward, decide_forward_resilient and decide_backward, each job's CadenceConfig, the run/skip gate and overnight skip windows.
- [Storage layer (readers and writers)](storage-layer.md) — Map of odds_lambda/storage/: one reader/writer pair per source, the commit/flush contract, upsert verb conventions and known drift between copies.
- [The Odds API client and key rotation](odds-api-client.md) — TheOddsAPIClient endpoints, tenacity retry, key rotation on 401/429 with the active index in SSM, and rules for changing it.
- [Training config (MLTrainingConfig)](training-config.md) — The YAML schema for training.data, training.model and training.features - field meanings, choosing data for an experiment, and invalid values that are accepted silently.
- [Walk-forward cross-validation](walk-forward-cv.md) — cross_validation.py - sliding and expanding event windows, validation steps, fold construction and metrics, plus the test_split and silent-downgrade traps.
- [agent-run job and wake-up scheduling](job-agent-run.md) — The local-only job that runs one agent wake-up: fixture-proximity cadence, agent_wakeups overrides, the AGENT_DATABASE_URL role and the odds agent run command.
- [alert_history table, alert types and heartbeats](table-alert-history.md) — The alert-type taxonomy, severity-blind rate limiting, and the heartbeat rows that share the table, with their retention.
- [daily-digest job](job-daily-digest.md) — A local-only EPL Discord embed of recent results and upcoming predictions, built on a single-model, home-side convention.
- [events table](table-events.md) — Event rows (sport_key, status, scores) and the id-prefix convention that encodes each row's source.
- [fetch-betfair-exchange job](job-fetch-betfair-exchange.md) — Polls Betfair Exchange best back/lay prices per sport on a kickoff-proximity cadence: self-scheduling on AWS, a cron floor locally.
- [fetch-odds job](job-fetch-odds.md) — The per-sport Odds API job: event sync from the free /events endpoint, game-aware gate, ingest, tier-based reschedule and quota alerts.
- [fetch-oddsportal job](job-fetch-oddsportal.md) — The per-league OddsPortal upcoming-odds scrape (scraper Lambda or local): kickoff-proximity cadence, tiering and inline CLV scoring.
- [fetch-oddsportal-results job](job-fetch-oddsportal-results.md) — A daily ~08:00 UTC scrape of results and closing odds that marks events FINAL and reschedules itself in `finally`.
- [fetch-scores job](job-fetch-scores.md) — The Odds API scores job that marks events FINAL, its backward-looking cadence, and the `odds fetch scores` CLI copy of it.
- [fetch_logs table](table-fetch-logs.md) — Success, quota and timing for each fetch, written by ingestion and read by the status CLI and health checks.
- [find_retail_edges tool](mcp-find-retail-edges.md) — The agent's retail-vs-sharp screener: inputs, ranking, dispersion statistics, vig-inclusive divergence and its dependence on the hold maths.
- [odds CLI](cli-overview.md) — The odds Typer app, its 18 command groups and the wiki page that owns each, plus shared conventions (--sport default, --db override, exit styles).
- [odds table](table-odds.md) — Normalized odds, one row per bookmaker/market/outcome, with American prices and the query indexes.
- [odds-analytics package](package-odds-analytics.md) — Module map for odds-analytics (backtesting, strategies, features, training, backfill planning) and the modules whose names mislead.
- [odds-core package](package-odds-core.md) — The shared base package (models, config, database, alerts, odds maths, helpers) and the dependency rule that keeps it at the bottom of the graph.
- [odds-lambda package](package-odds-lambda.md) — Module map for odds-lambda (fetchers, ingestion, storage, scheduling, jobs, services) and the cross-package imports its pyproject does not declare.
- [odds-mcp server](mcp-server.md) — The FastMCP server package: launch, the tool catalogue with the page that owns each tool, sport sub-server mounting, the tool-name uniqueness check and the dependency on odds-analytics.
- [odds_snapshots table and hybrid storage](table-odds-snapshots.md) — One raw JSON snapshot per fetch, tagged with fetch_tier. Covers the hybrid raw-plus-normalized design, the raw_data format and the source vocabulary.
- [update-status job](job-update-status.md) — Marks SCHEDULED events that have started as LIVE, so pre-game fetching stops.

## Decisions

- [Pinnacle with Betfair Exchange fallback as sharp reference](decision-hybrid-sharp-reference.md) — Why sharp_bookmakers is [pinnacle, betfair_exchange] with first-available semantics.
- [Pivot to information edge](decision-pivot-to-information-edge.md) — Why the agent's breadth of information synthesis, not the CLV model, is treated as the primary edge.
- [Rejected agent data sources and lineup sourcing](decision-rejected-data-sources.md) — Why each evaluated source was rejected, and why lineups come from web search rather than an API.
- [Walk-forward CV with sliding-380:50 and no test split](decision-cv-protocol.md) — Why walk-forward CV is the primary metric, test_split is 0.0, and the window is one season with 50-event steps.
- [bet365 as the CLV target](decision-bet365-target.md) — Why the model predicts bet365 line movement rather than Pinnacle's.

## Experiments

- [Betting simulation and line shopping (Exp 7, 7b)](exp-betting-simulation.md) — A flat-stake simulation on bet365 loses at every useful threshold, and line shopping cuts the overround but not by enough.
- [Bookmaker line release timing](exp-bookmaker-line-release.md) — When NBA bookmakers first post lines (soft books days out, Pinnacle 12-24 h before), and what that means for tiers and opening lines.
- [CV protocol grid](exp-cv-protocol-grid.md) — A 15-cell grid of window type x validation step on EPL - sliding-380 gives the lowest MSE, and a validation step of 50 keeps feature sets distinguishable.
- [EPL feature group ablation](exp-epl-feature-group-ablation.md) — Per-group EPL ablation plus the FPL availability runs - match_stats helps, standings is neutral, and schedule and FPL hurt.
- [EPL sharp reference (Pinnacle, then hybrid)](exp-epl-sharp-reference.md) — EPL runs on combined OddsPortal and FDUK data - signal needs a sharp reference, the
- [Early NBA signal search (XGBoost v1, Exp 1, Exp 2)](exp-early-nba-signal-search.md) — February 2026 NBA runs on about 230 events - no signal, sharp-retail divergence the strongest correlate, an IN_PLAY leak found, and a Polymarket vs sportsbook cross-check.
- [Hours-to-game effect (Exp 4)](exp-hours-to-game.md) — CLV delta grows with hours to the game, and sharp-retail divergence peaks 3-6 h before NBA games.
- [LSTM sequence models](exp-lstm-sequence-models.md) — Three LSTM variants on the Pinnacle target, all worse than predicting the mean, so LSTM was ruled out.
- [NBA injury features](exp-nba-injury-features.md) — Injury and rest features looked positive until both arms were tuned the same way; they add nothing at the sharp and closing tiers.
- [NBA target bookmaker and data volume](exp-nba-target-and-data-volume.md) — bet365 at about 5K events plateaus at R² near 0.036, Pinnacle CLV is unpredictable, and the learning curve is flat after about 1.5K events.
- [Principled search space re-test](exp-principled-search-space.md) — Sliding-380:50 tuning collapsed to constant predictors; a revised Optuna space fixed this and produced the deployed model.

## How-tos

- [Add a betting strategy](howto-add-betting-strategy.md) — Subclass BettingStrategy, implement evaluate_opportunity with BacktestEvent, register it in STRATEGIES, and test it.
- [Add a scheduler job](howto-add-scheduler-job.md) — Entry point, registry entry, self-scheduling, job_alert_context, per-sport or cron registration, local testing and the deployment-matrix update.
- [Backfill historical odds and scores](howto-historical-backfill.md) — Discover, plan, execute, scores, status (or gaps, then execute) for loading historical Odds API snapshots, with the gap planner, game selector and executor and their NBA-only defaults.
- [Create and apply a database migration](howto-database-migration.md) — Change a model, autogenerate and review an Alembic migration, then apply it local, then CI dev, then prod, following the safety rules.
- [Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md) — Read-only diagnostic workflow: rules, CloudWatch logs, database vs fixture list, tier coverage, common failures and manual Lambda invoke.
- [Deploy to AWS](howto-deploy-aws.md) — Build and push both images, apply Terraform per environment, bootstrap and publish a model on first deploy, and how CI deploys.
- [Run a backtest](howto-run-backtest.md) — Backtests from the CLI or from code, data prerequisites, and viewing, comparing and exporting results.
- [Run and record a modelling experiment](howto-run-experiment.md) — Script conventions, the required FINDINGS.md, verification artifacts and their checks, and updating the experiment record in the wiki's exp pages.
- [Run the scheduler on an always-on host](howto-deploy-local-host.md) — Install the systemd unit, redeploy with deploy.sh (pull, sync, migrate, smoke, restart), and stop the service first for destructive migrations.
- [Run the test suite](howto-run-tests.md) — Unit and integration tests on in-process py-pglite, migration tests left to the deploy workflows, and the test layout and conventions.
- [Set up a local development environment](howto-local-setup.md) — uv sync, a working .env, Docker Postgres on port 5433, migrations, and the first commands to run.
- [Train, tune and publish the CLV model](howto-train-and-publish-model.md) — Write a config, run odds train run or tune --train-best, check the artifacts, run odds model publish, and confirm scoring picks the model up.
- [Use the issue tracker](howto-use-issue-tracker.md) — gh conventions for issues, the Personal board and draft-PR linking, sub-issues, triage labels and wayfinding maps.
