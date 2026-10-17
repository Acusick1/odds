---
status: accepted
---

# Run production on the mini-PC and leave AWS dormant

## Context

By October 2026 the pipeline ran on two hosts that wrote to two databases. The audit in #483 recorded the state.

- **AWS.**
  - The scheduler and scraper Lambdas ran image `prod-1e33618` (#372, 2026-04-28), with no production deploy since.
  - Their EventBridge rules wrote to the Neon database.
  - The scraper had stored no OddsPortal snapshot since 2026-08-10 (#484).
  - Inline scoring failed on every run (#454).
- **Mini-PC.**
  - A local scheduler wrote to its own Postgres.
  - It ran the OddsPortal, Betfair, ESPN and MLB probables jobs, and it was the only host scoring predictions.
- **Why two databases.**
  - #357 (closing #355) moved scoring and the fixed-schedule jobs to the local scheduler so the agent, scraper and scorer share one database.
  - #355 expected Neon to remain a "cold scrape mirror" of the Lambda jobs.
  - No record decides that jobs AWS already ran should also run locally. `fetch-betfair-exchange-epl` and `fetch-oddsportal-results-epl` ran on both hosts.
- **Monitoring gap.** The AWS health check read Neon, so it could not see local heartbeats. When the mini-PC's WSL guest was shut down on 2026-10-14, collection there stopped for three days with no alert.
- **Data split.**
  - The local database held 1017 MB, including all MLB history, the predictions and the agent's tables.
  - Neon held 434 MB, mostly Odds API and Betfair EPL snapshots.

The owner considered two single-database topologies. One kept Neon as the database of record, with both hosts writing to it. Neon's free tier caps storage at 0.5 GB, and the merged data is about 1.4 GB. The owner, who is on the free tier, chose the other: the mini-PC (#483).

## Decision

The mini-PC is the only production host, and its Postgres is the production database.

- Every scheduled job runs on the mini-PC's local scheduler, managed by the systemd unit `odds-scheduler.service`.
- The defaults in `odds_core.config.SchedulerConfig` `bootstrap_jobs` include the jobs that moved from AWS: `fetch-odds`, `fetch-scores`, `update-status` and `check-health`.
- The AWS EventBridge rules are disabled.
  - The Lambdas, ECR images and Terraform stay in place, so the move is reversible.
  - `deploy-prod.yml` is not run, by dispatch or by a published release, because its post-deploy invocations re-arm every rule.
- The Neon database is a frozen archive. Its EPL and MLB events, snapshots and odds were merged into the mini-PC database on 2026-10-17 (#483).
- The agent runs on the mini-PC, by hand, as the restricted `odds_agent` role (`AGENT_DATABASE_URL`, `odds_lambda.jobs.agent_run`).

## Consequences

- One database holds every job's heartbeats, so `check-health` sees all of them. Nothing outside the host notices if the host itself stops; #489 tracks an external dead-man's switch.
- Availability depends on one Windows machine running a WSL guest. The guest must start at boot and must not be idled out by WSL. `odds-scheduler.service` restarts the scheduler when it dies ([Run the scheduler as a service](../../wiki/scheduling/howto-run-scheduler-service.md)).
- Deploys reach production only through `deploy.sh` on the mini-PC, which also runs the migrations. CI does not deploy.
- "Production" in `AGENTS.md` now means the mini-PC database. A development machine still never points `DATABASE_URL` at it.
- `odds copy from-prod` reads `PROD_DATABASE_URL`, which on existing machines still names the Neon archive.
- Odds API quota is spent from the mini-PC's key pool. `fetch-odds` and `fetch-scores` stay EPL-only, matching the AWS deployment, through the sport allowlist in `odds_lambda.scheduling.jobs`.
- Collection jobs keep running a sport through its off-season (#487).
- The AWS-facing workflows lose their target: `schedule-health-check.yml` fails every disabled rule, and `daily-validation.yml` reads the `production` environment's `DATABASE_URL`, the Neon archive.

## Evidence

- Audit and follow-ups: #483 (snapshot and decisions), #484, #485, #486, #487, #489.
- Earlier decisions and their records: #355, #357, #372, #454.
- Commits: 1e33618 (last production image), 250733a (`agent-run` and `daily-digest` removed from the default bootstrap).
- Code and config: `odds_core.config.SchedulerConfig`, `odds_lambda.scheduling.jobs`, `odds_lambda.jobs.agent_run`, `deploy.sh`, `odds-scheduler.service`, `.github/workflows/deploy-prod.yml`, `.github/workflows/schedule-health-check.yml`, `.github/workflows/daily-validation.yml`, `deployment/aws/terraform/eventbridge.tf`.
- Wiki: [Where each job runs](../../wiki/scheduling/job-deployment-matrix.md), [Local scheduler](../../wiki/scheduling/local-scheduler.md).
