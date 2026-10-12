---
title: Scheduling
description: The job registry and where each job runs, self-scheduling and its decision engine, the AWS EventBridge and local APScheduler backends, and the Lambda handler.
---

# Scheduling

Every data job is registered once, decides its own next run, and books that run through a backend: an EventBridge rule on AWS or an APScheduler entry on a local host. This area covers the registry, the decision logic, both backends, and the how-tos for adding and debugging jobs.

<!-- pages:start -->
### Processes

- [Self-scheduling and the decision engine](self-scheduling.md) — How each job decides and books its own next run, the forward/backward decisions, every job's cadence and overnight window, and the per-job failure policies that can break a chain.

### Components

- [Jobs: registry, context and where each runs](jobs-and-deployment-matrix.md) — The 14 registered jobs, per-sport compound names and JobContext, and for each job where it can run (scheduler Lambda, scraper Lambda, local default or opt-in) and on what schedule, plus how to check the deployed state.
- [Local scheduler and always-on host](local-scheduler.md) — The APScheduler-on-Postgres backend, odds scheduler start/run-once/smoke/list-jobs, bootstrap jobs and cron floors, and running it as a systemd service redeployed by deploy.sh.
- [Scheduler backends, EventBridge rules and the Lambda handler](scheduler-backends.md) — The SchedulerBackend interface and selection, how the AWS backend rewrites placeholder EventBridge rules, the Lambda handler's payload routing and extra failure alert, and the Railway no-op shim.

### How-tos

- [Add a scheduler job](howto-add-scheduler-job.md) — Steps to add a job (entry point, registry, per-sport or cron registration, self-scheduling, alert wrapper and heartbeat, local test, Terraform rule and matrix update) and the traps in each.
- [Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md) — A read-only workflow for stale data or missing tiers on AWS covering rules, CloudWatch logs, database versus fixtures, tier coverage, common failures and a safe manual invoke.
<!-- pages:end -->
