---
title: Scheduling
description: The job registry and where each job runs, self-scheduling and its decision engine, the AWS EventBridge and local APScheduler backends, and the Lambda handler.
---

# Scheduling

Every data job is registered once, decides its own next run, and books that run through a backend: an EventBridge rule on AWS or an APScheduler entry on a local host. This area covers the registry, the decision logic, both backends, and the how-tos for adding and debugging jobs.

<!-- pages:start -->
### Reference

- [Job registry](job-registry.md) — Which jobs exist in the registry, the bootstrap and smoke maps, which jobs are per sport and how compound names are built, and what JobContext carries.
- [Local scheduler](local-scheduler.md) — The APScheduler-on-Postgres backend, its scheduling primitives, the odds scheduler commands, bootstrap jobs and the cron table.
- [Scheduler backends](scheduler-backends.md) — The SchedulerBackend interface and selection, the AWS EventBridge backend and its env contract, the Railway shim, health checks, and the Lambda handler EventBridge invokes.
- [Scheduling decision engine](decision-engine.md) — decide_forward and decide_backward and their resilient wrappers, CadenceConfig and per-job cadences, overnight windows, and the self_schedule and kickoff helpers.
- [Where each job runs](job-deployment-matrix.md) — The AWS and local gates, the per-job placement and schedule table, heartbeat expectations per job, and how to check deployed state.

### Explanation

- [Self-scheduling](self-scheduling.md) — Why jobs book their own next run instead of a fixed cron, the pre-schedule, work and re-schedule pattern, and how each job's failure policy can end its chain.

### How-to guides

- [Add a scheduler job](howto-add-scheduler-job.md) — Steps to add a job (entry point, registry, per-sport or cron registration, self-scheduling, alert wrapper and heartbeat, local test, Terraform rule and matrix update) and the traps in each.
- [Debug the AWS self-scheduling jobs](howto-debug-aws-scheduler.md) — A read-only workflow for stale data or missing tiers on AWS covering rules, CloudWatch logs, database versus fixtures, tier coverage and a safe manual invoke.
- [Run the scheduler as a systemd service](howto-run-scheduler-service.md) — Install the scheduler's systemd unit on an always-on host, start it at Windows boot on a WSL host, and redeploy it with deploy.sh.
<!-- pages:end -->
