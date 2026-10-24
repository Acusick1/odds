---
title: Operations
description: AWS infrastructure and deployment, CI/CD workflows, Discord alerts, heartbeats and the health monitor, and reporting surfaces including the daily digest.
---

# Operations

How the system is deployed, verified and watched: production is deployed with `deploy.sh` ([Run the scheduler as a systemd service](../scheduling/howto-run-scheduler-service.md)); this area covers the dormant Lambdas and their Terraform, the GitHub Actions that test, deploy and check them, the alert and heartbeat machinery that tells an operator something stalled, and the reports the system produces. Deployed AWS state is not in the repo; pages say so and give the command that checks it.

<!-- pages:start -->
### Reference

- [Alerts and the alert_history table](alerts.md) — Discord delivery, the alert helpers and job_alert_context, AlertConfig, rate limiting and the alert_history taxonomy.
- [AWS infrastructure and deployment](aws-infrastructure.md) — The dormant AWS stack — the Lambda function and image, Terraform and bootstrap layout and variables, IAM, SSM, logs, secrets, constraints and costs.
- [CI/CD workflows](ci-cd-workflows.md) — What each GitHub Actions workflow runs and when — tests, lint, ephemeral dev deploy, the AWS prod deploy that must not run, daily validation, live-API and schedule health checks — and what they do not exercise.
- [Health monitor, check-health job and odds status](health-monitor.md) — HealthMonitor, the check-health job's six checks, heartbeat purge, its own alert path, and what odds status show, quota and events print and read.
- [The daily-digest job](daily-digest.md) — Queries, embed layout, delivery, schedule and heartbeat of odds_lambda.jobs.daily_digest.

### Explanation

- [How failures are detected](alerting-and-heartbeats.md) — How event-driven alerts, heartbeats and the polling health monitor fit together, which jobs each covers, and the blind spots.
- [Reporting surfaces](reporting-surfaces.md) — What the system reports about itself and its outputs, through which channel, and what each surface cannot show.

### How-to guides

- [Deploy to AWS by hand](howto-deploy-aws.md) — Deploy the dormant AWS stack by hand — first-deploy bootstrap, then build, apply, migrate, activate rules and verify.
<!-- pages:end -->
