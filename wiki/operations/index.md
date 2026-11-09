---
title: Operations
description: AWS infrastructure and deployment, CI/CD workflows, Discord alerts, heartbeats and the health monitor, and reporting surfaces.
---

# Operations

How the system is deployed, verified and watched: production is deployed with `deploy.sh` ([Run the scheduler as a systemd service](../scheduling/howto-run-scheduler-service.md)); this area covers the dormant Lambdas and their Terraform, the GitHub Actions that test, deploy and check them, the alert and heartbeat machinery that tells an operator something stalled, and the reports the system produces. Deployed AWS state is not in the repo; pages say so and give the command that checks it.

<!-- pages:start -->
### Reference

- [Alerts and the alert_history table](alerts.md) — Discord delivery, the alert helpers and job_alert_context, AlertConfig, rate limiting and the alert_history taxonomy.
- [AWS infrastructure and deployment](aws-infrastructure.md) — The dormant AWS stack — the Lambda function and image, Terraform and bootstrap layout and variables, IAM, SSM, logs, secrets, constraints and costs.
- [CI/CD workflows](ci-cd-workflows.md) — What each GitHub Actions workflow runs and when — tests, lint, ephemeral dev deploy, the AWS prod deploy that must not run, daily validation, live-API and schedule health checks — and what they do not exercise.
- [Health monitor, check-health job and odds status](health-monitor.md) — HealthMonitor, the check-health job's six checks, heartbeat purge, its own alert path, and what odds status show, quota and events print and read.
- [odds-web, the read-only web UI](odds-web.md) — What odds web serve runs, the database role and session it reads through, the report data behind the portfolio page, the status strip and the data-health page, its cache and failure bounds, the vendored scripts, and how deploy.sh and systemd run it.

### Explanation

- [How failures are detected](alerting-and-heartbeats.md) — How event-driven alerts, heartbeats and the polling health monitor fit together, which jobs each covers, and the blind spots.
- [Reporting surfaces](reporting-surfaces.md) — What the system reports about itself and its outputs, through which channel, and what each surface cannot show.

### How-to guides

- [Deploy to AWS by hand](howto-deploy-aws.md) — Deploy the dormant AWS stack by hand — first-deploy bootstrap, then build, apply, migrate, activate rules and verify.
- [Set up the odds-web host](howto-set-up-odds-web-host.md) — Run the one-time, sudo-only wizard that creates the odds_web role, writes WEB_DATABASE_URL, installs the unit and its sudo rules, and exposes the UI on the tailnet, before the first deploy.
<!-- pages:end -->
