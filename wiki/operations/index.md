---
title: Operations
description: AWS infrastructure and deployment, CI/CD workflows, Discord alerts, heartbeats and the health monitor, and reporting surfaces including the daily digest.
---

# Operations

How the system is deployed, verified and watched: the two Lambdas and their Terraform, the GitHub Actions that deploy and check them, the alert and heartbeat machinery that tells an operator something stalled, and the reports the system produces. Deployed AWS state is not in the repo; pages say so and give the command that checks it.

<!-- pages:start -->
### Processes

- [Reporting surfaces and the daily digest](reporting-and-daily-digest.md) — What the system reports and through which channel (Discord, odds status, MCP, traces), and the opt-in, local-only EPL daily-digest job with its single-model, home-side convention.

### Components

- [Alerts, heartbeats and the health monitor](alerts-and-health.md) — How alerts reach Discord, the alert-type taxonomy and severity-blind rate limiting in alert_history, heartbeats and their expectations, the check-health job's five checks, and which jobs have no heartbeat at all.
- [AWS infrastructure and deployment](aws-infrastructure.md) — The two Lambda functions and images, Terraform and bootstrap layout and variables, IAM/SSM/logs and costs, and the manual first-deploy and per-deploy steps, including activating self-scheduling rules.
- [CI/CD workflows](ci-cd-workflows.md) — What each GitHub Actions workflow runs and when — tests, lint, ephemeral dev deploy, gated prod deploy, daily validation, live-API and schedule health checks — and what they do not exercise.
<!-- pages:end -->
