---
type: howto
title: Add a setting
description: Steps to add a configuration setting so it loads from .env and reaches the production host.
sources:
  - odds_core.config
  - .env.example
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/scraper.tf
  - deployment/aws/terraform/variables.tf
status: stable
---

# Add a setting

How settings are structured and which variables bypass them is in [Configuration](configuration.md).

1. **Add the field** to the right class in `odds_core.config` with a type and a default.
   - A section with an `env_prefix` reads `<PREFIX><FIELD>`; a bare section reads the upper-cased field name ([Sections](configuration.md#sections)).
2. **Add it to `.env.example`.**
   - Write list fields as JSON arrays and dict fields (such as `BOOKMAKERS`) as JSON objects; the template's comma-separated lists make `odds_core.config.get_settings` raise ([List fields and the template](configuration.md#list-fields-and-the-template)).
3. **Set it on the production host** if it needs a non-default value: add it to the `.env` in the mini-PC's repo checkout, which `odds-scheduler.service` reads, then redeploy ([Run the scheduler as a systemd service](../scheduling/howto-run-scheduler-service.md)).
   - The AWS Lambdas are dormant ([ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)). For them, a setting would also need a variable in `deployment/aws/terraform/variables.tf` passed in the `environment` block of `deployment/aws/terraform/lambda.tf` (scheduler) or `deployment/aws/terraform/scraper.tf` (scraper); each block sets its own subset, so an unlisted setting runs on its code default ([AWS infrastructure](../operations/aws-infrastructure.md)).
4. **Read it through Settings.**
   - A variable read with `os.getenv` bypasses `.env` ([Configuration](configuration.md#variables-read-outside-settings)).

## Related

- [Configuration](configuration.md)
- [Set up and run locally](howto-develop-locally.md)
