---
type: howto
title: Add a setting
description: Steps to add a configuration setting so it loads from .env and reaches the Lambdas.
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
   - Write list fields as JSON arrays; the template's comma-separated lists make `odds_core.config.get_settings` raise ([List fields and the template](configuration.md#list-fields-and-the-template)).
3. **Wire Terraform if a Lambda needs it.** Declare a variable in `deployment/aws/terraform/variables.tf` and pass it in the `environment` block of `deployment/aws/terraform/lambda.tf` (scheduler) or `deployment/aws/terraform/scraper.tf` (scraper).
   - Each block sets its own subset of variables, so a setting not listed there runs on its code default ([AWS infrastructure](../operations/aws-infrastructure.md)).
4. **Read it through Settings.**
   - A variable read with `os.getenv` bypasses `.env` ([Configuration](configuration.md#variables-read-outside-settings)).

## Related

- [Configuration](configuration.md)
- [Set up and run locally](howto-develop-locally.md)
