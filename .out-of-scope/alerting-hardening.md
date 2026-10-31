# Alerting hardening: second channel, atomic rate limit, injected manager

Alerting stays a single Discord channel, with a module-level `alert_manager` and a rate limit that reads `alert_history` and then records to it.

## Why this is out of scope

This is a single-user system, and one person reads its alerts. Each proposed change adds machinery without changing what that person sees:

- **A second channel** means a second set of credentials and a second delivery path to keep working, so that alerts still arrive if Discord breaks. Alerts are recorded in `alert_history` whether or not delivery succeeds, so a broken webhook can already be seen in the database. Silence from the health check is itself a signal to whoever runs the system.
- **An atomic check-and-record** would only prevent two jobs that raise the same alert type at the same moment from both sending it. The cost of that race is one duplicate Discord message.
- **An injected or lazily built `alert_manager`** would make tests slightly simpler and settings re-readable at runtime. Settings aren't changed at runtime, and the tests already patch the manager.

## Prior requests

- #457: "Alerting: add second channel, atomic rate limiting, injectable alert_manager"
