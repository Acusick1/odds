# Betting Odds Pipeline & Agent

A single-user, sport-agnostic betting odds pipeline with per-league LLM betting agents.
The goal is profit; any approach (rules, models, agents) is a candidate judged on
live paper bets ([goal and strategy](wiki/foundations/goal-and-strategy.md)).
Shared infrastructure (odds collection, storage, paper betting, scheduling, alerting) is
reused across leagues; each league adds its own data sources, feature extractors, agent
prompts, and MCP tools.

Current collection and agent status: [deployment matrix](wiki/scheduling/job-deployment-matrix.md#where-each-job-runs).

The pipeline collects book prices (The Odds API), UK retail odds (OddsPortal,
football-data.co.uk) and Betfair Exchange prices, stores hybrid raw + normalized data, and
trains models on that data. The betting agent researches events via web search and Playwright,
consumes pipeline data through MCP tools, and places paper bets with explicit reasoning.

## Repository Layout

```
packages/
├── odds-core/      # Shared models, config, database, alerts
├── odds-lambda/    # Data fetching, storage, scheduling, scheduled jobs
├── odds-analytics/ # Candidate replay backtest and report, ML training / feature extraction
├── odds-mcp/       # MCP server — the betting agent's tool interface
└── odds-cli/       # CLI commands
```

## Quick Start

Prerequisites: Python 3.12+, Docker, [uv](https://github.com/astral-sh/uv) and a [The Odds API](https://the-odds-api.com/) key. Setup, configuration and the first fetch are in [Develop locally](wiki/foundations/howto-develop-locally.md).

## Documentation

- [wiki/index.md](wiki/index.md): how every subsystem works, why, and its hazards, with citations into the code. Start here.
- [docs/adr/](docs/adr/): architecture decisions.
- [AGENTS.md](AGENTS.md): constraints and conventions for coding agents.

## License

Private project — all rights reserved.
