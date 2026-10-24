# Betting Odds Pipeline & Agent

A single-user, sport-agnostic betting odds pipeline with sport-specific LLM betting agents.
The goal is profit; any approach (today the betting agents; models and mechanical rules qualify
too) is a candidate judged on live paper trades ([goal and strategy](wiki/foundations/goal-and-strategy.md)).
Shared infrastructure (odds collection, storage, paper trading, scheduling, alerting) is
reused across sports; each sport adds its own data sources, feature extractors, agent
prompts, and MCP tools.

**Active sports:** EPL football and MLB baseball (agents in interactive evaluation).
NBA infrastructure exists but is deprioritized.

The pipeline collects bookmaker odds (The Odds API), UK retail odds (OddsPortal,
football-data.co.uk) and Betfair Exchange prices, stores hybrid raw + normalized data, and
trains models on that data. The betting agent researches matches via web search and Playwright,
consumes pipeline data through MCP tools, and places paper trades with explicit reasoning.

## Repository Layout

```
packages/
├── odds-core/      # Shared models, config, database, alerts
├── odds-lambda/    # Data fetching, storage, scheduling, scheduled jobs
├── odds-analytics/ # Backtesting, strategies, ML / feature extraction
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
