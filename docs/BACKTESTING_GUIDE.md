# Backtesting Reference

Strategies, bet sizing, metrics and extension points for the backtesting engine
(`packages/odds-analytics/odds_analytics/backtesting/`). Command flags are in
[CLI.md](CLI.md#backtesting-commands).

## Quick Start

```bash
# List strategies
odds backtest list-strategies

# Run backtest
odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31 --output-json results.json

# View results
odds backtest show results.json --verbose
```

## Available Strategies

### Flat Betting Strategy

**Name:** `flat`

Simple baseline strategy that bets a fixed amount on every game.

**Parameters:**

- `market`: "h2h", "spreads", or "totals" (default: "h2h")
- `outcome_pattern`: "home", "away", "draw", or "favorite" (default: "home")
- `bookmaker`: Which bookmaker to use (default: "fanduel")

**Example:**

```bash
odds backtest run --strategy flat --start 2024-10-01 --end 2024-12-31
```

### Expected Value (EV) Strategy

**Name:** `basic_ev`

Bets when retail bookmakers offer odds that differ significantly from sharp bookmakers, indicating positive expected value.

**Parameters:**

- `sharp_book`: Sharp bookmaker for "true" odds (default: "pinnacle")
- `retail_books`: Retail books to find +EV bets (default: ["fanduel", "draftkings", "betmgm"])
- `min_ev_threshold`: Minimum EV required to bet (default: 0.03 = 3%)
- `markets`: Markets to consider (default: ["h2h", "spreads", "totals"])

**Example:**

```bash
odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31 --kelly-fraction 0.25
```

### Arbitrage Strategy

**Name:** `arbitrage`

Finds risk-free profit opportunities by betting both sides of a market across different bookmakers.

**Parameters:**

- `min_profit_margin`: Minimum profit margin to pursue (default: 0.01 = 1%)
- `max_hold`: Maximum market hold to consider (default: 0.10 = 10%)
- `bookmakers`: Bookmakers to consider (default: all major books)

**Example:**

```bash
odds backtest run --strategy arbitrage --start 2024-10-01 --end 2024-12-31
```

### Model-Driven Strategies

**Names:** `xgb_line_movement`, `lstm_line_movement`

Bet on outcomes where a trained line-movement model predicts the market will move in their
favour. Pass the model artifact with `--model-path` (`XGBoostLineMovementStrategy` in
`xgboost_line_movement.py`, `LSTMLineMovementStrategy` in `lstm_line_movement.py`).

**Example:**

```bash
odds backtest run --strategy xgb_line_movement --start 2024-10-01 --end 2024-12-31 --model-path models/model.pkl
```

## Bet Sizing Methods

### Fractional Kelly (Default)

```bash
--bet-sizing fractional_kelly --kelly-fraction 0.25
```

Uses the Kelly Criterion to size bets from edge and odds. Quarter-Kelly (0.25) is the default, chosen for lower volatility while capturing ~75% of full Kelly returns (inferred).

### Flat Betting

```bash
--bet-sizing flat
```

Bets a fixed amount on every opportunity (good for baseline comparison). The stake
amount comes from `BetSizingConfig.flat_stake_amount` (default 100) and is set
programmatically; the CLI uses the default.

### Percentage Betting

```bash
--bet-sizing percentage
```

Bets a fixed percentage of current bankroll on every opportunity. The percentage comes
from `BetSizingConfig` (default 0.02) and is set programmatically; the CLI uses the default.

## Performance Metrics

### Return on Investment (ROI)

`(Total Profit / Initial Bankroll) × 100`

Percentage return on starting bankroll.

### Win Rate

`(Winning Bets / Total Decided Bets) × 100`

Percentage of bets that won (excludes pushes).

### Sharpe Ratio

`Average Daily Return / Standard Deviation of Daily Returns` (the engine uses a risk-free rate of 0)

Risk-adjusted return metric. Higher is better:

- >1.0 is good
- >2.0 is excellent
- >3.0 is exceptional

### Sortino Ratio

Similar to Sharpe but only penalizes downside volatility. Generally higher than Sharpe for positive-returning strategies.

### Max Drawdown

Largest peak-to-trough decline in bankroll. Important for understanding worst-case risk.

### Profit Factor

`Total Winning Profit / Total Losing Loss`

How much you make per dollar lost:

- >1.0 means profitable
- >2.0 is very good

### Calmar Ratio

`ROI / |Max Drawdown %|`

Return per unit of drawdown risk. Higher is better.

## Creating Custom Strategies

To create a new strategy, inherit from `BettingStrategy`:

```python
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.backtesting import BettingStrategy, BacktestEvent, BetOpportunity, BacktestConfig
from odds_core.models import Odds

class MyStrategy(BettingStrategy):
    def __init__(self, my_param: float = 0.05):
        super().__init__(
            name="MyStrategy",
            my_param=my_param,
        )

    async def evaluate_opportunity(
        self,
        event: BacktestEvent,  # Type-safe event with guaranteed scores
        odds_snapshot: list[Odds],
        config: BacktestConfig,
        session: AsyncSession | None = None,  # engine passes the reader's session
    ) -> list[BetOpportunity]:
        opportunities = []

        # Your strategy logic here
        # event.home_score and event.away_score are guaranteed to exist (not None)

        return opportunities
```

**Key Points:**

- Use `BacktestEvent` instead of `Event` for the event parameter
- `BacktestEvent` guarantees `home_score` and `away_score` are not None (type-safe)
- No need to check if scores exist - they're always present during backtesting

Register it in the `STRATEGIES` dict in `packages/odds-cli/odds_cli/commands/backtest.py`:

```python
STRATEGIES = {
    # ...existing strategies...
    "my_strategy": MyStrategy,
}
```

## Programmatic Usage

```python
import asyncio
from datetime import UTC, datetime

from odds_analytics.backtesting import BacktestConfig, BacktestEngine, BetSizingConfig
from odds_analytics.strategies import BasicEVStrategy
from odds_core.database import async_session_maker
from odds_lambda.storage.readers import OddsReader

async def run_backtest():
    strategy = BasicEVStrategy()

    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=datetime(2024, 10, 1, tzinfo=UTC),
        end_date=datetime(2024, 12, 31, tzinfo=UTC),
        sport_key="soccer_epl",
        sizing=BetSizingConfig(
            method="fractional_kelly",
            kelly_fraction=0.25,
        ),
    )

    async with async_session_maker() as session:
        # BacktestEngine takes an OddsReader, not a raw session
        engine = BacktestEngine(strategy, config, OddsReader(session))
        result = await engine.run()

    # Access results programmatically
    print(f"ROI: {result.roi:.2f}%")
    print(f"Sharpe: {result.sharpe_ratio:.2f}")

    # Export
    result.to_json("my_results.json")
    result.to_csv("my_bets.csv")

    return result

asyncio.run(run_backtest())
```

## Data Requirements

The backtesting system requires:

- Historical events with final scores (`Event.status = FINAL`)
- Historical odds snapshots (`Odds` records with timestamps)
- Odds available at decision time (typically 1 hour before game)

Use the backfill commands to populate historical data (see
[Historical Backfill Workflow](CLI.md#historical-backfill-workflow)).
