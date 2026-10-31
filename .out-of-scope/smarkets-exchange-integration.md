# Smarkets exchange integration

The pipeline doesn't integrate Smarkets' own API as a second exchange.

## Why this is out of scope

Every candidate is scored by EV at the price taken against the Betfair Exchange close ([goal and strategy](../wiki/foundations/goal-and-strategy.md)). Betfair is therefore the exchange reference the project is built on, and its price, depth and matched volume are already collected live by the Betfair job.

Smarkets lists the same EPL markets with roughly a tenth of Betfair's liquidity. Its order book would be a thinner copy of a signal that is already collected, and integrating it would mean another account, another client, and another source to keep healthy in the stale-source check.

The Odds API's `uk` region already returns Smarkets prices, as the `smarkets` bookmaker key ([sport and market keys](../wiki/collection/sport-and-market-keys.md)). A candidate that wants a second exchange's price can name that bookmaker in a fetch without a new integration.

## Prior requests

- #428: "Evaluate Smarkets as an exchange-volume data source"
