"""Configuration objects used by the backtesting framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

__all__ = [
    "BetSizingConfig",
    "BetConstraintsConfig",
    "BacktestConfig",
]


@dataclass
class BetSizingConfig:
    """Configuration for bet sizing strategy."""

    method: str = "fractional_kelly"
    kelly_fraction: float = 0.25
    flat_stake_amount: float = 100.0
    percentage_stake: float = 0.02

    def to_dict(self) -> dict:
        """Convert to primitive dictionary."""
        return {
            "method": self.method,
            "kelly_fraction": self.kelly_fraction,
            "flat_stake_amount": self.flat_stake_amount,
            "percentage_stake": self.percentage_stake,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BetSizingConfig:
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class BetConstraintsConfig:
    """Configuration for bet constraints and filters."""

    min_bet_size: float = 10.0
    max_bet_size: float | None = None
    max_bet_percentage: float = 0.05
    min_odds: int | None = None
    max_odds: int | None = None
    allowed_markets: list[str] | None = None
    allowed_bookmakers: list[str] | None = None
    include_transaction_costs: bool = False
    transaction_cost_rate: float = 0.0

    def to_dict(self) -> dict:
        """Convert to primitive dictionary."""
        return {
            "min_bet_size": self.min_bet_size,
            "max_bet_size": self.max_bet_size,
            "max_bet_percentage": self.max_bet_percentage,
            "min_odds": self.min_odds,
            "max_odds": self.max_odds,
            "allowed_markets": self.allowed_markets,
            "allowed_bookmakers": self.allowed_bookmakers,
            "include_transaction_costs": self.include_transaction_costs,
            "transaction_cost_rate": self.transaction_cost_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BetConstraintsConfig:
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class BacktestConfig:
    """Configuration for running a backtest."""

    initial_bankroll: float
    start_date: datetime
    end_date: datetime
    decision_hours_before_game: float = 1.0
    sizing: BetSizingConfig = field(default_factory=BetSizingConfig)
    constraints: BetConstraintsConfig = field(default_factory=BetConstraintsConfig)

    @property
    def bet_sizing_method(self) -> str:
        return self.sizing.method

    @property
    def kelly_fraction(self) -> float:
        return self.sizing.kelly_fraction

    @property
    def flat_stake_amount(self) -> float:
        return self.sizing.flat_stake_amount

    @property
    def percentage_stake(self) -> float:
        return self.sizing.percentage_stake

    @property
    def min_bet_size(self) -> float:
        return self.constraints.min_bet_size

    @property
    def max_bet_size(self) -> float | None:
        return self.constraints.max_bet_size

    @property
    def max_bet_percentage(self) -> float:
        return self.constraints.max_bet_percentage

    @property
    def min_odds(self) -> int | None:
        return self.constraints.min_odds

    @property
    def max_odds(self) -> int | None:
        return self.constraints.max_odds

    @property
    def allowed_markets(self) -> list[str] | None:
        return self.constraints.allowed_markets

    @property
    def allowed_bookmakers(self) -> list[str] | None:
        return self.constraints.allowed_bookmakers

    @property
    def include_transaction_costs(self) -> bool:
        return self.constraints.include_transaction_costs

    @property
    def transaction_cost_rate(self) -> float:
        return self.constraints.transaction_cost_rate

    def to_dict(self) -> dict:
        """Convert to primitive dictionary."""
        return {
            "initial_bankroll": self.initial_bankroll,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "decision_hours_before_game": self.decision_hours_before_game,
            "sizing": self.sizing.to_dict(),
            "constraints": self.constraints.to_dict(),
        }
