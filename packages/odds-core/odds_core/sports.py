"""Supported sport keys for the odds pipeline.

Values match The Odds API's sport key convention and are used as the
``sport_key`` on ``Event`` rows and as the ``sport`` argument across
scheduler jobs, MCP tools, and CLI commands.
"""

from typing import Literal, get_args

SportKey = Literal["soccer_epl", "baseball_mlb"]

SUPPORTED_SPORTS: tuple[SportKey, ...] = get_args(SportKey)
