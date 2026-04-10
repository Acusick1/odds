"""normalize team names in events table

Revision ID: e7f2a1b3c4d5
Revises: 021f43706c10
Create Date: 2026-04-10 12:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "e7f2a1b3c4d5"
down_revision = "021f43706c10"
branch_labels = None
depends_on = None

# Alias -> canonical (same mappings as odds_core.team._TEAM_ALIASES).
# Only includes names that could exist in the events table.
_ALIAS_MAP: dict[str, str] = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nottingham",
    "Sheffield United": "Sheffield Utd",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


def upgrade() -> None:
    for alias, canonical in _ALIAS_MAP.items():
        for col in ("home_team", "away_team"):
            op.execute(
                f"UPDATE events SET {col} = '{canonical}' WHERE {col} = '{alias}'"  # noqa: S608
            )


def downgrade() -> None:
    # Reverse mapping is lossy (multiple aliases map to one canonical).
    # Use the longest/most-specific alias as the reverse form.
    _REVERSE: dict[str, str] = {
        "Bournemouth": "AFC Bournemouth",
        "Brighton": "Brighton and Hove Albion",
        "Cardiff": "Cardiff City",
        "Huddersfield": "Huddersfield Town",
        "Hull": "Hull City",
        "Ipswich": "Ipswich Town",
        "Leicester": "Leicester City",
        "Leeds": "Leeds United",
        "Manchester Utd": "Manchester United",
        "Newcastle": "Newcastle United",
        "Norwich": "Norwich City",
        "Nottingham": "Nottingham Forest",
        "Sheffield Utd": "Sheffield United",
        "Stoke": "Stoke City",
        "Swansea": "Swansea City",
        "Tottenham": "Tottenham Hotspur",
        "West Brom": "West Bromwich Albion",
        "West Ham": "West Ham United",
        "Wolves": "Wolverhampton Wanderers",
    }
    for canonical, alias in _REVERSE.items():
        for col in ("home_team", "away_team"):
            op.execute(
                f"UPDATE events SET {col} = '{alias}' WHERE {col} = '{canonical}'"  # noqa: S608
            )
