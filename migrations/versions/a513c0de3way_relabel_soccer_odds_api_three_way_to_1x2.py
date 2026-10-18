"""relabel soccer Odds API three-way market from h2h to 1x2

The Odds API keys the home/draw/away market ``h2h`` and names teams in full.
Ingest now stores it as ``1x2`` with the canonical team names of the events
table. This rewrites the market key and the team outcome names of rows stored
before that change. Only soccer events and markets with a Draw outcome are
touched, so the migration is idempotent and leaves MLB/NBA ``h2h`` alone.

``down_revision`` may need rebasing if other migrations land first.

Revision ID: a513c0de3way
Revises: e6f24ef60549
Create Date: 2026-10-18 12:00:00.000000

"""

import json

from alembic import op

revision = "a513c0de3way"
down_revision = "e6f24ef60549"
branch_labels = None
depends_on = None

TEAM_NAMES = json.dumps(
    {
        "Brighton and Hove Albion": "Brighton",
        "Coventry City": "Coventry",
        "Hull City": "Hull",
        "Ipswich Town": "Ipswich",
        "Leeds United": "Leeds",
        "Manchester United": "Manchester Utd",
        "Newcastle United": "Newcastle",
        "Nottingham Forest": "Nottingham",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves",
    }
)

RELABEL_ODDS_SQL = f"""
        UPDATE odds
        SET market_key = '1x2',
            outcome_name = COALESCE(CAST('{TEAM_NAMES}' AS jsonb) ->> outcome_name, outcome_name)
        WHERE market_key = 'h2h'
          AND event_id IN (SELECT id FROM events WHERE sport_key LIKE 'soccer\\_%')
          AND (event_id, bookmaker_key, odds_timestamp) IN (
            SELECT event_id, bookmaker_key, odds_timestamp
            FROM odds
            WHERE market_key = 'h2h' AND outcome_name = 'Draw'
          )
"""

_IS_THREE_WAY_H2H = """
    m->>'key' = 'h2h' AND EXISTS (
        SELECT 1 FROM jsonb_array_elements(m->'outcomes') AS d WHERE d->>'name' = 'Draw'
    )
"""

# raw_data is json, not jsonb: cast in, and Postgres converts back on assignment.
# Only bookmakers holding a matching market are rebuilt, so the others stay as they are.
RELABEL_SNAPSHOTS_SQL = f"""
        UPDATE odds_snapshots s
        SET raw_data = jsonb_set(
            s.raw_data::jsonb,
            '{{bookmakers}}',
            (
                SELECT jsonb_agg(
                    CASE WHEN EXISTS (
                        SELECT 1 FROM jsonb_array_elements(bk->'markets') AS m
                        WHERE {_IS_THREE_WAY_H2H}
                    )
                    THEN jsonb_set(
                        bk,
                        '{{markets}}',
                        (
                            SELECT jsonb_agg(
                                CASE WHEN {_IS_THREE_WAY_H2H}
                                THEN jsonb_set(
                                    jsonb_set(m, '{{key}}', '"1x2"'),
                                    '{{outcomes}}',
                                    (
                                        SELECT jsonb_agg(
                                            jsonb_set(
                                                o,
                                                '{{name}}',
                                                to_jsonb(COALESCE(
                                                    CAST('{TEAM_NAMES}' AS jsonb) ->> (o->>'name'),
                                                    o->>'name'
                                                ))
                                            )
                                        )
                                        FROM jsonb_array_elements(m->'outcomes') AS o
                                    )
                                )
                                ELSE m END
                            )
                            FROM jsonb_array_elements(bk->'markets') AS m
                        )
                    )
                    ELSE bk END
                )
                FROM jsonb_array_elements(s.raw_data::jsonb->'bookmakers') AS bk
            )
        )
        WHERE s.event_id IN (SELECT id FROM events WHERE sport_key LIKE 'soccer\\_%')
          AND EXISTS (
            SELECT 1
            FROM jsonb_array_elements(s.raw_data::jsonb->'bookmakers') AS bk,
                 jsonb_array_elements(bk->'markets') AS m
            WHERE {_IS_THREE_WAY_H2H}
          )
"""


def upgrade() -> None:
    op.execute(RELABEL_ODDS_SQL)
    op.execute(RELABEL_SNAPSHOTS_SQL)


def downgrade() -> None:
    # Irreversible: OddsPortal and Betfair rows are also 1x2, so the Odds API
    # rows cannot be told apart afterwards.
    pass
