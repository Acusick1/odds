"""relabel 1x2 market_key from h2h to 1x2

Odds rows with outcome_name='Draw' are 1x2 (3-way) data that was stored
under market_key='h2h'. This migration relabels them to '1x2', including
sibling home/away rows in the same snapshot group, and patches the
raw_data JSON on the parent odds_snapshots.

Revision ID: 920aee156d9c
Revises: 5eca7168d060
Create Date: 2026-04-15 14:10:32.881407

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "920aee156d9c"
down_revision = "5eca7168d060"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Step 1: Update odds table — relabel all h2h rows that share a
    # (event_id, bookmaker_key, odds_timestamp) group with a Draw outcome.
    op.execute("""
        UPDATE odds
        SET market_key = '1x2'
        WHERE market_key = 'h2h'
          AND (event_id, bookmaker_key, odds_timestamp) IN (
            SELECT event_id, bookmaker_key, odds_timestamp
            FROM odds
            WHERE market_key = 'h2h' AND outcome_name = 'Draw'
          )
    """)

    # Step 2: Update raw_data JSON in odds_snapshots — replace "h2h" with
    # "1x2" in any bookmaker market entry that contains a Draw outcome.
    # Uses jsonb_set on each matching market object.
    op.execute("""
        UPDATE odds_snapshots
        SET raw_data = (
            SELECT jsonb_set(
                raw_data,
                '{bookmakers}',
                (
                    SELECT jsonb_agg(
                        CASE
                            WHEN (
                                SELECT bool_or(outcome->>'name' = 'Draw')
                                FROM jsonb_array_elements(bk->'markets') AS m,
                                     jsonb_array_elements(m->'outcomes') AS outcome
                            )
                            THEN jsonb_set(
                                bk,
                                '{markets}',
                                (
                                    SELECT jsonb_agg(
                                        CASE
                                            WHEN m->>'key' = 'h2h'
                                                 AND (
                                                     SELECT bool_or(o->>'name' = 'Draw')
                                                     FROM jsonb_array_elements(m->'outcomes') AS o
                                                 )
                                            THEN jsonb_set(m, '{key}', '"1x2"')
                                            ELSE m
                                        END
                                    )
                                    FROM jsonb_array_elements(bk->'markets') AS m
                                )
                            )
                            ELSE bk
                        END
                    )
                    FROM jsonb_array_elements(raw_data->'bookmakers') AS bk
                )
            )
        )
        WHERE id IN (
            SELECT s.id
            FROM odds_snapshots s,
                 jsonb_array_elements(s.raw_data->'bookmakers') AS bk,
                 jsonb_array_elements(bk->'markets') AS m,
                 jsonb_array_elements(m->'outcomes') AS outcome
            WHERE m->>'key' = 'h2h'
              AND outcome->>'name' = 'Draw'
        )
    """)


def downgrade() -> None:
    # Reverse: relabel 1x2 back to h2h in odds table
    op.execute("""
        UPDATE odds
        SET market_key = 'h2h'
        WHERE market_key = '1x2'
    """)

    # Reverse: patch raw_data JSON back
    op.execute("""
        UPDATE odds_snapshots
        SET raw_data = (
            SELECT jsonb_set(
                raw_data,
                '{bookmakers}',
                (
                    SELECT jsonb_agg(
                        jsonb_set(
                            bk,
                            '{markets}',
                            (
                                SELECT jsonb_agg(
                                    CASE
                                        WHEN m->>'key' = '1x2'
                                        THEN jsonb_set(m, '{key}', '"h2h"')
                                        ELSE m
                                    END
                                )
                                FROM jsonb_array_elements(bk->'markets') AS m
                            )
                        )
                    )
                    FROM jsonb_array_elements(raw_data->'bookmakers') AS bk
                )
            )
        )
        WHERE id IN (
            SELECT s.id
            FROM odds_snapshots s,
                 jsonb_array_elements(s.raw_data->'bookmakers') AS bk,
                 jsonb_array_elements(bk->'markets') AS m
            WHERE m->>'key' = '1x2'
        )
    """)
