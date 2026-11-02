"""drop the old odds representations: odds, the legacy snapshot and event columns, polymarket_*

Revision ID: b8e2d5f7a1c3
Revises: c7d9e1f3a5b6
Create Date: 2026-10-28 16:30:00.000000

Prices live in ``prices`` rows, archived envelopes in ``snapshot_payloads``, gamePks in
``event_external_ids`` (source ``mlb_stats_api``) and the reason an MLB event has none in a
health-check acknowledgement (#658). This drops what they replace:

- the ``odds`` table;
- ``odds_snapshots.raw_data``, ``hours_until_commence``, ``fetch_tier``, ``bookmaker_count``
  and ``api_request_id``;
- ``events.mlb_game_pk`` and ``no_game_pk_reason``;
- the ``polymarket_*`` tables and their ``polymarketmarkettype`` enum.

The upgrade first refuses, dropping nothing, while any of them holds data found nowhere else
(``_GUARDS`` and ``_UNHELD_ODDS_PRICES``): odds rows and their prices, raw_data, gamePks,
no-gamePk reasons, legacy Polymarket moneyline prices of linked events and order books. It
drops knowingly the legacy Polymarket rows the history migration left out (games linked to no
event, spread and total markets, probabilities outside 0 to 1) and ``polymarket_fetch_logs``.
The downgrade recreates the structures empty; the dropped data does not come back.
"""

import sqlalchemy as sa
from alembic import op

revision = "b8e2d5f7a1c3"
down_revision = "c7d9e1f3a5b6"
branch_labels = None
depends_on = None

# (what the rows hold that nothing else does, a query counting them)
_GUARDS: tuple[tuple[str, str], ...] = (
    (
        "odds rows without a snapshot: their prices are in no price row",
        "SELECT count(*) FROM odds WHERE snapshot_id IS NULL",
    ),
    (
        "snapshots whose odds rows have no price rows",
        "SELECT count(*) FROM (SELECT DISTINCT snapshot_id FROM odds"
        " WHERE snapshot_id IS NOT NULL) o"
        " WHERE NOT EXISTS (SELECT 1 FROM prices p WHERE p.snapshot_id = o.snapshot_id)",
    ),
    (
        "snapshots whose raw_data no payload archives",
        "SELECT count(*) FROM odds_snapshots WHERE raw_data IS NOT NULL AND payload_id IS NULL",
    ),
    (
        "snapshots without a source or time basis",
        "SELECT count(*) FROM odds_snapshots WHERE source IS NULL OR time_basis IS NULL",
    ),
    (
        "events whose mlb_game_pk is not their mlb_stats_api external id",
        "SELECT count(*) FROM events e WHERE e.mlb_game_pk IS NOT NULL AND NOT EXISTS ("
        " SELECT 1 FROM event_external_ids x WHERE x.event_id = e.surrogate_id"
        " AND x.source = 'mlb_stats_api' AND x.external_id = e.mlb_game_pk::text)",
    ),
    (
        "events whose no_game_pk_reason has no missing_mlb_game_pk acknowledgement",
        "SELECT count(*) FROM events e WHERE e.no_game_pk_reason IS NOT NULL AND NOT EXISTS ("
        " SELECT 1 FROM health_check_acknowledgements a WHERE a.event_id = e.surrogate_id"
        " AND a.check_name = 'missing_mlb_game_pk')",
    ),
    (
        "legacy Polymarket moneyline prices of a linked event with no Polymarket snapshot at"
        " their time",
        "SELECT count(*) FROM polymarket_price_snapshots ps"
        " JOIN polymarket_markets m ON m.id = ps.polymarket_market_id"
        " JOIN polymarket_events pe ON pe.id = m.polymarket_event_id"
        " WHERE m.market_type = 'MONEYLINE' AND pe.event_id IS NOT NULL"
        " AND ps.outcome_0_price > 0 AND ps.outcome_0_price < 1"
        " AND ps.outcome_1_price > 0 AND ps.outcome_1_price < 1"
        " AND NOT EXISTS (SELECT 1 FROM odds_snapshots s WHERE s.event_id = pe.event_id"
        " AND s.source = 'polymarket' AND s.snapshot_time = ps.snapshot_time)",
    ),
    (
        "legacy Polymarket order books no payload archives",
        "SELECT count(*) FROM polymarket_orderbook_snapshots b WHERE NOT EXISTS ("
        " SELECT 1 FROM snapshot_payloads p WHERE p.payload ? 'polymarket_order_book'"
        " AND (p.payload -> 'polymarket_order_book' ->> 'id')::int = b.id)",
    ),
)

_POLYMARKET_TABLES = (
    "polymarket_orderbook_snapshots",
    "polymarket_price_snapshots",
    "polymarket_markets",
    "polymarket_events",
    "polymarket_fetch_logs",
)


# Distinct odds-row prices that no price row of their snapshot holds, counted per key so a
# missing price is caught even when another outcome of its class has the same price. A row's
# price row has the book's canonical key for the snapshot's source (``book_aliases``), side
# ``lay`` for the Odds API's ``h2h_lay`` market, ``mid`` for Polymarket and ``back`` otherwise,
# the line on spreads and totals only, and the outcome Draw, Over or Under by name; a team name
# is home or away, which the writer decided, so team rows are counted against home and away
# price rows together. Prices compare rounded to 8 decimals: an American price converted to
# decimal twice can differ in the last binary digit.
_UNHELD_ODDS_PRICES = """
    WITH odds_rows AS (
        SELECT DISTINCT snapshot_id, bookmaker_key, market_key, outcome_name, point,
               round(price_decimal::numeric, 8) AS price
        FROM odds WHERE snapshot_id IS NOT NULL
    ),
    needed AS (
        SELECT o.snapshot_id,
               coalesce(a.canonical_key, o.bookmaker_key) AS bookmaker,
               CASE
                   WHEN o.market_key = 'h2h_lay' THEN 'lay'
                   WHEN s.source = 'polymarket' THEN 'mid'
                   ELSE 'back'
               END AS side,
               CASE WHEN s.market IN ('spreads', 'totals') THEN o.point END AS line,
               o.price,
               CASE o.outcome_name
                   WHEN 'Draw' THEN 'draw'
                   WHEN 'Over' THEN 'over'
                   WHEN 'Under' THEN 'under'
                   ELSE 'team'
               END AS outcome_class,
               count(*) AS rows
        FROM odds_rows o
        JOIN odds_snapshots s ON s.id = o.snapshot_id
        LEFT JOIN book_aliases a ON a.source = s.source AND a.raw_key = o.bookmaker_key
        GROUP BY 1, 2, 3, 4, 5, 6
    ),
    held AS (
        SELECT p.snapshot_id, p.bookmaker, p.side, p.line::float8 AS line,
               round(p.price, 8) AS price,
               CASE WHEN p.outcome IN ('home', 'away') THEN 'team' ELSE p.outcome END
                   AS outcome_class,
               count(*) AS rows
        FROM prices p
        WHERE p.snapshot_id IN (SELECT snapshot_id FROM odds_rows)
        GROUP BY 1, 2, 3, 4, 5, 6
    )
    SELECT coalesce(sum(n.rows - coalesce(h.rows, 0)), 0)
    FROM needed n
    LEFT JOIN held h
      ON h.snapshot_id = n.snapshot_id
     AND h.bookmaker = n.bookmaker
     AND h.side = n.side
     AND h.line IS NOT DISTINCT FROM n.line
     AND h.price = n.price
     AND h.outcome_class = n.outcome_class
    WHERE n.rows > coalesce(h.rows, 0)
"""


def _unheld_odds_prices(bind: sa.engine.Connection) -> int:
    if bind.execute(sa.text("SELECT NOT EXISTS (SELECT 1 FROM odds)")).scalar_one():
        return 0
    # The writers' alias maps; imported only when there are odds rows to check.
    from odds_core.bookmakers import BOOKMAKER_ALIASES_BY_SOURCE

    bind.execute(
        sa.text(
            "CREATE TEMP TABLE book_aliases (source text, raw_key text, canonical_key text)"
            " ON COMMIT DROP"
        )
    )
    aliases = [
        {"source": source.value, "raw_key": raw, "canonical_key": canonical}
        for source, mapping in BOOKMAKER_ALIASES_BY_SOURCE.items()
        for raw, canonical in mapping.items()
    ]
    if aliases:
        bind.execute(
            sa.text("INSERT INTO book_aliases VALUES (:source, :raw_key, :canonical_key)"),
            aliases,
        )
    count = bind.execute(sa.text(_UNHELD_ODDS_PRICES)).scalar_one()
    bind.execute(sa.text("DROP TABLE book_aliases"))
    return count


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    bind = op.get_bind()
    held = [
        f"{count} {what}"
        for what, query in _GUARDS
        if (count := bind.execute(sa.text(query)).scalar_one())
    ]
    if unheld := _unheld_odds_prices(bind):
        held.append(f"{unheld} odds-row prices no price row of their snapshot holds")
    if held:
        raise RuntimeError(
            "refusing to drop the old odds representations while they hold data found nowhere "
            "else: " + "; ".join(held) + ". Carry that data into the new shape first."
        )

    op.drop_table("odds")

    op.drop_index("ix_event_tier", table_name="odds_snapshots")
    op.drop_index("ix_odds_snapshots_fetch_tier", table_name="odds_snapshots")
    for column in (
        "raw_data",
        "hours_until_commence",
        "fetch_tier",
        "bookmaker_count",
        "api_request_id",
    ):
        op.drop_column("odds_snapshots", column)

    op.drop_index("uq_events_mlb_game_pk", table_name="events")
    op.drop_constraint("ck_events_game_pk_or_no_game_pk_reason", "events", type_="check")
    op.drop_column("events", "mlb_game_pk")
    op.drop_column("events", "no_game_pk_reason")

    for table in _POLYMARKET_TABLES:
        op.drop_table(table)
    op.execute(sa.text("DROP TYPE polymarketmarkettype"))


def downgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    for statement in (
        "CREATE TYPE polymarketmarkettype AS ENUM"
        " ('MONEYLINE', 'SPREAD', 'TOTAL', 'PLAYER_PROP', 'OTHER')",
        """CREATE TABLE polymarket_fetch_logs (
            id SERIAL PRIMARY KEY,
            fetch_time TIMESTAMPTZ,
            job_type VARCHAR NOT NULL,
            events_count INTEGER NOT NULL,
            markets_count INTEGER NOT NULL,
            snapshots_stored INTEGER NOT NULL,
            success BOOLEAN NOT NULL,
            error_message VARCHAR,
            response_time_ms INTEGER
        )""",
        "CREATE INDEX ix_polymarket_fetch_logs_fetch_time ON polymarket_fetch_logs (fetch_time)",
        """CREATE TABLE polymarket_events (
            id SERIAL PRIMARY KEY,
            pm_event_id VARCHAR NOT NULL,
            ticker VARCHAR NOT NULL,
            slug VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            event_id VARCHAR REFERENCES events (id),
            start_date TIMESTAMPTZ,
            end_date TIMESTAMPTZ,
            active BOOLEAN NOT NULL,
            closed BOOLEAN NOT NULL,
            volume DOUBLE PRECISION,
            liquidity DOUBLE PRECISION,
            markets_count INTEGER,
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ
        )""",
        "CREATE INDEX ix_pm_event_active_closed ON polymarket_events (active, closed)",
        "CREATE INDEX ix_polymarket_events_event_id ON polymarket_events (event_id)",
        "CREATE UNIQUE INDEX ix_polymarket_events_pm_event_id ON polymarket_events (pm_event_id)",
        "CREATE INDEX ix_polymarket_events_start_date ON polymarket_events (start_date)",
        "CREATE INDEX ix_polymarket_events_ticker ON polymarket_events (ticker)",
        """CREATE TABLE polymarket_markets (
            id SERIAL PRIMARY KEY,
            polymarket_event_id INTEGER NOT NULL REFERENCES polymarket_events (id),
            pm_market_id VARCHAR NOT NULL,
            condition_id VARCHAR NOT NULL,
            question VARCHAR NOT NULL,
            clob_token_ids JSON,
            outcomes JSON,
            market_type polymarketmarkettype NOT NULL,
            group_item_title VARCHAR,
            point DOUBLE PRECISION,
            active BOOLEAN NOT NULL,
            closed BOOLEAN NOT NULL,
            accepting_orders BOOLEAN NOT NULL,
            created_at TIMESTAMPTZ
        )""",
        "CREATE INDEX ix_pm_market_event_type ON polymarket_markets"
        " (polymarket_event_id, market_type)",
        "CREATE INDEX ix_polymarket_markets_market_type ON polymarket_markets (market_type)",
        "CREATE UNIQUE INDEX ix_polymarket_markets_pm_market_id ON polymarket_markets"
        " (pm_market_id)",
        "CREATE INDEX ix_polymarket_markets_polymarket_event_id ON polymarket_markets"
        " (polymarket_event_id)",
        """CREATE TABLE polymarket_price_snapshots (
            id SERIAL PRIMARY KEY,
            polymarket_market_id INTEGER NOT NULL REFERENCES polymarket_markets (id),
            snapshot_time TIMESTAMPTZ,
            outcome_0_price DOUBLE PRECISION NOT NULL,
            outcome_1_price DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION,
            liquidity DOUBLE PRECISION,
            fetch_tier VARCHAR,
            hours_until_commence DOUBLE PRECISION,
            CONSTRAINT uq_pm_price_market_time UNIQUE (polymarket_market_id, snapshot_time)
        )""",
        "CREATE INDEX ix_pm_price_market_tier ON polymarket_price_snapshots"
        " (polymarket_market_id, fetch_tier)",
        "CREATE INDEX ix_polymarket_price_snapshots_fetch_tier ON polymarket_price_snapshots"
        " (fetch_tier)",
        "CREATE INDEX ix_polymarket_price_snapshots_polymarket_market_id"
        " ON polymarket_price_snapshots (polymarket_market_id)",
        "CREATE INDEX ix_polymarket_price_snapshots_snapshot_time ON polymarket_price_snapshots"
        " (snapshot_time)",
        """CREATE TABLE polymarket_orderbook_snapshots (
            id SERIAL PRIMARY KEY,
            polymarket_market_id INTEGER NOT NULL REFERENCES polymarket_markets (id),
            snapshot_time TIMESTAMPTZ,
            token_id VARCHAR NOT NULL,
            raw_book JSON,
            best_bid DOUBLE PRECISION,
            best_ask DOUBLE PRECISION,
            spread DOUBLE PRECISION,
            midpoint DOUBLE PRECISION,
            bid_levels INTEGER,
            ask_levels INTEGER,
            bid_depth_total DOUBLE PRECISION,
            ask_depth_total DOUBLE PRECISION,
            imbalance DOUBLE PRECISION,
            weighted_mid DOUBLE PRECISION,
            fetch_tier VARCHAR,
            hours_until_commence DOUBLE PRECISION
        )""",
        "CREATE INDEX ix_pm_orderbook_market_time ON polymarket_orderbook_snapshots"
        " (polymarket_market_id, snapshot_time)",
        "CREATE INDEX ix_polymarket_orderbook_snapshots_polymarket_market_id"
        " ON polymarket_orderbook_snapshots (polymarket_market_id)",
        "CREATE INDEX ix_polymarket_orderbook_snapshots_snapshot_time"
        " ON polymarket_orderbook_snapshots (snapshot_time)",
        "ALTER TABLE events ADD COLUMN mlb_game_pk BIGINT, ADD COLUMN no_game_pk_reason TEXT",
        "ALTER TABLE events ADD CONSTRAINT ck_events_game_pk_or_no_game_pk_reason"
        " CHECK (mlb_game_pk IS NULL OR no_game_pk_reason IS NULL)",
        "CREATE UNIQUE INDEX uq_events_mlb_game_pk ON events (mlb_game_pk)"
        " WHERE mlb_game_pk IS NOT NULL",
        # Snapshots stored meanwhile have no count to give; 0 stands in for it.
        "ALTER TABLE odds_snapshots ADD COLUMN raw_data JSON,"
        " ADD COLUMN bookmaker_count INTEGER NOT NULL DEFAULT 0,"
        " ADD COLUMN api_request_id VARCHAR, ADD COLUMN fetch_tier VARCHAR,"
        " ADD COLUMN hours_until_commence DOUBLE PRECISION",
        "ALTER TABLE odds_snapshots ALTER COLUMN bookmaker_count DROP DEFAULT",
        "CREATE INDEX ix_odds_snapshots_fetch_tier ON odds_snapshots (fetch_tier)",
        "CREATE INDEX ix_event_tier ON odds_snapshots (event_id, fetch_tier)",
        """CREATE TABLE odds (
            id SERIAL PRIMARY KEY,
            event_id VARCHAR NOT NULL REFERENCES events (id),
            bookmaker_key VARCHAR NOT NULL,
            bookmaker_title VARCHAR NOT NULL,
            market_key VARCHAR NOT NULL,
            outcome_name VARCHAR NOT NULL,
            point DOUBLE PRECISION,
            odds_timestamp TIMESTAMPTZ NOT NULL,
            last_update TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            is_valid BOOLEAN NOT NULL,
            validation_notes VARCHAR,
            price_decimal DOUBLE PRECISION NOT NULL,
            snapshot_id INTEGER REFERENCES odds_snapshots (id),
            CONSTRAINT ck_odds_price_decimal_gt_one CHECK (price_decimal > 1.0)
        )""",
        "CREATE INDEX ix_bookmaker_time ON odds (bookmaker_key, odds_timestamp)",
        "CREATE INDEX ix_event_bookmaker_market ON odds (event_id, bookmaker_key, market_key)",
        "CREATE INDEX ix_odds_bookmaker_key ON odds (bookmaker_key)",
        "CREATE INDEX ix_odds_event_id ON odds (event_id)",
        "CREATE INDEX ix_odds_market_key ON odds (market_key)",
        "CREATE INDEX ix_odds_odds_timestamp ON odds (odds_timestamp)",
        "CREATE INDEX ix_odds_snapshot_id ON odds (snapshot_id)",
    ):
        op.execute(sa.text(statement))
