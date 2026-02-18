"""add_polymarket_tables

Revision ID: 7a9cf0e00ca5
Revises: a1b2c3d4e5f6
Create Date: 2026-02-09 10:13:16.040121

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "7a9cf0e00ca5"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "polymarket_fetch_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fetch_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("job_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("events_count", sa.Integer(), nullable=False),
        sa.Column("markets_count", sa.Integer(), nullable=False),
        sa.Column("snapshots_stored", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("response_time_ms", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_polymarket_fetch_logs_fetch_time"),
        "polymarket_fetch_logs",
        ["fetch_time"],
        unique=False,
    )
    op.create_table(
        "polymarket_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pm_event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("ticker", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("slug", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("title", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("closed", sa.Boolean(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("liquidity", sa.Float(), nullable=True),
        sa.Column("markets_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["events.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pm_event_active_closed", "polymarket_events", ["active", "closed"], unique=False
    )
    op.create_index(
        op.f("ix_polymarket_events_event_id"), "polymarket_events", ["event_id"], unique=False
    )
    op.create_index(
        op.f("ix_polymarket_events_pm_event_id"), "polymarket_events", ["pm_event_id"], unique=True
    )
    op.create_index(
        op.f("ix_polymarket_events_start_date"), "polymarket_events", ["start_date"], unique=False
    )
    op.create_index(
        op.f("ix_polymarket_events_ticker"), "polymarket_events", ["ticker"], unique=False
    )
    op.create_table(
        "polymarket_markets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("polymarket_event_id", sa.Integer(), nullable=False),
        sa.Column("pm_market_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("condition_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("question", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("clob_token_ids", sa.JSON(), nullable=True),
        sa.Column("outcomes", sa.JSON(), nullable=True),
        sa.Column(
            "market_type",
            sa.Enum(
                "MONEYLINE", "SPREAD", "TOTAL", "PLAYER_PROP", "OTHER", name="polymarketmarkettype"
            ),
            nullable=False,
        ),
        sa.Column("group_item_title", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("point", sa.Float(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("closed", sa.Boolean(), nullable=False),
        sa.Column("accepting_orders", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["polymarket_event_id"],
            ["polymarket_events.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pm_market_event_type",
        "polymarket_markets",
        ["polymarket_event_id", "market_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_polymarket_markets_market_type"),
        "polymarket_markets",
        ["market_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_polymarket_markets_pm_market_id"),
        "polymarket_markets",
        ["pm_market_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_polymarket_markets_polymarket_event_id"),
        "polymarket_markets",
        ["polymarket_event_id"],
        unique=False,
    )
    op.create_table(
        "polymarket_orderbook_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("polymarket_market_id", sa.Integer(), nullable=False),
        sa.Column("snapshot_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("token_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("raw_book", sa.JSON(), nullable=True),
        sa.Column("best_bid", sa.Float(), nullable=True),
        sa.Column("best_ask", sa.Float(), nullable=True),
        sa.Column("spread", sa.Float(), nullable=True),
        sa.Column("midpoint", sa.Float(), nullable=True),
        sa.Column("bid_levels", sa.Integer(), nullable=True),
        sa.Column("ask_levels", sa.Integer(), nullable=True),
        sa.Column("bid_depth_total", sa.Float(), nullable=True),
        sa.Column("ask_depth_total", sa.Float(), nullable=True),
        sa.Column("imbalance", sa.Float(), nullable=True),
        sa.Column("weighted_mid", sa.Float(), nullable=True),
        sa.Column("fetch_tier", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("hours_until_commence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["polymarket_market_id"],
            ["polymarket_markets.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pm_orderbook_market_time",
        "polymarket_orderbook_snapshots",
        ["polymarket_market_id", "snapshot_time"],
        unique=False,
    )
    op.create_index(
        op.f("ix_polymarket_orderbook_snapshots_polymarket_market_id"),
        "polymarket_orderbook_snapshots",
        ["polymarket_market_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_polymarket_orderbook_snapshots_snapshot_time"),
        "polymarket_orderbook_snapshots",
        ["snapshot_time"],
        unique=False,
    )
    op.create_table(
        "polymarket_price_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("polymarket_market_id", sa.Integer(), nullable=False),
        sa.Column("snapshot_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("outcome_0_price", sa.Float(), nullable=False),
        sa.Column("outcome_1_price", sa.Float(), nullable=False),
        sa.Column("best_bid", sa.Float(), nullable=True),
        sa.Column("best_ask", sa.Float(), nullable=True),
        sa.Column("spread", sa.Float(), nullable=True),
        sa.Column("midpoint", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("liquidity", sa.Float(), nullable=True),
        sa.Column("fetch_tier", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("hours_until_commence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["polymarket_market_id"],
            ["polymarket_markets.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pm_price_market_tier",
        "polymarket_price_snapshots",
        ["polymarket_market_id", "fetch_tier"],
        unique=False,
    )
    op.create_index(
        "ix_pm_price_market_time",
        "polymarket_price_snapshots",
        ["polymarket_market_id", "snapshot_time"],
        unique=False,
    )
    op.create_index(
        op.f("ix_polymarket_price_snapshots_fetch_tier"),
        "polymarket_price_snapshots",
        ["fetch_tier"],
        unique=False,
    )
    op.create_index(
        op.f("ix_polymarket_price_snapshots_polymarket_market_id"),
        "polymarket_price_snapshots",
        ["polymarket_market_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_polymarket_price_snapshots_snapshot_time"),
        "polymarket_price_snapshots",
        ["snapshot_time"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_polymarket_price_snapshots_snapshot_time"), table_name="polymarket_price_snapshots"
    )
    op.drop_index(
        op.f("ix_polymarket_price_snapshots_polymarket_market_id"),
        table_name="polymarket_price_snapshots",
    )
    op.drop_index(
        op.f("ix_polymarket_price_snapshots_fetch_tier"), table_name="polymarket_price_snapshots"
    )
    op.drop_index("ix_pm_price_market_time", table_name="polymarket_price_snapshots")
    op.drop_index("ix_pm_price_market_tier", table_name="polymarket_price_snapshots")
    op.drop_table("polymarket_price_snapshots")
    op.drop_index(
        op.f("ix_polymarket_orderbook_snapshots_snapshot_time"),
        table_name="polymarket_orderbook_snapshots",
    )
    op.drop_index(
        op.f("ix_polymarket_orderbook_snapshots_polymarket_market_id"),
        table_name="polymarket_orderbook_snapshots",
    )
    op.drop_index("ix_pm_orderbook_market_time", table_name="polymarket_orderbook_snapshots")
    op.drop_table("polymarket_orderbook_snapshots")
    op.drop_index(
        op.f("ix_polymarket_markets_polymarket_event_id"), table_name="polymarket_markets"
    )
    op.drop_index(op.f("ix_polymarket_markets_pm_market_id"), table_name="polymarket_markets")
    op.drop_index(op.f("ix_polymarket_markets_market_type"), table_name="polymarket_markets")
    op.drop_index("ix_pm_market_event_type", table_name="polymarket_markets")
    op.drop_table("polymarket_markets")
    op.drop_index(op.f("ix_polymarket_events_ticker"), table_name="polymarket_events")
    op.drop_index(op.f("ix_polymarket_events_start_date"), table_name="polymarket_events")
    op.drop_index(op.f("ix_polymarket_events_pm_event_id"), table_name="polymarket_events")
    op.drop_index(op.f("ix_polymarket_events_event_id"), table_name="polymarket_events")
    op.drop_index("ix_pm_event_active_closed", table_name="polymarket_events")
    op.drop_table("polymarket_events")
    op.drop_index(op.f("ix_polymarket_fetch_logs_fetch_time"), table_name="polymarket_fetch_logs")
    op.drop_table("polymarket_fetch_logs")
    sa.Enum(
        "MONEYLINE", "SPREAD", "TOTAL", "PLAYER_PROP", "OTHER", name="polymarketmarkettype"
    ).drop(op.get_bind())
