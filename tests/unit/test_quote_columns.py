"""The candidate rules' own quote query reads the same columns as odds_core.quotes."""

from __future__ import annotations

from odds_core.quotes import QUOTE_COLUMNS
from odds_lambda.candidate_rules import inputs
from sqlalchemy import select
from sqlalchemy.dialects import postgresql


class TestQuoteColumns:
    def test_rule_query_columns_equal_the_quote_columns(self) -> None:
        dialect = postgresql.dialect()
        assert str(select(*inputs._COLUMNS).compile(dialect=dialect)) == str(
            select(*QUOTE_COLUMNS).compile(dialect=dialect)
        )
        assert [c.name for c in inputs._COLUMNS] == [c.name for c in QUOTE_COLUMNS]
