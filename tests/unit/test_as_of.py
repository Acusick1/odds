"""Unit tests for the as-of selection of stored quotes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from odds_core.as_of import by_snapshot, latest_snapshot
from odds_core.quotes import Quote

from tests.test_helpers import make_epl_event, make_quotes, three_way_bookmaker

T = datetime(2026, 11, 7, 14, 0, tzinfo=UTC)
EVENT = make_epl_event("evt", T + timedelta(hours=1))


def _snap(snapshot_id: int, minutes_before_t: float, *books: str) -> list[Quote]:
    return make_quotes(
        EVENT,
        snapshot_id,
        T - timedelta(minutes=minutes_before_t),
        [three_way_bookmaker(book, (2.0, 3.0, 4.0)) for book in books],
        "oddsportal_live",
    )


class TestBySnapshot:
    def test_groups_in_time_then_id_order(self) -> None:
        late, tied, early = _snap(3, 1, "bet365"), _snap(2, 1, "betway"), _snap(9, 5, "coral")
        groups = by_snapshot(late + early + tied)
        assert [g[0].snapshot_id for g in groups] == [9, 2, 3]


class TestLatestSnapshot:
    def test_latest_at_or_before_wins_whole(self) -> None:
        old, new = _snap(1, 5, "bet365", "betway"), _snap(2, 1, "coral")
        chosen = latest_snapshot(old + new, T)
        assert chosen is not None and {q.bookmaker for q in chosen} == {"coral"}

    def test_later_snapshots_are_ignored(self) -> None:
        assert latest_snapshot(_snap(1, -1, "bet365"), T) is None

    def test_a_snapshot_accept_rejects_is_skipped(self) -> None:
        old, new = _snap(1, 5, "bet365"), _snap(2, 1, "coral")
        chosen = latest_snapshot(
            old + new, T, lambda group: any(q.bookmaker == "bet365" for q in group)
        )
        assert chosen is not None and chosen[0].snapshot_id == 1

    def test_a_tie_in_time_goes_to_the_higher_id(self) -> None:
        chosen = latest_snapshot(_snap(7, 1, "bet365") + _snap(4, 1, "coral"), T)
        assert chosen is not None and chosen[0].snapshot_id == 7
