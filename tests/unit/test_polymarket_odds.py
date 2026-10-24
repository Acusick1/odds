"""Unit tests for Polymarket match markets as odds snapshots."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from odds_core.models import OddsSnapshot
from odds_core.odds_math import devig_probabilities
from odds_core.snapshot_utils import extract_odds_from_snapshot
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.polymarket_fetcher import MAX_HISTORY_WINDOW, PolymarketClient
from odds_lambda.polymarket_odds import (
    MAX_HORIZON,
    MAX_STALENESS,
    TIER_STEPS,
    Observation,
    Prices,
    Side,
    build_raw_data,
    canonical_team,
    grid_offsets,
    pair_events,
    parse_match_event,
    resample,
)
from odds_lambda.polymarket_sports import MATCH_ODDS_SPORTS, POLYMARKET_SPORTS
from odds_lambda.tier_utils import calculate_tier

from tests.test_helpers import make_epl_event

EPL = POLYMARKET_SPORTS["soccer_epl"]
KICKOFF = datetime(2025, 12, 6, 15, 0, tzinfo=UTC)


def flat_series(prices: Prices, start: datetime, end: datetime) -> dict[Side, list[Observation]]:
    minutes = int((end - start).total_seconds() // 60)
    return {
        side: [(start + timedelta(minutes=i), p) for i in range(minutes + 1)]
        for side, p in prices.by_side().items()
    }


class TestParseMatchEvent:
    def test_classifies_markets_by_slug_suffix(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)

        assert match is not None
        assert (match.home_team, match.away_team) == ("Bournemouth", "Chelsea")
        assert match.start_time == KICKOFF
        assert match.home.yes_token_id == "yes-bou"
        assert match.draw.yes_token_id == "yes-draw"
        assert match.away.yes_token_id == "yes-che"

    def test_midpoints_from_best_bid_and_ask(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)

        assert match is not None
        prices = match.midpoints()
        assert prices is not None
        assert prices.by_side() == pytest.approx({"home": 0.53, "draw": 0.25, "away": 0.21})

    def test_max_spread_is_the_widest_outcome(self, gamma_event) -> None:
        match = parse_match_event(
            gamma_event(quotes={"che": (0.2, 0.22), "draw": (0.2, 0.26), "bou": (0.5, 0.52)}), EPL
        )

        assert match is not None
        assert match.max_spread() == pytest.approx(0.06)

    @pytest.mark.parametrize("bid_ask", [(0.0, 0.0), (1.0, 1.0)])
    def test_midpoints_none_when_a_midpoint_is_not_a_probability(
        self, gamma_event, bid_ask: tuple[float, float]
    ) -> None:
        match = parse_match_event(
            gamma_event(quotes={"che": (0.2, 0.22), "draw": (0.2, 0.26), "bou": bid_ask}), EPL
        )

        assert match is not None
        assert match.midpoints() is None

    def test_midpoints_none_when_a_side_has_no_book(self, gamma_event) -> None:
        event = gamma_event()
        event["markets"][0]["bestBid"] = None
        match = parse_match_event(event, EPL)

        assert match is not None
        assert match.midpoints() is None

    @pytest.mark.parametrize(
        "slug",
        [
            "epl-bou-che-2025-12-06-total-corners",
            "epl-bou-che-2025-12-06-exact-score",
            "nba-dal-mil-2025-12-06",
        ],
    )
    def test_ignores_prop_and_foreign_events(self, gamma_event, slug: str) -> None:
        assert parse_match_event(gamma_event(slug=slug), EPL) is None

    def test_rejects_event_with_unknown_market(self, gamma_event) -> None:
        event = gamma_event()
        event["markets"][1]["slug"] = "epl-bou-che-2025-12-06-other"

        assert parse_match_event(event, EPL) is None

    def test_rejects_event_missing_a_market(self, gamma_event) -> None:
        event = gamma_event()
        event["markets"].pop()

        assert parse_match_event(event, EPL) is None


class TestCanonicalTeam:
    @pytest.mark.parametrize(
        ("polymarket_name", "canonical"),
        [
            ("Nottingham Forest FC", "Nottingham"),
            ("Brighton & Hove Albion FC", "Brighton"),
            ("AFC Bournemouth", "Bournemouth"),
            ("Manchester United FC", "Manchester Utd"),
            ("Tottenham Hotspur FC", "Tottenham"),
            ("Wolverhampton Wanderers FC", "Wolves"),
            ("Arsenal FC", "Arsenal"),
        ],
    )
    def test_matches_events_table_names(self, polymarket_name: str, canonical: str) -> None:
        assert canonical_team(polymarket_name) == canonical


class TestGrid:
    def test_offsets_ascend_from_kickoff_to_max_horizon(self) -> None:
        offsets = grid_offsets()

        assert offsets[0] == timedelta(0)
        assert offsets[-1] == MAX_HORIZON
        assert offsets == sorted(set(offsets))

    @pytest.mark.parametrize("hours", [1, 3, 6, 12, 24, 72, 168])
    def test_standard_horizons_are_grid_points(self, hours: int) -> None:
        assert timedelta(hours=hours) in grid_offsets()

    def test_spacing_is_the_tier_step_of_the_later_point(self) -> None:
        offsets = grid_offsets()

        for earlier, later in zip(offsets, offsets[1:], strict=False):
            tier = calculate_tier(later.total_seconds() / 3600)
            assert later - earlier == TIER_STEPS[tier]

    def test_tier_steps_cover_every_pre_game_tier(self) -> None:
        assert set(TIER_STEPS) == set(FetchTier) - {FetchTier.IN_PLAY}


class TestResample:
    def test_flat_series_fills_whole_grid(self) -> None:
        prices = Prices(0.5, 0.25, 0.25)
        start = KICKOFF - MAX_HORIZON - MAX_STALENESS
        result = resample(flat_series(prices, start, KICKOFF), KICKOFF, KICKOFF)

        assert len(result.points) == len(grid_offsets())
        assert result.missing == 0
        assert result.off_sum == 0
        assert result.points[-1].time == KICKOFF
        assert all(point.prices == prices for point in result.points)

    def test_uses_last_observation_at_or_before_each_grid_time(self) -> None:
        early = KICKOFF - timedelta(hours=2)
        series: dict[Side, list[Observation]] = {
            "home": [(early, 0.5), (early + timedelta(minutes=1), 0.6)],
            "draw": [(early, 0.3), (early + timedelta(minutes=1), 0.25)],
            "away": [(early, 0.2), (early + timedelta(minutes=1), 0.15)],
        }

        result = resample(series, KICKOFF, early + timedelta(minutes=10))

        assert result.points[0].prices == Prices(0.5, 0.3, 0.2)
        assert result.points[-1].time == early + timedelta(minutes=10)
        assert result.points[-1].prices == Prices(0.6, 0.25, 0.15)

    def test_never_looks_past_the_cutoff(self) -> None:
        start = KICKOFF - timedelta(hours=1)
        series = flat_series(Prices(0.5, 0.25, 0.25), start, KICKOFF)
        cutoff = KICKOFF - timedelta(minutes=20)

        result = resample(series, KICKOFF, cutoff)

        assert result.points
        assert max(point.time for point in result.points) <= cutoff

    def test_stale_side_drops_the_grid_point(self) -> None:
        start = KICKOFF - timedelta(minutes=10)
        series = flat_series(Prices(0.5, 0.25, 0.25), start, KICKOFF)
        series["draw"] = [(start - 2 * MAX_STALENESS, 0.25)]

        result = resample(series, KICKOFF, KICKOFF)

        assert result.points == []
        assert result.missing > 0

    def test_prices_outside_open_interval_are_missing(self) -> None:
        start = KICKOFF - timedelta(minutes=10)
        series = flat_series(Prices(1.0, 0.0, 0.0), start, KICKOFF)

        result = resample(series, KICKOFF, KICKOFF)

        assert result.points == []
        assert result.missing > 0

    def test_prices_off_sum_are_counted_and_dropped(self) -> None:
        start = KICKOFF - timedelta(minutes=10)
        series = flat_series(Prices(0.5, 0.4, 0.4), start, KICKOFF)

        result = resample(series, KICKOFF, KICKOFF)

        assert result.points == []
        assert result.off_sum > 0

    def test_sum_just_inside_tolerance_is_kept(self) -> None:
        start = KICKOFF - timedelta(minutes=10)
        series = flat_series(Prices(0.51, 0.26, 0.25), start, KICKOFF)

        result = resample(series, KICKOFF, KICKOFF)

        assert result.points[-1].time == KICKOFF
        assert result.off_sum == 0


class TestPairEvents:
    def test_pairs_by_teams_and_kickoff(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        event = make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea")

        pairing = pair_events([event], [match])

        assert pairing.pairs == [(event, match)]
        assert pairing.unmatched_matches == []

    def test_twins_pair_only_the_api_event(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        events = [
            make_epl_event("op_2025-2026_BOU_CHE_2025-12-06", KICKOFF, "Bournemouth", "Chelsea"),
            make_epl_event("fduk_1", KICKOFF, "Bournemouth", "Chelsea"),
            make_epl_event("hex1", KICKOFF, "Bournemouth", "Chelsea"),
        ]

        pairing = pair_events(events, [match])

        assert [(e.id, m) for e, m in pairing.pairs] == [("hex1", match)]
        assert [(m, sorted(e.id for e in evs)) for m, evs in pairing.twins] == [
            (match, ["fduk_1", "hex1", "op_2025-2026_BOU_CHE_2025-12-06"])
        ]

    def test_twins_without_an_api_event_pair_the_lowest_id(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        events = [
            make_epl_event("op_b", KICKOFF, "Bournemouth", "Chelsea"),
            make_epl_event("op_a", KICKOFF, "Bournemouth", "Chelsea"),
        ]

        assert [e.id for e, _ in pair_events(events, [match]).pairs] == ["op_a"]

    def test_reversed_fixture_does_not_pair(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        event = make_epl_event("e1", KICKOFF, "Chelsea", "Bournemouth")

        pairing = pair_events([event], [match])

        assert pairing.pairs == []
        assert pairing.unmatched_events == [event]
        assert pairing.unmatched_matches == [match]

    def test_kickoff_beyond_window_does_not_pair(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        event = make_epl_event("e1", KICKOFF + timedelta(hours=5), "Bournemouth", "Chelsea")

        assert pair_events([event], [match]).unmatched_events == [event]

    def test_two_matches_in_window_are_ambiguous(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        duplicate = parse_match_event(gamma_event(), EPL)
        assert duplicate is not None
        event = make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea")

        assert pair_events([event], [match, duplicate]).ambiguous_events == [event]


class TestBuildRawData:
    def test_pipeline_extracts_devigged_polymarket_probs(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        event = make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea")
        prices = Prices(0.53, 0.25, 0.22)
        raw_data = build_raw_data(match, EPL, event, prices, KICKOFF - timedelta(hours=1))
        snapshot = OddsSnapshot(
            event_id="e1",
            snapshot_time=KICKOFF - timedelta(hours=1),
            raw_data=raw_data,
            bookmaker_count=1,
        )

        odds = extract_odds_from_snapshot(snapshot, "e1", market="1x2")
        implied = {o.outcome_name: o.implied_probability for o in odds}
        probs = devig_probabilities(implied["Bournemouth"], implied["Draw"], implied["Chelsea"])

        assert raw_data["source"] == "polymarket"
        assert probs == pytest.approx((0.53, 0.25, 0.22), abs=2e-3)

    def test_outcomes_carry_the_reciprocal_decimal_price(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        event = make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea")

        outcomes = build_raw_data(match, EPL, event, Prices(0.53, 0.25, 0.22), KICKOFF)[
            "bookmakers"
        ][0]["markets"][0]["outcomes"]

        assert all("price" not in o for o in outcomes)
        assert sorted(o["price_decimal"] for o in outcomes) == pytest.approx(
            sorted([1 / 0.53, 1 / 0.25, 1 / 0.22])
        )

    def test_meta_keeps_exact_probabilities(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        event = make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea")

        meta = build_raw_data(match, EPL, event, Prices(0.53, 0.25, 0.22), KICKOFF)["bookmakers"][
            0
        ]["polymarket_meta"]

        assert meta["yes_probabilities"] == {"home": 0.53, "draw": 0.25, "away": 0.22}
        assert meta["pm_event_id"] == "900"
        assert "best_bid" not in meta

    def test_live_snapshot_keeps_quotes(self, gamma_event) -> None:
        match = parse_match_event(gamma_event(), EPL)
        assert match is not None
        event = make_epl_event("e1", KICKOFF, "Bournemouth", "Chelsea")

        meta = build_raw_data(
            match, EPL, event, Prices(0.53, 0.25, 0.22), KICKOFF, include_quotes=True
        )["bookmakers"][0]["polymarket_meta"]

        assert meta["best_bid"] == {"home": 0.52, "draw": 0.24, "away": 0.20}
        assert meta["best_ask"] == {"home": 0.54, "draw": 0.26, "away": 0.22}


class TestSportConfig:
    def test_epl_has_three_way_markets_and_no_nba_ids(self) -> None:
        assert MATCH_ODDS_SPORTS == ("soccer_epl",)
        assert (EPL.series_id, EPL.tag_id, EPL.market_key) == ("10188", None, "1x2")

    def test_match_slug_per_sport(self) -> None:
        nba = POLYMARKET_SPORTS["basketball_nba"]

        assert nba.match_slug.match("nba-dal-mil-2026-01-25")
        assert not nba.match_slug.match("epl-bou-che-2025-12-06")
        assert EPL.match_slug.match("epl-bou-che-2025-12-06")


class TestPriceSeries:
    async def test_splits_range_into_windows_within_the_cap(self) -> None:
        client = PolymarketClient()
        calls: list[tuple[int, int]] = []

        async def history(token_id: str, **kwargs: Any) -> list[dict[str, float]]:
            calls.append((kwargs["start_ts"], kwargs["end_ts"]))
            return [{"t": kwargs["start_ts"], "p": 0.5}, {"t": kwargs["end_ts"], "p": 0.5}]

        client.get_price_history = AsyncMock(side_effect=history)  # type: ignore[method-assign]
        start = datetime(2025, 8, 1, tzinfo=UTC)
        end = start + timedelta(days=30)

        series = await client.get_price_series("tok", start, end)

        assert len(calls) == 3
        assert all(
            window_end - window_start <= MAX_HISTORY_WINDOW.total_seconds()
            for window_start, window_end in calls
        )
        assert calls[0][0] == int(start.timestamp())
        assert calls[-1][1] == int(end.timestamp())
        times = [t for t, _ in series]
        assert times == sorted(set(times))
        assert client.get_price_history.call_args.kwargs["interval"] is None

    async def test_events_request_omits_tag_and_active_when_unset(self) -> None:
        client = PolymarketClient()
        client._make_request = AsyncMock(return_value=([], 1))  # type: ignore[method-assign]

        await client.get_events("soccer_epl", active=None, closed=True)

        params = client._make_request.call_args.kwargs["params"]
        assert params == {"series_id": "10188", "closed": "true", "limit": 100, "offset": 0}

    async def test_events_request_sends_nba_tag(self) -> None:
        client = PolymarketClient()
        client._make_request = AsyncMock(return_value=([], 1))  # type: ignore[method-assign]

        await client.get_events("basketball_nba")

        params = client._make_request.call_args.kwargs["params"]
        assert (params["series_id"], params["tag_id"], params["active"]) == (
            "10345",
            "100639",
            "true",
        )

    async def test_get_all_events_drops_repeats_across_pages(self) -> None:
        client = PolymarketClient()
        client._make_request = AsyncMock(  # type: ignore[method-assign]
            side_effect=[([{"id": "1"}, {"id": "2"}], 1), ([{"id": "2"}, {"id": "3"}], 1), ([], 1)]
        )

        events = await client.get_all_events("soccer_epl")

        assert [e["id"] for e in events] == ["1", "2", "3"]
        offsets = [c.kwargs["params"]["offset"] for c in client._make_request.call_args_list]
        assert offsets == [0, 2, 4]

    async def test_get_all_events_pages_until_empty(self) -> None:
        client = PolymarketClient()
        client._make_request = AsyncMock(  # type: ignore[method-assign]
            side_effect=[([{"id": "1"}, {"id": "2"}], 1), ([{"id": "3"}], 1), ([], 1)]
        )

        events = await client.get_all_events("soccer_epl")

        assert [e["id"] for e in events] == ["1", "2", "3"]
        offsets = [c.kwargs["params"]["offset"] for c in client._make_request.call_args_list]
        assert offsets == [0, 2, 3]
