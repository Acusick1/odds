"""Unit tests for feature_groups helpers."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_analytics.feature_groups import (
    _STATIC_FEATURE_GROUPS,
    EventDataBundle,
    SnapshotQuotes,
    TierSampler,
    _compute_target,
    prepare_training_data,
    raw_closes,
)
from odds_analytics.training.config import FeatureConfig
from odds_core.models import Event, EventStatus
from odds_core.odds_math import american_to_decimal, calculate_implied_probability
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis
from pydantic import ValidationError

KICKOFF = datetime(2025, 12, 1, 15, 0, tzinfo=UTC)


def _make_event() -> Event:
    return Event(
        id="evt_1",
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=KICKOFF,
        home_team="Arsenal",
        away_team="Chelsea",
        status=EventStatus.FINAL,
        home_score=2,
        away_score=1,
    )


def _snapshot(
    *,
    hours_before: float = 0.5,
    source: Source = Source.ODDS_API,
    time_basis: TimeBasis = TimeBasis.OBSERVED,
    bookmakers: tuple[str, ...] = ("bet365",),
    prices: tuple[int, int] = (-125, 260),
    snapshot_id: int = 1,
) -> SnapshotQuotes:
    """One snapshot of home and away back quotes (American prices) from each bookmaker, stamped
    ``hours_before`` kick-off (negative: after it)."""
    snapshot_time = KICKOFF - timedelta(hours=hours_before)
    return SnapshotQuotes(
        tuple(
            Quote(
                snapshot_id=snapshot_id,
                event_id="evt_1",
                snapshot_time=snapshot_time,
                created_at=snapshot_time,
                source=source,
                market=Market.H2H,
                time_basis=time_basis,
                in_play=None,
                market_status=None,
                total_matched=None,
                market_start_time=None,
                bookmaker=bookmaker,
                outcome=outcome,
                line=None,
                side=Side.BACK,
                price=american_to_decimal(price),
                size=None,
                matched=None,
                changed_at=None,
            )
            for bookmaker in bookmakers
            for outcome, price in zip((PriceOutcome.HOME, PriceOutcome.AWAY), prices, strict=True)
        )
    )


class TestSnapshotQuotes:
    def test_has_bookmaker_present(self) -> None:
        snap = _snapshot(bookmakers=("pinnacle", "bet365"))

        assert snap.has_bookmaker("pinnacle") is True

    def test_has_bookmaker_absent(self) -> None:
        snap = _snapshot(bookmakers=("bet365",))

        assert snap.has_bookmaker("pinnacle") is False

    def test_outcome_quotes_keep_only_that_outcome(self) -> None:
        snap = _snapshot(bookmakers=("pinnacle", "bet365"))

        quotes = snap.outcome_quotes(PriceOutcome.AWAY)

        assert [(q.bookmaker, q.outcome) for q in quotes] == [
            ("pinnacle", PriceOutcome.AWAY),
            ("bet365", PriceOutcome.AWAY),
        ]

    def test_snapshot_identity_comes_from_its_quotes(self) -> None:
        snap = _snapshot(hours_before=2, source=Source.ODDSPORTAL)

        assert snap.snapshot_time == KICKOFF - timedelta(hours=2)
        assert snap.source is Source.ODDSPORTAL
        assert snap.time_basis is TimeBasis.OBSERVED


class TestFeatureGroupValidation:
    """FeatureConfig.feature_groups rejects names no code dispatches on."""

    def test_unknown_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(target_type="raw", feature_groups=("tabular", "bogus"))

    @pytest.mark.parametrize("name", ["epl_lineup", "trajectory"])
    def test_removed_or_unhandled_names_rejected(self, name: str) -> None:
        with pytest.raises(ValidationError):
            FeatureConfig(target_type="raw", feature_groups=("tabular", name))

    def test_every_dispatched_group_accepted(self) -> None:
        names = tuple(sorted(_STATIC_FEATURE_GROUPS))
        assert FeatureConfig(target_type="raw", feature_groups=names).feature_groups == names


class TestComputeRawTarget:
    """The raw target is the implied-probability delta of ``FeatureConfig.outcome``."""

    @staticmethod
    def _target(outcome: str) -> float | None:
        config = FeatureConfig(
            target_type="raw",
            markets=["h2h"],
            outcome=outcome,  # type: ignore[arg-type]
        )
        snapshot = _snapshot(hours_before=6, prices=(100, 300))
        closing = _snapshot(hours_before=0.1, prices=(-125, 380))
        return _compute_target(snapshot, closing, config)

    def test_home_delta_is_the_implied_probability_change(self) -> None:
        expected = calculate_implied_probability(-125) - calculate_implied_probability(100)

        assert self._target("home") == pytest.approx(expected)

    def test_away_delta_is_the_away_side_change(self) -> None:
        expected = calculate_implied_probability(380) - calculate_implied_probability(300)

        assert self._target("away") == pytest.approx(expected)


def _history(*, snapshot_id: int, hours_before: float) -> SnapshotQuotes:
    return _snapshot(
        snapshot_id=snapshot_id,
        hours_before=hours_before,
        source=Source.ODDSPORTAL,
        time_basis=TimeBasis.SOURCE_STAMPED,
    )


def raw_close(snapshots: list[SnapshotQuotes], kickoff: datetime) -> SnapshotQuotes | None:
    """The close of single-source ``snapshots``."""
    closes = raw_closes(snapshots, kickoff)
    assert len(closes) <= 1
    return next(iter(closes.values()), None)


class TestRawClose:
    """A source's raw-target close is its latest snapshot whose derived tier is CLOSING."""

    def test_fduk_close_is_the_close_and_its_opening_never_is(self) -> None:
        opening = _snapshot(
            snapshot_id=1,
            hours_before=2,
            source=Source.FOOTBALL_DATA_UK,
            time_basis=TimeBasis.UPPER_BOUND,
        )
        close = _snapshot(
            snapshot_id=2,
            hours_before=0,
            source=Source.FOOTBALL_DATA_UK,
            time_basis=TimeBasis.CLOSE,
        )

        assert raw_close([opening, close], KICKOFF) is close
        assert raw_close([opening], KICKOFF) is opening, (
            "without FDUK's close, an opening under 3 h out covers the closing tier"
        )

    def test_oddsportal_history_close_stamped_after_kickoff_is_the_close(self) -> None:
        opening = _history(snapshot_id=1, hours_before=30)
        closing = _history(snapshot_id=2, hours_before=-1)

        assert raw_close([opening, closing], KICKOFF) is closing

    def test_oddsportal_history_close_stamped_hours_out_is_the_close(self) -> None:
        opening = _history(snapshot_id=1, hours_before=30)
        closing = _history(snapshot_id=2, hours_before=5)

        assert raw_close([opening, closing], KICKOFF) is closing

    def test_lone_oddsportal_history_stamp_hours_out_is_not_the_close(self) -> None:
        assert raw_close([_history(snapshot_id=1, hours_before=5)], KICKOFF) is None

    def test_observed_after_kickoff_is_never_the_close(self) -> None:
        before = _snapshot(snapshot_id=1, hours_before=0.5)
        after = _snapshot(snapshot_id=2, hours_before=-0.5)

        assert raw_close([before, after], KICKOFF) is before

    def test_observed_more_than_three_hours_out_is_not_the_close(self) -> None:
        assert raw_close([_snapshot(hours_before=3.5)], KICKOFF) is None

    @pytest.mark.parametrize("hours_before", [0.0, 1.0, 3.0])
    def test_observed_within_three_hours_is_the_close(self, hours_before: float) -> None:
        snap = _snapshot(hours_before=hours_before)

        assert raw_close([snap], KICKOFF) is snap

    def test_upper_bound_on_the_three_hour_boundary_is_not_the_close(self) -> None:
        on_boundary = _snapshot(hours_before=3, time_basis=TimeBasis.UPPER_BOUND)

        assert raw_close([on_boundary], KICKOFF) is None


class TestRawClosesBySource:
    """A decision snapshot is measured against its own source's close."""

    @staticmethod
    def _snapshots() -> tuple[SnapshotQuotes, SnapshotQuotes, SnapshotQuotes]:
        decision = _snapshot(
            snapshot_id=1,
            hours_before=6,
            source=Source.ODDSPORTAL,
            bookmakers=("pinnacle", "bet365"),
            prices=(-150, 130),
        )
        oddsportal_close = _snapshot(
            snapshot_id=2,
            hours_before=1,
            source=Source.ODDSPORTAL,
            bookmakers=("pinnacle", "bet365"),
            prices=(-200, 170),
        )
        betfair_close = _snapshot(
            snapshot_id=3,
            hours_before=0.2,
            source=Source.BETFAIR,
            bookmakers=("betfair_exchange",),
            prices=(-110, -110),
        )
        return decision, oddsportal_close, betfair_close

    def test_each_source_keeps_its_own_latest_close(self) -> None:
        decision, oddsportal_close, betfair_close = self._snapshots()

        closes = raw_closes([decision, oddsportal_close, betfair_close], KICKOFF)

        assert closes == {Source.ODDSPORTAL: oddsportal_close, Source.BETFAIR: betfair_close}

    @pytest.mark.asyncio
    async def test_target_ignores_a_later_close_of_another_source(self) -> None:
        decision, oddsportal_close, betfair_close = self._snapshots()
        snapshots = [decision, oddsportal_close, betfair_close]
        bundle = EventDataBundle(
            event=_make_event(),
            snapshots=snapshots,
            closes=raw_closes(snapshots, KICKOFF),
        )
        config = FeatureConfig(
            target_type="raw",
            markets=["h2h"],
            outcome="home",
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            sampling={"strategy": "tier", "decision_tier": "pregame"},
        )

        with patch(
            "odds_analytics.feature_groups.collect_event_data",
            new=AsyncMock(return_value=bundle),
        ):
            result = await prepare_training_data([_make_event()], AsyncMock(), config)

        expected = calculate_implied_probability(-200) - calculate_implied_probability(-150)
        assert result.y.tolist() == [pytest.approx(expected, abs=1e-6)]


class TestSamplerSourcesWithASharpClose:
    """Decisions come only from sources whose close has a sharp book."""

    def test_tier_sampler_skips_a_later_snapshot_of_another_source(self) -> None:
        betfair = _snapshot(
            snapshot_id=1, hours_before=8, source=Source.BETFAIR, bookmakers=("betfair_exchange",)
        )
        oddsportal = _snapshot(
            snapshot_id=2, hours_before=4, source=Source.ODDSPORTAL, bookmakers=("pinnacle",)
        )
        bundle = EventDataBundle(event=_make_event(), snapshots=[betfair, oddsportal], closes={})
        sampler = TierSampler("pregame", required_bookmakers=["pinnacle", "betfair_exchange"])

        assert sampler.sample(bundle) == [oddsportal]
        assert sampler.sample(bundle, sources={Source.BETFAIR}) == [betfair]

    @pytest.mark.asyncio
    async def test_row_comes_from_the_source_with_a_sharp_close(self) -> None:
        betfair_decision = _snapshot(
            snapshot_id=1,
            hours_before=8,
            source=Source.BETFAIR,
            bookmakers=("betfair_exchange",),
            prices=(-150, 130),
        )
        oddsportal_decision = _snapshot(
            snapshot_id=2,
            hours_before=4,
            source=Source.ODDSPORTAL,
            bookmakers=("pinnacle", "bet365"),
            prices=(-120, 100),
        )
        oddsportal_close = _snapshot(
            snapshot_id=3,
            hours_before=1,
            source=Source.ODDSPORTAL,
            bookmakers=("bet365",),
            prices=(-300, 250),
        )
        betfair_close = _snapshot(
            snapshot_id=4,
            hours_before=0.2,
            source=Source.BETFAIR,
            bookmakers=("betfair_exchange",),
            prices=(-200, 170),
        )
        snapshots = [betfair_decision, oddsportal_decision, oddsportal_close, betfair_close]
        bundle = EventDataBundle(
            event=_make_event(),
            snapshots=snapshots,
            closes=raw_closes(snapshots, KICKOFF),
        )
        config = FeatureConfig(
            target_type="raw",
            markets=["h2h"],
            outcome="home",
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            sampling={"strategy": "tier", "decision_tier": "pregame"},
        )

        with patch(
            "odds_analytics.feature_groups.collect_event_data",
            new=AsyncMock(return_value=bundle),
        ):
            result = await prepare_training_data([_make_event()], AsyncMock(), config)

        expected = calculate_implied_probability(-200) - calculate_implied_probability(-150)
        assert result.y.tolist() == [pytest.approx(expected, abs=1e-6)]


class TestTierSamplerDerivedTiers:
    """A snapshot's tier is derived from its time, source and time basis, never stored."""

    @staticmethod
    def _bundle(snapshots: list[SnapshotQuotes]) -> EventDataBundle:
        return EventDataBundle(event=_make_event(), snapshots=snapshots, closes={})

    def test_picks_the_latest_snapshot_of_the_decision_tier_or_earlier(self) -> None:
        early = _snapshot(snapshot_id=1, hours_before=48)
        pregame = _snapshot(snapshot_id=2, hours_before=6)
        closing = _snapshot(snapshot_id=3, hours_before=1)

        result = TierSampler("pregame").sample(self._bundle([early, pregame, closing]))

        assert result == [pregame]

    def test_oddsportal_history_close_after_kickoff_is_excluded(self) -> None:
        pregame = _snapshot(snapshot_id=1, hours_before=6)
        stamped_after = _snapshot(
            snapshot_id=2,
            hours_before=-1,
            source=Source.ODDSPORTAL,
            time_basis=TimeBasis.SOURCE_STAMPED,
        )

        result = TierSampler("pregame").sample(self._bundle([pregame, stamped_after]))

        assert result == [pregame]

    def test_in_play_decision_tier_takes_the_after_kickoff_snapshot(self) -> None:
        pregame = _snapshot(snapshot_id=1, hours_before=6)
        in_play = _snapshot(snapshot_id=2, hours_before=-1)

        result = TierSampler("in_play").sample(self._bundle([pregame, in_play]))

        assert result == [in_play]

    def test_fduk_opening_is_tiered_by_its_time(self) -> None:
        opening = _snapshot(
            snapshot_id=1,
            hours_before=30,
            source=Source.FOOTBALL_DATA_UK,
            time_basis=TimeBasis.UPPER_BOUND,
            bookmakers=("pinnacle",),
        )

        result = TierSampler("sharp", required_bookmakers=["pinnacle"]).sample(
            self._bundle([opening])
        )

        assert result == [opening]
