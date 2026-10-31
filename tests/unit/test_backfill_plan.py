"""Unit tests for the Odds API historical backfill planner."""

import random
from datetime import UTC, datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
from odds_analytics.backfill_plan import (
    Fixture,
    Manifest,
    Season,
    Window,
    load_fixtures,
    plan,
    targets,
)
from odds_core.models import Event, EventStatus
from pydantic import ValidationError

from tests.test_helpers import add_event

REPO = Path(__file__).resolve().parents[2]
FIVE_MINUTES_FROM = datetime(2022, 9, 1, tzinfo=UTC)


def manifest_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "sport": "soccer_epl",
        "seasons": {"2022-23": {"start": "2022-07-01T00:00:00Z", "end": "2023-07-01T00:00:00Z"}},
        "markets": [{"key": "h2h"}, {"key": "totals", "line": 2.5}],
        "bookmakers": [f"book{i}" for i in range(10)],
        "windows": [
            {"minutes_before_kickoff": [120, 60, 45, 30, 15, 10, 5], "max_input_age_minutes": 10},
            {"minutes_before_kickoff": [180, 1440, 10080], "max_input_age_minutes": 60},
        ],
        "snapshot_grid": [
            {"start": "2020-06-06T00:00:00Z", "step_minutes": 10},
            {"start": "2022-09-01T00:00:00Z", "step_minutes": 5},
        ],
        "probe": {
            "snapshots": ["2022-09-17T10:00:00Z"],
            "min_fixture_lead_hours": 169,
            "required_bookmakers": {"h2h": ["book0"]},
            "units_per_call": 20,
        },
    }
    return data | overrides


def with_units(data: dict[str, Any], units: int) -> dict[str, Any]:
    return data | {"probe": data["probe"] | {"units_per_call": units}}


@pytest.fixture
def manifest() -> Manifest:
    return Manifest.model_validate(manifest_data())


def random_fixtures(seed: int, n: int) -> list[Fixture]:
    """Kick-offs on any minute either side of the grid change, many sharing a slot."""
    rng = random.Random(seed)
    base = FIVE_MINUTES_FROM - timedelta(days=20)
    slots = [base + timedelta(minutes=rng.randrange(40 * 24 * 60)) for _ in range(n // 3)]
    return [Fixture(i, rng.choice(slots)) for i in range(n)]


class TestPlan:
    @pytest.mark.parametrize("seed", range(5))
    def test_every_call_lies_inside_each_window_it_serves(
        self, manifest: Manifest, seed: int
    ) -> None:
        calls = plan(manifest, random_fixtures(seed, 120))

        for call in calls:
            for target in call.serves:
                assert target.start <= call.time <= target.end, (call.time, target)

    @pytest.mark.parametrize("seed", range(5))
    def test_every_window_is_served_once(self, manifest: Manifest, seed: int) -> None:
        fixtures = random_fixtures(seed, 120)

        served = [target for call in plan(manifest, fixtures) for target in call.serves]

        assert sorted(served, key=repr) == sorted(targets(manifest, fixtures), key=repr)

    @pytest.mark.parametrize("seed", range(20))
    def test_no_fewer_grid_calls_serve_every_window(self, seed: int) -> None:
        small = Manifest.model_validate(
            manifest_data(
                windows=[
                    {"minutes_before_kickoff": [30, 15, 5], "max_input_age_minutes": 10},
                    {"minutes_before_kickoff": [60], "max_input_age_minutes": 20},
                ]
            )
        )
        rng = random.Random(seed)
        base = FIVE_MINUTES_FROM - timedelta(minutes=45)
        fixtures = [Fixture(i, base + timedelta(minutes=rng.randrange(90))) for i in range(3)]
        windows = targets(small, fixtures)
        lo, hi = min(t.start for t in windows), max(t.end for t in windows)
        points = sorted(
            {
                small.floor_to_grid(lo + timedelta(minutes=m))
                for m in range(int((hi - lo).total_seconds() // 60) + 1)
            }
        )
        points = [p for p in points if any(t.contains(p) for t in windows)]

        fewest = next(
            k
            for k in range(1, len(windows) + 1)
            if any(
                all(any(t.contains(p) for p in chosen) for t in windows)
                for chosen in combinations(points, k)
            )
        )

        assert len(plan(small, fixtures)) == fewest

    def test_calls_fall_on_the_snapshot_grid(self, manifest: Manifest) -> None:
        for call in plan(manifest, random_fixtures(0, 120)):
            step = 5 if call.time >= FIVE_MINUTES_FROM else 10
            assert call.time.second == call.time.microsecond == 0
            assert call.time.minute % step == 0

    def test_fixtures_sharing_a_kickoff_share_calls(self, manifest: Manifest) -> None:
        kickoff = datetime(2022, 10, 1, 14, 0, tzinfo=UTC)

        calls = plan(manifest, [Fixture(1, kickoff), Fixture(2, kickoff)])

        assert len(calls) == len(plan(manifest, [Fixture(1, kickoff)]))
        assert all({t.fixture.event_id for t in call.serves} == {1, 2} for call in calls)

    def test_a_call_serves_overlapping_windows_of_different_kickoffs(
        self, manifest: Manifest
    ) -> None:
        early = Fixture(1, datetime(2022, 10, 1, 14, 0, tzinfo=UTC))
        late = Fixture(2, datetime(2022, 10, 1, 14, 30, tzinfo=UTC))

        calls = plan(manifest, [early, late])

        # 14:00 KO-30 and 14:30 KO-60 both end at 13:30.
        shared = next(c for c in calls if c.time == datetime(2022, 10, 1, 13, 30, tzinfo=UTC))
        assert {(t.fixture.event_id, t.window.lead) for t in shared.serves} >= {
            (1, timedelta(minutes=30)),
            (2, timedelta(minutes=60)),
        }

    def test_a_window_holding_no_snapshot_raises(self) -> None:
        narrow = Manifest.model_validate(
            manifest_data(windows=[{"minutes_before_kickoff": [5], "max_input_age_minutes": 3}])
        )

        with pytest.raises(ValueError, match="no snapshot"):
            plan(narrow, [Fixture(1, datetime(2021, 10, 1, 19, 44, tzinfo=UTC))])

    def test_a_window_before_the_first_snapshot_raises(self, manifest: Manifest) -> None:
        with pytest.raises(ValueError, match="before the first snapshot"):
            plan(manifest, [Fixture(1, datetime(2020, 6, 6, 1, 0, tzinfo=UTC))])


class TestManifest:
    def test_the_committed_manifest_plans(self) -> None:
        committed = Manifest.load(REPO / "backfills" / "odds-api-epl-2020-26.yaml")

        assert committed.units_per_call == 20
        assert plan(committed, [Fixture(1, datetime(2024, 10, 5, 14, 0, tzinfo=UTC))])

    @pytest.mark.parametrize(
        ("books", "regions", "units"),
        [(10, (), 20), (11, (), 40), (0, ("uk",), 20), (0, ("uk", "eu"), 40)],
    )
    def test_units_per_call(self, books: int, regions: tuple[str, ...], units: int) -> None:
        data = manifest_data(bookmakers=[f"book{i}" for i in range(books)], regions=list(regions))
        data["probe"]["required_bookmakers"] = {}

        assert Manifest.model_validate(with_units(data, units)).units_per_call == units

    @pytest.mark.parametrize(("books", "regions"), [([], []), (["pinnacle"], ["uk"])])
    def test_names_either_bookmakers_or_regions(self, books: list[str], regions: list[str]) -> None:
        with pytest.raises(ValidationError, match="either bookmakers or regions|probe requires"):
            Manifest.model_validate(manifest_data(bookmakers=books, regions=regions))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"markets": [{"key": "h2h"}, {"key": "h2h"}]}, "requested once"),
            ({"markets": [{"key": "1x2"}]}, "markets"),
            ({"regions": ["mars"], "bookmakers": []}, "regions"),
            (
                {
                    "seasons": {
                        "a": {"start": "2023-01-01T00:00:00Z", "end": "2022-01-01T00:00:00Z"}
                    }
                },
                "at or after its end",
            ),
            (
                {
                    "seasons": {
                        "a": {"start": "2022-01-01T00:00:00Z", "end": "2023-01-01T00:00:00Z"},
                        "b": {"start": "2022-06-01T00:00:00Z", "end": "2023-06-01T00:00:00Z"},
                    }
                },
                "overlap",
            ),
        ],
    )
    def test_rejects_an_invalid_request(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            Manifest.model_validate(manifest_data(**overrides))

    @pytest.mark.parametrize(
        ("markets", "probe", "message"),
        [
            (None, {"min_fixture_lead_hours": 168}, "fixture lead"),
            (None, {"units_per_call": 10}, "units per call"),
            (None, {"required_bookmakers": {"h2h": ["pinnacle"]}}, "unrequested books"),
            (
                [{"key": "h2h"}],
                {"required_bookmakers": {"totals": ["book0"]}, "units_per_call": 10},
                "unrequested market",
            ),
        ],
    )
    def test_the_probe_matches_the_request(
        self, markets: list[dict[str, str]] | None, probe: dict[str, Any], message: str
    ) -> None:
        data = manifest_data(**({"markets": markets} if markets else {}))
        data["probe"] = data["probe"] | probe

        with pytest.raises(ValidationError, match=message):
            Manifest.model_validate(data)

    def test_an_offset_belongs_to_one_window_group(self) -> None:
        windows = [
            {"minutes_before_kickoff": [60], "max_input_age_minutes": 10},
            {"minutes_before_kickoff": [60], "max_input_age_minutes": 60},
        ]

        with pytest.raises(ValidationError, match="one window group"):
            Manifest.model_validate(manifest_data(windows=windows))

    def test_window_cuts_drop_one_window_each(self, manifest: Manifest) -> None:
        cuts = manifest.window_cuts()

        assert [window for window, _ in cuts][:2] == [
            Window(timedelta(minutes=5), timedelta(minutes=10)),
            Window(timedelta(minutes=10), timedelta(minutes=10)),
        ]
        for window, rest in cuts:
            assert rest is not None
            kept = {m for params in rest.windows for m in params.minutes_before_kickoff}
            assert len(kept) == 9
            assert window.lead not in {timedelta(minutes=m) for m in kept}

    def test_the_only_window_cuts_to_none(self) -> None:
        single = Manifest.model_validate(
            manifest_data(windows=[{"minutes_before_kickoff": [5], "max_input_age_minutes": 10}])
        )

        assert single.window_cuts() == [(Window(timedelta(minutes=5), timedelta(minutes=10)), None)]


class TestLoadFixtures:
    async def test_loads_the_seasons_final_events_of_the_sport(self, test_session) -> None:
        season = Season(
            start=datetime(2022, 7, 1, tzinfo=UTC), end=datetime(2023, 7, 1, tzinfo=UTC)
        )
        rows = [
            (1, "soccer_epl", datetime(2022, 8, 6, 14, 0), EventStatus.FINAL),
            (2, "soccer_epl", datetime(2022, 8, 6, 11, 30), EventStatus.FINAL),
            (3, "soccer_epl", datetime(2022, 8, 7, 14, 0), EventStatus.SCHEDULED),
            (4, "baseball_mlb", datetime(2022, 8, 6, 18, 0), EventStatus.FINAL),
            (5, "soccer_epl", datetime(2023, 7, 1, 0, 0), EventStatus.FINAL),
            (6, "soccer_epl", datetime(2022, 7, 1, 0, 0), EventStatus.FINAL),
        ]
        for event_id, sport, kickoff, status in rows:
            await add_event(
                test_session,
                Event(
                    id=event_id,
                    sport_key=sport,
                    sport_title=sport,
                    commence_time=kickoff.replace(tzinfo=UTC),
                    home_team=f"Home {event_id}",
                    away_team=f"Away {event_id}",
                    status=status,
                ),
            )

        fixtures = await load_fixtures(test_session, "soccer_epl", season)

        assert [f.event_id for f in fixtures] == [6, 2, 1]
