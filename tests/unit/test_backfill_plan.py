"""Unit tests for the Odds API historical backfill planner."""

import bisect
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
    load_fixtures,
    plan,
    targets,
)
from odds_core.models import Event, EventStatus
from pydantic import ValidationError

from tests.test_helpers import add_event

REPO = Path(__file__).resolve().parents[2]
COMMITTED = sorted((REPO / "backfills").glob("*.yaml"))
REFERENCE: Manifest = Manifest.load(REPO / "backfills" / "odds-api-epl-2022-26.yaml")
# A close sample covers only the last minutes before kick-off, with a band of its own; every other
# manifest is a full grid and starts with the reference's nearest band.
CLOSE_SAMPLE_REACH_MINUTES = 10


def is_full_grid(manifest: Manifest) -> bool:
    return max(band.max_lead_minutes for band in manifest.coverage) > CLOSE_SAMPLE_REACH_MINUTES


FULL_GRIDS: list[Path] = [path for path in COMMITTED if is_full_grid(Manifest.load(path))]
FIVE_MINUTES_FROM = datetime(2022, 9, 1, 0, 0, 38, tzinfo=UTC)


def band(min_lead: int, max_lead: int, age: int) -> dict[str, int]:
    return {
        "min_lead_minutes": min_lead,
        "max_lead_minutes": max_lead,
        "max_input_age_minutes": age,
    }


def manifest_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "sport": "soccer_epl",
        "seasons": {"2022-23": {"start": "2022-07-01T00:00:00Z", "end": "2023-07-01T00:00:00Z"}},
        "markets": ["h2h", "totals"],
        "bookmakers": [f"book{i}" for i in range(10)],
        "coverage": [
            band(1, 60, 10),
            band(61, 180, 30),
            band(181, 600, 120),
        ],
        "snapshot_grid": [
            {"start": "2020-06-06T00:05:00Z", "step_minutes": 10},
            {"start": "2022-09-01T00:00:38Z", "step_minutes": 5},
        ],
        "probe": {
            "snapshots": ["2022-09-17T09:55:38Z"],
            "min_fixture_lead_hours": 12,
            "required_bookmakers": {"h2h": ["book0"]},
        },
    }
    return data | overrides


@pytest.fixture
def manifest() -> Manifest:
    return Manifest.model_validate(manifest_data())


def random_fixtures(seed: int, n: int) -> list[Fixture]:
    """Kick-offs on any minute either side of the grid change, many sharing a slot."""
    rng = random.Random(seed)
    base = datetime(2022, 9, 1, tzinfo=UTC) - timedelta(days=20)
    slots = [base + timedelta(minutes=rng.randrange(40 * 24 * 60)) for _ in range(n // 3)]
    return [Fixture(i, rng.choice(slots)) for i in range(n)]


def decisions(fixtures: list[Fixture], manifest: Manifest) -> list[tuple[datetime, timedelta]]:
    """Every whole-minute decision time of every fixture's bands, with its max age."""
    return [
        (f.kickoff - timedelta(minutes=lead), b.max_age)
        for f in fixtures
        for b in manifest.coverage
        for lead in range(b.min_lead_minutes, b.max_lead_minutes + 1)
    ]


class TestPlan:
    @pytest.mark.parametrize("seed", range(5))
    def test_every_decision_minute_has_a_fresh_enough_call(
        self, manifest: Manifest, seed: int
    ) -> None:
        fixtures = random_fixtures(seed, 30)
        times = [call.time for call in plan(manifest, fixtures)]

        for decision, age in decisions(fixtures, manifest):
            latest = times[bisect.bisect_right(times, decision) - 1]
            assert decision - age <= latest <= decision, (decision, age, latest)

    @pytest.mark.parametrize("seed", range(5))
    def test_every_call_lies_inside_each_band_it_serves(
        self, manifest: Manifest, seed: int
    ) -> None:
        fixtures = random_fixtures(seed, 30)
        calls = plan(manifest, fixtures)

        for call in calls:
            for target in call.serves:
                assert target.contains(call.time), (call.time, target)
        served = {target for call in calls for target in call.serves}
        assert served == set(targets(manifest, fixtures))

    @pytest.mark.parametrize("seed", range(20))
    def test_no_fewer_grid_calls_cover_every_decision(self, seed: int) -> None:
        small = Manifest.model_validate(manifest_data(coverage=[band(1, 20, 10), band(21, 40, 20)]))
        rng = random.Random(seed)
        base = datetime(2022, 9, 1, tzinfo=UTC) - timedelta(minutes=45)
        fixtures = [Fixture(i, base + timedelta(minutes=rng.randrange(30))) for i in range(2)]
        needs = decisions(fixtures, small)
        lo = min(d - age for d, age in needs)
        hi = max(d for d, _ in needs)
        points = sorted(
            {
                small.floor_to_grid(lo + timedelta(minutes=m))
                for m in range(int((hi - lo).total_seconds() // 60) + 1)
            }
        )
        points = [p for p in points if p >= lo]

        def covers(chosen: tuple[datetime, ...]) -> bool:
            return all(any(d - age <= p <= d for p in chosen) for d, age in needs)

        fewest = next(
            k for k in range(1, len(points) + 1) if any(map(covers, combinations(points, k)))
        )

        assert len(plan(small, fixtures)) == fewest

    def test_calls_fall_on_the_phased_snapshot_grid(self, manifest: Manifest) -> None:
        for call in plan(manifest, random_fixtures(0, 30)):
            if call.time >= FIVE_MINUTES_FROM:
                assert (call.time.minute % 5, call.time.second) == (0, 38)
            else:
                assert (call.time.minute % 10, call.time.second) == (5, 0)

    @pytest.mark.parametrize("seed", range(5))
    def test_no_call_is_after_a_decision_it_is_the_input_to(
        self, manifest: Manifest, seed: int
    ) -> None:
        fixtures = random_fixtures(seed, 30)
        times = [call.time for call in plan(manifest, fixtures)]

        for decision, _ in decisions(fixtures, manifest):
            index = bisect.bisect_right(times, decision) - 1
            assert index >= 0 and times[index] <= decision

    def test_a_phased_grid_takes_the_snapshot_before_the_decision_minute(self) -> None:
        single = Manifest.model_validate(manifest_data(coverage=[band(1, 10, 10)]))
        kickoff = datetime(2022, 10, 1, 14, 0, tzinfo=UTC)

        calls = plan(single, [Fixture(1, kickoff)])

        assert [c.time for c in calls] == [
            datetime(2022, 10, 1, 13, 45, 38, tzinfo=UTC),
            datetime(2022, 10, 1, 13, 55, 38, tzinfo=UTC),
        ]

    def test_fixtures_sharing_a_kickoff_share_calls(self, manifest: Manifest) -> None:
        kickoff = datetime(2022, 10, 1, 14, 0, tzinfo=UTC)

        calls = plan(manifest, [Fixture(1, kickoff), Fixture(2, kickoff)])

        assert len(calls) == len(plan(manifest, [Fixture(1, kickoff)]))
        assert all({t.fixture.event_id for t in call.serves} == {1, 2} for call in calls)

    def test_a_max_age_shorter_than_the_grid_raises(self) -> None:
        narrow = Manifest.model_validate(manifest_data(coverage=[band(1, 30, 3)]))

        with pytest.raises(ValueError, match="no snapshot"):
            plan(narrow, [Fixture(1, datetime(2021, 10, 1, 19, 45, tzinfo=UTC))])

    def test_a_kickoff_between_minutes_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not on a whole minute"):
            Fixture(1, datetime(2022, 10, 1, 15, 50, 30, tzinfo=UTC))

    def test_a_decision_before_the_first_snapshot_raises(self, manifest: Manifest) -> None:
        with pytest.raises(ValueError, match="before the first snapshot"):
            plan(manifest, [Fixture(1, datetime(2020, 6, 6, 1, 0, tzinfo=UTC))])


class TestManifest:
    @pytest.mark.parametrize("path", COMMITTED, ids=lambda p: p.stem)
    def test_every_committed_manifest_plans(self, path: Path) -> None:
        committed = Manifest.load(path)
        season = next(iter(committed.seasons.values()))
        kickoff = season.start + timedelta(days=60, hours=14)

        assert plan(committed, [Fixture(1, kickoff)])

    @pytest.mark.parametrize("path", COMMITTED, ids=lambda p: p.stem)
    def test_every_committed_manifest_requests_the_same_prices(self, path: Path) -> None:
        committed = Manifest.load(path)

        assert (committed.bookmakers, committed.regions) == (
            REFERENCE.bookmakers,
            REFERENCE.regions,
        )
        assert committed.probe.required_bookmakers == REFERENCE.probe.required_bookmakers
        assert committed.markets == REFERENCE.markets

    @pytest.mark.parametrize("path", FULL_GRIDS, ids=lambda p: p.stem)
    def test_a_full_grid_manifest_starts_with_the_reference_close_band(self, path: Path) -> None:
        assert Manifest.load(path).coverage[0] == REFERENCE.coverage[0]

    @pytest.mark.parametrize(
        ("bands", "full"),
        [
            ([band(1, 6, 5)], False),
            ([band(1, 1, 5)], False),
            ([band(1, 90, 10)], True),
            ([band(1, 120, 10), band(121, 480, 60)], True),
        ],
    )
    def test_a_manifest_is_a_full_grid_unless_it_covers_only_the_last_minutes(
        self, bands: list[dict[str, int]], full: bool
    ) -> None:
        assert is_full_grid(Manifest.model_validate(manifest_data(coverage=bands))) is full

    @pytest.mark.parametrize(
        ("books", "regions", "units"),
        [(10, (), 20), (11, (), 40), (0, ("uk",), 20), (0, ("uk", "eu"), 40)],
    )
    def test_units_per_call(self, books: int, regions: tuple[str, ...], units: int) -> None:
        data = manifest_data(bookmakers=[f"book{i}" for i in range(books)], regions=list(regions))
        data["probe"]["required_bookmakers"] = {}

        assert Manifest.model_validate(data).units_per_call == units

    @pytest.mark.parametrize(("books", "regions"), [([], []), (["pinnacle"], ["uk"])])
    def test_names_either_bookmakers_or_regions(self, books: list[str], regions: list[str]) -> None:
        with pytest.raises(ValidationError, match="either bookmakers or regions|probe requires"):
            Manifest.model_validate(manifest_data(bookmakers=books, regions=regions))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"markets": ["h2h", "h2h"]}, "requested once"),
            ({"markets": ["1x2"]}, "markets"),
            ({"markets": ["h2h", {"key": "totals", "line": 2.5}]}, "markets"),
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
            (None, {"min_fixture_lead_hours": 11}, "fixture lead"),
            (None, {"required_bookmakers": {"h2h": ["pinnacle"]}}, "unrequested books"),
            (
                ["h2h"],
                {"required_bookmakers": {"totals": ["book0"]}},
                "unrequested market",
            ),
            (None, {"snapshots": ["2023-09-16T10:00:00Z"]}, "outside every season"),
            (None, {"snapshots": ["2020-01-01T00:00:00Z"]}, "outside every season"),
            (None, {"snapshots": ["2022-09-17T10:00:00Z"]}, "off the grid"),
            (None, {"snapshots": ["2022-09-17T09:55:37Z"]}, "off the grid"),
        ],
    )
    def test_the_probe_matches_the_request(
        self, markets: list[str] | None, probe: dict[str, Any], message: str
    ) -> None:
        data = manifest_data(**({"markets": markets} if markets else {}))
        data["probe"] = data["probe"] | probe

        with pytest.raises(ValidationError, match=message):
            Manifest.model_validate(data)

    @pytest.mark.parametrize(
        ("coverage", "message"),
        [
            ([band(1, 60, 10), band(60, 120, 30)], "overlap"),
            ([band(1, 60, 10), band(30, 40, 5)], "overlap"),
            ([band(60, 1, 10)], "after max lead"),
        ],
    )
    def test_rejects_invalid_coverage(self, coverage: list[dict[str, int]], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            Manifest.model_validate(manifest_data(coverage=coverage))

    def test_grid_step_is_the_interval_in_force(self, manifest: Manifest) -> None:
        assert manifest.grid_step_at(FIVE_MINUTES_FROM - timedelta(minutes=1)) == timedelta(
            minutes=10
        )
        assert manifest.grid_step_at(FIVE_MINUTES_FROM) == timedelta(minutes=5)

    @pytest.mark.parametrize(
        ("snapshot", "requested"),
        [
            (
                datetime(2022, 9, 17, 9, 55, 38, tzinfo=UTC),
                datetime(2022, 9, 17, 9, 58, 8, tzinfo=UTC),
            ),
            (datetime(2021, 9, 18, 9, 55, tzinfo=UTC), datetime(2021, 9, 18, 10, 0, tzinfo=UTC)),
        ],
    )
    def test_a_snapshot_is_requested_half_its_step_later(
        self, manifest: Manifest, snapshot: datetime, requested: datetime
    ) -> None:
        assert manifest.request_time(snapshot) == requested

    def test_grid_step_before_the_first_snapshot_raises(self, manifest: Manifest) -> None:
        with pytest.raises(ValueError, match="before the first snapshot"):
            manifest.grid_step_at(datetime(2020, 1, 1, tzinfo=UTC))

    def test_band_cuts_drop_one_band_each(self, manifest: Manifest) -> None:
        cuts = manifest.band_cuts()

        assert [b.min_lead_minutes for b, _ in cuts] == [1, 61, 181]
        for cut_band, rest in cuts:
            assert rest is not None
            assert cut_band not in rest.coverage
            assert len(rest.coverage) == 2

    def test_the_only_band_cuts_to_none(self) -> None:
        single = Manifest.model_validate(manifest_data(coverage=[band(1, 60, 10)]))

        assert [rest for _, rest in single.band_cuts()] == [None]


class TestLoadFixtures:
    async def test_loads_the_seasons_final_events_of_the_sport(self, test_session) -> None:
        season = Season(
            start=datetime(2022, 7, 1, tzinfo=UTC), end=datetime(2023, 7, 1, tzinfo=UTC)
        )
        rows = [
            (1, "soccer_epl", datetime(2022, 8, 6, 14, 0, tzinfo=UTC), EventStatus.FINAL),
            (2, "soccer_epl", datetime(2022, 8, 6, 11, 30, tzinfo=UTC), EventStatus.FINAL),
            (3, "soccer_epl", datetime(2022, 8, 7, 14, 0, tzinfo=UTC), EventStatus.SCHEDULED),
            (4, "baseball_mlb", datetime(2022, 8, 6, 18, 0, tzinfo=UTC), EventStatus.FINAL),
            (5, "soccer_epl", datetime(2023, 7, 1, 0, 0, tzinfo=UTC), EventStatus.FINAL),
            (6, "soccer_epl", datetime(2022, 7, 1, 0, 0, tzinfo=UTC), EventStatus.FINAL),
        ]
        for event_id, sport, kickoff, status in rows:
            await add_event(
                test_session,
                Event(
                    id=event_id,
                    sport_key=sport,
                    sport_title=sport,
                    commence_time=kickoff,
                    home_team=f"Home {event_id}",
                    away_team=f"Away {event_id}",
                    status=status,
                ),
            )

        fixtures = await load_fixtures(test_session, "soccer_epl", season)

        assert [f.event_id for f in fixtures] == [6, 2, 1]
