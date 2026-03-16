"""Unit tests for all-competition fixture cache."""

from datetime import UTC, datetime

from odds_analytics.fixture_cache import (
    FixtureCache,
    FixtureRecord,
    get_last_fixture_date,
    load_fixture_cache,
)


def _dt(year: int, month: int, day: int, hour: int = 15, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _make_cache() -> FixtureCache:
    """Build a small fixture cache for testing."""
    return {
        "Arsenal": [
            FixtureRecord(date=_dt(2025, 1, 4, 15), competition="Premier League"),
            FixtureRecord(date=_dt(2025, 1, 8, 20), competition="Champions League"),
            FixtureRecord(date=_dt(2025, 1, 11, 15), competition="Premier League"),
            FixtureRecord(date=_dt(2025, 1, 15, 20), competition="FA Cup"),
            FixtureRecord(date=_dt(2025, 1, 18, 15), competition="Premier League"),
        ],
        "Chelsea": [
            FixtureRecord(date=_dt(2025, 1, 4, 15), competition="Premier League"),
            FixtureRecord(date=_dt(2025, 1, 11, 15), competition="Premier League"),
        ],
    }


class TestGetLastFixtureDate:
    def test_finds_most_recent_before(self) -> None:
        cache = _make_cache()
        result = get_last_fixture_date(cache, "Arsenal", _dt(2025, 1, 11, 15))
        # Should find CL match on Jan 8, not EPL on Jan 4
        assert result == _dt(2025, 1, 8, 20)

    def test_strictly_before(self) -> None:
        cache = _make_cache()
        # Exactly at the match time should NOT return that match
        result = get_last_fixture_date(cache, "Arsenal", _dt(2025, 1, 8, 20))
        assert result == _dt(2025, 1, 4, 15)

    def test_before_first_fixture(self) -> None:
        cache = _make_cache()
        result = get_last_fixture_date(cache, "Arsenal", _dt(2025, 1, 1))
        assert result is None

    def test_team_not_in_cache(self) -> None:
        cache = _make_cache()
        result = get_last_fixture_date(cache, "Wolves", _dt(2025, 1, 11))
        assert result is None

    def test_empty_cache(self) -> None:
        result = get_last_fixture_date({}, "Arsenal", _dt(2025, 1, 11))
        assert result is None

    def test_after_all_fixtures(self) -> None:
        cache = _make_cache()
        result = get_last_fixture_date(cache, "Arsenal", _dt(2025, 2, 1))
        assert result == _dt(2025, 1, 18, 15)

    def test_between_fixtures(self) -> None:
        cache = _make_cache()
        # Between Jan 8 CL and Jan 11 EPL
        result = get_last_fixture_date(cache, "Arsenal", _dt(2025, 1, 10))
        assert result == _dt(2025, 1, 8, 20)


class TestLoadFixtureCache:
    def test_loads_from_data_dir(self, tmp_path: object) -> None:
        """Load fixture CSVs from a temp directory."""
        import csv
        from pathlib import Path

        data_dir = Path(str(tmp_path))
        csv_path = data_dir / "fixtures_2024-25.csv"

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["date", "team", "opponent", "competition", "home_away"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "date": "2025-01-04T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Chelsea",
                    "competition": "Premier League",
                    "home_away": "home",
                }
            )
            writer.writerow(
                {
                    "date": "2025-01-08T20:00Z",
                    "team": "Arsenal",
                    "opponent": "PSG",
                    "competition": "Champions League",
                    "home_away": "home",
                }
            )

        cache = load_fixture_cache(data_dir)
        assert "Arsenal" in cache
        assert len(cache["Arsenal"]) == 2
        assert cache["Arsenal"][0].competition == "Premier League"
        assert cache["Arsenal"][1].competition == "Champions League"
        # Opponents are also indexed
        assert "Chelsea" in cache
        assert len(cache["Chelsea"]) == 1
        assert "PSG" in cache
        assert len(cache["PSG"]) == 1

    def test_empty_dir(self, tmp_path: object) -> None:
        from pathlib import Path

        cache = load_fixture_cache(Path(str(tmp_path)))
        assert cache == {}

    def test_multiple_seasons_sorted(self, tmp_path: object) -> None:
        """Fixtures from multiple seasons are merged and sorted."""
        import csv
        from pathlib import Path

        data_dir = Path(str(tmp_path))

        for label, date_str in [("2023-24", "2024-01-04T15:00Z"), ("2024-25", "2025-01-04T15:00Z")]:
            csv_path = data_dir / f"fixtures_{label}.csv"
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["date", "team", "opponent", "competition", "home_away"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "date": date_str,
                        "team": "Arsenal",
                        "opponent": "Chelsea",
                        "competition": "Premier League",
                        "home_away": "home",
                    }
                )

        cache = load_fixture_cache(data_dir)
        assert len(cache["Arsenal"]) == 2
        assert cache["Arsenal"][0].date.year == 2024
        assert cache["Arsenal"][1].date.year == 2025
        # Chelsea appears as opponent in both seasons
        assert len(cache["Chelsea"]) == 2

    def test_opponent_indexing_for_promoted_teams(self, tmp_path: object) -> None:
        """Teams only appearing as opponents are still indexed."""
        import csv
        from pathlib import Path

        data_dir = Path(str(tmp_path))
        csv_path = data_dir / "fixtures_2024-25.csv"

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["date", "team", "opponent", "competition", "home_away"]
            )
            writer.writeheader()
            # Ipswich only appears as an opponent (promoted, not in ESPN teams API)
            writer.writerow(
                {
                    "date": "2025-01-04T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Ipswich",
                    "competition": "Premier League",
                    "home_away": "home",
                }
            )
            writer.writerow(
                {
                    "date": "2025-01-11T15:00Z",
                    "team": "Liverpool",
                    "opponent": "Ipswich",
                    "competition": "Premier League",
                    "home_away": "home",
                }
            )

        cache = load_fixture_cache(data_dir)
        assert "Ipswich" in cache
        assert len(cache["Ipswich"]) == 2
        # Can look up Ipswich's most recent fixture
        result = get_last_fixture_date(cache, "Ipswich", _dt(2025, 1, 12))
        assert result == _dt(2025, 1, 11, 15)

    def test_deduplicates_team_opponent_overlap(self, tmp_path: object) -> None:
        """When same match appears for team and opponent, no duplicates."""
        import csv
        from pathlib import Path

        data_dir = Path(str(tmp_path))
        csv_path = data_dir / "fixtures_2024-25.csv"

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["date", "team", "opponent", "competition", "home_away"]
            )
            writer.writeheader()
            # Same match from both perspectives
            writer.writerow(
                {
                    "date": "2025-01-04T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Chelsea",
                    "competition": "Premier League",
                    "home_away": "home",
                }
            )
            writer.writerow(
                {
                    "date": "2025-01-04T15:00Z",
                    "team": "Chelsea",
                    "opponent": "Arsenal",
                    "competition": "Premier League",
                    "home_away": "away",
                }
            )

        cache = load_fixture_cache(data_dir)
        # Arsenal: indexed as team + indexed as opponent = 2 raw, deduped to 1
        assert len(cache["Arsenal"]) == 1
        assert len(cache["Chelsea"]) == 1
