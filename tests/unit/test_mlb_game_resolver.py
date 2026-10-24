"""Tests for resolving MLB events to their gamePk (#612).

Schedule slots are copied from the 2026 MLB Stats API schedule (``/schedule`` with
``hydrate=linescore``), trimmed to the fields the fetcher reads. Hand-built slots say so.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from odds_lambda.mlb_game_resolver import (
    GameResolution,
    GameRow,
    MlbScheduleUnavailable,
    ResolutionRule,
    ResolvedGame,
    UnresolvedGame,
    UnresolvedReason,
    resolve_batch,
)
from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher

METS, ROCKIES = "New York Mets", "Colorado Rockies"
ROYALS, BREWERS = "Kansas City Royals", "Milwaukee Brewers"
PADRES, CARDINALS = "San Diego Padres", "St. Louis Cardinals"
BRAVES, GIANTS = "Atlanta Braves", "San Francisco Giants"

FINAL = {"abstractGameState": "Final", "codedGameState": "F", "detailedState": "Final"}
POSTPONED = {"abstractGameState": "Final", "codedGameState": "D", "detailedState": "Postponed"}
SCHEDULED = {"abstractGameState": "Preview", "codedGameState": "S", "detailedState": "Scheduled"}
LIVE = {"abstractGameState": "Live", "codedGameState": "I", "detailedState": "In Progress"}
CANCELLED = {"abstractGameState": "Final", "codedGameState": "C", "detailedState": "Cancelled"}
OTHER = {"abstractGameState": "Other", "codedGameState": "W", "detailedState": "Other"}


def _picks(result: list[GameResolution]) -> list[tuple[int, ResolutionRule] | GameResolution]:
    return [(r.game_pk, r.rule) if isinstance(r, ResolvedGame) else r for r in result]


def _slot(
    pk: int,
    game_date: str,
    official_date: str,
    home: str,
    away: str,
    status: dict[str, Any] = FINAL,
    game_number: int = 1,
    tbd: bool = False,
    resumed_from: str | None = None,
) -> dict[str, Any]:
    game: dict[str, Any] = {
        "gamePk": pk,
        "gameDate": game_date,
        "officialDate": official_date,
        "gameNumber": game_number,
        "status": {**status, "startTimeTBD": tbd},
        "teams": {"home": {"team": {"name": home}}, "away": {"team": {"name": away}}},
    }
    if resumed_from is not None:
        game["resumedFrom"] = resumed_from
    return game


def _mets_rockies(game_two_status: dict[str, Any] = FINAL) -> dict[str, list[dict[str, Any]]]:
    """Mets-Rockies 04-24 to 04-26: 04-25 postponed into a doubleheader whose game 2 is TBD."""
    return {
        "2026-04-24": [
            _slot(823638, "2026-04-24T23:10:00Z", "2026-04-24", METS, ROCKIES),
        ],
        "2026-04-25": [
            _slot(823637, "2026-04-25T20:10:00Z", "2026-04-26", METS, ROCKIES, POSTPONED),
        ],
        "2026-04-26": [
            _slot(823635, "2026-04-26T17:40:00Z", "2026-04-26", METS, ROCKIES),
            _slot(
                823637,
                "2026-04-26T17:45:00Z",
                "2026-04-26",
                METS,
                ROCKIES,
                game_two_status,
                game_number=2,
                tbd=True,
            ),
        ],
    }


ROYALS_BREWERS = {
    "2026-04-03": [
        _slot(824134, "2026-04-03T23:45:00Z", "2026-04-04", ROYALS, BREWERS, POSTPONED),
    ],
    "2026-04-04": [
        _slot(824132, "2026-04-04T18:10:00Z", "2026-04-04", ROYALS, BREWERS),
        _slot(824134, "2026-04-04T23:10:00Z", "2026-04-04", ROYALS, BREWERS, game_number=2),
    ],
}

PADRES_CARDINALS = {
    "2026-05-07": [_slot(823306, "2026-05-08T02:10:00Z", "2026-05-07", PADRES, CARDINALS)],
    "2026-05-08": [_slot(823303, "2026-05-09T01:45:00Z", "2026-05-08", PADRES, CARDINALS)],
}

BRAVES_GIANTS = {
    "2026-06-16": [_slot(824912, "2026-06-16T23:15:00Z", "2026-06-16", BRAVES, GIANTS)],
    "2026-06-17": [
        _slot(
            824912,
            "2026-06-17T18:00:00Z",
            "2026-06-16",
            BRAVES,
            GIANTS,
            resumed_from="2026-06-16T23:15:00Z",
        ),
        _slot(824913, "2026-06-17T23:15:00Z", "2026-06-17", BRAVES, GIANTS),
    ],
}


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


class ScheduleServer:
    def __init__(self) -> None:
        self.days: dict[str, list[dict[str, Any]]] = {}
        self.requests: list[httpx.Request] = []
        self.response: httpx.Response | None = None

    def add(self, days: dict[str, list[dict[str, Any]]]) -> None:
        for day, games in days.items():
            self.days.setdefault(day, []).extend(games)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.response is not None:
            return self.response
        start, end = request.url.params["startDate"], request.url.params["endDate"]
        dates = [
            {"date": day, "games": games}
            for day, games in sorted(self.days.items())
            if start <= day <= end
        ]
        return httpx.Response(200, json={"dates": dates})


@pytest.fixture
def server() -> ScheduleServer:
    return ScheduleServer()


@pytest.fixture
def resolve(server: ScheduleServer):
    async def run(rows: list[GameRow]) -> list[GameResolution]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(server.handle))
        async with MlbStatsFetcher(client=client, request_delay_seconds=0) as fetcher:
            return await resolve_batch(rows, fetcher)

    return run


class TestSingleGame:
    async def test_takes_the_only_game_of_the_pair_that_day(self, server, resolve) -> None:
        server.add(_mets_rockies())

        result = await resolve([GameRow(METS, ROCKIES, _utc("2026-04-24T23:10"))])

        assert _picks(result) == [(823638, ResolutionRule.SINGLE_GAME)]

    async def test_matches_betfair_team_names(self, server, resolve) -> None:
        server.add(PADRES_CARDINALS)

        result = await resolve([GameRow(PADRES, "St Louis Cardinals", _utc("2026-05-09T01:45"))])

        assert _picks(result) == [(823303, ResolutionRule.SINGLE_GAME)]

    async def test_late_night_utc_start_resolves_on_its_eastern_date(self, server, resolve) -> None:
        server.add(PADRES_CARDINALS)

        result = await resolve(
            [
                GameRow(PADRES, CARDINALS, _utc("2026-05-08T02:10")),
                GameRow(PADRES, CARDINALS, _utc("2026-05-09T01:45")),
            ]
        )

        assert [r.game_pk for r in result if isinstance(r, ResolvedGame)] == [823306, 823303]

    async def test_resumed_continuation_slot_is_not_a_game(self, server, resolve) -> None:
        server.add(BRAVES_GIANTS)

        result = await resolve([GameRow(BRAVES, GIANTS, _utc("2026-06-17T23:15"))])

        assert _picks(result) == [(824913, ResolutionRule.SINGLE_GAME)]

    async def test_three_game_series_listed_newest_first(self, server, resolve) -> None:
        server.add(_mets_rockies())
        rows = [
            GameRow(METS, ROCKIES, _utc("2026-04-26T21:11")),
            GameRow(METS, ROCKIES, _utc("2026-04-26T17:40")),
            GameRow(METS, ROCKIES, _utc("2026-04-25T20:10")),
            GameRow(METS, ROCKIES, _utc("2026-04-24T23:10")),
        ]

        result = await resolve(rows)

        assert [r.game_pk for r in result if isinstance(r, ResolvedGame)] == [
            823637,
            823635,
            823637,
            823638,
        ]


class TestPostponedSlot:
    async def test_row_at_a_postponed_slot_takes_its_game(self, server, resolve) -> None:
        server.add(_mets_rockies())

        result = await resolve([GameRow(METS, ROCKIES, _utc("2026-04-25T20:10"))])

        assert _picks(result) == [(823637, ResolutionRule.SINGLE_GAME)]

    async def test_postponed_slot_beside_another_game_is_matched_by_start_time(
        self, server, resolve
    ) -> None:
        # Hand-built: a date holding a postponed slot and a later game of the same pair.
        server.add(
            {
                "2026-07-01": [
                    _slot(900001, "2026-07-01T17:05:00Z", "2026-07-02", ROYALS, BREWERS, POSTPONED),
                    _slot(900002, "2026-07-01T23:05:00Z", "2026-07-01", ROYALS, BREWERS, SCHEDULED),
                ]
            }
        )

        # OddsPortal often lists a start one minute after MLB's.
        result = await resolve([GameRow(ROYALS, BREWERS, _utc("2026-07-01T17:06"))])

        assert _picks(result) == [(900001, ResolutionRule.START_TIME)]


class TestDoubleheader:
    async def test_split_doubleheader_pairs_rows_by_start_order(self, server, resolve) -> None:
        server.add(ROYALS_BREWERS)

        result = await resolve(
            [
                GameRow(ROYALS, BREWERS, _utc("2026-04-04T23:10")),
                GameRow(ROYALS, BREWERS, _utc("2026-04-04T18:10")),
            ]
        )

        assert _picks(result) == [
            (824134, ResolutionRule.START_ORDER),
            (824132, ResolutionRule.START_ORDER),
        ]

    async def test_tbd_game_two_listed_at_an_estimated_start_pairs_by_order(
        self, server, resolve
    ) -> None:
        # OddsPortal listed game 2 at 21:11; MLB's placeholder is 17:45.
        server.add(_mets_rockies(game_two_status=SCHEDULED))

        result = await resolve(
            [
                GameRow(METS, ROCKIES, _utc("2026-04-26T17:40")),
                GameRow(METS, ROCKIES, _utc("2026-04-26T21:11")),
            ]
        )

        assert _picks(result) == [
            (823635, ResolutionRule.START_ORDER),
            (823637, ResolutionRule.START_ORDER),
        ]

    async def test_betfair_game_number_picks_the_game(self, server, resolve) -> None:
        server.add(_mets_rockies(game_two_status=SCHEDULED))

        result = await resolve(
            [GameRow("NY Mets", ROCKIES, _utc("2026-04-26T17:45"), game_number=2)]
        )

        assert _picks(result) == [(823637, ResolutionRule.GAME_NUMBER)]

    @pytest.mark.parametrize("game_one", [FINAL, LIVE])
    async def test_lone_row_takes_the_game_not_yet_started(
        self, server, resolve, game_one: dict[str, Any]
    ) -> None:
        days = _mets_rockies(game_two_status=SCHEDULED)
        days["2026-04-26"][0]["status"] = {**game_one, "startTimeTBD": False}
        server.add(days)

        result = await resolve([GameRow(METS, ROCKIES, _utc("2026-04-26T21:11"))])

        assert _picks(result) == [(823637, ResolutionRule.SIBLING_STARTED)]

    async def test_lone_game_one_row_while_game_two_is_pending(self, server, resolve) -> None:
        days = {
            day: [{**g, "status": {**SCHEDULED, "startTimeTBD": False}} for g in games]
            for day, games in ROYALS_BREWERS.items()
        }
        server.add(days)

        result = await resolve([GameRow(ROYALS, BREWERS, _utc("2026-04-04T18:11"))])

        assert _picks(result) == [(824132, ResolutionRule.START_TIME)]

    async def test_lone_game_one_row_while_game_one_is_live(self, server, resolve) -> None:
        days = {
            "2026-04-04": [
                {**ROYALS_BREWERS["2026-04-04"][0], "status": {**LIVE, "startTimeTBD": False}},
                {**ROYALS_BREWERS["2026-04-04"][1], "status": {**SCHEDULED, "startTimeTBD": False}},
            ]
        }
        server.add(days)

        result = await resolve([GameRow(ROYALS, BREWERS, _utc("2026-04-04T18:11"))])

        assert _picks(result) == [(824132, ResolutionRule.START_TIME)]


# The four 2026 doubleheaders whose game 2 had startTimeTBD: (date, home, away, game 1, game 2),
# each game as (gamePk, gameDate). Game 2's start is MLB's placeholder, 5 min after game 1.
TBD_DOUBLEHEADERS = [
    (
        "2026-04-05",
        "Cleveland Guardians",
        "Chicago Cubs",
        (824459, "2026-04-05T17:10"),
        (824460, "2026-04-05T17:15"),
    ),
    ("2026-04-26", METS, ROCKIES, (823635, "2026-04-26T17:40"), (823637, "2026-04-26T17:45")),
    (
        "2026-04-30",
        "Baltimore Orioles",
        "Houston Astros",
        (824848, "2026-04-30T16:35"),
        (824850, "2026-04-30T16:40"),
    ),
    (
        "2026-09-25",
        "New York Yankees",
        "Baltimore Orioles",
        (823491, "2026-09-25T20:05"),
        (823489, "2026-09-25T20:10"),
    ),
]


def _tbd_doubleheader(
    day: str,
    home: str,
    away: str,
    game_one: tuple[int, str],
    game_two: tuple[int, str],
    game_one_status: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        day: [
            _slot(game_one[0], f"{game_one[1]}:00Z", day, home, away, game_one_status),
            _slot(
                game_two[0],
                f"{game_two[1]}:00Z",
                day,
                home,
                away,
                SCHEDULED,
                game_number=2,
                tbd=True,
            ),
        ]
    }


@pytest.mark.parametrize(("day", "home", "away", "game_one", "game_two"), TBD_DOUBLEHEADERS)
class TestTbdDoubleheaderLoneRowAtPlaceholder:
    async def test_takes_game_two_while_game_one_is_live(
        self, server, resolve, day, home, away, game_one, game_two
    ) -> None:
        server.add(_tbd_doubleheader(day, home, away, game_one, game_two, LIVE))

        result = await resolve([GameRow(home, away, _utc(game_two[1]) + timedelta(minutes=1))])

        assert _picks(result) == [(game_two[0], ResolutionRule.SIBLING_STARTED)]

    async def test_unresolved_while_game_one_is_scheduled(
        self, server, resolve, day, home, away, game_one, game_two
    ) -> None:
        server.add(_tbd_doubleheader(day, home, away, game_one, game_two, SCHEDULED))

        [result] = await resolve([GameRow(home, away, _utc(game_two[1]) + timedelta(minutes=1))])

        assert isinstance(result, UnresolvedGame)
        assert result.reason is UnresolvedReason.AMBIGUOUS_DOUBLEHEADER


class TestUnresolved:
    async def test_no_game_of_the_pair(self, server, resolve) -> None:
        server.add(_mets_rockies())

        result = await resolve([GameRow(ROCKIES, METS, _utc("2026-04-24T23:10"))])

        assert result == [UnresolvedGame(UnresolvedReason.NO_GAME)]

    async def test_single_game_starting_far_from_the_row(self, server, resolve) -> None:
        # op_live_ATLBRA_SANGIA_2026-06-17T1605 matches no slot of that day.
        server.add(BRAVES_GIANTS)

        [result] = await resolve([GameRow(BRAVES, GIANTS, _utc("2026-06-17T16:05"))])

        assert isinstance(result, UnresolvedGame)
        assert result.reason is UnresolvedReason.START_TOO_FAR

    async def test_unknown_game_number(self, server, resolve) -> None:
        server.add(ROYALS_BREWERS)

        [result] = await resolve(
            [GameRow(ROYALS, BREWERS, _utc("2026-04-04T23:10"), game_number=3)]
        )

        assert isinstance(result, UnresolvedGame)
        assert result.reason is UnresolvedReason.GAME_NUMBER_NOT_FOUND

    async def test_lone_row_when_both_games_are_final(self, server, resolve) -> None:
        server.add(ROYALS_BREWERS)

        [result] = await resolve([GameRow(ROYALS, BREWERS, _utc("2026-04-04T20:30"))])

        assert isinstance(result, UnresolvedGame)
        assert result.reason is UnresolvedReason.AMBIGUOUS_DOUBLEHEADER

    async def test_two_games_within_the_start_window(self, server, resolve) -> None:
        # Hand-built: two games of the pair 10 minutes apart, neither TBD.
        server.add(
            {
                "2026-07-01": [
                    _slot(900001, "2026-07-01T17:05:00Z", "2026-07-01", ROYALS, BREWERS, SCHEDULED),
                    _slot(
                        900002,
                        "2026-07-01T17:15:00Z",
                        "2026-07-01",
                        ROYALS,
                        BREWERS,
                        SCHEDULED,
                        game_number=2,
                    ),
                ]
            }
        )

        [result] = await resolve([GameRow(ROYALS, BREWERS, _utc("2026-07-01T17:10"))])

        assert isinstance(result, UnresolvedGame)
        assert result.reason is UnresolvedReason.AMBIGUOUS_DOUBLEHEADER

    @pytest.mark.parametrize("game_two", [CANCELLED, OTHER])
    async def test_lone_row_when_the_other_game_cannot_be_played(
        self, server, resolve, game_two: dict[str, Any]
    ) -> None:
        days = {
            "2026-04-04": [
                ROYALS_BREWERS["2026-04-04"][0],
                {**ROYALS_BREWERS["2026-04-04"][1], "status": {**game_two, "startTimeTBD": False}},
            ]
        }
        server.add(days)

        [result] = await resolve([GameRow(ROYALS, BREWERS, _utc("2026-04-04T23:40"))])

        assert isinstance(result, UnresolvedGame)
        assert result.reason is UnresolvedReason.AMBIGUOUS_DOUBLEHEADER

    async def test_doubleheader_rows_at_the_same_start(self, server, resolve) -> None:
        server.add(ROYALS_BREWERS)
        row = GameRow(ROYALS, BREWERS, _utc("2026-04-04T18:10"))

        result = await resolve([row, row])

        assert result == [UnresolvedGame(UnresolvedReason.SAME_START_ROWS)] * 2


class TestResolveBatch:
    async def test_one_schedule_request_over_the_eastern_dates_spanned(
        self, server, resolve
    ) -> None:
        server.add(PADRES_CARDINALS)

        await resolve(
            [
                GameRow(PADRES, CARDINALS, _utc("2026-05-09T01:45")),
                GameRow(PADRES, CARDINALS, _utc("2026-05-08T02:10")),
            ]
        )

        [request] = server.requests
        assert (request.url.params["startDate"], request.url.params["endDate"]) == (
            "2026-05-07",
            "2026-05-08",
        )

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(503),
            httpx.Response(200, content=b"<html>not json</html>"),
            httpx.Response(200, json={"copyright": "MLB"}),
            httpx.Response(200, json={"dates": []}),
        ],
        ids=["http-error", "bad-json", "no-dates", "no-games"],
    )
    async def test_unusable_schedule_raises_unavailable(
        self, server, resolve, monkeypatch, response: httpx.Response
    ) -> None:
        monkeypatch.setattr("odds_lambda.mlb_stats_fetcher._RETRY_BACKOFF_SECONDS", 0)
        server.response = response

        with pytest.raises(MlbScheduleUnavailable):
            await resolve([GameRow(METS, ROCKIES, _utc("2026-04-24T23:10"))])

    async def test_empty_batch_makes_no_request(self, server, resolve) -> None:
        assert await resolve([]) == []
        assert server.requests == []
