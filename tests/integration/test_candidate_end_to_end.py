"""A decision taken by the runner, replayed by the parity check."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_core.candidates import load_candidate_file
from odds_core.models import OddsSnapshot
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_lambda.jobs.run_candidates import candidate_events, run_decisions, runnable_candidates
from sqlalchemy import select

from tests.integration.test_candidate_backtest import (
    EXCHANGE,
    KICKOFF,
    RETAIL,
    T,
    betfair,
    candidate,
)
from tests.test_helpers import (
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
)

pytestmark = pytest.mark.integration

NOW = T + timedelta(minutes=1)


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("odds_lambda.jobs.run_candidates.candidates_dir", lambda: tmp_path)
    monkeypatch.setattr("odds_lambda.paper_trading.candidates_dir", lambda: tmp_path)
    fields = candidate().model_dump(mode="json")
    (tmp_path / "gap.yaml").write_text(yaml.safe_dump(fields))
    assert load_candidate_file(tmp_path / "gap.yaml").name == "gap"
    return tmp_path


async def seed(
    session: Any, event_id: str, retail: tuple[float, float, float], away_team: str = "Chelsea"
) -> None:
    session.add(make_epl_event(event_id, KICKOFF, away_team=away_team))
    books = [three_way_bookmaker("bet365", retail), betfair(EXCHANGE)]
    snapshots = [
        make_tagged_snapshot(event_id, T - timedelta(minutes=8), books, "oddsportal_live"),
        make_tagged_snapshot(event_id, T - timedelta(minutes=5), books, "oddsportal_live"),
        make_tagged_snapshot(
            event_id, T - timedelta(minutes=3), [betfair(EXCHANGE)], "betfair_api"
        ),
    ]
    for snapshot in snapshots:
        snapshot.created_at = NOW - timedelta(hours=1)
    for snapshot in snapshots:
        await store_tagged(session, snapshot)
    await session.commit()


async def decide(session: Any) -> None:
    runnable = await runnable_candidates("soccer_epl", NOW)
    events = await candidate_events(session, "soccer_epl", NOW)
    summary = await run_decisions(session, runnable, events, NOW)
    assert summary.failed == 0


class TestRunnerThenParity:
    async def test_every_decision_matches(self, pglite_async_session, registry) -> None:
        await seed(pglite_async_session, "bet", RETAIL)
        await seed(pglite_async_session, "none", (1.9, 3.0, 3.0), away_team="Spurs")
        await decide(pglite_async_session)

        report = await check_parity(pglite_async_session, candidate())

        assert report.checked == 2
        assert [d.status for d in report.decisions] == [ParityStatus.MATCH] * 2
        assert report.data_differences == []
        trades = (await pglite_async_session.execute(select(PaperTrade))).scalars().all()
        assert len(trades) == 1

    async def test_recorded_input_is_the_snapshot_decided_from(
        self, pglite_async_session, registry
    ) -> None:
        await seed(pglite_async_session, "bet", RETAIL)
        await decide(pglite_async_session)

        [decision] = (await pglite_async_session.execute(select(CandidateDecision))).scalars()
        [latest] = (
            await pglite_async_session.execute(
                select(OddsSnapshot.id).where(
                    OddsSnapshot.snapshot_time == T - timedelta(minutes=5)
                )
            )
        ).scalars()
        assert decision.input_snapshot_ids == [latest]

    async def test_a_later_twin_merge_is_a_data_difference_not_a_mismatch(
        self, pglite_async_session, registry
    ) -> None:
        await seed(pglite_async_session, "bet", RETAIL)
        await decide(pglite_async_session)
        # A twin's snapshot, stored before the run but moved onto the event after it.
        merged = make_tagged_snapshot(
            "bet",
            T - timedelta(minutes=2),
            [three_way_bookmaker("bet365", (2.5, 3.0, 3.0)), betfair(EXCHANGE)],
            "oddsportal_live",
        )
        merged.created_at = NOW - timedelta(hours=2)
        await store_tagged(pglite_async_session, merged)
        await pglite_async_session.commit()

        report = await check_parity(pglite_async_session, candidate())

        assert [d.status for d in report.decisions] == [ParityStatus.MATCH]
        [difference] = report.data_differences
        assert difference.replayed == (("home", "bet365", 2.2, 1.0),)
        assert difference.db_replayed == (("home", "bet365", 2.5, 1.0),)
        assert report.ok
