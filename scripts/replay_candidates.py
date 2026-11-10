"""Write the replay of candidate files as canonical JSON, reading the database read-only.

Run with a checkout's own environment (``uv run --project <checkout> python <this file>``) to
replay under that checkout's code; ``odds candidates rehash`` compares two such runs. It uses
only interfaces that exist across revisions of the replay backtest.
"""

import argparse
import asyncio
import dataclasses
import json
from datetime import UTC, date, datetime
from pathlib import Path

from odds_analytics.candidate_backtest import _Counts, _decide, final_events, replay
from odds_core.candidates import MechanicalCandidate, load_candidate_file
from odds_core.database import async_session_maker
from odds_lambda.candidate_rules import resolve_rule
from sqlalchemy import text

# Fixed so the summary's pending counts do not depend on when it runs.
NOW = datetime(2000, 1, 1, tzinfo=UTC)


async def dump(paths: list[Path], start: date, end: date) -> dict[str, object]:
    out: dict[str, object] = {}
    async with async_session_maker() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))
        for path in paths:
            candidate = load_candidate_file(path)
            assert isinstance(candidate, MechanicalCandidate)  # noqa: S101
            spec, params = resolve_rule(candidate)
            events = await final_events(session, candidate.sport, start, end)
            counts = _Counts()
            bets = await _decide(session, candidate, spec, params, events, counts)
            summary = dataclasses.asdict(await replay(session, candidate, start, end, 1, now=NOW))
            summary.pop("hash_status")
            out[candidate.name] = {
                "counts": dataclasses.asdict(counts),
                "bets": [dataclasses.asdict(b) for b in bets],
                "summary": summary,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="end", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    result = asyncio.run(dump(args.paths, args.start, args.end))
    args.out.write_text(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
