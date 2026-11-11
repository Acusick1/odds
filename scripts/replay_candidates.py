"""Write the replay of candidate files as canonical JSON, reading the database read-only.

Run with a checkout's own environment (``uv run --project <checkout> python <this file>``) to
replay under that checkout's code; ``odds candidates rehash`` compares two such runs. With
``--snapshot`` the replay reads the exported database snapshot, so two runs see the same data.
It uses only interfaces that exist across revisions of the replay backtest.
"""

import argparse
import asyncio
import dataclasses
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml
from odds_analytics.candidate_backtest import _Counts, _decide, final_events, replay
from odds_core.candidates import MechanicalCandidate
from odds_core.database import async_session_maker
from odds_lambda.candidate_rules import resolve_rule
from sqlalchemy import text

# Fixed so the summary's pending counts do not depend on when it runs.
NOW = datetime(2000, 1, 1, tzinfo=UTC)

# The fields a replayed bet has in every revision; a later field is left out, so a field added
# to the bet does not make two revisions' replays differ.
BET_FIELDS = (
    "candidate_name",
    "candidate_version",
    "event_id",
    "commence_time",
    "stake",
    "pnl",
    "price_decimal",
    "p_close",
    "placed_at",
    "close_grace",
)
# Summary fields that only some revisions have, or that restate the candidate file (whose
# league field is named differently across revisions), rather than what the replay decided.
SUMMARY_DROPPED = ("hash_status", "league_coverage")
CANDIDATE_SUMMARY_DROPPED = ("registered", "leagues")


def current_rule_sha256(rule: str) -> str:
    """The rule's hash under this checkout's code, wherever this revision defines it."""
    try:
        from odds_lambda.candidate_rules.pinning import rule_sha256
    except ImportError:
        from odds_lambda.candidate_rules import rule_sha256  # type: ignore[attr-defined,no-redef]
    return rule_sha256(rule)


def load(path: Path) -> MechanicalCandidate:
    """The candidate file parsed by this checkout's model: a revision before pooled leagues
    names its one league ``sport``, a later one lists ``sports``."""
    raw = yaml.safe_load(path.read_text())
    if "sports" in MechanicalCandidate.model_fields:
        if "sport" in raw:
            raw["sports"] = [raw.pop("sport")]
    elif "sports" in raw:
        (raw["sport"],) = raw.pop("sports")
    return MechanicalCandidate.model_validate(raw)


def leagues(candidate: MechanicalCandidate) -> list[str]:
    raw: Any = candidate
    return list(raw.sports) if hasattr(raw, "sports") else [raw.sport]


def in_league(candidate: MechanicalCandidate, league: str) -> MechanicalCandidate:
    """``candidate`` restricted to ``league``, whose development days differ from another's."""
    if "sports" in MechanicalCandidate.model_fields:
        return candidate.model_copy(update={"sports": (league,)})
    return candidate


def _summary(result: object) -> dict[str, Any]:
    summary = dataclasses.asdict(result)  # type: ignore[call-overload]
    for key in SUMMARY_DROPPED:
        summary.pop(key, None)
    for key in CANDIDATE_SUMMARY_DROPPED:
        summary["summary"].pop(key, None)
    return summary


async def dump(
    paths: list[Path],
    start: date,
    end: date,
    snapshot: str | None,
    segments: dict[str, list[tuple[date, date]]],
) -> dict[str, object]:
    """The replay of each candidate over each league's segments (``start``..``end`` when it has
    none), so a sealed holdout window is never replayed."""
    out: dict[str, object] = {}
    async with async_session_maker() as session:
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        if snapshot is not None:
            await session.execute(text(f"SET TRANSACTION SNAPSHOT '{snapshot}'"))
        for path in paths:
            candidate = load(path)
            spec, params = resolve_rule(candidate)
            counts = _Counts()
            bets = []
            summaries = []
            for league in leagues(candidate):
                single = in_league(candidate, league)
                for first, last in segments.get(league, [(start, end)]):
                    events = await final_events(session, league, first, last)
                    bets += await _decide(session, single, spec, params, events, counts)
                    summaries.append(
                        _summary(await replay(session, single, first, last, 1, now=NOW))
                    )
            out[candidate.name] = {
                "counts": dataclasses.asdict(counts),
                "bets": [{f: getattr(b, f) for f in BET_FIELDS} for b in bets],
                "summary": summaries,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="end", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", help="An exported snapshot id to read")
    parser.add_argument(
        "--segments",
        default="{}",
        help="JSON of sport to [from, to] day ranges to replay in place of --from to --to",
    )
    parser.add_argument("--rule", help="Also write this rule's hash to --hash-out")
    parser.add_argument("--hash-out", type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.rule is not None:
        args.hash_out.write_text(current_rule_sha256(args.rule))
    segments = {
        sport: [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in ranges]
        for sport, ranges in json.loads(args.segments).items()
    }
    result = asyncio.run(dump(args.paths, args.start, args.end, args.snapshot, segments))
    args.out.write_text(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
