import bz2
import collections
import json
import sys
from datetime import UTC, datetime

p = sys.argv[1]
L = [json.loads(l) for l in bz2.open(p, "rt")]
keys = collections.Counter()
mk = {}
for m in L:
    for mc in m.get("mc", []):
        md = mc.get("marketDefinition")
        if md:
            mk[mc["id"]] = md
        for r in mc.get("rc", []):
            keys.update(r.keys())
        keys.update("mc." + k for k in mc.keys())
print(keys)
types = collections.Counter(md.get("marketType") for md in mk.values())
print(len(mk), "markets", dict(types))
print(
    set(
        (md["eventName"], md["marketTime"], md.get("name"))
        for md in mk.values()
        if md.get("marketType") == "MATCH_ODDS"
    )
)
mo = [i for i, md in mk.items() if md.get("marketType") == "MATCH_ODDS"][0]
md = mk[mo]
print([(r["id"], r["name"], r["status"]) for r in md["runners"]])
ko = datetime.fromisoformat(md["marketTime"].replace("Z", "+00:00")).timestamp() * 1000
rows = []
for m in L:
    for mc in m.get("mc", []):
        if mc["id"] == mo:
            if "marketDefinition" in mc:
                print(
                    "def",
                    datetime.fromtimestamp(m["pt"] / 1000, UTC),
                    mc["marketDefinition"]["status"],
                    mc["marketDefinition"]["inPlay"],
                )
            for r in mc.get("rc", []):
                rows.append((m["pt"], r["id"], r.get("ltp"), r.get("tv")))
print(len(rows), "MO price updates")
ts = sorted(set(r[0] for r in rows))
print(
    "first",
    datetime.fromtimestamp(ts[0] / 1000, UTC),
    "last",
    datetime.fromtimestamp(ts[-1] / 1000, UTC),
    "ko",
    datetime.fromtimestamp(ko / 1000, UTC),
)
import statistics

d = [(b - a) / 1000 for a, b in zip(ts, ts[1:])]
print("gap median/min/max s", statistics.median(d), min(d), max(d))
for h in [1440, 720, 360, 120, 60, 30, 10, 5, 1]:
    t = ko - h * 60000
    out = {}
    for pt, sel, ltp, tv in rows:
        if pt <= t and ltp is not None:
            out[sel] = (ltp, round((t - pt) / 60000, 1))
    print(f"T-{h}m", out)
