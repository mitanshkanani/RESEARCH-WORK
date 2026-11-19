"""EXP-14 extra check (not in Part A): the stream joining rules compared at the SAME settings.

In the main reading each rule picks its own smoothing and handicap in the rehearsal, so part of a
rule's gain over "never" can come from a different handicap rather than from joining. Here every
rule is read at the settings chosen for "never" (and, as a second view, at each handicap from 0 to
0.2), on the test, per seed; the gain over "never" is then paired by seed.

Run:  python src/extra_matched.py   ->  results/extra_matched.json
"""
import json
import pathlib
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def hm(o, n):
    return np.where(o + n > 0, 2 * o * n / np.maximum(o + n, 1e-12), 0.0)


def main():
    meta = json.loads((RES / "meta.json").read_text())
    summ = json.loads((RES / "summary.json").read_text())
    G1, pols = meta["g1"], list(meta["policies"])
    sessions = list(range(1, meta["sessions"] + 1))
    stream = defaultdict(dict)
    for line in (RES / "runs.jsonl").read_text().splitlines():
        if '"stream"' not in line:
            continue
        r = json.loads(line)
        if r.get("kind") == "stream" and r["task"] == "part1":
            stream[(r["policy"], r["gamma"], r["session"])][r["seed"]] = r
    seeds = sorted(stream[(pols[0], meta["gammas"][0], 1)])

    def score(pol, g, j):
        """Per seed: stream score (harmonic mean averaged over sessions) and per-session H at handicap index j."""
        o = np.array([[np.mean(stream[(pol, g, s)][sd]["old"], 0)[j] / 100 for s in sessions] for sd in seeds])
        n = np.array([[np.mean(stream[(pol, g, s)][sd]["new"], 0)[j] / 100 for s in sessions] for sd in seeds])
        H = hm(o, n)
        return H.mean(1), H, o.mean(1)

    nv = summ["stream"]["never"]["handicap"]
    g, j = nv["gamma"], G1.index(nv["h"])
    base, base_s, _ = score("never", g, j)
    at_never = {}
    for p in pols:
        sc, ss, old = score(p, g, j)
        d = sc - base
        at_never[p] = dict(score=round(float(sc.mean()), 2), old=round(float(old.mean()), 2),
                           gain_vs_never=round(float(d.mean()), 2), gain_sd=round(float(d.std()), 2),
                           seeds_positive=int((d > 0).sum()),
                           sessions=[round(float(x), 2) for x in ss.mean(0)])
    sweep = {}
    for h in [0.0, 0.05, 0.1, 0.15, 0.2]:
        jj = G1.index(h)
        b, _, _ = score("never", g, jj)
        sweep[str(h)] = {p: round(float((score(p, g, jj)[0] - b).mean()), 2) for p in pols}
    out = dict(note="extra check, not in Part A; test rows; every rule at the never rule's settings",
               gamma=g, h=nv["h"], at_never_settings=at_never, gain_vs_never_by_handicap=sweep)
    (RES / "extra_matched.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
