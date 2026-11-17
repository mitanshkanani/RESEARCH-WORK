"""Build the EXP-10 summary, the verdict and the figures from the real result file.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes results/summary.json and figures/*.svg. Handicap values are chosen on the rehearsal rows only
(Part 1: memory after stage 6, stages 7-8 as pretend-new, named among 80; stream: memory after stage 4,
stages 5-6 as pretend-new, named among 60), by the highest harmonic mean, as fixed in JOURNAL.md.
Harmonic means are computed per seed, then averaged.
"""
import json
import pathlib
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
BLUE, RED, GREEN, AMBER, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a63d2"
INK, INK2, MUTED, GRID, AXIS, SURF, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#b9b8b0"
EYES = ["P", "F"]
ENAME = {"P": "patch eyes (no backprop)", "F": "frozen backprop eyes"}
CORR = ["C0", "C1", "C2", "C3", "C4"]
CNAME = {"C0": "none (EXP-9)", "C1": "handicap", "C2": "certainty", "C3": "certainty +\nhandicap", "C4": "new classes in\nview + handicap"}
CCOL = {"C0": NEUTRAL, "C1": BLUE, "C2": AMBER, "C3": RED, "C4": GREEN}
EXP9 = {"old": 52.21, "new": 7.05, "old_80": 53.26}

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "lines.markersize": 5,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def hm(o, n):
    o, n = np.asarray(o, float), np.asarray(n, float)
    return np.where(o + n > 0, 2 * o * n / np.maximum(o + n, 1e-12), 0.0)


def ms(v, nd=2):
    v = np.asarray(v, float)
    return [round(float(v.mean()), nd), round(float(v.std()), nd)]


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "runs.jsonl"
    rows = [json.loads(l) for l in open(src)]
    meta_p = src.parent / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    G = {"C1": meta.get("g1"), "C4": meta.get("g1"), "C3": meta.get("g3")}
    T = defaultdict(list)          # (eyes, task, shots, corr) -> rows sorted by seed
    S = defaultdict(list)          # (eyes, session, corr) -> rows
    ref, costs = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["task"] == "cost":
            costs[r["eyes"]].append(r)
        elif r["task"] == "part1_ref_all":
            ref[r["eyes"]].append(r)
        elif r["task"] == "stream":
            S[(r["eyes"], r["session"], r["corr"])].append(r)
        else:
            T[(r["eyes"], r["task"], r["shots"], r["corr"])].append(r)
    for d in (T, S):
        for v in d.values():
            v.sort(key=lambda r: r["seed"])
    seeds = len(costs["P"])
    shots = sorted({k[2] for k in T if k[1] == "part1"})
    sessions = sorted({k[1] for k in S})

    def arr(recs, key):
        """[seeds, n_handicaps] in percent (n_handicaps = 1 when there is no handicap)."""
        return np.array([[x] if isinstance(x, (int, float)) else x for x in (r[key] for r in recs)], float) * 100

    def choose(e, task, k, c):
        recs = T[(e, task, k, c)]
        return int(np.argmax(hm(arr(recs, "old"), arr(recs, "new")).mean(0)))

    def read(recs, i):
        o, n = arr(recs, "old")[:, i], arr(recs, "new")[:, i]
        out = dict(old=ms(o), new=ms(n), H=ms(hm(o, n)))
        if "new_only" in recs[0]:
            out["new_only"] = ms([r["new_only"] * 100 for r in recs])
        return out

    s = {"source": src.name, "seeds": seeds, "shots": shots}
    chosen, part1, reh = {}, {}, {}
    for e in EYES:
        chosen[e], part1[e], reh[e] = {}, {}, {}
        for c in CORR:
            chosen[e][c], part1[e][c], reh[e][c] = {}, {}, {}
            for k in shots:
                i = choose(e, "rehearsal", k, c)
                chosen[e][c][str(k)] = G[c][i] if c in G else None
                reh[e][c][str(k)] = read(T[(e, "rehearsal", k, c)], i)
                part1[e][c][str(k)] = read(T[(e, "part1", k, c)], i)
    sel = {e: max(CORR, key=lambda c: reh[e][c]["3"]["H"][0]) for e in EYES}
    s.update(chosen_handicap=chosen, selected=sel, rehearsal=reh, part1=part1)
    s["part1_ref_all"] = {e: dict(old=ms([r["old"] * 100 for r in ref[e]]), new=ms([r["new"] * 100 for r in ref[e]]),
                                  H=ms(hm([r["old"] for r in ref[e]], [r["new"] for r in ref[e]]) * 100)) for e in EYES}
    # best possible handicap on the test itself (diagnostic only, never used to choose)
    s["part1_best_handicap_on_test_3"] = {}
    for e in EYES:
        s["part1_best_handicap_on_test_3"][e] = {}
        for c in ("C1", "C3", "C4"):
            recs = T[(e, "part1", 3, c)]
            h = hm(arr(recs, "old"), arr(recs, "new")).mean(0)
            i = int(np.argmax(h))
            s["part1_best_handicap_on_test_3"][e][c] = dict(handicap=G[c][i], H=round(float(h[i]), 2))

    # ---- stream: handicap from the stream rehearsal (5 pictures)
    k_s = next(k[2] for k in T if k[1] == "stream_rehearsal")
    stream, s_chosen = {}, {}
    for e in EYES:
        stream[e], s_chosen[e] = {}, {}
        base0 = [r["old"] * 100 for r in S[(e, 0, "C0")]]
        for c in CORR:
            i = choose(e, "stream_rehearsal", k_s, c)
            s_chosen[e][c] = G[c][i] if c in G else None
            curve = [dict(session=0, all=ms(base0), old=ms(base0))]
            for ss in sessions[1:]:
                recs = S[(e, ss, c)]
                o, n = arr(recs, "old")[:, i], arr(recs, "new")[:, i]
                no, nn = np.array([r["n_old_test"] for r in recs]), np.array([r["n_new_test"] for r in recs])
                al = (o * no + n * nn) / (no + nn)
                curve.append(dict(session=ss, n_classes=recs[0]["n_classes"], all=ms(al), old=ms(o), new=ms(n), H=ms(hm(o, n))))
            stream[e][c] = curve
    s.update(stream_handicap=s_chosen, stream=stream)

    def cst(e):
        out = {}
        for k in ("build_seconds", "train_seconds", "peak_mem_mb", "train_wh", "build_wh"):
            if all(r.get(k) is not None for r in costs[e]):
                out[k] = ms([r[k] for r in costs[e]], 3)
        out["seconds"] = {k: ms([r["seconds"][k] for r in costs[e]], 2) for k in costs[e][0]["seconds"]}
        if all(r.get("wh") for r in costs[e]):
            out["wh"] = {k: ms([r["wh"][k] for r in costs[e]], 4) for k in costs[e][0]["wh"]}
        out["feature_dim"] = costs[e][0]["feature_dim"]
        return out
    s["cost"] = {e: cst(e) for e in EYES}

    # ---- verdict, by JOURNAL.md section 5
    p = part1["P"]
    g3 = lambda c, key: p[c]["3"][key][0]
    v = {"step0_rebuild": dict(old=[g3("C0", "old"), EXP9["old"]], new=[g3("C0", "new"), EXP9["new"]])}
    v["step0_passed"] = abs(g3("C0", "old") - EXP9["old"]) <= 1.5 and abs(g3("C0", "new") - EXP9["new"]) <= 1.5
    c = sel["P"]
    h, o = g3(c, "H"), g3(c, "old")
    v["step1"] = dict(selected=c, handicap=chosen["P"][c]["3"], old=o, new=g3(c, "new"), H=h, new_only=g3(c, "new_only"),
                      verdict="works" if h >= 30 and o >= 48.3 else ("helps, but not enough" if h >= 20 and o >= 48.3 else "does not work"))
    v["step2"] = {cc: dict(H=g3(cc, "H"), gain=round(g3(cc, "H") - g3("C0", "H"), 2), counts=g3(cc, "H") - g3("C0", "H") >= 3)
                  for cc in CORR[1:]}
    v["step3"] = dict(C4=g3("C4", "H"), C1=g3("C1", "H"), gain=round(g3("C4", "H") - g3("C1", "H"), 2),
                      verdict="yes" if g3("C4", "H") - g3("C1", "H") >= 2 else "no")
    last = lambda cc: stream["P"][cc][-1]
    v["step4"] = dict(selected=c, sel_all=last(c)["all"][0], C0_all=last("C0")["all"][0],
                      gain=round(last(c)["all"][0] - last("C0")["all"][0], 2),
                      verdict="helps the stream" if last(c)["all"][0] - last("C0")["all"][0] >= 3 else "does not help the stream",
                      start_old=stream["P"][c][0]["old"][0], end_old=last(c)["old"][0], end_new=last(c)["new"][0])
    best = max(CORR, key=lambda cc: g3(cc, "H"))
    v["step5"] = dict(selected=c, selected_H=h, best_on_test=best, best_H=g3(best, "H"), trusted=g3(best, "H") - h <= 1.5)
    s["verdict"] = v
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(v, indent=1))

    # ---- figure 1: Part 1, 3 pictures, every correction
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, e in zip(axes, EYES):
        x = np.arange(len(CORR))
        wd = 0.27
        for j, (key, lab, col) in enumerate((("old", "old classes", NEUTRAL), ("new", "new classes", RED), ("H", "harmonic mean", BLUE))):
            vals = [part1[e][cc]["3"][key][0] for cc in CORR]
            ax.bar(x + (j - 1) * wd, vals, width=wd, color=col, label=lab)
            for xi, val in zip(x, vals):
                ax.text(xi + (j - 1) * wd, val + 0.8, f"{val:.0f}", ha="center", fontsize=7)
        ax.set_xticks(x, [CNAME[cc] for cc in CORR], fontsize=7.5)
        ax.grid(False, axis="x")
        ax.set_title(ENAME[e])
    axes[0].set_ylabel("% among all 100 classes, 3 pictures per new class")
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig1_corrections")

    # ---- figure 2: the trade-off as the handicap varies (Part 1, patch eyes, 3 pictures)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for cc in ("C1", "C3", "C4"):
        recs = T[("P", "part1", 3, cc)]
        o, n = arr(recs, "old").mean(0), arr(recs, "new").mean(0)
        ax.plot(o, n, color=CCOL[cc], lw=1.6, label=CNAME[cc].replace("\n", " "))
        ch = part1["P"][cc]["3"]
        ax.scatter([ch["old"][0]], [ch["new"][0]], s=55, facecolor="none", edgecolor=CCOL[cc], lw=1.5, zorder=3)
    for cc in ("C0", "C2"):
        ch = part1["P"][cc]["3"]
        ax.scatter([ch["old"][0]], [ch["new"][0]], s=40, color=CCOL[cc], zorder=3, label=CNAME[cc].replace("\n", " "))
    rf = s["part1_ref_all"]["P"]
    ax.scatter([rf["old"][0]], [rf["new"][0]], marker="*", s=110, color=VIOLET, zorder=3, label="new classes from all pictures (reference)")
    ax.axvline(48.3, color=AXIS, ls=":", lw=1)
    ax.text(48.6, 1, "old 48.3%\n(rule limit)", fontsize=7, color=MUTED)
    ax.set_xlabel("old classes % (among all 100)")
    ax.set_ylabel("new classes % (among all 100)")
    ax.set_title("patch eyes, 3 pictures: old against new as the handicap varies")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig2_tradeoff")

    # ---- figure 3: harmonic mean against pictures per new class
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, e in zip(axes, EYES):
        for cc in CORR:
            ax.plot(shots, [part1[e][cc][str(k)]["H"][0] for k in shots], marker="o", color=CCOL[cc], label=CNAME[cc].replace("\n", " "))
        ax.set_xscale("log")
        ax.set_xticks(shots, [str(k) for k in shots])
        ax.set_xlabel("pictures per new class")
        ax.set_title(ENAME[e])
    axes[0].set_ylabel("harmonic mean of old and new %")
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig3_pictures")

    # ---- figure 4: the stream
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, e in zip(axes, EYES):
        for cc in CORR:
            ax.plot(sessions, [pt["all"][0] for pt in stream[e][cc]], marker="o", color=CCOL[cc], label=CNAME[cc].replace("\n", " "))
        ax.set_xticks(sessions, [f"{s_}\n{60 + 5 * s_}" for s_ in sessions], fontsize=7.5)
        ax.set_xlabel("session / classes known")
        ax.set_title(ENAME[e])
    axes[0].set_ylabel("accuracy over all classes seen %")
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig4_stream")

    # ---- figure 5: stream, old against new for the selected correction and C0 (patch eyes)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for cc, ls in (("C0", ":"), (sel["P"], "-")):
        ax.plot(sessions[1:], [pt["old"][0] for pt in stream["P"][cc][1:]], color=NEUTRAL, ls=ls, marker="o", label=f"old classes, {CNAME[cc].splitlines()[0]}")
        ax.plot(sessions[1:], [pt["new"][0] for pt in stream["P"][cc][1:]], color=RED, ls=ls, marker="o", label=f"new classes, {CNAME[cc].splitlines()[0]}")
    ax.set_xlabel("session")
    ax.set_ylabel("% among all classes seen")
    ax.set_title("patch eyes, stream: old and new classes")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig5_stream_old_new")


if __name__ == "__main__":
    main()
