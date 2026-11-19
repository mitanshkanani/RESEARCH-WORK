"""Build the EXP-11 summary, the verdict and the figures from the real result file.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes results/summary.json and figures/*.svg. Every setting is chosen on the rehearsal rows only (memory
after stage 6, stages 7-8 as pretend-new, named among 80), by the rules fixed in JOURNAL.md section 3:
  T1 smoothing: best 60-class accuracy (distance through the smoothed table)
  T2 smoothing: best pretend-new accuracy among themselves (C0, angle)
  T3 setting:   over smoothing x (C1, C3) x handicap, keep old >= C0 old (same smoothing) - 5, take best H
  Selected eyes: best rehearsal T3 H at 3 pictures among versions with peak GPU memory <= 3.5 GB.
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
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]
ANAME = {"A0": "A0 EXP-10 eyes", "A1": "A1 + mirror", "A2": "A2 + two-sided", "A3": "A3 + 3x3 regions",
         "A4": "A4 + 3,200 stencils", "A5": "A5 all three, 800 st."}
ACOL = {"A0": NEUTRAL, "A1": BLUE, "A2": AMBER, "A3": GREEN, "A4": RED, "A5": VIOLET}
EXP10 = {"old": 52.2, "new": 7.02, "new_only": 33.54}
EXP8_T1 = 55.2
MEM_LIMIT_MB = 3.5 * 1024
LIMIT = 5.0

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
    gammas = meta.get("gammas", [0.01, 0.03, 0.1, 0.3, 1.0])
    GRID_OF = {"C1": meta["g1"], "C3": meta["g3"]}
    arms = [a for a in ARMS if any(r["eyes"] == a for r in rows)]
    T1, C, REF, COST = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        if r["task"] == "cost":
            COST[r["eyes"]].append(r)
        elif r.get("kind") == "ref":
            REF[(r["eyes"], r["gamma"])].append(r)
        elif r.get("kind") == "t1":
            T1[(r["eyes"], r["task"], r["gamma"])].append(r)
        else:
            C[(r["eyes"], r["task"], r["gamma"], r["shots"], r["corr"])].append(r)
    for d in (T1, C, REF):
        for v in d.values():
            v.sort(key=lambda r: r["seed"])
    seeds = len(COST[arms[0]])
    shots = sorted({k[3] for k in C})

    def arr(recs, key):
        """[seeds, n_handicaps] in percent."""
        return np.array([[x] if isinstance(x, (int, float)) else x for x in (r[key] for r in recs)], float) * 100

    s = {"source": src.name, "seeds": seeds, "shots": shots, "gammas": gammas}
    out = {}
    for a in arms:
        o = {}
        # ---- T1
        reh = {g: np.mean([r["lda"] for r in T1[(a, "rehearsal", g)]]) for g in gammas}
        g1 = max(gammas, key=lambda g: reh[g])
        o["T1"] = dict(gamma=g1, rehearsal=round(reh[g1] * 100, 2), test=ms([r["lda"] * 100 for r in T1[(a, "part1", g1)]]),
                       test_angle=ms([r["angle"] * 100 for r in T1[(a, "part1", g1)]]),
                       by_gamma={str(g): ms([r["lda"] * 100 for r in T1[(a, "part1", g)]]) for g in gammas},
                       rehearsal_by_gamma={str(g): round(reh[g] * 100, 2) for g in gammas})
        o["T2"], o["T3"], o["ref"] = {}, {}, {}
        for k in shots:
            # ---- T2
            rh = {g: np.mean([r["new_only"] for r in C[(a, "rehearsal", g, k, "C0")]]) for g in gammas}
            g2 = max(gammas, key=lambda g: rh[g])
            o["T2"][str(k)] = dict(gamma=g2, rehearsal=round(rh[g2] * 100, 2),
                                   test=ms([r["new_only"] * 100 for r in C[(a, "part1", g2, k, "C0")]]),
                                   by_gamma={str(g): ms([r["new_only"] * 100 for r in C[(a, "part1", g, k, "C0")]]) for g in gammas})
            # ---- T3: candidates within the old-class limit in the rehearsal
            best = None
            for g in gammas:
                c0_old = arr(C[(a, "rehearsal", g, k, "C0")], "old")[:, 0].mean()
                for corr in ("C1", "C3"):
                    recs = C[(a, "rehearsal", g, k, corr)]
                    ol, nw = arr(recs, "old"), arr(recs, "new")
                    h = hm(ol, nw).mean(0)
                    ok = ol.mean(0) >= c0_old - LIMIT
                    for i in np.nonzero(ok)[0]:
                        if best is None or h[i] > best[0]:
                            best = (float(h[i]), g, corr, int(i), float(ol.mean(0)[i]), float(c0_old))
            h_r, g3, corr, i, old_r, c0_r = best
            recs = C[(a, "part1", g3, k, corr)]
            ol, nw = arr(recs, "old")[:, i], arr(recs, "new")[:, i]
            c0t = C[(a, "part1", g3, k, "C0")]
            c2t = C[(a, "part1", g3, k, "C2")]
            o["T3"][str(k)] = dict(
                gamma=g3, corr=corr, handicap=GRID_OF[corr][i], rehearsal=dict(H=round(h_r, 2), old=round(old_r, 2), C0_old=round(c0_r, 2)),
                old=ms(ol), new=ms(nw), H=ms(hm(ol, nw)),
                C0=dict(old=ms(arr(c0t, "old")[:, 0]), new=ms(arr(c0t, "new")[:, 0]), H=ms(hm(arr(c0t, "old")[:, 0], arr(c0t, "new")[:, 0]))),
                C2=dict(old=ms(arr(c2t, "old")[:, 0]), new=ms(arr(c2t, "new")[:, 0]), H=ms(hm(arr(c2t, "old")[:, 0], arr(c2t, "new")[:, 0]))))
            if k == 3:
                rf = REF[(a, g3)]
                ro, rn = np.array([r["old"] for r in rf]) * 100, np.array([r["new"] for r in rf]) * 100
                o["ref"] = dict(gamma=g3, old=ms(ro), new=ms(rn), H=ms(hm(ro, rn)),
                                ratio=round(float(np.mean(hm(ol, nw)) / np.mean(hm(ro, rn))), 3))
        cs = COST[a]
        o["cost"] = dict(feature_dim=cs[0]["feature_dim"], peak_mem_mb=max(r["peak_mem_mb"] for r in cs),
                         peak_reserved_mb=max(r["peak_reserved_mb"] for r in cs),
                         cpu_peak_mb=max((r["cpu_peak_mb"] or 0) for r in cs) or None,
                         seconds={k: ms([r["seconds"][k] for r in cs], 1) for k in cs[0]["seconds"]},
                         total_seconds=ms([sum(r["seconds"].values()) for r in cs], 1))
        if all(r.get("wh") for r in cs):
            o["cost"]["wh"] = ms([sum(v for v in r["wh"].values() if v) for r in cs], 3)
        out[a] = o
    s["arms"] = out

    # ---- verdict, by JOURNAL.md section 5
    v = {}
    a0 = out["A0"]
    c0_01 = C[("A0", "part1", 0.1, 3, "C0")]
    st0 = dict(old=ms(arr(c0_01, "old")[:, 0]), new=ms(arr(c0_01, "new")[:, 0]), new_only=ms([r["new_only"] * 100 for r in c0_01]),
               t1_at_0_01=a0["T1"]["by_gamma"]["0.01"])
    st0["passed"] = bool(abs(st0["old"][0] - EXP10["old"]) <= 0.5 and abs(st0["new"][0] - EXP10["new"]) <= 0.5
                         and abs(st0["new_only"][0] - EXP10["new_only"]) <= 0.5 and abs(st0["t1_at_0_01"][0] - EXP8_T1) <= 1.0)
    v["step0"] = st0
    allowed = [a for a in arms if out[a]["cost"]["peak_mem_mb"] <= MEM_LIMIT_MB]
    selected = max(allowed, key=lambda a: out[a]["T3"]["3"]["rehearsal"]["H"])
    S_ = out[selected]
    t1g = S_["T1"]["test"][0] - a0["T1"]["test"][0]
    t2g = S_["T2"]["3"]["test"][0] - a0["T2"]["3"]["test"][0]
    t3g = S_["T3"]["3"]["H"][0] - a0["T3"]["3"]["H"][0]
    old_ok = S_["T3"]["3"]["old"][0] >= S_["T3"]["3"]["C0"]["old"][0] - 6
    cond = dict(a=t1g >= 2.0, b=t2g >= 2.0, c=bool(t3g >= 3.0 and old_ok), d=S_["cost"]["peak_mem_mb"] <= MEM_LIMIT_MB)
    n_abc = sum(cond[x] for x in "abc")
    verdict = "works" if n_abc == 3 and cond["d"] else ("helps, but not enough" if cond["d"] and n_abc >= 1 else "does not work")
    v["step1"] = dict(selected=selected, allowed=allowed, T1_gain=round(t1g, 2), T2_gain=round(t2g, 2), T3_gain=round(t3g, 2),
                      T3_old=S_["T3"]["3"]["old"][0], T3_C0_old=S_["T3"]["3"]["C0"]["old"][0], old_within_6=bool(old_ok),
                      peak_mem_mb=S_["cost"]["peak_mem_mb"], conditions=cond, verdict=verdict)
    pairs = [("A1", "A0", "mirror"), ("A2", "A1", "two-sided"), ("A3", "A1", "3x3 regions"), ("A4", "A1", "3,200 stencils"),
             ("A5", "A1", "all three, 800 stencils")]
    st2 = {}
    for x, y, lab in pairs:
        if x in out and y in out:
            g = dict(T1=round(out[x]["T1"]["test"][0] - out[y]["T1"]["test"][0], 2),
                     T2=round(out[x]["T2"]["3"]["test"][0] - out[y]["T2"]["3"]["test"][0], 2),
                     T3=round(out[x]["T3"]["3"]["H"][0] - out[y]["T3"]["3"]["H"][0], 2))
            st2[f"{x}-{y}"] = dict(upgrade=lab, **g, counts={t: g[t] >= 1.0 for t in g})
    v["step2"] = st2
    v["step3"] = dict(selected_ratio=S_["ref"]["ratio"], A0_ratio=a0["ref"]["ratio"],
                      rise=round(S_["ref"]["ratio"] - a0["ref"]["ratio"], 3),
                      verdict="the gap shrinks" if S_["ref"]["ratio"] - a0["ref"]["ratio"] >= 0.05 else "the gap does not shrink")
    best = max(arms, key=lambda a: out[a]["T3"]["3"]["H"][0])
    v["step4"] = dict(selected=selected, selected_H=S_["T3"]["3"]["H"][0], best_on_test=best, best_H=out[best]["T3"]["3"]["H"][0],
                      trusted=bool(out[best]["T3"]["3"]["H"][0] - S_["T3"]["3"]["H"][0] <= 1.0))
    s["verdict"] = v
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(v, indent=1))
    for a in arms:
        o = out[a]
        print(f"{a}: T1 {o['T1']['test'][0]:.1f} (g {o['T1']['gamma']}) | T2 {o['T2']['3']['test'][0]:.1f} (g {o['T2']['3']['gamma']}) | "
              f"T3 H {o['T3']['3']['H'][0]:.1f} old {o['T3']['3']['old'][0]:.1f} new {o['T3']['3']['new'][0]:.1f} "
              f"({o['T3']['3']['corr']} g {o['T3']['3']['gamma']} h {o['T3']['3']['handicap']}) | ref H {o['ref']['H'][0]:.1f} "
              f"ratio {o['ref']['ratio']} | mem {o['cost']['peak_mem_mb']:.0f} MB")

    # ---- figure 1: the three tests, every version
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    tests = [("T1", "T1: 80 classes, all pictures", lambda o: o["T1"]["test"]),
             ("T2", "T2: 20 new from 3 pictures, among themselves", lambda o: o["T2"]["3"]["test"]),
             ("T3", "T3: 20 new (3 pictures) among 100,\nharmonic mean, old kept within the limit", lambda o: o["T3"]["3"]["H"])]
    for ax, (key, title, get) in zip(axes, tests):
        vals = [get(out[a]) for a in arms]
        x = np.arange(len(arms))
        ax.bar(x, [m for m, _ in vals], yerr=[sd for _, sd in vals], color=[ACOL[a] for a in arms], width=0.7,
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for xi, (m, _) in zip(x, vals):
            ax.text(xi, m * 0.5, f"{m:.1f}", ha="center", fontsize=7.5, color="white" if xi else INK, rotation=90)
        lo = min(m for m, _ in vals)
        ax.set_ylim(max(0, lo - 8), max(m for m, _ in vals) + 2)
        ax.set_xticks(x, arms)
        ax.grid(False, axis="x")
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("% (5 seeds, test)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACOL[a]) for a in arms]
    fig.legend(handles, [ANAME[a] for a in arms], loc="lower center", ncol=6, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig1_three_tests")

    # ---- figure 2: old against new as the handicap varies (3 pictures, C1 at each version's T3 smoothing)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for a in arms:
        t3 = out[a]["T3"]["3"]
        recs = C[(a, "part1", t3["gamma"], 3, t3["corr"])]
        ax.plot(arr(recs, "old").mean(0), arr(recs, "new").mean(0), color=ACOL[a], lw=1.5, label=ANAME[a])
        ax.scatter([t3["old"][0]], [t3["new"][0]], s=50, facecolor="none", edgecolor=ACOL[a], lw=1.5, zorder=3)
        ax.scatter([out[a]["ref"]["old"][0]], [out[a]["ref"]["new"][0]], marker="*", s=90, color=ACOL[a], zorder=3)
    ax.set_xlabel("old classes % (among all 100)")
    ax.set_ylabel("new classes % (among all 100)")
    ax.set_title("3 pictures: old against new as the handicap varies")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig2_tradeoff")

    # ---- figure 3: T2 and T3 against pictures per new class
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for a in arms:
        axes[0].plot(shots, [out[a]["T2"][str(k)]["test"][0] for k in shots], marker="o", color=ACOL[a], label=ANAME[a])
        axes[1].plot(shots, [out[a]["T3"][str(k)]["H"][0] for k in shots], marker="o", color=ACOL[a], label=ANAME[a])
    for ax, t in zip(axes, ("T2: new among themselves %", "T3: harmonic mean among 100")):
        ax.set_xscale("log")
        ax.set_xticks(shots, [str(k) for k in shots])
        ax.set_xlabel("pictures per new class")
        ax.set_title(t)
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig3_pictures")

    # ---- figure 4: T1 against smoothing
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for a in arms:
        ax.plot(gammas, [out[a]["T1"]["by_gamma"][str(g)][0] for g in gammas], marker="o", color=ACOL[a], label=ANAME[a])
    ax.set_xscale("log")
    ax.set_xticks(gammas, [str(g) for g in gammas])
    ax.set_xlabel("smoothing of the shared table")
    ax.set_ylabel("80-class accuracy % (test)")
    ax.set_title("T1 against smoothing")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig4_smoothing")

    # ---- figure 5: cost
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    x = np.arange(len(arms))
    axes[0].bar(x, [out[a]["cost"]["peak_mem_mb"] / 1024 for a in arms], color=[ACOL[a] for a in arms], width=0.7)
    axes[0].axhline(3.5, color=RED, ls=":", lw=1)
    axes[0].text(-0.4, 3.55, "3.5 GB limit", fontsize=7, color=RED)
    axes[0].set_ylabel("peak GPU memory, GB")
    axes[1].bar(x, [out[a]["cost"]["total_seconds"][0] for a in arms], color=[ACOL[a] for a in arms], width=0.7)
    axes[1].set_ylabel("seconds per seed")
    for ax, t in zip(axes, ("peak GPU memory", "time per seed (all settings)")):
        ax.set_xticks(x, arms)
        ax.grid(False, axis="x")
        ax.set_title(t)
    save(fig, "fig5_cost")


if __name__ == "__main__":
    main()
