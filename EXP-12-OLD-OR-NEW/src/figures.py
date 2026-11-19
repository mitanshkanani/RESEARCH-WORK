"""Build the EXP-12 summary, the verdict and the figures from the real result file.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes results/summary.json and figures/*.svg. Every setting is chosen on the rehearsal rows only (memory after
stage 6, stages 7-8 as pretend-new, named among 80), by the rules fixed in JOURNAL.md section 3:
  for each switch and each number of pictures, over smoothing x settings, keep rehearsal old >= no-correction old
  (S0 at h = 0, same smoothing) - 5, and take the highest rehearsal harmonic mean.
  Selected switch: the highest rehearsal harmonic mean at 3 pictures among the eight familiarity switches.
Harmonic means are computed per seed (from the old and new accuracy averaged over the 20 draws), then averaged.
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
SIGNALS = ["F1", "F2", "F3", "F4"]
SNAME = {"F1": "F1 inside the view", "F2": "F2 outside the view", "F3": "F3 everything", "F4": "F4 relative"}
SCOL = {"F1": BLUE, "F2": GREEN, "F3": AMBER, "F4": VIOLET}
EXP11 = {"old": 52.47, "new": 17.28, "H": 26.0}
EXP11_PERFECT = 45.0
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
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "runs.jsonl"
    meta = json.loads((path.parent / "meta.json").read_text())
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    gammas, shots, H, Q = meta["gammas"], meta["shots"], meta["h_grid"], meta["q_grid"]
    ih0, ih8, ihx = H.index(0.0), H.index(0.08), len(H) - 1          # h = 0, h = 0.08, "no handicap test"
    iq0 = 0                                                           # "no familiarity test"

    SW = defaultdict(lambda: {"old": [], "new": []})                  # (task, gamma, k, signal) -> per seed [Q, H] in %
    OR = defaultdict(lambda: {"old": [], "new": []})
    AF, AM, DEC = defaultdict(list), defaultdict(list), defaultdict(list)
    cost = []
    for r in rows:
        kind = r.get("kind")
        if kind == "switch":
            key = (r["task"], r["gamma"], r["shots"], r["signal"])
            SW[key]["old"].append(np.array(r["old"]) / 100)
            SW[key]["new"].append(np.array(r["new"]) / 100)
        elif kind == "oracle":
            key = (r["task"], r["gamma"], r["shots"])
            OR[key]["old"].append(100 * r["old"])
            OR[key]["new"].append(100 * r["new"])
        elif kind == "auc_f":
            AF[(r["task"], r["gamma"])].append({s: r[s] for s in SIGNALS})
        elif kind == "auc_m":
            AM[(r["task"], r["gamma"], r["shots"])].append(r["margin"])
        elif kind == "deciles":
            DEC[(r["task"], r["gamma"])].append({s: r[s] for s in SIGNALS})
        elif r.get("task") == "cost":
            cost.append(r)
    get = lambda task, g, k, s: (np.stack(SW[(task, g, k, s)]["old"]), np.stack(SW[(task, g, k, s)]["new"]))

    def family_mask(fam):
        m = np.zeros((len(Q), len(H)), bool)
        if fam == "S0":
            m[iq0, :ihx] = True
        elif fam == "only":
            m[1:, ihx] = True
        else:
            m[1:, :ihx] = True
        return m

    def choose(fam, sig, k):
        """Best rehearsal harmonic mean under the old-class limit; then the test readings at that setting."""
        best = None
        for g in gammas:
            o, n = get("rehearsal", g, k, sig)
            om, hmn = o.mean(0), hm(o, n).mean(0)
            ok = family_mask(fam) & (om >= om[iq0, ih0] - LIMIT)
            if ok.any():
                v = np.where(ok, hmn, -1)
                qi, hi = np.unravel_index(v.argmax(), v.shape)
                if best is None or v[qi, hi] > best[0]:
                    best = (float(v[qi, hi]), g, int(qi), int(hi), float(om[qi, hi]), float(om[iq0, ih0]))
        _, g, qi, hi, r_old, r_c0 = best
        o, n = get("part1", g, k, sig)
        return dict(gamma=g, q=Q[qi], h=None if hi == ihx else H[hi], qi=qi, hi=hi,
                    rehearsal=dict(H=round(best[0], 2), old=round(r_old, 2), C0_old=round(r_c0, 2)),
                    old=ms(o[:, qi, hi]), new=ms(n[:, qi, hi]), H=ms(hm(o[:, qi, hi], n[:, qi, hi])),
                    C0_old=ms(o[:, iq0, ih0]))

    switches = {"S0": ("S0", "F1")}
    for s in SIGNALS:
        switches[f"{s} only"] = ("only", s)
        switches[f"{s} + h"] = ("and", s)
    out = {name: {str(k): choose(fam, sig, k) for k in shots} for name, (fam, sig) in switches.items()}

    perfect = {}
    for k in shots:
        rh = {g: hm(OR[("rehearsal", g, k)]["old"], OR[("rehearsal", g, k)]["new"]).mean() for g in gammas}
        g = max(rh, key=rh.get)
        o, n = np.array(OR[("part1", g, k)]["old"]), np.array(OR[("part1", g, k)]["new"])
        perfect[str(k)] = dict(gamma=g, rehearsal_H=round(float(rh[g]), 2), old=ms(o), new=ms(n), H=ms(hm(o, n)))

    auc = {}
    for s in SIGNALS:
        g = out[f"{s} only"]["3"]["gamma"]
        auc[s] = dict(gamma=g, test=ms([a[s] for a in AF[("part1", g)]], 3),
                      by_gamma={str(gg): ms([a[s] for a in AF[("part1", gg)]], 3) for gg in gammas})
    auc["margin"] = {str(k): dict(by_gamma={str(gg): ms(AM[("part1", gg, k)], 3) for gg in gammas}) for k in shots}

    c = cost
    peak = max(x["peak_mem_mb"] or 0 for x in c)
    cost_out = dict(feature_dim=c[0]["feature_dim"], peak_mem_mb=peak, peak_reserved_mb=max(x["peak_reserved_mb"] or 0 for x in c),
                    cpu_peak_mb=max((x["cpu_peak_mb"] or 0) for x in c),
                    seconds={k: ms([x["seconds"][k] for x in c], 1) for k in c[0]["seconds"]},
                    total_seconds=ms([sum(x["seconds"].values()) for x in c], 1),
                    wh=ms([sum(v for v in x["wh"].values() if v) for x in c], 3) if c[0]["wh"] else None)

    # ---- verdict, by JOURNAL.md section 5
    fam_names = [n for n in switches if n != "S0"]
    selected = max(fam_names, key=lambda n: out[n]["3"]["rehearsal"]["H"])
    o0, n0 = get("part1", 0.1, 3, "F1")
    step0 = dict(old=ms(o0[:, iq0, ih8]), new=ms(n0[:, iq0, ih8]))
    step0["passed"] = abs(step0["old"][0] - EXP11["old"]) <= 0.5 and abs(step0["new"][0] - EXP11["new"]) <= 0.5
    gain = {k: round(out[selected][k]["H"][0] - out["S0"][k]["H"][0], 2) for k in map(str, shots)}
    sel3 = out[selected]["3"]
    cond = dict(a=gain["3"] >= 3.0 and sel3["old"][0] >= sel3["C0_old"][0] - 6.0,
                b=gain["1"] >= 1.5 and gain["10"] >= 1.5,
                c=peak <= MEM_LIMIT_MB)
    if all(cond.values()):
        verdict = "works"
    elif cond["c"] and gain["3"] >= 1.0:
        verdict = "helps, but not enough"
    else:
        verdict = "does not work"
    p3 = perfect["3"]["H"][0]
    best_test = max(switches, key=lambda n: out[n]["3"]["H"][0])
    summary = dict(
        switches=out, perfect=perfect, auc=auc, cost=cost_out,
        verdict=dict(
            step0=step0,
            step1=dict(selected=selected, gain=gain, old=sel3["old"][0], C0_old=sel3["C0_old"][0], peak_mem_mb=peak,
                       conditions=cond, verdict=verdict),
            step2=dict(perfect_H=p3, expected=EXP11_PERFECT, confirmed=abs(p3 - EXP11_PERFECT) <= 3.0),
            step3=dict(F2_minus_F1=round(auc["F2"]["test"][0] - auc["F1"]["test"][0], 3),
                       F4_minus_F1=round(auc["F4"]["test"][0] - auc["F1"]["test"][0], 3),
                       outside_counts=auc["F2"]["test"][0] - auc["F1"]["test"][0] >= 0.02,
                       relative_counts=auc["F4"]["test"][0] - auc["F1"]["test"][0] >= 0.02),
            step4=dict(fraction_closed=round((sel3["H"][0] - out["S0"]["3"]["H"][0]) / max(p3 - out["S0"]["3"]["H"][0], 1e-9), 3)),
            step5=dict(selected_H=sel3["H"][0], best_on_test=best_test, best_H=out[best_test]["3"]["H"][0],
                       trusted=out[best_test]["3"]["H"][0] - sel3["H"][0] <= 1.0)))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["verdict"], indent=1))
    for name in switches:
        x = out[name]["3"]
        print(f"{name:8s} k=3: H {x['H'][0]:5.1f} old {x['old'][0]:5.1f} new {x['new'][0]:5.1f} | g {x['gamma']} q {x['q']} h {x['h']}"
              f" | reh H {x['rehearsal']['H']} | k=1 {out[name]['1']['H'][0]} k=10 {out[name]['10']['H'][0]}")
    print("perfect", perfect)
    print("auc", {s: auc[s]["test"] for s in SIGNALS}, "margin k=3", auc["margin"]["3"])

    # ---- figure 1: every switch at 3 pictures (test)
    names = list(switches) + ["perfect"]
    vals = [out[n]["3"]["H"] for n in switches] + [perfect["3"]["H"]]
    cols = [NEUTRAL] + [SCOL[s] for s in SIGNALS for _ in (0,)] + [SCOL[s] for s in SIGNALS] + [INK2]
    cols = [NEUTRAL] + [SCOL[n.split()[0]] for n in list(switches)[1:]] + [INK2]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(names))
    ax.bar(x, [m for m, _ in vals], yerr=[sd for _, sd in vals], color=cols, width=0.7,
           error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    for xi, n in enumerate(names):
        if "+ h" in n:
            ax.patches[xi].set_hatch("///")
            ax.patches[xi].set_edgecolor(SURF)
        m = vals[xi][0]
        ax.text(xi, m + 1.0, f"{m:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [n.replace(" + h", "\n+ handicap").replace(" only", "\nonly") for n in names], fontsize=7.5)
    ax.set_ylabel("harmonic mean, 3 pictures (test)")
    ax.set_title("Every switch, at the settings chosen in the rehearsal (old classes kept within the limit)")
    ax.grid(False, axis="x")
    save(fig, "fig1_switches")

    # ---- figure 2: how well each signal separates old from new pictures (test)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for s in SIGNALS:
        ax.plot(gammas, [auc[s]["by_gamma"][str(g)][0] for g in gammas], marker="o", color=SCOL[s], label=SNAME[s])
    for k, ls in zip(shots, (":", "--", "-")):
        ax.plot(gammas, [auc["margin"][str(k)]["by_gamma"][str(g)][0] for g in gammas], color=NEUTRAL, ls=ls, lw=1.5,
                label=f"handicap's score, {k} picture{'s' if k > 1 else ''}")
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.text(gammas[0], 0.505, "chance", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks(gammas, [str(g) for g in gammas])
    ax.set_xlabel("smoothing of the shared table")
    ax.set_ylabel("AUC: old against new pictures")
    ax.set_title("How well each signal tells new pictures from old (test)")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig2_auc")

    # ---- figure 3: old against new at 3 pictures (test, mean of seeds)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    g0 = out["S0"]["3"]["gamma"]
    o, n = get("part1", g0, 3, "F1")
    ax.plot(o.mean(0)[iq0, :ihx], n.mean(0)[iq0, :ihx], color=NEUTRAL, lw=1.8, label=f"S0 handicap (smoothing {g0})")
    sel_sig = selected.split()[0]
    gs = out[selected]["3"]["gamma"]
    o, n = get("part1", gs, 3, sel_sig)
    om, nm = o.mean(0), n.mean(0)
    fm = family_mask("and" if "+ h" in selected else "only")
    pts = np.stack([om[fm], nm[fm]], 1)
    order = np.argsort(-pts[:, 0])
    env, best_new = [], -1
    for p in pts[order]:
        if p[1] > best_new:
            env.append(p)
            best_new = p[1]
    env = np.array(env)
    ax.plot(env[:, 0], env[:, 1], color=SCOL[sel_sig], lw=1.8, label=f"{selected}: best reachable (smoothing {gs})")
    for name, col in (("S0", NEUTRAL), (selected, SCOL[sel_sig])):
        ax.scatter([out[name]["3"]["old"][0]], [out[name]["3"]["new"][0]], s=55, facecolor="none", edgecolor=col, lw=1.6, zorder=3)
    ax.scatter([perfect["3"]["old"][0]], [perfect["3"]["new"][0]], marker="*", s=110, color=INK2, zorder=3, label="perfect switch")
    ax.set_xlabel("old classes % (among all 100)")
    ax.set_ylabel("new classes % (among all 100)")
    ax.set_title("3 pictures: old against new")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig3_tradeoff")

    # ---- figure 4: 1, 3, 10 pictures
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for name, col, ls in (("S0", NEUTRAL, "-"), (selected, SCOL[sel_sig], "-")):
        ax.plot(shots, [out[name][str(k)]["H"][0] for k in shots], marker="o", color=col, ls=ls, label=name)
    ax.plot(shots, [perfect[str(k)]["H"][0] for k in shots], marker="*", color=INK2, ls="--", label="perfect switch")
    ax.set_xscale("log")
    ax.set_xticks(shots, [str(k) for k in shots])
    ax.set_xlabel("pictures per new class")
    ax.set_ylabel("harmonic mean among 100 (test)")
    ax.set_title("Harmonic mean against pictures per new class")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig4_pictures")

    # ---- figure 5: familiarity of old, new and in-hand pictures (selected signal, test, mean of seeds' deciles)
    dq = meta["deciles"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for part, col, lab in (("old", BLUE, "old classes' test pictures"), ("new", RED, "new classes' test pictures"),
                           ("pool", AMBER, "new pictures in hand (set the threshold)")):
        v = np.mean([dd[sel_sig][part] for dd in DEC[("part1", gs)]], 0)
        ax.plot(v, dq, color=col, label=lab)
    ax.set_xlabel(f"{SNAME[sel_sig]} (larger = less familiar)")
    ax.set_ylabel("fraction of pictures below")
    ax.set_title(f"Familiarity, {SNAME[sel_sig]}, smoothing {gs} (test)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "fig5_familiarity")


if __name__ == "__main__":
    main()
