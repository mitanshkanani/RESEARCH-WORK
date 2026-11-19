"""Build the EXP-13 summary, the verdict and the figures from the real result file.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes results/summary.json and figures/*.svg. Every setting is chosen on the rehearsal rows only (memory after
stage 6, stages 7-8 as pretend-new, named among 80), by the rules fixed in JOURNAL.md section 3:
  with the handicap: over smoothing x handicap, keep rehearsal old >= old at h = 0 (same smoothing) - 5, best H;
  no correction:     h = 0, the smoothing with the best rehearsal H;
  selected copies:   among moved, enlarged and both, the best rehearsal H at 3 pictures (with the handicap).
Harmonic means are computed per seed (from old and new averaged over the draws), then averaged.
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
CNAME = {"none": "no copies", "shift": "moved (8 copies)", "zoom": "enlarged (5 copies)", "both": "both (13 copies)"}
CCOL = {"none": NEUTRAL, "shift": BLUE, "zoom": GREEN, "both": VIOLET}
EXP12 = {"old": 52.47, "new": 17.28}
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


def log_interp(x_pts, y_pts, y):
    """Pictures at which the curve (y against log x, straight between points) reaches y."""
    lx = np.log(np.asarray(x_pts, float))
    for i in range(len(y_pts) - 1):
        a, b = y_pts[i], y_pts[i + 1]
        if (a - y) * (b - y) <= 0 and a != b:
            return float(np.exp(lx[i] + (y - a) / (b - a) * (lx[i + 1] - lx[i])))
    return None


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "runs.jsonl"
    meta = json.loads((path.parent / "meta.json").read_text())
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    gammas, G1 = meta["gammas"], meta["g1"]
    i0, i8 = G1.index(0.0), G1.index(0.08)

    CV = defaultdict(lambda: {"old": [], "new": [], "new_only": []})
    cost = []
    for r in rows:
        if r.get("kind") == "curve":
            v = CV[(r["task"], r["gamma"], r["shots"], r["copies"])]
            v["old"].append(np.array(r["old"]) / 100)
            v["new"].append(np.array(r["new"]) / 100)
            v["new_only"].append(100 * r["new_only"])
        elif r.get("task") == "cost":
            cost.append(r)
    shots_of = {t: sorted({k for (tt, _, k, _) in CV if tt == t}) for t in ("rehearsal", "part1")}
    # labels: the measured numbers of pictures, with the largest ("all") matched across rehearsal and test
    labels = [str(k) for k in shots_of["part1"][:-1]] + ["all"]
    kk = {t: dict(zip(labels, shots_of[t])) for t in shots_of}
    assert len(shots_of["rehearsal"]) == len(shots_of["part1"])
    n_all = shots_of["part1"][-1]
    copy_shots = [k for k in meta["shots"]]

    def arr(task, g, lab, cs):
        v = CV[(task, g, kk[task][lab], cs)]
        return np.stack(v["old"]), np.stack(v["new"]), np.array(v["new_only"])

    def choose(cs, lab, handicap=True):
        best = None
        for g in gammas:
            o, n, _ = arr("rehearsal", g, lab, cs)
            om, hh = o.mean(0), hm(o, n).mean(0)
            ok = om >= om[i0] - LIMIT if handicap else np.arange(len(G1)) == i0
            v = np.where(ok, hh, -1)
            j = int(v.argmax())
            if best is None or v[j] > best[0]:
                best = (float(v[j]), g, j, float(om[j]), float(om[i0]))
        _, g, j, r_old, r_c0 = best
        o, n, no = arr("part1", g, lab, cs)
        return dict(gamma=g, h=G1[j], rehearsal=dict(H=round(best[0], 2), old=round(r_old, 2), C0_old=round(r_c0, 2)),
                    old=ms(o[:, j]), new=ms(n[:, j]), H=ms(hm(o[:, j], n[:, j])), C0_old=ms(o[:, i0]), new_only=ms(no))

    curve = {lab: dict(handicap=choose("none", lab), none=choose("none", lab, handicap=False)) for lab in labels}
    copies = {cs: {str(k): choose(cs, str(k)) for k in copy_shots} for cs in meta["copy_sets"]}

    c = cost
    peak = max(x["peak_mem_mb"] or 0 for x in c)
    cost_out = dict(feature_dim=c[0]["feature_dim"], peak_mem_mb=peak, peak_reserved_mb=max(x["peak_reserved_mb"] or 0 for x in c),
                    cpu_peak_mb=max((x["cpu_peak_mb"] or 0) for x in c),
                    seconds={k: ms([x["seconds"][k] for x in c], 1) for k in c[0]["seconds"]},
                    total_seconds=ms([sum(x["seconds"].values()) for x in c], 1),
                    wh=ms([sum(v for v in x["wh"].values() if v) for x in c], 3) if c[0]["wh"] else None)

    # ---- verdict, by JOURNAL.md section 5
    o, n, _ = arr("part1", 0.1, "3", "none")
    step0 = dict(old=ms(o[:, i8]), new=ms(n[:, i8]))
    step0["passed"] = abs(step0["old"][0] - EXP12["old"]) <= 0.5 and abs(step0["new"][0] - EXP12["new"]) <= 0.5
    cands = [cs for cs in meta["copy_sets"] if cs != "none"]
    selected = max(cands, key=lambda cs: copies[cs]["3"]["rehearsal"]["H"])
    gain = {str(k): round(copies[selected][str(k)]["H"][0] - copies["none"][str(k)]["H"][0], 2) for k in copy_shots}
    gain_no = round(copies[selected]["3"]["new_only"][0] - copies["none"]["3"]["new_only"][0], 2)
    s3 = copies[selected]["3"]
    cond = dict(a=gain["3"] >= 3.0 and s3["old"][0] >= s3["C0_old"][0] - 6.0,
                b=gain["1"] >= 1.5 and gain["10"] >= 1.5, c=gain_no >= 2.0, d=peak <= MEM_LIMIT_MB)
    if all(cond.values()):
        verdict = "works"
    elif cond["d"] and gain["3"] >= 1.0:
        verdict = "helps, but not enough"
    else:
        verdict = "does not work"
    xs = [float(lab) if lab != "all" else float(n_all) for lab in labels]
    hcurve = [curve[lab]["handicap"]["H"][0] for lab in labels]
    equiv = log_interp(xs, hcurve, s3["H"][0])
    need90 = log_interp(xs, hcurve, 0.9 * hcurve[-1])
    best_test = max(cands, key=lambda cs: copies[cs]["3"]["H"][0])
    summary = dict(
        labels=labels, n_all=n_all, curve=curve, copies=copies, cost=cost_out,
        verdict=dict(
            step0=step0,
            step1=dict(selected=selected, gain_H=gain, gain_new_only_3=gain_no, old=s3["old"][0], C0_old=s3["C0_old"][0],
                       peak_mem_mb=peak, conditions=cond, verdict=verdict),
            step2=dict(picture_equivalent=None if equiv is None else round(equiv, 2),
                       counts_more=equiv is not None and equiv >= 4.5),
            step3=dict(pictures_for_90pct=None if need90 is None else round(need90, 1), H_all=hcurve[-1]),
            step4=dict(H_all_handicap=curve["all"]["handicap"]["H"][0], H_all_none=curve["all"]["none"]["H"][0],
                       difference=round(curve["all"]["handicap"]["H"][0] - curve["all"]["none"]["H"][0], 2),
                       handicap_needed=curve["all"]["handicap"]["H"][0] - curve["all"]["none"]["H"][0] >= 1.0),
            step5=dict(selected_H=s3["H"][0], best_on_test=best_test, best_H=copies[best_test]["3"]["H"][0],
                       trusted=copies[best_test]["3"]["H"][0] - s3["H"][0] <= 1.0)))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["verdict"], indent=1))
    for lab in labels:
        a, b = curve[lab]["handicap"], curve[lab]["none"]
        print(f"{lab:>4s}: handicap H {a['H'][0]:5.1f} (old {a['old'][0]:.1f} new {a['new'][0]:.1f} g {a['gamma']} h {a['h']})"
              f" | none H {b['H'][0]:5.1f} (old {b['old'][0]:.1f} new {b['new'][0]:.1f} g {b['gamma']}) | new-only {a['new_only'][0]}")
    for cs in meta["copy_sets"]:
        print(cs, {k: (v["H"][0], v["old"][0], v["new"][0], v["new_only"][0], v["gamma"], v["h"], v["rehearsal"]["H"])
                   for k, v in copies[cs].items()})

    # ---- figure 1: the curve, with the copies placed on it
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(xs, hcurve, marker="o", color=INK2, label="with the handicap")
    ax.plot(xs, [curve[lab]["none"]["H"][0] for lab in labels], marker="o", color=NEUTRAL, ls="--", label="no correction")
    ax.plot(xs, [curve[lab]["handicap"]["new_only"][0] for lab in labels], marker="s", color=AMBER, lw=1.4,
            label="new classes among themselves")
    for cs in cands:
        ax.scatter(copy_shots, [copies[cs][str(k)]["H"][0] for k in copy_shots], color=CCOL[cs], zorder=3, s=28,
                   label=f"with copies: {CNAME[cs]}")
    ax.set_xscale("log")
    ax.set_xticks(xs, labels[:-1] + [f"all ({n_all})"])
    ax.set_xlabel("pictures per new class")
    ax.set_ylabel("% (test)")
    ax.set_title("Harmonic mean of old and new among all 100, against pictures per new class")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig1_curve")

    # ---- figure 2: copies at 1, 3, 10 pictures
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    w = 0.2
    for i, cs in enumerate(meta["copy_sets"]):
        x = np.arange(len(copy_shots)) + (i - 1.5) * w
        for ax, key in zip(axes, ("H", "new_only")):
            vals = [copies[cs][str(k)][key] for k in copy_shots]
            ax.bar(x, [m for m, _ in vals], yerr=[sd for _, sd in vals], width=w, color=CCOL[cs], label=CNAME[cs],
                   error_kw=dict(ecolor=INK2, lw=0.7, capsize=1.5))
    for ax, t in zip(axes, ("harmonic mean among 100 (test)", "new classes among themselves (test)")):
        ax.set_xticks(np.arange(len(copy_shots)), [f"{k} picture{'s' if k > 1 else ''}" for k in copy_shots])
        ax.set_title(t)
        ax.grid(False, axis="x")
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig2_copies")

    # ---- figure 3: old against new as the handicap varies, at 3 pictures (with and without copies) and with all pictures
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for cs, lab, col, ls in (("none", "3", NEUTRAL, "-"), (selected, "3", CCOL[selected], "-"), ("none", "30", AMBER, "--"),
                             ("none", "all", INK2, "--")):
        ch = curve[lab]["handicap"] if cs == "none" else copies[cs][lab]
        o, n, _ = arr("part1", ch["gamma"], lab, cs)
        name = f"{lab} pictures, {CNAME[cs]}" if lab != "all" else "all pictures"
        ax.plot(o.mean(0), n.mean(0), color=col, ls=ls, lw=1.6, label=name)
        ax.scatter([ch["old"][0]], [ch["new"][0]], s=45, facecolor="none", edgecolor=col, lw=1.5, zorder=3)
    ax.set_xlabel("old classes % (among all 100)")
    ax.set_ylabel("new classes % (among all 100)")
    ax.set_title("Old against new as the handicap varies (test)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig3_tradeoff")

    # ---- figure 4: old and new separately against pictures, with and without the handicap
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for mode, ls in (("handicap", "-"), ("none", "--")):
        ax.plot(xs, [curve[lab][mode]["old"][0] for lab in labels], color=BLUE, ls=ls, marker="o",
                label=f"old classes, {'with the handicap' if mode == 'handicap' else 'no correction'}")
        ax.plot(xs, [curve[lab][mode]["new"][0] for lab in labels], color=RED, ls=ls, marker="o",
                label=f"new classes, {'with the handicap' if mode == 'handicap' else 'no correction'}")
    ax.set_xscale("log")
    ax.set_xticks(xs, labels[:-1] + [f"all ({n_all})"])
    ax.set_xlabel("pictures per new class")
    ax.set_ylabel("% among all 100 (test)")
    ax.set_title("Old and new classes against pictures per new class")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig4_old_new")


if __name__ == "__main__":
    main()
