"""Build the EXP-9 summary, the verdict and the figures from the real result file.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes results/summary.json and figures/*.svg. Settings are chosen on the rehearsal rows only
(memory after stage 6, classes of stages 7 and 8 as pretend-new, 3 pictures), as fixed in
JOURNAL.md sections 3 and 5, and then read on the surprise rows.
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
FAMILIES = ["W0", "W1", "W2", "W2+B", "W3", "W3+B"]
FNAME = {"W0": "plain\ncards", "W1": "shared table,\nsmoothed", "W2": "tidy", "W2+B": "tidy +\nborrow",
         "W3": "learned\nview", "W3+B": "learned view\n+ borrow"}
FCOL = {"W0": NEUTRAL, "W1": AMBER, "W2": BLUE, "W2+B": VIOLET, "W3": GREEN, "W3+B": RED}
# EXP-8 numbers quoted as targets (EXP-8-SAME-MEMORY/results/summary.json, exam half, 5 seeds)
EXP8 = {"P_W0_3": 17.01, "P_W1_g100_3": 21.17, "F_W0_3": 25.23, "P_best_3": 21.17, "F_best_3": 25.57,
        "G_best": {"1": 19.39, "3": 29.78, "10": 41.12}, "R_best": {"1": 25.44, "3": 38.46, "10": 51.54}}
TARGET_G, TARGET_R = 29.78, 38.46

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


def ms(v, scale=100.0, nd=2):
    v = np.array(v, dtype=float) * scale
    return [round(float(v.mean()), nd), round(float(v.std()), nd)]


def family(way):
    if way.startswith("W3"):
        return "W3+B" if "+B" in way else "W3"
    if way.startswith("W2"):
        return "W2+B" if "+B" in way else "W2"
    if way.startswith("W1"):
        return None if way == "W1_g0.01" else "W1"       # 0.01 is EXP-8's value, a reference only
    return "W0"


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "runs.jsonl"
    rows = [json.loads(l) for l in open(src)]
    fs = defaultdict(list)        # (eyes, task, known, way) -> rows
    chk = defaultdict(list)
    costs = defaultdict(list)
    for r in rows:
        if r["task"] == "cost":
            costs[r["eyes"]].append(r)
        elif r["task"] == "all100":
            chk[(r["eyes"], r["way"])].append(r)
        else:
            fs[(r["eyes"], r["task"], r["known"], r["way"])].append(r)
    seeds = len(costs["P"])
    ways = {e: list(dict.fromkeys(k[3] for k in fs if k[0] == e and k[1] == "surprise")) for e in EYES}
    known_test = sorted({k[2] for k in fs if k[1] == "surprise"})
    k_reh = next(k[2] for k in fs if k[1] == "rehearsal")
    K = known_test[-1]

    def m(e, task, known, way, shots="3"):
        return ms([r["few_shot"][shots] for r in fs[(e, task, known, way)]])

    s = {"source": src.name, "seeds": seeds, "rehearsal_known": k_reh, "known_test": known_test}
    sel, chosen, rehearsal = {}, {}, {}
    for e in EYES:
        reh = {w: m(e, "rehearsal", k_reh, w)[0] for w in ways[e]}
        rehearsal[e] = reh
        chosen[e] = {}
        for f in FAMILIES:
            cands = [w for w in ways[e] if family(w) == f]
            chosen[e][f] = max(cands, key=lambda w: reh[w])
        sel[e] = max(FAMILIES, key=lambda f: reh[chosen[e][f]])
    s["chosen_setting"] = chosen
    s["selected_family"] = sel
    s["rehearsal_3"] = {e: {f: rehearsal[e][chosen[e][f]] for f in FAMILIES} for e in EYES}

    s["surprise"] = {e: {f: {"way": chosen[e][f], **{str(k): {sh: m(e, "surprise", k, chosen[e][f], sh)
                                                                for sh in ("1", "3", "10")} for k in known_test}}
                         for f in FAMILIES} for e in EYES}
    s["reference_W1_g0.01"] = {e: {sh: m(e, "surprise", K, "W1_g0.01", sh) for sh in ("1", "3", "10")} for e in EYES}
    s["oracle_best_on_surprise_3"] = {}
    for e in EYES:
        sc = {w: m(e, "surprise", K, w)[0] for w in ways[e] if family(w)}
        b = max(sc, key=sc.get)
        s["oracle_best_on_surprise_3"][e] = dict(way=b, score=sc[b])
    s["all_settings_80_3"] = {e: {w: m(e, "surprise", K, w)[0] for w in ways[e]} for e in EYES}

    def c100(e, w):
        recs = chk[(e, w)]
        return {k: ms([r[k] for r in recs]) for k in ("new", "old_100", "old_80")}
    s["all100"] = {e: {"selected": dict(way=chosen[e][sel[e]], **c100(e, chosen[e][sel[e]])),
                       "W0": c100(e, "W0"),
                       "EXP-8 table (W1_g0.01)": c100(e, "W1_g0.01"),
                       **{f: dict(way=chosen[e][f], **c100(e, chosen[e][f])) for f in FAMILIES}} for e in EYES}

    def cst(e):
        out = {}
        for k in ("build_seconds", "train_seconds", "describe_seconds", "tables_seconds", "solve_seconds",
                  "eval_seconds", "peak_mem_mb", "train_wh", "build_wh"):
            if all(r.get(k) is not None for r in costs[e]):
                out[k] = ms([r[k] for r in costs[e]], 1.0, 3)
        if all(r.get("wh") for r in costs[e]):
            out["wh"] = {k: ms([r["wh"][k] for r in costs[e]], 1.0, 4) for k in costs[e][0]["wh"]}
        out["feature_dim"] = costs[e][0]["feature_dim"]
        return out
    s["cost"] = {e: cst(e) for e in EYES}

    # ---- verdict, by JOURNAL.md section 5
    sv = lambda e, f, k=K, sh="3": s["surprise"][e][f][str(k)][sh][0]
    v = {}
    v["step0_rebuild"] = {
        "P_W0": [sv("P", "W0"), EXP8["P_W0_3"]],
        "P_W1_g100": [m("P", "surprise", K, "W1_g100.0")[0], EXP8["P_W1_g100_3"]],
        "F_W0": [sv("F", "W0"), EXP8["F_W0_3"]]}
    v["step0_passed"] = all(abs(a - b) <= 2.0 for a, b in v["step0_rebuild"].values())
    p = sv("P", sel["P"])
    v["step1"] = dict(score=p, family=sel["P"], way=chosen["P"][sel["P"]],
                      verdict="works" if p >= TARGET_G else ("helps, but not enough" if p >= EXP8["P_best_3"] + 3 else "does not work"))
    rise = sv("P", sel["P"], K) - sv("P", sel["P"], known_test[0])
    v["step2"] = dict(score_10=sv("P", sel["P"], known_test[0]), score_80=sv("P", sel["P"], K), rise=round(rise, 2),
                      verdict="yes" if rise >= 3 else "no")
    ladder = ["W0", "W2", "W3", "W3+B"]
    v["step3"] = {f: sv("P", f) for f in ladder}
    v["step3_gains"] = {f"{a}->{b}": round(sv("P", b) - sv("P", a), 2) for a, b in zip(ladder, ladder[1:])}
    v["step3_counts"] = {k: g >= 1.5 for k, g in v["step3_gains"].items()}
    fsc = sv("F", sel["F"])
    v["step4"] = dict(score=fsc, family=sel["F"], gain=round(fsc - EXP8["F_best_3"], 2),
                      verdict="yes" if fsc - EXP8["F_best_3"] >= 3 else "no")
    v["step5"] = {}
    for e in EYES:
        best_f = max(FAMILIES, key=lambda f: sv(e, f))
        v["step5"][e] = dict(selected=sel[e], selected_score=sv(e, sel[e]), best_family_on_test=best_f,
                             best_score=sv(e, best_f), trusted=sv(e, best_f) - sv(e, sel[e]) <= 1.0)
    s["verdict"] = v
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(v, indent=1))

    # ---- figure 1: every way at 3 pictures, 80 known classes
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, e in zip(axes, EYES):
        vals = [s["surprise"][e][f][str(K)]["3"] for f in FAMILIES]
        x = np.arange(len(FAMILIES))
        ax.bar(x, [a for a, _ in vals], yerr=[b for _, b in vals], color=[FCOL[f] for f in FAMILIES], width=0.65,
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for xi, (a, _) in zip(x, vals):
            ax.text(xi, a / 2, f"{a:.1f}", ha="center", fontsize=8, color=INK)
        ax.axhline(TARGET_G, color=RED, ls="--", lw=1)
        ax.axhline(TARGET_R, color=AMBER, ls="--", lw=1)
        ax.text(-0.45, TARGET_G + 0.6, "grow and lock (EXP-8)", ha="left", fontsize=7.5, color=RED)
        ax.text(-0.45, TARGET_R + 0.6, "retrain (EXP-8)", ha="left", fontsize=7.5, color=AMBER)
        ax.set_xticks(x, [FNAME[f] for f in FAMILIES], fontsize=7.5)
        ax.grid(False, axis="x")
        ax.set_title(ENAME[e])
        ax.set_ylim(0, 45)
    axes[0].set_ylabel("20 new classes, 3 pictures each: accuracy %")
    save(fig, "fig1_ways")

    # ---- figure 2: does knowing more classes help?
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for ax, e in zip(axes, EYES):
        for f in FAMILIES:
            y = [s["surprise"][e][f][str(k)]["3"][0] for k in known_test]
            ax.plot(known_test, y, marker="o", color=FCOL[f], label=FNAME[f].replace("\n", " "))
        ax.set_xticks(known_test)
        ax.set_xlabel("classes already known")
        ax.set_title(ENAME[e])
    axes[0].set_ylabel("3 pictures each: accuracy %")
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig2_known_classes")

    # ---- figure 3: number of pictures
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    sh = ["1", "3", "10"]
    xs = [1, 3, 10]
    for e, col in (("P", GREEN), ("F", BLUE)):
        ax.plot(xs, [s["surprise"][e][sel[e]][str(K)][k][0] for k in sh], marker="o", color=col,
                label=f"{ENAME[e]}, selected way")
        ax.plot(xs, [s["surprise"][e]["W0"][str(K)][k][0] for k in sh], marker="o", color=col, ls=":",
                label=f"{ENAME[e]}, plain cards")
    ax.plot(xs, [EXP8["G_best"][k] for k in sh], marker="s", color=RED, ls="--", lw=1.2, label="grow and lock (quoted, EXP-8)")
    ax.plot(xs, [EXP8["R_best"][k] for k in sh], marker="s", color=AMBER, ls="--", lw=1.2, label="retrain (quoted, EXP-8)")
    ax.set_xscale("log")
    ax.set_xticks(xs, ["1", "3", "10"])
    ax.set_xlabel("pictures per new class")
    ax.set_ylabel("accuracy %, 20 new classes")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig3_pictures")

    # ---- figure 4: rehearsal against the real test, every setting
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, e in zip(axes, EYES):
        for w in ways[e]:
            f = family(w)
            if not f:
                continue
            ax.scatter(rehearsal[e][w], s["all_settings_80_3"][e][w], s=12, color=FCOL[f], alpha=0.75, lw=0)
        c = chosen[e][sel[e]]
        ax.scatter(rehearsal[e][c], s["all_settings_80_3"][e][c], s=60, facecolor="none", edgecolor=INK, lw=1.2)
        lo = min(min(rehearsal[e].values()), min(s["all_settings_80_3"][e].values())) - 1
        hi = max(max(rehearsal[e].values()), max(s["all_settings_80_3"][e].values())) + 1
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=0.8, ls=":")
        ax.set_xlabel(f"rehearsal ({k_reh} known, pretend-new classes) %")
        ax.set_title(ENAME[e])
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0].set_ylabel(f"real test ({K} known, surprise classes) %")
    handles = [plt.Line2D([], [], marker="o", ls="", color=FCOL[f], label=FNAME[f].replace("\n", " ")) for f in FAMILIES]
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, label="selected"))
    axes[1].legend(handles=handles, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig4_rehearsal")

    # ---- figure 5: all 100 classes together
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    keys = [("W0", "plain cards"), ("EXP-8 table (W1_g0.01)", "shared table\n(EXP-8)"), ("selected", "selected way")]
    for ax, e in zip(axes, EYES):
        x = np.arange(len(keys))
        wdt = 0.27
        for i, (part, lab, col) in enumerate((("old_80", "old classes, before", NEUTRAL), ("old_100", "old classes, with new", BLUE),
                                             ("new", "new classes (3 pictures)", RED))):
            vals = [s["all100"][e][k][part][0] for k, _ in keys]
            ax.bar(x + (i - 1) * wdt, vals, width=wdt, color=col, label=lab)
            for xi, val in zip(x, vals):
                ax.text(xi + (i - 1) * wdt, val + 0.8, f"{val:.0f}", ha="center", fontsize=7)
        ax.set_xticks(x, [l for _, l in keys], fontsize=8)
        ax.set_title(ENAME[e])
    axes[0].set_ylabel("accuracy % among all classes")
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig5_all100")


if __name__ == "__main__":
    main()
