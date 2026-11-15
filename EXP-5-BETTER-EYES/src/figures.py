"""Draw the EXP-5 figures from the real result file and write the summary tables and the verdict.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes figures/*.svg and results/summary.json. Every mark is a mean over seeds read from
results/runs.jsonl. The way (T1, T2 or T3) is selected on the practice half and judged on the
exam half, as fixed in JOURNAL.md sections 4 and 6.
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
WAYS = ["T0", "T1", "T2", "T3"]
WNAME = {"T0": "T0 as before", "T1": "T1 turned, name the turn", "T2": "T2 turned, same answer (control)",
         "T3": "T3 two copies agree"}
BLUE, RED, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
WCOL = {"T0": RED, "T1": GREEN, "T2": "#8a63d2", "T3": BLUE}
INK, INK2, MUTED, GRID, AXIS, SURF, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#b9b8b0"
SPLITS = ["full", "practice", "exam"]
EXP4 = {"R7": 39.5, "open": 41.0, "frozen": 32.9, "retrain": 52.7}   # EXP-4 exam-half values for step 0

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


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "runs.jsonl"
    rows = [json.loads(l) for l in open(src)]
    by = defaultdict(list)                      # (way, passes, rule, stage) -> records, sorted by seed
    for r in rows:
        by[(r["way"], r["passes"], r["rule"], r["stage"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["seed"])
    S = max(r["stage"] for r in rows)
    passes = sorted({r["passes"] for r in rows if r["way"] == "T0"}, reverse=True)
    P0 = passes[0]
    stages = list(range(1, S + 1))

    def cell(recs, sp):
        older = [r[sp]["err_older"] / max(r[sp]["err_older"] + r[sp]["err_newer"], 1) for r in recs]
        out = dict(all=ms([r[sp]["acc_all"] for r in recs]), per_seed_all=[round(r[sp]["acc_all"] * 100, 2) for r in recs],
                   first=ms([r[sp]["acc_groups"][0] for r in recs]),
                   groups=[round(float(np.mean([r[sp]["acc_groups"][g] for r in recs])) * 100, 2)
                           for g in range(len(recs[0][sp]["acc_groups"]))],
                   older_share=ms(older), first_own=ms([r["first_own"][sp] for r in recs]))
        return out

    def cost(recs):
        out = dict(train_seconds=ms([r["train_seconds"] for r in recs], 1.0, 1))
        if all(r.get("energy_wh") is not None for r in recs):
            out["energy_wh"] = ms([r["energy_wh"] for r in recs], 1.0, 3)
        out["peak_mem_mb"] = round(float(np.max([r["peak_mem_mb"] for r in recs])), 0)
        return out

    def shots(recs):
        return {k: ms([r["few_shot"][k] for r in recs]) for k in recs[0]["few_shot"]}

    summary = {"source": src.name, "seeds": len(by[("T0", P0, "R7", S)]), "final": {}, "refs": {}, "cost": {},
               "few_shot": {}, "first_own_over_time": {}, "over_time": {}}
    for sp in SPLITS:
        summary["final"][sp] = {str(p): {w: {ru: cell(by[(w, p, ru, S)], sp) for ru in ("R0", "R7", "open")}
                                         for w in WAYS} for p in passes}
        summary["refs"][sp] = {"frozen": cell(by[("frozen", P0, "open", S)], sp),
                               "retrain": {str(p): cell(by[("retrain", p, "open", S)], sp) for p in passes}}
    for p in passes:
        summary["cost"][str(p)] = {w: cost(by[(w, p, "R7", S)]) for w in WAYS}
        summary["cost"][str(p)]["retrain"] = cost(by[("retrain", p, "open", S)])
        summary["few_shot"][str(p)] = {w: shots(by[(w, p, "open", S)]) for w in WAYS}
        summary["few_shot"][str(p)]["retrain"] = shots(by[("retrain", p, "open", S)])
    summary["cost"][str(P0)]["frozen"] = cost(by[("frozen", P0, "open", S)])
    summary["few_shot"][str(P0)]["frozen"] = shots(by[("frozen", P0, "open", S)])
    for w in WAYS:                               # exam half, main number of passes, after every stage
        summary["over_time"][w] = {ru: [ms([r["exam"]["acc_all"] for r in by[(w, P0, ru, s)]]) for s in stages]
                                   for ru in ("R7", "open")}
        summary["first_own_over_time"][w] = {
            "own_cards_only": [ms([r["first_own"]["exam"] for r in by[(w, P0, "R7", s)]]) for s in stages],
            "among_all": [ms([r["exam"]["acc_groups"][0] for r in by[(w, P0, "R7", s)]]) for s in stages]}

    # ------------------------------------------------------------------ verdict (JOURNAL.md section 6)
    def own_drop(w, sp):
        a = np.mean([r["first_own"][sp] for r in by[(w, P0, "R7", 1)]])
        b = np.mean([r["first_own"][sp] for r in by[(w, P0, "R7", S)]])
        return round(float(a - b) * 100, 2), round(float(a) * 100, 2), round(float(b) * 100, 2)

    pr, ex = summary["final"]["practice"][str(P0)], summary["final"]["exam"][str(P0)]
    passed = [w for w in WAYS[1:] if own_drop(w, "practice")[0] <= 3.0]
    chosen = max(passed, key=lambda w: pr[w]["R7"]["all"][0]) if passed else None
    t0, fro, ret = ex["T0"], summary["refs"]["exam"]["frozen"], summary["refs"]["exam"]["retrain"][str(P0)]
    v = {"selection": {"passed_step3_on_practice": passed, "selected": chosen,
                       "practice_scores_R7": {w: pr[w]["R7"]["all"][0] for w in WAYS}},
         "step0_rebuild_exam": {"T0_R7": t0["R7"]["all"][0], "T0_open": t0["open"]["all"][0],
                                "frozen": fro["all"][0], "retrain": ret["all"][0], "exp4": EXP4,
                                "passes": bool(abs(t0["R7"]["all"][0] - EXP4["R7"]) <= 3 and
                                               abs(t0["open"]["all"][0] - EXP4["open"]) <= 3 and
                                               abs(fro["all"][0] - EXP4["frozen"]) <= 3 and
                                               abs(ret["all"][0] - EXP4["retrain"]) <= 3)}}
    if chosen:
        sel = ex[chosen]
        gap = ret["all"][0] - t0["R7"]["all"][0]
        share = (sel["R7"]["all"][0] - t0["R7"]["all"][0]) / gap
        seeds_share = [(a - b) / (c - b) for a, b, c in zip(sel["R7"]["per_seed_all"], t0["R7"]["per_seed_all"],
                                                            ret["per_seed_all"])]
        d = own_drop(chosen, "exam")
        open_gain = round(sel["open"]["all"][0] - t0["open"]["all"][0], 2)
        v["step1_eyes"] = {"open_book_selected": sel["open"]["all"][0], "open_book_T0": t0["open"]["all"][0],
                           "gain": open_gain, "eyes_improved": bool(open_gain >= 2.0),
                           "few_shot_3": {w: summary["few_shot"][str(P0)][w]["3"][0] for w in WAYS + ["frozen", "retrain"]}}
        v["step2_gap"] = {"selected_R7": sel["R7"]["all"][0], "T0_R7": t0["R7"]["all"][0], "retrain": ret["all"][0],
                          "gap": round(gap, 2), "share": round(share, 3),
                          "share_per_seed": [round(x, 3) for x in seeds_share]}
        v["step3_forgetting"] = {"first_group_own_cards_stage1": d[1], f"first_group_own_cards_stage{S}": d[2],
                                 "drop": d[0], "passes": bool(d[0] <= 3.0),
                                 "first_group_among_all": sel["R7"]["first"][0], "retrain_first_group": ret["first"][0]}
        v["verdict"] = ("does not work" if d[0] > 3.0 or share < 0.25 else "partly works" if share < 0.5 else "works")
        if chosen == "T1":
            diff = round(ex["T1"]["R7"]["all"][0] - ex["T2"]["R7"]["all"][0], 2)
            v["control"] = {"T1_minus_T2": diff, "gain_credited_to": "more altered data" if abs(diff) <= 1.0 else "the extra job"}
    summary["verdict"] = v

    # ------------------------------------------------------------------ figures (exam half)
    x = np.arange(len(WAYS))
    # fig 1: overall score per way, R7 and open-book, with floor and ceiling
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    r7 = [ex[w]["R7"]["all"] for w in WAYS]
    op = [ex[w]["open"]["all"] for w in WAYS]
    ax.bar(x - 0.18, [a[0] for a in r7], 0.34, yerr=[a[1] for a in r7], color=[WCOL[w] for w in WAYS],
           error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    ax.bar(x + 0.18, [a[0] for a in op], 0.34, yerr=[a[1] for a in op], color=NEUTRAL,
           error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for i in range(len(WAYS)):
        ax.text(i - 0.18, r7[i][0] + r7[i][1] + 0.5, f"{r7[i][0]:.1f}", ha="center", fontsize=8.5)
        ax.text(i + 0.18, op[i][0] + op[i][1] + 0.5, f"{op[i][0]:.1f}", ha="center", fontsize=8.5, color=INK2)
    ax.axhline(ret["all"][0], color=AMBER, lw=1.4, ls=(0, (4, 3)), label=f"retrain everything (ceiling) {ret['all'][0]:.1f}")
    ax.axhline(fro["all"][0], color=BLUE, lw=1.4, ls=(0, (4, 3)), label=f"frozen eyes (floor) {fro['all'][0]:.1f}")
    ax.bar([0], [0], color=NEUTRAL, label="grey bar: open-book on the same eyes (not allowed)")
    ax.set_xticks(x)
    ax.set_xticklabels([WNAME[w].replace(" ", "\n", 1) for w in WAYS], fontsize=8)
    ax.set_ylim(0, ret["all"][0] * 1.12)
    ax.set_ylabel("accuracy on all 80 classes, %")
    ax.set_title(f"Fig. Overall score for each way of training the new units. Exam half, {P0} passes, 5 seeds, bar = 1 s.d.\n"
                 f"coloured bar = real test with rule R7. Selected on the practice half: {chosen}", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig1_ways")

    # fig 2: fewer passes
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ps = sorted(passes)
    for w in WAYS:
        y = [summary["final"]["exam"][str(p)][w]["R7"]["all"][0] for p in ps]
        ax.plot(ps, y, color=WCOL[w], marker="o", markeredgecolor=SURF, markeredgewidth=1, label=WNAME[w])
    y = [summary["refs"]["exam"]["retrain"][str(p)]["all"][0] for p in ps]
    ax.plot(ps, y, color=AMBER, marker="^", ls=(0, (4, 2)), markeredgecolor=SURF, markeredgewidth=1, label="retrain everything")
    ax.axhline(fro["all"][0], color=BLUE, lw=1.2, ls=(0, (1, 2)), label=f"frozen eyes, {P0} passes")
    ax.set_xscale("log")
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.minorticks_off()
    ax.set_xlabel("passes over the pictures of each stage")
    ax.set_ylabel("accuracy on all 80 classes, %")
    ax.set_ylim(0, None)
    ax.set_title("Fig. How the score falls with fewer passes. Exam half, rule R7, mean of 5 seeds", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig2_passes")

    # fig 3: first group over time, own cards only (solid) and among all classes (dashed)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for w in WAYS:
        d = summary["first_own_over_time"][w]
        ax.plot(stages, [a[0] for a in d["own_cards_only"]], color=WCOL[w], marker="o", markeredgecolor=SURF,
                markeredgewidth=1, label=f"{WNAME[w]}: own 10 cards only")
        ax.plot(stages, [a[0] for a in d["among_all"]], color=WCOL[w], ls=(0, (3, 2)), lw=1.4)
    ax.plot([], [], color=INK2, ls=(0, (3, 2)), lw=1.4, label="dashed: same group among all classes so far")
    ax.set_xticks(stages)
    ax.set_xlabel("stage (10 new classes arrive at each stage)")
    ax.set_ylabel("accuracy on the first group, %")
    ax.set_ylim(0, 100)
    ax.set_title(f"Fig. Is the first group really forgotten? Exam half, {P0} passes, rule R7", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig3_first_group")

    # fig 4: three-picture test on the surprise classes
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    names = ["frozen"] + WAYS + ["retrain"]
    fs = summary["few_shot"][str(P0)]
    vals = [fs[n]["3"] for n in names]
    cols = [BLUE] + [WCOL[w] for w in WAYS] + [AMBER]
    ax.bar(range(len(names)), [a[0] for a in vals], yerr=[a[1] for a in vals], color=cols, width=0.62,
           error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for i, a in enumerate(vals):
        ax.text(i, a[0] + a[1] + 0.4, f"{a[0]:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["frozen\neyes"] + [WNAME[w].replace(" ", "\n", 1) for w in WAYS] + ["retrain\neverything"], fontsize=7.5)
    ax.set_ylabel("accuracy on 20 surprise classes, %")
    ax.set_title(f"Fig. Learning 20 never-seen classes from 3 pictures each. {P0} passes, mean of 5 seeds", fontsize=9.5)
    save(fig, "fig4_surprise")

    # fig 5: cost
    names = WAYS + ["retrain"]
    c = summary["cost"][str(P0)]
    has_e = all("energy_wh" in c[n] for n in names)
    fig, axes = plt.subplots(1, 2 if has_e else 1, figsize=(8.4, 3.2))
    axes = np.atleast_1d(axes)
    for ax, key, lab in zip(axes, ["train_seconds", "energy_wh"], ["training time per seed, seconds", "GPU energy per seed, Wh"]):
        vals = [c[n][key][0] for n in names]
        ax.bar(range(len(names)), vals, color=[WCOL[w] for w in WAYS] + [AMBER], width=0.62)
        for i, a in enumerate(vals):
            ax.text(i, a, f"{a:.0f}" if key == "train_seconds" else f"{a:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(WAYS + ["retrain\nonce"], fontsize=8)
        ax.set_ylabel(lab)
    axes[0].set_title(f"Fig. Cost of each way, all 8 stages, {P0} passes (Kaggle T4)", fontsize=9.5)
    save(fig, "fig5_cost")

    # fig 6: score per group
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    series = [("T0 as before", ex["T0"]["R7"]["groups"], RED, "s", "-")]
    if chosen:
        series.append((f"{WNAME[chosen]} (selected)", ex[chosen]["R7"]["groups"], WCOL[chosen], "D", "-"))
    series.append(("retrain everything", ret["groups"], AMBER, "^", (0, (4, 2))))
    for name, y, col, m, ls in series:
        ax.plot(stages, y, color=col, marker=m, ls=ls, markeredgecolor=SURF, markeredgewidth=1, label=name)
    ax.set_xticks(stages)
    ax.set_xlabel("group of classes (1 = learned first; last = newest)")
    ax.set_ylabel("accuracy on that group, %")
    ax.set_ylim(0, None)
    ax.set_title(f"Fig. Score per group after the last stage. Exam half, {P0} passes, rule R7", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig6_per_group")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
