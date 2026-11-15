"""Draw the EXP-6 figures from the real result file and write the summary tables and the verdict.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes figures/*.svg and results/summary.json. Every mark is a mean over seeds read from
results/runs.jsonl, exam half of the test pictures, after the last stage, as fixed in JOURNAL.md.
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
EXP5 = {"A_open": 40.9, "A_R7": 39.6, "retrain": 53.2, "frozen": 32.8}
SP = "exam"

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
    by = defaultdict(list)
    for r in rows:
        by[(r["way"], r["width"], r["rule"], r["stage"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["seed"])
    S = max(r["stage"] for r in rows)
    stages = list(range(1, S + 1))
    final = {"A": 144, "B": 288, "C": 144}
    start = {"A": 32, "B": 64, "C": 32}

    def cell(recs):
        out = dict(all=ms([r[SP]["acc_all"] for r in recs]), per_seed=[round(r[SP]["acc_all"] * 100, 2) for r in recs],
                   first=ms([r[SP]["acc_groups"][0] for r in recs]), first_own=ms([r["first_own"][SP] for r in recs]),
                   groups=[round(float(np.mean([r[SP]["acc_groups"][g] for r in recs])) * 100, 2)
                           for g in range(len(recs[0][SP]["acc_groups"]))],
                   train_seconds=ms([r["train_seconds"] for r in recs], 1.0, 1),
                   energy_wh=ms([r["energy_wh"] for r in recs], 1.0, 3),
                   peak_mem_mb=round(float(max(r["peak_mem_mb"] for r in recs)), 0))
        if "few_shot" in recs[0]:
            out["few_shot"] = {k: ms([r["few_shot"][k] for r in recs]) for k in recs[0]["few_shot"]}
        return out

    s = {"source": src.name, "split": SP, "seeds": len(by[("A", 144, "open", S)]), "runs": {}, "refs": {}}
    for a in "ABC":
        s["runs"][a] = {ru: cell(by[(a, final[a], ru, S)]) for ru in ("R0", "R7", "open")}
        s["runs"][a]["first_own_R7_by_stage"] = [
            ms([r["first_own"][SP] for r in by[(a, start[a] + (final[a] - start[a]) // (S - 1) * i, "R7", i + 1)]])
            for i in range(S)]
        s["runs"][a]["open_by_stage"] = [
            ms([r[SP]["acc_all"] for r in by[(a, start[a] + (final[a] - start[a]) // (S - 1) * i, "open", i + 1)]])
            for i in range(S)]
    for w in (144, 288):
        s["refs"][f"retrain_{w}"] = cell(by[("retrain", w, "open", S)])
        s["refs"][f"frozen_{w}"] = cell(by[("frozen", w, "open", S)])

    A, B, C = (s["runs"][a] for a in "ABC")
    r144, r288 = s["refs"]["retrain_144"], s["refs"]["retrain_288"]
    ao, bo, co = A["open"]["all"][0], B["open"]["all"][0], C["open"]["all"][0]
    gap144, gap288 = r144["all"][0] - ao, r288["all"][0] - bo

    def name(x):
        return "main cause" if x >= 0.5 else "contributing cause" if x >= 0.25 else "minor"

    narrow, lock = (co - ao) / gap144, (r144["all"][0] - co) / gap144
    ps = lambda k: np.array(k["per_seed"])
    narrow_seeds = (ps(C["open"]) - ps(A["open"])) / (ps(r144) - ps(A["open"]))
    shrink = (gap144 - gap288) / gap144
    b_share = (B["R7"]["all"][0] - A["R7"]["all"][0]) / (r144["all"][0] - A["R7"]["all"][0])
    b_drop = round(B["first_own_R7_by_stage"][0][0] - B["first_own_R7_by_stage"][-1][0], 2)
    a_drop = round(A["first_own_R7_by_stage"][0][0] - A["first_own_R7_by_stage"][-1][0], 2)
    s["verdict"] = {
        "step0_rebuild": {"A_open": ao, "A_R7": A["R7"]["all"][0], "retrain_144": r144["all"][0],
                          "frozen_144": s["refs"]["frozen_144"]["all"][0], "exp5": EXP5,
                          "passes": bool(abs(ao - EXP5["A_open"]) <= 3 and abs(A["R7"]["all"][0] - EXP5["A_R7"]) <= 3 and
                                         abs(r144["all"][0] - EXP5["retrain"]) <= 3 and
                                         abs(s["refs"]["frozen_144"]["all"][0] - EXP5["frozen"]) <= 3)},
        "step1_split_at_144": {"A_open": ao, "C_open": co, "retrain_144": r144["all"][0], "gap": round(gap144, 2),
                               "narrow_view_points": round(co - ao, 2), "narrow_view_share": round(narrow, 3),
                               "narrow_view_share_per_seed": [round(float(x), 3) for x in narrow_seeds],
                               "narrow_view_name": name(narrow),
                               "locking_points": round(r144["all"][0] - co, 2), "locking_share": round(lock, 3),
                               "locking_name": name(lock)},
        "step2_size": {"gap_144": round(gap144, 2), "B_open": bo, "retrain_288": r288["all"][0], "gap_288": round(gap288, 2),
                       "gap_shrinks_by": round(shrink, 3), "size_is_a_cause_of_the_gap": bool(shrink >= 0.25)},
        "step3_double_units": {"A_R7": A["R7"]["all"][0], "B_R7": B["R7"]["all"][0], "share": round(b_share, 3),
                               "first_group_own_cards_drop_B": b_drop, "first_group_own_cards_drop_A": a_drop,
                               "peak_mem_mb_B": B["R7"]["peak_mem_mb"],
                               "verdict": "does not work" if b_share < 0.25 or b_drop > 3 else
                               "partly works" if b_share < 0.5 else "works"},
        "few_shot_3": {"A": A["open"]["few_shot"]["3"][0], "B": B["open"]["few_shot"]["3"][0],
                       "C": C["open"]["few_shot"]["3"][0], "retrain_144": r144["few_shot"]["3"][0],
                       "retrain_288": r288["few_shot"]["3"][0], "frozen_144": s["refs"]["frozen_144"]["few_shot"]["3"][0]},
    }

    # fig 1: how good the eyes are (open-book), all runs
    labels = ["A\nas now\n(size 144)", "C\nwide view\n(size 144)", "retrain\neverything\n(size 144)",
              "B\ndouble units\n(size 288)", "retrain\neverything\n(size 288)"]
    cells = [A["open"], C["open"], r144, B["open"], r288]
    cols = [RED, VIOLET, AMBER, GREEN, AMBER]
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    xs = [0, 1, 2, 3.6, 4.6]
    ax.bar(xs, [c["all"][0] for c in cells], yerr=[c["all"][1] for c in cells], color=cols, width=0.66,
           error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for x, c in zip(xs, cells):
        ax.text(x, c["all"][0] + c["all"][1] + 0.6, f"{c['all'][0]:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("open-book accuracy on all 80 classes, %")
    ax.set_ylim(0, r288["all"][0] * 1.14)
    ax.set_title("Fig. How good are the eyes? Open-book score, exam half, 5 seeds, bar = 1 s.d.", fontsize=9.5)
    save(fig, "fig1_eyes")

    # fig 2: the gap at size 144 split into the two suspects
    fig, ax = plt.subplots(figsize=(7.6, 1.9))
    ax.barh([0], [co - ao], color=VIOLET, height=0.5, label=f"narrow view: {co - ao:.1f} points ({narrow * 100:.0f}%)")
    ax.barh([0], [r144["all"][0] - co], left=[co - ao], color=NEUTRAL, height=0.5,
            label=f"locking and building in steps: {r144['all'][0] - co:.1f} points ({lock * 100:.0f}%)")
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlim(0, gap144)
    ax.set_xlabel(f"points between run A ({ao:.1f}%) and retrain everything ({r144['all'][0]:.1f}%), open-book")
    ax.set_title("Fig. Where the gap comes from, at size 144", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=1)
    save(fig, "fig2_gap_split")

    # fig 3: three-picture test on surprise classes
    names = ["frozen_144", "A", "C", "retrain_144", "B", "retrain_288"]
    vals = [s["refs"][n]["few_shot"]["3"] if n in s["refs"] else s["runs"][n]["open"]["few_shot"]["3"] for n in names]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    xs = [0, 1, 2, 3, 4.6, 5.6]
    ax.bar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], color=[BLUE, RED, VIOLET, AMBER, GREEN, AMBER], width=0.66,
           error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for x, v in zip(xs, vals):
        ax.text(x, v[0] + v[1] + 0.4, f"{v[0]:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["frozen\n(144)", "A\nas now", "C\nwide view", "retrain\n(144)", "B\ndouble units", "retrain\n(288)"], fontsize=7.8)
    ax.set_ylabel("accuracy on 20 surprise classes, %")
    ax.set_title("Fig. Learning 20 never-seen classes from 3 pictures each, mean of 5 seeds", fontsize=9.5)
    save(fig, "fig3_surprise")

    # fig 4: open-book score after every stage for A, B, C
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for a, col, lab in [("A", RED, "A as now"), ("B", GREEN, "B double units"), ("C", VIOLET, "C wide view (diagnosis)")]:
        y = [v[0] for v in s["runs"][a]["open_by_stage"]]
        ax.plot(stages, y, color=col, marker="o", markeredgecolor=SURF, markeredgewidth=1, label=lab)
        ax.text(stages[-1] + 0.12, y[-1], f"{y[-1]:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_xticks(stages)
    ax.set_xlim(stages[0] - 0.2, stages[-1] + 0.8)
    ax.set_xlabel("stage (10 new classes arrive at each stage)")
    ax.set_ylabel("open-book accuracy on all classes so far, %")
    ax.set_title("Fig. Open-book score after each stage, exam half", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig4_over_time")

    # fig 5: cost
    names = [("A", A["R7"], RED), ("B", B["R7"], GREEN), ("C", C["R7"], VIOLET), ("retrain\n144", r144, AMBER), ("retrain\n288", r288, AMBER)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, key, lab, fmt in [(axes[0], "train_seconds", "training time per seed, seconds", "{:.0f}"),
                              (axes[1], "energy_wh", "GPU energy per seed, Wh", "{:.2f}")]:
        vals = [c[key][0] for _, c, _ in names]
        ax.bar(range(len(names)), vals, color=[c for _, _, c in names], width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n for n, _, _ in names], fontsize=8)
        ax.set_ylabel(lab)
    axes[0].set_title("Fig. Cost per seed, 30 passes (Kaggle T4). Retrain = one training on all 80 classes", fontsize=9.5)
    save(fig, "fig5_cost")

    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s["verdict"], indent=1))


if __name__ == "__main__":
    main()
