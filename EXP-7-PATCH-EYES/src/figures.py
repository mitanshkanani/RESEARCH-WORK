"""Draw the EXP-7 figures from the real result file and write the summary tables and the verdict.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes figures/*.svg and results/summary.json. Every mark is a mean over seeds read from
results/runs.jsonl. One combination (answer rule + memory) of the fixed patch eyes with random
detectors is selected on the practice half and judged on the exam half, as fixed in JOURNAL.md.
Reference numbers of the backprop-trained eyes are quoted from EXP-6's summary.json.
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
# EXP-6, exam half: all-80 score, 3-picture surprise test, training seconds, GPU Wh
REF = {"frozen": dict(all=33.17, few3=25.63, sec=25.7, wh=0.269),
       "grow_lock": dict(all=39.63, few3=29.67, sec=130.7, wh=1.364),
       "retrain": dict(all=53.19, few3=38.5, sec=202.2, wh=2.123)}
E1 = ["E1_rp_tri", "E1_rp_t0", "E1_rp_t0.1", "E1_rp_t0.25"]
ANAME = {"E1_rp_tri": "fixed, random patches,\ntriangle rule", "E1_rp_t0": "fixed, random,\nthreshold 0",
         "E1_rp_t0.1": "fixed, random,\nthreshold 0.1", "E1_rp_t0.25": "fixed, random,\nthreshold 0.25",
         "E1_km_tri": "fixed, K-means,\ntriangle", "E1_small_tri": "fixed, 400\ndetectors, triangle",
         "E3_all_tri": "patches from all\nclasses (diagnosis)", "E2_grow_t0.1": "growing,\nthreshold 0.1"}

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


def refs(ax, key="all"):
    for name, lab, c in [("retrain", "retrain everything, backprop", AMBER), ("grow_lock", "grow and lock, backprop", RED),
                         ("frozen", "frozen backprop eyes", BLUE)]:
        ax.axhline(REF[name][key], color=c, lw=1.4, ls=(0, (4, 3)), label=f"{lab} {REF[name][key]:.1f} (EXP-6)")


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "runs.jsonl"
    rows = [json.loads(l) for l in open(src)]
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["rule"], r["stage"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["seed"])
    S = max(r["stage"] for r in rows)
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    rules = {a: list(dict.fromkeys(r["rule"] for r in rows if r["arm"] == a and r["stage"] == S)) for a in arms}

    def cell(a, ru, sp):
        recs, first = by[(a, ru, S)], by[(a, ru, 1)]
        out = dict(all=ms([r[sp]["acc_all"] for r in recs]), per_seed=[round(r[sp]["acc_all"] * 100, 2) for r in recs],
                   first=ms([r[sp]["acc_groups"][0] for r in recs]),
                   groups=[round(float(np.mean([r[sp]["acc_groups"][g] for r in recs])) * 100, 2) for g in range(S)],
                   first_own_end=ms([r["first_own"][sp] for r in recs]))
        if first:
            out["first_own_start"] = ms([r["first_own"][sp] for r in first])
            out["first_own_drop"] = round(out["first_own_start"][0] - out["first_own_end"][0], 2)
        if "few_shot" in recs[0]:
            out["few_shot"] = {k: ms([r["few_shot"][k] for r in recs]) for k in recs[0]["few_shot"]}
        return out

    def cost(a, ru):
        recs = by[(a, ru, S)]
        out = {k: ms([r[k] for r in recs], 1.0, 2) for k in ("build_seconds", "describe_train_seconds")}
        for k in ("build_wh", "describe_train_wh"):
            if all(r.get(k) is not None for r in recs):
                out[k] = ms([r[k] for r in recs], 1.0, 4)
        for k in ("tables_seconds", "solve_seconds"):
            if all(k in r for r in recs):
                out[k] = ms([r[k] for r in recs], 1.0, 2)
        out["detectors"], out["feature_dim"] = recs[0]["detectors"], recs[0]["feature_dim"]
        if recs[0].get("peak_mem_mb") is not None:
            out["peak_mem_mb"] = max(r["peak_mem_mb"] for r in recs)
        return out

    s = {"source": src.name, "seeds": len(by[(arms[0], rules[arms[0]][0], S)]), "stages": S, "reference_exp6": REF,
         "final": {sp: {a: {ru: cell(a, ru, sp) for ru in rules[a]} for a in arms} for sp in ("exam", "practice", "full")},
         "cost": {a: {ru: cost(a, ru) for ru in rules[a]} for a in arms}, "over_time": {}}
    san = RESULTS / "sanity.json"
    if san.exists():
        s["sanity"] = json.loads(san.read_text())

    # ---- selection on the practice half, verdict on the exam half
    pr, ex = s["final"]["practice"], s["final"]["exam"]
    cands = [(a, ru) for a in E1 if a in arms for ru in rules[a]]
    ok = [(a, ru) for a, ru in cands if pr[a][ru].get("first_own_drop", 99) <= 3.0]
    sel = max(ok, key=lambda t: pr[t[0]][t[1]]["all"][0]) if ok else None
    v = {"selection": {"candidates": len(cands), "passed_step2_on_practice": len(ok),
                       "selected_arm": sel[0] if sel else None, "selected_rule": sel[1] if sel else None,
                       "practice_top5": sorted(([a, ru, pr[a][ru]["all"][0]] for a, ru in ok), key=lambda t: -t[2])[:5]},
         "step0_cifar10": s.get("sanity", {})}
    if sel:
        c = ex[sel[0]][sel[1]]
        x = c["all"][0]
        gl, rt = REF["grow_lock"]["all"], REF["retrain"]["all"]
        v["step1_score"] = {"selected": x, "sd": c["all"][1], "per_seed": c["per_seed"],
                            "share_of_gap": round((x - gl) / (rt - gl), 3),
                            "verdict": "fails" if x < REF["frozen"]["all"] else "weaker than grow and lock" if x < gl - 2 else
                            "matches grow and lock" if x < gl + 0.25 * (rt - gl) else
                            "partly closes the gap" if x < gl + 0.5 * (rt - gl) else "works"}
        v["step2_forgetting"] = {"first_own_start": c["first_own_start"][0], "first_own_end": c["first_own_end"][0],
                                 "drop": c["first_own_drop"], "passes": bool(c["first_own_drop"] <= 3.0),
                                 "first_group_among_all": c["first"][0]}
        v["few_shot_3_selected_arm"] = {ru: ex[sel[0]][ru]["few_shot"]["3"][0] for ru in rules[sel[0]] if "few_shot" in ex[sel[0]][ru]}
        v["cost_selected"] = s["cost"][sel[0]][sel[1]]
    s["verdict"] = v

    def best(a, prefix, sp="practice"):
        c = [ru for ru in rules[a] if ru.startswith(prefix)]
        return max(c, key=lambda ru: s["final"][sp][a][ru]["all"][0]) if c else None

    arm0 = sel[0] if sel else E1[0]
    # fig 1: memories on the selected fixed patch eyes
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    mem = [("cos", "plain cards"), ("sqrt_cos", "cards after\nsquare root"), (best(arm0, "lda"), "cards + shared\ntable (lda)"),
           (best(arm0, "ridge"), "one-step straight-line\nclassifier (ridge)")]
    mem = [(ru, lab) for ru, lab in mem if ru]
    vals = [ex[arm0][ru]["all"] for ru, _ in mem]
    ax.bar(range(len(mem)), [a[0] for a in vals], yerr=[a[1] for a in vals], color=GREEN, width=0.6,
           error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for i, a in enumerate(vals):
        ax.text(i, a[0] + a[1] + 0.6, f"{a[0]:.1f}", ha="center", fontsize=8.5)
    refs(ax)
    ax.set_xticks(range(len(mem)))
    ax.set_xticklabels([lab for _, lab in mem], fontsize=8)
    ax.set_ylim(0, max(REF["retrain"]["all"], max(a[0] for a in vals)) * 1.14)
    ax.set_ylabel("accuracy on all 80 classes, %")
    ax.set_title(f"Fig. Patch eyes ({ANAME[arm0].replace(chr(10), ' ')}) with four memories. Exam half, 5 seeds, bar = 1 s.d.", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig1_memories")

    # fig 2: every kind of eyes, plain cards and best table memory
    fixed = [a for a in arms if a != "E2_grow_t0.1"]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(len(fixed))
    v1 = [ex[a]["sqrt_cos"]["all"] for a in fixed]
    bestru = [max((ru for ru in rules[a] if ru.startswith(("lda", "ridge"))), key=lambda ru: pr[a][ru]["all"][0]) for a in fixed]
    v2 = [ex[a][ru]["all"] for a, ru in zip(fixed, bestru)]
    ax.bar(x - 0.19, [a[0] for a in v1], 0.36, yerr=[a[1] for a in v1], color=NEUTRAL, error_kw=dict(ecolor=INK2, lw=1, capsize=2),
           label="cards after square root")
    ax.bar(x + 0.19, [a[0] for a in v2], 0.36, yerr=[a[1] for a in v2], color=GREEN, error_kw=dict(ecolor=INK2, lw=1, capsize=2),
           label="best table memory (picked on the practice half)")
    for i in range(len(fixed)):
        ax.text(i - 0.19, v1[i][0] + v1[i][1] + 0.5, f"{v1[i][0]:.1f}", ha="center", fontsize=8, color=INK2)
        ax.text(i + 0.19, v2[i][0] + v2[i][1] + 0.5, f"{v2[i][0]:.1f}", ha="center", fontsize=8)
    refs(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([ANAME[a] for a in fixed], fontsize=7.3)
    ax.set_ylim(0, max(REF["retrain"]["all"], max(a[0] for a in v2)) * 1.14)
    ax.set_ylabel("accuracy on all 80 classes, %")
    ax.set_title("Fig. Every kind of patch eyes. Exam half, after the last stage, 5 seeds", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig2_eyes")

    # fig 3: score after every stage for the selected eyes
    stages = list(range(1, S + 1))
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for ru, lab, col, mk in [("sqrt_cos", "cards after square root", NEUTRAL, "o"), (best(arm0, "lda"), "cards + shared table", GREEN, "D"),
                             (best(arm0, "ridge"), "ridge", VIOLET, "s")]:
        if not ru:
            continue
        pts = [(st, ms([r["exam"]["acc_all"] for r in by[(arm0, ru, st)]])[0]) for st in stages if by[(arm0, ru, st)]]
        s["over_time"][ru] = pts
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, markeredgecolor=SURF, markeredgewidth=1, label=lab)
        ax.text(pts[-1][0] + 0.12, pts[-1][1], f"{pts[-1][1]:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_xticks(stages)
    ax.set_xlim(0.8, S + 0.8)
    ax.set_xlabel("stage (10 new classes arrive at each stage)")
    ax.set_ylabel("accuracy on all classes so far, %")
    ax.set_title("Fig. Score after each stage, selected patch eyes, exam half", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig3_over_time")

    # fig 4: three-picture test on surprise classes
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    fr = [(ru, lab) for ru, lab in [("cos", "plain cards"), ("sqrt_cos", "cards after\nsquare root"),
                                    (best(arm0, "lda"), "cards + shared\ntable")] if ru and "few_shot" in ex[arm0][ru]]
    vals = [ex[arm0][ru]["few_shot"]["3"] for ru, _ in fr]
    ax.bar(range(len(fr)), [a[0] for a in vals], yerr=[a[1] for a in vals], color=GREEN, width=0.55, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for i, a in enumerate(vals):
        ax.text(i, a[0] + a[1] + 0.4, f"{a[0]:.1f}", ha="center", fontsize=8.5)
    refs(ax, "few3")
    ax.set_xticks(range(len(fr)))
    ax.set_xticklabels([lab for _, lab in fr], fontsize=8)
    ax.set_ylim(0, max(REF["retrain"]["few3"], max(a[0] for a in vals)) * 1.15)
    ax.set_ylabel("accuracy on 20 surprise classes, %")
    ax.set_title("Fig. Learning 20 never-seen classes from 3 pictures each, selected patch eyes", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig4_surprise")

    # fig 5: cost of learning all 80 classes
    if sel:
        c = s["cost"][sel[0]][sel[1]]
        parts = [("build the eyes", c["build_seconds"][0]), ("describe every training picture once", c["describe_train_seconds"][0]),
                 ("update the tables", c.get("tables_seconds", [0])[0]), ("solve once", c.get("solve_seconds", [0])[0])]
        fig, ax = plt.subplots(figsize=(8.0, 2.6))
        left = 0.0
        for (lab, val), col in zip(parts, [BLUE, GREEN, VIOLET, NEUTRAL]):
            ax.barh([2], [val], left=[left], color=col, height=0.55, label=f"{lab}: {val:.1f} s")
            left += val
        ax.text(left + 2, 2, f"{left:.0f} s", va="center", fontsize=8.5)
        ax.barh([1], [REF["grow_lock"]["sec"]], color=RED, height=0.55)
        ax.text(REF["grow_lock"]["sec"] + 2, 1, f"{REF['grow_lock']['sec']:.0f} s", va="center", fontsize=8.5)
        ax.barh([0], [REF["retrain"]["sec"]], color=AMBER, height=0.55)
        ax.text(REF["retrain"]["sec"] + 2, 0, f"{REF['retrain']['sec']:.0f} s", va="center", fontsize=8.5)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["patch eyes (selected)", "grow and lock (EXP-6)", "retrain once (EXP-6)"], fontsize=8)
        ax.grid(False)
        ax.set_xlabel("seconds per seed on a Kaggle T4, to learn all 80 classes")
        ax.set_title("Fig. Cost of learning", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        s["verdict"]["cost_selected"]["total_seconds"] = round(left, 1)
        save(fig, "fig5_cost")

    # fig 6: growing eyes
    if "E2_grow_t0.1" in arms and "E1_rp_t0.1" in arms:
        g = ex["E2_grow_t0.1"]
        fig, ax = plt.subplots(figsize=(7.4, 3.3))
        items = [("fixed eyes\n(all detectors from\nthe first group)", ex["E1_rp_t0.1"]["sqrt_cos"]["all"], GREEN),
                 ("growing eyes,\nold cards on their\nown units (R0)", g["sqrt_R0"]["all"], NEUTRAL),
                 ("growing eyes,\nold cards filled in\n(R7)", g["sqrt_R7"]["all"], VIOLET),
                 ("growing eyes,\nopen-book\n(not allowed)", g["sqrt_open"]["all"], AMBER)]
        ax.bar(range(4), [i[1][0] for i in items], yerr=[i[1][1] for i in items], color=[i[2] for i in items], width=0.6,
               error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        for i, it in enumerate(items):
            ax.text(i, it[1][0] + it[1][1] + 0.4, f"{it[1][0]:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels([i[0] for i in items], fontsize=7.6)
        ax.set_ylabel("accuracy on all 80 classes, %")
        ax.set_title("Fig. Fixed against growing patch eyes (threshold 0.1, cards after square root), exam half", fontsize=9.5)
        save(fig, "fig6_growing")

    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
