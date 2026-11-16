"""Draw the EXP-8 figures from the real result file and write the summary tables and the verdict.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes figures/*.svg and results/summary.json. Every mark is a mean over seeds read from
results/runs.jsonl. For each kind of eyes the memory is chosen on the practice half and scored on
the exam half, as fixed in JOURNAL.md sections 3 and 5.
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
EYES = ["P", "F", "G", "R"]
ENAME = {"P": "patch eyes\n(no backprop)", "F": "frozen\nbackprop eyes", "G": "grow and lock", "R": "retrain everything\n(reference)"}
ECOL = {"P": GREEN, "F": BLUE, "G": RED, "R": AMBER}
EARLIER = {"P_lda_g0.01": 55.2, "F_cos": 33.2, "G_R7": 39.6, "R_cos": 53.2}      # EXP-7 / EXP-6, exam half

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


def gval(rule):
    return float(rule.rsplit("_g", 1)[1])


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "runs.jsonl"
    rows = [json.loads(l) for l in open(src)]
    by = defaultdict(list)
    for r in rows:
        by[(r["eyes"], r["rule"], r["stage"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["seed"])
    S = max(r["stage"] for r in rows)
    rules = {e: list(dict.fromkeys(r["rule"] for r in rows if r["eyes"] == e and r["stage"] == S)) for e in EYES}

    def cell(e, ru, sp):
        recs, first = by[(e, ru, S)], by[(e, ru, 1)]
        out = {}
        if sp in recs[0]:
            out.update(all=ms([r[sp]["acc_all"] for r in recs]), per_seed=[round(r[sp]["acc_all"] * 100, 2) for r in recs],
                       first=ms([r[sp]["acc_groups"][0] for r in recs]),
                       groups=[round(float(np.mean([r[sp]["acc_groups"][g] for r in recs])) * 100, 2) for g in range(S)])
        if "first_own" in recs[0]:
            out["first_own_end"] = ms([r["first_own"][sp] for r in recs])
            if first and "first_own" in first[0]:
                out["first_own_start"] = ms([r["first_own"][sp] for r in first])
                out["first_own_drop"] = round(out["first_own_start"][0] - out["first_own_end"][0], 2)
        if "few_shot" in recs[0]:
            out["few_shot"] = {k: ms([r["few_shot"][k] for r in recs]) for k in recs[0]["few_shot"]}
        return out

    def cost(e, ru):
        recs = by[(e, ru, S)]
        out = {}
        for k in ("train_seconds", "train_wh", "build_seconds", "build_wh", "describe_train_seconds", "describe_train_wh",
                  "tables_seconds", "tables_wh", "solve_seconds", "solve_wh", "peak_mem_mb"):
            if all(r.get(k) is not None for r in recs):
                out[k] = ms([r[k] for r in recs], 1.0, 4)
        out["feature_dim"] = recs[0].get("feature_dim")
        return out

    s = {"source": src.name, "seeds": len(by[("P", "cos", S)]), "stages": S,
         "final": {sp: {e: {ru: cell(e, ru, sp) for ru in rules[e]} for e in EYES} for sp in ("exam", "practice", "full")},
         "cost": {e: {ru: cost(e, ru) for ru in rules[e] if not ru.startswith("fs_")} for e in EYES}}
    ex, pr = s["final"]["exam"], s["final"]["practice"]

    def is_table(e, ru):
        return ru.startswith("lda_fill_g") if e == "G" else ru.startswith(("lda_g", "ridge_g"))

    def pick(e, pred):
        c = [ru for ru in rules[e] if pred(ru) and "all" in pr[e][ru] and pr[e][ru].get("first_own_drop", 0) <= 3.0]
        return max(c, key=lambda ru: pr[e][ru]["all"][0]) if c else None

    best = {e: pick(e, lambda ru, e=e: is_table(e, ru)) for e in EYES}
    plain = {"P": "cos", "F": "cos", "G": "R7", "R": "cos"}
    g_open = pick("G", lambda ru: ru.startswith(("lda_open_g", "ridge_open_g")))

    def fs_all(e):
        out = {}
        for ru in rules[e]:
            c = ex[e][ru]
            if "few_shot" in c:
                out[ru[3:] if ru.startswith("fs_") else ru] = c["few_shot"]["3"][0]
        return out

    fs = {e: fs_all(e) for e in EYES}
    sc = lambda e, ru: ex[e][ru]["all"][0]
    own = sc("R", "own_layer")
    ceiling = max(own, sc("R", best["R"]))
    p_best = sc("P", best["P"])
    dist = round(ceiling - p_best, 2)
    lda_end = lambda e: min(GV := sorted({gval(r) for r in rules[e] if is_table(e, r)}))
    v = {
        "selected_memory": best, "plain_memory": plain, "grow_open_book_best": g_open,
        "step0_rebuild": {"P_lda_g0.01": sc("P", "lda_g0.01"), "F_cos": sc("F", "cos"), "G_R7": sc("G", "R7"), "R_cos": sc("R", "cos"),
                          "earlier": EARLIER,
                          "passes": bool(abs(sc("P", "lda_g0.01") - 55.2) <= 3 and abs(sc("F", "cos") - 33.2) <= 3 and
                                         abs(sc("G", "R7") - 39.6) <= 3 and abs(sc("R", "cos") - 53.2) <= 3)},
        "step1_ceiling": {"retrain_own_answer_layer": own, "retrain_best_table": sc("R", best["R"]), "ceiling": ceiling},
        "step2_patch_vs_ceiling": {"patch_best": p_best, "sd": ex["P"][best["P"]]["all"][1], "below_ceiling": dist,
                                   "verdict": "matches the trained network" if dist <= 3 else
                                   "close, at a fraction of the cost" if dist <= 10 else "clearly behind"},
        "step3_grow_and_lock": {"R7_cards": sc("G", "R7"), "filled_in_tables": sc("G", best["G"]) if best["G"] else None,
                                "gain": round(sc("G", best["G"]) - sc("G", "R7"), 2) if best["G"] else None,
                                "held_back_by_memory": bool(best["G"] and sc("G", best["G"]) - sc("G", "R7") >= 5),
                                "open_book_tables": sc("G", g_open) if g_open else None, "open_book_cards": sc("G", "open_cos")},
        "step4_frozen": {"plain_cards": sc("F", "cos"), "best_table": sc("F", best["F"]),
                         "gain": round(sc("F", best["F"]) - sc("F", "cos"), 2),
                         "memory_was_a_bottleneck": bool(sc("F", best["F"]) - sc("F", "cos") >= 5)},
        "step5_few_shot_3": {e: {"all": fs[e], "best": max(fs[e].values()), "best_rule": max(fs[e], key=fs[e].get)} for e in EYES},
        "same_memory_ranking": sorted(((e, sc(e, best[e])) for e in EYES if best[e]), key=lambda t: -t[1]),
        "smallest_smoothing_is_best": {e: bool(best[e] and gval(best[e]) == lda_end(e)) for e in EYES},
        "forgetting_selected": {e: ex[e][best[e]].get("first_own_drop") for e in EYES if best[e]},
    }

    def total_cost(e, ru, tables=True):
        c = s["cost"][e][ru]
        parts = ["train", "build", "describe_train"] + (["tables", "solve"] if tables else [])
        t = sum(c.get(k + "_seconds", [0])[0] for k in parts)
        wh = sum(c.get(k + "_wh", [0])[0] for k in parts)
        return round(t, 1), round(wh, 3), c.get("peak_mem_mb", [None])[0]

    v["cost_selected"] = {e: dict(zip(("seconds", "wh", "peak_mem_mb"), total_cost(e, best[e]))) for e in EYES if best[e]}
    v["cost_plain"] = {e: dict(zip(("seconds", "wh", "peak_mem_mb"), total_cost(e, plain[e], tables=False))) for e in EYES}
    s["verdict"] = v

    # fig 1: plain cards against the best table memory, per kind of eyes
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(EYES))
    a = [ex[e][plain[e]]["all"] for e in EYES]
    b = [ex[e][best[e]]["all"] for e in EYES]
    ax.bar(x - 0.19, [t[0] for t in a], 0.36, yerr=[t[1] for t in a], color=NEUTRAL, error_kw=dict(ecolor=INK2, lw=1, capsize=2),
           label="plain cards (as in EXP-2 to EXP-6)")
    ax.bar(x + 0.19, [t[0] for t in b], 0.36, yerr=[t[1] for t in b], color=[ECOL[e] for e in EYES],
           error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    ax.bar([0], [0], color=INK2, label="best table memory, chosen on the practice half")
    for i in range(len(EYES)):
        ax.text(i - 0.19, a[i][0] + a[i][1] + 0.6, f"{a[i][0]:.1f}", ha="center", fontsize=8.5, color=INK2)
        ax.text(i + 0.19, b[i][0] + b[i][1] + 0.6, f"{b[i][0]:.1f}", ha="center", fontsize=8.5)
    ax.axhline(own, color=INK, lw=1.4, ls=(0, (4, 3)), label=f"retrained network with its own answer layer {own:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels([ENAME[e] for e in EYES], fontsize=8)
    ax.set_ylim(0, max(ceiling, max(t[0] for t in b)) * 1.15)
    ax.set_ylabel("accuracy on all 80 classes, %")
    ax.set_title("Fig. The same memories on every kind of eyes. Exam half, 5 seeds, bar = 1 s.d.", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig1_same_memory")

    # fig 2: effect of the smoothing value
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for e in EYES:
        pre = "lda_fill_g" if e == "G" else "lda_g"
        pts = sorted((gval(ru), ex[e][ru]["all"][0]) for ru in rules[e] if ru.startswith(pre))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ECOL[e], marker="o", markeredgecolor=SURF, markeredgewidth=1,
                label=ENAME[e].replace("\n", " ") + (" (filled-in table)" if e == "G" else ""))
    ax.set_xscale("log")
    ax.set_xlabel("smoothing of the shared table (times the average spread)")
    ax.set_ylabel("accuracy on all 80 classes, %")
    ax.set_title("Fig. Cards + shared table: score against smoothing, exam half", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig2_smoothing")

    # fig 3: three-picture surprise test
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    width = 0.26
    for j, (lab, key, col) in enumerate([("plain cards", "cos", NEUTRAL), ("cards after square root", "sqrt_cos", MUTED),
                                         ("cards + shared table, best smoothing tried", None, None)]):
        vals = []
        for e in EYES:
            if key:
                vals.append(fs[e].get(key, fs[e].get("open_cos", 0.0)))
            else:
                vals.append(max([val for k, val in fs[e].items() if k.startswith("lda")], default=0.0))
        ax.bar(x + (j - 1) * width, vals, width, color=col if col else [ECOL[e] for e in EYES], label=lab if col else None)
        for i, val in enumerate(vals):
            ax.text(i + (j - 1) * width, val + 0.4, f"{val:.1f}", ha="center", fontsize=8)
    ax.bar([0], [0], color=INK2, label="cards + shared table, best smoothing tried")
    ax.set_xticks(x)
    ax.set_xticklabels([ENAME[e] for e in EYES], fontsize=8)
    ax.set_ylabel("accuracy on 20 surprise classes, %")
    ax.set_title("Fig. Learning 20 never-seen classes from 3 pictures each, mean of 5 seeds", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig3_surprise")

    # fig 4: cost against score for each full system
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for e in EYES:
        t, wh, _ = total_cost(e, best[e])
        ax.scatter([t], [sc(e, best[e])], s=70, color=ECOL[e], edgecolor=SURF, linewidth=1.2, zorder=3)
        ax.annotate(f"{ENAME[e].replace(chr(10), ' ')}\n{sc(e, best[e]):.1f}%, {t:.0f} s, {wh:.2f} Wh", (t, sc(e, best[e])),
                    textcoords="offset points", xytext=(8, -4), fontsize=8, color=INK2)
    ax.set_xlabel("seconds per seed to learn all 80 classes (training or describing, plus the tables), Kaggle T4")
    ax.set_ylabel("accuracy on all 80 classes, %")
    ax.set_xlim(0, None)
    ax.set_title("Fig. Score against cost, each kind of eyes with its best table memory", fontsize=9.5)
    save(fig, "fig4_cost")

    # fig 5: grow and lock in detail
    items = [("cards, own units\n(R0, EXP-3)", "R0", NEUTRAL), ("cards filled in\n(R7, EXP-4)", "R7", RED),
             ("filled-in tables\n(real method)", best["G"], GREEN), ("open-book cards\n(not allowed)", "open_cos", MUTED),
             ("open-book tables\n(not allowed)", g_open, AMBER)]
    items = [(l, r, c) for l, r, c in items if r]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    vals = [ex["G"][r]["all"] for _, r, _ in items]
    ax.bar(range(len(items)), [t[0] for t in vals], yerr=[t[1] for t in vals], color=[c for _, _, c in items], width=0.6,
           error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for i, t in enumerate(vals):
        ax.text(i, t[0] + t[1] + 0.5, f"{t[0]:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([l for l, _, _ in items], fontsize=7.8)
    ax.set_ylabel("accuracy on all 80 classes, %")
    ax.set_title("Fig. Grow and lock with each memory, exam half", fontsize=9.5)
    save(fig, "fig5_grow_and_lock")

    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
