"""Draw the EXP-4 figures from the real result file and write the summary tables and the verdict.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes figures/*.svg and results/summary.json. Every mark is a mean over seeds read from
results/runs.jsonl. The rule is selected on the practice half and judged on the exam half,
as fixed in JOURNAL.md section 6.
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
RULES = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
RNAME = {"R0": "R0 as in EXP-3", "R1": "R1 blank fill", "R2": "R2 group-centred", "R3": "R3 group-standardised",
         "R4": "R4 stranger (running)", "R5": "R5 stranger (same crowd)", "R6": "R6 spread-aware",
         "R7": "R7 predicted fill-in"}
BLUE, RED, GREEN, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, SURF, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#b9b8b0"

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


def ms(v):
    v = np.array(v, dtype=float) * 100
    return [round(float(v.mean()), 2), round(float(v.std()), 2)]


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "runs.jsonl"
    rows = [json.loads(l) for l in open(src)]
    by = defaultdict(list)                      # (dataset, way, rule, stage) -> records over seeds
    for r in rows:
        by[(r["dataset"], r["way"], r["rule"], r["stage"])].append(r)
    summary = {"source": src.name, "datasets": {}}

    for ds in sorted({r["dataset"] for r in rows}):
        S = max(r["stage"] for r in rows if r["dataset"] == ds)
        stages = list(range(1, S + 1))

        def cell(way, rule, stage, split):
            rs = sorted(by[(ds, way, rule, stage)], key=lambda r: r["seed"])
            g = np.array([r[split]["acc_groups"] for r in rs]) * 100
            older = np.array([r[split]["err_older"] for r in rs], dtype=float)
            newer = np.array([r[split]["err_newer"] for r in rs], dtype=float)
            same = np.array([r[split]["err_same"] for r in rs], dtype=float)
            cross = np.maximum(older + newer, 1)
            wrong = np.maximum(older + newer + same, 1)
            return dict(all=ms([r[split]["acc_all"] for r in rs]), first=ms([r[split]["acc_groups"][0] for r in rs]),
                        per_seed_all=[round(100 * r[split]["acc_all"], 2) for r in rs],
                        groups=[round(float(x), 2) for x in g.mean(0)],
                        older_share_of_cross_group_errors=ms(older / cross),
                        wrong_older=ms(older / wrong), wrong_same=ms(same / wrong), wrong_newer=ms(newer / wrong))

        out = {"stages": S, "final": {}, "over_time": {}}
        for split in ("full", "practice", "exam"):
            f = {"frozen": cell("frozen", "R0", S, split), "retrain": cell("retrain", "open", S, split),
                 "open_book": cell("grow", "open", S, split), "rules": {r: cell("grow", r, S, split) for r in RULES}}
            out["final"][split] = f
        for key, way, rule in [("frozen", "frozen", "R0"), ("open_book", "grow", "open")] + [(r, "grow", r) for r in RULES]:
            out["over_time"][key] = [cell(way, rule, s, "exam")["all"] for s in stages]

        # selection on the practice half: best all-classes score among rules that keep the first group
        pr = out["final"]["practice"]
        ok = [r for r in RULES if pr["rules"][r]["first"][0] >= pr["frozen"]["first"][0] - 5]
        chosen = max(ok, key=lambda r: pr["rules"][r]["all"][0]) if ok else None
        ex, fu = out["final"]["exam"], out["final"]["full"]
        sel = dict(passed_step3_on_practice=ok, selected=chosen,
                   practice_scores={r: pr["rules"][r]["all"][0] for r in RULES})
        verdict = dict(selection=sel)
        if chosen:
            fro, ret, got = ex["frozen"]["all"][0], ex["retrain"]["all"][0], ex["rules"][chosen]["all"][0]
            r0, op = ex["rules"]["R0"]["all"][0], ex["open_book"]["all"][0]
            gap = ret - fro
            share = (got - fro) / gap if gap >= 3 else None
            per_seed = [round((g - f) / (t - f), 3) for g, f, t in zip(
                ex["rules"][chosen]["per_seed_all"], ex["frozen"]["per_seed_all"], ex["retrain"]["per_seed_all"])]
            keeps = ex["rules"][chosen]["first"][0] >= ex["frozen"]["first"][0] - 5
            cause = ex["rules"]["R0"]["older_share_of_cross_group_errors"][0]
            verdict.update(
                step0_rebuild_full_test=dict(R0=fu["rules"]["R0"]["all"][0], frozen=fu["frozen"]["all"][0],
                                             retrain=fu["retrain"]["all"][0], open_book=fu["open_book"]["all"][0]),
                step1_cause=dict(older_share_pct_exam=cause, older_share_pct_full=fu["rules"]["R0"]["older_share_of_cross_group_errors"][0],
                                 conclusion="confirmed" if cause >= 65 else "unclear" if cause >= 55 else "not supported"),
                step2_gap=dict(frozen=fro, retrain=ret, selected_rule_score=got, R0=r0, open_book=op, gap=round(gap, 2),
                               share=None if share is None else round(share, 3), share_per_seed=per_seed,
                               beats_frozen=got > fro, lost_points_won_back=round(got - r0, 2),
                               share_of_lost_points=round((got - r0) / (op - r0), 3) if op - r0 > 0 else None),
                step3_keeps_old=dict(selected_first_group=ex["rules"][chosen]["first"][0],
                                     frozen_first_group=ex["frozen"]["first"][0], passes=bool(keeps)),
                verdict=("no verdict: gap under 3 points" if share is None else "does not work" if (share < 0.25 or not keeps)
                         else "partly works" if share < 0.5 else "works"))
        out["verdict"] = verdict
        # cost of each rule: scoring time on the final test set and what the memory keeps
        out["score_ms"] = {r: round(float(np.mean([x["score_ms"] for x in by[(ds, "grow", r, S)]])), 2) for r in RULES}
        out["memory_fit_seconds"] = round(float(np.mean([x["memory_fit_seconds"] for x in by[(ds, "grow", "R0", S)]])), 3)
        out["train_seconds"] = {w: round(float(np.mean([x["train_seconds"] for x in by[(ds, w, ru, S)]])), 1)
                                for w, ru in [("frozen", "R0"), ("retrain", "open")]}
        out["train_seconds"]["grow_all_stages"] = round(float(np.mean(
            [sum(x["train_seconds"] for s in stages for x in by[(ds, "grow", "R0", s)] if x["seed"] == seed)
             for seed in sorted({x["seed"] for x in by[(ds, "grow", "R0", S)]})])), 1)
        summary["datasets"][ds] = out

        label = "Photos (CIFAR-100)" if ds == "cifar100" else "Digits, then clothes"
        short = "photos" if ds == "cifar100" else "digits_clothes"

        # fig A: all-classes score of every rule on the exam half, with the three reference levels
        fig, ax = plt.subplots(figsize=(8.6, 3.6))
        vals = [ex["rules"][r]["all"][0] for r in RULES]
        errs = [ex["rules"][r]["all"][1] for r in RULES]
        pr = out["final"]["practice"]["rules"]
        top = max(RULES, key=lambda r: pr[r]["all"][0])  # highest practice-half score, ignoring step 3
        cols = [RED if r == "R0" else GREEN if r in (chosen, top) else NEUTRAL for r in RULES]
        ax.bar(range(len(RULES)), vals, yerr=errs, color=cols, width=0.62, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        for name, val, c in [("retrain everything (ceiling)", ex["retrain"]["all"][0], AMBER),
                             ("open-book, same eyes (not allowed)", ex["open_book"]["all"][0], INK2),
                             ("frozen eyes (floor)", ex["frozen"]["all"][0], BLUE)]:
            ax.axhline(val, color=c, lw=1.4, ls=(0, (4, 3)), label=f"{name} {val:.1f}")
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.set_xticks(range(len(RULES)))
        ax.set_xticklabels([RNAME[r].replace(" ", "\n", 1).replace(" (", "\n(") for r in RULES], fontsize=7)
        ax.set_xlim(-0.6, len(RULES) - 0.4)
        ax.set_ylim(0, max(ex["retrain"]["all"][0], max(vals)) * 1.12)
        ax.set_ylabel("accuracy on all classes, %")
        ax.set_title(f"Fig. Overall score of each comparison rule. {label}, exam half, mean of 5 seeds, bar = 1 s.d.\n"
                     f"red = the EXP-3 rule (R0). Selected by the written rule: {chosen}. "
                     f"Highest score on the practice half: {top}", fontsize=9.5)
        save(fig, f"fig1_rules_{short}")

        # fig B: score per group, oldest to newest
        fig, ax = plt.subplots(figsize=(7.4, 3.5))
        gx = list(range(1, S + 1))
        series = [("frozen eyes", ex["frozen"]["groups"], BLUE, "o", "-"), ("R0 as in EXP-3", ex["rules"]["R0"]["groups"], RED, "s", "-")]
        if chosen and chosen != "R0":
            series.append((f"{RNAME[chosen]} (selected)", ex["rules"][chosen]["groups"], GREEN, "D", "-"))
        if top not in ("R0", chosen):
            series.append((f"{RNAME[top]} (highest score)", ex["rules"][top]["groups"], GREEN, "D", "-"))
        series.append(("open-book (not allowed)", ex["open_book"]["groups"], AMBER, "^", (0, (4, 2))))
        for name, y, c, m, ls in series:
            ax.plot(gx, y, color=c, marker=m, ls=ls, markeredgecolor=SURF, markeredgewidth=1, label=name)
        ax.set_xticks(gx)
        ax.set_xlabel("group of classes (1 = learned first, oldest card; last = newest card)")
        ax.set_ylabel("accuracy on that group, %")
        ax.set_ylim(0, None)
        ax.set_title(f"Fig. Is any age of class favoured? {label}, exam half, after the last stage", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        save(fig, f"fig2_per_group_{short}")

        # fig C: where the wrong answers go
        fig, ax = plt.subplots(figsize=(7.4, 3.3))
        names = RULES + ["open"]
        cells = [ex["rules"][r] for r in RULES] + [ex["open_book"]]
        left = np.zeros(len(names))
        for key, c, lab in [("wrong_older", RED, "to a class from an OLDER group"), ("wrong_same", NEUTRAL, "to a class of the SAME group"),
                            ("wrong_newer", BLUE, "to a class from a NEWER group")]:
            v = np.array([x[key][0] for x in cells])
            ax.barh(range(len(names)), v, left=left, color=c, height=0.62, edgecolor=SURF, linewidth=1.5, label=lab)
            for i, (a, b) in enumerate(zip(left, v)):
                if b >= 7:
                    ax.text(a + b / 2, i, f"{b:.0f}%", ha="center", va="center", fontsize=8, color=INK)
            left += v
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([RNAME[r] for r in RULES] + ["open-book (not allowed)"], fontsize=8)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of the wrong answers, %")
        ax.set_title(f"Fig. Where do the wrong answers go? {label}, exam half, after the last stage", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.2))
        save(fig, f"fig3_wrong_answers_{short}")

        # fig D: overall score after each stage
        if S > 2:
            fig, ax = plt.subplots(figsize=(7.4, 3.5))
            series = [("frozen eyes", "frozen", BLUE, "o", "-"), ("R0 as in EXP-3", "R0", RED, "s", "-")]
            if chosen and chosen != "R0":
                series.append((f"{RNAME[chosen]} (selected)", chosen, GREEN, "D", "-"))
            if top not in ("R0", chosen):
                series.append((f"{RNAME[top]} (highest score)", top, GREEN, "D", "-"))
            series.append(("open-book (not allowed)", "open_book", AMBER, "^", (0, (4, 2))))
            for name, key, c, m, ls in series:
                y = np.array([v[0] for v in out["over_time"][key]])
                sd = np.array([v[1] for v in out["over_time"][key]])
                ax.fill_between(stages, y - sd, y + sd, color=c, alpha=0.12, lw=0)
                ax.plot(stages, y, color=c, marker=m, ls=ls, markeredgecolor=SURF, markeredgewidth=1, label=name)
                ax.text(stages[-1] + 0.12, y[-1], f"{y[-1]:.1f}", va="center", fontsize=8, color=INK2)
            ax.set_xticks(stages)
            ax.set_xlim(stages[0] - 0.2, stages[-1] + 0.8)
            ax.set_xlabel("stage (10 new classes arrive at each stage)")
            ax.set_ylabel("accuracy on all classes so far, %")
            ax.set_title(f"Fig. Overall score after each stage. {label}, exam half, band = 1 s.d.", fontsize=9.5)
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
            save(fig, f"fig4_over_time_{short}")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    for ds, o in summary["datasets"].items():
        print(ds, json.dumps(o["verdict"], indent=1))


if __name__ == "__main__":
    main()
