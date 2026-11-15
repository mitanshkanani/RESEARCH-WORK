"""Draw the EXP-3 figures from the real result file and write the summary tables and the verdict.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes figures/*.svg and results/summary.json. Nothing here is illustrative: every mark is a mean
over seeds read from results/runs.jsonl.
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

WAYS = ["frozen", "ordinary", "grow", "retrain"]
NAME = {"frozen": "1 Frozen eyes", "ordinary": "2 Ordinary training", "grow": "3 Grow and lock",
        "retrain": "4 Retrain everything"}
COL = dict(zip(WAYS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARK = dict(zip(WAYS, ["o", "s", "D", "^"]))
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

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


def spread(ys, gap):
    order, out = np.argsort(ys), np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def legend(fig, y=-0.06):
    fig.legend(handles=[plt.Line2D([], [], color=COL[w], marker=MARK[w], label=NAME[w]) for w in WAYS],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, y), fontsize=8.5)


def curves(ax, stat, ds, field, stages, title, ylabel=None, chance=None):
    """One line per way: mean over seeds, band = 1 s.d., value labelled at the last stage."""
    ends = []
    for w in WAYS:
        m = np.array([stat[(ds, w, s)][field][0] for s in stages])
        sd = np.array([stat[(ds, w, s)][field][1] for s in stages])
        ax.fill_between(stages, m - sd, m + sd, color=COL[w], alpha=0.12, lw=0)
        ax.plot(stages, m, color=COL[w], marker=MARK[w], markeredgecolor=SURF, markeredgewidth=1)
        ends.append(m[-1])
    if chance is not None:
        ax.plot(stages, chance, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(stages[0], chance[0] + 1.5, "guessing", fontsize=7.5, color=MUTED, va="bottom")
    lo, hi = ax.get_ylim()
    for w, e, y in zip(WAYS, ends, spread(ends, 0.06 * (hi - lo))):
        ax.text(stages[-1] + 0.15, y, f"{e:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_xticks(stages)
    ax.set_xlim(stages[0] - 0.2, stages[-1] + 0.9)
    ax.set_title(title)
    ax.set_xlabel("stage (10 new classes arrive at each stage)")
    if ylabel:
        ax.set_ylabel(ylabel)


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "runs.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["way"], r["stage"])].append(r)

    def ms(vals):
        vals = [v for v in vals if v is not None]
        return (round(float(np.mean(vals)), 3), round(float(np.std(vals)), 3)) if vals else (None, None)

    stat = {}
    for key, rs in g.items():
        stat[key] = dict(
            first=ms([100 * r["acc_groups"][0] for r in rs]), all=ms([100 * r["acc_all"] for r in rs]),
            open_first=ms([100 * r["open_acc_groups"][0] for r in rs]), open_all=ms([100 * r["open_acc_all"] for r in rs]),
            newest=ms([100 * r["acc_groups"][-1] for r in rs]),
            train_seconds=ms([r["train_seconds"] for r in rs]), energy_wh=ms([r.get("energy_wh") for r in rs]),
            peak_mem_mb=ms([r.get("peak_mem_mb") for r in rs]), units=rs[0]["units_per_layer"],
            params_total=rs[0]["params_total"], params_trained=rs[0]["params_trained"],
            lock_drift=max([r.get("lock_drift", 0.0) for r in rs]), seeds=len(rs))
        if "few_shot" in rs[0]:
            for k in rs[0]["few_shot"]:
                stat[key][f"few{k}"] = ms([100 * r["few_shot"][k] for r in rs])

    datasets = sorted({k[0] for k in g})
    summary = {"source": str(src.name), "datasets": {}}
    for ds in datasets:
        stages = sorted({k[2] for k in g if k[0] == ds})
        last = stages[-1]
        tab = {w: {str(s): stat[(ds, w, s)] for s in stages} for w in WAYS}
        cum = {w: dict(train_minutes=round(sum(stat[(ds, w, s)]["train_seconds"][0] for s in stages) / 60, 2),
                       energy_wh=(round(sum(stat[(ds, w, s)]["energy_wh"][0] for s in stages), 2)
                                  if all(stat[(ds, w, s)]["energy_wh"][0] is not None for s in stages) else None),
                       peak_mem_mb=max(stat[(ds, w, s)]["peak_mem_mb"][0] or 0 for s in stages),
                       final_units=stat[(ds, w, last)]["units"], final_params=stat[(ds, w, last)]["params_total"])
               for w in WAYS}
        summary["datasets"][ds] = dict(stages=stages, seeds=stat[(ds, "grow", last)]["seeds"], by_way=tab, cost=cum)

        # the rule from JOURNAL.md section 6
        f = lambda w, field, s=last: stat[(ds, w, s)][field][0]
        verdict = {}
        o1, oN = f("ordinary", "first", 1), f("ordinary", "first")
        verdict["step0_ordinary_first_group"] = dict(after_stage_1=o1, at_end=oN, lost_share=round(1 - oN / o1, 3),
                                                     valid=bool(oN <= 0.5 * o1))
        shares = {}
        for label, field in (("M2_all_classes", "all"), ("M3_few_shot_3", "few3")):
            if field not in stat[(ds, "grow", last)]:
                continue
            fr, re, gr, od = f("frozen", field), f("retrain", field), f("grow", field), f("ordinary", field)
            gap = re - fr
            shares[label] = dict(frozen=fr, retrain=re, grow=gr, ordinary=od, gap=round(gap, 2),
                                 grow_share=round((gr - fr) / gap, 3) if gap >= 3 else None,
                                 ordinary_share=round((od - fr) / gap, 3) if gap >= 3 else None)
        verdict["step1_gap_closed"] = shares
        verdict["step2_keeps_old"] = dict(grow_first_at_end=f("grow", "first"), frozen_first_at_end=f("frozen", "first"),
                                          passes=bool(f("grow", "first") >= f("frozen", "first") - 5))
        vals = [v["grow_share"] for v in shares.values() if v["grow_share"] is not None]
        if not vals or len(vals) < len(shares):
            verdict["verdict"] = "no verdict on at least one measurement (floor and ceiling under 3 points apart)"
        if vals:
            if not verdict["step2_keeps_old"]["passes"] or min(vals) < 0.25:
                word = "does not work"
            elif min(vals) >= 0.5:
                word = "works"
            else:
                word = "partly works"
            verdict["verdict_on_computable"] = word
        summary["datasets"][ds]["verdict"] = verdict
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))

    # ------------------------------------------------------------------ photos
    ds = "cifar100"
    if ds in datasets:
        stages = summary["datasets"][ds]["stages"]
        n = summary["datasets"][ds]["seeds"]
        note = f"mean of {n} seeds, band = 1 s.d."

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
        curves(axes[0], stat, ds, "first", stages, "As tested: no old pictures kept", "accuracy on the FIRST 10 classes, %")
        curves(axes[1], stat, ds, "open_first", stages, "Open-book check: memory cards rewritten from old pictures")
        fig.suptitle(f"Fig 1. Does it still know the first group? Photos (CIFAR-100), {note}", x=0.01, ha="left",
                     fontsize=10.5, fontweight="bold")
        legend(fig)
        fig.tight_layout()
        save(fig, "fig1_first_group_over_time")

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
        ch = [100 / (10 * s) for s in stages]
        curves(axes[0], stat, ds, "all", stages, "As tested: no old pictures kept", "accuracy on ALL classes so far, %", ch)
        curves(axes[1], stat, ds, "open_all", stages, "Open-book check: memory cards rewritten from old pictures", None, ch)
        fig.suptitle(f"Fig 2. How good is it overall, with no hint? Photos (CIFAR-100), {note}", x=0.01, ha="left",
                     fontsize=10.5, fontweight="bold")
        legend(fig)
        fig.tight_layout()
        save(fig, "fig2_all_classes_over_time")

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for ax, k in zip(axes, ("1", "3", "10")):
            curves(ax, stat, ds, f"few{k}", stages, f"{k} picture{'s' if k != '1' else ''} per surprise class",
                   "accuracy on 20 surprise classes, %" if k == "1" else None, [5] * len(stages))
        fig.suptitle(f"Fig 3. Did the eyes get better at learning brand-new things? Photos (CIFAR-100), {note}",
                     x=0.01, ha="left", fontsize=10.5, fontweight="bold")
        legend(fig)
        fig.tight_layout()
        save(fig, "fig3_few_shot_over_time")

        sh = summary["datasets"][ds]["verdict"]["step1_gap_closed"]
        labels = {"M2_all_classes": "M2: all classes so far", "M3_few_shot_3": "M3: surprise classes, 3 pictures"}
        fig, axes = plt.subplots(1, len(sh), figsize=(4.6 * len(sh), 3.4), squeeze=False)
        for ax, (key, v) in zip(axes[0], sh.items()):
            vals = [v[w] for w in WAYS]
            ax.bar(range(4), vals, width=0.62, color=[COL[w] for w in WAYS], edgecolor=SURF, linewidth=2)
            for i, val in enumerate(vals):
                ax.text(i, val + 0.5, f"{val:.1f}", ha="center", fontsize=8.5, color=INK)
            ax.set_xticks(range(4), ["1 Frozen\n(floor)", "2 Ordinary", "3 Grow\nand lock", "4 Retrain\n(ceiling)"])
            share = v["grow_share"]
            ax.set_title(labels[key] + (f"\ngrow and lock closes {100 * share:.0f}% of the gap" if share is not None
                                        else "\ngap under 3 points: no share computed"))
            ax.set_ylim(0, max(vals) * 1.18)
        axes[0][0].set_ylabel("accuracy after the last stage, %")
        fig.suptitle("Fig 4. The verdict numbers: floor, ceiling, and where the tested idea lands",
                     x=0.01, ha="left", fontsize=10.5, fontweight="bold")
        fig.tight_layout()
        save(fig, "fig4_verdict")

    # ------------------------------------------------------------------ digits then clothes
    ds = "digits_clothes"
    if ds in datasets:
        n = summary["datasets"][ds]["seeds"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
        panels = [("first", "As tested: no old pictures kept"), ("open_first", "Open-book check: cards rewritten from old pictures")]
        for ax, (field, title) in zip(axes, panels):
            x = np.arange(4)
            before = [stat[(ds, w, 1)]["first"][0] for w in WAYS]
            after = [stat[(ds, w, 2)][field][0] for w in WAYS]
            ax.bar(x - 0.2, before, width=0.36, color=[COL[w] for w in WAYS], alpha=0.35, edgecolor=SURF, linewidth=2)
            ax.bar(x + 0.2, after, width=0.36, color=[COL[w] for w in WAYS], edgecolor=SURF, linewidth=2)
            for i in range(4):
                ax.text(i - 0.2, before[i] + 1.5, f"{before[i]:.1f}", ha="center", fontsize=8, color=INK2)
                ax.text(i + 0.2, after[i] + 1.5, f"{after[i]:.1f}", ha="center", fontsize=8, color=INK)
            ax.axhline(5, color=MUTED, lw=1, ls=(0, (3, 3)))
            ax.set_xticks(x, ["1 Frozen", "2 Ordinary", "3 Grow\nand lock", "4 Retrain"])
            ax.set_ylim(0, 112)
            ax.set_title(title)
        axes[0].set_ylabel("accuracy on the 10 digits, %")
        fig.suptitle(f"Fig 5. Digits, then clothes: accuracy on digits before (pale) and after (solid) learning clothes, "
                     f"mean of {n} seeds", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
        fig.tight_layout()
        save(fig, "fig5_digits_then_clothes")

    # ------------------------------------------------------------------ cost
    ds = "cifar100"
    if ds in datasets:
        stages = summary["datasets"][ds]["stages"]
        has_energy = all(summary["datasets"][ds]["cost"][w]["energy_wh"] is not None for w in WAYS)
        panels = [("train_seconds", "Training time per stage, seconds"), ("params", "Size of the network, thousands of numbers")]
        if has_energy:
            panels.insert(1, ("energy_wh", "GPU energy per stage, watt-hours"))
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
        for ax, (field, title) in zip(axes, panels):
            for w in WAYS:
                if field == "params":
                    y = [stat[(ds, w, s)]["params_total"] / 1000 for s in stages]
                else:
                    y = [stat[(ds, w, s)][field][0] for s in stages]
                ax.plot(stages, y, color=COL[w], marker=MARK[w], markeredgecolor=SURF, markeredgewidth=1)
            ax.set_xticks(stages)
            ax.set_title(title)
            ax.set_xlabel("stage")
            ax.set_ylim(bottom=0)
        fig.suptitle("Fig 6. What it cost at each stage. Photos (CIFAR-100), mean over seeds",
                     x=0.01, ha="left", fontsize=10.5, fontweight="bold")
        legend(fig, -0.08)
        fig.tight_layout()
        save(fig, "fig6_cost")

    print(json.dumps({d: summary["datasets"][d]["verdict"] for d in datasets}, indent=1))


if __name__ == "__main__":
    main()
