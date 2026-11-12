"""Draw the EXP-2 figures from the real result file and write the summary tables.

Run:  python src/figures.py --tag full
Writes figures/*.svg and results/summary_<tag>.json. Nothing here is illustrative: every mark is
a mean over seeds read from results/<tag>.jsonl.
"""
import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import FIGURES, RESULTS

SRC = ["A", "B", "C", "D", "E", "F"]
SRC_NAME = {"A": "A raw pixels", "B": "B random expansion", "C": "C random conv net",
            "D": "D trained, no labels", "E": "E trained, labels", "F": "F ImageNet ResNet-18"}
SRC_COL = dict(zip(SRC, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
LRN = ["ncm", "slda", "fly", "clp"]
LRN_COL = dict(zip(LRN, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
DS_NAME = {"cifar100": "CIFAR-100 (80 classes)", "omniglot": "Omniglot (659 classes)",
           "mnist_fashion": "MNIST then Fashion (20 classes)"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "lines.markersize": 4.5,
})


def load(tag):
    rows = [json.loads(l) for l in (RESULTS / f"{tag}.jsonl").read_text().splitlines() if l.strip()]
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["source"], r["learner"], r["shots"])].append(r)
    return g


def mean(g, key, field):
    return float(np.mean([r[field] for r in g[key]]))


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def spread_labels(ys, gap):
    """Move end-of-line labels apart so they do not overlap; keeps their order."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    g = load(a.tag)
    datasets = [d for d in DS_NAME if any(k[0] == d for k in g)]
    shots = sorted({k[3] for k in g})
    n_seeds = max(len(v) for v in g.values())
    nd = len(datasets)

    # best learner per (dataset, source), chosen at 3 shots as the decision rule says
    best, summary = {}, {"tag": a.tag, "seeds": n_seeds, "datasets": {}}
    for d in datasets:
        tab = {s: {l: {str(k): round(100 * mean(g, (d, s, l, k), "final_acc"), 2) for k in shots}
                   for l in LRN if (d, s, l, 3) in g} for s in SRC if any((d, s, l, 3) in g for l in LRN)}
        for s in tab:
            best[(d, s)] = max(tab[s], key=lambda l: tab[s][l]["3"])
        f3 = tab["F"][best[(d, "F")]]["3"] if "F" in tab else None
        rule = {s: dict(best_learner=best[(d, s)], acc3=tab[s][best[(d, s)]]["3"],
                        sd3=round(100 * float(np.std([r["final_acc"] for r in g[(d, s, best[(d, s)], 3)]])), 2),
                        ratio_to_F=round(tab[s][best[(d, s)]]["3"] / f3, 3) if f3 else None) for s in tab}
        forget = {s: {l: dict(alone=round(100 * mean(g, (d, s, l, 3), "first_half_acc_when_alone"), 2),
                              at_end=round(100 * mean(g, (d, s, l, 3), "first_half_acc_at_end"), 2))
                      for l in tab[s]} for s in tab}
        cost = {l: round(float(np.mean([mean(g, (d, s, l, 3), "learn_ms_per_class") for s in tab if l in tab[s]])), 3)
                for l in LRN}
        protos = {s: round(mean(g, (d, s, "clp", max(shots)), "prototypes"), 1) for s in tab if "clp" in tab[s]}
        summary["datasets"][d] = dict(final_acc=tab, rule_at_3_shots=rule, first_half_at_3_shots=forget,
                                      learn_ms_per_class_at_3_shots=cost, clp_prototypes_at_max_shots=protos,
                                      classes=g[(d, "A", "ncm", 3)][0]["classes"])
    (RESULTS / f"summary_{a.tag}.json").write_text(json.dumps(summary, indent=1))

    # ---- Fig 1: accuracy vs examples per class, each source with its best learner
    fig, axes = plt.subplots(1, nd, figsize=(4.3 * nd, 3.6), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        ends = []
        for s in SRC:
            if (d, s) not in best:
                continue
            l = best[(d, s)]
            m = np.array([100 * mean(g, (d, s, l, k), "final_acc") for k in shots])
            sd = np.array([100 * np.std([r["final_acc"] for r in g[(d, s, l, k)]]) for k in shots])
            ax.fill_between(range(len(shots)), m - sd, m + sd, color=SRC_COL[s], alpha=0.12, lw=0)
            ax.plot(range(len(shots)), m, color=SRC_COL[s], marker="o", markeredgecolor=SURF, markeredgewidth=1)
            ends.append((s, l, m[-1]))
        lo, hi = ax.get_ylim()
        ys = spread_labels([e[2] for e in ends], 0.055 * (hi - lo))
        for (s, l, _), y in zip(ends, ys):
            ax.text(len(shots) - 0.85, y, f"{s} · {l}", va="center", fontsize=8, color=INK2)
            ax.plot([len(shots) - 0.97], [y], marker="s", ms=4, color=SRC_COL[s], clip_on=False)
        ax.axvline(shots.index(3), color=AXIS, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.set_xticks(range(len(shots)), shots)
        ax.set_xlim(-0.2, len(shots) + 0.25)
        ax.set_title(DS_NAME[d])
        ax.set_xlabel("examples per class")
    axes[0][0].set_ylabel("accuracy on all classes, %")
    fig.legend(handles=[plt.Line2D([], [], color=SRC_COL[s], marker="o", label=SRC_NAME[s]) for s in SRC],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.suptitle(f"Final accuracy after learning every class, by feature source (best learner each, mean of "
                 f"{n_seeds} seeds, band = 1 s.d.)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_accuracy_vs_shots")

    # ---- Fig 2: the decision rule. Share of F's accuracy at 3 shots
    fig, axes = plt.subplots(1, nd, figsize=(4.3 * nd, 3.2), squeeze=False, sharey=True)
    for ax, d in zip(axes[0], datasets):
        rule = summary["datasets"][d]["rule_at_3_shots"]
        ss = [s for s in SRC if s in rule]
        vals = [100 * rule[s]["ratio_to_F"] for s in ss]
        ax.bar(ss, vals, width=0.62, color=[SRC_COL[s] for s in ss], edgecolor=SURF, linewidth=2)
        for i, (s, v) in enumerate(zip(ss, vals)):
            ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
            ax.text(i, 3, f"{rule[s]['acc3']:.1f}\n{rule[s]['best_learner']}", ha="center", fontsize=7,
                    color="#0b0b0b" if s == "D" else "#ffffff")
        ax.axhline(90, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(len(ss) - 0.4, 90, "90%\n= close", ha="left", va="center", fontsize=8, color=INK2)
        ax.set_xlim(-0.6, len(ss) + 0.5)
        ax.set_title(DS_NAME[d])
        ax.set_ylim(0, max(125, max(vals) + 12))
    axes[0][0].set_ylabel("share of source F's accuracy, %")
    fig.suptitle("Decision rule at 3 examples per class: how close each cheap source gets to the pretrained "
                 "ResNet-18 (in bars: accuracy % and learner)", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_decision_rule")

    # ---- Fig 3: every source x learner at 3 shots
    fig, axes = plt.subplots(1, nd, figsize=(3.6 * nd, 3.4), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        tab = summary["datasets"][d]["final_acc"]
        M = np.array([[tab[s][l]["3"] if s in tab and l in tab[s] else np.nan for l in LRN] for s in SRC])
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=np.nanmax(M), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                            color="#ffffff" if M[i, j] > 0.5 * np.nanmax(M) else INK)
        ax.set_xticks(range(len(LRN)), LRN)
        ax.set_yticks(range(len(SRC)), SRC)
        ax.set_xticks(np.arange(-.5, len(LRN)), minor=True)
        ax.set_yticks(np.arange(-.5, len(SRC)), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color=SURF, linewidth=2)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(DS_NAME[d])
    axes[0][0].set_ylabel("feature source")
    fig.suptitle("Accuracy % at 3 examples per class, every feature source and learner",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_source_by_learner")

    # ---- Fig 4: accuracy as classes are added (3 shots, best learner per source)
    fig, axes = plt.subplots(1, nd, figsize=(4.3 * nd, 3.4), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for s in SRC:
            if (d, s) not in best:
                continue
            rs = g[(d, s, best[(d, s)], 3)]
            xs = sorted(int(n) for n in rs[0]["acc_seen"])
            ax.plot(xs, [100 * np.mean([r["acc_seen"][str(n)] for r in rs]) for n in xs],
                    color=SRC_COL[s], marker="o", markeredgecolor=SURF, markeredgewidth=1)
        ax.set_title(DS_NAME[d])
        ax.set_xlabel("classes learned so far")
        ax.set_ylim(0, 100)
    axes[0][0].set_ylabel("accuracy on classes seen so far, %")
    fig.legend(handles=[plt.Line2D([], [], color=SRC_COL[s], marker="o", label=SRC_NAME[s]) for s in SRC],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.suptitle("Accuracy while classes arrive one at a time (3 examples each, best learner per source)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_accuracy_as_classes_arrive")

    # ---- Fig 5: first half of the classes, alone vs after everything is learned
    fig, axes = plt.subplots(1, nd, figsize=(4.3 * nd, 3.6), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        fz = summary["datasets"][d]["first_half_at_3_shots"]
        labels, y = [], 0
        for s in SRC:
            if s not in fz:
                continue
            for l in LRN:
                if l not in fz[s]:
                    continue
                v = fz[s][l]
                ax.plot([v["at_end"], v["alone"]], [y, y], color=LRN_COL[l], lw=2, solid_capstyle="round")
                ax.plot(v["alone"], y, marker="o", ms=5, mfc=SURF, mec=LRN_COL[l], mew=1.5)
                ax.plot(v["at_end"], y, marker="o", ms=5, color=LRN_COL[l], mec=SURF, mew=1)
                y += 1
            labels.append((s, y - len(fz[s]) / 2 - 0.5))
            y += 0.8
        ax.set_yticks([p for _, p in labels], [s for s, _ in labels])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlim(0, 100)
        ax.set_xlabel("accuracy on the first half of the classes, %")
        ax.set_title(DS_NAME[d])
    axes[0][0].set_ylabel("feature source")
    hs = [plt.Line2D([], [], color=LRN_COL[l], lw=2, label=l) for l in LRN]
    hs += [plt.Line2D([], [], ls="", marker="o", mfc=SURF, mec=INK2, mew=1.5, label="when only they were known"),
           plt.Line2D([], [], ls="", marker="o", color=INK2, label="after all classes were learned")]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.suptitle("Old classes after new ones are added (3 examples each): the drop from hollow to filled dot",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_first_half_retention")

    # ---- Fig 6: time to learn one class
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    w = 0.8 / nd
    for i, d in enumerate(datasets):
        c = summary["datasets"][d]["learn_ms_per_class_at_3_shots"]
        xs = np.arange(len(LRN)) + (i - (nd - 1) / 2) * w
        ax.bar(xs, [c[l] for l in LRN], width=w, color=["#2a78d6", "#eb6834", "#1baf7a"][i],
               edgecolor=SURF, linewidth=2, label=DS_NAME[d])
        for x, l in zip(xs, LRN):
            ax.text(x, c[l] * 1.12, f"{c[l]:.2f}" if c[l] < 10 else f"{c[l]:.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(LRN)), LRN)
    ax.set_ylabel("milliseconds per class (log scale)")
    ax.set_title("Time to learn one new class from 3 examples (GPU, averaged over sources)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 4)
    fig.tight_layout()
    save(fig, "fig6_learning_time")


if __name__ == "__main__":
    main()
