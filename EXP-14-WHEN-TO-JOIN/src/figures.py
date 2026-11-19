"""Build the EXP-14 summary, the verdict and the figures from the real result file.

Run:  python src/figures.py [path/to/runs.jsonl]
Writes results/summary.json and figures/*.svg. Every setting is chosen on the rehearsal rows only (memory after
stage 6, stages 7-8 as pretend-new), by the rules fixed in JOURNAL.md section 3:
  with the handicap: over smoothing x handicap, keep rehearsal old >= old at h = 0 (same smoothing) - 5, best H;
  no correction:     h = 0, the smoothing with the best rehearsal H;
  the stream:        the same, with old accuracy and harmonic mean averaged over the 4 sessions (stream score);
  selected rule N*:  the joining rule with the best rehearsal stream score.
Harmonic means are computed per seed (from old and new averaged over draws or runs), then averaged over seeds.
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
WNAME = {"apart": "apart (EXP-13)", "view": "joins the view", "table": "joins table and view"}
WCOL = {"apart": NEUTRAL, "view": BLUE, "table": GREEN}
PNAME = {"never": "never", "once": "at once (3)", "at10": "at 10", "at30": "at 30", "at100": "at 100"}
PCOL = {"never": NEUTRAL, "once": RED, "at10": AMBER, "at30": BLUE, "at100": VIOLET}
STEP0A = {"old": 52.47, "new": 17.28}
STEP0B = {"old": 54.09, "new": 51.67}
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
    gammas, G1 = meta["gammas"], meta["g1"]
    i0, i8 = G1.index(0.0), G1.index(0.08)
    seeds = sorted({r["seed"] for r in rows})

    # cross[(task, way, gamma, label)][seed] = list of per-draw (old[71], new[71], new_only), in draw order
    cross = defaultdict(lambda: defaultdict(list))
    stream = defaultdict(dict)                      # [(task, policy, gamma, session)][seed] = row
    ref = defaultdict(list)                         # [gamma] = [(old, new)] over seeds, part 1
    cost = []
    for r in rows:
        k = r.get("kind")
        if k == "cross":
            key = (r["task"], r["way"], r["gamma"], str(r["shots"]))
            for o, n, no in zip(r["old"], r["new"], r["new_only"]):
                cross[key][r["seed"]].append((np.array(o) / 100, np.array(n) / 100, no / 100))
        elif k == "stream":
            stream[(r["task"], r["policy"], r["gamma"], r["session"])][r["seed"]] = r
        elif k == "ref" and r["task"] == "part1":
            ref[r["gamma"]].append((100 * r["old"], 100 * r["new"]))
        elif r.get("task") == "cost":
            cost.append(r)
    labels = [l for l in ("1", "3", "10", "30", "100", "all") if ("part1", "apart", gammas[0], l) in cross]
    n_all = meta.get("n_all", 500)
    xs = {l: (float(l) if l != "all" else 500.0) for l in labels}

    def arr(task, way, g, lab, draws=None):
        """Per seed: old and new (71,) averaged over the draws (all, or the first `draws`), new-only."""
        o, n, no = [], [], []
        for s in seeds:
            v = cross[(task, way, g, lab)][s][: draws or None]
            o.append(np.mean([x[0] for x in v], 0))
            n.append(np.mean([x[1] for x in v], 0))
            no.append(np.mean([x[2] for x in v]))
        return np.stack(o), np.stack(n), np.array(no)

    def pick(get, handicap):
        best = None
        for g in gammas:
            o, n = get("rehearsal", g)
            om, hh = o.mean(0), hm(o, n).mean(0)
            ok = om >= om[i0] - LIMIT if handicap else np.arange(len(G1)) == i0
            v = np.where(ok, hh, -1)
            j = int(v.argmax())
            if best is None or v[j] > best[0]:
                best = (float(v[j]), g, j, float(om[j]), float(om[i0]))
        return best

    def choose(way, lab, handicap=True, draws=None):
        best = pick(lambda t, g: arr(t, way, g, lab, draws)[:2], handicap)
        _, g, j, r_old, r_c0 = best
        o, n, no = arr("part1", way, g, lab, draws)
        return dict(gamma=g, h=G1[j], rehearsal=dict(H=round(best[0], 2), old=round(r_old, 2), C0_old=round(r_c0, 2)),
                    old=ms(o[:, j]), new=ms(n[:, j]), H=ms(hm(o[:, j], n[:, j])), C0_old=ms(o[:, i0]), new_only=ms(no))

    curve = {w: {l: dict(handicap=choose(w, l), none=choose(w, l, handicap=False)) for l in labels}
             for w in ("apart", "view")}
    t_labels = [l for l in labels if ("part1", "table", gammas[0], l) in cross]
    t_draws = {l: len(cross[("part1", "table", gammas[0], l)][seeds[0]]) for l in t_labels}
    table = {l: dict(table=dict(handicap=choose("table", l), none=choose("table", l, handicap=False)),
                     view=dict(handicap=choose("view", l, draws=t_draws[l]), none=choose("view", l, False, t_draws[l])),
                     draws=t_draws[l]) for l in t_labels}

    # ---- the stream
    policies = list(meta["policies"])
    sessions = list(range(1, meta["sessions"] + 1))

    def sarr(task, pol, g):
        """Per seed, per session: old and new (71,) averaged over runs. Returns (seeds, sessions, 71) arrays."""
        o = np.stack([np.stack([np.mean(stream[(task, pol, g, s)][sd]["old"], 0) / 1e4 * 100 for s in sessions])
                      for sd in seeds])
        n = np.stack([np.stack([np.mean(stream[(task, pol, g, s)][sd]["new"], 0) / 1e4 * 100 for s in sessions])
                      for sd in seeds])
        return o, n

    def schoose(pol, handicap=True):
        def get(t, g):
            o, n = sarr(t, pol, g)
            return o.mean(1), hm(o, n).mean(1)         # old averaged over sessions; stream score per seed
        best = None
        for g in gammas:
            om_s, sc_s = get("rehearsal", g)
            om, sc = om_s.mean(0), sc_s.mean(0)
            ok = om >= om[i0] - LIMIT if handicap else np.arange(len(G1)) == i0
            v = np.where(ok, sc, -1)
            j = int(v.argmax())
            if best is None or v[j] > best[0]:
                best = (float(v[j]), g, j, float(om[j]), float(om[i0]))
        _, g, j, r_old, r_c0 = best
        o, n = sarr("part1", pol, g)
        H = hm(o, n)
        return dict(gamma=g, h=G1[j], rehearsal=dict(score=round(best[0], 2), old=round(r_old, 2), C0_old=round(r_c0, 2)),
                    score=ms(H[:, :, j].mean(1)), old=ms(o[:, :, j].mean(1)), new=ms(n[:, :, j].mean(1)),
                    C0_old=ms(o[:, :, i0].mean(1)),
                    sessions={str(s): dict(H=ms(H[:, i, j]), old=ms(o[:, i, j]), new=ms(n[:, i, j]))
                              for i, s in enumerate(sessions)})

    strm = {p: dict(handicap=schoose(p), none=schoose(p, handicap=False)) for p in policies}

    c = cost
    peak = max(x["peak_mem_mb"] or 0 for x in c)
    cost_out = dict(feature_dim=c[0]["feature_dim"], peak_mem_mb=peak,
                    peak_reserved_mb=max(x["peak_reserved_mb"] or 0 for x in c),
                    cpu_peak_mb=max((x["cpu_peak_mb"] or 0) for x in c),
                    seconds={k: ms([x["seconds"][k] for x in c], 1) for k in c[0]["seconds"]},
                    total_seconds=ms([sum(x["seconds"].values()) for x in c], 1),
                    wh=ms([sum(v for v in x["wh"].values() if v) for x in c], 3) if c[0]["wh"] else None)

    # ---- verdict, by JOURNAL.md section 5
    o, n, _ = arr("part1", "apart", 0.1, "3")
    step0a = dict(old=ms(o[:, i8]), new=ms(n[:, i8]))
    step0a["passed"] = abs(step0a["old"][0] - STEP0A["old"]) <= 0.5 and abs(step0a["new"][0] - STEP0A["new"]) <= 0.5
    rb = np.array(ref[0.1]) if 0.1 in ref else None
    step0b = None if rb is None else dict(old=ms(rb[:, 0]), new=ms(rb[:, 1]))
    if step0b:
        step0b["passed"] = abs(step0b["old"][0] - STEP0B["old"]) <= 0.5 and abs(step0b["new"][0] - STEP0B["new"]) <= 0.5
    gain = {l: dict(handicap=round(curve["view"][l]["handicap"]["H"][0] - curve["apart"][l]["handicap"]["H"][0], 2),
                    none=round(curve["view"][l]["none"]["H"][0] - curve["apart"][l]["none"]["H"][0], 2)) for l in labels}
    crossover = None
    for i, l in enumerate(labels):
        if all(gain[m]["handicap"] >= 1.0 for m in labels[i:]):
            crossover = l
            break
    t_all = table.get("all")
    step2 = None if t_all is None else dict(
        table_H=t_all["table"]["handicap"]["H"][0], view_H=t_all["view"]["handicap"]["H"][0],
        difference=round(t_all["table"]["handicap"]["H"][0] - t_all["view"]["handicap"]["H"][0], 2))
    if step2:
        step2["table_matters"] = step2["difference"] >= 1.0
    sel = max(policies, key=lambda p: strm[p]["handicap"]["rehearsal"]["score"])
    s_sel, s_never, s_once = strm[sel]["handicap"], strm["never"]["handicap"], strm["once"]["handicap"]
    g3 = round(s_sel["score"][0] - s_never["score"][0], 2)
    cond = dict(a=g3 >= 3.0, b=s_sel["old"][0] >= s_sel["C0_old"][0] - 6.0, c=peak <= MEM_LIMIT_MB)
    if all(cond.values()):
        verdict = "works"
    elif cond["c"] and g3 >= 1.0:
        verdict = "helps, but not enough"
    else:
        verdict = "does not work"
    apart_matters = sel != "once" and s_sel["score"][0] - s_once["score"][0] >= 1.0
    va = curve["view"]["all"]
    best_test = max(policies, key=lambda p: strm[p]["handicap"]["score"][0])
    summary = dict(
        labels=labels, curve=curve, table=table, stream=strm, cost=cost_out,
        verdict=dict(
            step0=dict(a=step0a, b=step0b),
            step1=dict(gain=gain, crossover=crossover),
            step2=step2,
            step3=dict(selected=sel, score=s_sel["score"][0], never=s_never["score"][0], once=s_once["score"][0], gain=g3,
                       old=s_sel["old"][0], C0_old=s_sel["C0_old"][0], peak_mem_mb=peak, conditions=cond, verdict=verdict),
            step4=dict(selected_minus_once=round(s_sel["score"][0] - s_once["score"][0], 2), holding_apart_matters=apart_matters),
            step5=dict(H_handicap=va["handicap"]["H"][0], H_none=va["none"]["H"][0],
                       difference=round(va["handicap"]["H"][0] - va["none"]["H"][0], 2),
                       handicap_needed=va["handicap"]["H"][0] - va["none"]["H"][0] >= 1.0),
            step6=dict(best_on_test=best_test, best_score=strm[best_test]["handicap"]["score"][0],
                       trusted=strm[best_test]["handicap"]["score"][0] - s_sel["score"][0] <= 1.0)))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["verdict"], indent=1))
    for w in ("apart", "view"):
        for l in labels:
            a, b = curve[w][l]["handicap"], curve[w][l]["none"]
            print(f"{w:5s} {l:>4s}: handicap H {a['H'][0]:5.1f} (old {a['old'][0]:.1f} new {a['new'][0]:.1f} g {a['gamma']} "
                  f"h {a['h']}) | none H {b['H'][0]:5.1f} (old {b['old'][0]:.1f} new {b['new'][0]:.1f} g {b['gamma']}) "
                  f"| new-only {a['new_only'][0]}")
    for l, v in table.items():
        print(f"table {l:>4s} ({v['draws']} draws): table H {v['table']['handicap']['H'][0]} / none {v['table']['none']['H'][0]}"
              f" | view same draws H {v['view']['handicap']['H'][0]} / none {v['view']['none']['H'][0]}")
    for p in policies:
        a, b = strm[p]["handicap"], strm[p]["none"]
        print(f"stream {p:6s}: score {a['score'][0]:5.1f} (old {a['old'][0]:.1f} new {a['new'][0]:.1f} g {a['gamma']} h {a['h']};"
              f" rehearsal {a['rehearsal']['score']}) | none {b['score'][0]:5.1f} | sessions "
              + " ".join(f"{a['sessions'][str(s)]['H'][0]:.1f}" for s in sessions))

    xv = [xs[l] for l in labels]
    ticks = labels[:-1] + [f"all ({n_all})"]

    # ---- figure 1: the crossover
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for w in ("apart", "view"):
        ax.plot(xv, [curve[w][l]["handicap"]["H"][0] for l in labels], marker="o", color=WCOL[w] if w == "view" else INK2,
                label=f"{WNAME[w]}, with the handicap")
        ax.plot(xv, [curve[w][l]["none"]["H"][0] for l in labels], marker="o", ls="--",
                color=WCOL[w], lw=1.4, label=f"{WNAME[w]}, no correction")
    ax.scatter([xs[l] for l in t_labels], [table[l]["table"]["handicap"]["H"][0] for l in t_labels], color=GREEN,
               zorder=3, s=34, label=f"{WNAME['table']}, with the handicap (1-2 draws)")
    ax.set_xscale("log")
    ax.set_xticks(xv, ticks)
    ax.set_xlabel("pictures per new class")
    ax.set_ylabel("% (test)")
    ax.set_title("Harmonic mean of old and new among all 100: apart against joined")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig1_crossover")

    # ---- figure 2: gain from joining the view
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    w = 0.38
    x = np.arange(len(labels))
    for i, (mode, col, name) in enumerate((("handicap", BLUE, "with the handicap"), ("none", NEUTRAL, "no correction"))):
        vals = [gain[l][mode] for l in labels]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=col, label=name)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7, color=INK2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(x, ticks)
    ax.set_xlabel("pictures per new class")
    ax.set_ylabel("points of harmonic mean")
    ax.set_title("Gain from joining the view (joins the view minus apart, test)")
    ax.grid(False, axis="x")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "fig2_gain")

    # ---- figure 3: the stream
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    for p in policies:
        v = strm[p]["handicap"]["sessions"]
        ax.plot(sessions, [v[str(s)]["H"][0] for s in sessions], marker="o", color=PCOL[p], lw=2.4 if p == sel else 1.5,
                label=PNAME[p] + (" (selected)" if p == sel else ""))
    ax.set_xticks(sessions, [f"session {s}" for s in sessions])
    ax.set_ylabel("harmonic mean % (test)")
    ax.set_title("The stream: harmonic mean after each session")
    ax.legend(fontsize=7, frameon=False, title="joins the view", title_fontsize=7)
    ax = axes[1]
    vals = [strm[p]["handicap"]["score"] for p in policies]
    ax.bar(range(len(policies)), [m for m, _ in vals], yerr=[sd for _, sd in vals], color=[PCOL[p] for p in policies],
           error_kw=dict(ecolor=INK2, lw=0.7, capsize=2))
    for i, (m, _) in enumerate(vals):
        ax.text(i, m + 0.6, f"{m:.1f}", ha="center", fontsize=7, color=INK2)
    lo = min(m for m, _ in vals)
    ax.set_ylim(max(0, lo - 8), max(m for m, _ in vals) + 4)
    ax.set_xticks(range(len(policies)), [PNAME[p] for p in policies], fontsize=8)
    ax.set_title("Stream score (mean over the 4 sessions)")
    ax.grid(False, axis="x")
    save(fig, "fig3_stream")

    # ---- figure 4: old against new as the handicap varies, all pictures and 3 pictures
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for w, lab, ls in (("apart", "3", ":"), ("view", "3", ":"), ("apart", "all", "-"), ("view", "all", "-")):
        ch = curve[w][lab]["handicap"]
        o, n, _ = arr("part1", w, ch["gamma"], lab)
        ax.plot(o.mean(0), n.mean(0), color=WCOL[w] if w == "view" else INK2, ls=ls, lw=1.6,
                label=f"{WNAME[w]}, {lab} pictures" if lab != "all" else f"{WNAME[w]}, all pictures")
        ax.scatter([ch["old"][0]], [ch["new"][0]], s=45, facecolor="none", edgecolor=WCOL[w] if w == "view" else INK2,
                   lw=1.5, zorder=3)
    if "all" in table:
        ch = table["all"]["table"]["handicap"]
        o, n, _ = arr("part1", "table", ch["gamma"], "all")
        ax.plot(o.mean(0), n.mean(0), color=GREEN, lw=1.6, label=f"{WNAME['table']}, all pictures")
        ax.scatter([ch["old"][0]], [ch["new"][0]], s=45, facecolor="none", edgecolor=GREEN, lw=1.5, zorder=3)
    ax.set_xlabel("old classes % (among all 100)")
    ax.set_ylabel("new classes % (among all 100)")
    ax.set_title("Old against new as the handicap varies (test)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig4_tradeoff")


if __name__ == "__main__":
    main()
