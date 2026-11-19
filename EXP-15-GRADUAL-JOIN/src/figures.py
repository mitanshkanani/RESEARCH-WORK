"""EXP-15: turn the result files into results/summary.json and the figures, with the choosing rules of JOURNAL.md
section 3 and the verdict steps of section 5.

  part 1:  per way and number of pictures, among smoothings and handicaps h1, keep those where the old classes stay
           within 5 points of no correction (h1 = 0, same smoothing, same way) in the rehearsal; take the best rehearsal
           harmonic mean. G1* = the gradual way with the best rehearsal harmonic mean averaged over the 6 counts.
  stream:  the same over smoothings, h1 and h2, with old accuracy and harmonic mean averaged over the 8 sessions
           (stream score); "one handicap" = h2 fixed at 0. G* = the gradual rule with the best rehearsal stream score.

Run:  python src/figures.py [results folder]
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
FIGURES = ROOT / "figures"
BLUE, RED, GREEN, AMBER, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a63d2"
INK, INK2, MUTED, GRID, AXIS, SURF, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#b9b8b0"
STEP0A = {"old": 52.47, "new": 17.28}
STEP0B = {"old": 55.35, "new": 9.91}
STEP0C = [34.82, 38.65, 39.20]
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


def gradual(name):
    return name.startswith(("counted", "measured", "card"))


def pretty(name):
    if name.startswith("counted"):
        return f"counted, k = {name[7:]}"
    if name.startswith("measured"):
        return f"measured, b = {name[8:]}"
    if name.startswith("card"):
        return f"card only, b = {name[4:]}"
    return {"once": "threshold 3 (at once)", "at10": "threshold 10", "at30": "threshold 30", "at100": "threshold 100",
            "never": "never", "apart": "apart", "full": "full"}.get(name, name)


def main():
    res = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    meta = json.loads((res / "meta.json").read_text())
    rows = [json.loads(l) for l in (res / "runs.jsonl").read_text().splitlines() if l.strip()]
    gammas, G1, G2 = meta["gammas"], meta["g1"], meta["g2"]
    ways, rules = meta["cross_ways"], meta["stream_rules"]
    i0, i8, i12 = G1.index(0.0), G1.index(0.08), G1.index(0.12)
    j0 = G2.index(0.0)
    seeds = sorted({r["seed"] for r in rows if r.get("kind") == "files"})
    files = {(r["task"], r["seed"]): r for r in rows if r.get("kind") == "files"}
    labels = files[("part1", seeds[0])]["labels"]
    xs = {l: (float(l) if l != "all" else 500.0) for l in labels}
    sessions = list(range(1, meta["sessions"] + 1))

    cross = {(t, s): np.load(res / f"cross_{t}_s{s}.npz") for t in ("rehearsal", "part1") for s in seeds}
    strm = {(t, s): np.load(res / f"stream_{t}_s{s}.npz") for t in ("rehearsal", "part1") for s in seeds}

    # ---------------- part 1
    def arr(task, way, g, lab):
        o = np.stack([cross[task, s][f"{way}|{g}|{lab}|old"].mean(0) / 100 for s in seeds])
        n = np.stack([cross[task, s][f"{way}|{g}|{lab}|new"].mean(0) / 100 for s in seeds])
        no = np.array([cross[task, s][f"{way}|{g}|{lab}|no"].mean() / 100 for s in seeds])
        return o, n, no

    def choose(way, lab, handicap=True):
        best = None
        for g in gammas:
            o, n, _ = arr("rehearsal", way, g, lab)
            om, hh = o.mean(0), hm(o, n).mean(0)
            ok = om >= om[i0] - LIMIT if handicap else np.arange(len(G1)) == i0
            v = np.where(ok, hh, -1)
            j = int(v.argmax())
            if best is None or v[j] > best[0]:
                best = (float(v[j]), g, j, float(om[j]), float(om[i0]))
        _, g, j, r_old, r_c0 = best
        o, n, no = arr("part1", way, g, lab)
        return dict(gamma=g, h=G1[j], rehearsal=dict(H=round(best[0], 2), old=round(r_old, 2), C0_old=round(r_c0, 2)),
                    old=ms(o[:, j]), new=ms(n[:, j]), H=ms(hm(o[:, j], n[:, j])), C0_old=ms(o[:, i0]), new_only=ms(no),
                    H_seeds=hm(o[:, j], n[:, j]).tolist())

    curve = {w: {l: dict(handicap=choose(w, l), none=choose(w, l, handicap=False)) for l in labels} for w in ways}
    g_ways = [w for w in ways if gradual(w)]
    g1_score = {w: round(float(np.mean([curve[w][l]["handicap"]["rehearsal"]["H"] for l in labels])), 3) for w in g_ways}
    g1 = max(g_ways, key=lambda w: g1_score[w])

    # ---------------- the stream
    def sarr(task, rule, gi):
        a = np.stack([strm[task, s][rule][gi] / 100 for s in seeds])        # (seeds, sessions, 2, H2, H1)
        return a[:, :, 0], a[:, :, 1]

    def schoose(rule, two=True, handicap=True):
        best = None
        for gi, g in enumerate(gammas):
            o, n = sarr("rehearsal", rule, gi)
            om, sc = o.mean(1).mean(0), hm(o, n).mean(1).mean(0)                # (H2, H1)
            ok = om >= om[j0, i0] - LIMIT
            if not two:
                ok &= (np.arange(len(G2)) == j0)[:, None]
            if not handicap:
                ok = np.zeros_like(ok)
                ok[j0, i0] = True
            v = np.where(ok, sc, -1)
            j2, j1 = np.unravel_index(int(v.argmax()), v.shape)
            if best is None or v[j2, j1] > best[0]:
                best = (float(v[j2, j1]), g, gi, int(j2), int(j1), float(om[j2, j1]), float(om[j0, i0]))
        _, g, gi, j2, j1, r_old, r_c0 = best
        o, n = sarr("part1", rule, gi)
        H = hm(o, n)
        return dict(gamma=g, gi=gi, h1=G1[j1], h2=G2[j2], j1=j1, j2=j2,
                    rehearsal=dict(score=round(best[0], 2), old=round(r_old, 2), C0_old=round(r_c0, 2)),
                    score=ms(H[:, :, j2, j1].mean(1)), old=ms(o[:, :, j2, j1].mean(1)), new=ms(n[:, :, j2, j1].mean(1)),
                    C0_old=ms(o[:, :, j0, i0].mean(1)), score_first4=ms(H[:, :4, j2, j1].mean(1)),
                    seeds=H[:, :, j2, j1].mean(1).tolist(),
                    sessions={str(s): dict(H=ms(H[:, i, j2, j1]), old=ms(o[:, i, j2, j1]), new=ms(n[:, i, j2, j1]))
                              for i, s in enumerate(sessions)})

    stream = {r: dict(two=schoose(r), one=schoose(r, two=False), none=schoose(r, handicap=False)) for r in rules}
    g_rules = [r for r in rules if gradual(r)]
    gstar = max(g_rules, key=lambda r: stream[r]["two"]["rehearsal"]["score"])
    fam_best = {f: max([r for r in g_rules if r.startswith(f)], key=lambda r: stream[r]["two"]["rehearsal"]["score"])
                for f in ("counted", "measured", "card")}

    def at_settings(rule, gi, j2, j1):
        o, n = sarr("part1", rule, gi)
        return hm(o, n)[:, :, j2, j1].mean(1)

    # ---------------- noise, weights, cost
    noise = defaultdict(list)
    for r in rows:
        if r.get("kind") == "noise":
            noise[(r["task"], r["gamma"])].append((r["ratio"], r["s2"], r["within"]))
    noise_out = {f"{t}|{g}": dict(ratio=ms([x[0] for x in v], 3), s2=ms([x[1] for x in v], 1))
                 for (t, g), v in sorted(noise.items())}
    wts = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("kind") == "weights" and r["task"] == "part1":
            for w, v in r["mean_weight"].items():
                wts[(r["gamma"], r["shots"])][w].append(v)
    weights_out = {f"{g}|{l}": {w: round(float(np.mean(v)), 3) for w, v in d.items()} for (g, l), d in wts.items()}
    cost = [r for r in rows if r.get("task") == "cost"]
    peak = max(x["peak_mem_mb"] or 0 for x in cost)
    cost_out = dict(feature_dim=cost[0]["feature_dim"], peak_mem_mb=peak,
                    peak_reserved_mb=max(x["peak_reserved_mb"] or 0 for x in cost),
                    cpu_peak_mb=max((x["cpu_peak_mb"] or 0) for x in cost),
                    seconds={k: ms([x["seconds"][k] for x in cost], 1) for k in cost[0]["seconds"]},
                    total_seconds=ms([sum(x["seconds"].values()) for x in cost], 1),
                    wh=ms([sum(v for v in x["wh"].values() if v) for x in cost], 3) if cost[0]["wh"] else None,
                    minutes=meta.get("total_minutes"))

    # ---------------- verdict, by JOURNAL.md section 5
    o, n, _ = arr("part1", "apart", 0.1, "3")
    s0a = dict(old=ms(o[:, i8]), new=ms(n[:, i8]))
    s0a["passed"] = abs(s0a["old"][0] - STEP0A["old"]) <= 0.5 and abs(s0a["new"][0] - STEP0A["new"]) <= 0.5
    o, n, _ = arr("part1", "full", 0.1, "3")
    s0b = dict(old=ms(o[:, i8]), new=ms(n[:, i8]))
    s0b["passed"] = abs(s0b["old"][0] - STEP0B["old"]) <= 0.5 and abs(s0b["new"][0] - STEP0B["new"]) <= 0.5
    o, n = sarr("part1", "at30", gammas.index(0.03))
    hc = [round(float(hm(o[:, i, j0, i12], n[:, i, j0, i12]).mean()), 2) for i in range(3)]
    s0c = dict(H=hc, passed=all(abs(a - b) <= 0.5 for a, b in zip(hc, STEP0C)))

    step1 = {}
    for l in labels:
        a, f, g = (curve[w][l]["handicap"]["H"][0] for w in ("apart", "full", g1))
        step1[l] = dict(G1=g, apart=a, full=f, minus_better=round(g - max(a, f), 2), minus_apart=round(g - a, 2),
                        minus_full=round(g - f, 2))
    s1 = dict(G1=g1, rehearsal_mean=g1_score, per_count=step1,
              removes_harm=all(v["minus_better"] >= -0.5 for v in step1.values()))

    cb, mb = stream[fam_best["counted"]]["two"], stream[fam_best["measured"]]["two"]
    s2 = dict(noise=noise_out, best_measured=fam_best["measured"], best_counted=fam_best["counted"],
              best_card=fam_best["card"], counted_score=cb["score"][0], measured_score=mb["score"][0],
              counted_minus_measured=round(cb["score"][0] - mb["score"][0], 2),
              same=abs(cb["score"][0] - mb["score"][0]) < 0.5)

    G, T1, NV = stream[gstar]["two"], stream["at30"]["one"], stream["never"]["one"]
    gain_t = round(G["score"][0] - T1["score"][0], 2)
    gain_n = round(G["score"][0] - NV["score"][0], 2)
    cond = dict(a=gain_t >= 1.0, b=gain_n >= 3.0, c=G["old"][0] >= G["C0_old"][0] - 6.0, d=peak <= MEM_LIMIT_MB)
    if all(cond.values()):
        verdict = "works"
    elif cond["c"] and cond["d"] and gain_t >= 0.5:
        verdict = "helps, but not enough"
    else:
        verdict = "does not work"
    s3 = dict(G=gstar, score=G["score"][0], at30_one=T1["score"][0], never=NV["score"][0], minus_at30_one=gain_t,
              minus_never=gain_n, old=G["old"][0], C0_old=G["C0_old"][0], peak_mem_mb=peak, conditions=cond,
              verdict=verdict, paired_minus_at30_one=ms(np.array(G["seeds"]) - np.array(T1["seeds"])),
              paired_minus_never=ms(np.array(G["seeds"]) - np.array(NV["seeds"])))
    s4 = {r: dict(two=stream[r]["two"]["score"][0], one=stream[r]["one"]["score"][0],
                  gain=round(stream[r]["two"]["score"][0] - stream[r]["one"]["score"][0], 2),
                  matters=stream[r]["two"]["score"][0] - stream[r]["one"]["score"][0] >= 1.0,
                  h2=stream[r]["two"]["h2"]) for r in (gstar, "at30")}
    s5 = dict(at30_one=T1["score"][0], never=NV["score"][0], gain=round(T1["score"][0] - NV["score"][0], 2),
              pays_off=T1["score"][0] - NV["score"][0] >= 3.0)
    nv = stream["never"]["one"]
    g_at_nv = at_settings(gstar, nv["gi"], j0, nv["j1"])
    nv_sc = at_settings("never", nv["gi"], j0, nv["j1"])
    s6 = dict(G_at_never_settings=ms(g_at_nv), never=ms(nv_sc), gain=round(float((g_at_nv - nv_sc).mean()), 2),
              helps_alone=float((g_at_nv - nv_sc).mean()) >= 1.0)
    best_test = max(g_rules, key=lambda r: stream[r]["two"]["score"][0])
    s7 = dict(best_on_test=best_test, best_score=stream[best_test]["two"]["score"][0], G_score=G["score"][0],
              trusted=stream[best_test]["two"]["score"][0] - G["score"][0] <= 1.0)

    summary = dict(labels=labels, curve=curve, stream=stream, weights=weights_out, cost=cost_out,
                   verdict=dict(step0=dict(a=s0a, b=s0b, c=s0c), step1=s1, step2=s2, step3=s3, step4=s4, step5=s5,
                                step6=s6, step7=s7))
    (res / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["verdict"], indent=1))
    for w in ways:
        print(f"{w:13s} " + " ".join(f"{l}:{curve[w][l]['handicap']['H'][0]:5.1f}" for l in labels)
              + f" | none " + " ".join(f"{curve[w][l]['none']['H'][0]:5.1f}" for l in labels) + f" | reh {g1_score.get(w, '')}")
    for r in rules:
        a, b = stream[r]["two"], stream[r]["one"]
        print(f"stream {r:13s}: two {a['score'][0]:5.1f} (g {a['gamma']} h1 {a['h1']} h2 {a['h2']}; reh {a['rehearsal']['score']})"
              f" | one {b['score'][0]:5.1f} (h1 {b['h1']}) | none {stream[r]['none']['score'][0]:5.1f} | sessions "
              + " ".join(f"{a['sessions'][str(s)]['H'][0]:.1f}" for s in sessions))

    # ---------------- figure 1: the crossover
    xv = [xs[l] for l in labels]
    ticks = labels[:-1] + ["all (500)"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for w, col, lab in (("apart", INK2, "apart (EXP-13)"), ("full", BLUE, "full (EXP-14)"),
                        (fam_best["counted"] if fam_best["counted"] != g1 else None, AMBER, None),
                        (g1, GREEN, None)):
        if w is None:
            continue
        ax.plot(xv, [curve[w][l]["handicap"]["H"][0] for l in labels], marker="o", color=col,
                lw=2.4 if w == g1 else 1.6, label=lab or (pretty(w) + (" (G1*)" if w == g1 else "")))
    best_meas = max([w for w in g_ways if w.startswith("measured")], key=lambda w: g1_score[w])
    if best_meas != g1:
        ax.plot(xv, [curve[best_meas][l]["handicap"]["H"][0] for l in labels], marker="o", color=VIOLET, lw=1.6,
                label=pretty(best_meas))
    ax.set_xscale("log")
    ax.set_xticks(xv, ticks)
    ax.set_xlabel("pictures per new class")
    ax.set_ylabel("harmonic mean % (test)")
    ax.set_title("Harmonic mean among all 100, with the handicap: all-or-nothing against gradual")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig1_crossover")

    # ---------------- figure 2: gain over the better of apart and full, every gradual way
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    cols = {"counted": AMBER, "measured": VIOLET, "card": RED}
    for w in g_ways:
        fam = next(f for f in cols if w.startswith(f))
        v = [curve[w][l]["handicap"]["H"][0] - max(curve["apart"][l]["handicap"]["H"][0], curve["full"][l]["handicap"]["H"][0])
             for l in labels]
        ax.plot(range(len(labels)), v, marker="o", lw=2.2 if w == g1 else 0.9, color=cols[fam], alpha=1 if w == g1 else 0.6,
                label=pretty(w) + (" (G1*)" if w == g1 else ""))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axhline(-0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(labels)), ticks)
    ax.set_xlabel("pictures per new class")
    ax.set_ylabel("points of harmonic mean")
    ax.set_title("Gradual minus the better of apart and full (test)")
    ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig2_gain")

    # ---------------- figure 3: the weights against pictures
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    chosen = [curve[g1][l]["handicap"]["gamma"] for l in labels]
    gw = max(set(chosen), key=chosen.count)                  # the smoothing G1* chose most often
    for w in g_ways:
        if w.startswith("card"):
            continue
        fam = "counted" if w.startswith("counted") else "measured"
        v = [weights_out.get(f"{gw}|{l}", {}).get(w, np.nan) for l in labels]
        ax.plot(xv, v, marker="o", color=cols[fam], lw=2.2 if w == g1 else 1.0, alpha=1 if w == g1 else 0.7,
                label=pretty(w))
    ax.set_xscale("log")
    ax.set_xticks(xv, ticks)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("pictures per new class")
    ax.set_ylabel("weight of the new question")
    ax.set_title(f"Weight given to a new class's question (part 1, smoothing {gw}, mean over classes and seeds)")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig3_weights")

    # ---------------- figure 4: the stream
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    show = [("never", "one", NEUTRAL, "never"), ("at30", "one", BLUE, "threshold 30, one handicap (EXP-14)"),
            ("at30", "two", BLUE, "threshold 30, two handicaps"), (gstar, "two", GREEN, pretty(gstar) + " (G*)")]
    for rule, kind, col, lab in show:
        v = stream[rule][kind]["sessions"]
        ax.plot(sessions, [v[str(s)]["H"][0] for s in sessions], marker="o", color=col,
                ls="--" if (rule == "at30" and kind == "two") else "-", lw=2.4 if rule == gstar else 1.5, label=lab)
    ax.set_xticks(sessions, [str(s) for s in sessions])
    ax.set_xlabel("session")
    ax.set_ylabel("harmonic mean % (test)")
    ax.set_title("The long stream: harmonic mean after each session")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    order = rules
    two = [stream[r]["two"]["score"][0] for r in order]
    one = [stream[r]["one"]["score"][0] for r in order]
    colr = [NEUTRAL if r == "never" else BLUE if r in meta["thresh"] else cols[next(f for f in cols if r.startswith(f))]
            for r in order]
    ax.barh(range(len(order)), two, color=colr)
    ax.scatter(one, range(len(order)), color=INK, s=12, zorder=3, label="one handicap (h2 = 0)")
    for i, v in enumerate(two):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(order)), [pretty(r) + (" *" if r == gstar else "") for r in order], fontsize=7)
    ax.invert_yaxis()
    lo = min(min(two), min(one))
    ax.set_xlim(max(0, lo - 4), max(two) + 3)
    ax.set_title("Stream score, 8 sessions (bars: two handicaps)")
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "fig4_stream")

    # ---------------- figure 5: the second handicap
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for rule, col in ((gstar, GREEN), ("at30", BLUE)):
        ch = stream[rule]["two"]
        o, n = sarr("part1", rule, ch["gi"])
        v = hm(o, n)[:, :, :, ch["j1"]].mean(1).mean(0)
        ax.plot(G2, v, color=col, marker="o", ms=3, label=f"{pretty(rule)} (smoothing {ch['gamma']}, h1 {ch['h1']})")
        ax.scatter([ch["h2"]], [ch["score"][0]], s=50, facecolor="none", edgecolor=col, lw=1.5, zorder=3)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("second handicap h2 (subtracted from new classes, times their weight)")
    ax.set_ylabel("stream score % (test)")
    ax.set_title("Stream score against the second handicap (circles: chosen in the rehearsal)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig5_h2")


if __name__ == "__main__":
    main()
