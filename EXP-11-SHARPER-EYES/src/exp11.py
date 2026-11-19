"""EXP-11 sharper eyes: do cheap upgrades of the patch eyes (no backprop) make better cards?

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

Six versions of the patch eyes (A0 = EXP-10's eyes, unchanged):
  A0  1,600 stencils, one-sided, 4 quarters, no mirror           6,400 numbers
  A1  A0 + mirror (tables learn both copies; answers and new cards use the average of the two)
  A2  A1 + two-sided (positive and negative parts)               12,800
  A3  A1 + 3x3 regions                                            14,400
  A4  A1 + 3,200 stencils                                         12,800
  A5  A1 + two-sided + 3x3 regions, 800 stencils                  14,400
Memory as EXP-10 (running tables, learned view with all directions, borrowing from 2 nearest known cards
with weight 1 picture), but the shared table is kept in single precision around a fixed reference point
(stage 1's mean description) and the view is computed in place in one full-precision working table.
Smoothing 0.01 / 0.03 / 0.1 / 0.3 / 1 (times the mean of the table's diagonal), all logged.
Tests: T1 80 classes by distance through the smoothed table (EXP-8's memory); T2 20 new classes from
1/3/10 pictures among themselves (EXP-9); T3 the 20 new among all 100 with C0 (none), C1 (handicap),
C2 (certainty), C3 (certainty + handicap) (EXP-10); reference: new classes from all their pictures.
Rehearsal: memory after stage 6 (60 classes), stages 7-8 as pretend-new. Every setting is logged; the
choosing is done afterwards by src/figures.py from the rehearsal rows only.

Run:  python src/exp11.py                full run (5 seeds)
      EXP11_SMOKE=1 python src/exp11.py  tiny run to check the code
      EXP11_ARMS=A3 EXP11_SEEDS=1 EXP11_STOP=6 python src/exp11.py   one arm, one seed, up to stage 6
"""
import glob
import json
import os
import pathlib
import shutil
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or (os.name == "posix" and pathlib.Path("/kaggle/working").exists())
try:
    ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    ROOT = pathlib.Path.cwd()
OUT = pathlib.Path(os.environ.get("EXP11_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP11_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP11_SMOKE") == "1"

SEEDS = int(os.environ.get("EXP11_SEEDS", 1 if SMOKE else 5))
STOP = int(os.environ.get("EXP11_STOP", 8))           # last stage to run (8 = everything)
STAGES, PER_STAGE = 8, 10
N_PATCHES = 20_000 if SMOKE else 200_000
PATCH = 6
NORM_EPS = 10.0
ZCA_EPS = 0.1
THR = 0.1
SURPRISE_SEED = 1234
SHOTS = (1, 3, 10)
EPISODES = 20
CHUNK = 1000                                          # pictures per chunk when filling the tables

ARMS = {  # stencils, sides, regions per side, mirror
    "A0": dict(k=1600, sides=1, grid=2, mirror=False),
    "A1": dict(k=1600, sides=1, grid=2, mirror=True),
    "A2": dict(k=1600, sides=2, grid=2, mirror=True),
    "A3": dict(k=1600, sides=1, grid=3, mirror=True),
    "A4": dict(k=3200, sides=1, grid=2, mirror=True),
    "A5": dict(k=800, sides=2, grid=3, mirror=True),
}
RUN_ARMS = os.environ.get("EXP11_ARMS", ",".join(ARMS)).split(",")
K_MAX = max(a["k"] for a in ARMS.values())
SMOKE_DIV = 10

GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0]
ALPHA = 1.0
N_BORROW = 2
G1 = [round(x, 3) for x in np.arange(-0.2, 0.5001, 0.01)]     # handicap on cosine scores (C1)
G3 = [round(x, 3) for x in np.arange(-0.5, 1.0001, 0.02)]     # handicap on C2 scores, times the number of directions (C3)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F64 = torch.float64


# ----------------------------------------------------------------------------- power and timing (as EXP-10)
class PowerMeter:
    def __init__(self):
        self.samples, self.ok, self.proc = [], False, None
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "1000"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except OSError:
            pass

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.time(), float(line.strip().split("\n")[0])))
                self.ok = True
            except ValueError:
                pass

    def wh(self, t0, t1):
        w = [p for t, p in self.samples if t0 <= t <= t1]
        return float(np.mean(w)) * (t1 - t0) / 3600 if w else None

    def stop(self):
        if self.proc:
            self.proc.terminate()


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


class Clock:
    def __init__(self, meter):
        self.meter = meter

    def __enter__(self):
        sync()
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        sync()
        self.seconds = time.time() - self.t0
        self.wh = self.meter.wh(self.t0, self.t0 + self.seconds)


def cpu_peak_mb():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0)   # Linux: KB
    except ImportError:
        return None


# ----------------------------------------------------------------------------- data
def copy_kaggle_dataset():
    """Copy CIFAR-100 from the private Kaggle dataset (mitanshkanani/cifar100-python) so nothing is downloaded."""
    dst = DATA / "cifar-100-python"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test", "meta"):
        hits = [h for h in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if os.path.isfile(h)]
        if hits and not (dst / name).exists():
            shutil.copy(hits[0], dst / name)
            print("copied", hits[0], flush=True)


def load():
    import torchvision.datasets as tv
    DATA.mkdir(parents=True, exist_ok=True)
    if ON_KAGGLE:
        copy_kaggle_dataset()
    tr, te = tv.CIFAR100(DATA, train=True, download=True), tv.CIFAR100(DATA, train=False, download=True)
    f = lambda d: (torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(d.targets))
    (xtr, ytr), (xte, yte) = f(tr), f(te)
    if SMOKE:
        keep = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0))[: len(xtr) // 10]
        xtr, ytr = xtr[keep], ytr[keep]
    return xtr.to(dev), ytr.long().to(dev), xte.to(dev), yte.long().to(dev)


# ----------------------------------------------------------------------------- patch eyes
def norm_patches(x_u8):
    p = F.unfold(x_u8.float(), PATCH).transpose(1, 2)
    p = p - p.mean(2, keepdim=True)
    return p / torch.sqrt(p.var(2, unbiased=False, keepdim=True) + NORM_EPS)


def sample_patches(x_u8, n, gen):
    n_img = min(len(x_u8), max(64, n // 100))
    idx = torch.randperm(len(x_u8), generator=gen)[:n_img].to(dev)
    p = torch.cat([norm_patches(x_u8[idx[i:i + 500]]).reshape(-1, 3 * PATCH * PATCH) for i in range(0, n_img, 500)])
    return p[torch.randperm(len(p), generator=gen)[:n].to(dev)]


def fit_whitening(p):
    mu = p.mean(0)
    cov = torch.cov((p - mu).T.double())
    d, v = torch.linalg.eigh(cov)
    w = (v * (d + ZCA_EPS).rsqrt()) @ v.T
    return mu, w.float()


class Eyes:
    def __init__(self, mu, w, c, sides, grid, mirror):
        self.mu, self.w, self.sides, self.mirror = mu, w, sides, mirror
        self.cn = F.normalize(c, dim=1)
        pos = 32 - PATCH + 1
        r = (torch.arange(pos, device=dev) >= (pos + 1) // 2).long() if grid == 2 else torch.arange(pos, device=dev) // (pos // grid)
        self.q = F.one_hot((r[:, None] * grid + r[None, :]).reshape(-1), grid * grid).float()
        self.dim = len(c) * sides * grid * grid
        self.bs = max(1, int(3e7 // (pos * pos * len(c) * sides)))

    @torch.no_grad()
    def raw(self, x_u8):
        """Description of each picture as it is (square root of the matches added up per region)."""
        out = []
        for i in range(0, len(x_u8), self.bs):
            p = (norm_patches(x_u8[i:i + self.bs]) - self.mu) @ self.w
            z = F.normalize(p, dim=2) @ self.cn.T
            parts = [F.relu(z - THR)] + ([F.relu(-z - THR)] if self.sides == 2 else [])
            out.append(torch.cat([(f.transpose(1, 2) @ self.q).reshape(len(p), -1) for f in parts], 1).sqrt())
        return torch.cat(out)

    def table_views(self, x_u8):
        """What goes into the tables: the picture, and its mirror image as a second example."""
        return [self.raw(x_u8)] + ([self.raw(torch.flip(x_u8, dims=[3]))] if self.mirror else [])

    def answer(self, x_u8):
        """Description used to answer and to make new cards: the average of the picture and its mirror."""
        return 0.5 * (self.raw(x_u8) + self.raw(torch.flip(x_u8, dims=[3]))) if self.mirror else self.raw(x_u8)


# ----------------------------------------------------------------------------- the memory: compact running tables
class Tables:
    """Class sums and one shared table. The table is single precision, as sums around a fixed reference point."""

    def __init__(self, d, ref):
        self.a = torch.zeros(d, d, device=dev)
        self.ref = ref
        self.cls, self.sums, self.counts, self.pics = [], {}, {}, {}

    def register(self, classes):
        for c in classes:
            self.cls.append(c)
            self.sums[c] = torch.zeros(len(self.ref), device=dev, dtype=F64)
            self.counts[c], self.pics[c] = 0, 0

    def add(self, g, y, real):
        gc = g - self.ref
        self.a.addmm_(gc.T, gc)
        for c in torch.unique(y).tolist():
            m = y == c
            self.sums[c] += g[m].double().sum(0)
            self.counts[c] += int(m.sum())
            if real:
                self.pics[c] += int(m.sum())

    def info(self):
        s = torch.stack([self.sums[c] for c in self.cls])
        n = torch.tensor([self.counts[c] for c in self.cls], device=dev, dtype=F64)
        m = s / n[:, None]
        return dict(m=m, n=n, pics=torch.tensor([self.pics[c] for c in self.cls], device=dev, dtype=F64),
                    mu=s.sum(0) / n.sum(), cls=list(self.cls))

    def fill(self, buf, info):
        """buf <- within-class shared table, full precision."""
        buf.copy_(self.a)
        mc = (info["m"] - self.ref.double()) * info["n"].sqrt()[:, None]
        buf.addmm_(mc.T, mc, alpha=-1)
        buf /= info["n"].sum() - len(info["cls"])
        return buf


def chol_inplace(a, b=1024):
    """Lower Cholesky factor of a, written over a (blocked, so no second full-size table is made)."""
    n = a.shape[0]
    for k in range(0, n, b):
        e = min(k + b, n)
        a[k:e, k:e] = torch.linalg.cholesky(a[k:e, k:e])
        if e < n:
            a[e:, k:e] = torch.linalg.solve_triangular(a[k:e, k:e].mT, a[e:, k:e], upper=True, left=False)
            for i in range(e, n, b):
                i2 = min(i + b, n)
                a[i:i2, e:i2].addmm_(a[i:i2, k:e], a[e:i2, k:e].T, alpha=-1)
    return a.tril_()


def bundle(tb, buf, info, gamma):
    """The learned view at one smoothing: P (whitened by the smoothed table) and Q (whitened inside the view)."""
    tb.fill(buf, info)
    md = buf.diagonal().mean()
    buf.diagonal().add_(gamma * md)
    chol_inplace(buf)
    mc = (info["m"] - info["mu"]).T
    u, _, _ = torch.linalg.svd(torch.linalg.solve_triangular(buf, mc, upper=False), full_matrices=False)
    r = len(info["cls"]) - 1
    P = torch.linalg.solve_triangular(buf.mT, u[:, :r].contiguous(), upper=True)
    sv = torch.eye(r, device=dev, dtype=F64) - gamma * md * (P.T @ P)      # = P' (table without smoothing) P
    sv.diagonal().add_(1e-6 * sv.diagonal().mean())
    lv = torch.linalg.cholesky(sv)
    Q = torch.linalg.solve_triangular(lv, P.T, upper=False).T
    return P.float(), Q.float(), r


# ----------------------------------------------------------------------------- scoring when old and new compete (as EXP-10)
def new_cards(zp, idx, compare, known, k):
    c = zp[idx].mean(2)
    if compare == "angle":
        near = torch.einsum("ecr,jr->ecj", F.normalize(c, dim=2), known)
    else:
        near = 2 * torch.einsum("ecr,jr->ecj", c, known) - known.pow(2).sum(1)
    nb = near.topk(N_BORROW, dim=2).indices
    c = (k * c + ALPHA * known[nb].mean(2)) / (k + ALPHA)
    return F.normalize(c, dim=2) if compare == "angle" else c


def raw_scores(compare, known, pics_old, z_sets, zpool, idx, k):
    """Scores of each projected test set against old cards then new cards: {name: [N, E, C_old + C_new]}."""
    if compare == "angle":
        f = lambda z: F.normalize(z, dim=1)
        nc = new_cards(f(zpool), idx, "angle", known, k)
        cards = torch.cat([known[None].expand(len(nc), -1, -1), nc], 1)
        return {n: torch.einsum("nr,ecr->nec", f(z), cards) for n, z in z_sets.items()}
    nc = new_cards(zpool, idx, "dist", known, k)
    cards = torch.cat([known[None].expand(len(nc), -1, -1), nc], 1)
    infl = torch.cat([1 + 1 / pics_old, torch.full((nc.shape[1],), 1 + 1 / (k + ALPHA), device=dev)])
    out = {}
    for n, z in z_sets.items():
        d2 = z.pow(2).sum(1)[:, None, None] - 2 * torch.einsum("nr,ecr->nec", z, cards) + cards.pow(2).sum(2)[None]
        out[n] = -d2 / infl
    return out


def sweep(s, y, cls, n_old, grid):
    mask = torch.zeros(s.shape[2], device=dev)
    mask[:n_old] = 1
    return [round((cls[(s - g * mask).argmax(2)] == y[:, None]).float().mean().item(), 4) for g in grid]


def compete(ka, kd, pics_old, r, zo, oy, zn, ny, zp, draws, cls, k):
    """C0-C3 for one smoothing. zo/zn/zp: (angle part, distance part) of old test, new test and new pool."""
    n_old = len(ka)
    out = {}
    for comp, names, known, i in (("angle", ("C0", "C1"), ka, 0), ("dist", ("C2", "C3"), kd, 1)):
        sc = raw_scores(comp, known, pics_old, {"old": zo[i], "new": zn[i]}, zp[i], draws, k)
        new_only = round((cls[n_old:][sc["new"][:, :, n_old:].argmax(2)] == ny[:, None]).float().mean().item(), 4)
        out[names[0]] = dict(old=sweep(sc["old"], oy, cls, n_old, [0.0])[0], new=sweep(sc["new"], ny, cls, n_old, [0.0])[0],
                             new_only=new_only)
        grid = G1 if comp == "angle" else [g * r for g in G3]
        out[names[1]] = dict(old=sweep(sc["old"], oy, cls, n_old, grid), new=sweep(sc["new"], ny, cls, n_old, grid),
                             new_only=new_only)
        del sc
    return out


def make_draws(y_pool, classes, rng, shots, episodes=EPISODES):
    idx = {c: torch.nonzero(y_pool == c).squeeze(1).cpu().numpy() for c in classes}
    return {k: torch.tensor(np.array([[rng.choice(idx[c], k, replace=False) for c in classes]
                                      for _ in range(episodes)]), device=dev) for k in shots}


# ----------------------------------------------------------------------------- one seed
def run(seed, data, meter, log):
    xtr, ytr, xte, yte = data
    rng = np.random.default_rng(seed)
    surprise = sorted(np.random.default_rng(SURPRISE_SEED).permutation(100)[:20].tolist())
    rest = rng.permutation([c for c in range(100) if c not in surprise]).tolist()
    groups = [rest[i * PER_STAGE:(i + 1) * PER_STAGE] for i in range(STAGES)]
    upto = lambda s: [c for g in groups[:s] for c in g]

    def sel(x, y, classes):
        m = torch.isin(y, torch.tensor(classes, device=dev))
        return x[m], y[m]

    def write(**kw):
        log.write(json.dumps(dict(seed=seed, **kw)) + "\n")
        log.flush()

    def rw(x):
        return None if x is None else round(x, 5)

    gx0, _ = sel(xtr, ytr, groups[0])

    # stencils: cut from stage 1's pictures exactly as EXP-10 (same calls, same generator), K_MAX of them
    gen = torch.Generator().manual_seed(seed * 100 + 2)
    torch.manual_seed(seed * 100 + 2)
    meter_t0 = time.time()
    p = sample_patches(gx0, N_PATCHES, gen)
    mu, w = fit_whitening(p)
    stencils = ((p - mu) @ w)[torch.randperm(len(p), generator=gen)[:K_MAX].to(dev)]
    del p
    sync()
    build_seconds = time.time() - meter_t0

    for name in RUN_ARMS:
        cfg = ARMS[name]
        k_st = cfg["k"] // SMOKE_DIV if SMOKE else cfg["k"]
        eyes = Eyes(mu, w, stencils[:k_st], cfg["sides"], cfg["grid"], cfg["mirror"])
        d = eyes.dim
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        times = dict(describe=0.0, tables=0.0, view=0.0, scoring=0.0)
        wh = {k: 0.0 for k in times} if meter.ok else None

        def tick(key, c):
            times[key] += c.seconds
            if wh is not None and c.wh is not None:
                wh[key] += c.wh

        def describe_answer(x):
            with Clock(meter) as c:
                g = torch.cat([eyes.answer(x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)])
            tick("describe", c)
            return g

        def fill_tables(tb, x, y):
            for i in range(0, len(x), CHUNK):
                with Clock(meter) as c:
                    views = eyes.table_views(x[i:i + CHUNK])
                tick("describe", c)
                with Clock(meter) as c:
                    for j, v in enumerate(views):
                        tb.add(v, y[i:i + CHUNK], real=(j == 0))
                tick("tables", c)
                del views

        def views_all(tb):
            """The learned view at every smoothing, in one working table that is freed afterwards."""
            info = tb.info()
            with Clock(meter) as c:
                buf = torch.empty(d, d, device=dev, dtype=F64)
                out = [bundle(tb, buf, info, gm) for gm in GAMMAS]
                del buf
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            tick("view", c)
            return info, out

        def project(x, info, vs):
            """Describe pictures (in chunks) and keep only their coordinates in every view."""
            mu_g = info["mu"].float()
            wcat = torch.cat([torch.cat([P, Q], 1) for P, Q, _ in vs], 1)
            zs = []
            for i in range(0, len(x), CHUNK):
                g = describe_answer(x[i:i + CHUNK])
                zs.append((g - mu_g) @ wcat)
                del g
            return torch.cat(zs)

        def evaluate(task, tb, new_cls):
            info, vs = views_all(tb)
            r = vs[0][2]
            n_old = len(info["cls"])
            ox, oy = sel(xte, yte, info["cls"])
            px, py = sel(xtr, ytr, new_cls)
            nx, ny = sel(xte, yte, new_cls)
            zo_all, zp_all, zn_all = project(ox, info, vs), project(px, info, vs), project(nx, info, vs)
            draws = make_draws(py, new_cls, np.random.default_rng({"part1": 10_000, "rehearsal": 20_000}[task] + seed), SHOTS)
            cls = torch.tensor(info["cls"] + list(new_cls), device=dev)
            cls_old = cls[:n_old]
            with Clock(meter) as c:
                km = info["m"].float() - info["mu"].float()
                for gi, (P, Q, _) in enumerate(vs):
                    sl = lambda z, j: z[:, (2 * gi + j) * r:(2 * gi + j + 1) * r]
                    kp, kd = km @ P, km @ Q
                    ka = F.normalize(kp, dim=1)
                    # T1: old classes among themselves, distance through the smoothed table (EXP-8's memory) and angle
                    zo_p = sl(zo_all, 0)
                    d2 = zo_p.pow(2).sum(1)[:, None] - 2 * zo_p @ kp.T + kp.pow(2).sum(1)[None]
                    t1_lda = round((cls_old[d2.argmin(1)] == oy).float().mean().item(), 4)
                    t1_ang = round((cls_old[(F.normalize(zo_p, dim=1) @ ka.T).argmax(1)] == oy).float().mean().item(), 4)
                    write(eyes=name, task=task, kind="t1", gamma=GAMMAS[gi], known=n_old, lda=t1_lda, angle=t1_ang)
                    del d2
                    zo, zn, zp = (sl(zo_all, 0), sl(zo_all, 1)), (sl(zn_all, 0), sl(zn_all, 1)), (sl(zp_all, 0), sl(zp_all, 1))
                    for k, idx in draws.items():
                        res = compete(ka, kd, info["pics"].float(), r, zo, oy, zn, ny, zp, idx, cls, k)
                        for corr, rr in res.items():
                            write(eyes=name, task=task, kind="compete", gamma=GAMMAS[gi], shots=k, corr=corr, known=n_old, **rr)
                    c0 = res["C0"]
                    print(f"seed {seed} {name} {task} gamma {GAMMAS[gi]} | T1 {t1_lda:.3f} | k={k} C0 old {c0['old']:.3f} "
                          f"new {c0['new']:.3f} new-only {c0['new_only']:.3f}", flush=True)
            tick("scoring", c)
            del zo_all, zp_all, zn_all, vs

        def reference_all(tb):
            """New classes learned from all their pictures (not a method): angle, no handicap, every smoothing."""
            px, py = sel(xtr, ytr, surprise)
            tb.register(surprise)
            fill_tables(tb, px, py)
            info, vs = views_all(tb)
            r = vs[0][2]
            ox, oy = sel(xte, yte, info["cls"][:80])
            nx, ny = sel(xte, yte, surprise)
            zo_all, zn_all = project(ox, info, vs), project(nx, info, vs)
            cls = torch.tensor(info["cls"], device=dev)
            km = info["m"].float() - info["mu"].float()
            for gi, (P, Q, _) in enumerate(vs):
                ka = F.normalize(km @ P, dim=1)
                acc = lambda z, y: round((cls[(F.normalize(z[:, 2 * gi * r:(2 * gi + 1) * r], dim=1) @ ka.T).argmax(1)] == y)
                                         .float().mean().item(), 4)
                write(eyes=name, task="part1_ref_all", kind="ref", gamma=GAMMAS[gi], old=acc(zo_all, oy), new=acc(zn_all, ny))
            del zo_all, zn_all, vs

        tb = None
        for s, group in enumerate(groups[:STOP]):
            stage = s + 1
            gx, gy = sel(xtr, ytr, group)
            if tb is None:      # the fixed reference point of the compact table: stage 1's mean description
                with Clock(meter) as c:
                    ref = sum(eyes.raw(gx[i:i + CHUNK]).double().sum(0) for i in range(0, len(gx), CHUNK)) / len(gx)
                tick("describe", c)
                tb = Tables(d, ref.float())
            tb.register(group)
            fill_tables(tb, gx, gy)
            if stage == 6:
                evaluate("rehearsal", tb, groups[6] + groups[7])
            if stage == 8:
                evaluate("part1", tb, surprise)
                reference_all(tb)
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 20, 0) if dev.type == "cuda" else None
        peak_res = round(torch.cuda.max_memory_reserved() / 2 ** 20, 0) if dev.type == "cuda" else None
        write(eyes=name, task="cost", seconds={k: round(v, 2) for k, v in times.items()},
              wh={k: rw(v) for k, v in wh.items()} if wh else None, feature_dim=d, peak_mem_mb=peak,
              peak_reserved_mb=peak_res, cpu_peak_mb=cpu_peak_mb(), stencil_build_seconds=round(build_seconds, 2))
        print(f"seed {seed} {name} done: dim {d}, peak GPU {peak} MB (reserved {peak_res}), times {times}", flush=True)
        del tb, eyes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "none"
    print("device", dev, gpu, "torch", torch.__version__, "smoke", SMOKE, "arms", RUN_ARMS, flush=True)
    if dev.type != "cuda" and not SMOKE:
        raise SystemExit("No GPU: refusing to start the full run on CPU.")
    meter = PowerMeter()
    t0 = time.time()
    data = load()
    print(f"data ready, {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(OUT / "runs.jsonl", "w") as log:
        for seed in range(SEEDS):
            run(seed, data, meter, log)
            print(f"== seed {seed} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    meter.stop()
    meta = dict(gpu=gpu, torch=torch.__version__, on_kaggle=ON_KAGGLE, smoke=SMOKE, seeds=SEEDS, stop=STOP, stages=STAGES,
                per_stage=PER_STAGE, arms={a: ARMS[a] for a in RUN_ARMS}, n_patches=N_PATCHES, thr=THR, shots=SHOTS,
                episodes=EPISODES, gammas=GAMMAS, alpha=ALPHA, n_borrow=N_BORROW, g1=G1, g3=G3,
                surprise_seed=SURPRISE_SEED, power_reported=meter.ok, total_minutes=round((time.time() - t0) / 60, 1),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
