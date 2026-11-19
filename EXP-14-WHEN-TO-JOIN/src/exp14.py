"""EXP-14 when to join: after how many pictures should a new class join the memory's learned view, and does a stream that
holds new classes apart and lets them join once they have N pictures (consolidation) beat never or always joining?
No backpropagation anywhere.

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

Eyes and memory: EXP-11's version A2 (1,600 stencils, two-sided, 4 quarters, mirrored; 12,800 numbers), compact tables,
learned view of the old classes with all directions, new cards by angle with borrowing from the 2 nearest known cards
(weight 1 picture). Same seeds, classes and draws as EXP-13 (no copies).
Three ways to treat the new classes:
  apart  card only, view of the old classes (EXP-13)
  view   the new cards add their own directions to the view; the shared table is unchanged. With all directions kept,
         this equals rebuilding the view from the old and new cards with the old table
  table  the new pictures (and mirrors) also go into the running tables and the view is rebuilt (EXP-10's C4, EXP-11's
         reference); fewer draws because each needs a full rebuild
Part 1, the crossover: 1, 3, 10, 30, 100 pictures per new class and all of them.
Part 2, the stream: 20 new classes arrive 5 per session over 4 sessions; a class has 3, 10, 30, 100 pictures in the
session it arrives and the next ones; it joins the view once it has N pictures (never / 3 / 10 / 30 / 100).
Every handicap h (subtracted from the old classes' scores; h = 0 is no correction) and smoothing is logged; the choosing
is done afterwards by src/figures.py from the rehearsal rows.

Run:  python src/exp14.py                full run (5 seeds)
      EXP14_SMOKE=1 python src/exp14.py  tiny run to check the code
      EXP14_SEEDS=1 EXP14_STOP=6 python src/exp14.py   one seed, up to the stage-6 rehearsal
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
OUT = pathlib.Path(os.environ.get("EXP14_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP14_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP14_SMOKE") == "1"

SEEDS = int(os.environ.get("EXP14_SEEDS", 1 if SMOKE else 5))
STOP = int(os.environ.get("EXP14_STOP", 8))           # last stage to run (8 = everything)
STAGES, PER_STAGE = 8, 10
N_PATCHES = 20_000 if SMOKE else 200_000
PATCH = 6
NORM_EPS = 10.0
ZCA_EPS = 0.1
THR = 0.1
SURPRISE_SEED = 1234
SHOTS = (1, 3, 10)                                    # drawn exactly as EXP-11 to EXP-13 (same random generator order)
BIG = {30: 20, 100: 10}                               # more pictures per new class: draws each (as EXP-13)
EPISODES = 20
TABLE_DRAWS = {3: 2, 10: 2, 30: 1, 100: 1, "all": 1}  # "table and view": the first draws of each list
STREAM_RUNS = 10
SESSIONS, PER_SESSION = 4, 5
LEVELS = (3, 10, 30, 100)                             # pictures of a class in the session it arrives and the next ones
POLICIES = {"never": None, "once": 3, "at10": 10, "at30": 30, "at100": 100}
CHUNK = 1000

EYES = dict(k=1600, sides=2, grid=2, mirror=True)     # EXP-11's A2
K_MAX = 3200                                           # stencils are cut exactly as in EXP-11 (A2 used the first 1,600)
SMOKE_DIV = 10

GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0]
ALPHA = 1.0
N_BORROW = 2
G1 = [round(x, 3) for x in np.arange(-0.2, 0.5001, 0.01)]     # handicap on cosine scores, as EXP-10 to EXP-13

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F64 = torch.float64


# ----------------------------------------------------------------------------- power and timing (as EXP-11)
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


# ----------------------------------------------------------------------------- data (as EXP-11)
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


# ----------------------------------------------------------------------------- patch eyes (as EXP-11; image-specific)
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
        out = []
        for i in range(0, len(x_u8), self.bs):
            p = (norm_patches(x_u8[i:i + self.bs]) - self.mu) @ self.w
            z = F.normalize(p, dim=2) @ self.cn.T
            parts = [F.relu(z - THR)] + ([F.relu(-z - THR)] if self.sides == 2 else [])
            out.append(torch.cat([(f.transpose(1, 2) @ self.q).reshape(len(p), -1) for f in parts], 1).sqrt())
        return torch.cat(out)

    def table_views(self, x_u8):
        return [self.raw(x_u8)] + ([self.raw(torch.flip(x_u8, dims=[3]))] if self.mirror else [])

    def answer(self, x_u8):
        return 0.5 * (self.raw(x_u8) + self.raw(torch.flip(x_u8, dims=[3]))) if self.mirror else self.raw(x_u8)


# ----------------------------------------------------------------------------- the memory: compact running tables (as EXP-11)
class Tables:
    def __init__(self, d, ref):
        self.a = torch.zeros(d, d, device=dev)
        self.ref = ref
        self.cls, self.sums, self.counts, self.pics = [], {}, {}, {}

    def register(self, classes):
        for c in classes:
            self.cls.append(c)
            self.sums[c] = torch.zeros(len(self.ref), device=dev, dtype=F64)
            self.counts[c], self.pics[c] = 0, 0

    def forget(self, classes):
        """Undo register() + add() for these classes (the shared table itself is restored by the caller)."""
        for c in classes:
            self.cls.remove(c)
            del self.sums[c], self.counts[c], self.pics[c]

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
        buf.copy_(self.a)                        # works with the table on the CPU too (used while views are joined)
        mc = (info["m"] - self.ref.double()) * info["n"].sqrt()[:, None]
        buf.addmm_(mc.T, mc, alpha=-1)
        buf /= info["n"].sum() - len(info["cls"])
        return buf


def chol_inplace(a, b=1024):
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


def bundle(tb, buf, info, gamma, keep=False):
    """The learned view P at one smoothing (as EXP-11 to EXP-13). keep=True also returns the whitened directions U and
    leaves the smoothed table's Cholesky factor L in buf (used to add directions for joining classes)."""
    tb.fill(buf, info)
    md = buf.diagonal().mean()
    buf.diagonal().add_(gamma * md)
    chol_inplace(buf)
    mc = (info["m"] - info["mu"]).T
    u, _, _ = torch.linalg.svd(torch.linalg.solve_triangular(buf, mc, upper=False), full_matrices=False)
    r = len(info["cls"]) - 1
    U = u[:, :r].contiguous()
    P = torch.linalg.solve_triangular(buf.mT, U, upper=True)
    return (P.float(), r, U) if keep else (P.float(), r)


def join_directions(L, U, mc, sizes):
    """Extra view directions for groups of joining classes, with the old table: for each group, the part of its cards
    (whitened by L) that the old directions U cannot express, made orthonormal, mapped back (as P = L^-T U).
    mc: (d, sum(sizes)) cards minus the old centre, float64. Returns (d, sum(sizes)) float32."""
    b = torch.linalg.solve_triangular(L, mc, upper=False)
    for _ in range(2):                                  # twice, for numerical safety
        b -= U @ (U.T @ b)
    q, s = [], 0
    for m in sizes:
        blk = b[:, s:s + m]
        blk = torch.linalg.qr(blk).Q                    # orthonormal directions for this group
        q.append(blk)
        s += m
    q = torch.cat(q, 1)
    del b
    return torch.linalg.solve_triangular(L.mT, q, upper=True).float()


# ----------------------------------------------------------------------------- scoring (as EXP-11's C0 and C1)
def new_cards(zp, idx, known, k):
    """New cards by angle with borrowing from the 2 nearest known cards, weight ALPHA pictures (as EXP-9 to EXP-13).
    zp: (N, D) or (E, N, D) normalized picture coordinates; idx: (E, C, k); known: (C_old, D) or (E, C_old, D)."""
    if zp.dim() == 2:
        c = zp[idx].mean(2)
    else:
        c = torch.stack([zp[e][idx[e]] for e in range(len(idx))]).mean(2)
    if known.dim() == 2:
        near = torch.einsum("ecr,jr->ecj", F.normalize(c, dim=2), known)
        nb = near.topk(N_BORROW, dim=2).indices
        borrow = known[nb].mean(2)
    else:
        near = torch.einsum("ecr,ejr->ecj", F.normalize(c, dim=2), known)
        nb = near.topk(N_BORROW, dim=2).indices
        borrow = torch.stack([known[e][nb[e]] for e in range(len(nb))]).mean(2)
    c = (k * c + ALPHA * borrow) / (k + ALPHA)
    return F.normalize(c, dim=2)


def compete(so, sn, ka, nc, oy, ny, cls_old, cls_new, hg):
    """Old and new accuracy among all classes for every handicap, per draw. so, sn: (E, N, D) normalized test
    coordinates (E may be 1 and broadcast); ka: (E, C_old, D); nc: (E, C_new, D). Returns old, new (E, len(hg)), and
    the new classes named among themselves (E,)."""
    cos_oo = so @ ka.mT
    oc_old = (cls_old[cos_oo.argmax(2)] == oy).float()
    best_old_o = cos_oo.max(2).values
    del cos_oo
    best_old_n = (sn @ ka.mT).max(2).values
    m_old = (so @ nc.mT).max(2).values - best_old_o
    cos_nn = sn @ nc.mT
    m_new = cos_nn.max(2).values - best_old_n
    cn = (cls_new[cos_nn.argmax(2)] == ny).float()
    # answer is new iff best new score > best old score - h (EXP-11's C1; h = 0 is no correction)
    old = ((m_old[:, :, None] <= -hg).float() * oc_old[:, :, None]).mean(1)
    new = ((m_new[:, :, None] > -hg).float() * cn[:, :, None]).mean(1)
    return old, new, cn.mean(1)


def ints(t):
    return (t * 10000).round().int().tolist()


def make_draws(y_pool, classes, rng, shots, episodes=EPISODES):
    idx = {c: torch.nonzero(y_pool == c).squeeze(1).cpu().numpy() for c in classes}
    eps = episodes if isinstance(episodes, dict) else {k: episodes for k in shots}
    return {k: torch.tensor(np.array([[rng.choice(idx[c], k, replace=False) for c in classes]
                                      for _ in range(eps[k])]), device=dev) for k in shots}


# ----------------------------------------------------------------------------- one seed
def run(seed, data, meter, log):
    xtr, ytr, xte, yte = data
    rng = np.random.default_rng(seed)
    surprise = sorted(np.random.default_rng(SURPRISE_SEED).permutation(100)[:20].tolist())
    rest = rng.permutation([c for c in range(100) if c not in surprise]).tolist()
    groups = [rest[i * PER_STAGE:(i + 1) * PER_STAGE] for i in range(STAGES)]

    def sel(x, y, classes):
        m = torch.isin(y, torch.tensor(classes, device=dev))
        return x[m], y[m]

    def write(**kw):
        log.write(json.dumps(dict(seed=seed, **kw)) + "\n")
        log.flush()

    gx0, _ = sel(xtr, ytr, groups[0])
    gen = torch.Generator().manual_seed(seed * 100 + 2)
    torch.manual_seed(seed * 100 + 2)
    p = sample_patches(gx0, N_PATCHES, gen)
    mu, w = fit_whitening(p)
    stencils = ((p - mu) @ w)[torch.randperm(len(p), generator=gen)[:K_MAX].to(dev)]
    del p

    k_st = EYES["k"] // SMOKE_DIV if SMOKE else EYES["k"]
    eyes = Eyes(mu, w, stencils[:k_st], EYES["sides"], EYES["grid"], EYES["mirror"])
    d = eyes.dim
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    times = dict(describe=0.0, tables=0.0, view=0.0, join=0.0, table_view=0.0, scoring=0.0)
    wh = {k: 0.0 for k in times} if meter.ok else None
    hg = torch.tensor(G1, device=dev, dtype=torch.float32)

    def tick(key, c):
        times[key] += c.seconds
        if wh is not None and c.wh is not None:
            wh[key] += c.wh

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

    def to_host(t):
        t = t.cpu()                                   # ordinary (not page-locked) memory: kinder to an 8-16 GB laptop
        if ON_KAGGLE and dev.type == "cuda":
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
        return t

    def project(x, mu_g, wcat):
        """Describe pictures (picture and mirror averaged); keep their coordinates in every view (on the GPU) and the
        full descriptions (on the CPU, for the views that change later)."""
        out, raw = [], []
        with Clock(meter) as c:
            for i in range(0, len(x), CHUNK):
                g = eyes.answer(x[i:i + CHUNK])
                out.append((g - mu_g) @ wcat)
                raw.append(g.cpu())
                del g
        tick("describe", c)
        return torch.cat(out), to_host(torch.cat(raw))

    def stream_through(xh, mu_g, w):
        """(x - mu) @ w for descriptions kept on the CPU, in chunks."""
        return torch.cat([(xh[i:i + CHUNK].to(dev, non_blocking=True) - mu_g) @ w for i in range(0, len(xh), CHUNK)])

    def evaluate(task, tb, new_cls):
        info = tb.info()
        mu_g = info["mu"].float()
        n_old = len(info["cls"])
        with Clock(meter) as c:
            buf = torch.empty(d, d, device=dev, dtype=F64)
            vs = [bundle(tb, buf, info, g) for g in GAMMAS]
            del buf
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        tick("view", c)
        r = vs[0][1]
        wcat = torch.cat([P for P, _ in vs], 1)
        ox, oy = sel(xte, yte, info["cls"])
        px, py = sel(xtr, ytr, new_cls)
        nx, ny = sel(xte, yte, new_cls)
        draw_rng = np.random.default_rng({"part1": 10_000, "rehearsal": 20_000}[task] + seed)
        draws = make_draws(py, new_cls, draw_rng, SHOTS)                         # identical to EXP-11 to EXP-13
        n_min = min(int((py == c).sum()) for c in new_cls)
        big = {k: e for k, e in BIG.items() if k < n_min}
        draws.update(make_draws(py, new_cls, draw_rng, tuple(big), big))
        draws[n_min] = torch.stack([torch.nonzero(py == c).squeeze(1)[:n_min] for c in new_cls])[None]   # all pictures
        label = lambda k: "all" if k == n_min else k
        # the stream: arrival order of the 20 new classes and each class's pictures in the order they arrive
        srng = np.random.default_rng({"part1": 30_000, "rehearsal": 40_000}[task] + seed)
        levels = [min(n, n_min) for n in LEVELS]
        s_order, s_pics = [], []
        for _ in range(STREAM_RUNS):
            s_order.append(srng.permutation(len(new_cls)))
            s_pics.append(torch.tensor(np.array([srng.permutation(torch.nonzero(py == c).squeeze(1).cpu().numpy())[:levels[-1]]
                                                 for c in new_cls]), device=dev))
        zo_all, xo_h = project(ox, mu_g, wcat)
        zn_all, xn_h = project(nx, mu_g, wcat)
        zp_all, xp_h = project(px, mu_g, wcat)
        cls_old = torch.tensor(info["cls"], device=dev)
        cls_new = torch.tensor(list(new_cls), device=dev)
        km = info["m"].float() - mu_g

        # ---------------- apart (EXP-13's way): every draw, every smoothing
        with Clock(meter) as c:
            for gi, gamma in enumerate(GAMMAS):
                sl = lambda z: F.normalize(z[:, gi * r:(gi + 1) * r], dim=1)
                ka = F.normalize(km @ vs[gi][0], dim=1)
                so, sn, sp = sl(zo_all), sl(zn_all), sl(zp_all)
                oc_old = (cls_old[(so @ ka.T).argmax(1)] == oy).float()
                write(task=task, kind="old_only", gamma=gamma, known=n_old, old=round(oc_old.mean().item(), 4))
                for k, idx in draws.items():
                    nc = new_cards(sp, idx, ka, k)
                    old, new, no = compete(so[None], sn[None], ka[None], nc, oy, ny, cls_old, cls_new, hg)
                    write(task=task, kind="cross", way="apart", gamma=gamma, shots=label(k), known=n_old,
                          old=[ints(o) for o in old], new=[ints(o) for o in new], new_only=ints(no))
                    if k == 3:
                        ih = G1.index(0.08)
                        print(f"seed {seed} {task} gamma {gamma} k=3 apart | h=0.08 old {old[:, ih].mean():.4f} "
                              f"new {new[:, ih].mean():.4f} | new-only {no.mean():.3f}", flush=True)
        tick("scoring", c)

        # ---------------- table and view: new pictures go into the tables, view rebuilt (fewer draws)
        a_host = to_host(tb.a)
        for key, n_dr in TABLE_DRAWS.items():
            k = n_min if key == "all" else key
            if k not in draws:
                continue
            for e in range(n_dr):
                idx = draws[k][e]                                                   # (20, k)
                tb.register(new_cls)
                flat = idx.reshape(-1)
                fill_tables(tb, px[flat], py[flat])
                info2 = tb.info()
                mu2 = info2["mu"].float()
                with Clock(meter) as c:
                    buf = torch.empty(d, d, device=dev, dtype=F64)
                    vs2 = [bundle(tb, buf, info2, g) for g in GAMMAS]
                    del buf
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                tick("table_view", c)
                tb.a.copy_(a_host)                                                  # the old tables, exactly
                tb.forget(new_cls)
                r2 = vs2[0][1]
                w2 = torch.cat([P for P, _ in vs2], 1)
                with Clock(meter) as c:
                    zo2, zn2 = stream_through(xo_h, mu2, w2), stream_through(xn_h, mu2, w2)
                    zp2 = (xp_h[flat.cpu()].to(dev) - mu2) @ w2
                tick("join", c)
                km2 = info2["m"].float() - mu2
                loc = torch.arange(len(flat), device=dev).reshape(1, *idx.shape)
                with Clock(meter) as c:
                    for gi, gamma in enumerate(GAMMAS):
                        sl = lambda z: F.normalize(z[:, gi * r2:(gi + 1) * r2], dim=1)
                        kall = km2 @ vs2[gi][0]
                        ka = F.normalize(kall[:n_old], dim=1)
                        so, sn = sl(zo2), sl(zn2)
                        nc = new_cards(sl(zp2), loc, ka, k)
                        old, new, no = compete(so[None], sn[None], ka[None], nc, oy, ny, cls_old, cls_new, hg)
                        write(task=task, kind="cross", way="table", gamma=gamma, shots=label(k), draw=e, known=n_old,
                              old=[ints(o) for o in old], new=[ints(o) for o in new], new_only=ints(no))
                        if key == "all":                                            # EXP-11's reference cards, h = 0
                            ref = F.normalize(kall[n_old:], dim=1)[None]
                            o0, n0, no0 = compete(so[None], sn[None], ka[None], ref, oy, ny, cls_old, cls_new,
                                                  torch.zeros(1, device=dev))
                            write(task=task, kind="ref", gamma=gamma, old=round(o0.item(), 4), new=round(n0.item(), 4),
                                  new_only=round(no0.item(), 4))
                tick("scoring", c)
                del zo2, zn2, zp2, vs2, w2, info2

        # ---------------- view: the new cards add their own directions; the table stays as it was
        # every group of joining classes: part 1 (all 20 classes, every draw) and the stream (every run/session/rule)
        groups_j = []                                                    # (name, class positions, picture lists)
        for k, idx in draws.items():
            for e in range(idx.shape[0]):
                groups_j.append((("cross", k, e), list(range(len(new_cls))), [idx[e, j] for j in range(len(new_cls))]))
        arrived_n = {}                                                   # (run, session) -> {position: pictures}
        for e in range(STREAM_RUNS):
            for s in range(1, SESSIONS + 1):
                have = {int(j): levels[s - 1 - pos // PER_SESSION]
                        for pos, j in enumerate(s_order[e][:s * PER_SESSION])}
                arrived_n[e, s] = have
                for pol, nmin in POLICIES.items():
                    if nmin is None:
                        continue
                    join = [j for j, n in have.items() if n >= nmin]
                    if join:
                        groups_j.append((("stream", e, s, pol), join, [s_pics[e][j, :have[j]] for j in join]))
        # the cards (class averages of the full descriptions) of every joining class, once
        with Clock(meter) as c:
            means = torch.stack([xp_h[pics.cpu()].double().mean(0) for _, _, pl in groups_j for pics in pl], 1)
        tick("join", c)
        sizes = [len(js) for _, js, _ in groups_j]
        starts = np.concatenate([[0], np.cumsum(sizes)])
        col = {g[0]: (int(starts[i]), int(starts[i + 1])) for i, g in enumerate(groups_j)}
        a_dev = tb.a
        tb.a = a_host                                                    # the shared table waits on the CPU meanwhile
        del a_dev
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        buf = torch.empty(d, d, device=dev, dtype=F64)
        for gi, gamma in enumerate(GAMMAS):
            with Clock(meter) as c:
                P_old, _, U = bundle(tb, buf, info, gamma, keep=True)
                padd = torch.cat([join_directions(buf, U, (means[:, s:s + sum(sizes_blk)].to(dev) - info["mu"][:, None]), sizes_blk)
                                  for s, sizes_blk in blocks(sizes, 400)], 1)
                del U
                zo_a, zn_a = stream_through(xo_h, mu_g, padd), stream_through(xn_h, mu_g, padd)
                zp_a = stream_through(xp_h, mu_g, padd)
                ko_a = km @ padd
            tick("join", c)
            with Clock(meter) as c:
                zo_g, zn_g, zp_g = zo_all[:, gi * r:(gi + 1) * r], zn_all[:, gi * r:(gi + 1) * r], zp_all[:, gi * r:(gi + 1) * r]
                ko_g = km @ vs[gi][0]
                # part 1 in the view: all 20 classes join, per draw
                for k, idx in draws.items():
                    E = idx.shape[0]
                    cs = [col["cross", k, e] for e in range(E)]
                    ext = lambda z, za: F.normalize(torch.cat([z[None].expand(E, -1, -1),
                                                               torch.stack([za[:, a:b] for a, b in cs])], 2), dim=2)
                    so, sn, sp = ext(zo_g, zo_a), ext(zn_g, zn_a), ext(zp_g, zp_a)
                    ka = ext(ko_g, ko_a)
                    nc = new_cards(sp, idx, ka, k)
                    old, new, no = compete(so, sn, ka, nc, oy, ny, cls_old, cls_new, hg)
                    write(task=task, kind="cross", way="view", gamma=gamma, shots=label(k), known=n_old,
                          old=[ints(o) for o in old], new=[ints(o) for o in new], new_only=ints(no))
                    if k == 3:
                        ih = G1.index(0.08)
                        print(f"seed {seed} {task} gamma {gamma} k=3 view  | h=0.08 old {old[:, ih].mean():.4f} "
                              f"new {new[:, ih].mean():.4f} | new-only {no.mean():.3f}", flush=True)
                    del so, sn, sp, ka, nc
                # the stream
                for pol in POLICIES:
                    rows = {s: dict(old=[], new=[], new_only=[], joined=[]) for s in range(1, SESSIONS + 1)}
                    for e in range(STREAM_RUNS):
                        for s in range(1, SESSIONS + 1):
                            have = arrived_n[e, s]
                            pos = list(have)
                            key = ("stream", e, s, pol)
                            if key in col:
                                a, b = col[key]
                                one = lambda z, za: F.normalize(torch.cat([z, za[:, a:b]], 1), dim=1)
                                nj = b - a
                            else:
                                one = lambda z, za: F.normalize(z, dim=1)
                                nj = 0
                            so, ka = one(zo_g, zo_a), one(ko_g, ko_a)
                            arr = cls_new[pos]
                            mask = torch.isin(ny, arr)
                            sn = one(zn_g, zn_a)[mask]
                            sp = one(zp_g, zp_a)
                            cards = torch.empty(len(pos), so.shape[1], device=dev)
                            for n in sorted(set(have.values())):
                                which = [i for i, j in enumerate(pos) if have[j] == n]
                                idx = torch.stack([s_pics[e][pos[i], :n] for i in which])[None]
                                cards[which] = new_cards(sp, idx, ka, n)[0]
                            old, new, no = compete(so[None], sn[None], ka[None], cards[None], oy, ny[mask], cls_old, arr, hg)
                            rows[s]["old"].append(ints(old[0]))
                            rows[s]["new"].append(ints(new[0]))
                            rows[s]["new_only"].append(round(no.item(), 4))
                            rows[s]["joined"].append(nj)
                    for s, rw in rows.items():
                        write(task=task, kind="stream", policy=pol, gamma=gamma, session=s, known=n_old, **rw)
            tick("scoring", c)
            del zo_a, zn_a, zp_a, ko_a, padd
        del buf
        tb.a = a_host.to(dev)
        del zo_all, zn_all, zp_all, vs, wcat, xo_h, xn_h, xp_h, means
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    tb = None
    for s, group in enumerate(groups[:STOP]):
        stage = s + 1
        gx, gy = sel(xtr, ytr, group)
        if tb is None:
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
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 20, 0) if dev.type == "cuda" else None
    peak_res = round(torch.cuda.max_memory_reserved() / 2 ** 20, 0) if dev.type == "cuda" else None
    write(task="cost", seconds={k: round(v, 2) for k, v in times.items()},
          wh={k: (None if v is None else round(v, 5)) for k, v in wh.items()} if wh else None, feature_dim=d,
          peak_mem_mb=peak, peak_reserved_mb=peak_res, cpu_peak_mb=cpu_peak_mb())
    print(f"seed {seed} done: dim {d}, peak GPU {peak} MB (reserved {peak_res}), times {times}", flush=True)


def blocks(sizes, cap):
    """Split consecutive groups into blocks of at most `cap` columns (never splitting a group): (start column, sizes)."""
    out, start, cur, n = [], 0, [], 0
    for m in sizes:
        if cur and n + m > cap:
            out.append((start, cur))
            start, cur, n = start + n, [], 0
        cur.append(m)
        n += m
    if cur:
        out.append((start, cur))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "none"
    print("device", dev, gpu, "torch", torch.__version__, "smoke", SMOKE, flush=True)
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
                per_stage=PER_STAGE, eyes=EYES, n_patches=N_PATCHES, thr=THR, shots=SHOTS, big=BIG, episodes=EPISODES,
                table_draws={str(k): v for k, v in TABLE_DRAWS.items()}, stream_runs=STREAM_RUNS, sessions=SESSIONS,
                per_session=PER_SESSION, levels=LEVELS, policies=POLICIES, gammas=GAMMAS, alpha=ALPHA,
                n_borrow=N_BORROW, g1=G1, surprise_seed=SURPRISE_SEED, power_reported=meter.ok,
                total_minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
