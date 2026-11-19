"""EXP-15 gradual joining: does letting a new class's question (its own direction in the learned view) count for little
at first and more as its pictures arrive, with its own handicap, beat EXP-14's all-or-nothing joining? And does holding
apart and joining pay off over a longer stream? No backpropagation anywhere.

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

Eyes and memory: EXP-14's (A2 eyes, compact tables, learned view of the old classes, new cards by angle with borrowing).
Every new class gets one extra direction, as EXP-14's "view" (its card through the old table, minus what the old
directions express; classes in arrival order); each way only changes the WEIGHT of those directions:
  apart                 0                                  (EXP-13)
  full                  1                                  (EXP-14's view)
  counted k             n / (n + k)
  measured b            max(0, 1 - b * s2 / (n * f2))      f2: squared novelty of the class's average description;
                                                            s2: expected squared novelty of one description that is pure
                                                            chance, from the shared table
  card b                pictures 1, the card's new part times the measured weight
  threshold N (stream)  0 below N pictures, 1 from N       (EXP-14's rules)
Part 1, the crossover: 1, 3, 10, 30, 100 pictures per new class and all of them, every way, every handicap h1.
Part 2, the stream: 8 sessions; 5 classes arrive in each of sessions 1-4; a class has 3, 10, 30, 60, 100, 200, 300, 500
pictures in the session it arrives and the next ones. Every h1 (old classes) and h2 (new classes, times their weight).
The choosing is done afterwards by src/figures.py from the rehearsal rows.

Outputs: runs.jsonl (small rows), cross_<task>_s<seed>.npz and stream_<task>_s<seed>.npz (accuracy tables, x 10^4).

Run:  python src/exp15.py                full run (5 seeds)
      EXP15_SMOKE=1 python src/exp15.py  tiny run to check the code
      EXP15_SEEDS=1 EXP15_STOP=6 python src/exp15.py   one seed, up to the stage-6 rehearsal
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
OUT = pathlib.Path(os.environ.get("EXP15_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP15_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP15_SMOKE") == "1"

SEEDS = int(os.environ.get("EXP15_SEEDS", 1 if SMOKE else 5))
STOP = int(os.environ.get("EXP15_STOP", 8))           # last stage to run (8 = everything)
STAGES, PER_STAGE = 8, 10
N_PATCHES = 20_000 if SMOKE else 200_000
PATCH = 6
NORM_EPS = 10.0
ZCA_EPS = 0.1
THR = 0.1
SURPRISE_SEED = 1234
SHOTS = (1, 3, 10)                                    # drawn exactly as EXP-11 to EXP-14 (same random generator order)
BIG = {30: 20, 100: 10}
EPISODES = 20
STREAM_RUNS = 10
ARRIVE_SESSIONS, PER_SESSION = 4, 5                   # 5 classes arrive in each of sessions 1-4
LEVELS = (3, 10, 30, 60, 100, 200, 300, 500)          # pictures of a class in the session it arrives and the next ones
SESSIONS = len(LEVELS)
CHUNK = 1000
PROBES = 128                                          # random probes for the trace of the smoothed table's inverse
NOISE_PICS = 30                                       # pictures per class for the noise check

EYES = dict(k=1600, sides=2, grid=2, mirror=True)     # EXP-11's A2
K_MAX = 3200
SMOKE_DIV = 10

GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0]
ALPHA = 1.0
N_BORROW = 2
G1 = [round(x, 3) for x in np.arange(-0.2, 0.5001, 0.01)]     # h1, on the old classes' cosine scores
G2 = [round(x, 3) for x in np.arange(-0.2, 0.2001, 0.02)]     # h2, on new classes' scores, times their weight
KS = (3, 10, 30, 100)
BETAS = (0.25, 0.5, 1.0, 2.0)
THRESH = {"once": 3, "at10": 10, "at30": 30, "at100": 100}
CROSS_WAYS = (["apart", "full"] + [f"counted{k}" for k in KS] + [f"measured{b}" for b in BETAS]
              + [f"card{b}" for b in BETAS])
STREAM_RULES = (["never"] + list(THRESH) + [f"counted{k}" for k in KS] + [f"measured{b}" for b in BETAS]
                + [f"card{b}" for b in BETAS])

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
    """The learned view P at one smoothing (as EXP-11 to EXP-14). keep=True also returns the whitened directions U, the
    chance spread s2 (below), and leaves the smoothed table's Cholesky factor L in buf."""
    tb.fill(buf, info)
    md = buf.diagonal().mean()
    buf.diagonal().add_(gamma * md)
    chol_inplace(buf)
    mc = (info["m"] - info["mu"]).T
    u, _, _ = torch.linalg.svd(torch.linalg.solve_triangular(buf, mc, upper=False), full_matrices=False)
    r = len(info["cls"]) - 1
    U = u[:, :r].contiguous()
    P = torch.linalg.solve_triangular(buf.mT, U, upper=True)
    if not keep:
        return P.float(), r
    s2 = chance_spread(buf, P, r, gamma * md)
    return P.float(), r, U, s2


def chance_spread(L, P, r, gm):
    """Expected squared novelty of one table description that is pure within-class chance: the trace of the whitened
    within-class table over the directions the old view does not use. With T = W + gm I = L L^T:
    tr(P_perp L^-1 W L^-T P_perp) = (d - r) - gm (tr(T^-1) - ||P||^2); tr(T^-1) by random +-1 probes."""
    d = L.shape[0]
    g = torch.Generator(device=L.device).manual_seed(7)
    v = (torch.randint(0, 2, (d, PROBES), generator=g, device=L.device).to(F64) * 2 - 1)
    tr_inv = float((torch.linalg.solve_triangular(L, v, upper=False) ** 2).sum() / PROBES)
    return float((d - r) - float(gm) * (tr_inv - float((P ** 2).sum())))


def join_directions(L, U, mc, sizes):
    """Extra view directions for groups of joining classes, with the old table (as EXP-14), and each class's squared
    novelty f2 (the size of the part of its whitened card that the old directions U cannot express).
    mc: (d, sum(sizes)) cards minus the old centre, float64. Returns (d, sum(sizes)) float32 and f2 (sum(sizes),)."""
    b = torch.linalg.solve_triangular(L, mc, upper=False)
    for _ in range(2):                                  # twice, for numerical safety
        b -= U @ (U.T @ b)
    f2 = (b ** 2).sum(0)
    q, s = [], 0
    for m in sizes:
        q.append(torch.linalg.qr(b[:, s:s + m]).Q)      # orthonormal directions for this group, in its column order
        s += m
    q = torch.cat(q, 1)
    del b
    return torch.linalg.solve_triangular(L.mT, q, upper=True).float(), f2


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


# ----------------------------------------------------------------------------- weights
def weights(way, n, f2, s2):
    """(picture weight, card weight) of each new class's direction. n, f2: tensors (..., C)."""
    one = torch.ones_like(f2)
    if way in ("apart", "never"):
        return 0 * one, 0 * one
    if way == "full":
        return one, one
    if way in THRESH:
        w = (n >= THRESH[way]).to(f2.dtype)
        return w, w
    if way.startswith("counted"):
        k = float(way[len("counted"):])
        w = n / (n + k)
        return w, w
    b = float(way[len("measured"):] if way.startswith("measured") else way[len("card"):])
    w = (1 - b * s2 / (n * f2.clamp_min(1e-12))).clamp(0, 1)
    return (w, w) if way.startswith("measured") else (one, w)


# ----------------------------------------------------------------------------- scoring (as EXP-11's C0 and C1)
def new_cards(zp, idx, known, k):
    """New cards by angle with borrowing from the 2 nearest known cards, weight ALPHA pictures (as EXP-9 to EXP-14).
    zp: (N, D) or (E, N, D) normalized picture coordinates; idx: (E, C, k); known: (C_old, D) or (E, C_old, D).
    Returns unit cards (E, C, D)."""
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


def shrink_cards(nc, r, wc):
    """'card' ways: the card's new part (coordinates after the first r) times the card weight, then unit length again.
    nc: (E, C, D); wc: (E, C)."""
    return F.normalize(torch.cat([nc[..., :r], nc[..., r:] * wc[..., None]], -1), dim=-1)


def compete(so, sn, ka, nc, oy, ny, cls_old, cls_new, hg):
    """As EXP-14: old and new accuracy among all classes for every h1, per draw (E, len(hg)); new among themselves (E,)."""
    cos_oo = so @ ka.mT
    oc_old = (cls_old[cos_oo.argmax(2)] == oy).float()
    best_old_o = cos_oo.max(2).values
    del cos_oo
    best_old_n = (sn @ ka.mT).max(2).values
    m_old = (so @ nc.mT).max(2).values - best_old_o
    cos_nn = sn @ nc.mT
    m_new = cos_nn.max(2).values - best_old_n
    cn = (cls_new[cos_nn.argmax(2)] == ny).float()
    old = ((m_old[:, :, None] <= -hg).float() * oc_old[:, :, None]).mean(1)
    new = ((m_new[:, :, None] > -hg).float() * cn[:, :, None]).mean(1)
    return old, new, cn.mean(1)


def compete2(so, sn, ka, nc, oy, ny, cls_old, cls_new, hg, h2, wh):
    """One stream session with two handicaps: h1 (hg) off the old classes' scores, h2 * wh off each new class's score.
    so, sn: (N, D) unit test coordinates; ka: (C_old, D); nc: (C_new, D); wh: (C_new,). Returns old, new (len(h2), len(hg))
    and the new classes among themselves at h2 = 0."""
    cos_oo = so @ ka.T
    oc_old = (cls_old[cos_oo.argmax(1)] == oy).float()
    best_old_o = cos_oo.max(1).values
    del cos_oo
    best_old_n = (sn @ ka.T).max(1).values
    off = h2[:, None] * wh[None, :]                                            # (H2, C_new)
    s_on = (so @ nc.T)[None] - off[:, None, :]                                 # (H2, N_old, C_new)
    m_old = s_on.max(2).values - best_old_o[None]
    del s_on
    s_nn = (sn @ nc.T)[None] - off[:, None, :]
    m_new = s_nn.max(2).values - best_old_n[None]
    cn = (cls_new[s_nn.argmax(2)] == ny[None]).float()                         # (H2, N_new)
    old = ((m_old[:, :, None] <= -hg).float() * oc_old[None, :, None]).mean(1)
    new = ((m_new[:, :, None] > -hg).float() * cn[:, :, None]).mean(1)
    i0 = int(torch.argmin(h2.abs()))
    return old, new, cn[i0].mean()


def u16(t):
    return (t * 10000).round().to(torch.int32).cpu().numpy().astype(np.uint16)


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
    times = dict(describe=0.0, tables=0.0, view=0.0, join=0.0, scoring=0.0)
    wh = {k: 0.0 for k in times} if meter.ok else None
    hg = torch.tensor(G1, device=dev, dtype=torch.float32)
    h2g = torch.tensor(G2, device=dev, dtype=torch.float32)

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
        draws = make_draws(py, new_cls, draw_rng, SHOTS)                         # identical to EXP-11 to EXP-14
        n_min = min(int((py == c).sum()) for c in new_cls)
        big = {k: e for k, e in BIG.items() if k < n_min}
        draws.update(make_draws(py, new_cls, draw_rng, tuple(big), big))
        draws[n_min] = torch.stack([torch.nonzero(py == c).squeeze(1)[:n_min] for c in new_cls])[None]   # all pictures
        label = lambda k: "all" if k == n_min else str(k)
        # the stream (the same generator calls as EXP-14, so sessions 1-3 are identical)
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
        with Clock(meter) as c:
            for gi, gamma in enumerate(GAMMAS):
                so = F.normalize(zo_all[:, gi * r:(gi + 1) * r], dim=1)
                ka = F.normalize(km @ vs[gi][0], dim=1)
                oc_old = (cls_old[(so @ ka.T).argmax(1)] == oy).float()
                write(task=task, kind="old_only", gamma=gamma, known=n_old, old=round(oc_old.mean().item(), 4))
        tick("scoring", c)

        # every group of joining classes: part 1 (all 20 classes, every draw) and the stream (every run and session:
        # all classes that have arrived, in arrival order, so that a threshold rule's joined classes come first)
        groups_j = []                                                    # (key, picture lists)
        for k, idx in draws.items():
            for e in range(idx.shape[0]):
                groups_j.append((("cross", k, e), [idx[e, j] for j in range(len(new_cls))]))
        arrived = {}                                                     # (run, session) -> (positions, pictures each)
        for e in range(STREAM_RUNS):
            for s in range(1, SESSIONS + 1):
                pos = [int(j) for j in s_order[e][:min(s, ARRIVE_SESSIONS) * PER_SESSION]]
                nn = [levels[s - 1 - i // PER_SESSION] for i in range(len(pos))]
                arrived[e, s] = (pos, nn)
                groups_j.append((("stream", e, s), [s_pics[e][j, :n] for j, n in zip(pos, nn)]))
        with Clock(meter) as c:
            means = torch.stack([xp_h[pics.cpu()].double().mean(0) for _, pl in groups_j for pics in pl], 1)
        tick("join", c)
        sizes = [len(pl) for _, pl in groups_j]
        starts = np.concatenate([[0], np.cumsum(sizes)])
        col = {g[0]: (int(starts[i]), int(starts[i + 1])) for i, g in enumerate(groups_j)}
        a_host = to_host(tb.a)
        tb.a = a_host                                                    # the shared table waits on the CPU meanwhile
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        cross = {}                                                       # npz arrays
        stream = {rule: np.zeros((len(GAMMAS), SESSIONS, 2, len(G2), len(G1)), np.uint16) for rule in STREAM_RULES}
        stream_no = {rule: np.zeros((len(GAMMAS), SESSIONS), np.uint16) for rule in STREAM_RULES}
        buf = torch.empty(d, d, device=dev, dtype=F64)
        for gi, gamma in enumerate(GAMMAS):
            with Clock(meter) as c:
                _, _, U, s2 = bundle(tb, buf, info, gamma, keep=True)
                parts = [join_directions(buf, U, (means[:, s:s + sum(sb)].to(dev) - info["mu"][:, None]), sb)
                         for s, sb in blocks(sizes, 400)]
                padd = torch.cat([p for p, _ in parts], 1)
                f2_all = torch.cat([f for _, f in parts]).float()
                del parts
                # noise check: the chance part measured on the new classes' own pictures (the first 30-picture draw)
                kn = NOISE_PICS if NOISE_PICS in draws else max(k for k in draws if k != n_min)
                ids = draws[kn][0]                                                       # (20, kn)
                g = xp_h[ids.reshape(-1).cpu()].to(dev).double() - info["mu"]
                bb = torch.linalg.solve_triangular(buf, g.T, upper=False)
                for _ in range(2):
                    bb -= U @ (U.T @ bb)
                bb = bb.T.reshape(len(new_cls), kn, -1)
                within = float(((bb - bb.mean(1, keepdim=True)) ** 2).sum() / (len(new_cls) * (kn - 1)))
                del g, bb, U
                zo_a, zn_a = stream_through(xo_h, mu_g, padd), stream_through(xn_h, mu_g, padd)
                zp_a = stream_through(xp_h, mu_g, padd)
                ko_a = km @ padd
            tick("join", c)
            write(task=task, kind="noise", gamma=gamma, s2=round(s2, 3), within=round(within, 3), pics=kn,
                  ratio=round(within / s2, 4))
            with Clock(meter) as c:
                zo_g, zn_g, zp_g = zo_all[:, gi * r:(gi + 1) * r], zn_all[:, gi * r:(gi + 1) * r], zp_all[:, gi * r:(gi + 1) * r]
                ko_g = km @ vs[gi][0]
                # ---------------- part 1: all 20 classes join, per draw
                for k, idx in draws.items():
                    E = idx.shape[0]
                    cs = [col["cross", k, e] for e in range(E)]
                    za = lambda z: torch.stack([z[:, a:b] for a, b in cs])                    # (E, N, 20)
                    f2 = torch.stack([f2_all[a:b] for a, b in cs])                             # (E, 20)
                    nk = torch.full_like(f2, float(k))
                    zo_e, zn_e, zp_e, ko_e = za(zo_a), za(zn_a), za(zp_a), za(ko_a)
                    wsum = {}
                    for way in CROSS_WAYS:
                        wp, wc = weights(way, nk, f2, s2)
                        ext = lambda z, ze: F.normalize(torch.cat([z[None].expand(E, -1, -1), ze * wp[:, None, :]], 2), dim=2)
                        so, sn, sp, ka = ext(zo_g, zo_e), ext(zn_g, zn_e), ext(zp_g, zp_e), ext(ko_g, ko_e)
                        nc = new_cards(sp, idx, ka, k)
                        if way.startswith("card"):
                            nc = shrink_cards(nc, r, wc)
                        old, new, no = compete(so, sn, ka, nc, oy, ny, cls_old, cls_new, hg)
                        key = f"{way}|{gamma}|{label(k)}"
                        cross[key + "|old"], cross[key + "|new"], cross[key + "|no"] = u16(old), u16(new), u16(no)
                        wsum[way] = round(float(wc.mean()), 4)
                        if k == 3 and way in ("apart", "full"):
                            ih = G1.index(0.08)
                            print(f"seed {seed} {task} gamma {gamma} k=3 {way:5s} | h=0.08 old {old[:, ih].mean():.4f} "
                                  f"new {new[:, ih].mean():.4f} | new-only {no.mean():.3f}", flush=True)
                        del so, sn, sp, ka, nc
                    write(task=task, kind="weights", gamma=gamma, shots=label(k), f2=round(float(f2.mean()), 3),
                          s2=round(s2, 3), mean_weight=wsum)
                    del zo_e, zn_e, zp_e, ko_e
                # ---------------- the stream
                acc = {rule: torch.zeros(SESSIONS, 2, len(G2), len(G1), device=dev) for rule in STREAM_RULES}
                acc_no = {rule: torch.zeros(SESSIONS, device=dev) for rule in STREAM_RULES}
                for e in range(STREAM_RUNS):
                    for s in range(1, SESSIONS + 1):
                        pos, nn = arrived[e, s]
                        a, b = col["stream", e, s]
                        f2 = f2_all[a:b]
                        nt = torch.tensor(nn, device=dev, dtype=f2.dtype)
                        arr = cls_new[pos]
                        mask = torch.isin(ny, arr)
                        zn_m, zna_m, nym = zn_g[mask], zn_a[mask, a:b], ny[mask]
                        for rule in STREAM_RULES:
                            wp, wc = weights(rule, nt, f2, s2)
                            one = lambda z, ze: F.normalize(torch.cat([z, ze * wp[None]], 1), dim=1)
                            so, ka = one(zo_g, zo_a[:, a:b]), one(ko_g, ko_a[:, a:b])
                            sn, sp = one(zn_m, zna_m), one(zp_g, zp_a[:, a:b])
                            cards = torch.empty(len(pos), so.shape[1], device=dev)
                            for n in sorted(set(nn)):
                                which = [i for i, m in enumerate(nn) if m == n]
                                idx = torch.stack([s_pics[e][pos[i], :n] for i in which])[None]
                                cards[which] = new_cards(sp, idx, ka, n)[0]
                            if rule.startswith("card"):
                                cards = shrink_cards(cards[None], r, wc[None])[0]
                            old, new, no = compete2(so, sn, ka, cards, oy, nym, cls_old, arr, hg, h2g, wc)
                            acc[rule][s - 1, 0] += old
                            acc[rule][s - 1, 1] += new
                            acc_no[rule][s - 1] += no
                for rule in STREAM_RULES:
                    stream[rule][gi] = u16(acc[rule] / STREAM_RUNS)
                    stream_no[rule][gi] = u16(acc_no[rule] / STREAM_RUNS)
                ih, i2 = G1.index(0.12), G2.index(0.0)
                for rule in ("never", "at30"):
                    v = stream[rule][gi][:, :, i2, ih] / 100
                    print(f"seed {seed} {task} gamma {gamma} stream {rule:6s} h1=0.12 H by session "
                          + " ".join(f"{2 * o * n / max(o + n, 1e-9):.1f}" for o, n in v), flush=True)
            tick("scoring", c)
            del zo_a, zn_a, zp_a, ko_a, padd, f2_all
        del buf
        tb.a = a_host.to(dev)
        np.savez_compressed(OUT / f"cross_{task}_s{seed}.npz", **cross)
        np.savez_compressed(OUT / f"stream_{task}_s{seed}.npz",
                            **{rule: stream[rule] for rule in STREAM_RULES},
                            **{rule + "|no": stream_no[rule] for rule in STREAM_RULES})
        write(task=task, kind="files", n_min=n_min, known=n_old, labels=[label(k) for k in draws],
              draws={label(k): int(v.shape[0]) for k, v in draws.items()})
        del zo_all, zn_all, zp_all, vs, wcat, xo_h, xn_h, xp_h, means, a_host
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
                stream_runs=STREAM_RUNS, sessions=SESSIONS, arrive_sessions=ARRIVE_SESSIONS, per_session=PER_SESSION,
                levels=LEVELS, gammas=GAMMAS, alpha=ALPHA, n_borrow=N_BORROW, g1=G1, g2=G2, ks=KS, betas=BETAS,
                thresh=THRESH, cross_ways=CROSS_WAYS, stream_rules=STREAM_RULES, probes=PROBES,
                surprise_seed=SURPRISE_SEED, power_reported=meter.ok,
                total_minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
