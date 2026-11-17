"""EXP-10 new classes among old ones: can few-picture classes hold their place among many-picture classes?

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

System exactly as EXP-9: patch eyes P (1,600 fixed patches, no backprop) and frozen backprop eyes F,
running tables (class sums, one shared table), learned view (P: smoothing 0.1, F: 1; all directions),
borrowing from the 2 nearest known cards with weight 1 picture. Corrections when old and new compete:
  C0  none: angle in the view (EXP-9)
  C1  C0 minus a handicap on every old class's score (grid G1)
  C2  distance in the view, whitened by the view's own within-class table, divided by (1 + 1/n)
  C3  C2 minus a handicap on old classes (grid G3, in units of the number of directions)
  C4  new classes added to the tables, view recomputed, then C1
Part 1: 20 surprise classes (1/3/10 pictures, 20 draws) join the 80 known; rehearsal after stage 6 with
        stages 7-8 as pretend-new (named among 80). Reference: new classes from all their pictures.
Part 2: stream: 60 classes (stages 1-6), then 8 sessions of 5 new classes x 5 pictures (stages 7-8 +
        surprise, random order); its rehearsal after stage 4 with stages 5-6 as pretend-new (among 60).
Every handicap value is logged; choosing is done afterwards by src/figures.py from the rehearsal rows.

Run:  python src/exp10.py                full run (5 seeds)
      EXP10_SMOKE=1 python src/exp10.py  tiny run to check the code
"""
import json
import os
import pathlib
import subprocess
import threading
import time

import numpy as np
import torch
import math

import torch.nn as nn
import torch.nn.functional as F

ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or (os.name == "posix" and pathlib.Path("/kaggle/working").exists())
try:
    ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    ROOT = pathlib.Path.cwd()
OUT = pathlib.Path(os.environ.get("EXP10_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP10_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP10_SMOKE") == "1"

SEEDS = 1 if SMOKE else 5
STAGES, PER_STAGE = 8, 10
K_BIG = int(os.environ.get("EXP10_KBIG", 120 if SMOKE else 1600))
N_PATCHES = 20_000 if SMOKE else 200_000
PATCH = 6
NORM_EPS = 10.0
ZCA_EPS = 0.1
SURPRISE_SEED = 1234
PASSES = 1 if SMOKE else 30
BATCH, LR = 128, 1e-3
START, ADD = 32, 16
FINAL_WIDTH = START + ADD * (STAGES - 1)
SHOTS = (1, 3, 10)
EPISODES = 20
BS = 50

VIEW_GAMMA = {"P": 0.1, "F": 1.0}       # EXP-9's chosen smoothing for the learned view
ALPHA = 1.0                              # EXP-9's chosen borrowing weight
N_BORROW = 2
G1 = [round(x, 3) for x in np.arange(-0.2, 0.5001, 0.01)]     # handicap on cosine scores (C1, C4)
G3 = [round(x, 3) for x in np.arange(-0.5, 1.0001, 0.02)]     # handicap on C2 scores, times the number of directions
STREAM_SESSIONS, STREAM_WAYS, STREAM_SHOTS = 8, 5, 5
CORR = ["C0", "C1", "C2", "C3", "C4"]

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F64 = torch.float64


# ----------------------------------------------------------------------------- power and timing (as EXP-9)
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


# ----------------------------------------------------------------------------- data (as EXP-9)
def load():
    import torchvision.datasets as tv
    DATA.mkdir(parents=True, exist_ok=True)
    tr, te = tv.CIFAR100(DATA, train=True, download=True), tv.CIFAR100(DATA, train=False, download=True)
    f = lambda d: (torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(d.targets))
    (xtr, ytr), (xte, yte) = f(tr), f(te)
    if SMOKE:
        keep = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0))[: len(xtr) // 10]
        xtr, ytr = xtr[keep], ytr[keep]
    return xtr.to(dev), ytr.long().to(dev), xte.to(dev), yte.long().to(dev)


# ----------------------------------------------------------------------------- patch eyes (as EXP-9)
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
    def __init__(self, mu, w, thr, c):
        self.mu, self.w, self.thr, self.c = mu, w, thr, c
        pos = 32 - PATCH + 1
        half = (pos + 1) // 2
        r = (torch.arange(pos, device=dev) >= half).long()
        self.q = F.one_hot((r[:, None] * 2 + r[None, :]).reshape(-1), 4).float()

    @torch.no_grad()
    def describe(self, x_u8):
        out = []
        cn = F.normalize(self.c, dim=1)
        for i in range(0, len(x_u8), BS):
            p = (norm_patches(x_u8[i:i + BS]) - self.mu) @ self.w
            f = F.relu(F.normalize(p, dim=2) @ cn.T - self.thr)
            out.append((f.transpose(1, 2) @ self.q).reshape(len(p), -1))
        return torch.cat(out)


# ----------------------------------------------------------------------------- backprop eyes (as EXP-9)
def augment(x):
    n = len(x)
    scale = torch.empty(n, device=dev).uniform_(0.7, 1.0)
    ang = torch.zeros(n, device=dev)
    tx, ty = torch.empty(2, n, device=dev).uniform_(-0.15, 0.15)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.stack([
        torch.stack([scale * torch.cos(ang) * flip, -scale * torch.sin(ang), tx], 1),
        torch.stack([scale * torch.sin(ang) * flip, scale * torch.cos(ang), ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * c + m) * b
    grey = torch.rand(n, 1, 1, 1, device=dev) < 0.2
    x = torch.where(grey, x.mean(1, keepdim=True).expand_as(x), x)
    return x.clamp(0, 1)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cin = cin
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x[:, :self.cin]))


class GrowNet(nn.Module):
    def __init__(self, in_ch, depth=4):
        super().__init__()
        self.in_ch, self.width = in_ch, 0
        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(depth)])

    def grow(self, n):
        for p in self.parameters():
            p.requires_grad_(False)
        for i, blocks in enumerate(self.layers):
            blocks.append(Block(self.in_ch if i == 0 else self.width + n, n))
        self.width += n
        return self.to(dev)

    def train(self, mode=True):
        super().train(mode)
        for blocks in self.layers:
            for b in blocks[:-1]:
                b.eval()
        return self

    def forward(self, x):
        x = (x - 0.5) / 0.5
        for blocks in self.layers:
            x = F.max_pool2d(F.relu(torch.cat([b(x) for b in blocks], 1)), 2)
        return x.flatten(1)


@torch.no_grad()
def features(net, x_u8, bs=500):
    net.eval()
    return torch.cat([net(x_u8[i:i + bs].float() / 255) for i in range(0, len(x_u8), bs)])


def train_t0(net, x_u8, y, classes, passes, meter):
    lut = torch.full((int(max(classes)) + 1,), -1, device=dev)
    lut[torch.tensor(classes, device=dev)] = torch.arange(len(classes), device=dev)
    y = lut[y]
    net.train()
    with torch.no_grad():
        dim = net(x_u8[:2].float() / 255).shape[1]
    net.train()
    head = nn.Linear(dim, len(classes)).to(dev)
    params = [p for p in net.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, passes * math.ceil(len(x_u8) / BATCH))
    sync()
    t0 = time.time()
    for _ in range(passes):
        perm = torch.randperm(len(x_u8), device=dev)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(head(net(augment(x_u8[idx].float() / 255))), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    sync()
    t1 = time.time()
    return dict(seconds=t1 - t0, wh=meter.wh(t0, t1))


# ----------------------------------------------------------------------------- the memory: running tables only
class Tables:
    def __init__(self, d):
        self.a = torch.zeros(d, d, device=dev, dtype=F64)
        self.sums, self.counts, self.cls = [], [], []

    def add_stage(self, g, y, classes):
        for i in range(0, len(g), 5000):
            gg = g[i:i + 5000].double()
            self.a += gg.T @ gg
        for c in classes:
            m = y == c
            self.cls.append(c)
            self.sums.append(g[m].double().sum(0))
            self.counts.append(int(m.sum()))

    def state(self):
        s, n = torch.stack(self.sums), torch.tensor(self.counts, device=dev, dtype=F64)
        m = s / n[:, None]
        big_n = n.sum()
        sw = (self.a - (m * n[:, None]).T @ m) / (big_n - len(n))
        return dict(m=m, n=n, mu=s.sum(0) / big_n, sw=sw, cls=list(self.cls))


def add_classes(st, groups_g, classes):
    """The state the tables would have after adding new classes from their pictures (list of [k_j, d])."""
    n_old, c_old = st["n"].sum(), len(st["cls"])
    scatter = st["sw"] * (n_old - c_old)
    means, counts, tot = [], [], st["mu"] * n_old
    for g in groups_g:
        g = g.double()
        mj = g.mean(0)
        dg = g - mj
        scatter = scatter + dg.T @ dg
        means.append(mj)
        counts.append(len(g))
        tot = tot + g.sum(0)
    n_new = n_old + sum(counts)
    c_new = c_old + len(classes)
    return dict(m=torch.cat([st["m"], torch.stack(means)]), n=torch.cat([st["n"], torch.tensor(counts, device=dev, dtype=F64)]),
                mu=tot / n_new, sw=scatter / (n_new - c_new), cls=st["cls"] + list(classes))


def bundle(st, gv):
    """The learned view of a state, for angle (P) and for distance whitened by the view's own table (Q)."""
    sw = st["sw"]
    a = sw.clone()
    a.diagonal().add_(gv * sw.diagonal().mean())
    L = torch.linalg.cholesky(a)
    del a
    mc = (st["m"] - st["mu"]).T
    u, _, _ = torch.linalg.svd(torch.linalg.solve_triangular(L, mc, upper=False), full_matrices=False)
    r = len(st["cls"]) - 1
    P = torch.linalg.solve_triangular(L.T, u[:, :r], upper=True)
    del L
    sv = P.T @ sw @ P
    sv.diagonal().add_(1e-6 * sv.diagonal().mean())
    lv = torch.linalg.cholesky(sv)
    Q = torch.linalg.solve_triangular(lv, P.T, upper=False).T          # (g - mu) @ Q has unit within-class spread
    mu = st["mu"].float()
    P, Q = P.float(), Q.float()
    km = (st["m"].float() - mu)
    return dict(P=P, Q=Q, mu=mu, ka=F.normalize(km @ P, dim=1), kd=km @ Q, n=st["n"].float(), r=r)


# ----------------------------------------------------------------------------- scoring when old and new compete
def new_cards(zp, idx, compare, known, k):
    """Cards of new classes [E, C, r] from k pictures each, borrowing from the 2 nearest known cards."""
    c = zp[idx].mean(2)
    if compare == "angle":
        near = torch.einsum("ecr,jr->ecj", F.normalize(c, dim=2), known)
    else:
        near = 2 * torch.einsum("ecr,jr->ecj", c, known) - known.pow(2).sum(1)
    nb = near.topk(N_BORROW, dim=2).indices
    c = (k * c + ALPHA * known[nb].mean(2)) / (k + ALPHA)
    return F.normalize(c, dim=2) if compare == "angle" else c


def raw_scores(b, compare, g_sets, gp, idx, k, n_old):
    """Scores of each test set against old cards then new cards: {name: [N, E, C_old + C_new]}."""
    if compare == "angle":
        f = lambda g: F.normalize((g - b["mu"]) @ b["P"], dim=1)
        nc = new_cards(f(gp), idx, "angle", b["ka"], k)
        cards = torch.cat([b["ka"][None].expand(len(nc), -1, -1), nc], 1)
        return {n: torch.einsum("nr,ecr->nec", f(g), cards) for n, g in g_sets.items()}
    f = lambda g: (g - b["mu"]) @ b["Q"]
    nc = new_cards(f(gp), idx, "dist", b["kd"], k)
    cards = torch.cat([b["kd"][None].expand(len(nc), -1, -1), nc], 1)
    infl = torch.cat([1 + 1 / b["n"][:n_old], torch.full((nc.shape[1],), 1 + 1 / (k + ALPHA), device=dev)])
    out = {}
    for n, g in g_sets.items():
        z = f(g)
        d2 = z.pow(2).sum(1)[:, None, None] - 2 * torch.einsum("nr,ecr->nec", z, cards) + cards.pow(2).sum(2)[None]
        out[n] = -d2 / infl
    return out


def sweep(s, y, cls, n_old, grid):
    """Accuracy of each test picture's answer as the handicap on old classes varies. s [N, E, C]."""
    mask = torch.zeros(s.shape[2], device=dev)
    mask[:n_old] = 1
    return [round((cls[(s - g * mask).argmax(2)] == y[:, None]).float().mean().item(), 4) for g in grid]


def compete(st, b, gv, n_old, old_te, old_y, new_te, new_y, gp, draws, new_cls, k, corrections=CORR):
    """Old and new classes named together. draws: [E, C_new, k] indices into gp.
    Returns {corr: dict(old=[...], new=[...], new_only=x)} with one entry per handicap value."""
    cls = torch.tensor(st["cls"][:n_old] + list(new_cls), device=dev)
    sets = {"old": old_te, "new": new_te}
    ys = {"old": old_y, "new": new_y}
    out = {}
    for comp, names in (("angle", ("C0", "C1")), ("dist", ("C2", "C3"))):
        if not any(c in corrections for c in names):
            continue
        sc = raw_scores(b, comp, sets, gp, draws, k, n_old)
        new_only = round((cls[n_old:][sc["new"][:, :, n_old:].argmax(2)] == new_y[:, None]).float().mean().item(), 4)
        if names[0] in corrections:
            out[names[0]] = dict(old=sweep(sc["old"], old_y, cls, n_old, [0.0])[0],
                                 new=sweep(sc["new"], new_y, cls, n_old, [0.0])[0], new_only=new_only)
        if names[1] in corrections:
            grid = G1 if comp == "angle" else [g * b["r"] for g in G3]
            out[names[1]] = dict(old=sweep(sc["old"], old_y, cls, n_old, grid),
                                 new=sweep(sc["new"], new_y, cls, n_old, grid), new_only=new_only)
        del sc
    if "C4" in corrections:     # new classes join the tables; the view is recomputed for every draw
        olds, news, nos = [], [], []
        for e in range(draws.shape[0]):
            st2 = add_classes(st, [gp[draws[e, j]] for j in range(draws.shape[1])], new_cls)
            b2 = bundle(st2, gv)
            b2["ka"] = b2["ka"][:n_old]           # borrow from the old cards only, as C0
            sc = raw_scores(b2, "angle", sets, gp, draws[e:e + 1], k, n_old)
            olds.append(sweep(sc["old"], old_y, cls, n_old, G1))
            news.append(sweep(sc["new"], new_y, cls, n_old, G1))
            nos.append((cls[n_old:][sc["new"][:, :, n_old:].argmax(2)] == new_y[:, None]).float().mean().item())
            del st2, b2, sc
        out["C4"] = dict(old=[round(float(x), 4) for x in np.mean(olds, 0)], new=[round(float(x), 4) for x in np.mean(news, 0)],
                         new_only=round(float(np.mean(nos)), 4))
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
    stream_new = np.random.default_rng(30_000 + seed).permutation(groups[6] + groups[7] + surprise).tolist()

    def sel(x, y, classes):
        m = torch.isin(y, torch.tensor(classes, device=dev))
        return x[m], y[m]

    def write(**kw):
        log.write(json.dumps(dict(seed=seed, **kw)) + "\n")
        log.flush()

    def rw(x):
        return None if x is None else round(x, 5)

    def peak():
        return round(torch.cuda.max_memory_allocated() / 2 ** 20, 0) if dev.type == "cuda" else None

    def reset_peak():
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    gx0, gy0 = sel(xtr, ytr, groups[0])

    def one_eyes(name, describe, build_cost):
        g = lambda x: describe(x).sqrt()
        gv = VIEW_GAMMA[name]
        times = dict(describe=0.0, tables=0.0, part1=0.0, rehearsal=0.0, stream=0.0, stream_rehearsal=0.0)
        wh = {k: 0.0 for k in times} if meter.ok else None

        def tick(key, c):
            times[key] += c.seconds
            if wh is not None and c.wh is not None:
                wh[key] += c.wh

        def task_compete(task, st, n_known, new_cls, shots, extra=None):
            """Describe the needed pictures and run every correction for each number of pictures."""
            with Clock(meter) as c:
                ox, oy = sel(xte, yte, st["cls"])
                px, py = sel(xtr, ytr, new_cls)
                nx, ny = sel(xte, yte, new_cls)
                g_old, g_pool, g_new = g(ox), g(px), g(nx)
            tick("describe", c)
            draws = make_draws(py, new_cls, np.random.default_rng(
                {"part1": 10_000, "rehearsal": 20_000, "stream_rehearsal": 40_000}[task] + seed), shots)
            with Clock(meter) as c:
                b = bundle(st, gv)
                for k, idx in draws.items():
                    res = compete(st, b, gv, n_known, g_old, oy, g_new, ny, g_pool, idx, new_cls, k)
                    for corr, r in res.items():
                        write(eyes=name, task=task, shots=k, corr=corr, known=n_known, **r)
                    print(f"seed {seed} {name} {task} k={k} | " + " ".join(
                        f"{c_}:{r['old'] if isinstance(r['old'], float) else r['old'][G1.index(0.0) if c_ != 'C3' else G3.index(0.0)]:.3f}/"
                        f"{r['new'] if isinstance(r['new'], float) else r['new'][G1.index(0.0) if c_ != 'C3' else G3.index(0.0)]:.3f}"
                        for c_, r in res.items()), flush=True)
                if extra:
                    extra(st, g_old, oy, g_pool, py, g_new, ny)
            tick(task, c)
            del g_old, g_pool, g_new, b

        def reference_all(st, g_old, oy, g_pool, py, g_new, ny):
            """New classes learned from all their pictures (not a method): view of all 100, angle, no handicap."""
            st2 = add_classes(st, [g_pool[py == c] for c in surprise], surprise)
            b2 = bundle(st2, gv)
            cls = torch.tensor(st2["cls"], device=dev)
            f = lambda x: F.normalize((x - b2["mu"]) @ b2["P"], dim=1)
            acc = lambda x, y: round((cls[(f(x) @ b2["ka"].T).argmax(1)] == y).float().mean().item(), 4)
            write(eyes=name, task="part1_ref_all", corr="ref_all", known=len(st["cls"]), old=acc(g_old, oy), new=acc(g_new, ny))
            del st2, b2

        def stream(st):
            """60 classes, then 8 sessions of 5 new classes with 5 pictures each."""
            with Clock(meter) as c:
                base_cls = st["cls"]
                ox, oy = sel(xte, yte, base_cls)
                g_old = g(ox)
                srng = np.random.default_rng(50_000 + seed)
                shots_g, new_te, new_y = [], [], []
                for cl in stream_new:
                    ids = torch.nonzero(ytr == cl).squeeze(1).cpu().numpy()
                    shots_g.append(g(xtr[torch.tensor(srng.choice(ids, STREAM_SHOTS, replace=False), device=dev)]))
                nx, ny = sel(xte, yte, stream_new)
                g_newall = g(nx)
            tick("describe", c)
            with Clock(meter) as c:
                b = bundle(st, gv)
                n0 = len(base_cls)
                st_c4 = st
                for sess in range(STREAM_SESSIONS + 1):
                    seen = stream_new[:sess * STREAM_WAYS]
                    if sess == 0:
                        cls = torch.tensor(base_cls, device=dev)
                        za = F.normalize((g_old - b["mu"]) @ b["P"], dim=1)
                        write(eyes=name, task="stream", session=0, corr="C0", n_classes=n0,
                              old=round((cls[(za @ b["ka"].T).argmax(1)] == oy).float().mean().item(), 4))
                        continue
                    m = torch.isin(ny, torch.tensor(seen, device=dev))
                    gp = torch.cat(shots_g[:len(seen)])
                    idx = torch.arange(len(gp), device=dev).reshape(1, len(seen), STREAM_SHOTS)
                    res = compete(st, b, gv, n0, g_old, oy, g_newall[m], ny[m], gp, idx, seen, STREAM_SHOTS,
                                  corrections=["C0", "C1", "C2", "C3"])
                    # C4 in a stream: the tables keep the new classes from session to session
                    sess_cls = stream_new[(sess - 1) * STREAM_WAYS:sess * STREAM_WAYS]
                    st_c4 = add_classes(st_c4, shots_g[(sess - 1) * STREAM_WAYS:sess * STREAM_WAYS], sess_cls)
                    b4 = bundle(st_c4, gv)
                    b4["ka"] = b4["ka"][:n0]
                    sc = raw_scores(b4, "angle", {"old": g_old, "new": g_newall[m]}, gp, idx, STREAM_SHOTS, n0)
                    clsall = torch.tensor(base_cls + seen, device=dev)
                    res["C4"] = dict(old=sweep(sc["old"], oy, clsall, n0, G1), new=sweep(sc["new"], ny[m], clsall, n0, G1))
                    del b4, sc
                    for corr, r in res.items():
                        write(eyes=name, task="stream", session=sess, corr=corr, n_classes=n0 + len(seen),
                              n_old_test=len(oy), n_new_test=int(m.sum()), **r)
                    r0 = res["C0"]
                    print(f"seed {seed} {name} stream session {sess} C0 old {r0['old']:.3f} new {r0['new']:.3f}", flush=True)
                del st_c4, b
            tick("stream", c)
            del g_old, g_newall, shots_g

        reset_peak()
        tb = None
        for s, group in enumerate(groups):
            stage = s + 1
            with Clock(meter) as c:
                gx, gy = sel(xtr, ytr, group)
                g_stage = g(gx)
            tick("describe", c)
            with Clock(meter) as c:
                if tb is None:
                    tb = Tables(g_stage.shape[1])
                tb.add_stage(g_stage, gy, group)
            tick("tables", c)
            del g_stage
            if stage == 4:
                st = tb.state()
                task_compete("stream_rehearsal", st, 40, groups[4] + groups[5], (STREAM_SHOTS,))
                del st
            if stage == 6:
                st = tb.state()
                task_compete("rehearsal", st, 60, groups[6] + groups[7], SHOTS)
                stream(st)
                del st
            if stage == 8:
                st = tb.state()
                task_compete("part1", st, 80, surprise, SHOTS, extra=reference_all)
                del st
        write(eyes=name, task="cost", seconds={k: round(v, 2) for k, v in times.items()},
              wh={k: rw(v) for k, v in wh.items()} if wh else None, feature_dim=tb.a.shape[0], peak_mem_mb=peak(), **build_cost)
        del tb

    # ---- P: patch eyes, exactly as EXP-8 / EXP-9
    reset_peak()
    gen = torch.Generator().manual_seed(seed * 100 + 2)
    torch.manual_seed(seed * 100 + 2)
    with Clock(meter) as c_b:
        p = sample_patches(gx0, N_PATCHES, gen)
        mu, w = fit_whitening(p)
        eyes = Eyes(mu, w, 0.1, ((p - mu) @ w)[torch.randperm(len(p), generator=gen)[:K_BIG].to(dev)])
        del p
    one_eyes("P", eyes.describe, dict(build_seconds=round(c_b.seconds, 2), build_wh=rw(c_b.wh)))
    del eyes

    # ---- F: frozen backprop eyes, exactly as EXP-8 / EXP-9
    reset_peak()
    torch.manual_seed(seed * 1000 + 800)
    net = GrowNet(3).grow(FINAL_WIDTH)
    c = train_t0(net, gx0, gy0, groups[0], PASSES, meter)
    one_eyes("F", lambda x: features(net, x), dict(train_seconds=round(c["seconds"], 2), train_wh=rw(c["wh"])))
    del net


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "none"
    print("device", dev, gpu, "torch", torch.__version__, "smoke", SMOKE, flush=True)
    if dev.type != "cuda" and not SMOKE:
        raise SystemExit("No GPU: refusing to start the full run on CPU.")
    if dev.type == "cuda":
        nn.Conv2d(3, 8, 3).to(dev)(torch.zeros(2, 3, 8, 8, device=dev)).sum().item()
        torch.backends.cudnn.benchmark = True
    meter = PowerMeter()
    t0 = time.time()
    data = load()
    print(f"data ready, {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(OUT / "runs.jsonl", "w") as log:
        for seed in range(SEEDS):
            run(seed, data, meter, log)
            print(f"== seed {seed} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    meter.stop()
    meta = dict(gpu=gpu, torch=torch.__version__, on_kaggle=ON_KAGGLE, smoke=SMOKE, seeds=SEEDS, stages=STAGES,
                per_stage=PER_STAGE, passes=PASSES, k_big=K_BIG, n_patches=N_PATCHES, bs=BS, shots=SHOTS,
                episodes=EPISODES, view_gamma=VIEW_GAMMA, alpha=ALPHA, n_borrow=N_BORROW, g1=G1, g3=G3,
                stream=dict(sessions=STREAM_SESSIONS, ways=STREAM_WAYS, shots=STREAM_SHOTS),
                surprise_seed=SURPRISE_SEED, power_reported=meter.ok, total_minutes=round((time.time() - t0) / 60, 1),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
