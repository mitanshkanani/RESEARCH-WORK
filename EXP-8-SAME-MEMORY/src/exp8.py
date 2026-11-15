"""EXP-8 same memory for every eyes: is it the eyes or the memory?

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

Four kinds of eyes on the 80-class protocol of EXP-3..7 (CIFAR-100, 8 stages of 10, 5 seeds):
  P  patch eyes: 1,600 fixed whitened patches from the first group, threshold 0.1 (no backprop)
  F  frozen backprop eyes: full-size network trained on the first group only
  G  grow and lock: units added and locked at each stage (as EXP-6 run A)
  R  retrain everything: fresh network on all classes; also scored with its own answer layer
Memories on each: class cards by angle (cos), the same on square-rooted features (sqrt_cos), class
means + one shared covariance (lda) and a closed-form ridge classifier, at five smoothing values.
For G the shared table has missing blocks for old classes: `lda_fill` fills them from current
pictures (a real method); `lda_open` / `ridge_open` use all old pictures (diagnosis).

Run:  python src/exp8.py              full run (5 seeds)
      EXP8_SMOKE=1 python src/exp8.py   tiny run to check the code
Writes runs.jsonl and meta.json into the output folder.
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
OUT = pathlib.Path(os.environ.get("EXP8_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP8_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP8_SMOKE") == "1"

SEEDS = 1 if SMOKE else 5
STAGES, PER_STAGE = (3 if SMOKE else 8), 10
K_BIG, K_SMALL = (120, 60) if SMOKE else (1600, 400)
N_PATCHES = 20_000 if SMOKE else 200_000
PATCH = 6
NORM_EPS = 10.0                      # added to a patch's variance (pixels on the 0..255 scale)
ZCA_EPS = 0.1                        # whitening constant
KMEANS_ITERS = 10
GAMMAS = [0.0001, 0.001, 0.01, 0.1, 1.0]            # smoothing of the shared tables, times the average variance
TABLE_STAGES = sorted({1, (STAGES + 1) // 2, STAGES})   # stages at which lda / ridge are solved
SURPRISE_SEED = 1234
SPLIT_SEED = 4321
RIDGE_R7 = 1e-2
FS_GAMMAS = [10.0, 100.0]             # extra smoothing values, used for the surprise test only
PASSES = 1 if SMOKE else 30
BATCH, LR = 128, 1e-3
START, ADD = 32, 16                  # grow and lock, as EXP-3..6
SHOTS = (1, 3, 10)
EPISODES = 20
BS = 100

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F64 = torch.float64


# ----------------------------------------------------------------------------- power (as EXP-3)
class PowerMeter:
    """Reads the GPU's power draw once a second through nvidia-smi, if the GPU reports it."""

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
    """Times a block and reads its GPU energy."""

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


# ----------------------------------------------------------------------------- data
def load(name):
    import torchvision.datasets as tv
    DATA.mkdir(parents=True, exist_ok=True)
    cls = tv.CIFAR100 if name == "cifar100" else tv.CIFAR10
    tr, te = cls(DATA, train=True, download=True), cls(DATA, train=False, download=True)
    f = lambda d: (torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(d.targets))
    (xtr, ytr), (xte, yte) = f(tr), f(te)
    if SMOKE:
        keep = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0))[: len(xtr) // 10]
        xtr, ytr = xtr[keep], ytr[keep]
    g = torch.Generator().manual_seed(SPLIT_SEED)
    practice = torch.zeros(len(yte), dtype=torch.bool)
    for c in yte.unique().tolist():
        idx = torch.nonzero(yte == c).squeeze(1)
        practice[idx[torch.randperm(len(idx), generator=g)[: len(idx) // 2]]] = True
    return xtr.to(dev), ytr.long().to(dev), xte.to(dev), yte.long().to(dev), practice.to(dev)


# ----------------------------------------------------------------------------- patch eyes
def norm_patches(x_u8):
    """All 6x6 patches of a batch of pictures, each with its mean removed and its spread set to 1."""
    p = F.unfold(x_u8.float(), PATCH).transpose(1, 2)            # [B, positions, 108]
    p = p - p.mean(2, keepdim=True)
    return p / torch.sqrt(p.var(2, unbiased=False, keepdim=True) + NORM_EPS)


def sample_patches(x_u8, n, gen):
    """n random normalised patches from random pictures of x_u8."""
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


def kmeans(p, k, gen):
    c = p[torch.randperm(len(p), generator=gen)[:k].to(dev)].clone()
    for _ in range(KMEANS_ITERS):
        assign = torch.cat([torch.cdist(p[i:i + 20000], c).argmin(1) for i in range(0, len(p), 20000)])
        sums = torch.zeros_like(c).index_add_(0, assign, p)
        cnt = torch.bincount(assign, minlength=k).unsqueeze(1)
        c = torch.where(cnt > 0, sums / cnt.clamp_min(1), c)
    return c


class Eyes:
    """Whitening plus a list of detectors (whitened patches). Nothing here is trained by backprop."""

    def __init__(self, mu, w, enc):
        self.mu, self.w, self.enc = mu, w, enc           # enc: "tri" or a threshold (float)
        self.c = torch.empty(0, mu.numel(), device=dev)
        pos = 32 - PATCH + 1
        half = (pos + 1) // 2
        r = (torch.arange(pos, device=dev) >= half).long()
        quad = (r[:, None] * 2 + r[None, :]).reshape(-1)
        self.q = F.one_hot(quad, 4).float()              # [positions, 4]: which quarter a position is in

    def add(self, c):
        self.c = torch.cat([self.c, c])
        return self

    @property
    def k(self):
        return len(self.c)

    @torch.no_grad()
    def describe(self, x_u8):
        """Pictures -> [N, K*4] descriptions. Columns are detector-major: detector k owns columns 4k..4k+3."""
        out = []
        cn, c2 = F.normalize(self.c, dim=1), self.c.pow(2).sum(1)
        for i in range(0, len(x_u8), BS):
            p = (norm_patches(x_u8[i:i + BS]) - self.mu) @ self.w
            if self.enc == "tri":
                z = (p.pow(2).sum(2, keepdim=True) - 2 * p @ self.c.T + c2).clamp_min(0).sqrt()
                f = F.relu(z.mean(2, keepdim=True) - z)
            else:
                f = F.relu(F.normalize(p, dim=2) @ cn.T - self.enc)
            out.append((f.transpose(1, 2) @ self.q).reshape(len(p), -1))
        return torch.cat(out)


# ----------------------------------------------------------------------------- scoring helpers
def report(pred, y, groups, masks):
    gid = torch.full((int(max(max(g) for g in groups)) + 1,), -1, device=dev)
    for i, g in enumerate(groups):
        gid[torch.tensor(g, device=dev)] = i
    out = {}
    for name, m in masks.items():
        p, t = pred[m], y[m]
        pg, tg = gid[p], gid[t]
        wrong = p != t
        out[name] = dict(
            n=int(m.sum().item()), acc_all=round((~wrong).float().mean().item(), 4),
            acc_groups=[round((p[tg == i] == t[tg == i]).float().mean().item(), 4) for i in range(len(groups))],
            err_older=int((wrong & (pg < tg)).sum().item()), err_newer=int((wrong & (pg > tg)).sum().item()),
            err_same=int((wrong & (pg == tg)).sum().item()))
    return out


def own_cards_only(scores, cls, y, group, masks):
    """First group's pictures, answers restricted to the first group's own classes."""
    m = torch.isin(y, torch.tensor(group, device=dev))
    k = len(group)
    ok = cls[:k][scores[:, :k].argmax(1)] == y
    return {name: round(ok[m & mm].float().mean().item(), 4) for name, mm in masks.items()}


def cards(f, y, classes):
    fn = F.normalize(f, dim=1)
    return torch.stack([F.normalize(fn[y == c].mean(0), dim=0) for c in classes])


class Tables:
    """Running sums for the shared-covariance (lda) and ridge memories. No pictures are kept."""

    def __init__(self, d):
        self.a = torch.zeros(d, d, device=dev, dtype=F64)      # sum of g g^T over all pictures so far
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

    def solve(self, gamma):
        """Returns the lda scorer (W, b, cholesky factor) and the ridge scorer (W, mu) for one smoothing value."""
        d = self.a.shape[0]
        s, n = torch.stack(self.sums), torch.tensor(self.counts, device=dev, dtype=F64)
        m = s / n[:, None]
        big_n, eye = n.sum(), torch.eye(d, device=dev, dtype=F64)
        sw = (self.a - (m * n[:, None]).T @ m) / (big_n - len(n))
        l_lda = torch.linalg.cholesky(sw + gamma * sw.diagonal().mean() * eye)
        w_lda = torch.cholesky_solve(m.T, l_lda)
        b_lda = -0.5 * (m * w_lda.T).sum(1)
        mu = s.sum(0) / big_n
        st = (self.a - big_n * torch.outer(mu, mu)) / big_n
        l_r = torch.linalg.cholesky(st + gamma * st.diagonal().mean() * eye)
        w_r = torch.cholesky_solve((s - n[:, None] * mu).T / big_n, l_r)
        return dict(lda=(w_lda.float(), b_lda.float(), l_lda), ridge=(w_r.float(), mu.float()))


class GrowMemory:
    """Cards of different lengths with R0 (own units) and R7 (predicted fill-in), as EXP-4."""

    def __init__(self):
        self.cls, self.dim, self.card, self.full = [], [], [], None

    def add_stage(self, f, y, classes):
        d_new = f.shape[1]
        if self.full is not None:
            d_old = self.full.shape[1]
            X, Y = f[:, :d_old].double(), f[:, d_old:].double()
            xm, ym = X.mean(0), Y.mean(0)
            Xc, Yc = X - xm, Y - ym
            lam = RIDGE_R7 * Xc.pow(2).sum() / d_old
            W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(d_old, device=dev, dtype=F64), Xc.T @ Yc)
            self.full = torch.cat([self.full, ((self.full.double() - xm) @ W + ym).float()], 1)
        fn = F.normalize(f, dim=1)
        new = []
        for c in classes:
            m = y == c
            self.cls.append(c)
            self.dim.append(d_new)
            self.card.append(F.normalize(fn[m].mean(0), dim=0))
            new.append(f[m].mean(0))
        new = torch.stack(new)
        self.full = new if self.full is None else torch.cat([self.full, new])

    def scores(self, f, rule):
        if rule == "R7":
            return F.normalize(f, dim=1) @ F.normalize(self.full, dim=1).T
        out = torch.empty(len(f), len(self.cls), device=dev)
        for d in sorted(set(self.dim)):
            cols = [i for i, dd in enumerate(self.dim) if dd == d]
            out[:, cols] = F.normalize(f[:, :d], dim=1) @ torch.stack([self.card[i] for i in cols]).T
        return out


def few_shot_cos(f_pool, y_pool, f_te, y_te, classes, rng):
    f_pool, f_te = F.normalize(f_pool, dim=1), F.normalize(f_te, dim=1)
    idx = {c: torch.nonzero(y_pool == c).squeeze(1).cpu().numpy() for c in classes}
    cls = torch.tensor(classes, device=dev)
    out = {}
    for k in SHOTS:
        accs = []
        for _ in range(EPISODES):
            cd = torch.stack([F.normalize(f_pool[torch.from_numpy(rng.choice(idx[c], k, replace=False)).to(dev)]
                                          .mean(0), dim=0) for c in classes])
            accs.append((cls[(f_te @ cd.T).argmax(1)] == y_te).float().mean().item())
        out[str(k)] = round(float(np.mean(accs)), 4)
    return out


def few_shot_lda(g_pool, y_pool, g_te, y_te, classes, rng, chol):
    """Surprise classes from k pictures each, judged with the shared covariance of the 80 learned classes."""
    idx = {c: torch.nonzero(y_pool == c).squeeze(1).cpu().numpy() for c in classes}
    cls = torch.tensor(classes, device=dev)
    out = {}
    for k in SHOTS:
        means = torch.stack([g_pool[torch.from_numpy(rng.choice(idx[c], k, replace=False)).to(dev)].mean(0)
                             for _ in range(EPISODES) for c in classes]).double()        # [E*C, d]
        w = torch.cholesky_solve(means.T, chol)                                           # [d, E*C]
        b = -0.5 * (means * w.T).sum(1)
        s = (g_te @ w.float() + b.float()).reshape(len(g_te), EPISODES, len(classes))
        out[str(k)] = round((cls[s.argmax(2)] == y_te[:, None]).float().mean().item(), 4)
    return out


# ----------------------------------------------------------------------------- backprop network (as EXP-3..6)
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


# ----------------------------------------------------------------------------- network (as EXP-3 / EXP-4)
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
def features(net, x_u8, bs=1000):
    net.eval()
    return torch.cat([net(x_u8[i:i + bs].float() / 255) for i in range(0, len(x_u8), bs)])


def train_stage(net, x_u8, y, classes, way, passes, meter):
    """Train the unlocked units of `net` on one stage's pictures under one of the four jobs."""
    lut = torch.full((int(max(classes)) + 1,), -1, device=dev)
    lut[torch.tensor(classes, device=dev)] = torch.arange(len(classes), device=dev)
    y = lut[y]
    k = len(classes)
    net.train()
    with torch.no_grad():
        dim = net(x_u8[:2].float() / 255).shape[1]
    net.train()
    head = nn.Linear(dim, 4 * k if way == "T1" else k).to(dev)
    extra = list(head.parameters())
    proj = None
    if way == "T3":
        proj = nn.Linear(dim, T3_DIM).to(dev)
        extra += list(proj.parameters())
    params = [p for p in net.parameters() if p.requires_grad] + extra
    steps_per_epoch = math.ceil(len(x_u8) / BATCH)
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, passes * steps_per_epoch)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(passes):
        perm = torch.randperm(len(x_u8), device=dev)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x_u8[idx].float() / 255, y[idx]
            if way == "T0":
                loss = F.cross_entropy(head(net(augment(xb))), yb)
            elif way in ("T1", "T2"):
                xa = augment(xb)
                x4 = torch.cat([torch.rot90(xa, r, (2, 3)) for r in range(4)])
                if way == "T1":
                    y4 = torch.cat([yb * 4 + r for r in range(4)])
                else:
                    y4 = yb.repeat(4)
                loss = F.cross_entropy(head(net(x4)), y4)
            else:  # T3: class job on both copies + the two copies of a picture must agree
                n = len(xb)
                f = net(torch.cat([augment(xb), augment(xb)]))
                loss = F.cross_entropy(head(f), yb.repeat(2))
                z = F.normalize(proj(f), dim=1)
                sim = z @ z.T / T3_TEMP
                sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=dev), float("-inf"))
                partner = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(dev)
                loss = loss + T3_WEIGHT * F.cross_entropy(sim, partner)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    return dict(head=head, seconds=t1 - t0, wh=meter.wh(t0, t1),
                peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0.0)


# ----------------------------------------------------------------------------- tables for growing eyes
class GrowTables:
    """Class means and one shared covariance when the description grows over time and old classes
    never had the newer units. Missing parts are filled from the current stage's pictures: how the
    new units follow from the old ones (as R7 does for cards; block formulas as for monotone
    missing data). No pictures are kept."""

    def __init__(self):
        self.sigma, self.dof, self.means, self.cls = None, 0.0, None, []

    def add_stage(self, g, y, classes):
        g = g.double()
        d_new = g.shape[1]
        ms = torch.stack([g[y == c].mean(0) for c in classes])
        s_cur = torch.zeros(d_new, d_new, device=dev, dtype=F64)
        for i, c in enumerate(classes):
            r = g[y == c] - ms[i]
            s_cur += r.T @ r
        n_cur = float(len(g) - len(classes))
        if self.sigma is None:
            self.sigma, self.dof, self.means = s_cur / n_cur, n_cur, ms
        else:
            d_old = self.sigma.shape[0]
            # old cards: predict their new units from their old units (as R7)
            X, Y = g[:, :d_old], g[:, d_old:]
            xm, ym = X.mean(0), Y.mean(0)
            Xc, Yc = X - xm, Y - ym
            eye = torch.eye(d_old, device=dev, dtype=F64)
            W = torch.linalg.solve(Xc.T @ Xc + RIDGE_R7 * Xc.pow(2).sum() / d_old * eye, Xc.T @ Yc)
            self.means = torch.cat([torch.cat([self.means, (self.means - xm) @ W + ym], 1), ms])
            # shared table: old block pooled over all stages; new blocks through the within-class regression
            s11, s21, s22 = s_cur[:d_old, :d_old], s_cur[d_old:, :d_old], s_cur[d_old:, d_old:]
            P = torch.linalg.solve(s11 + RIDGE_R7 * s11.diagonal().mean() * eye, s21.T).T      # [new, old]
            Q = (s22 - P @ s21.T) / n_cur
            sig11 = (self.dof * self.sigma + s11) / (self.dof + n_cur)
            top = torch.cat([sig11, sig11 @ P.T], 1)
            bot = torch.cat([P @ sig11, Q + P @ sig11 @ P.T], 1)
            self.sigma = torch.cat([top, bot])
            self.sigma = (self.sigma + self.sigma.T) / 2
            self.dof += n_cur
        self.cls += list(classes)

    def solve(self, gamma):
        d = self.sigma.shape[0]
        chol = torch.linalg.cholesky(self.sigma + gamma * self.sigma.diagonal().mean() * torch.eye(d, device=dev, dtype=F64))
        w = torch.cholesky_solve(self.means.T, chol)
        return w.float(), (-0.5 * (self.means * w.T).sum(1)).float(), chol


# ----------------------------------------------------------------------------- one seed
def run(seed, data, meter, log):
    xtr, ytr, xte, yte, practice = data
    rng = np.random.default_rng(seed)
    surprise = sorted(np.random.default_rng(SURPRISE_SEED).permutation(100)[:20].tolist())
    rest = rng.permutation([c for c in range(100) if c not in surprise]).tolist()
    groups = [rest[i * PER_STAGE:(i + 1) * PER_STAGE] for i in range(STAGES)]
    allc = [c for g in groups for c in g]
    final_width = START + ADD * (STAGES - 1)

    def sel(x, y, classes, extra=None):
        m = torch.isin(y, torch.tensor(classes, device=dev))
        return (x[m], y[m]) if extra is None else (x[m], y[m], extra[m])

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

    sx_pool, sy_pool = sel(xtr, ytr, surprise)
    sx_te, sy_te = sel(xte, yte, surprise)
    ax, ay = sel(xtr, ytr, allc)
    tx, ty, tp = sel(xte, yte, allc, practice)
    gid = torch.full((100,), -1, device=dev)
    for i, g in enumerate(groups):
        gid[torch.tensor(g, device=dev)] = i
    a_stage, t_stage = gid[ay], gid[ty]
    cls_all = torch.tensor(allc, device=dev)
    gx0, gy0 = sel(xtr, ytr, groups[0])
    fs_rng = lambda: np.random.default_rng(10_000 + seed)

    def stage_masks(s):
        te_m = t_stage <= s
        p_s = tp[te_m]
        return te_m, ty[te_m], {"full": torch.ones_like(p_s), "practice": p_s, "exam": ~p_s}

    def fixed(eyes_name, f_tr, f_te, f_sp, f_st, cost):
        """Every memory on eyes that never change after they are built. One pass over each stage's pictures."""
        g_tr, g_te, g_sp, g_st = f_tr.sqrt(), f_te.sqrt(), f_sp.sqrt(), f_st.sqrt()
        tb = Tables(g_tr.shape[1])
        card_cos, card_sqrt, line = [], [], []
        t_tab, wh_tab = 0.0, 0.0 if meter.ok else None
        for s, group in enumerate(groups):
            last = s == STAGES - 1
            seen = [c for g in groups[:s + 1] for c in g]
            m_tr = a_stage == s
            card_cos.append(cards(f_tr[m_tr], ay[m_tr], group))
            card_sqrt.append(cards(g_tr[m_tr], ay[m_tr], group))
            with Clock(meter) as c_t:
                tb.add_stage(g_tr[m_tr], ay[m_tr], group)
            t_tab += c_t.seconds
            if wh_tab is not None and c_t.wh is not None:
                wh_tab += c_t.wh
            te_m, y_s, masks = stage_masks(s)
            cls = cls_all[:len(seen)]
            scored = {"cos": F.normalize(f_te[te_m], dim=1) @ torch.cat(card_cos).T,
                      "sqrt_cos": F.normalize(g_te[te_m], dim=1) @ torch.cat(card_sqrt).T}
            extra, fs = {}, {}
            if s + 1 in TABLE_STAGES:
                for g in GAMMAS + (FS_GAMMAS if last else []):
                    with Clock(meter) as c_s:
                        sol = tb.solve(g)
                    if g in GAMMAS:
                        scored[f"lda_g{g}"] = g_te[te_m] @ sol["lda"][0] + sol["lda"][1]
                        scored[f"ridge_g{g}"] = (g_te[te_m] - sol["ridge"][1]) @ sol["ridge"][0]
                        extra[f"lda_g{g}"] = extra[f"ridge_g{g}"] = dict(solve_seconds=round(c_s.seconds, 3), solve_wh=rw(c_s.wh))
                    if last:
                        fs[f"lda_g{g}"] = few_shot_lda(g_sp, sy_pool, g_st, sy_te, surprise, fs_rng(), sol["lda"][2])
                    del sol
            if last:
                fs["cos"] = few_shot_cos(f_sp, sy_pool, f_st, sy_te, surprise, fs_rng())
                fs["sqrt_cos"] = few_shot_cos(g_sp, sy_pool, g_st, sy_te, surprise, fs_rng())
            for rule, S in scored.items():
                rep = report(cls[S.argmax(1)], y_s, groups[:s + 1], masks)
                kw = dict(extra.get(rule, {}))
                if rule in fs:
                    kw["few_shot"] = fs.pop(rule)
                write(eyes=eyes_name, rule=rule, stage=s + 1, **rep, first_own=own_cards_only(S, cls, y_s, groups[0], masks),
                      tables_seconds=round(t_tab, 3), tables_wh=rw(wh_tab), feature_dim=f_tr.shape[1], peak_mem_mb=peak(), **cost, **kw)
                if last:
                    line.append(f"{rule} {rep['full']['acc_all']:.3f}")
            for rule, val in fs.items():          # few-shot only (extra smoothing values)
                write(eyes=eyes_name, rule="fs_" + rule, stage=s + 1, few_shot=val)
        print(f"seed {seed} {eyes_name} | " + " ".join(line), flush=True)

    def bp_cost(c):
        return dict(train_seconds=round(c["seconds"], 2), train_wh=rw(c["wh"]))

    # ---- P: patch eyes (EXP-7's selected: random patches from the first group, threshold 0.1)
    reset_peak()
    gen = torch.Generator().manual_seed(seed * 100 + 2)
    torch.manual_seed(seed * 100 + 2)
    with Clock(meter) as c_b:
        p = sample_patches(gx0, N_PATCHES, gen)
        mu, w = fit_whitening(p)
        eyes = Eyes(mu, w, 0.1).add(((p - mu) @ w)[torch.randperm(len(p), generator=gen)[:K_BIG].to(dev)])
        del p
    with Clock(meter) as c_d:
        f_tr = eyes.describe(ax)
    fixed("P", f_tr, eyes.describe(tx), eyes.describe(sx_pool), eyes.describe(sx_te),
          dict(build_seconds=round(c_b.seconds, 2), build_wh=rw(c_b.wh), describe_train_seconds=round(c_d.seconds, 2),
               describe_train_wh=rw(c_d.wh)))
    del f_tr, eyes

    # ---- F: frozen backprop eyes (full-size network trained on the first group only)
    reset_peak()
    torch.manual_seed(seed * 1000 + 800)
    net = GrowNet(3).grow(final_width)
    c = train_stage(net, gx0, gy0, groups[0], "T0", PASSES, meter)
    fixed("F", features(net, ax), features(net, tx), features(net, sx_pool), features(net, sx_te), bp_cost(c))
    del net

    # ---- R: retrain everything, with its own answer layer as the ceiling
    reset_peak()
    torch.manual_seed(seed * 1000 + 900)
    net = GrowNet(3).grow(final_width)
    c = train_stage(net, ax, ay, allc, "T0", PASSES, meter)
    f_te = features(net, tx)
    with torch.no_grad():
        pred = cls_all[c["head"](f_te).argmax(1)]
    _, y_s, masks = stage_masks(STAGES - 1)
    rep = report(pred, y_s, groups, masks)
    write(eyes="R", rule="own_layer", stage=STAGES, **rep, **bp_cost(c), feature_dim=f_te.shape[1])
    print(f"seed {seed} R own answer layer {rep['full']['acc_all']:.3f}", flush=True)
    fixed("R", features(net, ax), f_te, features(net, sx_pool), features(net, sx_te), bp_cost(c))
    del net

    # ---- G: grow and lock
    reset_peak()
    torch.manual_seed(seed * 1000)
    net, mem, gt = GrowNet(3), GrowMemory(), GrowTables()
    tot = dict(seconds=0.0, wh=0.0 if meter.ok else None)
    t_tab, line = 0.0, []
    for s, group in enumerate(groups):
        last = s == STAGES - 1
        seen = [c for g in groups[:s + 1] for c in g]
        gx, gy = sel(xtr, ytr, group)
        net.grow(START if s == 0 else ADD)
        c = train_stage(net, gx, gy, group, "T0", PASSES, meter)
        tot["seconds"] += c["seconds"]
        if tot["wh"] is not None and c["wh"] is not None:
            tot["wh"] += c["wh"]
        f_cur = features(net, gx)
        mem.add_stage(f_cur, gy, group)
        with Clock(meter) as c_t:
            gt.add_stage(f_cur.sqrt(), gy, group)
        t_tab += c_t.seconds
        te_m, y_s, masks = stage_masks(s)
        f_te = features(net, tx[te_m])
        g_te = f_te.sqrt()
        cls = torch.tensor(mem.cls, device=dev)
        cost = dict(**bp_cost(tot), tables_seconds=round(t_tab, 3), feature_dim=f_te.shape[1], peak_mem_mb=peak())
        scored = {"R0": mem.scores(f_te, "R0"), "R7": mem.scores(f_te, "R7")}
        chols = {}
        if s + 1 in TABLE_STAGES:
            for g in GAMMAS + (FS_GAMMAS if last else []):
                w_, b_, chols[g] = gt.solve(g)
                if g in GAMMAS:
                    scored[f"lda_fill_g{g}"] = g_te @ w_ + b_
        for rule, S in scored.items():
            rep = report(cls[S.argmax(1)], y_s, groups[:s + 1], masks)
            write(eyes="G", rule=rule, stage=s + 1, **rep, first_own=own_cards_only(S, cls, y_s, groups[0], masks), **cost)
            if last:
                line.append(f"{rule} {rep['full']['acc_all']:.3f}")
        if last:   # open-book memories (diagnosis) and the surprise test on the final eyes
            f_all, f_sp, f_st = features(net, ax), features(net, sx_pool), features(net, sx_te)
            ga, gsp, gst = f_all.sqrt(), f_sp.sqrt(), f_st.sqrt()
            S = F.normalize(f_te, dim=1) @ cards(f_all, ay, allc).T
            write(eyes="G", rule="open_cos", stage=s + 1, **report(cls_all[S.argmax(1)], y_s, groups, masks),
                  first_own=own_cards_only(S, cls_all, y_s, groups[0], masks), **cost,
                  few_shot=few_shot_cos(f_sp, sy_pool, f_st, sy_te, surprise, fs_rng()))
            write(eyes="G", rule="fs_sqrt_cos", stage=s + 1, few_shot=few_shot_cos(gsp, sy_pool, gst, sy_te, surprise, fs_rng()))
            tb = Tables(ga.shape[1])
            for st in range(STAGES):
                tb.add_stage(ga[a_stage == st], ay[a_stage == st], groups[st])
            for g in GAMMAS:
                sol = tb.solve(g)
                for name, S in [(f"lda_open_g{g}", g_te @ sol["lda"][0] + sol["lda"][1]),
                                (f"ridge_open_g{g}", (g_te - sol["ridge"][1]) @ sol["ridge"][0])]:
                    rep = report(cls_all[S.argmax(1)], y_s, groups, masks)
                    write(eyes="G", rule=name, stage=s + 1, **rep, first_own=own_cards_only(S, cls_all, y_s, groups[0], masks), **cost)
                    line.append(f"{name} {rep['full']['acc_all']:.3f}")
                del sol
            for g, ch in chols.items():   # surprise test judged with the filled-in table (a real method)
                write(eyes="G", rule=f"fs_lda_fill_g{g}", stage=s + 1,
                      few_shot=few_shot_lda(gsp, sy_pool, gst, sy_te, surprise, fs_rng(), ch))
    print(f"seed {seed} G | " + " ".join(line), flush=True)
    del net, mem, gt


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
    data = load("cifar100")
    with open(OUT / "runs.jsonl", "w") as log:
        for seed in range(SEEDS):
            run(seed, data, meter, log)
            print(f"== seed {seed} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    meter.stop()
    meta = dict(gpu=gpu, torch=torch.__version__, on_kaggle=ON_KAGGLE, smoke=SMOKE, seeds=SEEDS, stages=STAGES,
                per_stage=PER_STAGE, passes=PASSES, start=START, add=ADD, batch=BATCH, lr=LR, k_big=K_BIG,
                n_patches=N_PATCHES, patch=PATCH, norm_eps=NORM_EPS, zca_eps=ZCA_EPS, gammas=GAMMAS, fs_gammas=FS_GAMMAS,
                table_stages=TABLE_STAGES, ridge_r7=RIDGE_R7, split_seed=SPLIT_SEED, shots=SHOTS, episodes=EPISODES,
                power_reported=meter.ok, total_minutes=round((time.time() - t0) / 60, 1),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
