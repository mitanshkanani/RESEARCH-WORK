"""EXP-9 a new class from a few pictures: can the memory's knowledge of known classes help?

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

Two kinds of eyes, built exactly as in EXP-8:
  P  patch eyes: 1,600 fixed whitened patches from the first group, threshold 0.1 (no backprop)
  F  frozen backprop eyes: network trained on the first group only
Known classes arrive in 8 stages of 10 (CIFAR-100, same orders as EXP-3..8). Only running tables are
kept (class sums and one sum of outer products), never pictures. With the memory as it stands after
stages 1, 4 and 8 (10, 40, 80 known classes), the 20 surprise classes are learned from 1, 3 or 10
pictures (20 draws, the same draws as EXP-8) by these ways, all on square-rooted descriptions:
  W0   plain cards by angle (EXP-8 "sqrt_cos")
  W1   shared table (lda) at smoothing 0.01 (EXP-8's, reference) and 1, 10, 100, 1000
  W2   tidy: subtract the known average, compare by angle (SimpleShot-style)
  W3   learned view: discriminant directions of the known classes (smoothing x {all, half} x {angle, dist})
  +B   borrowing: blend each new card with the mean of its 2 nearest known cards (weight 0.5, 1, 2)
Rehearsal (for choosing settings): memory after stage 6, the classes of stages 7 and 8 as pretend-new.
After stage 8, a 100-class check: surprise classes (3 pictures) added to the 80 known.
Every setting is logged; the choosing is done afterwards by src/figures.py from the rehearsal rows.

Run:  python src/exp9.py               full run (5 seeds)
      EXP9_SMOKE=1 python src/exp9.py  tiny run to check the code
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
OUT = pathlib.Path(os.environ.get("EXP9_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP9_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP9_SMOKE") == "1"

SEEDS = 1 if SMOKE else 5
STAGES, PER_STAGE = 8, 10
K_BIG = int(os.environ.get("EXP9_KBIG", 120 if SMOKE else 1600))
N_PATCHES = 20_000 if SMOKE else 200_000
PATCH = 6
NORM_EPS = 10.0
ZCA_EPS = 0.1
SURPRISE_SEED = 1234
SPLIT_SEED = 4321
PASSES = 1 if SMOKE else 30
BATCH, LR = 128, 1e-3
START, ADD = 32, 16
FINAL_WIDTH = START + ADD * (STAGES - 1)
SHOTS = (1, 3, 10)
EPISODES = 20
CHECK_EPISODES = 5                      # draws used in the 100-class check
BS = 50                                 # pictures described at once (EXP-8 used 100; halved for GPU memory)

W1_GAMMAS = [0.01, 1.0, 10.0, 100.0, 1000.0]   # 0.01 is EXP-8's selected value, kept as a reference only
W3_GAMMAS = [0.01, 0.1, 1.0, 10.0, 100.0]
ALPHAS = [0.5, 1.0, 2.0]
N_BORROW = 2
REHEARSAL_STAGE = 6
TEST_STAGES = (1, 4, 8)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F64 = torch.float64


# ----------------------------------------------------------------------------- power and timing (as EXP-8)
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


# ----------------------------------------------------------------------------- data (as EXP-8)
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


# ----------------------------------------------------------------------------- patch eyes (as EXP-8)
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


# ----------------------------------------------------------------------------- backprop eyes (as EXP-8, T0 only)
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
    """Sum of g g^T over all pictures so far, and per-class sums and counts. No pictures are kept."""

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
        """Everything the ways need, from the tables alone: class means, the known average and the
        within-class table. Cholesky factors are made on demand by chol() (one kept at a time, to save GPU memory)."""
        s, n = torch.stack(self.sums), torch.tensor(self.counts, device=dev, dtype=F64)
        m = s / n[:, None]
        big_n = n.sum()
        mu = s.sum(0) / big_n
        sw = (self.a - (m * n[:, None]).T @ m) / (big_n - len(n))
        return dict(m=m, mu=mu, sw=sw, avg=sw.diagonal().mean(), cls=list(self.cls), cache=(None, None))


def chol(st, gamma):
    if st["cache"][0] != gamma:
        st["cache"] = (None, None)
        a = st["sw"].clone()
        a.diagonal().add_(gamma * st["avg"])
        st["cache"] = (gamma, torch.linalg.cholesky(a))
        del a
    return st["cache"][1]


def view(st, gamma):
    """Learned view: whiten by the smoothed within-class table, then keep the directions along which the
    known class means differ. Returns P [d, C-1] (columns strongest first); a picture's view is (g - mu) @ P."""
    L = chol(st, gamma)
    mc = (st["m"] - st["mu"]).T                                      # [d, C]
    wm = torch.linalg.solve_triangular(L, mc, upper=False)          # whitened, centred class means
    u, _, _ = torch.linalg.svd(wm, full_matrices=False)
    r = len(st["cls"]) - 1
    return torch.linalg.solve_triangular(L.T, u[:, :r], upper=True).float()


# ----------------------------------------------------------------------------- scoring a way
def make_draws(y_pool, classes, rng, shots=SHOTS, episodes=EPISODES):
    """Indices of the pictures each new class is learned from, in EXP-8's order: [E, C, k] per shot count."""
    idx = {c: torch.nonzero(y_pool == c).squeeze(1).cpu().numpy() for c in classes}
    out = {}
    for k in shots:
        out[k] = torch.tensor(np.array([[rng.choice(idx[c], k, replace=False) for c in classes]
                                        for _ in range(episodes)]), device=dev)
    return out


def new_cards(zp, idx, compare, known, k, alpha):
    """Cards of new classes [E, C, r] from their k pictures; optionally borrow from the nearest known cards."""
    c = zp[idx].mean(2)
    if alpha is not None:
        if compare == "angle":
            near = torch.einsum("ecr,jr->ecj", F.normalize(c, dim=2), known)
        else:
            near = 2 * torch.einsum("ecr,jr->ecj", c, known) - known.pow(2).sum(1)
        nb = near.topk(N_BORROW, dim=2).indices                        # [E, C, 2]
        c = (k * c + alpha * known[nb].mean(2)) / (k + alpha)
    return F.normalize(c, dim=2) if compare == "angle" else c


def scores(zt, c, compare):
    """zt [N, r], c [E, C', r] -> [N, E, C']"""
    if compare == "angle":
        return torch.einsum("nr,ecr->nec", zt, c)
    return 2 * torch.einsum("nr,ecr->nec", zt, c) - c.pow(2).sum(2)[None]


def acc(s, cls, y):
    return round((cls[s.argmax(2)] == y[:, None]).float().mean().item(), 4)


def space_ways(st, g_sets):
    """Yields (name, compare, z_sets, known, borrow) for every non-table way. z_sets transforms each
    feature set in g_sets (a dict of [N, d] square-rooted descriptions)."""
    mu = st["mu"].float()
    m = st["m"].float()
    yield "W0", "angle", {k: F.normalize(v, dim=1) for k, v in g_sets.items()}, F.normalize(m, dim=1), False
    yield "W2", "angle", {k: F.normalize(v - mu, dim=1) for k, v in g_sets.items()}, F.normalize(m - mu, dim=1), True
    for g in W3_GAMMAS:
        P = view(st, g)
        z = {k: (v - mu) @ P for k, v in g_sets.items()}
        kn = (m - mu) @ P
        r = P.shape[1]
        for part, rr in (("all", r), ("half", max(1, r // 2))):
            zz = {k: v[:, :rr] for k, v in z.items()}
            kk = kn[:, :rr]
            yield f"W3_g{g}_{part}_angle", "angle", {k: F.normalize(v, dim=1) for k, v in zz.items()}, F.normalize(kk, dim=1), True
            yield f"W3_g{g}_{part}_dist", "dist", zz, kk, True
        del P, z


def few_shot_all(st, g_pool, g_test, y_test, classes, draws):
    """Every way on one few-picture task. Returns {way: {shots: accuracy}}."""
    cls = torch.tensor(classes, device=dev)
    out = {}
    for name, compare, z, known, borrow in space_ways(st, {"p": g_pool, "t": g_test}):
        for alpha in ([None] + ALPHAS if borrow else [None]):
            key = name if alpha is None else f"{name}+B_a{alpha}"
            out[key] = {str(k): acc(scores(z["t"], new_cards(z["p"], idx, compare, known, k, alpha), compare), cls, y_test)
                        for k, idx in draws.items()}
    for g in W1_GAMMAS:
        out[f"W1_g{g}"] = {str(k): lda_acc(st, g, g_pool, g_test, y_test, cls, idx) for k, idx in draws.items()}
    return out


def lda_acc(st, g, g_pool, g_test, y_test, cls, idx):
    e, c, _ = idx.shape
    means = g_pool[idx].mean(2).reshape(e * c, -1).double()
    w = torch.cholesky_solve(means.T, chol(st, g))
    b = -0.5 * (means * w.T).sum(1)
    s = (g_test @ w.float() + b.float()).reshape(len(g_test), e, c)
    return acc(s, cls, y_test)


def check_100(st, g_pool, idx, sy, g_known_te, ky, g_new_te, ny, surprise):
    """80 known classes (cards from all their pictures, via the tables) plus 20 new ones from 3 pictures.
    Returns {way: dict(new=..., old_100=..., old_80=...)}; 'old_80' is the known pictures among known classes only."""
    known_cls = st["cls"]
    allc = torch.tensor(known_cls + list(surprise), device=dev)
    nk = len(known_cls)
    out = {}

    def rec(s_old, s_new):
        return dict(new=acc(s_new, allc, ny), old_100=acc(s_old, allc, ky),
                    old_80=acc(s_old[:, :, :nk], allc[:nk], ky))

    for name, compare, z, known, borrow in space_ways(st, {"p": g_pool, "k": g_known_te, "n": g_new_te}):
        for alpha in ([None] + ALPHAS if borrow else [None]):
            key = name if alpha is None else f"{name}+B_a{alpha}"
            nc = new_cards(z["p"], idx, compare, known, 3, alpha)                    # [E, 20, r]
            kc = known[None].expand(len(nc), -1, -1)
            cards_all = torch.cat([kc, nc], 1)
            out[key] = rec(scores(z["k"], cards_all, compare), scores(z["n"], cards_all, compare))
    for g in W1_GAMMAS:
        e, c, _ = idx.shape
        means = torch.cat([st["m"][None].expand(e, -1, -1), g_pool[idx].mean(2).double()], 1)   # [E, 100, d]
        flat = means.reshape(e * means.shape[1], -1)
        w = torch.cholesky_solve(flat.T, chol(st, g))
        b = -0.5 * (flat * w.T).sum(1)
        sk = (g_known_te @ w.float() + b.float()).reshape(len(g_known_te), e, -1)
        sn = (g_new_te @ w.float() + b.float()).reshape(len(g_new_te), e, -1)
        out[f"W1_g{g}"] = rec(sk, sn)
    return out


# ----------------------------------------------------------------------------- one seed
def run(seed, data, meter, log):
    xtr, ytr, xte, yte = data
    rng = np.random.default_rng(seed)
    surprise = sorted(np.random.default_rng(SURPRISE_SEED).permutation(100)[:20].tolist())
    rest = rng.permutation([c for c in range(100) if c not in surprise]).tolist()
    groups = [rest[i * PER_STAGE:(i + 1) * PER_STAGE] for i in range(STAGES)]
    allc = [c for g in groups for c in g]
    pretend = groups[REHEARSAL_STAGE] + groups[REHEARSAL_STAGE + 1]

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

    sx_pool, sy_pool = sel(xtr, ytr, surprise)
    sx_te, sy_te = sel(xte, yte, surprise)
    rx_pool, ry_pool = sel(xtr, ytr, pretend)
    rx_te, ry_te = sel(xte, yte, pretend)
    kx_te, ky_te = sel(xte, yte, allc)
    gx0, gy0 = sel(xtr, ytr, groups[0])
    # the same draws as EXP-8 for the surprise classes; a separate stream for the rehearsal
    s_draws = make_draws(sy_pool, surprise, np.random.default_rng(10_000 + seed))
    r_draws = make_draws(ry_pool, pretend, np.random.default_rng(20_000 + seed))

    def one_eyes(name, describe, build_cost):
        g = lambda x: describe(x).sqrt()
        reset_peak()
        cost = dict(build_cost)
        t_desc = t_tab = t_solve = t_eval = 0.0
        wh = dict(desc=0.0, tab=0.0, solve=0.0, eval=0.0) if meter.ok else None

        def add(key, clock):
            if wh is not None and clock.wh is not None:
                wh[key] += clock.wh

        with Clock(meter) as c:
            g_sp, g_st = g(sx_pool), g(sx_te)
        t_desc += c.seconds
        add("desc", c)
        tb = None
        for s, group in enumerate(groups):
            stage = s + 1
            with Clock(meter) as c:
                gx, gy = sel(xtr, ytr, group)
                g_stage = g(gx)
            t_desc += c.seconds
            add("desc", c)
            with Clock(meter) as c:
                if tb is None:
                    tb = Tables(g_stage.shape[1])
                tb.add_stage(g_stage, gy, group)
            t_tab += c.seconds
            add("tab", c)
            del g_stage
            if stage != REHEARSAL_STAGE and stage not in TEST_STAGES:
                continue
            with Clock(meter) as c:
                st = tb.state()
            t_solve += c.seconds
            add("solve", c)
            line = []
            if stage == REHEARSAL_STAGE:
                with Clock(meter) as c:
                    g_rp, g_rt = g(rx_pool), g(rx_te)
                t_desc += c.seconds
                add("desc", c)
                with Clock(meter) as c:
                    res = few_shot_all(st, g_rp, g_rt, ry_te, pretend, r_draws)
                t_eval += c.seconds
                add("eval", c)
                del g_rp, g_rt
                for way, fs in res.items():
                    write(eyes=name, task="rehearsal", known=len(st["cls"]), way=way, few_shot=fs)
                line.append(f"rehearsal W0 {res['W0']['3']:.3f} best {max(v['3'] for v in res.values()):.3f}")
            if stage in TEST_STAGES:
                with Clock(meter) as c:
                    res = few_shot_all(st, g_sp, g_st, sy_te, surprise, s_draws)
                t_eval += c.seconds
                add("eval", c)
                for way, fs in res.items():
                    write(eyes=name, task="surprise", known=len(st["cls"]), way=way, few_shot=fs,
                          eval_seconds=round(c.seconds, 2))
                best = max(res, key=lambda w: res[w]["3"])
                line.append(f"known {len(st['cls'])}: W0 {res['W0']['3']:.3f} W1_g100 {res['W1_g100.0']['3']:.3f} "
                            f"W2 {res['W2']['3']:.3f} best {best} {res[best]['3']:.3f}")
            if stage == STAGES:
                with Clock(meter) as c:
                    g_kt = g(kx_te)
                t_desc += c.seconds
                add("desc", c)
                with Clock(meter) as c:
                    res = check_100(st, g_sp, s_draws[3][:CHECK_EPISODES], sy_pool, g_kt, ky_te, g_st, sy_te, surprise)
                t_eval += c.seconds
                add("eval", c)
                for way, r in res.items():
                    write(eyes=name, task="all100", known=len(st["cls"]), way=way, **r)
                del g_kt
            del st
            print(f"seed {seed} {name} stage {stage} | " + " | ".join(line), flush=True)
        write(eyes=name, task="cost", describe_seconds=round(t_desc, 2), tables_seconds=round(t_tab, 3),
              solve_seconds=round(t_solve, 2), eval_seconds=round(t_eval, 2), wh={k: rw(v) for k, v in wh.items()} if wh else None,
              feature_dim=g_sp.shape[1], peak_mem_mb=peak(), **cost)

    # ---- P: patch eyes, exactly as EXP-8
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

    # ---- F: frozen backprop eyes, exactly as EXP-8
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
    with open(OUT / "runs.jsonl", "w") as log:
        for seed in range(SEEDS):
            run(seed, data, meter, log)
            print(f"== seed {seed} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    meter.stop()
    meta = dict(gpu=gpu, torch=torch.__version__, on_kaggle=ON_KAGGLE, smoke=SMOKE, seeds=SEEDS, stages=STAGES,
                per_stage=PER_STAGE, passes=PASSES, k_big=K_BIG, n_patches=N_PATCHES, bs=BS, shots=SHOTS,
                episodes=EPISODES, check_episodes=CHECK_EPISODES, w1_gammas=W1_GAMMAS, w3_gammas=W3_GAMMAS,
                alphas=ALPHAS, n_borrow=N_BORROW, rehearsal_stage=REHEARSAL_STAGE, test_stages=TEST_STAGES,
                surprise_seed=SURPRISE_SEED, power_reported=meter.ok, total_minutes=round((time.time() - t0) / 60, 1),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
