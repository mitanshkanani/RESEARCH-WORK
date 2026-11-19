"""EXP-12 old or new: a switch in front of the memory, computed in one step from what the memory stores.

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

Eyes and memory: EXP-11's selected version A2 (1,600 stencils, two-sided, 4 quarters, mirrored; 12,800 numbers),
compact tables, learned view with all directions, borrowing from the 2 nearest known cards. Same seeds, classes and
random draws of new-class pictures as EXP-11.

Naming within a group: by angle in the learned view (old cards only, or new cards only).
Switches (a picture goes to the new group when the switch says "new"):
  S0     margin: best new score > best old score - h  (= EXP-11's C1 handicap)
  F only familiarity F > threshold, the threshold being the q-th fraction of F over the new pictures in hand
  F + h  both
  perfect switch (a ceiling): the picture's true group
Familiarity signals, all from the old cards and the smoothed shared table:
  F1 in-view:  min over old cards of the squared distance in the view's coordinates (P)
  F2 residual: squared length of the whitened description outside the view
  F3 full:     F1 + F2 (distance through the whole table to the nearest old card)
  F4 relative: F1 - distance to the background (within table + spread of the cards); the residual cancels
Every (smoothing, pictures, signal, q, h) is logged as old and new accuracy, averaged over the 20 draws. The choosing
is done afterwards by src/figures.py from the rehearsal rows only.

Run:  python src/exp12.py                full run (5 seeds)
      EXP12_SMOKE=1 python src/exp12.py  tiny run to check the code
      EXP12_SEEDS=1 EXP12_STOP=6 python src/exp12.py   one seed, up to the stage-6 rehearsal
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
OUT = pathlib.Path(os.environ.get("EXP12_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP12_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP12_SMOKE") == "1"

SEEDS = int(os.environ.get("EXP12_SEEDS", 1 if SMOKE else 5))
STOP = int(os.environ.get("EXP12_STOP", 8))           # last stage to run (8 = everything)
STAGES, PER_STAGE = 8, 10
N_PATCHES = 20_000 if SMOKE else 200_000
PATCH = 6
NORM_EPS = 10.0
ZCA_EPS = 0.1
THR = 0.1
SURPRISE_SEED = 1234
SHOTS = (1, 3, 10)
EPISODES = 20
CHUNK = 1000

EYES = dict(k=1600, sides=2, grid=2, mirror=True)     # EXP-11's A2
K_MAX = 3200                                           # stencils are cut exactly as in EXP-11 (A2 used the first 1,600)
SMOKE_DIV = 10

GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0]
ALPHA = 1.0
N_BORROW = 2
G1 = [round(x, 3) for x in np.arange(-0.2, 0.5001, 0.01)]     # handicap on cosine scores, as EXP-10 and EXP-11
H_GRID = G1 + [9.0]                                           # 9.0 = "no handicap test" (the margin always says new)
Q_REAL = [round(x, 3) for x in np.arange(0.0, 1.0001, 0.02)]
Q_GRID = [-1.0] + Q_REAL                                      # -1 = "no familiarity test"
SIGNALS = ("F1", "F2", "F3", "F4")
DECILES = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]

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


# ----------------------------------------------------------------------------- patch eyes (as EXP-11)
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
        buf.copy_(self.a)
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


def bundle(tb, buf, info, gamma, sets):
    """The learned view P at one smoothing (as EXP-11), plus the residual of every description in `sets`:
    the squared length of the whitened description outside the view. sets: CPU float32 tensors, centred at mu."""
    tb.fill(buf, info)
    md = buf.diagonal().mean()
    buf.diagonal().add_(gamma * md)
    chol_inplace(buf)
    mc = (info["m"] - info["mu"]).T
    u, _, _ = torch.linalg.svd(torch.linalg.solve_triangular(buf, mc, upper=False), full_matrices=False)
    r = len(info["cls"]) - 1
    ur = u[:, :r].contiguous()
    P = torch.linalg.solve_triangular(buf.mT, ur, upper=True)
    res = []
    for G in sets:
        out = []
        for i in range(0, len(G), CHUNK):
            w = torch.linalg.solve_triangular(buf, G[i:i + CHUNK].to(dev).double().T, upper=False)
            out.append((w.pow(2).sum(0) - (ur.T @ w).pow(2).sum(0)).float())
            del w
        res.append(torch.cat(out))
    return P.float(), r, res


# ----------------------------------------------------------------------------- the switch
def new_cards(zp, idx, known, k):
    """New cards by angle with borrowing from the 2 nearest known cards (as EXP-9 to EXP-11)."""
    c = zp[idx].mean(2)
    near = torch.einsum("ecr,jr->ecj", F.normalize(c, dim=2), known)
    nb = near.topk(N_BORROW, dim=2).indices
    c = (k * c + ALPHA * known[nb].mean(2)) / (k + ALPHA)
    return F.normalize(c, dim=2)


def auc(score_old, score_new):
    """Chance that a random new picture has a higher score than a random old one."""
    x = torch.cat([score_old, score_new]).double()
    rank = torch.empty_like(x)
    rank[x.argsort()] = torch.arange(1, len(x) + 1, device=x.device, dtype=x.dtype)
    n0, n1 = len(score_old), len(score_new)
    return round(((rank[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)).item(), 4)


def switch_tables(oc_old, m_old, f_old, cn, m_new, f_new, thr):
    """Old and new accuracy for every (q, h), one draw.
    oc_old: old test picture named right within the old group; cn: new test picture named right within the new group;
    m_*: best new score - best old score; f_*: familiarity; thr: [Q] thresholds (-inf for 'no familiarity test')."""
    hg = torch.tensor(H_GRID, device=dev)
    a_old = (oc_old[None] & (f_old[None] > thr[:, None])).float()                 # [Q, N_old]: right, and unfamiliar
    go_old = (m_old[None] > -hg[:, None]).float()                                  # [H, N_old]: margin says new
    old = (oc_old.float().sum() - a_old @ go_old.T) / len(oc_old)                  # right answers kept in the old group
    a_new = (cn[None] & (f_new[None] > thr[:, None])).float()
    go_new = (m_new[None] > -hg[:, None]).float()
    new = (a_new @ go_new.T) / len(cn)
    return old, new


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
    times = dict(describe=0.0, tables=0.0, view=0.0, scoring=0.0)
    wh = {k: 0.0 for k in times} if meter.ok else None

    def tick(key, c):
        times[key] += c.seconds
        if wh is not None and c.wh is not None:
            wh[key] += c.wh

    def describe_cpu(x, mu_g):
        """Descriptions (picture and mirror averaged), centred, kept in ordinary memory."""
        out = []
        with Clock(meter) as c:
            for i in range(0, len(x), CHUNK):
                out.append((eyes.answer(x[i:i + CHUNK]) - mu_g).cpu())
        tick("describe", c)
        return torch.cat(out)

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

    def proj(G, P):
        return torch.cat([G[i:i + CHUNK].to(dev) @ P for i in range(0, len(G), CHUNK)])

    def evaluate(task, tb, new_cls):
        info = tb.info()
        mu_g = info["mu"].float()
        n_old = len(info["cls"])
        ox, oy = sel(xte, yte, info["cls"])
        px, py = sel(xtr, ytr, new_cls)
        nx, ny = sel(xte, yte, new_cls)
        draws = make_draws(py, new_cls, np.random.default_rng({"part1": 10_000, "rehearsal": 20_000}[task] + seed), SHOTS)
        used = torch.unique(torch.cat([v.reshape(-1) for v in draws.values()]))
        where = torch.full((len(py),), -1, dtype=torch.long, device=dev)
        where[used] = torch.arange(len(used), device=dev)
        draws = {k: where[v] for k, v in draws.items()}                            # positions among the used pool pictures
        Go, Gn, Gp = describe_cpu(ox, mu_g), describe_cpu(nx, mu_g), describe_cpu(px[used], mu_g)
        cls_old = torch.tensor(info["cls"], device=dev)
        cls_new = torch.tensor(list(new_cls), device=dev)
        km = info["m"].float() - mu_g
        wts = (info["n"] / info["n"].sum()).float()
        buf = torch.empty(d, d, device=dev, dtype=F64)
        for gamma in GAMMAS:
            with Clock(meter) as c:
                P, r, (res_o, res_n, res_p) = bundle(tb, buf, info, gamma, [Go, Gn, Gp])
            tick("view", c)
            with Clock(meter) as c:
                zo, zn, zp = proj(Go, P), proj(Gn, P), proj(Gp, P)
                kp = km @ P
                ka = F.normalize(kp, dim=1)
                bg = torch.linalg.inv(torch.eye(r, device=dev) + (kp.T * wts) @ kp)    # background spread inside the view

                def fam(z, res):
                    f1 = torch.cdist(z, kp).pow(2).min(1).values
                    return dict(F1=f1, F2=res, F3=f1 + res, F4=f1 - ((z @ bg) * z).sum(1))

                fo, fn_, fp = fam(zo, res_o), fam(zn, res_n), fam(zp, res_p)
                write(task=task, kind="auc_f", gamma=gamma, known=n_old, **{s: auc(fo[s], fn_[s]) for s in SIGNALS})
                qs = torch.tensor(DECILES, device=dev, dtype=torch.float32)
                write(task=task, kind="deciles", gamma=gamma,
                      **{s: dict(old=torch.quantile(fo[s], qs).tolist(), new=torch.quantile(fn_[s], qs).tolist(),
                                 pool=torch.quantile(fp[s], qs).tolist()) for s in SIGNALS})
                so, sn, sp = F.normalize(zo, dim=1), F.normalize(zn, dim=1), F.normalize(zp, dim=1)
                cos_oo, cos_no = so @ ka.T, sn @ ka.T
                oc_old = cls_old[cos_oo.argmax(1)] == oy                          # named right among old cards only
                best_old_o, best_old_n = cos_oo.max(1).values, cos_no.max(1).values
                for k, idx in draws.items():
                    nc = new_cards(sp, idx, ka, k)                                 # [E, 20, r]
                    cos_on = torch.einsum("nr,ecr->enc", so, nc)
                    cos_nn = torch.einsum("nr,ecr->enc", sn, nc)
                    m_old = cos_on.max(2).values - best_old_o[None]                # [E, N_old]
                    m_new = cos_nn.max(2).values - best_old_n[None]
                    cn = cls_new[cos_nn.argmax(2)] == ny[None]                     # [E, N_new] named right among new only
                    write(task=task, kind="oracle", gamma=gamma, shots=k, old=round(oc_old.float().mean().item(), 4),
                          new=round(cn.float().mean().item(), 4))
                    write(task=task, kind="auc_m", gamma=gamma, shots=k,
                          margin=round(float(np.mean([auc(m_old[e], m_new[e]) for e in range(len(m_old))])), 4))
                    for s in SIGNALS:
                        fpool = fp[s][idx.reshape(len(idx), -1)]                   # [E, 20k] familiarity of the pictures in hand
                        thr = torch.quantile(fpool, torch.tensor(Q_REAL, device=dev, dtype=fpool.dtype), dim=1).T    # [E, Q_real]
                        thr = torch.cat([torch.full((len(thr), 1), -float("inf"), device=dev), thr], 1)
                        acc_o = torch.zeros(len(Q_GRID), len(H_GRID), device=dev)
                        acc_n = torch.zeros_like(acc_o)
                        for e in range(len(idx)):
                            o, nn_ = switch_tables(oc_old, m_old[e], fo[s], cn[e], m_new[e], fn_[s], thr[e])
                            acc_o += o
                            acc_n += nn_
                        acc_o, acc_n = acc_o / len(idx), acc_n / len(idx)
                        write(task=task, kind="switch", gamma=gamma, shots=k, signal=s, known=n_old,
                              old=(acc_o * 10000).round().int().tolist(), new=(acc_n * 10000).round().int().tolist())
                    if k == 3:
                        i0, ih = Q_GRID.index(-1.0), H_GRID.index(0.08)
                        print(f"seed {seed} {task} gamma {gamma} k=3 | S0 h=0.08 old {acc_o[i0, ih]:.3f} new {acc_n[i0, ih]:.3f}"
                              f" | perfect old {oc_old.float().mean():.3f} new {cn.float().mean():.3f}", flush=True)
                    del cos_on, cos_nn, m_old, m_new, cn, nc
                del zo, zn, zp, fo, fn_, fp
            tick("scoring", c)
        del buf, Go, Gn, Gp
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


def make_draws(y_pool, classes, rng, shots, episodes=EPISODES):
    idx = {c: torch.nonzero(y_pool == c).squeeze(1).cpu().numpy() for c in classes}
    return {k: torch.tensor(np.array([[rng.choice(idx[c], k, replace=False) for c in classes]
                                      for _ in range(episodes)]), device=dev) for k in shots}


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
                per_stage=PER_STAGE, eyes=EYES, n_patches=N_PATCHES, thr=THR, shots=SHOTS, episodes=EPISODES,
                gammas=GAMMAS, alpha=ALPHA, n_borrow=N_BORROW, h_grid=H_GRID, q_grid=Q_GRID, signals=SIGNALS,
                deciles=DECILES, surprise_seed=SURPRISE_SEED, power_reported=meter.ok,
                total_minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
