"""EXP-13 how many pictures: how fast do new classes catch up with old ones as they get more pictures, and can
shifted and enlarged copies of each new picture make it count for more? No backpropagation anywhere.

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

Eyes and memory: EXP-11's version A2 (1,600 stencils, two-sided, 4 quarters, mirrored; 12,800 numbers), compact tables,
learned view of the old classes with all directions, new cards by angle with borrowing from the 2 nearest known cards
(weight 1 picture). Same seeds, classes and draws of 1, 3 and 10 pictures as EXP-11 and EXP-12.
Part 1, the curve: 1, 3, 10, 30, 100 pictures per new class and all of them, every handicap (h = 0: no correction).
Part 2, copies (1, 3, 10 pictures): each new picture's description is averaged with its copies before the card is made:
  none   the picture (and its mirror, as always)
  shift  + 8 copies moved by 3 pixels
  zoom   + 5 copies: a 28 x 28 part enlarged to 32 x 32
  both   + all 13
Test pictures and old cards are never copied. Every setting is logged; the choosing is done by src/figures.py.

Run:  python src/exp13.py                full run (5 seeds)
      EXP13_SMOKE=1 python src/exp13.py  tiny run to check the code
      EXP13_SEEDS=1 EXP13_STOP=6 python src/exp13.py   one seed, up to the stage-6 rehearsal
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
OUT = pathlib.Path(os.environ.get("EXP13_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP13_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP13_SMOKE") == "1"

SEEDS = int(os.environ.get("EXP13_SEEDS", 1 if SMOKE else 5))
STOP = int(os.environ.get("EXP13_STOP", 8))           # last stage to run (8 = everything)
STAGES, PER_STAGE = 8, 10
N_PATCHES = 20_000 if SMOKE else 200_000
PATCH = 6
NORM_EPS = 10.0
ZCA_EPS = 0.1
THR = 0.1
SURPRISE_SEED = 1234
SHOTS = (1, 3, 10)                                    # drawn exactly as EXP-11 and EXP-12 (same random generator order)
BIG = {30: 20, 100: 10}                               # more pictures per new class: draws each
EPISODES = 20
COPY_SETS = ("none", "shift", "zoom", "both")
CHUNK = 1000

EYES = dict(k=1600, sides=2, grid=2, mirror=True)     # EXP-11's A2
K_MAX = 3200                                           # stencils are cut exactly as in EXP-11 (A2 used the first 1,600)
SMOKE_DIV = 10

GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0]
ALPHA = 1.0
N_BORROW = 2
G1 = [round(x, 3) for x in np.arange(-0.2, 0.5001, 0.01)]     # handicap on cosine scores, as EXP-10 and EXP-11

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




def bundle(tb, buf, info, gamma):
    """The learned view P at one smoothing (as EXP-11 and EXP-12)."""
    tb.fill(buf, info)
    md = buf.diagonal().mean()
    buf.diagonal().add_(gamma * md)
    chol_inplace(buf)
    mc = (info["m"] - info["mu"]).T
    u, _, _ = torch.linalg.svd(torch.linalg.solve_triangular(buf, mc, upper=False), full_matrices=False)
    r = len(info["cls"]) - 1
    P = torch.linalg.solve_triangular(buf.mT, u[:, :r].contiguous(), upper=True)
    return P.float(), r


# ----------------------------------------------------------------------------- copies of a new picture (image-specific)
def shifted(x, dx, dy, pad=3):
    """The picture moved by (dx, dy) pixels; the uncovered border is filled by reflection."""
    p = F.pad(x.float(), (pad, pad, pad, pad), mode="reflect")
    return p[:, :, pad + dy:pad + dy + 32, pad + dx:pad + dx + 32]


def zoomed(x, ox, oy, size=28):
    """A 28 x 28 part of the picture, enlarged back to 32 x 32."""
    c = x.float()[:, :, oy:oy + size, ox:ox + size]
    return F.interpolate(c, size=(32, 32), mode="bilinear", align_corners=False)


SHIFTS = [(dx, dy) for dy in (-3, 0, 3) for dx in (-3, 0, 3) if (dx, dy) != (0, 0)]       # 8 moved copies
ZOOMS = [(0, 0), (4, 0), (0, 4), (4, 4), (2, 2)]                                         # 5 enlarged parts


# ----------------------------------------------------------------------------- scoring (as EXP-11's C0 and C1)
def new_cards(zp, idx, known, k):
    """New cards by angle with borrowing from the 2 nearest known cards, weight ALPHA pictures (as EXP-9 to EXP-12)."""
    c = zp[idx].mean(2)
    near = torch.einsum("ecr,jr->ecj", F.normalize(c, dim=2), known)
    nb = near.topk(N_BORROW, dim=2).indices
    c = (k * c + ALPHA * known[nb].mean(2)) / (k + ALPHA)
    return F.normalize(c, dim=2)


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
    times = dict(describe=0.0, copies=0.0, tables=0.0, view=0.0, scoring=0.0)
    wh = {k: 0.0 for k in times} if meter.ok else None

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

    def project(x, mu_g, wcat):
        """Describe pictures (picture and mirror averaged), keep only their coordinates in every view."""
        out = []
        with Clock(meter) as c:
            for i in range(0, len(x), CHUNK):
                out.append((eyes.answer(x[i:i + CHUNK]) - mu_g) @ wcat)
        tick("describe", c)
        return torch.cat(out)

    def project_copies(x, mu_g, wcat):
        """Each copy set: the description averaged over the picture and its copies, in every view."""
        out = {s: [] for s in COPY_SETS[1:]}
        with Clock(meter) as c:
            for i in range(0, len(x), CHUNK):
                xb = x[i:i + CHUNK]
                g0 = eyes.answer(xb)
                gs = sum(eyes.answer(shifted(xb, dx, dy)) for dx, dy in SHIFTS)
                gz = sum(eyes.answer(zoomed(xb, ox, oy)) for ox, oy in ZOOMS)
                out["shift"].append(((g0 + gs) / (1 + len(SHIFTS)) - mu_g) @ wcat)
                out["zoom"].append(((g0 + gz) / (1 + len(ZOOMS)) - mu_g) @ wcat)
                out["both"].append(((g0 + gs + gz) / (1 + len(SHIFTS) + len(ZOOMS)) - mu_g) @ wcat)
                del g0, gs, gz
        tick("copies", c)
        return {s: torch.cat(v) for s, v in out.items()}

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
        draws = make_draws(py, new_cls, draw_rng, SHOTS)                         # identical to EXP-11 and EXP-12
        n_min = min(int((py == c).sum()) for c in new_cls)
        big = {k: e for k, e in BIG.items() if k < n_min}
        draws.update(make_draws(py, new_cls, draw_rng, tuple(big), big))
        draws[n_min] = torch.stack([torch.nonzero(py == c).squeeze(1)[:n_min] for c in new_cls])[None]   # all pictures
        used = torch.unique(torch.cat([draws[k].reshape(-1) for k in SHOTS]))
        where = torch.full((len(py),), -1, dtype=torch.long, device=dev)
        where[used] = torch.arange(len(used), device=dev)
        zo_all, zn_all, zp_all = project(ox, mu_g, wcat), project(nx, mu_g, wcat), project(px, mu_g, wcat)
        zc_all = project_copies(px[used], mu_g, wcat)
        cls_old = torch.tensor(info["cls"], device=dev)
        cls_new = torch.tensor(list(new_cls), device=dev)
        km = info["m"].float() - mu_g
        hg = torch.tensor(G1, device=dev, dtype=torch.float32)
        with Clock(meter) as c:
            for gi, gamma in enumerate(GAMMAS):
                sl = lambda z: F.normalize(z[:, gi * r:(gi + 1) * r], dim=1)
                ka = F.normalize(km @ vs[gi][0], dim=1)
                so, sn = sl(zo_all), sl(zn_all)
                cos_oo, cos_no = so @ ka.T, sn @ ka.T
                oc_old = (cls_old[cos_oo.argmax(1)] == oy).float()
                best_old_o, best_old_n = cos_oo.max(1).values, cos_no.max(1).values
                pools = {"none": (sl(zp_all), None)}
                pools.update({s: (sl(zc_all[s]), where) for s in COPY_SETS[1:]})
                for k, idx in draws.items():
                    for cs in (COPY_SETS if k in SHOTS else ("none",)):
                        zp, remap = pools[cs]
                        nc = new_cards(zp, idx if remap is None else remap[idx], ka, k)
                        m_old = torch.einsum("nr,ecr->enc", so, nc).max(2).values - best_old_o[None]
                        cos_nn = torch.einsum("nr,ecr->enc", sn, nc)
                        m_new = cos_nn.max(2).values - best_old_n[None]
                        cn = (cls_new[cos_nn.argmax(2)] == ny[None]).float()
                        # answer is new iff best new score > best old score - h (EXP-11's C1; h = 0 is no correction)
                        old = ((m_old[:, :, None] <= -hg).float() * oc_old[None, :, None]).mean(1).mean(0)
                        new = ((m_new[:, :, None] > -hg).float() * cn[:, :, None]).mean(1).mean(0)
                        write(task=task, kind="curve", gamma=gamma, shots=int(k), copies=cs, known=n_old,
                              draws=len(idx), new_only=round(cn.mean().item(), 4),
                              old=(old * 10000).round().int().tolist(), new=(new * 10000).round().int().tolist())
                        if k == 3 and cs in ("none", "both"):
                            ih = G1.index(0.08)
                            print(f"seed {seed} {task} gamma {gamma} k=3 {cs:5s} | h=0.08 old {old[ih]:.3f} new {new[ih]:.3f}"
                                  f" | new-only {cn.mean():.3f}", flush=True)
                        del m_old, m_new, cos_nn, cn, nc
                write(task=task, kind="old_only", gamma=gamma, known=n_old, old=round(oc_old.mean().item(), 4))
        tick("scoring", c)
        del zo_all, zn_all, zp_all, zc_all, vs, wcat

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
                copy_sets=COPY_SETS, shifts=SHIFTS, zooms=ZOOMS, gammas=GAMMAS, alpha=ALPHA, n_borrow=N_BORROW, g1=G1,
                surprise_seed=SURPRISE_SEED, power_reported=meter.ok,
                total_minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
