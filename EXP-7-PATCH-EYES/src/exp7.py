"""EXP-7 patch eyes: a feature extractor made of fixed whitened image patches. No backpropagation.

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

Eyes: cut 6x6 patches, normalise each, whiten, keep K of them as detectors. A picture is
described by how strongly each detector answers in each quarter of the picture (4K numbers).
  E1  fixed: detectors and whitening from the first group's pictures only
  E2  growing: 1/8 of the detectors from each new group, old ones locked (threshold rule)
  E3  all classes (diagnosis): detectors and whitening from all 80 classes' pictures
Memory, all one pass, no stored pictures: class cards by angle (cos), the same on square-rooted
features (sqrt_cos), class means + one shared covariance (lda), and a closed-form ridge classifier.
Protocol as EXP-3..6: CIFAR-100, 80 classes in 8 stages of 10, 20 surprise classes, 5 seeds.
Step 0 is an implementation check on CIFAR-10.

Run:  python src/exp7.py              full run (5 seeds)
      EXP7_SMOKE=1 python src/exp7.py   tiny run to check the code
Writes runs.jsonl, sanity.json, meta.json and preds.npz into the output folder.
"""
import json
import os
import pathlib
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
OUT = pathlib.Path(os.environ.get("EXP7_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP7_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP7_SMOKE") == "1"

SEEDS = 1 if SMOKE else 5
STAGES, PER_STAGE = (3 if SMOKE else 8), 10
K_BIG, K_SMALL = (120, 60) if SMOKE else (1600, 400)
N_PATCHES = 20_000 if SMOKE else 200_000
PATCH = 6
NORM_EPS = 10.0                      # added to a patch's variance (pixels on the 0..255 scale)
ZCA_EPS = 0.1                        # whitening constant
KMEANS_ITERS = 10
GAMMAS = [0.01, 0.1, 1.0]            # smoothing of the shared tables, times the average variance
TABLE_STAGES = sorted({1, (STAGES + 1) // 2, STAGES})   # stages at which lda / ridge are solved
SURPRISE_SEED = 1234
SPLIT_SEED = 4321
RIDGE_R7 = 1e-2
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


# ----------------------------------------------------------------------------- step 0: CIFAR-10 check
def sanity(meter):
    """Coates 2011 recipe on CIFAR-10 (K-means detectors, triangle rule), ridge / lda / cards on top."""
    xtr, ytr, xte, yte, _ = load("cifar100" if SMOKE else "cifar10")   # the smoke run only tests the code path
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    p = sample_patches(xtr, N_PATCHES, gen)
    mu, w = fit_whitening(p)
    eyes = Eyes(mu, w, "tri").add(kmeans((p - mu) @ w, K_BIG, gen))
    with Clock(meter) as c:
        ftr = eyes.describe(xtr)
    fte = eyes.describe(xte)
    classes = sorted(ytr.unique().tolist())
    cls = torch.tensor(classes, device=dev)
    res = {"dataset": "cifar100-smoke" if SMOKE else "cifar10", "detectors": eyes.k, "train_pictures": len(xtr),
           "describe_train_seconds": round(c.seconds, 2)}
    res["cos"] = round((cls[(F.normalize(fte, dim=1) @ cards(ftr, ytr, classes).T).argmax(1)] == yte).float().mean().item(), 4)
    gtr, gte = ftr.sqrt(), fte.sqrt()
    res["sqrt_cos"] = round((cls[(F.normalize(gte, dim=1) @ cards(gtr, ytr, classes).T).argmax(1)] == yte).float().mean().item(), 4)
    tb = Tables(gtr.shape[1])
    tb.add_stage(gtr, ytr, classes)
    for g in GAMMAS:
        sol = tb.solve(g)
        res[f"lda_g{g}"] = round((cls[(gte @ sol["lda"][0] + sol["lda"][1]).argmax(1)] == yte).float().mean().item(), 4)
        res[f"ridge_g{g}"] = round((cls[((gte - sol["ridge"][1]) @ sol["ridge"][0]).argmax(1)] == yte).float().mean().item(), 4)
        del sol
    res["best_ridge"] = max(res[f"ridge_g{g}"] for g in GAMMAS)
    res["passes_70"] = bool(res["best_ridge"] >= 0.70)
    print("STEP 0", json.dumps(res), flush=True)
    (OUT / "sanity.json").write_text(json.dumps(res, indent=1))
    del xtr, xte, ftr, fte, gtr, gte, tb
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------- one seed
ARMS = [  # name, source of patches, how detectors are chosen, number, answer rule
    ("E1_rp_tri", "first", "rp", K_BIG, "tri"),
    ("E1_rp_t0", "first", "rp", K_BIG, 0.0),
    ("E1_rp_t0.1", "first", "rp", K_BIG, 0.1),
    ("E1_rp_t0.25", "first", "rp", K_BIG, 0.25),
    ("E1_km_tri", "first", "km", K_BIG, "tri"),
    ("E1_small_tri", "first", "rp", K_SMALL, "tri"),
    ("E3_all_tri", "all", "rp", K_BIG, "tri"),
    ("E2_grow_t0.1", "grow", "rp", K_BIG, 0.1),
]


def run(seed, data, meter, log, preds):
    xtr, ytr, xte, yte, practice = data
    rng = np.random.default_rng(seed)
    surprise = sorted(np.random.default_rng(SURPRISE_SEED).permutation(100)[:20].tolist())
    rest = rng.permutation([c for c in range(100) if c not in surprise]).tolist()
    groups = [rest[i * PER_STAGE:(i + 1) * PER_STAGE] for i in range(STAGES)]
    allc = [c for g in groups for c in g]

    def sel(x, y, classes, extra=None):
        m = torch.isin(y, torch.tensor(classes, device=dev))
        return (x[m], y[m]) if extra is None else (x[m], y[m], extra[m])

    def write(**kw):
        log.write(json.dumps(dict(seed=seed, **kw)) + "\n")
        log.flush()

    def rw(x):
        return None if x is None else round(x, 5)

    sx_pool, sy_pool = sel(xtr, ytr, surprise)
    sx_te, sy_te = sel(xte, yte, surprise)
    ax, ay = sel(xtr, ytr, allc)
    tx, ty, tp = sel(xte, yte, allc, practice)
    gid = torch.full((100,), -1, device=dev)
    for i, g in enumerate(groups):
        gid[torch.tensor(g, device=dev)] = i
    a_stage, t_stage = gid[ay], gid[ty]                 # stage in which each picture's class arrives
    cls_all = torch.tensor(allc, device=dev)
    gx0, _ = sel(xtr, ytr, groups[0])

    for ai, (arm, source, kind, k, enc) in enumerate(ARMS):
        gen = torch.Generator().manual_seed(seed * 100 + ai)
        torch.manual_seed(seed * 100 + ai)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # ---- build the eyes (no labels, no backprop)
        with Clock(meter) as c_build:
            p = sample_patches(ax if source == "all" else gx0, N_PATCHES, gen)
            mu, w = fit_whitening(p)
            eyes = Eyes(mu, w, enc)
            if source == "grow":
                per = k // STAGES
                eyes.add(((p - mu) @ w)[torch.randperm(len(p), generator=gen)[:per].to(dev)])
                for g in groups[1:]:
                    pg = sample_patches(sel(xtr, ytr, g)[0], max(per * 20, 2000), gen)
                    eyes.add(((pg - mu) @ w)[:per])
            elif kind == "km":
                eyes.add(kmeans((p - mu) @ w, k, gen))
            else:
                eyes.add(((p - mu) @ w)[torch.randperm(len(p), generator=gen)[:k].to(dev)])
            del p
        # ---- describe every training picture once, and the test pictures
        with Clock(meter) as c_desc:
            f_tr = eyes.describe(ax)
        f_te, f_sp, f_st = eyes.describe(tx), eyes.describe(sx_pool), eyes.describe(sx_te)
        cost = dict(detectors=eyes.k, feature_dim=f_tr.shape[1], build_seconds=round(c_build.seconds, 2),
                    build_wh=rw(c_build.wh), describe_train_seconds=round(c_desc.seconds, 2), describe_train_wh=rw(c_desc.wh))
        fs_rng = lambda: np.random.default_rng(10_000 + seed)

        if source == "grow":
            per = k // STAGES
            for tag, tr_f, te_f, sp_f, st_f in [("", f_tr, f_te, f_sp, f_st), ("sqrt_", f_tr.sqrt(), f_te.sqrt(), f_sp.sqrt(), f_st.sqrt())]:
                mem = GrowMemory()
                for s, group in enumerate(groups):
                    d = (s + 1) * per * 4                      # units that exist at this stage
                    seen = [c for g in groups[:s + 1] for c in g]
                    mem.add_stage(tr_f[a_stage == s][:, :d], ay[a_stage == s], group)
                    te_m = t_stage <= s
                    y_s, p_s = ty[te_m], tp[te_m]
                    masks = {"full": torch.ones_like(p_s), "practice": p_s, "exam": ~p_s}
                    cls = torch.tensor(mem.cls, device=dev)
                    for rule in ("R0", "R7"):
                        S = mem.scores(te_f[te_m][:, :d], rule)
                        write(arm=arm, rule=tag + rule, stage=s + 1, **report(cls[S.argmax(1)], y_s, groups[:s + 1], masks),
                              first_own=own_cards_only(S, cls, y_s, groups[0], masks), **cost)
                    S = F.normalize(te_f[te_m][:, :d], dim=1) @ cards(tr_f[a_stage <= s][:, :d], ay[a_stage <= s], seen).T
                    seen_t = torch.tensor(seen, device=dev)
                    kw = {}
                    if s == STAGES - 1:
                        kw["few_shot"] = few_shot_cos(sp_f, sy_pool, st_f, sy_te, surprise, fs_rng())
                    rep = report(seen_t[S.argmax(1)], y_s, groups[:s + 1], masks)
                    write(arm=arm, rule=tag + "open", stage=s + 1, **rep,
                          first_own=own_cards_only(S, seen_t, y_s, groups[0], masks), **cost, **kw)
                print(f"seed {seed} {arm} {tag or 'raw_'}open {rep['full']['acc_all']:.3f}", flush=True)
        else:
            g_tr, g_te, g_sp, g_st = f_tr.sqrt(), f_te.sqrt(), f_sp.sqrt(), f_st.sqrt()
            tb = Tables(g_tr.shape[1])
            card_cos, card_sqrt, t_tables, line = [], [], 0.0, []
            for s, group in enumerate(groups):
                last = s == STAGES - 1
                seen = [c for g in groups[:s + 1] for c in g]
                m_tr = a_stage == s                              # only this stage's pictures are used
                card_cos.append(cards(f_tr[m_tr], ay[m_tr], group))
                card_sqrt.append(cards(g_tr[m_tr], ay[m_tr], group))
                with Clock(meter) as c_t:
                    tb.add_stage(g_tr[m_tr], ay[m_tr], group)
                t_tables += c_t.seconds
                te_m = t_stage <= s
                y_s, p_s = ty[te_m], tp[te_m]
                masks = {"full": torch.ones_like(p_s), "practice": p_s, "exam": ~p_s}
                cls = cls_all[:len(seen)]
                scored = {"cos": F.normalize(f_te[te_m], dim=1) @ torch.cat(card_cos).T,
                          "sqrt_cos": F.normalize(g_te[te_m], dim=1) @ torch.cat(card_sqrt).T}
                extra, fs = {}, {}
                if s + 1 in TABLE_STAGES:
                    for g in GAMMAS:
                        with Clock(meter) as c_s:
                            sol = tb.solve(g)
                        scored[f"lda_g{g}"] = g_te[te_m] @ sol["lda"][0] + sol["lda"][1]
                        scored[f"ridge_g{g}"] = (g_te[te_m] - sol["ridge"][1]) @ sol["ridge"][0]
                        extra[f"lda_g{g}"] = extra[f"ridge_g{g}"] = dict(solve_seconds=round(c_s.seconds, 2))
                        if last:
                            fs[f"lda_g{g}"] = few_shot_lda(g_sp, sy_pool, g_st, sy_te, surprise, fs_rng(), sol["lda"][2])
                        del sol
                if last:
                    fs["cos"] = few_shot_cos(f_sp, sy_pool, f_st, sy_te, surprise, fs_rng())
                    fs["sqrt_cos"] = few_shot_cos(g_sp, sy_pool, g_st, sy_te, surprise, fs_rng())
                for rule, S in scored.items():
                    pred = cls[S.argmax(1)]
                    rep = report(pred, y_s, groups[:s + 1], masks)
                    kw = dict(extra.get(rule, {}))
                    if rule in fs:
                        kw["few_shot"] = fs[rule]
                    write(arm=arm, rule=rule, stage=s + 1, **rep, first_own=own_cards_only(S, cls, y_s, groups[0], masks),
                          tables_seconds=round(t_tables, 2), **cost, **kw,
                          peak_mem_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20, 0) if dev.type == "cuda" else None)
                    if last:
                        line.append(f"{rule} {rep['full']['acc_all']:.3f}")
                        if arm.startswith("E1_rp"):
                            preds[f"s{seed}_{arm}_{rule}"] = pred.cpu().numpy().astype(np.int16)
            print(f"seed {seed} {arm} k {eyes.k} build {c_build.seconds:.1f}s describe {c_desc.seconds:.1f}s | " + " ".join(line), flush=True)
            del g_tr, g_te, g_sp, g_st, tb
        del f_tr, f_te, f_sp, f_st, eyes
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    preds[f"s{seed}_true"] = ty.cpu().numpy().astype(np.int16)
    preds[f"s{seed}_practice"] = tp.cpu().numpy()
    preds[f"s{seed}_groups"] = np.array(groups, dtype=np.int16)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "none"
    print("device", dev, gpu, "torch", torch.__version__, "smoke", SMOKE, flush=True)
    if dev.type != "cuda" and not SMOKE:
        raise SystemExit("No GPU: refusing to start the full run on CPU.")
    meter = PowerMeter()
    t0 = time.time()
    san = sanity(meter)
    data = load("cifar100")
    preds = {}
    with open(OUT / "runs.jsonl", "w") as log:
        for seed in range(SEEDS):
            run(seed, data, meter, log, preds)
            np.savez_compressed(OUT / "preds.npz", **preds)
            print(f"== seed {seed} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    meter.stop()
    meta = dict(gpu=gpu, torch=torch.__version__, on_kaggle=ON_KAGGLE, smoke=SMOKE, seeds=SEEDS, stages=STAGES,
                per_stage=PER_STAGE, k_big=K_BIG, k_small=K_SMALL, n_patches=N_PATCHES, patch=PATCH, norm_eps=NORM_EPS,
                zca_eps=ZCA_EPS, kmeans_iters=KMEANS_ITERS, gammas=GAMMAS, table_stages=TABLE_STAGES, ridge_r7=RIDGE_R7,
                split_seed=SPLIT_SEED, shots=SHOTS, episodes=EPISODES, arms=[a[0] for a in ARMS],
                power_reported=meter.ok, sanity=san,
                total_minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
