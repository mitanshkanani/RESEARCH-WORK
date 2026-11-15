"""EXP-4 fair cards: compare memory cards of different sizes fairly, without old pictures.

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

The feature extractor ("eyes") is built exactly as the `grow` arm of EXP-3: each stage adds new
units and locks the old ones. What changes is the rule that compares a picture with the memory
cards. Eight rules (R0..R7) are applied to the same eyes. Two reference points are rebuilt:
`frozen` (trained on the first group only) and `retrain` (a fresh network trained on all classes,
built for the final stage only).

Run:  python src/exp4.py              full run (5 seeds)
      EXP4_SMOKE=1 python src/exp4.py   tiny run to check the code
Writes runs.jsonl, meta.json and preds_<dataset>.npz into the output folder.
"""
import json
import math
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or (os.name == "posix" and pathlib.Path("/kaggle/working").exists())
try:
    ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    ROOT = pathlib.Path.cwd()
OUT = pathlib.Path(os.environ.get("EXP4_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP4_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP4_SMOKE") == "1"

SEEDS = 1 if SMOKE else 5
EPOCHS = 1 if SMOKE else 30
BATCH = 128
LR = 1e-3
SURPRISE_SEED = 1234                 # same 20 held-out classes as EXP-3 (not used here, kept out of training)
SPLIT_SEED = 4321                    # practice half / exam half of the test pictures
RULES = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
VAR_FLOOR = 0.1                      # R6: added to each unit's variance, as a share of the class's mean variance
RIDGE = 1e-2                         # R7: strength of the ridge penalty, relative to the mean input variance

CONFIGS = {
    "digits_clothes": dict(stages=2, per_stage=10, start=32, add=32),
    "cifar100": dict(stages=3 if SMOKE else 8, per_stage=10, start=32, add=16),
}

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- data (as EXP-3)
def load(name):
    import torchvision.datasets as tv
    DATA.mkdir(parents=True, exist_ok=True)
    if name == "cifar100":
        tr, te = tv.CIFAR100(DATA, train=True, download=True), tv.CIFAR100(DATA, train=False, download=True)
        f = lambda d: (torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(d.targets))
        (xtr, ytr), (xte, yte) = f(tr), f(te)
    else:  # digits are classes 0-9, clothes are classes 10-19
        parts = []
        for train in (True, False):
            a, b = tv.MNIST(DATA, train=train, download=True), tv.FashionMNIST(DATA, train=train, download=True)
            parts.append((torch.cat([a.data, b.data]).unsqueeze(1), torch.cat([a.targets, b.targets + 10])))
        (xtr, ytr), (xte, yte) = parts
    if SMOKE:
        keep = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0))[: len(xtr) // 10]
        xtr, ytr = xtr[keep], ytr[keep]
    # practice half / exam half: a random half of each class's test pictures, fixed once
    g = torch.Generator().manual_seed(SPLIT_SEED)
    practice = torch.zeros(len(yte), dtype=torch.bool)
    for c in yte.unique().tolist():
        idx = torch.nonzero(yte == c).squeeze(1)
        practice[idx[torch.randperm(len(idx), generator=g)[: len(idx) // 2]]] = True
    return xtr.to(dev), ytr.long().to(dev), xte.to(dev), yte.long().to(dev), practice.to(dev)


def augment(x, colour):
    n = len(x)
    scale = torch.empty(n, device=dev).uniform_(0.7, 1.0)
    ang = torch.empty(n, device=dev).uniform_(-0.26, 0.26) * (0.0 if colour else 1.0)
    tx, ty = torch.empty(2, n, device=dev).uniform_(-0.15, 0.15)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0) if colour else torch.ones(n, device=dev)
    theta = torch.stack([
        torch.stack([scale * torch.cos(ang) * flip, -scale * torch.sin(ang), tx], 1),
        torch.stack([scale * torch.sin(ang) * flip, scale * torch.cos(ang), ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * c + m) * b
    if colour:
        grey = torch.rand(n, 1, 1, 1, device=dev) < 0.2
        x = torch.where(grey, x.mean(1, keepdim=True).expand_as(x), x)
    return x.clamp(0, 1)


# ----------------------------------------------------------------------------- network (as EXP-3)
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


def train_stage(net, x_u8, y, classes, colour):
    lut = torch.full((int(max(classes)) + 1,), -1, device=dev)
    lut[torch.tensor(classes, device=dev)] = torch.arange(len(classes), device=dev)
    y = lut[y]
    net.train()
    with torch.no_grad():
        dim = net(x_u8[:2].float() / 255).shape[1]
    net.train()
    head = nn.Linear(dim, len(classes)).to(dev)
    params = [p for p in net.parameters() if p.requires_grad] + list(head.parameters())
    steps_per_epoch = math.ceil(len(x_u8) / BATCH)
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * steps_per_epoch)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x_u8), device=dev)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(head(net(augment(x_u8[idx].float() / 255, colour))), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return round(time.time() - t0, 2)


# ----------------------------------------------------------------------------- memory with eight rules
class Memory:
    """Everything the system keeps about the classes it has learned. No pictures are stored.

    Per class: the card (average of unit-length features on the units that existed when the class
    was learned), the per-unit average and variance of the raw features (R6), two numbers about
    how the card scores on strangers (R4 running, R5 latest stage), and a full-length card whose
    missing part is predicted (R7)."""

    def __init__(self):
        self.cls, self.dim, self.grp = [], [], []       # class id, card length, stage index
        self.card, self.mu, self.var = [], [], []
        self.run = []                                     # R4: [count, sum, sum of squares] per card
        self.latest = []                                  # R5: [mean, sd] per card
        self.full = None                                  # R7: raw-mean cards at full current length
        self.fit_seconds = 0.0

    def own_cos(self, f):
        """Cosine between each picture and each card, on that card's own units. [pictures, cards]"""
        out = torch.empty(len(f), len(self.cls), device=dev)
        for d in sorted(set(self.dim)):
            cols = [i for i, dd in enumerate(self.dim) if dd == d]
            out[:, cols] = F.normalize(f[:, :d], dim=1) @ torch.stack([self.card[i] for i in cols]).T
        return out

    def add_stage(self, f, y, classes, stage):
        """f, y: this stage's training pictures (features at the current full length) and labels."""
        t0 = time.time()
        d_new = f.shape[1]
        # R7: predict the new units of every old card from its old units, using a map fitted on
        # the current pictures (new units = old units x W + b, ridge regression)
        if self.full is not None:
            d_old = self.full.shape[1]
            X, Y = f[:, :d_old].double(), f[:, d_old:].double()
            xm, ym = X.mean(0), Y.mean(0)
            Xc, Yc = X - xm, Y - ym
            lam = RIDGE * Xc.pow(2).sum() / len(X) / d_old * len(X)
            W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(d_old, device=dev, dtype=torch.float64), Xc.T @ Yc)
            guess = ((self.full.double() - xm) @ W + ym).float()
            self.full = torch.cat([self.full, guess], 1)
        fn = F.normalize(f, dim=1)
        new_full = []
        for c in classes:
            m = y == c
            self.cls.append(c)
            self.dim.append(d_new)
            self.grp.append(stage)
            self.card.append(F.normalize(fn[m].mean(0), dim=0))
            self.mu.append(f[m].mean(0))
            self.var.append(f[m].var(0, unbiased=False))
            self.run.append([0, 0.0, 0.0])
            self.latest.append([0.0, 1.0])
            new_full.append(f[m].mean(0))
        new_full = torch.stack(new_full)
        self.full = new_full if self.full is None else torch.cat([self.full, new_full])
        # R4 / R5: how each card scores on this stage's pictures that are not its own class
        S = self.own_cos(f)
        cls = torch.tensor(self.cls, device=dev)
        stranger = y[:, None] != cls[None, :]
        n = stranger.sum(0).double()
        s1 = (S.double() * stranger).sum(0)
        s2 = (S.double().pow(2) * stranger).sum(0)
        for i in range(len(self.cls)):
            self.run[i][0] += n[i].item()
            self.run[i][1] += s1[i].item()
            self.run[i][2] += s2[i].item()
            mean = s1[i].item() / n[i].item()
            self.latest[i] = [mean, max(s2[i].item() / n[i].item() - mean ** 2, 1e-12) ** 0.5]
        self.fit_seconds += time.time() - t0

    def scores(self, f, rule):
        """Match score of every picture against every card under one rule. Highest wins."""
        if rule == "R7":
            return F.normalize(f, dim=1) @ F.normalize(self.full, dim=1).T
        if rule == "R6":
            out = torch.empty(len(f), len(self.cls), device=dev)
            for d in sorted(set(self.dim)):
                cols = [i for i, dd in enumerate(self.dim) if dd == d]
                mu = torch.stack([self.mu[i] for i in cols])
                var = torch.stack([self.var[i] for i in cols])
                var = var + VAR_FLOOR * var.mean(1, keepdim=True) + 1e-8
                for a in range(0, len(f), 2000):
                    x = f[a:a + 2000, None, :d]
                    out[a:a + 2000, cols] = -((x - mu[None]).pow(2) / var[None] + var.log()[None]).mean(2)
            return out
        S = self.own_cos(f)
        if rule == "R0":
            return S
        if rule == "R1":      # missing units count as zeros: cosine at full length
            norm_full = f.norm(dim=1, keepdim=True).clamp_min(1e-12)
            for d in sorted(set(self.dim)):
                cols = [i for i, dd in enumerate(self.dim) if dd == d]
                S[:, cols] = S[:, cols] * (f[:, :d].norm(dim=1, keepdim=True) / norm_full)
            return S
        if rule in ("R2", "R3"):
            for g in sorted(set(self.grp)):
                cols = [i for i, gg in enumerate(self.grp) if gg == g]
                part = S[:, cols] - S[:, cols].mean(1, keepdim=True)
                if rule == "R3":
                    part = part / part.std(1, keepdim=True).clamp_min(1e-6)
                S[:, cols] = part
            return S
        if rule == "R4":
            mean = torch.tensor([r[1] / r[0] for r in self.run], device=dev)
            sd = torch.tensor([max(r[2] / r[0] - (r[1] / r[0]) ** 2, 1e-12) ** 0.5 for r in self.run], device=dev)
            return (S - mean) / sd
        if rule == "R5":
            st = torch.tensor(self.latest, device=dev)
            return (S - st[:, 0]) / st[:, 1]
        raise ValueError(rule)


def report(pred, y, groups, masks):
    """Accuracy overall and per group, and where wrong answers go, for each split of the test set."""
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


def open_cards_predict(f_tr, y_tr, f_te, classes):
    """Nearest card when all cards are written from all training pictures at full length."""
    fn = F.normalize(f_tr, dim=1)
    cards = torch.stack([F.normalize(fn[y_tr == c].mean(0), dim=0) for c in classes])
    return torch.tensor(classes, device=dev)[(F.normalize(f_te, dim=1) @ cards.T).argmax(1)]


# ----------------------------------------------------------------------------- one run
def run(name, seed, data, log, preds):
    cfg = CONFIGS[name]
    xtr, ytr, xte, yte, practice = data
    colour = xtr.shape[1] == 3
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if name == "cifar100":
        surprise = sorted(np.random.default_rng(SURPRISE_SEED).permutation(100)[:20].tolist())
        rest = rng.permutation([c for c in range(100) if c not in surprise]).tolist()
    else:
        rest = list(range(20))
    groups = [rest[i * cfg["per_stage"]:(i + 1) * cfg["per_stage"]] for i in range(cfg["stages"])]
    final_width = cfg["start"] + cfg["add"] * (cfg["stages"] - 1)
    if SMOKE and name == "cifar100":
        final_width = 144

    def sel(x, y, classes, extra=None):
        m = torch.isin(y, torch.tensor(classes, device=dev))
        return (x[m], y[m]) if extra is None else (x[m], y[m], extra[m])

    def write(way, rule, stage, rep, **kw):
        log.write(json.dumps(dict(dataset=name, seed=seed, way=way, rule=rule, stage=stage, **rep, **kw)) + "\n")
        log.flush()

    frozen = GrowNet(xtr.shape[1]).grow(final_width)
    grow = GrowNet(xtr.shape[1])
    mem, frozen_mem = Memory(), Memory()
    train_s = {"frozen": 0.0, "grow": 0.0, "retrain": 0.0}

    for s, group in enumerate(groups):
        seen = [c for g in groups[:s + 1] for c in g]
        gx, gy = sel(xtr, ytr, group)
        tx, ty, tp = sel(xte, yte, seen, practice)
        masks = {"full": torch.ones_like(tp), "practice": tp, "exam": ~tp}
        last = s == len(groups) - 1

        if s == 0:
            train_s["frozen"] = train_stage(frozen, gx, gy, group, colour)
        frozen_mem.add_stage(features(frozen, gx), gy, group, s)
        pred = torch.tensor(frozen_mem.cls, device=dev)[frozen_mem.scores(features(frozen, tx), "R0").argmax(1)]
        write("frozen", "R0", s + 1, report(pred, ty, groups[:s + 1], masks), train_seconds=train_s["frozen"])
        frozen_acc = (pred == ty).float().mean().item()

        grow.grow(cfg["start"] if s == 0 else cfg["add"])
        t = train_stage(grow, gx, gy, group, colour)
        train_s["grow"] += t
        mem.add_stage(features(grow, gx), gy, group, s)   # the stage's pictures are not used again
        f_te = features(grow, tx)
        cls = torch.tensor(mem.cls, device=dev)
        line = []
        for rule in RULES:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            pred = cls[mem.scores(f_te, rule).argmax(1)]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            rep = report(pred, ty, groups[:s + 1], masks)
            write("grow", rule, s + 1, rep, train_seconds=round(t, 2), units_per_layer=grow.width,
                  feature_dim=f_te.shape[1], score_ms=round(1000 * (time.time() - t0), 2),
                  memory_fit_seconds=round(mem.fit_seconds, 3))
            line.append(f"{rule} {rep['full']['acc_all']:.3f}")
            if last:
                preds[f"s{seed}_{rule}"] = pred.cpu().numpy().astype(np.int16)

        ax, ay = sel(xtr, ytr, seen)                      # all pictures so far: open-book and retrain only
        pred = open_cards_predict(features(grow, ax), ay, f_te, seen)
        rep = report(pred, ty, groups[:s + 1], masks)
        write("grow", "open", s + 1, rep)
        msg = f"{name} seed {seed} stage {s + 1} frozen {frozen_acc:.3f} | grow " + " ".join(line) + \
              f" open {rep['full']['acc_all']:.3f}"

        if last:
            retrain = GrowNet(xtr.shape[1]).grow(final_width)
            train_s["retrain"] = train_stage(retrain, ax, ay, seen, colour)
            pred = open_cards_predict(features(retrain, ax), ay, features(retrain, tx), seen)
            rep = report(pred, ty, groups, masks)
            write("retrain", "open", s + 1, rep, train_seconds=train_s["retrain"])
            msg += f" | retrain {rep['full']['acc_all']:.3f}"
            preds[f"s{seed}_true"] = ty.cpu().numpy().astype(np.int16)
            preds[f"s{seed}_practice"] = tp.cpu().numpy()
            preds[f"s{seed}_groups"] = np.array(groups, dtype=np.int16)
            del retrain
        print(msg, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "none"
    print("device", dev, gpu, "torch", torch.__version__, "smoke", SMOKE, flush=True)
    if dev.type != "cuda" and not SMOKE:
        raise SystemExit("No GPU: refusing to start the full run on CPU.")
    if dev.type == "cuda":
        nn.Conv2d(3, 8, 3).to(dev)(torch.zeros(2, 3, 8, 8, device=dev)).sum().item()
        torch.backends.cudnn.benchmark = True
    t0 = time.time()
    with open(OUT / "runs.jsonl", "w") as log:
        for name in CONFIGS:
            data = load(name)
            preds = {}
            for seed in range(SEEDS):
                run(name, seed, data, log, preds)
                print(f"== {name} seed {seed} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
            np.savez_compressed(OUT / f"preds_{name}.npz", **preds)
            del data
    meta = dict(gpu=gpu, torch=torch.__version__, on_kaggle=ON_KAGGLE, smoke=SMOKE, seeds=SEEDS, epochs=EPOCHS,
                batch=BATCH, lr=LR, rules=RULES, var_floor=VAR_FLOOR, ridge=RIDGE, split_seed=SPLIT_SEED,
                configs=CONFIGS, total_minutes=round((time.time() - t0) / 60, 1),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
