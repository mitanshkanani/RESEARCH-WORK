"""EXP-5 better eyes: can the units added at each stage learn more general features?

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

The network, grow and lock, and card rule R7 are exactly those of EXP-4. What changes is the job
given to the newly added units while they train:
  T0  tell this stage's classes apart (as before)
  T1  pictures also shown turned by 90/180/270 degrees; name the class AND the turn (4x answers)
  T2  the same four copies, but only the class is asked (control: same data, no extra job)
  T3  two altered copies of a picture must get similar descriptions (label-free), plus the class job
Each way is run with 30, 10, 3 and 1 passes over the pictures of a stage.
Reference points: `frozen` (trained on the first group only, 30 passes) and `retrain` (a fresh
network trained on all classes at once, plain training, at each number of passes).

Run:  python src/exp5.py              full run (5 seeds)
      EXP5_SMOKE=1 python src/exp5.py   tiny run to check the code
Writes runs.jsonl, meta.json and preds.npz into the output folder.
"""
import json
import math
import os
import pathlib
import subprocess
import threading
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
OUT = pathlib.Path(os.environ.get("EXP5_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP5_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP5_SMOKE") == "1"

SEEDS = 1 if SMOKE else 5
PASSES = [1] if SMOKE else [30, 10, 3, 1]
MAIN_PASSES = PASSES[0]
WAYS = ["T0", "T1", "T2", "T3"]
BATCH = 128
LR = 1e-3
STAGES, PER_STAGE, START, ADD = (3 if SMOKE else 8), 10, 32, 16
SURPRISE_SEED = 1234                 # the same 20 held-out classes as EXP-3
SPLIT_SEED = 4321                    # practice half / exam half of the test pictures, as EXP-4
RIDGE = 1e-2                         # R7, as EXP-4
SHOTS = (1, 3, 10)
EPISODES = 20
T3_WEIGHT, T3_TEMP, T3_DIM = 1.0, 0.5, 64

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ----------------------------------------------------------------------------- data (as EXP-4)
def load():
    import torchvision.datasets as tv
    DATA.mkdir(parents=True, exist_ok=True)
    tr, te = tv.CIFAR100(DATA, train=True, download=True), tv.CIFAR100(DATA, train=False, download=True)
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
    return dict(seconds=t1 - t0, wh=meter.wh(t0, t1),
                peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0.0)


# ----------------------------------------------------------------------------- memory: R0 and R7, as EXP-4
class Memory:
    """What the system keeps about learned classes. No pictures are stored."""

    def __init__(self):
        self.cls, self.dim, self.card = [], [], []
        self.full = None                      # R7: raw-mean cards at full current length

    def add_stage(self, f, y, classes):
        d_new = f.shape[1]
        if self.full is not None:             # R7: predict the new units of old cards from their old units
            d_old = self.full.shape[1]
            X, Y = f[:, :d_old].double(), f[:, d_old:].double()
            xm, ym = X.mean(0), Y.mean(0)
            Xc, Yc = X - xm, Y - ym
            lam = RIDGE * Xc.pow(2).sum() / len(X) / d_old * len(X)
            W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(d_old, device=dev, dtype=torch.float64), Xc.T @ Yc)
            self.full = torch.cat([self.full, ((self.full.double() - xm) @ W + ym).float()], 1)
        fn = F.normalize(f, dim=1)
        new_full = []
        for c in classes:
            m = y == c
            self.cls.append(c)
            self.dim.append(d_new)
            self.card.append(F.normalize(fn[m].mean(0), dim=0))
            new_full.append(f[m].mean(0))
        new_full = torch.stack(new_full)
        self.full = new_full if self.full is None else torch.cat([self.full, new_full])

    def scores(self, f, rule):
        if rule == "R7":
            return F.normalize(f, dim=1) @ F.normalize(self.full, dim=1).T
        out = torch.empty(len(f), len(self.cls), device=dev)   # R0: each card on its own units
        for d in sorted(set(self.dim)):
            cols = [i for i, dd in enumerate(self.dim) if dd == d]
            out[:, cols] = F.normalize(f[:, :d], dim=1) @ torch.stack([self.card[i] for i in cols]).T
        return out


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


def own_cards_only(scores, cls, y, group, masks):
    """M4: first group's pictures, answers restricted to the first group's own cards."""
    m = torch.isin(y, torch.tensor(group, device=dev))
    k = len(group)
    ok = cls[:k][scores[:, :k].argmax(1)] == y
    return {name: round(ok[m & mm].float().mean().item(), 4) for name, mm in masks.items()}


def open_cards(f_tr, y_tr, classes):
    fn = F.normalize(f_tr, dim=1)
    return torch.stack([F.normalize(fn[y_tr == c].mean(0), dim=0) for c in classes])


def few_shot(f_pool, y_pool, f_te, y_te, classes, rng):
    """Learn the surprise classes from k pictures each (nearest card), averaged over random draws."""
    f_pool, f_te = F.normalize(f_pool, dim=1), F.normalize(f_te, dim=1)
    idx = {c: torch.nonzero(y_pool == c).squeeze(1).cpu().numpy() for c in classes}
    cls = torch.tensor(classes, device=dev)
    out = {}
    for k in SHOTS:
        accs = []
        for _ in range(EPISODES):
            cards = torch.stack([F.normalize(f_pool[torch.from_numpy(rng.choice(idx[c], k, replace=False)).to(dev)]
                                             .mean(0), dim=0) for c in classes])
            accs.append((cls[(f_te @ cards.T).argmax(1)] == y_te).float().mean().item())
        out[str(k)] = round(float(np.mean(accs)), 4)
    return out


# ----------------------------------------------------------------------------- one seed
def run(seed, data, meter, log, preds):
    xtr, ytr, xte, yte, practice = data
    rng = np.random.default_rng(seed)
    surprise = sorted(np.random.default_rng(SURPRISE_SEED).permutation(100)[:20].tolist())
    rest = rng.permutation([c for c in range(100) if c not in surprise]).tolist()
    groups = [rest[i * PER_STAGE:(i + 1) * PER_STAGE] for i in range(STAGES)]
    final_width = START + ADD * (STAGES - 1)
    allc = [c for g in groups for c in g]

    def sel(x, y, classes, extra=None):
        m = torch.isin(y, torch.tensor(classes, device=dev))
        return (x[m], y[m]) if extra is None else (x[m], y[m], extra[m])

    def write(**kw):
        log.write(json.dumps(dict(seed=seed, **kw)) + "\n")
        log.flush()

    def cost(c):
        return dict(train_seconds=round(c["seconds"], 2), energy_wh=None if c["wh"] is None else round(c["wh"], 4),
                    peak_mem_mb=round(c["peak_mb"], 1))

    sx_pool, sy_pool = sel(xtr, ytr, surprise)
    sx_te, sy_te = sel(xte, yte, surprise)
    ax, ay = sel(xtr, ytr, allc)                    # all training pictures: open-book and retrain only
    tx_all, ty_all, tp_all = sel(xte, yte, allc, practice)
    masks_all = {"full": torch.ones_like(tp_all), "practice": tp_all, "exam": ~tp_all}

    def fs(net):
        return few_shot(features(net, sx_pool), sy_pool, features(net, sx_te), sy_te, surprise,
                        np.random.default_rng(10_000 + seed))

    def whole(net, way, passes, c):
        """Score a network whose cards are all written from all pictures (frozen eyes, retrain)."""
        cards = open_cards(features(net, ax), ay, allc)
        S = F.normalize(features(net, tx_all), dim=1) @ cards.T
        cls = torch.tensor(allc, device=dev)
        rep = report(cls[S.argmax(1)], ty_all, groups, masks_all)
        write(way=way, passes=passes, rule="open", stage=STAGES, **rep, **cost(c), few_shot=fs(net),
              first_own=own_cards_only(S, cls, ty_all, groups[0], masks_all))
        return rep["full"]["acc_all"]

    # frozen eyes: full-size network trained on the first group only, then never again
    torch.manual_seed(seed)
    net = GrowNet(3).grow(final_width)
    gx, gy = sel(xtr, ytr, groups[0])
    acc = whole(net, "frozen", MAIN_PASSES, train_stage(net, gx, gy, groups[0], "T0", MAIN_PASSES, meter))
    print(f"seed {seed} frozen {acc:.3f}", flush=True)
    del net

    # retrain everything: fresh full-size network, all classes at once, plain training
    for pi, passes in enumerate(PASSES):
        torch.manual_seed(seed * 1000 + 900 + pi)
        net = GrowNet(3).grow(final_width)
        acc = whole(net, "retrain", passes, train_stage(net, ax, ay, allc, "T0", passes, meter))
        print(f"seed {seed} retrain passes {passes} {acc:.3f}", flush=True)
        del net

    # grow and lock under each of the four jobs
    for wi, way in enumerate(WAYS):
        for pi, passes in enumerate(PASSES):
            torch.manual_seed(seed * 1000 + wi * 10 + pi)
            net, mem = GrowNet(3), Memory()
            tot = dict(seconds=0.0, wh=0.0 if meter.ok else None, peak_mb=0.0)
            first_own_start = None
            for s, group in enumerate(groups):
                seen = [c for g in groups[:s + 1] for c in g]
                gx, gy = sel(xtr, ytr, group)
                tx, ty, tp = sel(xte, yte, seen, practice)
                masks = {"full": torch.ones_like(tp), "practice": tp, "exam": ~tp}
                last = s == STAGES - 1
                net.grow(START if s == 0 else ADD)
                c = train_stage(net, gx, gy, group, way, passes, meter)
                tot["seconds"] += c["seconds"]
                tot["peak_mb"] = max(tot["peak_mb"], c["peak_mb"])
                if tot["wh"] is not None and c["wh"] is not None:
                    tot["wh"] += c["wh"]
                mem.add_stage(features(net, gx), gy, group)      # the stage's pictures are not used again
                f_te = features(net, tx)
                cls = torch.tensor(mem.cls, device=dev)
                line = []
                for rule in ("R0", "R7"):
                    S = mem.scores(f_te, rule)
                    pred = cls[S.argmax(1)]
                    rep = report(pred, ty, groups[:s + 1], masks)
                    own = own_cards_only(S, cls, ty, groups[0], masks)
                    write(way=way, passes=passes, rule=rule, stage=s + 1, **rep, **cost(tot), first_own=own,
                          units_per_layer=net.width, feature_dim=f_te.shape[1])
                    line.append(f"{rule} {rep['full']['acc_all']:.3f} own {own['full']:.3f}")
                    if last and passes == MAIN_PASSES:
                        preds[f"s{seed}_{way}_{rule}"] = pred.cpu().numpy().astype(np.int16)
                sa_x, sa_y = sel(xtr, ytr, seen)                  # open-book: diagnostic only
                S = F.normalize(f_te, dim=1) @ open_cards(features(net, sa_x), sa_y, seen).T
                seen_t = torch.tensor(seen, device=dev)
                rep = report(seen_t[S.argmax(1)], ty, groups[:s + 1], masks)
                kw = dict(few_shot=fs(net)) if last else {}
                write(way=way, passes=passes, rule="open", stage=s + 1, **rep,
                      first_own=own_cards_only(S, seen_t, ty, groups[0], masks), **kw)
                if last or passes == MAIN_PASSES:
                    print(f"seed {seed} {way} passes {passes} stage {s + 1} " + " ".join(line) +
                          f" open {rep['full']['acc_all']:.3f}" + (f" few3 {kw['few_shot']['3']:.3f}" if last else ""),
                          flush=True)
            del net, mem
    preds[f"s{seed}_true"] = ty_all.cpu().numpy().astype(np.int16)
    preds[f"s{seed}_practice"] = tp_all.cpu().numpy()
    preds[f"s{seed}_groups"] = np.array(groups, dtype=np.int16)


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
    preds = {}
    with open(OUT / "runs.jsonl", "w") as log:
        for seed in range(SEEDS):
            run(seed, data, meter, log, preds)
            np.savez_compressed(OUT / "preds.npz", **preds)
            print(f"== seed {seed} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    meter.stop()
    meta = dict(gpu=gpu, torch=torch.__version__, on_kaggle=ON_KAGGLE, smoke=SMOKE, seeds=SEEDS, passes=PASSES,
                ways=WAYS, batch=BATCH, lr=LR, stages=STAGES, per_stage=PER_STAGE, start=START, add=ADD,
                ridge=RIDGE, split_seed=SPLIT_SEED, shots=SHOTS, episodes=EPISODES, t3_weight=T3_WEIGHT,
                t3_temp=T3_TEMP, t3_dim=T3_DIM, power_reported=meter.ok,
                total_minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
