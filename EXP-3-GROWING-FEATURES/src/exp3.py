"""EXP-3 growing features: can the feature extractor keep learning without forgetting?

One file so it can run as a Kaggle script or on the laptop. The plan is in JOURNAL.md (Part A).

Four ways of handling the feature extractor ("eyes") while groups of classes arrive in stages:
  frozen    trained on stage 1 only, never changed
  ordinary  the same network keeps training on each new stage
  grow      new units are added each stage, only they are trained, old units are locked
  retrain   a fresh network is trained on every picture seen so far, at every stage

All four use the same memory: one average feature vector ("memory card") per class, nearest card
wins. Cards are written when a class is learned and never rewritten (except in `retrain`, which
keeps all old pictures by definition).

Run:  python src/exp3.py            full run (5 seeds)
      EXP3_SMOKE=1 python src/exp3.py   tiny run to check the code
Writes runs.jsonl (one line per dataset, seed, way, stage) and meta.json into the output folder.
"""
import copy
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
OUT = pathlib.Path(os.environ.get("EXP3_OUT", "/kaggle/working" if ON_KAGGLE else ROOT / "results"))
DATA = pathlib.Path(os.environ.get("EXP3_DATA", "/kaggle/temp/data" if ON_KAGGLE else ROOT / "data"))
SMOKE = os.environ.get("EXP3_SMOKE") == "1"

SEEDS = 1 if SMOKE else 5
EPOCHS = 1 if SMOKE else 30          # every picture of a stage is shown this many times
BATCH = 128
LR = 1e-3
SHOTS = (1, 3, 10)                   # pictures per surprise class in the few-shot test
EPISODES = 20                        # random draws of those pictures, averaged
SURPRISE_SEED = 1234                 # the 20 surprise classes are the same in every seed

CONFIGS = {
    # name: stages, classes per stage, units at start, units added per stage
    "digits_clothes": dict(stages=2, per_stage=10, start=32, add=32),
    "cifar100": dict(stages=3 if SMOKE else 8, per_stage=10, start=32, add=16),
}

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- power
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

    def watts(self, t0, t1):
        w = [p for t, p in self.samples if t0 <= t <= t1]
        return float(np.mean(w)) if w else None

    def stop(self):
        if self.proc:
            self.proc.terminate()


# ----------------------------------------------------------------------------- data
def load(name):
    """Returns train x, train y, test x, test y as uint8 / long tensors on the device."""
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
    return xtr.to(dev), ytr.long().to(dev), xte.to(dev), yte.long().to(dev)


def augment(x, colour):
    """Random zoom, shift, small rotation (or flip for colour photos), brightness and contrast.
    Same as EXP-2."""
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


# ----------------------------------------------------------------------------- network
class Block(nn.Module):
    """One group of units in one layer. It reads only the first `cin` channels below it, which
    are the units that existed when this group was created."""

    def __init__(self, cin, cout):
        super().__init__()
        self.cin = cin
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x[:, :self.cin]))


class GrowNet(nn.Module):
    """Four layers of conv - batchnorm - ReLU - maxpool (the EXP-2 design). `grow(n)` locks every
    existing unit and adds n new units to each layer. New units read old and new units of the
    layer below; old units never read new ones, so their output cannot change."""

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
        for blocks in self.layers:          # locked groups also keep their batch-norm averages fixed
            for b in blocks[:-1]:
                b.eval()
        return self

    def forward(self, x):
        x = (x - 0.5) / 0.5
        for blocks in self.layers:
            x = F.max_pool2d(F.relu(torch.cat([b(x) for b in blocks], 1)), 2)
        return x.flatten(1)


def n_params(net, trainable=False):
    return sum(p.numel() for p in net.parameters() if p.requires_grad or not trainable)


@torch.no_grad()
def features(net, x_u8, bs=1000):
    net.eval()
    return torch.cat([net(x_u8[i:i + bs].float() / 255) for i in range(0, len(x_u8), bs)])


def train_stage(net, x_u8, y, classes, colour, meter):
    """Train the unlocked part of `net` to tell `classes` apart. The linear layer used for this is
    thrown away afterwards; only the features are kept."""
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
        torch.cuda.reset_peak_memory_stats()
    t0, loss = time.time(), None
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
    t1 = time.time()
    watts = meter.watts(t0, t1)
    return dict(train_seconds=round(t1 - t0, 2), steps=EPOCHS * steps_per_epoch, final_loss=round(loss.item(), 4),
                mean_watts=None if watts is None else round(watts, 1),
                energy_wh=None if watts is None else round(watts * (t1 - t0) / 3600, 4),
                peak_mem_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if dev.type == "cuda" else None,
                params_total=n_params(net), params_trained=n_params(net, trainable=True))


# ----------------------------------------------------------------------------- memory cards
def make_cards(f, y, classes, dim):
    """One card per class: the average of the unit-length features, using the first `dim` numbers."""
    f = F.normalize(f[:, :dim], dim=1)
    return [(c, dim, F.normalize(f[y == c].mean(0), dim=0)) for c in classes]


def predict(f, cards):
    """Nearest card by cosine similarity. A card that covers fewer units is compared on those units only."""
    cls = torch.tensor([c for c, _, _ in cards], device=dev)
    sims = torch.empty(len(f), len(cards), device=dev)
    for d in sorted({d for _, d, _ in cards}):
        cols = [i for i, (_, dd, _) in enumerate(cards) if dd == d]
        sims[:, cols] = F.normalize(f[:, :d], dim=1) @ torch.stack([cards[i][2] for i in cols]).T
    return cls[sims.argmax(1)]


def score(f_te, y_te, cards, groups):
    pred = predict(f_te, cards)
    per_group = [round((pred[torch.isin(y_te, torch.tensor(g, device=dev))] ==
                        y_te[torch.isin(y_te, torch.tensor(g, device=dev))]).float().mean().item(), 4) for g in groups]
    return per_group, round((pred == y_te).float().mean().item(), 4)


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


# ----------------------------------------------------------------------------- one run
def run(name, seed, data, meter, log):
    cfg = CONFIGS[name]
    xtr, ytr, xte, yte = data
    colour = xtr.shape[1] == 3
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if name == "cifar100":
        surprise = sorted(np.random.default_rng(SURPRISE_SEED).permutation(100)[:20].tolist())
        rest = rng.permutation([c for c in range(100) if c not in surprise]).tolist()
    else:
        surprise, rest = [], list(range(20))            # digits first, then clothes
    groups = [rest[i * cfg["per_stage"]:(i + 1) * cfg["per_stage"]] for i in range(cfg["stages"])]
    final_width = cfg["start"] + cfg["add"] * (cfg["stages"] - 1)
    if SMOKE and name == "cifar100":
        final_width = 144                                # keep the real size so timing is meaningful

    def sel(x, y, classes):
        m = torch.isin(y, torch.tensor(classes, device=dev))
        return x[m], y[m]

    if surprise:
        sx_pool, sy_pool = sel(xtr, ytr, surprise)
        sx_te, sy_te = sel(xte, yte, surprise)
    probe = xte[:256]

    ordinary = GrowNet(xtr.shape[1]).grow(final_width)
    grow = GrowNet(xtr.shape[1])
    frozen = None
    cards = {"frozen": [], "ordinary": [], "grow": []}

    for s, group in enumerate(groups):
        seen = [c for g in groups[:s + 1] for c in g]
        gx, gy = sel(xtr, ytr, group)                    # this stage's training pictures
        ax, ay = sel(xtr, ytr, seen)                     # every picture so far (retrain and open-book only)
        tx, ty = sel(xte, yte, seen)
        nets, costs, extra = {}, {}, {}

        costs["ordinary"] = train_stage(ordinary, gx, gy, group, colour, meter)
        nets["ordinary"] = ordinary
        if s == 0:
            frozen = copy.deepcopy(ordinary)
            costs["frozen"] = dict(costs["ordinary"])
        else:
            costs["frozen"] = dict(train_seconds=0.0, steps=0, energy_wh=0.0 if meter.ok else None,
                                   params_total=n_params(frozen), params_trained=0)
        nets["frozen"] = frozen

        old_dim = features(grow, probe).shape[1] if s else 0
        before = features(grow, probe) if s else None
        grow.grow(cfg["start"] if s == 0 else cfg["add"])
        costs["grow"] = train_stage(grow, gx, gy, group, colour, meter)
        nets["grow"] = grow
        if s:   # proof that locking works: old units give the same output as before this stage
            extra["grow"] = dict(lock_drift=float((features(grow, probe)[:, :old_dim] - before).abs().max().item()))

        if s == 0:
            nets["retrain"], costs["retrain"] = ordinary, dict(costs["ordinary"])
        else:
            nets["retrain"] = GrowNet(xtr.shape[1]).grow(final_width)
            costs["retrain"] = train_stage(nets["retrain"], ax, ay, seen, colour, meter)

        for way in ("frozen", "ordinary", "grow", "retrain"):
            net = nets[way]
            f_all = features(net, ax)                    # used for retrain's cards and the open-book check
            dim = f_all.shape[1]
            open_cards = make_cards(f_all, ay, seen, dim)
            if way == "retrain":
                use = open_cards
            else:                                        # write cards for the new classes only, keep old ones
                m = torch.isin(ay, torch.tensor(group, device=dev))
                cards[way] += make_cards(f_all[m], ay[m], group, dim)
                use = cards[way]
            f_te = features(net, tx)
            acc_groups, acc_all = score(f_te, ty, use, groups[:s + 1])
            open_groups, open_all = score(f_te, ty, open_cards, groups[:s + 1])
            rec = dict(dataset=name, seed=seed, way=way, stage=s + 1, classes_seen=len(seen),
                       units_per_layer=net.width, feature_dim=dim,
                       acc_groups=acc_groups, acc_all=acc_all, open_acc_groups=open_groups, open_acc_all=open_all,
                       **costs[way], **extra.get(way, {}))
            if surprise:
                rec["few_shot"] = few_shot(features(net, sx_pool), sy_pool, features(net, sx_te), sy_te, surprise,
                                           np.random.default_rng([seed, s]))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"{name} seed {seed} stage {s + 1} {way:9s} first {acc_groups[0]:.3f} all {acc_all:.3f} "
                  f"open {open_all:.3f} few3 {rec.get('few_shot', {}).get('3', float('nan')):.3f} "
                  f"train {costs[way]['train_seconds']:.0f}s", flush=True)
        if s:
            del nets["retrain"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "none"
    print("device", dev, gpu, "torch", torch.__version__, "smoke", SMOKE, flush=True)
    if dev.type != "cuda" and not SMOKE:
        raise SystemExit("No GPU: refusing to start the full run on CPU.")
    if dev.type == "cuda":                               # fail early if this GPU cannot run this PyTorch
        nn.Conv2d(3, 8, 3).to(dev)(torch.zeros(2, 3, 8, 8, device=dev)).sum().item()
        torch.backends.cudnn.benchmark = True
    meter = PowerMeter()
    t0 = time.time()
    with open(OUT / "runs.jsonl", "w") as log:
        for name in CONFIGS:
            data = load(name)
            for seed in range(SEEDS):
                run(name, seed, data, meter, log)
                print(f"== {name} seed {seed} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
            del data
    meter.stop()
    meta = dict(gpu=gpu, torch=torch.__version__, on_kaggle=ON_KAGGLE, smoke=SMOKE, seeds=SEEDS, epochs=EPOCHS,
                batch=BATCH, lr=LR, shots=SHOTS, episodes=EPISODES, configs=CONFIGS, power_reported=meter.ok,
                total_minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
