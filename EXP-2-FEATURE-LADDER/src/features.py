"""Build the six feature sources of EXP-2 and save their outputs to cache/.

  A  raw pixels                                             no training
  B  random sparse expansion of the pixels (2048 units)     no training
  C  small conv network, random weights, never trained      no training
  D  same network, trained WITHOUT labels on the base data  (contrastive, SimCLR-style)
  E  same network, trained WITH labels on the base classes
  F  ResNet-18 pretrained on ImageNet, frozen               trained by someone else

The base data never contains a test class (see data.py).
Run:  python src/features.py [dataset ...] [--sources A B ...]
"""
import argparse
import json
import time

import common  # noqa: F401  (sets download folders before torch is imported)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CACHE, DATASETS, WEIGHTS, device
from data import load_split

SOURCES = ("A", "B", "C", "D", "E", "F")
POOL_PER_CLASS = 100     # shots are drawn from at most this many images per class
BASE_KEEP = 10000        # base images kept for the learners' statistics
TRAIN_STEPS = 3000


def to_tensor(x_uint8):
    """(N, H, W, C) uint8 -> (N, C, H, W) float in [0, 1]"""
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float() / 255.0


class Conv4(nn.Module):
    """Four blocks of conv(64) - batchnorm - ReLU - maxpool. 28x28 -> 64 features, 32x32 -> 256."""

    def __init__(self, in_ch):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(4):
            layers += [nn.Conv2d(c, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)]
            c = 64
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body((x - 0.5) / 0.5).flatten(1)


def augment(x, colour):
    """Random zoom, shift, small rotation (or flip for colour photos), brightness and contrast."""
    n, dev = len(x), x.device
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


def train_conv4(base_x, base_y, mode, colour, dev, seed=0):
    """mode 'ssl': contrastive learning without labels. mode 'sup': ordinary classification."""
    torch.manual_seed(seed)
    net = Conv4(base_x.shape[1]).to(dev)
    d = net(base_x[:2].to(dev)).shape[1]
    if mode == "ssl":
        head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 64)).to(dev)
        bs = 512
    else:
        head = nn.Linear(d, int(base_y.max()) + 1).to(dev)
        bs = 256
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS)
    t0, last = time.time(), 0.0
    for step in range(TRAIN_STEPS):
        idx = torch.randint(len(base_x), (bs,))
        x = base_x[idx].to(dev)
        if mode == "ssl":
            z = F.normalize(head(net(torch.cat([augment(x, colour), augment(x, colour)]))), dim=1)
            sim = z @ z.T / 0.5
            sim.fill_diagonal_(-1e9)
            target = torch.cat([torch.arange(bs, 2 * bs), torch.arange(bs)]).to(dev)
            loss = F.cross_entropy(sim, target)
        else:
            loss = F.cross_entropy(head(net(augment(x, colour))), base_y[idx].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        last = 0.98 * last + 0.02 * loss.item() if step else loss.item()
    return net.eval(), dict(mode=mode, steps=TRAIN_STEPS, batch=bs, final_loss=round(last, 4),
                            train_seconds=round(time.time() - t0, 1), base_images=len(base_x))


def calibrated_random_conv4(base_x, dev, seed=0):
    """Random weights that are never trained. Only the batch-norm running averages are set, by
    passing base images through once (no weight is changed)."""
    torch.manual_seed(seed)
    net = Conv4(base_x.shape[1]).to(dev).train()
    with torch.no_grad():
        for i in range(0, min(len(base_x), 5120), 512):
            net(base_x[i:i + 512].to(dev))
    return net.eval()


def sparse_expansion(d, m=2048, frac_in=0.125, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.rand(m, d, generator=g).topk(max(2, round(frac_in * d)), dim=1).indices
    return torch.zeros(m, d).scatter_(1, idx, 1.0)


@torch.no_grad()
def extract(fn, x, dev, bs=256):
    return torch.cat([fn(x[i:i + bs].to(dev)).float().cpu() for i in range(0, len(x), bs)])


def build(name, sources, dev):
    s = load_split(name)
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.permutation(np.flatnonzero(s["pool_y"] == c))[:POOL_PER_CLASS]
                           for c in np.unique(s["pool_y"])])
    bkeep = rng.permutation(len(s["base_x"]))[:BASE_KEEP]
    parts = dict(pool=(s["pool_x"][keep], s["pool_y"][keep]), test=(s["test_x"], s["test_y"]),
                 base=(s["base_x"][bkeep], s["base_y"][bkeep]))
    base_x, base_y = to_tensor(s["base_x"]), torch.from_numpy(s["base_y"]).long()
    colour = base_x.shape[1] == 3
    info_path = CACHE / "feature_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    for src in sources:
        out = CACHE / f"feat_{name}_{src}.pt"
        if out.exists():
            print(f"{name} {src}: already built")
            continue
        t0, meta = time.time(), {}
        if src == "A":
            fn = lambda x: x.flatten(1)
        elif src == "B":
            theta = sparse_expansion(base_x[0].numel()).to(dev)
            def fn(x):
                phi = x.flatten(1) @ theta.T
                top = phi.topk(round(0.05 * phi.shape[1]), dim=1)
                return torch.zeros_like(phi).scatter_(1, top.indices, top.values)
        elif src == "C":
            fn = calibrated_random_conv4(base_x, dev)
        elif src in ("D", "E"):
            wpath = WEIGHTS / f"conv4_{name}_{src}.pt"
            net, meta = train_conv4(base_x, base_y, "ssl" if src == "D" else "sup", colour, dev)
            torch.save(net.state_dict(), wpath)
            fn = net
        elif src == "F":
            from torchvision.models import ResNet18_Weights, resnet18
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()
            net = net.to(dev).eval()
            mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
            def fn(x):
                x = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
                x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    return net((x - mean) / std)
        blob = {}
        for part, (px, py) in parts.items():
            blob[f"{part}_f"] = extract(fn, to_tensor(px), dev)
            blob[f"{part}_y"] = torch.from_numpy(py).long()
        torch.save(blob, out)
        meta.update(dim=blob["pool_f"].shape[1], build_seconds=round(time.time() - t0, 1))
        info[f"{name}_{src}"] = meta
        info_path.write_text(json.dumps(info, indent=1))
        print(f"{name} {src}: {meta}")
        del fn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    a = ap.parse_args()
    for ds in a.datasets:
        build(ds, a.sources, device())
