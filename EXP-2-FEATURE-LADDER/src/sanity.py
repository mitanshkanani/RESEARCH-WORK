"""Sanity check: reproduce a published number with this code before trusting anything new.

Published (Janson et al., arXiv 2210.04428): a nearest-class-mean classifier on frozen
ViT-B/16 (ImageNet-21k) features reaches 83.70% on CIFAR-100 with no stored examples.

Here: all 100 CIFAR-100 classes, full training set, the NCM learner from learners.py.
Run:  python src/sanity.py
"""
import json

import common  # noqa: F401
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

from common import CACHE, DATA, RESULTS, device
from learners import NCM

MODEL = "vit_base_patch16_224.augreg_in21k"


@torch.no_grad()
def main():
    dev = device()
    path = CACHE / "sanity_vitb16_in21k_cifar100.pt"
    if path.exists():
        blob = torch.load(path)
    else:
        net = timm.create_model(MODEL, pretrained=True, num_classes=0).to(dev).eval()
        cfg = timm.data.resolve_model_data_config(net)
        mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
        print("model", MODEL, "| input", cfg["input_size"], "| mean", cfg["mean"], "| std", cfg["std"])
        blob = {}
        for part, train in (("train", True), ("test", False)):
            ds = CIFAR100(DATA, train=train, download=False)
            x = torch.from_numpy(ds.data).permute(0, 3, 1, 2)
            feats = []
            for i in range(0, len(x), 64):
                b = x[i:i + 64].to(dev).float() / 255.0
                b = F.interpolate(b, size=cfg["input_size"][-1], mode="bicubic", align_corners=False)
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    feats.append(net((b - mean) / std).float().cpu())
            blob[f"{part}_f"] = torch.cat(feats)
            blob[f"{part}_y"] = torch.tensor(ds.targets)
        torch.save(blob, path)

    tr_f, tr_y = blob["train_f"].to(dev), blob["train_y"].to(dev)
    te_f, te_y = blob["test_f"].to(dev), blob["test_y"].to(dev)
    out = {"published_ncm_accuracy": 83.70, "model": MODEL}

    learner = NCM(tr_f.shape[1], 100, dev)
    for c in np.random.default_rng(0).permutation(100):     # classes arrive one at a time
        learner.learn(tr_f[tr_y == int(c)], int(c))
    out["ours_ncm_unit_length"] = round(100 * (learner.scores(te_f).argmax(1) == te_y).float().mean().item(), 2)

    means = torch.stack([tr_f[tr_y == c].mean(0) for c in range(100)])   # plain means, no scaling
    out["ours_ncm_plain"] = round(100 * (torch.cdist(te_f, means).argmin(1) == te_y).float().mean().item(), 2)

    print(json.dumps(out, indent=1))
    (RESULTS / "sanity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
