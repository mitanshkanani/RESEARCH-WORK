"""Shared paths and settings for EXP-2. Everything is kept inside the experiment folder."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"          # raw dataset downloads
WEIGHTS = ROOT / "weights"    # pretrained and trained feature networks
CACHE = ROOT / "cache"        # prepared splits and saved features
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Keep every download inside the experiment folder (must be set before importing torch hub / HF).
os.environ["TORCH_HOME"] = str(WEIGHTS / "torch_home")
os.environ["HF_HOME"] = str(WEIGHTS / "hf_home")

for p in (DATA, WEIGHTS, CACHE, RESULTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

DATASETS = ("cifar100", "omniglot", "mnist_fashion")
SHOTS = (1, 2, 3, 5, 10)
SPLIT_SEED = 0  # fixes which classes are used to build features and which are tested


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
