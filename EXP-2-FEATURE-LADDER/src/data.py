"""Prepare the fixed data splits for EXP-2.

Each dataset is cut into three parts that never share a class:
  base         images used only to BUILD features (feature sources D and E)
  novel_pool   images of the test classes from which the few "shots" are drawn
  novel_test   held-out images of the test classes, used only for scoring

Run:  python src/data.py          (downloads into data/, writes cache/splits_<name>.npz)
"""
import numpy as np

from common import CACHE, DATA, DATASETS, SPLIT_SEED


def _to_np(ds_images):
    return np.asarray(ds_images, dtype=np.uint8)


def prepare_cifar100():
    """20 classes build features, the other 80 are learned few-shot. Images 32x32x3."""
    from torchvision.datasets import CIFAR100
    tr = CIFAR100(DATA, train=True, download=True)
    te = CIFAR100(DATA, train=False, download=True)
    xtr, ytr = _to_np(tr.data), np.asarray(tr.targets)
    xte, yte = _to_np(te.data), np.asarray(te.targets)

    perm = np.random.default_rng(SPLIT_SEED).permutation(100)
    base_cls, novel_cls = perm[:20], perm[20:]
    relabel = {c: i for i, c in enumerate(novel_cls)}
    base_relabel = {c: i for i, c in enumerate(base_cls)}

    mb = np.isin(ytr, base_cls)
    mp = np.isin(ytr, novel_cls)
    mt = np.isin(yte, novel_cls)
    return dict(
        base_x=xtr[mb], base_y=np.array([base_relabel[c] for c in ytr[mb]]),
        pool_x=xtr[mp], pool_y=np.array([relabel[c] for c in ytr[mp]]),
        test_x=xte[mt], test_y=np.array([relabel[c] for c in yte[mt]]),
        base_classes=base_cls, novel_classes=novel_cls,
    )


def prepare_omniglot():
    """Background alphabets (964 characters) build features; the 659 evaluation characters
    are learned few-shot. Per test character: 10 images form the shot pool, 10 are for scoring.
    Images are resized to 28x28, single channel, ink = high values."""
    from PIL import Image
    from torchvision.datasets import Omniglot

    def load(background):
        ds = Omniglot(DATA, background=background, download=True)
        xs, ys = [], []
        for img, y in ds:
            a = 255 - np.asarray(img.convert("L").resize((28, 28), Image.BILINEAR), dtype=np.uint8)
            xs.append(a)
            ys.append(y)
        return np.stack(xs)[..., None], np.asarray(ys)

    bx, by = load(True)
    nx, ny = load(False)

    rng = np.random.default_rng(SPLIT_SEED)
    pool_idx, test_idx = [], []
    for c in np.unique(ny):
        idx = rng.permutation(np.flatnonzero(ny == c))
        pool_idx.extend(idx[:10])
        test_idx.extend(idx[10:])
    pool_idx, test_idx = np.array(pool_idx), np.array(test_idx)
    return dict(
        base_x=bx, base_y=by,
        pool_x=nx[pool_idx], pool_y=ny[pool_idx],
        test_x=nx[test_idx], test_y=ny[test_idx],
        base_classes=np.unique(by), novel_classes=np.unique(ny),
    )


def prepare_mnist_fashion():
    """The 'cats and dogs, then digits' case: the 10 MNIST digits arrive first (classes 0-9),
    then the 10 Fashion-MNIST items (classes 10-19). Features are built on KMNIST, an unrelated
    set of Japanese characters. Images 28x28x1."""
    from torchvision.datasets import KMNIST, MNIST, FashionMNIST

    def load(cls, train):
        ds = cls(DATA, train=train, download=True)
        return ds.data.numpy().astype(np.uint8)[..., None], ds.targets.numpy()

    kx, ky = load(KMNIST, True)
    mx, my = load(MNIST, True)
    fx, fy = load(FashionMNIST, True)
    mtx, mty = load(MNIST, False)
    ftx, fty = load(FashionMNIST, False)
    return dict(
        base_x=kx, base_y=ky,
        pool_x=np.concatenate([mx, fx]), pool_y=np.concatenate([my, fy + 10]),
        test_x=np.concatenate([mtx, ftx]), test_y=np.concatenate([mty, fty + 10]),
        base_classes=np.arange(10), novel_classes=np.arange(20),
    )


PREPARE = dict(cifar100=prepare_cifar100, omniglot=prepare_omniglot, mnist_fashion=prepare_mnist_fashion)


def split_path(name):
    return CACHE / f"splits_{name}.npz"


def load_split(name):
    return dict(np.load(split_path(name)))


if __name__ == "__main__":
    for name in DATASETS:
        if split_path(name).exists():
            s = load_split(name)
        else:
            s = PREPARE[name]()
            # the three parts must not share a class (base labels live in their own label space,
            # so the check is on original class ids where the dataset has them)
            if name == "cifar100":
                assert not set(s["base_classes"]) & set(s["novel_classes"])
            np.savez_compressed(split_path(name), **s)
        print(f"{name:14s} base {s['base_x'].shape} ({len(np.unique(s['base_y']))} classes) | "
              f"pool {s['pool_x'].shape} | test {s['test_x'].shape} "
              f"({len(np.unique(s['test_y']))} classes)")
