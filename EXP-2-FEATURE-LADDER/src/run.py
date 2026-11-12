"""Run the few-shot, class-by-class learning protocol of EXP-2.

For one dataset and one feature source: draw k examples per class, present the classes one at a
time (no task hint, no stored examples), and score on held-out images of the classes seen so far.

Run:  python src/run.py --tag pilot --datasets mnist_fashion --seeds 3
Results are appended to results/<tag>.jsonl; finished combinations are skipped on a rerun.
"""
import argparse
import json
import time

import common  # noqa: F401
import numpy as np
import torch

from common import CACHE, DATASETS, RESULTS, SHOTS, device
from features import SOURCES
from learners import LEARNERS, make_context


def accuracy(scores, y, seen):
    """Accuracy over test images whose class is in `seen`, choosing only among `seen` classes."""
    mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
    mask[seen] = True
    rows = mask[y]
    pred = scores[rows].masked_fill(~mask, float("-inf")).argmax(1)
    return (pred == y[rows]).float().mean().item()


def block_accuracy(scores, y, seen, block):
    """Accuracy on test images of `block` classes, while choosing among all `seen` classes."""
    mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
    mask[seen] = True
    bmask = torch.zeros_like(mask)
    bmask[block] = True
    rows = bmask[y]
    pred = scores[rows].masked_fill(~mask, float("-inf")).argmax(1)
    return (pred == y[rows]).float().mean().item()


def one_run(feats, ctx, learner_name, k, seed, fixed_order, dev):
    pool_f, pool_y, test_f, test_y = feats["pool_f"], feats["pool_y"], feats["test_f"], feats["test_y"]
    K, d = int(pool_y.max()) + 1, pool_f.shape[1]
    rng = np.random.default_rng(1000 * seed + k)
    order = np.arange(K) if fixed_order else rng.permutation(K)
    pool_y_np = pool_y.cpu().numpy()
    shots = {c: rng.permutation(np.flatnonzero(pool_y_np == c))[:k] for c in range(K)}

    checkpoints = sorted({int(round(v)) for v in np.linspace(K / 10, K, 10)})
    half = K // 2
    block = torch.as_tensor(order[:half], device=dev)
    learner = LEARNERS[learner_name](d, K, dev, ctx=ctx, seed=seed)

    acc_seen, acc_block, learn_s = {}, {}, 0.0
    for n, c in enumerate(order, start=1):
        x = pool_f[torch.as_tensor(shots[c], device=dev)]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        learner.learn(x, int(c))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        learn_s += time.perf_counter() - t0
        if not learner.frozen_past and (n in checkpoints or n == half):
            seen = torch.as_tensor(order[:n], device=dev)
            sc = learner.scores(test_f)
            if n in checkpoints:
                acc_seen[n] = accuracy(sc, test_y, seen)
            if n >= half:
                acc_block[n] = block_accuracy(sc, test_y, seen, block)

    if learner.frozen_past:   # older classes never change, so earlier checkpoints are exact
        sc = learner.scores(test_f)
        for n in sorted(set(checkpoints) | {half}):
            seen = torch.as_tensor(order[:n], device=dev)
            if n in checkpoints:
                acc_seen[n] = accuracy(sc, test_y, seen)
            if n >= half:
                acc_block[n] = block_accuracy(sc, test_y, seen, block)

    return dict(
        learner=learner_name, shots=k, seed=seed, dim=d, classes=K,
        final_acc=acc_seen[K],
        acc_seen={str(n): round(a, 5) for n, a in sorted(acc_seen.items())},
        first_half_acc_when_alone=acc_block[half], first_half_acc_at_end=acc_block[K],
        learn_ms_per_class=round(1000 * learn_s / K, 4),
        prototypes=len(learner.P) if learner_name == "clp" else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    ap.add_argument("--learners", nargs="*", default=list(LEARNERS))
    ap.add_argument("--shots", nargs="*", type=int, default=list(SHOTS))
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()

    dev = device()
    out = RESULTS / f"{a.tag}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["dataset"], r["source"], r["learner"], r["shots"], r["seed"]))

    t_all = time.time()
    for ds in a.datasets:
        for src in a.sources:
            feats = {k: v.to(dev) for k, v in torch.load(CACHE / f"feat_{ds}_{src}.pt").items()}
            ctx = make_context(feats["base_f"], feats["base_y"])
            t_cell = time.time()
            with open(out, "a") as fh:
                for seed in range(a.seeds):
                    for k in a.shots:
                        for ln in a.learners:
                            if (ds, src, ln, k, seed) in done:
                                continue
                            r = one_run(feats, ctx, ln, k, seed, ds == "mnist_fashion", dev)
                            r.update(dataset=ds, source=src, clp_theta=round(ctx["clp_theta"], 4))
                            fh.write(json.dumps(r) + "\n")
                            fh.flush()
            print(f"{ds} {src}: {time.time() - t_cell:.0f}s", flush=True)
            del feats
            torch.cuda.empty_cache()
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    with torch.no_grad():
        main()
