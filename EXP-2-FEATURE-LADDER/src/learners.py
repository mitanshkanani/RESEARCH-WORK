"""The four learners of EXP-2. None of them uses backpropagation or stores old examples.

Every learner sees classes one at a time and offers:
    learn(x, y)   x: (n, d) features of ONE class, y: its class index
    scores(X)     (N, K) score per class; the runner hides classes not yet seen
    frozen_past   True if learning a new class never changes what was stored for older classes
                  (then accuracy at earlier checkpoints can be read off the final state exactly)

Where a paper left a detail open, the choice made here is marked "OUR CHOICE".
"""
import torch
import torch.nn.functional as F


class NCM:
    """Nearest class mean: keep one running average per class, answer with the closest one."""
    name = "ncm"
    frozen_past = True

    def __init__(self, d, n_classes, dev, ctx=None, seed=0):
        self.sums = torch.zeros(n_classes, d, device=dev)
        self.counts = torch.zeros(n_classes, device=dev)

    def learn(self, x, y):
        self.sums[y] += F.normalize(x, dim=1).sum(0)   # OUR CHOICE: unit-length features
        self.counts[y] += len(x)

    def scores(self, X):
        means = self.sums / self.counts.clamp(min=1)[:, None]
        return -torch.cdist(F.normalize(X, dim=1), means) ** 2


class SLDA:
    """Deep Streaming LDA (Hayes & Kanan, arXiv 1909.01520): running class means plus one shared
    covariance, updated one sample at a time. Shrinkage 1e-4 as in the paper. The covariance
    starts from the feature-building ("base") data with the OAS estimator, as in the paper.
    OUR CHOICE: the sample counter starts at the number of base samples."""
    name = "slda"
    frozen_past = False   # the shared covariance keeps moving

    def __init__(self, d, n_classes, dev, ctx=None, seed=0, eps=1e-4):
        self.mu = torch.zeros(n_classes, d, device=dev, dtype=torch.float64)
        self.c = torch.zeros(n_classes, device=dev, dtype=torch.float64)
        self.sigma = ctx["sigma0"].to(dev, torch.float64).clone()
        self.t = float(ctx["n_base"])
        self.eps = eps

    def learn(self, x, y):
        for z in x.to(torch.float64):
            diff = z - self.mu[y]                       # deviation from the mean BEFORE the update
            delta = self.t * torch.outer(diff, diff) / (self.t + 1)
            self.sigma = (self.t * self.sigma + delta) / (self.t + 1)
            self.mu[y] = (self.c[y] * self.mu[y] + z) / (self.c[y] + 1)
            self.c[y] += 1
            self.t += 1

    def scores(self, X):
        d = self.sigma.shape[0]
        lam = torch.linalg.inv((1 - self.eps) * self.sigma + self.eps * torch.eye(d, device=X.device, dtype=torch.float64))
        W = self.mu @ lam                                # (K, d)
        b = -0.5 * (W * self.mu).sum(1)
        return (X.to(torch.float64) @ W.T + b).float()


class Fly:
    """FlyModel (Shen, Dasgupta & Navlakha, arXiv 2107.07617): a fixed random sparse expansion,
    winner-take-all, then only the weights from the active units to the correct class grow.
    alpha = 0 (no decay) and weights capped to [0, 1], as in the paper.
    The paper used m = 3200 / 20000 units, about 12% of inputs per unit, and l = m / classes winners.
    OUR CHOICES: m = 20000 everywhere; l = max(m // classes, 32); beta = 0.1; inputs clipped to be
    non-negative; min-max scaling done per sample before the winner-take-all step."""
    name = "fly"
    frozen_past = True

    def __init__(self, d, n_classes, dev, ctx=None, seed=0, m=20000, beta=0.1, frac_in=0.125):
        g = torch.Generator(device="cpu").manual_seed(seed)
        s = max(2, round(frac_in * d))
        idx = torch.rand(m, d, generator=g).topk(s, dim=1).indices
        self.theta = torch.zeros(m, d).scatter_(1, idx, 1.0).to(dev)
        self.l = max(m // n_classes, 32)
        self.beta = beta
        self.W = torch.zeros(n_classes, m, device=dev)

    def code(self, X):
        phi = X.clamp(min=0) @ self.theta.T
        lo, hi = phi.min(1, keepdim=True).values, phi.max(1, keepdim=True).values
        phi = (phi - lo) / (hi - lo + 1e-8)
        top = phi.topk(self.l, dim=1)
        return torch.zeros_like(phi).scatter_(1, top.indices, top.values)

    def learn(self, x, y):
        for z in self.code(x):
            self.W[y] = (self.W[y] + self.beta * z).clamp(max=1.0)

    def scores(self, X):
        out = []
        for i in range(0, len(X), 1024):
            out.append(self.code(X[i:i + 1024]) @ self.W.T)
        return torch.cat(out)


class CLP:
    """Prototype neurons added on novelty, after CLP / CLP-SNN (arXiv 2511.01553).
    Inputs are clipped to non-negative and scaled to unit length. If no stored prototype is similar
    enough (cosine < theta), a new prototype neuron is created from the input. If the best match
    has the right label it is pulled towards the input, with a learning rate that shrinks as the
    prototype proves itself (alpha = 1 / g).
    This is CLP-STYLE, not a reproduction. OUR CHOICES: a wrong best match also creates a new
    prototype (the paper's hardware version does this; its adaptive version pushes the wrong
    prototype away instead); g starts at 1 and grows by 0.5 per correct match (the paper's table
    was unreadable); theta is set from the base data (midpoint between the average same-class and
    different-class similarity) because the right value depends on the features; no pool limit."""
    name = "clp"
    frozen_past = True   # only prototypes of the class being learned are ever created or moved

    def __init__(self, d, n_classes, dev, ctx=None, seed=0, dg=0.5):
        self.theta = ctx["clp_theta"]
        self.P = torch.zeros(0, d, device=dev)
        self.labels = torch.zeros(0, dtype=torch.long, device=dev)
        self.g = torch.zeros(0, device=dev)
        self.n_classes = n_classes
        self.dg = dg

    @staticmethod
    def prep(X):
        return F.normalize(X.clamp(min=0), dim=1)

    def _add(self, z, y):
        self.P = torch.cat([self.P, z[None]])
        self.labels = torch.cat([self.labels, torch.tensor([y], device=z.device)])
        self.g = torch.cat([self.g, torch.ones(1, device=z.device)])

    def learn(self, x, y):
        for z in self.prep(x):
            if len(self.P) == 0:
                self._add(z, y)
                continue
            sim = self.P @ z
            k = int(sim.argmax())
            if sim[k] < self.theta or int(self.labels[k]) != y:
                self._add(z, y)
            else:
                self.g[k] += self.dg
                a = 1.0 / self.g[k]
                self.P[k] += a * z - a * sim[k] * self.P[k]   # self-normalising update, r = +1

    def scores(self, X):
        sim = self.prep(X) @ self.P.T                          # (N, n_prototypes)
        out = torch.full((len(X), self.n_classes), -2.0, device=X.device)
        idx = self.labels[None, :].expand(len(X), -1)
        return out.scatter_reduce(1, idx, sim, reduce="amax")  # best prototype of each class


LEARNERS = {c.name: c for c in (NCM, SLDA, Fly, CLP)}


def make_context(base_f, base_y, max_n=10000, seed=0):
    """Statistics taken from the feature-building data, used by SLDA and CLP only."""
    from sklearn.covariance import oas
    g = torch.Generator().manual_seed(seed)
    keep = torch.randperm(len(base_f), generator=g)[:max_n]
    f, y = base_f[keep].cpu().double(), base_y[keep].cpu()

    resid = f.clone()                                   # within-class residuals for the covariance
    for c in y.unique():
        resid[y == c] -= f[y == c].mean(0)
    sigma0, _ = oas(resid.numpy(), assume_centered=True)

    z = CLP.prep(f.float())
    i = torch.randint(len(z), (20000,), generator=g)
    j = torch.randint(len(z), (20000,), generator=g)
    sim, same = (z[i] * z[j]).sum(1), (y[i] == y[j]) & (i != j)
    theta = float((sim[same].mean() + sim[~same].mean()) / 2)
    return dict(sigma0=torch.from_numpy(sigma0), n_base=len(f), clp_theta=theta)
