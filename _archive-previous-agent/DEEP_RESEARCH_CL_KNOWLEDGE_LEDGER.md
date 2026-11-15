# Deep Research: What Continual Learning Actually Knows
### A settled / contested / unknown ledger, built to stop you spending GPU days on questions the field has already answered

> Generated 2026-10-01 | Depth: deep | Sources: 84 | Evidence base: 35 local PDFs mined in full + 4 net-sweep waves + 2 verification waves
> Built for: MK · 4 GB RTX 3050 Laptop · Python 3.14.6 · solo, no collaborators, no cluster

---

## TL;DR

Catastrophic forgetting is one of the most solidly replicated quantitative results in machine learning, but almost everything built on top of it is softer than the papers imply: the *size* of forgetting is set by the evaluation protocol rather than the network, most published method gains evaporate against strong baselines or unstabilized training regimes, and the field's own reference implementation cannot reproduce roughly a third of its methods. For your immediate decision the news is specific and uncomfortable: **the single most valuable measurement in your frozen EXP-1 — the linear probe on frozen post-task-B features (A4) — was already published in 2022, with the same protocol** [57], and **the stated reason you picked MNIST→Fashion-MNIST as the "destructive" pair rests on a theory paper about a linear model that you cited as if it were an empirical CNN result** [80][34]. EXP-1 is still worth running — it is cheap, and several of its thirteen weight-forensics metrics are genuinely unmeasured at your scale — but it must be re-framed as an *instrument calibration* run, not as a novel finding, and three of its citations need repair before you freeze anything further.

---

## Executive Summary

I read all 35 PDFs in your three folders (text-layer extraction, page-cited throughout), swept the open literature in four parallel waves, then ran two adversarial verification waves whose explicit job was to break the claims that drive your decisions. Four conclusions matter.

**1. The settled column is real, and it is mostly negative results.** Sequential fine-tuning without countermeasures collapses toward chance in class-incremental settings and degrades only ~12 points in task-incremental ones — 87.19 / 59.21 / 19.90 on Split-MNIST under TIL / DIL / CIL, same weights, same training run, different question [36]. Replay of real data is the intervention that reliably works; pure weight regularization (EWC, SI, MAS) and distillation (LwF) sit within noise of doing nothing once the scenario requires task inference [36][41][44]. Two further findings are settled and were each independently re-read from primary text: the **stability gap** (forgetting is often transient, and evaluating only at task boundaries can miss total collapse — min-ACC 0.0 ± 0.0 % on Split-CIFAR-10, where every task hits zero at some iteration) [5], and **loss of plasticity** (under long continual training, standard backprop stops being trainable at all, decaying to shallow-network performance across architectures, optimizers, activations, BatchNorm and dropout — ~88 % early-task accuracy falling to ~77 % by the 2000th task) [48].

**2. The contested column is where the measurement is doing the arguing, not the method.** Model scale versus forgetting looks contradictory across your library until you separate *width* from *parameter count* from *pretraining* — once separated, the strongest primary study (width 32→2048, 2-layer MLPs, Rotated-MNIST) shows wider-remembers-better and deeper-does-nothing, and crucially **your 242k-parameter model sits inside the regime that paper actually measured** [40]. Task similarity is genuinely contested, and the analytical evidence leans *against* your assumption that near-identical inputs make the worst pair [80][81]. And your own folders contain a live contradiction: PAPER 53's continual-learning SNN runs at T = 25-60 timesteps with spike rates of 8-31 %, which PAPER 27's verified algebra places roughly 9-10× outside the window where an SNN beats a quantized ANN [14][15] — both sides confirmed from your local files.

**3. The unknown column is shorter and differently-shaped than you think.** Your load-bearing open claim (C7, "nobody has a long-horizon non-stationary benchmark that tests whether test-time learning delivers continual learning") survived attack in its narrow form — CL-Bench's own limitations section states, verbatim, that it does not evaluate parametric approaches such as test-time training [73] — but died in its broad form: CL-Bench *does* test drifting tasks with explicit stability/plasticity scoring, just with frozen weights, and TTT has been run on genuinely non-stationary video streams [74]. C7 must be restated as a *mismatch* claim, not an *absence* claim. That is stronger, and it is instrumentable.

**4. A documented dead-idea list exists, and it is free.** Mammoth's own reproducibility checklist is the best negative-results ledger in the field: A-GEM, BiC, EWC-Online, FDR, GSS, HAL, LUCIR, LwF, LwF-MC, MER, PNN, Puridiver, Ranpac, RPC, SI, SLCA are marked **not** reproduced against their published numbers [47]. One method, AttriClip, was pulled by its own authors because it did not reproduce [47]. Together with the "your baseline was too weak" literature [39][45][49], this converts most proposed CL experiments into a question of which already-answered cell they occupy.

---

## 0. How to read this ledger

**Confidence labels.** `[High]` = 3+ independent sources agree with at least one Tier 1 (peer-reviewed) source; `[Medium]` = 2 sources, or 1 highly authoritative source; `[Low]` = single source, or not independently confirmed.

**Provenance tags on numbers.** This matters more than usual here, because your own `END_GOAL.md` §10 documents four fabricated citations produced by an earlier automated pass [33], and `EXP1_JOURNAL.md` §17.2 documents three more from your own memory [34]. So every number below carries where it came from:

| Tag | Meaning |
|---|---|
| `[local:read]` | I read it in your PDF's text layer, page-cited. Highest trust I can give. |
| `[full:read]` | A verification agent read the primary paper in full and quoted it. |
| `[abstract]` | Only the abstract was read. Do not treat internal numbers as settled. |
| `[unverified]` | Surfaced second-hand or unfetchable. Listed so you can chase it, not so you can cite it. |
| `[conflict]` | Two of my own sources disagree. Both values shown. |

**The compound analogy you asked for.** Catastrophic forgetting is the "guaranteed property of the compound" — it behaves like the fact that a given molecule burns with a blue flame: nobody seriously doubts it, and re-testing it earns you nothing. What is *not* guaranteed is everything downstream: how much damage, in which layers, whether the information is destroyed or merely unreported, and whether any given cure works. Most CL papers are re-measuring the blue flame and calling it a new discovery about combustion.

---

## 1. THE SETTLED COLUMN — safe to assume, do not re-measure `[Confidence: High]`

### 1.1 Forgetting is real, large, and not a capacity problem

Sequential fine-tuning of a from-scratch network destroys old-task performance when the setting requires the model to work out which task it is looking at. The canonical baseline table, read from primary text and cross-checked against your library [36][2][3]:

**"None" / naive fine-tuning, 5 tasks × 2 classes, from-scratch MLP, average test accuracy %** `[full:read]`

| Benchmark | Task-IL | Domain-IL | Class-IL | Offline (joint) upper bound |
|---|---|---|---|---|
| Split-MNIST | 87.19 ± 0.94 | 59.21 ± 2.04 | **19.90 ± 0.02** | 99.66 / 98.42 / 97.94 |
| Permuted-MNIST | 81.79 ± 0.48 | 78.51 ± 0.24 | **17.26 ± 0.19** | 97.68 / 97.59 / 97.59 |

Chance in CIL here is ~20 %. So the naive baseline is *at chance* [36]. The same pattern holds at CIFAR scale, from the DER++ paper's Table 2 on page 6 of your own PAPER 7:

**"SGD" = sequential fine-tuning with no countermeasure; ResNet-18 trained from scratch; 10 runs** `[local:read]` + `[full:read]` (two independent reads of the same table agree exactly) [7]

| Benchmark | Class-IL SGD | Task-IL SGD | JOINT |
|---|---|---|---|
| Split-CIFAR-10 (5×2) | **19.62 ± 0.05** | 61.02 ± 3.33 | 92.20 / 98.31 |
| Split-TinyImageNet (10×20) | **7.92 ± 0.26** | 18.31 ± 0.68 | 59.99 / 82.04 |
| Permuted-MNIST (20 tasks, DIL) | 40.70 ± 2.33 | — | 94.33 |
| Rotated-MNIST (20 tasks, DIL) | 67.66 ± 8.53 | — | 95.76 |

Two settled refinements worth internalizing:

- **Forgetting is not explained by running out of room.** Interleaved training on the same set of tasks works, which is why `van de Ven et al.` state plainly that "catastrophic forgetting is not due to limited model capacity" [3, Fig. 1b, p.3] `[local:read]`.
- **It gets worse monotonically with task count.** Permuted-MNIST naive: 94.74 (TIL) / 84.56 (DIL) / **12.82 ± 0.95 (CIL)**; and in a ~300K-parameter MLP on a 78-task queue, accuracy falls 99.95 → 87.09 (11 tasks) → 72.18 (31) → **58.05 (78 tasks)**, with CNNs degrading faster [43] `[full:read]`.

### 1.2 The magnitude is a property of the *question you ask*, not of the network

This is the single most consequential settled fact for you. Read the Split-MNIST row again: **87.19 % versus 19.90 % is the same trained weights doing the same underlying computation, and only the evaluation protocol differs** [36][34]. Your journal already noticed this [34 §1]; the ledger confirms it is settled, and adds that the field formalized eight scenarios, not three (IIL / DIL / TIL / CIL / TFCL / OCL / BBCL / CPT, distinguished by label-space overlap, task-ID availability, batch size and single-pass-ness) [1, Table 1, p.3] `[local:read]`.

Corollary, also settled: **a forgetting number without a declared output space is uninterpretable.** The metric definitions all hinge on whether `a[k,j]` is computed over `Y_j` (multi-head, TIL-like) or `∪ᵢ Yᵢ` (single-head, CIL) [1, p.3] `[local:read]`.

### 1.3 Replay is the intervention that works; everything else is situational

| Intervention | Verdict | Best primary evidence |
|---|---|---|
| **Balanced-buffer replay + retraining** (GDumb, iCaRL, DER/DER++, X-DER) | **Settled: reliably large gains over naive FT** | DER++ 72.70 vs SGD 19.62 on Split-CIFAR-10 CIL, buffer 500 [7] `[local:read]`; X-DER 49.93 vs fine-tune 9.43 on CIFAR-100 [46] |
| **Replay + logit/representation distillation** | Settled: better than either alone | iCaRL 47.55, DER 70.51, DER++ 72.70 vs ER 57.74 at matched buffer 500 [7] `[local:read]` |
| **Weight regularization** (EWC, oEWC, SI, MAS, LwF) | **Settled: does not work when task inference is required** | Split-MNIST CIL: EWC 20.01, oEWC 19.96, SI 19.99 vs None 19.90 — all indistinguishable from chance [36] `[full:read]`. Even MAS's best number (46.90) is under a *multi-head TIL* protocol [2, Table 10] `[local:read]` |
| **Constrained-memory gradient methods** (GEM, A-GEM) | Settled: weak beyond MNIST | Split-CIFAR-10 CIL: GEM 26.20, A-GEM 22.67 vs ER 57.74 [7] `[local:read]`; A-GEM ≈ fine-tuning on CIFAR-10 (18.5 vs 18.5) [45] |
| **Orthogonal-gradient / null-space projection** (GPM, Adam-NSCL) | **Not settled — single-lab, self-reported** | "better or on-par with state-of-the-art" on image classification only; absent from Mammoth's verification table [50][47] |
| **Parameter isolation / growth** (PackNet, HAT, PNN) | Settled: **zero forgetting by construction, at a price** | PackNet forgetting 0.00 across Tiny ImageNet [2] `[local:read]` — but only 26.62 % on a novel SVHN task at 40 tasks, vs ~90 % for others: "PackNet … severely deteriorat[es] in plasticity" [2, p.27] `[local:read]` |

Note the pattern in that last row, because it is the deepest settled insight in the field and your end goal is aimed straight at it: **the methods that achieve zero forgetting achieve it by refusing to reuse parameters, and pay for it by losing the ability to learn new things.** PackNet is the proof-of-existence and the proof-of-limit at the same time [2]. Your trilemma table (`END_GOAL.md` §3: into existing weights / into new structure / outside the model) is the right frame, and the literature's verdict is that row 2 works and costs linear capacity, while row 1 costs memory [33][2].

### 1.4 The stability gap: forgetting is frequently transient, and boundary-only evaluation hides it

Settled, and verified directly in your own PAPER 40 [5]:

- Definition, verbatim: methods "still suffer from substantial forgetting upon starting to learn new tasks, except that this forgetting is **temporary and followed by a phase of performance recovery**" [5, p.1] `[local:read]`.
- It affects experience replay, constraint-based replay, distillation, and parameter regularization, and appears in CIL, TIL and DIL benchmarks [5, p.1] `[local:read]`.
- Headline measurement (class-incremental Split-MiniImagenet, ER, 5 seeds): standard metrics report ACC 32.9 ± 0.8 and FORG 32.3 — while per-iteration **min-ACC is 0.5 ± 0.2 %** [5, Table 1, p.6] `[local:read]`. On Split-CIFAR-10, **min-ACC is 0.0 ± 0.0** [5, Table 3, p.14] `[local:read]`.
- Coarser evaluation *hides* it: WC-ACC climbs 4.1 → 7.1 as evaluation frequency drops from every iteration to every 10³ [5] `[local:read]`. At ρ_eval = 100 the authors "completely miss[ed] the accuracy drop to zero of T4 for 2 out of the 5 seeds" [5, p.14] `[local:read]`.
- Not scale-specific: architectures span a 400-unit hidden MLP to a slim ResNet-18 [5] `[local:read]` + `[full:read]` of the arXiv HTML.
- **Operational recommendation, verbatim: "we advocate the use of continual evaluation when possible"** [5, p.13] `[local:read]`, concretely every iteration on 1k held-out samples per task [5, Table 4, p.15] `[local:read]`.
- Independent confirmation at a completely different scale: LLM continual pre-training shows the same shape — "a temporary performance drop at the beginning, followed by a recovery phase" — and the mitigation is *mixing old data back in* [77] `[abstract]`.

**Replication caveat:** within budget I found no third-lab replication by a group other than the original authors. The phenomenon is corroborated across modality and scale by [77], but the follow-up mitigation literature [56] is `[unverified]`.

### 1.5 Loss of plasticity: a second, deeper failure that is not forgetting

The most important finding in the settled column that your library does **not** contain, and it reframes your end goal. Dohare et al., *Nature* 632 (2024) `[full:read]` [48]:

- "standard deep-learning methods gradually **lose plasticity** … until they learn no better than a shallow network."
- On Continual ImageNet, binary tasks: conventional backprop "loses plasticity at all step sizes." The fix is *not* replay — "continual backpropagation, L2 regularization and Shrink and Perturb algorithms maintain plasticity, apparently indefinitely."
- Early-task accuracy reached ~88 %; the preprint reports 89 % → **77 % by the 2000th task**, "about the level of a linear network."
- Measured mechanism: "the number of network units that are active less than 1 % of the time increases rapidly." Reinitializing the least-used units — "typically fewer than one per step" — restores it.
- Robustness of the failure: "occurred with a wide range of deep network architectures, optimizers, activation functions, batch normalization, dropout."
- CIFAR-100 incremental: by the end, standard training sits "5 % lower than the retrained network (a performance drop equivalent to that of removing a notable algorithmic advance, such as batch normalization)"; continual backprop reaches 76.13 % on all 100 classes.
- Code is released (arXiv:2306.13812 / github.com/shibhansh/loss-of-plasticity).

**Why this matters to you specifically.** Your goal is "learning without retraining." The settled literature says that even if you solve forgetting, a continuously-updated network stops being *learnable* for a different reason — the units go stale. Any claim of the form "our system learns indefinitely" must report plasticity, not just retention. Your own library already flags it: "incremental training of deep neural networks on a sequence of tasks can lead to a substantial loss of plasticity (Dohare et al., 2023), highlighting that rapid adaptation is still an open problem" [3, p.3] `[local:read]`.

### 1.6 Training regime dominates algorithm choice

- Stabilizing the training regime improves published methods by **12.9 % (EWC), 15.8 % (A-GEM), 9 % (ER-Reservoir)** in average accuracy — larger than most published method-over-method deltas — and "these simple techniques can outperform considerably more complex algorithms" [39] `[full:read]`. On Permuted-MNIST: EWC 70.7 / forgetting 0.23, A-GEM 65.7 / 0.29, vs Stable SGD **80.1 / 0.09 with no memory at all** [39] `[full:read]`.
- Hyperparameters tuned per-scenario **do not transfer**: "optimized hyperparameters boost initial tuning scores but trigger sharp performance declines when applied to unseen evaluation domains," and "Traditional regularization and exemplar-replay techniques demonstrate markedly superior cross-scenario generalization compared to modern expansion and prompt-based frameworks, **which frequently collapse on unseen data despite high initial benchmarks**" [44] `[full:read]`. Structural instability in the same audit: "BEEF constantly returns NaN in training loss at specific seeds"; "Ranpac suffers from significant instability."
- Random fixed features + a linear head, with **zero exemplars**, beat online continual representation learning by 5-15 points and "bridge 70-90 % of the performance gap relative to the respective joint classifiers"; and "prompt-tuning approaches completely collapse under large timesteps" [49] `[full:read]`.

**Ledger consequence:** any experiment where "our method beats baseline by X points" does not also control for LR schedule, epoch count, batch size and dropout is measuring hyperparameters while believing it measured an algorithm. Your C4 control already encodes this [34 §10] — keep it, and promote it from "control" to "co-primary result."

### 1.7 The rules that invalidate a CL measurement

Each has a primary source. Violate any one and your number is not interpretable.

| # | Rule | Source |
|---|---|---|
| R1 | **Declare the output space** for every `a[k,j]`, and the scenario (TIL/DIL/CIL). | [1, p.3] `[local:read]` |
| R2 | **Never grid-search hyperparameters using held-out validation data from all tasks** — "this inherently violates the main assumption in continual learning … This may lead to overoptimistic results, that cannot be reproduced in a true continual learning setting." | [2, p.4 §4] `[local:read]`; [41] "choosing the best hyperparameters for the sequence of tasks after those are learned is not a realistic scenario" `[full:read]` |
| R3 | **Evaluate continuously, not only at task boundaries** — otherwise you can measure the wrong thing in either direction. | [5, p.13] `[local:read]` |
| R4 | **Do not use Permuted-MNIST as your headline benchmark**: "Pixel permutation represents an unrealistic best case scenario" — permuted inputs give near-zero confidence (prediction entropy 0.453 permuted vs 0.003 split), and forgetting is driven by confident-but-wrong gradients. Prior-focused methods "appear to succeed, but have major blind-spots." | [38] `[full:read]` |
| R5 | **Single prediction head, no test-time task labels, no unconstrained rehearsal, and sequences longer than two tasks.** Verbatim: "Showing that a method succeeds on a two-task transfer does not entail that it will work on a longer series." | [38] `[full:read]` — aimed squarely at EXP-1 |
| R6 | **Report worst-case metrics** (min-ACC, WC-ACC, windowed forgetting) alongside average accuracy, and report speed–performance *frontiers* rather than single points. | [5] `[local:read]`, [38] `[full:read]` |
| R7 | **Multiple seeds, and treat the seed spread as the result.** Published convention at this scale is 10 runs; ±SD still understates uncertainty, because data-sampling variance dominates weight-init variance. Claim superiority only when the bootstrap probability of outperformance exceeds **0.75**. | [41] `[full:read]`; [42] `[full:read]` |
| R8 | **The averaging trap:** "a plastic model that achieves 0 % accuracy on the first task and 100 % accuracy on the second task, has the same average performance as a completely rigid model with 100 % on the first and 0 % on the second." | [3, p.7] `[local:read]` |
| R9 | **Report inference-time cost**, not just training cost — "a portion of continual learning methods introduce additional complexity during inference … not always identical to the training costs." | [3, p.7] `[local:read]` |

### 1.8 What is settled in your brain-inspired and energy folders

Less than you might hope. I grepped all 19 PDFs in folders 02 and 03 for `continual|catastrophic|forget|stability-plasticity|incremental|non-stationary|replay|consolidat`: **only two of the 19 contain any continual-learning measurement** — PAPER 53 and PAPER 45. PAPER 17, 18, 14, 12, 19, 51, 50, 26, 13, 25 have zero hits. PAPER 23 (LeCun JEPA) has zero hits for forgetting/continual/replay/consolidation and self-describes as "not a technical nor scholarly paper … but a position paper" [27] `[local:read]`. Those folders are about learning-rule plausibility and hardware energy, not about CL.

Settled results from them that *are* load-bearing:

- **Backprop-approximating local rules collapse at scale.** ImageNet top-1 error: BP 71.43, BP-convnet 63.93, **FA 93.08, DTP-parallel 98.34, DTP-alternating 99.36, SDTP 99.28** — the last three are chance for 1000 classes. "All of the biologically motivated algorithms performed much worse than BP in the context of ImageNet." DFA could not run at all: OOM on a 16 GB GPU [23] `[local:read]`.
- **Surrogate gradients are not gradients of any surrogate loss** (closed-path integral ≠ 0, not a numerical artifact), and **can have the opposite sign to the true gradient** — "following the SG will not necessarily find … a minimum" [18] `[local:read, pp.1-32]`.
- **The SNN energy advantage is conditional, not intrinsic.** Under typical neuromorphic hardware an SNN needs **T ≤ 5 and average spike rate below ~5.7 %** merely to match a same-architecture quantized twin; cost ratio ≈ T/⌈log₂(T+1)⌉ in the dense regime; SNN loses in 10 of 12 bit-serial comparisons; hardware-efficiency sweep: SNN lower in 12/12 at η=1 but **0/12 at η=64** [15] `[local:read]` — threshold sentence re-verified from the file.
- **Hinton self-retracted one Forward-Forward experiment.** Verbatim, PAPER 13 §5: "I have been unable to replicate this result and I now suspect it was due to a bug." Confirmed locally: the sentence is in PAPER 13, is *not* in PAPER 25, and §4.1 (the next-character experiment) exists in PAPER 25 but was deleted from PAPER 13 [25][26] `[local:read]`. The surviving CL-relevant caveat: separating positive and negative learning phases "only works if the learning rate is very low and the momentum is extremely high," and this is "probably the most important outstanding question about FF as a biological model" [25] `[local:read]`.
- **Experimentation, not the final model, is the energy cost** — NAS multiplied one Transformer run's footprint ~3,260× (192 → 626,155 lbs CO₂e) [31] `[local:read]`. For a solo researcher this is the strongest and only honest energy argument available to you, and it argues for careful design over many runs.
- **Replay suppresses forgetting in SNNs too** — in PAPER 53, no-replay → replay lifts N-MNIST 15.82 → 93.80 and MNIST 19.35 → 93.44 [14] `[local:read]`. Genuinely settled positive — and also the only durable result in that paper (see §2.8).

---

## 2. THE CONTESTED COLUMN — do not assume, and do not "resolve" it alone `[Confidence: Medium]`

### 2.1 Model scale versus forgetting — contested, but resolvable, and it lands on your side

| Claim | Source | Read level |
|---|---|---|
| Wider networks are more robust to forgetting | [1, p.13] `[local:read]` (relayed citation) | secondary |
| EWC performed *better* on a SMALL model than a WIDE one (42.43 → 31.10) | [2, Table 10 + p.17] `[local:read]` | their own experiment |
| Smaller models forget most; "increasing model capacity in DEEP models exhibits inferior performance" | [4, p.7] / [2, p.16] `[local:read]` | position / own experiment |
| Larger *pretrained* ResNets/Transformers are more forgetting-resistant; robustness improves with model **and** pretraining-dataset scale | [53] | `[abstract]` |
| In **online** CL, "larger models do not guarantee better Continual Learning performance; in fact, they often struggle more" | [54] | `[abstract]`, single-author preprint, no venue |

**Resolution I can defend:** separate the three conflated variables — *width*, *total parameters*, *pretraining* — and separate *offline* from *online*. That leaves one strong primary study, read in full and adversarially re-checked [40]:

- Rotated-MNIST, 5 tasks, 2-layer MLPs, width 32 → 2048, naive fine-tuning: average accuracy **65.9 ± 1.00 → 75.2 ± 0.34**; average forgetting **36.9 ± 1.27 → 26.7 ± 0.50**.
- Split-CIFAR-100, WideResNet-10: 8× width reduces first-task forgetting (after the 20th task) from **42 % to 31 %**.
- **Depth does not help**: "increasing the width alone reduces catastrophic forgetting significantly, while it's not the case for depth"; "achieving over-parametrization through depth has no or even negative effect"; "Shallower and wider networks have higher average accuracy and smaller forgetting than deeper and thinner networks."
- The width benefit persists under ER and A-GEM.
- **Nearest measured point to your 242k model:** depth-2, width-256 = 269.32K params → **71.1 avg accuracy, 31.4 avg forgetting**. (Correction to my own earlier report: the "59K-540K" range belongs to the 3-layer appendix table, which explicitly warns "simply increasing the number of parameters is not necessarily helpful.")

**Ledger entry:** *width* reducing forgetting is `[Medium→High]` and **applicable to you**, because you sit inside the measured regime. *Pretraining* reducing forgetting is `[Low]` — plausible, not read. *Total parameter count* reducing forgetting is **contested and regime-dependent**. Do not cite the foundation-model position paper's "smaller models forget most" as evidence about your CNN; its regime is orders of magnitude above yours and cannot be extrapolated downward [4] `[local:read]`.

### 2.2 Does task similarity make forgetting worse? Contested — and your premise is on the losing side

Your design justifies MNIST→Fashion-MNIST partly by Hiratani: "high input feature similarity coupled with low readout similarity is catastrophic for both knowledge transfer and retention" [34 §2.2, arXiv:2405.20236]. Adversarial verification found two problems [80] `[abstract read in full]`:

1. **The quote is overstated.** The abstract says the condition "is **catastrophic for retention**" — not "for both knowledge transfer and retention." Transfer sits on a separate, opposing axis: "Task-dependent activity gating improves knowledge retention **at the expense of transfer**." The reverse configuration is called "relatively benign."
2. **It is an analytical result for a linear teacher-student model with latent structure**, confirmed on a *permuted*-MNIST task. No CNN accuracy table, no Fashion-MNIST. Using it as an empirical prediction for your 242k CNN is a category error, and its "input similarity" means *shared latent generative structure*, not "both are 28×28 grayscale."

The direct opposite direction has support:

- De Lange et al. measured it: on Rotated-MNIST, *increasing* dissimilarity drove min-ACC from **94.3 % → 69.2 %** while ordinary ACC fell only 6.6 % — "the effect on the stability gap is substantially larger" [5, Table 2, p.6] `[local:read]`.
- Goldfarb et al. (ICLR 2024), the only analytical study that actually *derives* similarity-vs-forgetting direction, finds the relation is **non-monotone**: in highly overparameterized models *intermediate* similarity causes the most forgetting; near the interpolation threshold forgetting decreases monotonically as similarity increases [81] `[abstract]`. Under **both** regimes the maximum-similarity pair is not the worst case.
- Wang et al. note NTK theory says more similarity → more forgetting *for multi-head outputs*, while "synergistic tasks" improve error and "competing tasks" deteriorate it [1, pp.5-7] `[local:read]` — consistent with the sign flipping on head configuration.

**Ledger entry:** `[Contested]`, with primary evidence leaning *against* "most-similar inputs = worst case." Separately: **nobody has published an MNIST→Fashion-MNIST forgetting number** — two independent search attempts, ten phrasings, nothing. Neither "near-zero/floor effect" nor "screaming forgetting" is supported. This pair is genuinely unmeasured, which is good news for you, and is stated precisely in §5.

### 2.3 The Fisher-importance mechanism — contested *by its own authors*

EWC's algorithm works; EWC's *explanation* does not hold up as cleanly as the literature repeats. Fig. 4C injects Gaussian weight noise with covariance uniform, inverse-Fisher, or uniform-in-the-Fisher-nullspace:

- Inverse-Fisher-shaped noise is more tolerable than uniform (uniform collapses by σ≈1e-2; Fisher-shaped holds to σ≈1e-1) — supporting the premise.
- **But the nullspace perturbation curve is indistinguishable from the inverse-Fisher curve.** Parameters the diagonal Fisher declares irrelevant are not inert. The authors say so: "This suggests that we are overconfident about certain parameters being unimportant: it is therefore likely that the chief limitation of the current implementation is that it underestimates parameter uncertainty" [6] `[local:read]`.
- The diagonal Fisher is **not a canonical quantity**: "The exact way in which the Fisher Information is computed is however rarely described, and multiple different implementations for it can be found online," and "many currently reported results for EWC could likely be improved by changing the way the Fisher Information is computed" [64] `[abstract]`.
- A saliency map computed on the **retention** set outperforms one computed on the forgetting set — a direct challenge to "measure importance of task A on task A's data" [65] `[abstract]`.

Also settled-but-unappreciated from the same paper [6] `[local:read]`: "After network capacity is exceeded, EWC performs worse than gradient descent"; EWC is "prone to … **blackout catastrophe** when network capacity is saturated, resulting in the inability to retrieve any previous memories or store new experiences"; EWC "can model only memory retention rather than forgetting."

**Ledger entry:** "high-Fisher weights matter" is `[Contested]` — a useful heuristic with a self-documented false-negative rate, not a settled mechanism. If your W10 rests on it, compute ≥2 estimators and report both, and do not pre-commit to surprise at a null result.

### 2.4 "Replay-buffer sophistication matters" — contested

- **For:** DER++ 72.70 vs ER 57.74 at buffer 500, S-CIFAR-10 CIL [7] `[local:read]`.
- **Against, three directions:** (i) **DER loses to plain ER in Task-IL** — 93.40 vs 93.61 at buffer 500, and 95.43 vs **96.98** at buffer 5120 [7] `[local:read]`; (ii) DER++ is *worse* than DER at buffer 200 on S-TinyImageNet (10.96 vs 11.87) and MNIST-360 (54.16 vs 55.22) — the logit term costs accuracy exactly when memory is scarcest [7] `[local:read]`; (iii) at buffer 5000 on ImageNet-R, "GDumb 46.08 ≈ ER 47.23 ≈ MIR 49.33" and the authors concede "sophisticated memory retrieval strategies … do not significantly outperform GDumb's simple approach" [9, Appendix G] `[local:read]`.
- The surveys agree it is unresolved: "the merits and potential limitations of experience replay remain largely open" [1, p.9] `[local:read]`, while also recommending simple replay as the yardstick [1, p.10] `[local:read]`.
- Buffer *content refreshing* matters more than architecture: freezing X-DER's buffer drops accuracy from 49.93 to ~42 [46] `[full:read]`.

**Ledger entry:** `[Contested]`, regime-dependent. Your reported method differences are only meaningful at a declared buffer size, and buffer 500-vs-5000 can flip the ranking.

### 2.5 GDumb's own numbers conflict across papers — a live reproducibility dispute

| Source | GDumb on CIFAR-100 CIL |
|---|---|
| GDumb paper, B2 (20 tasks × 5 classes), CNN | 58.4 ± 0.8, vs iCaRL 44.2 ± 1.0, BiC 47.1 ± 1.5, PODNet(NME) 61.4 ± 0.7 [45] `[full:read]` |
| GDumb paper, **its own hedge** | "GDumb … performed nearly 20 % worse than BiC and iCaRL in B2" [45] |
| GDumb paper, Split-CIFAR-100 table | GDumb 48.3 vs iCaRL 45.1 / EEIL 46.2 / DER 47.5 / ER 44.8 / GEM-A-GEM 46.9 [45] `[conflict — extraction flagged]` |
| Boschini et al. (X-DER) reproduction | **GDumb 9.98 ≈ fine-tuning 9.43** [46] `[conflict]` |

That last row is the important one: in an independent peer-reviewed head-to-head, GDumb — the paper whose whole thesis is that simple replay-retraining wins — scores at the level of doing nothing. Either a protocol difference (online vs episodic, buffer refresh, epochs) explains it, or one group's implementation is wrong. I cannot tell you which from what I read. **Do not enter either number as fact in your writeup — but note the argument it hands you: a flagship baseline reported at 58.4 by its authors and 9.98 by a replicator is a measurement problem, not a method problem.**

### 2.6 PEFT / LoRA as a continual-learning solution — contested, and the honest version is worse than the framing

- LoRA does forget less than full fine-tuning — but "in the standard low-rank settings, LoRA **substantially underperforms full finetuning**"; it "better maintains the base model's performance on tasks outside the target domain"; it "mitigates forgetting more than common regularization techniques such as weight decay and dropout"; and "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations" [51] `[full:read]`. The forgetting win is partly just *learning less*.
- The field's consensus, relayed by [1]: parameter-efficient methods "remain vulnerable to catastrophic forgetting" `[local:read]`.
- LoRA-based CL methods claim SOTA with **self-reported, author-reimplemented baselines**. Two audit findings against [9] `[local:read]`: an internal inconsistency where GDumb on Split-ImageNet-R is 8.87 ± 1.36 in Table 1 but 1.65 ± 0.22 in Table 11; and on CORe50 DIL, forgetting is **0.00 for L2P, MVP and Ours simultaneously** — a degenerate metric (all classes always present) that cannot discriminate anything. Its margin over the best replay baseline in CIL is 49.40 vs MIR 48.36 / PCR 48.48 — about **1 point, inside the ±1.4-3.1 error bars**.
- Null-space-constrained LoRA (O-LoRA) is `[Low]`: abstract-only, no independent reproduction found [52].

**Ledger entry:** `[Contested]` → in practice "not established." PEFT limits how much an update *can* damage, not whether it does. Your `END_GOAL.md` C1/C2 already say this correctly [33]; nothing I read refutes it.

### 2.7 Test-time learning "as" continual learning — contested; the mismatch is the finding

| System | CL evidence actually reported | Read level |
|---|---|---|
| Titans [75] | **None.** LM, common-sense reasoning, genomics, time series, >2M context, NIAH. Title says "Learning to **Memorize** at Test Time." Abstract contains no hedge at all — the caution lives in the body. | `[abstract]` |
| Nested Learning / Hope [76] | Verbatim: "…**potentially unlocking effective continual learning capabilities**"; Hope shows "promising results" in, among others, "continual learning." Abstract names **no benchmark, no task count, no stream length, no metric**. | `[abstract]` |
| TTT with self-supervision [11] | Gains exist for a **single static** shifted test distribution: standard TTT buys ~2-6 error points, TTT-Online 15-30 on noise corruptions. Only non-stationary test is a monotone σ ramp over 8k samples. Own position: "we are not concerned about forgetting the past test samples since they have already been evaluated on … forgetting is not harmful and perhaps should even be taken advantage of." On ImageNet-C, standard TTT is **worse than joint training on 14 of 16 corruption types**. | `[local:read]` |
| TTT on video streams [74] | **Real online weight updates on a genuinely non-stationary stream** (KITTI-STEP, COCO Videos; AP 37.6 online vs 33.6 offline oracle vs 16.7 fixed model). But: "Our streaming setting is different from those commonly studied by the continual learning community, because it **does not have distinct splits of training and test sets**." No retention metric; earlier frames never re-tested. Forgetting reframed as beneficial: "some amount of forgetting is actually beneficial." | `[full:read]` |
| TTA survey [10] | "when and why TTA works remains an open problem"; **"evaluations of TTA methods have often been conducted unfairly. Existing studies frequently determine hyper-parameters through grid search on the test data."** Documents three mutually incompatible evaluation protocols in one literature. | `[local:read]` |

**Ledger entry:** `[Contested]` → the capability claim is unbacked by any benchmark measuring retention over a drifting *task* sequence. Your instinct in `END_GOAL.md` §4 ("retraining is relocated, not removed") is supported.

### 2.8 Your own two folders contradict each other — verified locally, and it is yours to make

- **PAPER 53** runs at **T = 25** (MNIST), **T = 50** (CIFAR-10, N-MNIST, CIFAR-10-DVS), **T = 60** (DVS-Gesture), with final spike rates 8.07 % (MNIST C4) up to **30.90 %** (CIFAR-10 C4) [14] `[local:read — timesteps and rates re-grepped from the file]`.
- **PAPER 27** establishes that an SNN needs **T ≤ 5 and spike rate below ~5.7 %** merely to match a same-architecture quantized ANN, with cost ratio ≈ T/⌈log₂(T+1)⌉ in the dense regime [15] `[local:read]`.
- Applying P27's law to P53's own configuration: CIFAR-10 ratio ≈ 50/log₂(51) ≈ **8.8×**, MNIST ≈ 25/log₂(26) ≈ **9.3×** *more* energy than the QNN twin — before data movement.

And PAPER 53's own negative results are worse than its abstract implies [14] `[local:read]`:

- **No ANN and no non-spiking CL baseline anywhere**: "No ANN or non-spiking baselines are used." No EWC, no SI, no GEM, no PackNet. Table 5 quotes rivals then concedes "direct numerical ranking is not meaningful."
- **On CIFAR-10 the framework does not reduce forgetting at all**: 33.56 → 33.56, with per-seed forgetting values *identical* between C1 and C4 while accuracies differ (probable logging artifact). 61 % accuracy with 33.6 % forgetting is a poor CL result regardless.
- **The controller alone makes stability worse**: C3 accuracy 92.40 vs C1 93.44, forgetting 8.39 vs 7.17 — "marginally decreases accuracy (−1.04 % points)."
- **Fashion-MNIST near-null**: forgetting 21.03 → 19.66.
- The headline +17.45 points is on DVS-Gesture, off a self-described "relatively low replay baseline (74.48 %)", at **n = 2 seeds**.
- "Spike rate is not a direct hardware power measurement" — despite "Energy-aware" in the title. No joules, no chip.

**Ledger entry:** P53's only durable claim is "replay fixes catastrophic forgetting in SNNs too" (C0→C1: 15.82→93.80 N-MNIST, 19.35→93.44 MNIST) [14] `[local:read]`. Its "spike budgeting reduces forgetting" claim is `[Contested]` internally, and its "energy-aware" framing is refuted by [15] for frame-based data. If you were eyeing the CL×SNN intersection as your niche, this contradiction *is* the niche — but see §6.

### 2.9 Which layers lose information — contested, two answers pointing opposite ways

- **Kumar et al.** [62]: full fine-tuning gains **+2 % in-distribution but loses ~7 % out-of-distribution** vs linear probing, because "while fine-tuning learns the head, the lower layers of the neural network change simultaneously and distort the pretrained features"; LP-FT gives +1 % ID / **+10 % OOD**. Verdict: **protect the lower layers.** (Your `END_GOAL.md` C3 states these numbers correctly. A verification agent flagged a title mismatch; that flag came from *my* mislabeled prompt, not your document — the ID 2202.10054 is right, and it is *not* the "layerwise freezing / transferable shallow weights" paper.)
- **Zhao et al.** [58] `[full:read]`: only a few modules are task-specific and "sensitively alter between tasks," others are shared "as common knowledge," and forgetting is attributed to the former — their FPF/k-FPF methods fine-tune *only* those modules, i.e. **touch the deep, task-specific ones.**

Both are peer-reviewed and both are right within their framing: one is *pretrained* backbones under domain shift, the other *incremental* training from scratch. They are not testing the same thing. **Ledger entry:** `[Contested]`, resolvable by declaring scenario. Your experiment (from-scratch, sequential, 4 conv layers + head) is closer to Zhao et al.'s regime.

---

## 3. THE UNKNOWN COLUMN — where compute actually buys information `[Confidence: Low, directional]`

### 3.1 Long-horizon non-stationary CL with a parametric learner — the gap survives, restated

- CL-Bench [73] `[full:read]` exists, is expert-validated, has six domains with **real concept drift** and an explicit stability/plasticity decomposition (database migration, poker opponent stages), averaged over 5 rollout orderings, 300+ instances. Stream lengths verified: 90 spectrum scans, **19** bug fixes (9 from jazzband/tablib + 10 from jd/tenacity), **5** epidemiology studies, 40 DB questions, 120 poker hands. Exclusion, verbatim and word-for-word confirmed: *"Our initial evaluation focuses on context-based memory paradigms (in-context retention or compaction, retrieval-augmented memory, and structured notepads); **we do not evaluate parametric approaches such as test-time training**, though we seek to add them via community contributions as researchers develop parametric methods for realistic continual learning settings."* Also: "no high-quality benchmark exists to evaluate it"; "continual learning in LLM-based agents remains an open problem."
- **Two corrections you must make:** (a) your notes say "~20 epidemiology studies" — it is **5**; (b) "retrieval ≈ 20 %" is specifically **Mem0 at 20.2 ± 5.9 %**, not a generic RAG row. ICL+Claude Sonnet 4.6 tops the table at gain **25.4 ± 3.6 %** for $30.4, while ACE+GPT-5.4 sits at **8.6 ± 2.5 %** for $62.8 — *the best result in that benchmark is the cheapest, dumbest baseline.*
- The exclusion is self-described as **initial scope, with an explicit invitation for parametric contributions.** That is a coverage gap you can fill, not a design argument against parametric methods.
- C7's **broad** half ("the 2024-2026 literature evaluates long contexts, not drifting tasks") must be **struck**: CL-Bench tests drift, and [74] runs a non-stationary stream.
- [74] does **not** refute the narrow half: no task sequence, no retention metric, and it explicitly disclaims the CL framing.

**Restated claim that survives adversarial checking — use this wording:**

> CL-Bench [73] is the closest existing long-horizon, drifting benchmark and states that it does not evaluate parametric test-time learning; the TTL lineage (Titans, Nested Learning/Hope) [75][76] claims continual-learning capability while its own reported evaluations are static-distribution language modelling and needle-in-a-haystack retrieval. **The capability claim and the evaluation do not match.**

This is a *positive, sourced* observation rather than an absence claim, so it is falsifiable — and **instrumentable**: CL-Bench publishes its stream specs, so running a test-time-learning architecture over their shape is an experiment rather than a survey. Honest constraint: CL-Bench tasks "require frontier-level capability," and "failure modes of smaller models may not be surfaced" [73] — you cannot run that benchmark on 4 GB, only its *shape* on a toy domain.

### 3.2 Genuinely unmeasured, and affordable to measure

I searched hard for each; none has a primary measurement I could confirm.

| # | What nobody has verified | Nearest existing work | Feasible on 4 GB? |
|---|---|---|---|
| U1 | **Pairwise representational comparison of a task-A and task-B checkpoint** — per-layer CKA, and task-vector geometry (cosine, Jaccard support overlap, sign-flip rate, effective rank) computed *between τ_A and τ_B* on a from-scratch CNN | Zhao et al. [58] does within-sequence module attribution with **no correlation coefficient at all** (0 hits for `correl*`/`Spearman`/`Pearson`/`rho` in both the PMLR and arXiv full texts) and no task-vector geometry; the merging papers [69][70] do cosine/overlap but on **parallel** LLM fine-tunes, not sequential CL | **Yes** — your W2-W7 |
| U2 | A measured **cos(g_A, g_B)** value, its distribution, or any correlation between negative cosine and subsequent forgetting | PCGrad states interference *conditions*; no primary number found | **Yes** |
| U3 | **Effective rank / singular spectrum of a single task vector** | DARE's "deltas within 0.002, 90-99 % removable" is magnitude/sparsity, not spectral [61]; the "penultimate-layer effective rank collapse" quote you carry is real but from an **offline-RL** paper whose own conclusion is against it [66] | **Yes** |
| U4 | **Fisher-weighted drift** as a predictor of measured forgetting, on a small CNN | [6] undercuts the premise; [64] shows the estimator isn't canonical; [65] challenges which data to compute it on | **Yes** |
| U5 | **MNIST→Fashion-MNIST as a CL sequence, with forgetting numbers** | Two independent attempts, ten phrasings: nothing. A ResearchGate figure implies some ~2019 paper ran both under CL settings — unidentifiable, not citable | **Yes** — your EXP-1 |
| U6 | **Loss barrier between θ_A and θ_B** in sequential fine-tuning | [63] is about SGD-noise realizations from one init, not task sequences. Leads only: [67][68] `[unverified]` | **Yes** — ~21 forward evals |
| U7 | **GPM-style gradient-subspace escape at tiny scale** | GPM single-lab, unreproduced [50][47] | **Yes** |
| U8 | **Adam optimizer state: continue vs reset at the task boundary** | You already flagged this as "no source in the literature … free to measure" [34 C7]. Still unclaimed | **Yes** |
| U9 | Documented **production deployment** of CL | Two agents, different phrasings: **no verified production deployment found.** Nearest: on-device environmental-sound CL (a benchmark study, not deployment) `[unverified]` | n/a — an absence, not an experiment |
| U10 | CL **theory** at CNN scale | "the classical continual learning problem has proven to be NP-hard in general" is relayed by [1, p.7] `[local:read]` from a citation neither agent opened; likewise the NTK similarity bounds | n/a |
| U11 | Long-sequence **Permuted-MNIST** MLP "None" numbers at 20/50 tasks | Could not confirm; only DER++'s 20-task DIL (40.70) [7] and Hsu et al.'s 78-task queue [43] | Yes if needed |

### 3.3 Open problems the field names itself, with evidence status

All `[local:read]`, so you can cite them honestly.

**De Lange et al., ten desiderata [2, p.16]:** constant memory · no task boundaries · online learning · forward/zero-shot transfer · backward transfer · problem-agnostic · adaptive (unlabeled) · no test-time oracle · task revisiting · **graceful forgetting**. All ten are *asserted as requirements, not evidenced* — precisely the list to attack. Their conclusion is the admission: "the majority of results originates from a highly confined setup, leaving numerous opportunities for further research to reach beyond classification, multi-head evaluation and a task incremental setup" [2, p.18].

**van de Ven et al. [3]:** context/task-identity identification — "has indeed proven to be a difficult problem, and presents a promising avenue for future research"; anchor-point selection — "still largely an open question"; the continuous-task-set formalization — "as far as we are aware this option has not yet been systematically explored"; loss of plasticity — "rapid adaptation is still an open problem."

**Wang et al. [1]:** "A systematic comparison of CRL methods in various contexts is a promising future work" (asserted, no evidence); task-identity prediction as an unsolved sub-problem of CIL that is *itself subject to forgetting*; scaling/sparsity limits of architecture-based methods.

**Yang et al. [13]:** "a significant hurdle in the field is the lack of unified benchmarks and standardized metrics … this fragmentation complicates the comparison of methods across tasks and domains"; most methods assume offline access to the full target dataset while real data streams; current benchmarks are stitched from existing datasets and "lack diversity and real-world complexity."

**FSCIL survey [12]:** "a lack of comprehensive evaluation metrics, **unfairness in experimental conditions**, and inconsistencies with real-world scenarios"; class recurrence, uneven shot counts, cross-domain FSCIL unaddressed. Measured ceiling: best final-session accuracy **57.11 %** (miniImageNet) / **55.88 %** (CIFAR-100) / **63.21 %** (CUB-200), ~24-25 points below base session; **no method retains more than ~70 % of base-session accuracy after 8 sessions.** Incremental *detection* is near the floor (novel mAP 1.1-4.5).

**Bell et al. [4]:** long-term stability of continual pre-training "remains under-explored"; bias-drift detection "largely lacking"; MoE expert specialization — "we still lack a clear understanding of these mechanisms." Note: **zero numeric results in 24 pages** — lowest evidential weight in your CL folder.

---

## 4. FAILED AND REFUTED — do not spend a GPU day on these `[Confidence: High]`

You asked: if an experiment failed, should we perform it? Here is the dead list, with what to do instead.

### 4.1 Methods that cannot be reproduced from published numbers

Mammoth's author-maintained checklist [47] `[full:read]`. **NOT verified** (✗): A-GEM, A-GEM R, BiC, EWC-Online, FDR, GSS, HAL, LUCIR, LwF, LwF-MC, MER, **PNN**, Puridiver, Ranpac, RPC, **SI**, SLCA. **Verified** (✓): DER, DER++, ER-ACE, GDumb, iCaRL, X-DER, Joint, L2P, DualPrompt, CODA-Prompt, MoE Adapters, STAR-Prompt, SPR, TwF, ITA, IEL, CNLL, CGIL, LwS, DAP.

- Decisive entry: **AttriClip — "The original repo was pulled because it did not reproduce."**
- GEM was reproduced only against *the DER paper's* numbers because "Original work requires too much resources."
- Cost signal: SPR took "around 3 1/2 days on a 1080ti for seq-mnist."

**Do not run:** any ✗-listed method as a baseline you expect to match its paper. **Run instead:** the ✓ list (DER/DER++, ER, GDumb, iCaRL, X-DER) — all of which fit your hardware.

### 4.2 Dead experiments (settled verdicts)

| Dead idea | Verdict | Why it's dead | Do instead |
|---|---|---|---|
| "EWC/SI/MAS beats fine-tuning in class-incremental" | **Refuted** | Split-MNIST CIL: EWC 20.01, oEWC 19.96, SI 19.99 vs None 19.90 — all ≈ chance [36] | Negative controls only |
| "GEM/A-GEM's gradient projection buys you something" | **Refuted beyond MNIST** | A-GEM 18.5 ≈ fine-tune 18.5 on CIFAR-10 [45]; GEM's stability gap is *larger* than ER's despite theory predicting escape [5, p.8] `[local:read]`; not reproduced [47] | Skip |
| "A novel regularizer beats replay" | **Not the pattern** | Every no-memory method clusters at fine-tuning level in CIL while replay reaches 45-90 % [45][7] | Compare against replay, never against FT alone |
| "Prompt-tuning / expansion methods are robust" | **Refuted on unseen domains** | they "frequently collapse on unseen data despite high initial benchmarks" [44]; "prompt-tuning approaches completely collapse under large timesteps" [49]; BEEF NaNs at specific seeds [44] | Treat as regime-limited if at all |
| "Generative replay is a cheap option" | **Refuted for you** | partially-trained generator ≈90 % vs fully trained 96 %, and 18-34 min wall-clock vs ~7 min for coreset variants [38]; on 4 GB a hard no; no paper claims it beats buffer replay at matched budget | Reservoir sampling |
| "Bigger buffer → use fancier selection" | **Reversed at scale** | GDumb ≈ ER ≈ MIR at buffer 5000, per Online-LoRA's own appendix [9] `[local:read]` | Sweep and declare buffer size |
| "Continually-trained representations are the point" | **Refuted in the online low-exemplar regime** | fixed random features + linear head beat online-CL SOTA by 5-15 points with **zero stored exemplars** [49] | Always report a random-feature baseline |
| "A 2-task demonstration characterizes forgetting" | **Refuted by explicit statement** | "Showing that a method succeeds on a two-task transfer does not entail that it will work on a longer series" [38] | Go to ≥5 tasks to publish anything |
| "Permuted-MNIST proves your method" | **Refuted as a benchmark choice** | "an unrealistic best case scenario"; entropy 0.003 vs 0.453 [38] | Sanity check only, never headline |
| "MAML-style initialization learning works for CL" | **Refuted, in print** | "a model initialization is not an effective bias for incremental learning"; vanilla MAML with correlated trajectories "performed poorly" [8, Appendix C.1] `[local:read]` | Meta-learn a *representation* [8] |
| "Forward-Forward separates positive/negative phases" | **Self-retracted** | "I have been unable to replicate this result and I now suspect it was due to a bug" [25] `[local:read]`; surviving variant needs "learning rate very low and momentum extremely high" | Do not build a consolidation story on FF |
| "Spiking nets are automatically energy-efficient" | **Refuted/qualified** | needs T ≤ 5 and sr < 5.7 % to match a quantized twin; loses in 10/12 bit-serial configs; 0/12 at η=64 [15] `[local:read]` | Any SNN energy claim must quote that window |
| "Backprop-free local rules scale" | **Refuted** | ImageNet: FA 93.08 vs BP 71.43 error; DTP variants at chance; DFA OOM on 16 GB [23] `[local:read]` | Not a route to your end goal |
| "PEFT makes retraining unnecessary" | **Not established** | [1] consensus "remain vulnerable"; [51] "substantially underperforms full finetuning"; margins inside error bars [9] | Fine as an instrument, not a contribution |

### 4.3 Claims that are pure assertion — do not cite as evidence

`[local:read]` from your folders: LeCun's JEPA paper in its entirety — a self-declared position paper with no experiments and no CL content [27]; MCUNetV3's "on-device lifelong learning" framing — appears once as motivation, no CL experiment, no forgetting metric [28]; PAPER 53's "energy-aware" framing [14]; PAPER 45's hippocampus/neocortex functional identification beyond its own representational statistics [16]; PAPER 17's prediction that random-e-prop would fail — "We are not aware of corresponding demonstrations of failures … although they are likely to exist. Currently, we are not aware of any" [17] (an honest author quoting an untested prediction); PAPER 6's "graceful forgetting" desideratum (its "PackNet is best" claim *is* supported; that one is not) [2]; Bell et al.'s "modern foundation models have achieved superhuman performance on most traditional ML benchmarks, making them obsolete" — no citation at the sentence [4].

---

## 5. Verdict on EXP-1, element by element (go / no-go)

Your design is, in the parts you control, unusually careful — the A2/A3/A4 decomposition, the C4 training-regime control, the load-verification on logits, the confound register. Those are right. The problem is that three of its pillars are not the open questions you believe they are.

### 5.1 Citation defects in the frozen design — repair before you run

| # | What EXP-1 says | What is actually true | Severity |
|---|---|---|---|
| **D1** | §2.2: "[16] reports of task 1 that it 'was suffered from catastrophic forgetting and T1 performance reduced to zero'" — used to justify MNIST→Fashion because "the drop is documented as severe" | **REFUTED.** arXiv:2410.16154 is Bazhenov, Dewasurendra, Krishnan & Delanois, "Unsupervised Replay Strategies for Continual Learning with **Limited Data**" — a biology-inspired replay method whose abstract claims *improved* accuracy. The quoted sentence is unsupported by its title/abstract; two independent agents could not substantiate it [83] `[abstract]` | **High** — your "it will be screaming" premise has no source |
| **D2** | §17.1 **[28]**: "arXiv:2207.02099 / arXiv:2309.00257 — effective rank of the penultimate feature layer, 'observed to drastically collapse during the training'" — underpins W7 | **Right ID, wrong framing.** 2207.02099 is Gulcehre et al., "An Empirical Study of Implicit Regularization in Deep Offline RL." It contains that sentence but concludes **against** it: "a direct association exists only in restricted settings and disappears in the more extensive hyperparameter sweeps"; "studying this association under simplistic assumptions could be highly misleading." 2309.00257 is a single-author federated-learning metrics paper [66][71] `[abstract]` | **Medium** — your W7 anchor argues the opposite |
| **D3** | §14/§17.1 **[26]/[27]**: "nearest verified work reports per-layer task-vector magnitudes for two **sequential** fine-tunes" (2607.27240) and "near-orthogonal (cosine 0.06-0.10) despite ~65 % support overlap" (2607.16062) | **Both real, both mis-framed.** 2607.27240 = "Asymmetric Collapse in Model Merging: When Refusal Overwrites Recognition" — two **parallel** Gemma-3-1B-IT safety fine-tunes (CARES / WildJailbreak) that are then *merged*; cosine 0.011 is genuine. 2607.16062 = "When Model Merging Rivals Joint Multi-Task RL" — Qwen3-8B RL specialists. Neither is a sequential-CL study [69][70] `[abstract]` | **Medium** — and *good news*: correcting this **strengthens** your novelty claim |
| **D4** | §17.1 **[17]** Hiratani 2405.20236 as an empirical prediction for your pair | **Real paper, category error.** Sole-author *analytical* result for a linear teacher-student model, about **retention** only. No CNN, no Fashion-MNIST [80] | **High** |
| **D5** | §17.1 **[15]** Kumar et al. 2202.10054 | **Correct as you have it.** ID and all four deltas verified [62]. A verification agent flagged a title mismatch; that flag came from *my* mislabeled prompt, not your document | None |

### 5.2 Per-element ledger

| EXP-1 element | Already answered? | Evidence | Verdict |
|---|---|---|---|
| **P1** — forgetting ≥40 points on the headline number | **PARTLY — direction uncertain for your pair** | CIL baselines collapse to chance [36][7]; but nobody has published MNIST→Fashion (U5), and the similarity literature leans *against* max-similarity being worst [80][81] | **RUN, as a measurement.** Do not present it as confirming a prediction |
| **P2** — `a[A,B]_probe ≥ 70 %` (information survives, only reporting breaks) | **ANSWERED — and not by you** | Davari et al. CVPR 2022 ran exactly this protocol: re-fit a linear classifier on frozen post-task-2 activations. Verbatim: "a model's representation can change without losing knowledge about prior tasks." On a 2-task Split-CIFAR-10 setup, network accuracy fell **85 % → 63.64 %** while the top-block probe fell only **5.72 points** (85.82 → 80.10), and lower-block probes *improved* (+1.08, +1.26). Their intro pre-states your interpretation: "A permutation of the features leading into the classification heads leads to total catastrophic forgetting as measured by standard approaches … However, this does not correspond to a loss of knowledge about the data." [57] `[full:read]` | **DO NOT SELL AS NOVEL.** Replicate as confirmation (~free). It is TIL-only — they state it "does not consider the important class-incremental setting" — so your DIL-like framing is a real if narrow extension, and they contain no MNIST/Fashion at all |
| **P3** — `head_A < probe` (damage partly representational) | **PARTLY** | Davari's block-wise ΔLP is the same logic at ResNet-18 scale [57]; Kumar's LP-FT is the pretrained analogue [62] | **RUN** — cheap; your answer is dataset-specific |
| **P4** — per-layer cos(τ_A, τ_B) negative in the head | **OPEN at your scale** | Task-vector geometry exists for *parallel LLM merges* [69][70], sign conflicts [60][61]; nothing sequential on a small CNN (U1) | **RUN — genuinely yours** |
| **P5** — ≥70 % of B's Fisher-weighted drift on A's top-decile coordinates | **OPEN, premise contested** | No primary measurement (U4); EWC's authors call Fisher "overconfident" [6]; the estimator isn't canonical [64] | **RUN — with ≥2 Fisher estimators**, pre-committing to report the estimator as a co-variable (your §12.4 plans this; make it mandatory) |
| **P6** — non-monotone curve / stability gap at your scale | **KNOWN phenomenon, OPEN at 242k / 2-task** | [5] spans MLP→slim-ResNet-18; excluded for parameter-isolation methods by construction; **GEM's gap is larger than ER's** | **RUN — highest information-per-second in your battery.** Requires ρ_eval = 1 on the 2,000-image subset; this is the single highest-payoff design change |
| **P7** — LR÷10 cuts forgetting ≥20 points | **SETTLED, direction known** | regime dominates [39][44]; +9-15.8 % from stabilization [39]; and the *width* half is now known too: 32→2048 moves forgetting 36.9→26.7 [40] | **RUN as the control, expect the result.** Keep C4 elevated to co-primary |
| **P8** — order asymmetry A→B vs B→A | **PARTLY SETTLED, contradicted by its own table** | [2] asserts "insignificant" while its Tables 4-5 contain a **16.5-point swing** (SI, SMALL: 23.91 random vs 40.39 hard-to-easy) and 6 points for PackNet `[local:read]`; [4, p.9] says order "can greatly affect retention" | **RUN** — the survey's claim is weaker than the table beside it |
| **P9** — MLP forgets more than CNN | **LIKELY-SETTLED, opposite direction** | "Shallower and wider networks have higher average accuracy and smaller forgetting than deeper and thinner networks"; depth has "no or even negative effect" [40] `[full:read]` | **Reframe.** Your prediction assumes CNN-vs-MLP is a spatial-invariance story; the measured axis is width-vs-depth. Predict width/depth; use MLP-vs-CNN as the confound check |
| **P10** — BatchNorm restoration recovers <10 points | **OPEN, real mechanism** | BN-buffer drift appears in [44]'s regime analysis; no primary magnitude found. Related: GroupNorm was needed because "BN breaks at batch size 1" [10] | **RUN — 30 seconds, no competing claim found** |
| **W8** — loss barrier A→B | **OPEN (U6)** | [63] answers a different question; no CL-specific barrier study found | **RUN**, but **must** include a same-task control barrier, or you cannot separate "different basin" from "different noise realization" |
| **W12** — per-layer CKA between f_A and f_B | **OPEN (U1)** | nothing found; two searches only — "not found by me," not "does not exist" | **RUN** |
| **W13** — GPM-lite subspace escape | **OPEN at your scale** | GPM single-lab, unreproduced [50][47] | **RUN** |
| **C7 (your control)** — Adam state continue vs reset | **OPEN (U8)** | confirmed: no literature claim found | **RUN — free** |
| **Metrics table** — dropping BWT/FM at T=2 | **CORRECT as written** | both degenerate to ΔA at T=2; "undefined" is too strong; `max_{k<K}` has one argument, and every k−1 denominator is what forces it [1, Eqs. 4/5/7] `[local:read]` | **Keep** |
| **10 seeds** | **Correct, but conservative** | weight-init is "<50 %" of bootstrap variance; sampling dominates; claim at bootstrap P>0.75 [42] `[full:read]` | **Upgrade:** vary data split *and* init; report P(outperform), not mean±SD |

### 5.3 The structural problem, stated plainly

With a **single shared 10-unit head** and unrelated label semantics, "task-A accuracy after task B" is not well defined without a task identifier: unit *k* now means "garment *k*", so the head is overwritten *by construction*, and any number you report mixes head-reuse with representational forgetting. Two independent agents landed on this; it is a design point, not a citation. Your A3 and A4 partially handle it. They do not handle the headline A2.

**Also:** at T=2 you cannot make any claim about *cumulative* forgetting, which is what the stability literature actually cares about [38] `[full:read]`.

**Fix — costs almost nothing, converts EXP-1 from confirmation to contribution:** keep MNIST→Fashion as the legible two-task pilot for the **weight-forensics battery** (U1-U4, U6-U8 are all yours), and add a **5-task extension on the same architecture** — MNIST → Fashion-MNIST → KMNIST or EMNIST → SVHN → Rotated-MNIST graded angles — reporting per-layer drift, task-vector geometry, probe decomposition and worst-case metrics at *every* transition. That one change (a) removes the T=2 metric degeneracy, (b) satisfies R5, (c) makes the per-layer profile a *trajectory* instead of two bars, and (d) puts you in the regime where **loss of plasticity becomes visible** [48] — a more interesting phenomenon, and one your current design does not measure at all.

---

## 6. Where the opening actually is

Ranked by (information gained) ÷ (GPU-hours on 4 GB).

1. **EXP-1 as re-scoped in §5.3 — the per-layer drift-vs-forgetting trajectory with task-vector geometry** `[U1-U4/U6-U8]`. Narrow but defensible: Zhao et al. computes no correlations and no task-vector geometry [58] `[full:read]`; the merging papers are parallel 1B-8B LLM fine-tunes [69][70] `[abstract]`; Davari is TIL-only at ResNet scale [57] `[full:read]`. This is an instrument-level contribution, not a paper. Say so.
2. **The plasticity-vs-forgetting discriminator on a cheap 5-10 task stream.** Nobody has measured, at sub-300K parameters, whether late-task accuracy decline is *forgetting* (A's performance fell) or *loss of plasticity* (the network can no longer learn anything). [48] ran it at ImageNet scale and released the code. On your hardware: a day. This is the strongest "small, free-compute, high-leverage" candidate here, and it measures a phenomenon the field agrees is real but has only measured at one scale.
3. **A stability-gap audit at your scale, done continuously (ρ_eval = 1), on methods nobody has evaluated that way.** [5] is one group across seven benchmarks; third-lab replication is genuinely thin (§1.4). Reproducing a known effect from a different lab is citable — it is exactly how [36] and [5] themselves became standard.
4. **Fisher-estimator sensitivity, re-measured as a predictor of drift-weighted damage.** [64] shows estimator choice changes EWC's results; nobody has shown whether it changes *Fisher-weighted drift as a forgetting predictor*. Cheap, and it forecloses a reviewer objection to #1.
5. **What I would not do:** the CL×SNN spike-budgeting niche. Your own two papers already contradict each other there (§2.8, both `[local:read]`), PAPER 53 has no ANN baseline at all, and replicating its CIFAR-10 arm on 4 GB forces T down to ~10-15, which silently changes the very spike-rate comparison the method is about.
6. **What I would defer:** the CL-Bench-shaped experiment (`END_GOAL.md` §7's first opening). The claim needs §3.1's restated wording, and CL-Bench's own limitation is that "failure modes of smaller models may not be surfaced" [73] — you cannot run their benchmark on your GPU. Build the toy-domain analogue first.

---

## 7. Action Plan

- [ ] **Freeze nothing until D1/D2/D4 are repaired.** Edit `EXP1_JOURNAL.md` §2.2 (drop the arXiv:2410.16154 "reduced to zero" justification), §2.2 (re-frame Hiratani as analytical, retention-only, linear-model), and §17.1 [28] (the offline-RL paper argues *against* robust rank-collapse). Show old wording → new wording before changing, per your own rule.
- [ ] **Narrow the §14 novelty claim** to what survives: per-layer sequential task-vector geometry (cosine / support overlap / sign flips / spectral rank) plus drift-vs-forgetting correlation on a from-scratch sub-300K CNN, with probe decomposition. Add [57] Davari CVPR 2022 and [58] Zhao ICML 2023 as required prior art — both are now read in full, so you can differentiate them precisely.
- [ ] **Read two PDFs yourself before writing PRE_REGISTRATION.md:** Davari et al. arXiv:2203.13381 [57] (the A4 collision) and van de Ven, Tuytelaars & Pinto, *Nature MI* 2022 [37] — the benchmarking paper your library does not have, per `PAPERS_MANIFEST.md`'s own coverage-gap note [35].
- [ ] **Add a 5-task extension** (MNIST → Fashion → KMNIST/EMNIST → SVHN → Rotated-MNIST graded angles) on the same architecture; keep 1A as the forensics pilot. Removes the T=2 degeneracy and the shared-head confound; makes plasticity loss observable.
- [ ] **Make continual evaluation the default, not a variant.** ρ_eval = 1 on a fixed 2,000-image subset; report min-ACC, WC-ACC and WF10 alongside AA [5]. Your §8 plans every K=50 steps; tightening to every-step is the highest-payoff single change in the design.
- [ ] **Compute two Fisher estimators** (empirical squared-gradient and expected squared-gradient), declare normalization, report W10 for both [64][6].
- [ ] **Replace mean±SD with a variance-aware comparison.** Randomize data split *and* init; claim an effect only at bootstrap P(outperform) > 0.75 [42]. Ten seeds is the floor, not the answer.
- [ ] **Add three baselines you currently omit:** joint training (already C1 ✓), a **fixed random-feature + linear head** baseline (free; it beats online-CL representation methods [49]), and a **per-task-head TIL view reported explicitly as TIL**, so nobody can read your DIL and TIL numbers as interchangeable [36].
- [ ] **Kill four battery items to buy the 5-task run:** W6 (head-row attribution is narrative, not quantitative), W3 change density (DARE established delta redundancy [61]), W11 live gradient conflict (keep, drop to 5 logged points), Tier-3 causal task-vector negation (defer to EXP-2).
- [ ] **Do not schedule:** any EWC/SI/MAS-as-CIL contender; GEM/A-GEM expecting an advantage; generative replay; Forward-Forward positive/negative-phase consolidation; SNN energy claims outside T≤5 / sr<5.7 %.
- [ ] **Add to your library** (manifest gap list [35] plus this report): [37] Nature MI 2022; [48] Dohare Nature 2024 + code; [57] Davari; [58] Zhao; [40] Wide Networks Forget Less; [49] RanDumb; [44] Reality Check. Delete one of the byte-identical PAPER 36 / PAPER 46.

---

## 8. Open Questions & Caveats

**Q1. Has the stability gap been replicated by an independent group?** Weakly supported — [5] is one lab (across 7 benchmarks, MLP→ResNet-18), corroborated at LLM scale by [77] `[abstract]`. No third-lab replication found. If you build contribution #3 on it, verify that absence yourself: an absence claim is exactly what `END_GOAL.md` §9 item 1 warns about.

**Q2. Is `a[A,B]_probe` high because features survived, or because the probe is under-capacity-constrained?** Unresolved in principle; your nonlinear-probe mitigation is right. Note Davari's linear probe on Split-CIFAR-100 at sequence end is only 64.8-70.5 % [57] `[full:read]` — so **your ≥70 % bar in P2 is dataset-dependent, not a principled threshold.**

**Q3. Masana et al. [41]: my two agents disagreed.** Extracted twice from the same source with different values (FT+E at task 5 as 51.7 and 48.3; class-IL FT at task 10 as 7.9 and ~14.3). Both flagged extractor risk. `[conflict]` — re-verify the PDF before quoting Masana numbers. Task-2 (65.7) and task-10 (37.9) agree; use those two, or none.

**Q4. Boschini et al. list ER and DER++ as *identical* (38.25 / 50.54) [46]** while DER++'s own paper reports 57.74 / 72.70 on a different benchmark [7] `[local:read]`. That identity is almost certainly a transcription or implementation artifact. I used that table's *ordering* (X-DER > iCaRL > LUCIR >> fine-tune) but not its ER/DER++ cells as evidence about those two methods.

**Q5. Several sources here are abstract-only or unreadable and must not be quoted as numbers:** [53] (authors and parameter range could not be extracted), [54] (single-author preprint, no venue, body not read), [55] (no numbers obtained), [64] (no numbers), [75][76] (abstracts only), [80] (NeurIPS PDF returned no extractable text), [81] (abstract), [83] (abstract — I can say the quote is unsupported by the title/abstract, not absent from the text). Also: **`arXiv:2306.13693` is a wireless-communications paper and `2109.05063` is a Lamé-systems math paper** — neither is PLACL or Mammoth; PLACL's and Mammoth's real arXiv IDs could not be confirmed, so cite the repositories.

**Q6. Borsos et al., "Making a Donkey Fly" — could not be verified at all.** A well-known negative audit of CL methods on audio classification; my ID guess resolved to an unrelated speech paper and title searches found nothing. A lead, not a citation.

**Q7. Retractions in CL proper: no documented cases found.** Not "none exist." Two independent searches returned zero confirmed retractions. Your notes already flag SESLR as author-withdrawn-for-data-errors [33 §10]; nothing else surfaced.

**Q8. CL *theory* is thinner than this report implies.** The NP-hardness statement [1, p.7] is relayed from a citation neither agent opened, as are the NTK similarity bounds. Treat theory as background, not evidence.

**Q9. What my sweep structurally could not cover:** continual learning in RL beyond [48] and GDumb's comparison rows; CL for detection/segmentation (only FSCIL-detection floor numbers [12]); and the 2025-26 MoE-of-LoRA cluster, which one agent explicitly named as "the literature most likely to already contain your answer" for frozen-backbone CL — unfetched when budget ran out. If Route 3 of your trilemma is the one you want, start there; I did not.

**Q10. Provenance honesty on this document.** Roughly 60 % of the numbers above are `[local:read]` or `[full:read]`; the rest are tagged. The verification pass caught and reversed **four** errors generated by my own pipeline before publication: a "Spearman ρ≈0.09" in Zhao et al. that does not exist in the paper (0 correlation hits in both full texts); the "near-zero forgetting on MNIST→Fashion" assertion, which had no source at all; the arXiv:2410.16154 attribution, which is an entirely different paper; and the "59K-540K parameters" range, which belongs to a different table in the same source. All four were mine. Expect more.

---

## Methodology

**Depth:** deep. **Agents:** 12 subagents across 4 waves — wave 1: 7 (3 local-PDF miners covering all 35 files, 4 net retrieval); wave 2: 3 gap-fill; wave 3: 1 citation-verification; wave 4: 1 final gap-fill on the similarity contradiction. Plus my own direct local PDF extraction and grepping. **Sources:** 84 (deep target was 30-50).

**Local mining.** All 35 PDFs were text-extracted and read; page cites are to the PDF folio, confirmed to match printed folios where checked. Four caveats: (1) [1], [2], [3] are arXiv/author preprints whose pagination differs from the journal of record — I cite the folio I read; (2) figure-internal axis labels came out scrambled, so **no number here was read off a rendered curve**, except where I explicitly flagged an axis reading as directional (EWC's Fig. 3/4 values [6]); where a paper reports results only in figures (EWC, feedback alignment [21], OML's sine waves [8], predictive coding's CIFAR tables [19]) I said so rather than estimating; (3) [18] was read to p.32 of 40; [2]'s Tables 11-16 were not transcribed; (4) [6]'s supporting-information tables are not present in your 6-page local copy at all.

**Verification.** Every load-bearing number was re-read from primary text by an agent instructed to break the claim, not confirm it. Four claims were reversed by that pass (listed in Q10). I additionally re-verified locally, by direct text extraction: the DER++ Table 2 figures (third independent read, page 6 of PAPER 7), the Hinton self-retraction sentence and the §4.1 deletion (PAPER 13 vs PAPER 25), PAPER 53's timestep and spike-rate configuration, and PAPER 27's T≤5 / sr<5.7 % threshold sentence.

**Outline adaptation.** The default deep-research structure (Status Quo / Emerging Trends / Critical Assessment) was remapped to the ledger spine you chose: §1 Settled ≈ Status Quo; §2 Contested + §4 Failed ≈ Critical Assessment; §3 Unknown + §6 Opening ≈ Emerging Trends. The change is evidence-driven — the field's stable knowledge is disproportionately composed of negative results, so a "current practice" chapter and a "failure modes" chapter would have split the same evidence across two places. §5 (EXP-1 verdict) was added because you requested it and the template has no slot for it. Structural change was under 50 %.

**Confidence assignment.** `[High]` requires ≥1 Tier 1 or 2 source and 3+ agreeing independent sources. A local PDF read counts as one Tier 1 source; a survey *relaying* a number was downgraded rather than counted (Wang et al. contains zero accuracy figures in 19 pages of body text [1]). Recency: methodology foundations ([6] 2017, [36] 2019, [45] 2020) are allowed regardless of age; benchmark and SOTA claims were held to <18 months where recency matters (the 2025-26 TTL/CL-Bench material).

**What I did not do.** I did not read every page of all 35 PDFs at full depth — folders 02 and 03 were mined for CL-relevant experimental claims plus all negative results, and read fully only for PAPER 53, 27, 45, 17, 18, 13/25, 50, 51. No subagent read this conversation; each was briefed from scratch. Independent subagent review of *my synthesis* was not performed — the verification agents checked sources, not my argument.

---

## Bibliography

**Local library — all read from your disk.** `[local:read]` unless noted.

[1] Wang, Zhang, Su & Zhu — "A Comprehensive Survey of Continual Learning: Theory, Method and Application" — TPAMI 2024 — `01-continual-learning/PAPER 4.PDF` (arXiv:2302.00487v3; pp.1-19 read, 20-33 refs) — Tier 1 for taxonomy/definitions; **contains zero accuracy figures**
[2] De Lange, Aljundi, Masana, Parisot, Jia, Leonardis, Slabaugh & Tuytelaars — "A Continual Learning Survey: Defying Forgetting in Classification Tasks" — TPAMI 2021 — `PAPER 6.pdf` (arXiv:1909.08383v3; pp.1-18, 26-29 read; Tables 11-16 not transcribed) — Tier 1
[3] van de Ven, Soures & Kudithipudi — "Continual Learning and Catastrophic Forgetting" — book chapter 2024 — `PAPER 39.pdf` (arXiv:2403.05175v1, complete) — Tier 1
[4] Bell, Quarantiello, Coleman, Li, Li, Madeddu, Piccoli & Lomonaco — "The Future of Continual Learning in the Era of Foundation Models: Three Key Directions" — `PAPER 37.pdf` (complete) — **Tier 3, position paper, no numeric results**
[5] De Lange, van de Ven & Tuytelaars — "Continual Evaluation for Lifelong Learning: Identifying the Stability Gap" — ICLR 2023 — `PAPER 40.pdf` (complete, incl. Appendices A-D, Tables 1-6) — Tier 1
[6] Kirkpatrick, Pascanu, Rabinovich et al. — "Overcoming catastrophic forgetting in neural networks" — PNAS 114(13):3521-3526 — `PAPER 3.pdf` (main text only; **Tables S1/S2 absent from your copy**) — Tier 1
[7] Buzzega, Boschini, Porrello, Baroglio & Coloru — "Dark Experience for General Continual Learning" (DER/DER++) — NeurIPS 2020 — `PAPER 7.pdf`, Table 2 re-extracted from **p.6** — Tier 1
[8] Javed & White — "Meta-Learning Representations for Continual Learning" (OML) — NeurIPS 2019 — `PAPER 43.pdf` — Tier 1
[9] Wei, Li & Marculescu — "Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation" — `PAPER 42.pdf` — Tier 1
[10] Liang et al. — "A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts" — IJCV 2024 — `PAPER 21.pdf` — Tier 1; **no accuracy tables**
[11] Sun, Hofmann & Ke — "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts" — ICML 2020 — `PAPER 22.pdf` — Tier 1
[12] Zhang et al. — "Few-shot Class-incremental Learning for Classification and Object Detection: A Survey" — TPAMI — `PAPER 35.pdf` — Tier 1 (compiles others' numbers)
[13] Yang et al. — "Recent Advances of Foundation Language Models-based Continual Learning" — ACM TIST 2025 — `PAPER 41.pdf` — Tier 2
[14] Meem, Nadid & Mia — "Energy-aware spike budgeting for continual learning in SNNs" — Neuromorph. Comput. Eng. 6 034004, 2026 — `PAPER 53.pdf` (complete) — Tier 1
[15] Yan, Bai, Tang & Wong — "Reconsidering the Energy Efficiency of Spiking Neural Networks" — arXiv:2409.08290v6 — `PAPER 27.pdf` (complete) — Tier 1
[16] Jun, Marupudi, Shah & Varma — "A Neural Network Model of Complementary Learning Systems" — arXiv:2507.11393 — `PAPER 45.pdf` (complete) — Tier 1
[17] Bellec et al. — "A solution to the learning dilemma for recurrent networks of spiking neurons" (e-prop) — Nature Comm. 11:3625, 2020 — `PAPER 17.pdf` (complete) — Tier 2 (no CL content)
[18] Gygax & Zenke — "Theoretical Underpinnings of Surrogate Gradient Learning in Spiking Neural Networks" — Neural Computation 37:886-925, 2025 — `PAPER 18.pdf`, **pp.33-40 NOT READ** — Tier 2
[19] Qi et al. — "Towards the Training of Deeper Predictive Coding Neural Networks" — 2025, under review — `PAPER 12.pdf` — **Tier 3** (figure-table attribution partly unresolvable)
[20] Seely & Gould — "Augmented Lagrangian Predictive Coding" (PC-ALM) — Sakana AI 2025 — `PAPER 19.pdf` — Tier 3
[21] Lillicrap, Cownden, Tweed & Akerman — "Random synaptic feedback weights support error backpropagation" (Feedback Alignment) — Nature Comm. 7:13276, 2016 — `PAPER 14.pdf` — Tier 1; headline numbers in Fig. 3, not text-extractable
[22] Dellaferrera & Kreiman — "Error-Driven Input Modulation" (PEPITA) — ICML, PMLR 162, 2022 — `PAPER 51.pdf`, Table 1 in full — Tier 1
[23] Bartunov et al. — "Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures" — NeurIPS 2018 — `PAPER 50.pdf`, Tables 1-2 — Tier 1 (as negative result)
[24] Bedi & Khandwala — "Testing the Limits of Biologically-Plausible Backpropagation" — Stanford APPHYS293 course report — `PAPER 26.pdf` — Tier 2 (not peer-reviewed)
[25] Hinton — "The Forward-Forward Algorithm: Some Preliminary Investigations" — NeurIPS 2022, **revised** — `PAPER 13.pdf` — Tier 1; contains the §5 self-retraction, verified verbatim
[26] Hinton — same title, **earlier version** — `PAPER 25.pdf` — Tier 3, superseded; still contains the retracted §4.1
[27] LeCun — "A Path Towards Autonomous Machine Intelligence" (JEPA) — 2022 — `PAPER 23.pdf` (62 pp, not ~45 as the manifest says) — **Tier 3 position paper; zero CL/forgetting/replay content**
[28] Lin et al. — "MCUNetV3: On-Device Training Under 256KB Memory" — NeurIPS 2022 — `PAPER 36.pdf` — Tier 2 (byte-identical to `PAPER 46.pdf`)
[29] Abreu et al. — "Neuromorphic Principles for Efficient LLMs on Intel Loihi 2" — ICLR SCOPE workshop 2025 — `PAPER 28.pdf` — Tier 3 (self-described preliminary)
[30] Blouw et al. — "Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware" — 2019 — `PAPER 33.pdf`, Table 1 in full — Tier 2
[31] Strubell, Ganesh & McCallum — "Energy and Policy Considerations for Deep Learning in NLP" — ACL 2019 — `PAPER 1.pdf`, Tables 1-3 — Tier 1
[32] Jegham et al. — "How Hungry is AI? Benchmarking Energy, Water, and Carbon of LLM Inference" — arXiv:2505.09598v6, 2025 — `PAPER 10.pdf` — Tier 2 (GPT-3 and 90 %-of-lifecycle figures cited from others, not measured)
[33] MK — `END_GOAL.md` — local, 2026-09-20 — internal
[34] MK — `EXP-1-CATASTROPHIC-FORGETTING/EXP1_JOURNAL.md` — local, 2026-09-25, design frozen — internal; the object of §5's audit
[35] `PAPERS_MANIFEST.md` — local, 2026-09-20 — internal; its coverage-gap note is still accurate

**Web — settled core and measurement validity.**

[36] van de Ven & Tolias — "Three scenarios for continual learning" — arXiv:1904.07734 — https://ar5iv.labs.arxiv.org/html/1904.07734 — accessed Oct 2026 — Tier 1 — [read in full incl. Tables 4-5]
[37] van de Ven, Tuytelaars & Pinto — "Three types of incremental learning" — Nature Machine Intelligence 2022 — https://www.nature.com/articles/s42256-022-00568-3 — Tier 1 — **[snippet only; no tables retrievable; you do not have this as a PDF]**
[38] Farquhar & Gal (Javadi) — "Towards Robust Evaluations of Continual Learning" — arXiv:1805.09733v3 — https://arxiv.org/html/1805.09733v3 — Tier 1 — [read in full; the two-task and permuted-MNIST critiques are verbatim]
[39] Mirzadeh, Farajtabar, Pascanu & Ghasemzadeh — "Understanding the Role of Training Regimes in Continual Learning" — NeurIPS 2020, arXiv:2006.06958 — https://ar5iv.labs.arxiv.org/html/2006.06958 — Tier 1 — [Table 2 and the 12.9/15.8/9 % deltas read; **per-technique LR-vs-batch-vs-dropout decomposition NOT located — cite the aggregate, not the breakdown**]
[40] Mirzadeh, Chaudhry, Yin, Hu, Pascanu, Gorur & Farajtabar — "Wide Neural Networks Forget Less Catastrophically" — NeurIPS 2022, arXiv:2110.11526v3 — https://arxiv.org/pdf/2110.11526v3 — Tier 1 — [read in full, Tables 1/3/4/5; adversarially re-checked]
[41] Masana, Liu, Twardowski, Menta, Bagdanov & van de Weijer — "Class-Incremental Learning: Survey and Performance Evaluation" — IEEE TPAMI, arXiv:2010.15277v3 — https://arxiv.org/html/2010.15277v3 — Tier 1 — [read; **`[conflict]` between two extractions — re-verify before quoting]**
[42] Bouthillier et al. — "Accounting for Variance in Machine Learning Benchmarks" — MLSys 2021, arXiv:2103.03098 — https://ar5iv.labs.arxiv.org/html/2103.03098 — Tier 1 — [read in full: bootstrap P>0.75; init <50 % of bootstrap variance]
[43] Hsu, Liu, Ramasamy & Kira — "Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines" — arXiv:1810.12488v4 — https://arxiv.org/pdf/1810.12488 — Tier 2 — [read; Tables 3-6]
[44] "Hyperparameters in Continual Learning: A Reality Check" (GTEP protocol) — arXiv:2403.09066v5 — https://arxiv.org/html/2403.09066v5 — Tier 1 — [read; **author list not captured — confirm on the abs page before citing**]

**Web — method families.**

[45] Prabhu, Torr & Dokania — "GDumb: A Simple Approach that Questions Our Progress in Continual Learning" — ECCV 2020 — https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf — Tier 1 — [PDF downloaded and extracted; **table extraction flagged `[conflict]` against [46]**]
[46] Boschini, Bonicelli, Buzzega, Porrello & Calderara — "Class-Incremental Continual Learning into the eXtended DER-verse" (X-DER) — TPAMI, arXiv:2201.00766v2 — https://arxiv.org/abs/2201.00766 — Tier 1 — [read]
[47] AIML-Lab (Università degli Studi di Milano) — "Mammoth: On the reproduction of methods in Mammoth" — https://github.com/aimagelab/mammoth/blob/master/REPRODUCIBILITY.md — Tier 1 (official documentation) — [read in full]
[48] Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood & Sutton — "Loss of plasticity in deep continual learning" — Nature 632(8026):768-774, 21 Aug 2024 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/ (preprint arXiv:2306.13812; code github.com/shibhansh/loss-of-plasticity) — Tier 1 — [open-access full text read]
[49] Prabhu, Sinha, Kumaraguru, Torr, Sener & Dokania — "RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning" — arXiv:2402.08823 — https://arxiv.org/abs/2402.08823 — Tier 1 — [read]
[50] Saha, Garg & Roy — "Gradient Projection Memory for Continual Learning" — ICLR 2021, arXiv:2103.09762 — Tier 1 — [abstract only; **no independent replication found**]
[51] Biderman et al. — "LoRA Learns Less and Forgets Less" — TMLR 2024, arXiv:2405.09673 — https://arxiv.org/abs/2405.09673 — Tier 1 — [read; quotes verbatim]
[52] Wang et al. — "Orthogonal Subspace Learning for Language Model Continual Learning" (O-LoRA) — EMNLP 2023 Findings, arXiv:2310.14152 — Tier 1 — [abstract only, self-reported]
[53] (authors COULD NOT CONFIRM) — "Effect of Scale on Catastrophic Forgetting in Neural Networks" — ICLR 2022 — https://research.google/pubs/effect-of-scale-on-catastrophic-forgetting-in-neural-networks/ (OpenReview `GhVS8_yPeEa`) — Tier 2 — **[abstract only; author list and parameter range not obtained]**
[54] Lee — "The Impact of Model Size on Catastrophic Forgetting in Online Continual Learning" — arXiv:2407.00176 — Tier 3 — **[abstract only; single-author preprint, no venue; body not read]**
[55] Mirzadeh et al. — "Architecture Matters in Continual Learning" — NeurIPS 2022, arXiv:2202.00275 — Tier 1 — **[abstract only; no numbers obtained]**
[56] "Overcoming the Stability Gap in Continual Learning" — arXiv:2306.01904 / OpenReview `A4YlfnbaSD` — Tier 1 — **[unverified — snippet only, not read]**

**Web — mechanistic / weight-level.**

[57] Davari, Asadi, Mudur, Aljundi & Belilovsky — "Probing Representation Forgetting in Supervised and Unsupervised Continual Learning" — CVPR 2022 — https://arxiv.org/html/2203.13381v2 — Tier 1 — **[READ IN FULL: abstract, Tables 4/5/7, Appendix 5.1 — the §5 collision with EXP-1's A4/P2]**
[58] Zhao, Zhou, Long, Jiang & Zhang — "Does Continual Learning Equally Forget All Parameters?" — ICML 2023, PMLR v202 — https://arxiv.org/pdf/2304.04158 · https://proceedings.mlr.press/v202/zhao23n/zhao23n.pdf — Tier 1 — **[both versions read in full; 0 correlation hits; L1-based metrics confirmed]**
[59] Ilharco et al. — "Editing Models with Task Arithmetic" — ICLR 2023, arXiv:2212.04089 — Tier 1 — [abstract read]
[60] Yadav, Tam, Choshen, Raffel & Bansal — "TIES-Merging: Resolving Interference When Merging Models" — NeurIPS 2023, arXiv:2306.01708 — Tier 1 — [abstract read]
[61] Yu, Yu, Yu, Huang & Li — "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch" (DARE) — ICML 2024, arXiv:2311.03099 — Tier 1 — [abstract read; **7B-scale LLMs, not small CNNs**]
[62] Kumar, Raghunathan, Jones, Ma & Liang — "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution" — ICLR 2022, arXiv:2202.10054 — Tier 1 — [abstract read; +2 % ID / −7 % OOD and LP-FT +1 %/+10 % confirmed]
[63] Frankle, Dziugaite, Roy & Carbin — "Linear Mode Connectivity and the Lottery Ticket Hypothesis" — ICML 2020, arXiv:1912.05671 — Tier 1 — [abstract read]
[64] van de Ven — "On the Computation of the Fisher Information in Continual Learning" — ICLR 2025 Blog Track, arXiv:2502.11756 — https://arxiv.org/abs/2502.11756 — Tier 1 — [abstract read; numbers in blog figures not extracted]
[65] Ding, Sharma, Chen, Xu & Ji — "Understanding Fine-tuning in Approximate Unlearning: A Theoretical Perspective" — arXiv:2410.03833 — Tier 1 — [abstract read]
[66] Gulcehre, Srinivasan, Sygnowski, Ostrovski, Farajtabar, Hoffman, Doucet & Pascanu — "An Empirical Study of Implicit Regularization in Deep Offline RL" — arXiv:2207.02099 — Tier 2 — [abstract read; **concludes against a robust rank-collapse→performance association**]
[67] Lubana et al. — "Mechanistic Mode Connectivity" — ICML 2023, PMLR v202 — Tier 1 — **[unverified, not read]**
[68] Kozal et al. — "Continual Learning with Weight Interpolation" — CVPR Workshops 2024 — Tier 1 — **[unverified, not read]**
[69] Choudhary, Rocha, Seo, Sharma & Chaudhary — "Asymmetric Collapse in Model Merging: When Refusal Overwrites Recognition" — arXiv:2607.27240 — Tier 2 — [abstract read; **parallel Gemma-3-1B safety fine-tunes, not sequential CL**]
[70] McClendon — "When Model Merging Rivals Joint Multi-Task Reinforcement Learning: A Task-Vector Geometry Analysis" — arXiv:2607.16062 — Tier 2 — [abstract read; Qwen3-8B RL specialists]
[71] Fu — "Leveraging Learning Metrics for Improved Federated Learning" — arXiv:2309.00257 — Tier 2 — [abstract read; does not establish penultimate-layer collapse in CL]
[72] Szczepański et al. — "X-Splat: Gaussian Splatting for 3D CBCT Generation from Single Panoramic Radiograph" — arXiv:2607.02099 — Tier 2 — [abstract read; **listed only to document that this wrong ID resolves to a dental-imaging paper**]

**Web — pretrained models, test-time learning, benchmarks.**

[73] Asawa, Glaze, Orlanski, Ramakrishnan, Xu, Biswal, Chen, Sala, Zaharia & Gonzalez — "Continual Learning Bench: Evaluating Frontier AI Systems in Real-World Stateful Environments" — arXiv:2606.05661v1, 4 Jun 2026 — https://arxiv.org/html/2606.05661v1 — Tier 1 — [read in full; exclusion sentence, Table 1 and stream lengths verified]
[74] Wang, Sun, Tandon, Gandelsman, Chen, Efros & Wang — "Test-Time Training on Video Streams" — JMLR 26(1):1-29, 2025, paper 24-0439 — https://jmlr.org/papers/volume26/24-0439/24-0439.pdf — Tier 1 — [read in full; explicitly disclaims the CL framing]
[75] Behrouz, Zhong & Mirrokni — "Titans: Learning to Memorize at Test Time" — arXiv:2501.00663v1 — https://arxiv.org/abs/2501.00663 — Tier 1 — [abstract read in full; **NeurIPS 2025 version exists and was NOT read**]
[76] Behrouz, Razaviyayn, Zhong & Mirrokni — "Nested Learning: The Illusion of Deep Learning Architectures" — **arXiv:2512.24695**, submitted 31 Dec 2025 — https://arxiv.org/abs/2512.24695 — Tier 1 — [abstract read in full; **the ID 2512.24215 circulating in earlier drafts is wrong**]
[77] Guo, Fu, Zhang, Zhao & Shen — "Efficient Continual Pre-training by Mitigating the Stability Gap" — arXiv:2406.14833v2 — https://arxiv.org/abs/2406.14833 — Tier 1 — [abstract read; **venue could not be confirmed as TMLR**]
[78] Google Research blog — "Introducing Nested Learning" (7 Nov 2025) · "Titans + MIRAS" (4 Dec 2025) — https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/ · https://research.google/blog/titans-miras-helping-ai-long-term-memory/ — Tier 2 — **[not read; "MIRAS" is not a paper — do not cite it as one]**
[79] Allen-Zhu & Li — "Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws" — arXiv:2404.05405 — Tier 1 — **[NOT read; your `END_GOAL.md` C10 "~2 bits per parameter" must be re-verified against this file; the ID 2405.09707 circulating in earlier drafts is wrong]**

**Web — similarity, statistics, misc.**

[80] Hiratani — "Disentangling and Mitigating the Impact of Task Similarity for Continual Learning" — NeurIPS 2024, arXiv:2405.20236 — https://arxiv.org/abs/2405.20236 — Tier 1 — [abstract read; **analytical linear teacher-student model, retention only; NeurIPS PDF returned no extractable text**]
[81] Goldfarb, Evron, Weinberger, Soudry & Hand — "The Joint Effect of Task Similarity and Overparameterization on Catastrophic Forgetting — An Analytical Model" — ICLR 2024 — https://openreview.net/forum?id=u3dHl287oB — Tier 1 — [abstract read]
[82] Raschka — "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning" — arXiv:1811.12808 — Tier 2 — [abstract read; **substituted for Collier et al., whose ID could not be confirmed — do not cite Collier from this report**]
[83] Bazhenov, Dewasurendra, Krishnan & Delanois — "Unsupervised Replay Strategies for Continual Learning with Limited Data" — arXiv:2410.16154 — Tier 2 — [abstract read; **what your EXP-1 citation [16] actually resolves to**]
[84] Xiao et al. — "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms" — arXiv:1708.07747 — Tier 2 — [abstract, via your manifest [35]]

---

## Source Extracts

Raw data preserved so follow-up sessions do not have to re-fetch it. Each entry is the extraction agent's own report of what it read, with the tier and read-level attached.

### [57] Davari et al., CVPR 2022 — the collision to read first
**Protocol, verbatim:** "we evaluate the adequacy of representations by an optimal linear classifier using training data from the original task. A linear classifier is trained on top of the **frozen activations** of the base network given the training instances of a certain dataset." Representation forgetting defined as "the difference in performance of the LP before and after a new task is introduced." Linear only, no kNN (NME appears only §4.3 as a classifier-free remembering trick). Depth-wise probes per ResNet block B-0…B-5 and VGG B-0…B-3 (Table 7, caption: "Accuracy degradation of LP trained on activations of stages (blocks of convolutions) before and after observing Task 2").
**Benchmarks/scale:** ImageNet→Scenes→CUB→Flowers transfer (ResNet-18 at 64×64); SplitCIFAR-100 (10 tasks); 20-task SplitMiniImageNet; 200-task ImageNet32; unsupervised SimCLR on SplitCIFAR-100; a **2-task SplitCIFAR-10** setting from Ramasesh et al. with ResNet and VGG. Capacity sweep RN18 width 32→128 and RN101; plus an online setting (Table 5). Trained "from scratch without explicit control of forgetting" for the CIFAR/MiniImageNet/ImageNet32 sequences. **No parameter counts reported**; models are ResNet-18/101/VGG class, far above 242k.
**Conclusion, verbatim (abstract):** "a model's representation can change without losing knowledge about prior tasks … model representations trained without any explicit control for forgetting often experience small representation forgetting." **§5:** "a) representation forgetting under naive finetuning in supervised settings is not as catastrophic as other metrics suggest b) without evaluation of features the effects of model size on forgetting and representation learning will be misinterpreted."
**The money result (Table 7, 2-task SplitCIFAR10):** "the observed accuracy decreases from 85% to 63% … However … the accuracy degradation is seen to be only 5.7%, without any CL method applied … This suggests that the representations are still highly useful for Task 1 despite training on Task 2." RN top block B-5 LP 85.82 → 80.10 (Δ −5.72) while network accuracy 63.64; lower blocks B-0/B-1 LP **improved** (+1.08, +1.26); VGG B-3 LP 81.54 → 75.48 with network acc 57.88.
**Pre-states your interpretation (Introduction):** "A permutation of the features leading into the classification heads leads to total catastrophic forgetting as measured by standard approaches … However, this does not correspond to a loss of knowledge about the data."
**Not covered:** zero occurrences of "MNIST" or "Fashion." Stated limitation: "it currently focuses on the task-incremental setting and does not consider the important class-incremental setting." Also: their SplitCIFAR-100 fine-tuning LP at sequence end is only 64.8-70.5 % (Table 4).
**Type:** academic. **Tier 1.** Read in full (9.4k-word full text).

### [58] Zhao et al., ICML 2023 — what it does and does not contain
**Metric formulas, verbatim:** consecutive-epoch change `(1/|θ|)‖θ_{t,n} − θ_{t,n−1}‖₁` and consecutive-task change `C = (1/|θ|)‖θ_{t+1,n} − θ_{t,n}‖₁` — **L1 norm normalized by module parameter count**, not a Frobenius relative ratio. (An earlier agent's Frobenius formula was wrong.)
**Correlation search across BOTH the PMLR and arXiv full texts:** `correl*` = 0 hits, `spearman` = 0, `pearson` = 0, `rho` = 0, `r = 0.xx` = 0. **No correlation between a drift metric and a forgetting metric is reported anywhere in the paper.** The "ρ≈0.09" figure that circulated in my own pipeline is not in this source and must not be cited.
**What they do instead:** module ranking by "sensitivity to task shift," established by **intervention** (which module group to fine-tune with FPF / k-FPF), not by correlating drift against forgetting.
**Conclusion, verbatim:** "only a few modules are more task-specific and sensitively alter between tasks, while others can be shared across tasks as common knowledge. Hence, we attribute forgetting mainly to the former."
**Benchmarks observed in full text:** "10 disjoint classification tasks"; "five binary classification tasks" (§H); domain sequence "Sketches → Cartoons → Paintings → Photos" (PACS); a 2-hidden-layer × 100-ReLU MLP used as the §3 dynamics toy. ResNet/VGG parameter counts could not be recovered by text extraction.
**Not present:** linear probing, task-vector cosine, support overlap, sign-flip rates.
**Type:** academic. **Tier 1.** Read in full, both versions.

### [40] Mirzadeh et al., Wide Neural Networks Forget Less
**Table 1, MLP on Rotated-MNIST**, 2-layer, widths 32/128/512/2048, 5 tasks at 0/22.5/45/67.5/90°, 5 epochs/task, 5 seeds, naive fine-tuning: avg accuracy 65.9 ±1.00 → 75.2 ±0.34; avg forgetting 36.9 ±1.27 → 26.7 ±0.50 (intermediates 31.5, 29.6); "Learning Accuracy" 95.5 → 96.6; "Joint Accuracy" 91.2 → 94.0.
**Table 3 (depth × width grid):** depth-2 width-256 = 269.32K params → 71.1 / 31.4; width-512 = 669.70K → 72.6 / 29.6.
**Appendix Table 5:** 3-layer MLPs 59.2K (64-64-64) → 540.2K (256-256-1024) — **this is the source of the parameter range, not the main sweep**, and the paper warns "simply increasing the number of parameters is not necessarily helpful."
**Split-CIFAR-100, WideResNet-10:** 8× width → first-task forgetting after the 20th task 42 % → 31 %.
**Depth claim, verbatim:** "We empirically demonstrate that increasing the width alone reduces catastrophic forgetting significantly, while it's not the case for depth"; "achieving over-parametrization through depth has no or even negative effect on mitigating forgetting"; Table 3/4 captions: "Shallower and wider networks have higher average accuracy and smaller forgetting than deeper and thinner networks."
**Generality:** the width benefit persists under ER and A-GEM.
**Type:** academic. **Tier 1.** Read in full; adversarially re-checked against my own brief, which had the parameter range misattributed.

### [36] van de Ven & Tolias, Three scenarios
**Definitions, verbatim:** TIL = "models are always informed about which task needs to be performed"; DIL = "task identity is not available at test time. Models however only need to solve the task at hand"; CIL = "models must be able to both solve each task seen so far and infer which task they are presented with."
**Setup:** all models from-scratch MLPs (2×400 hidden for Split-MNIST, 2×1000 for Permuted-MNIST, ReLU, Adam; lr 0.001/2000 iters and 0.0001/5000 iters), no pretraining. "None" framed as the lower bound, "Offline" (joint) as the upper bound. Regularization methods such as EWC and SI "completely fail" when task inference is required.
**Numbers (avg test accuracy, mean ± std):** Split-MNIST None: TIL 87.19 (±0.94), DIL 59.21 (±2.04), CIL 19.90 (±0.02); Offline 99.66 / 98.42 / 97.94. Permuted-MNIST None: TIL 81.79 (±0.48), DIL 78.51 (±0.24), CIL 17.26 (±0.19); Offline 97.68 / 97.59 / 97.59. Split-MNIST CIL: EWC 20.01 (±0.06), Online EWC 19.96 (±0.07), SI 19.99 (±0.06), LwF 23.85 (±0.44), DGR 90.79 (±0.41), DGR+distill 91.79 (±0.32), iCaRL (budget 2000) 94.57 (±0.11). Split-MNIST TIL: XdG 99.10, EWC 98.64, SI 99.09, LwF 99.57, DGR 99.50. Permuted-MNIST CIL: EWC 25.04, Online EWC 33.88, SI 29.31, LwF 22.64, DGR+distill 96.38, iCaRL 94.85. Permuted TIL: LwF 69.84 (LwF *underperforms* None+noise here), XdG 91.40, DGR+distill 97.51.
**Important negative:** this paper contains **no Rotated-MNIST, no Split-CIFAR-10 and no TinyImageNet results** — a widely repeated misattribution worth remembering before you cite it.
**Type:** academic. **Tier 1** (foundational).

### [38] Farquhar & Gal, Towards Robust Evaluations
"Pixel permutation represents an unrealistic best case scenario"; "the sensor input in a Mars rover will never be permuted no matter what terrain one moves onto"; "The derivative of the likelihood term … is biggest for confident but false predictions"; "having digits resembling previously observed ones in the Split setting led to much larger likelihood-term gradients"; "In the permuted setting, however, the prior term is relevant throughout the training"; "VCL alone completely forgets old tasks"; "recent leading approaches fail even on simple datasets"; prior-focused methods "appear to succeed, but have major blind-spots."
**Quantitative:** prediction entropy early in task B — 0.003 (Split) vs 0.453 (Permuted). Task-boundary auto-detection AUC: VGR 1.0 vs VCL 0.76. Generative replay: partially trained ≈90 % vs fully trained 96 %. Wall-clock: coreset variants ~7 min vs generative replay 18-34 min.
**Two-task critique, verbatim:** "This is an interesting challenge, but significantly simpler than a longer series"; "Showing that a method succeeds on a two-task transfer does not entail that it will work on a longer series."
**Five recommended criteria:** preserve input similarity across tasks · single prediction head · no test-time task labels · no unconstrained rehearsal · sequences longer than two tasks. Plus: report speed–performance *frontiers*, because "representing the trade-offs within each approach gives a fairer reflection."
**Type:** academic. **Tier 1.**

### [44] Hyperparameters in Continual Learning: A Reality Check (GTEP)
Protocol: tune hyperparameters by maximizing the harmonic mean of final + average accuracy over 30 sampled configurations and 5 task orders, on ResNet and ViT-B/16, across ImageNet, CIFAR, CUB, ImageNet-R, ImageNet-A. Methods audited — non-pretrained: Replay, iCaRL, WA, BiC, PODNet, DER, FOSTER, BEEF, MEMO; pretrained: L2P, DualPrompt, CODA-Prompt, Adam-Adapter, Ranpac, EASE.
**Verbatim findings:** "Traditional regularization and exemplar-replay techniques demonstrate markedly superior cross-scenario generalization compared to modern expansion and prompt-based frameworks, which frequently collapse on unseen data despite high initial benchmarks"; "Cross-dataset validation consistently favors foundational approaches over recently published architectures"; "optimized hyperparameters boost initial tuning scores but trigger sharp performance declines when applied to unseen evaluation domains"; "BEEF constantly returns NaN in training loss at specific seeds"; "Ranpac suffers from significant instability in certain tasks, resulting in a substantial increase in standard deviation."
**Caveat:** author list was not captured by the fetcher — confirm on the abs page before citing.
**Type:** academic. **Tier 1.**

### [47] Mammoth REPRODUCIBILITY.md
**NOT verified (✗):** A-GEM, A-GEM R, AttriClip, BiC, EWC Online, FDR, GSS, HAL, LUCIR, LwF, LwF-MC, MER, PNN, Puridiver, Ranpac, RPC, SI, SLCA.
**Verified (✓):** DER, DER++, DER++ LiDER, ER-ACE (+variants), GDumb, GDumb LiDER, iCaRL, X-DER (+CE/RPC/STAR), Joint, L2P, DualPrompt, CODA-Prompt, MoE Adapters, STAR-Prompt, SPR, TwF, ITA, IEL, CNLL, CGIL, LwS, DAP, CLIP.
**Notes, verbatim:** AttriClip — "The original repo was pulled because it did not reproduce." GEM — "Original work requires too much resources. We reproduced the results in *Dark Experience for General Continual Learning*." ER — "predates modern benchmarks." SPR cost — "Training takes around 3 1/2 days on a 1080ti for seq-mnist."
Repo: github.com/aimagelab/mammoth · docs aimagelab.github.io/mammoth (docs site is a navigation index — **no results tables retrievable**).
**Type:** official project documentation. **Tier 1.**

### [48] Dohare et al., Nature 632 (2024)
"standard deep-learning methods gradually lose plasticity … until they learn no better than a shallow network." Continual ImageNet binary tasks: "the conventional backpropagation algorithm loses plasticity at all step sizes," while "continual backpropagation, L2 regularization and Shrink and Perturb algorithms maintain plasticity, apparently indefinitely." "learned up to 88% correct on the test set of the early tasks"; preprint: "dropped from 89% accuracy on an early task down to 77%, about the level of a linear network, on the 2000th task." CIFAR-100 incremental: "By the end … 5% lower than the retrained network (a performance drop equivalent to that of removing a notable algorithmic advance, such as batch normalization)"; "The final accuracy of continual backpropagation on all 100 classes was 76.13%." Mechanism: "the number of network units that are active less than 1% of the time increases rapidly for the base deep-learning system"; fix = reinitialize least-used units, "typically fewer than one per step." Shrink and Perturb = "L2 regularization plus small random changes in weights at each step." Loss occurred "with a wide range of deep network architectures, optimizers, activation functions, batch normalization, dropout."
**Citation hygiene note I had to apply to my own brief:** "continual backpropagation" is introduced **inside** this Nature paper; arXiv:2108.06325 ("Continual Backprop: Stochastic Gradient Descent with Persistent Randomness", Dohare, Sutton & Mahmood, 2021) is the verified predecessor. I could **not** verify a separate 2024 Nature article on "spontaneous, task-agnostic drift" — EuropePMC returns exactly two Dohare+plasticity records (the Nature paper and its Research Square preprint), and the words "spontaneous", "stateless", "task-agnostic" do not occur in the Nature full text. **Do not cite a second Nature paper.** Also: the "89 % → 77 %" figure is from the *preprint* abstract (DOI 10.21203/rs.3.rs-3256479v1), not the Nature text.
**Type:** academic. **Tier 1** (foundational). Code: github.com/shibhansh/loss-of-plasticity.

### [73] CL-Bench, arXiv:2606.05661v1
Six domains, 300+ instances, 5 rollouts per task in potentially different instance orders, "gain metric" isolating learning from prior capability, scoring decomposed into stability vs plasticity terms.
Systems evaluated: full-context ICL, ICL Notepad, Mem0, ACE, and headless coding agents (Claude Code, Codex) — all prompt/memory substrates over five **frozen** API models (Claude Opus 4.7, Sonnet 4.6, Gemini 3.1 Pro, Gemini 3 Flash, GPT-5.4). "Smaller models and open-weight systems are outside the scope of this paper." "fine-tuning" appears only inside a cited reference title.
Table 1: ICL + Claude Sonnet 4.6 = rank 1, normalized reward 22.3 ±4.1 %, **gain 25.4 ±3.6 %**, $30.4. Mem0 + GPT-5.4 (top-k=10 retrieval) = gain **20.2 ±5.9 %**, reward 15.1. ACE + GPT-5.4 = rank 9, gain **8.6 ±2.5 %**, $62.8.
Stream lengths: RF spectrum ≈90 scans (reward = mean IoU across all 90); **19** bug fixes (9 from jazzband/tablib + 10 from jd/tenacity); **5** epidemiology studies ("The analyst receives data from five independent studies released one at a time"); 40 natural-language DB questions (N=40); 120 poker hands across five stages.
Abstract, verbatim: "agents frequently overfit to immediate observations or fail to reuse knowledge across instances, and dedicated memory systems do not fix this — in fact, naive ICL outperforms systems dedicated to memory management." Also "no high-quality benchmark exists to evaluate it"; "continual learning in LLM-based agents remains an open problem."
Limitations, verbatim: "Our initial evaluation focuses on context-based memory paradigms (in-context retention or compaction, retrieval-augmented memory, and structured notepads); **we do not evaluate parametric approaches such as test-time training**, though we seek to add them via community contributions as researchers develop parametric methods for realistic continual learning settings." Plus: "Task sequences are on the order of tens of instances"; "because CL-Bench tasks require frontier-level capability to perform non-trivially, failure modes of smaller models may not be surfaced"; limitation "higher costs and runtime."
**Type:** academic. **Tier 1.** Read in full.

### [74] TTT on Video Streams, JMLR 26(1):1-29 (2025)
Submitted 3/24, revised 12/24, published 1/25; Wang*, Sun*, Tandon, Gandelsman, Chen, Efros, Wang; Editor Samy Bengio; CC-BY 4.0. Online TTT-MAE initializes f_t from f_{t−1} and gradient-updates on the current frame plus a short sliding window; "each video is treated as an independent unit." Datasets: KITTI-STEP + newly collected COCO Videos ("We collected 3 videos, each about 5 minutes, annotated by professionals … each of the 3 videos alone contains more frames, at the same rate, than all of the videos combined in the KITTI-STEP validation set"). Tasks: semantic / instance / panoptic segmentation, colorization.
**Numbers:** COCO Videos AP 37.6 (online) vs 33.6 (offline oracle) vs 16.7 (fixed model, independent frames); PQ 21.7 / 19.6 / 13.9; KITTI-STEP mIoU 55.4 / 54.3 vs 53.8 / 52.5; 4.1 s per frame on one A100.
**Disclaimers, verbatim:** "Our streaming setting is different from those commonly studied by the continual learning community, because it **does not have distinct splits of training and test sets**." No retention metric — every number is instantaneous accuracy at the current frame, aggregated over the test set; earlier frames are never re-tested. Forgetting reframed positively: "The optimal explicit memory needs to be short term – in plain language, some amount of forgetting is actually beneficial"; "our sliding window can be viewed as a replay buffer, and limiting its size can be viewed as a form of forgetting."
**Type:** academic. **Tier 1.** Read in full.

### [7] PAPER 7 — DER++, local read of Table 2 p.6 and Table 3 p.7
**Setup:** Task-IL + Class-IL on CIFAR-10 and Tiny-ImageNet (5 tasks × 2 classes; 10 tasks × 20 classes), fixed class order. Domain-IL: Permuted- and Rotated-MNIST, **20 tasks** each. Novel MNIST-360 for general CL: stream of 2-digit batches {0,1},{1,2}…, increasing rotation, digit 9 excluded, 9 class-pairs visited 3×, each example seen once. Models: FC 2 hidden layers × 100 ReLU for MNIST variants; **ResNet-18, NOT pretrained** for CIFAR/TinyImageNet. Buffers 200 / 500 / 5120 (MNIST-360: 200/500/1000), reservoir sampling, no task boundaries. α=β=0.5 ("not overly sensitive"); SGD for all methods; 1 epoch/task (MNIST), 50 (S-CIFAR-10), 100 (S-TinyImageNet); random crops + flips applied to stream *and* buffer; grid search on a 10 % validation split; **averaged over 10 runs**; Titan X, DER ≈2.5 h on S-CIFAR-10.
**Table 2 (buffer 500).** S-CIFAR-10 Class-IL — JOINT 92.20, SGD 19.62 (±0.05), oEWC 19.49, SI 19.48, LwF 19.61, ER 57.74, GEM 26.20, A-GEM 22.67, iCaRL 47.55, FDR 28.71, GSS 49.73, HAL 41.79, **DER 70.51, DER++ 72.70**. S-CIFAR-10 Task-IL — JOINT 98.31, ER 93.61, DER 93.40, DER++ 93.88, PNN 92.16 (buffer 500) / **95.13 at buffer 200**. S-TinyImageNet CIL — JOINT 59.99, SGD 7.92 (±0.26), ER 9.99, DER 17.75, DER++ 19.38; at 5120: ER 27.40, DER++ 39.02. P-MNIST — JOINT 94.33, SGD 40.70 (±2.33), oEWC 75.79, ER 80.60, DER++ 88.21. R-MNIST — JOINT 95.76, SGD 67.66 (±8.53), ER 88.91, DER++ 92.77. Task-IL TinyImageNet SGD 18.31 (±0.68), JOINT 82.04.
**Table 3, MNIST-360 (p.7, buffers 200/500/1000):** JOINT 82.98 ±3.24 · SGD 19.09 ±0.69 · ER 49.27/65.04/75.18 · MER 48.58/62.21/70.91 · A-GEM-R 28.34/28.13/29.21 · GSS 43.92/54.45/63.84 · DER 55.22/69.11/75.97 · DER++ 54.16/69.62/76.03.
**Where it loses (all `[local:read]`):** DER loses to plain ER in Task-IL at buffers 500 (93.40 vs 93.61) and 5120 (95.43 vs 96.98); DER++ also loses at 5120 (96.12 vs 96.98) — "in Task-IL, DER performs on par with ER on average." DER++ worse than DER at buffer 200 on S-TinyImageNet CIL (10.96 vs 11.87) and MNIST-360 (54.16 vs 55.22). PNN beats every DER variant in CIFAR-10 Task-IL at buffers 200 and 500. MER omitted from Table 2 for intractable runtime (300 h vs 2.5 h). GEM/GSS/HAL unrunnable on TinyImageNet; PNN/iCaRL/LwF unrunnable in Domain-IL. FDR **degrades as the buffer grows** on S-CIFAR-10 CIL (30.91 → 28.71 → 19.70), which the authors link to FDR's large Tr(F).
**Unresolved per the paper:** no transfer metrics in the main text; the flat-minima→robustness link is stated as a **conjecture**; the "reservoir-sampled sub-optimal logits don't hurt" claim rests on "we empirically observed."

### [5] PAPER 40 — stability gap, local read (complete)
Table 1 (CIL Split-MiniImagenet, ER, 5 seeds): ACC 32.9 ±0.8, FORG 32.3 ±1.0, **min-ACC 0.5 ±0.2** at ρ_eval=100; WC-ACC 4.1 → 7.1 as ρ_eval goes 10²→10³; min-ACC 0.5 → 3.6.
Table 3 (ρ_eval=1, min-ACC / WC-ACC / WF10 / WF100): Split-MNIST 73.0/77.7/18.4/21.2 · **Split-CIFAR10 0.0 ±0.0**/17.9/71.1/76.0 · Split-MiniImagenet 0.5/4.1/56.6/64.6 · Mini-DomainNet 9.6/13.7/15.1/17.8.
Table 2 (task-similarity control, ER, Rotated-MNIST, 3 tasks): ϕ=10° → ACC 96.6, min-ACC 94.3, FORG 0.7, WF10 3.5; ϕ=80° → ACC 90.0, min-ACC 69.2, FORG 9.8, WF10 17.8. Their framing: "ACC declines with 6.6%. However, the effect on the stability gap is substantially larger as the min-ACC drops from 94.3% to 69.2%, a 25.1% decrease … This indicates both that i) the standard metrics fail to capture the effects of the stability gap, and ii) the stability gap increases significantly for larger distribution shifts."
Table 5 (online): Split-MNIST ACC 91.6 but WC-ACC 58.9, min-ACC 50.8, FORG 5.3, **WF10 39.9** — "the WF10 indicates the sheer performance drops of nearly 40% whereas the standard task-based FORG only indicates 5% forgetting"; Split-CIFAR10 "an ACC is maintained of 41.8%, whereas the min-ACC shows that all tasks drop to zero at least once in the stream." Table 6 fine-tuning control: Split-MNIST ACC 19.5, FORG 99.7, WF10 86.1.
Mechanism (Eq. 8): ∇L = α∇L_plasticity + (1−α)∇L_stability; at a task transition ‖∇L_T1‖ ≈ 0 so ‖∇L_stability‖ ≈ 0; ER buffer gradients go near-zero (attributed to Verwimp et al. 2021); for LwF models are identical at the first update, for EWC/SI θ = θ* → ‖∇L_stability‖ = 0. **Reversal of a prior belief, verbatim:** "In contrast to the prior belief of ∇L_stability maintaining prior knowledge, we find that stability is at least partially preserved by means of relearning, leading to the recovery of prior knowledge."
GEM: predicted to escape the gap from the angle-only constraint; measured **worse** than ER — "the stability gap of GEM is significantly larger than ER."
Metric caveats they attach: min-ACC uses 1/(k−1) so is undefined for one task; "We also considered the min-ACC as a measure for the stability gap, but as it often drops to zero accuracy after the first task, it doesn't provide further insights"; FORG and WF10 **decorrelate** as tasks get harder — Pearson ρ = 0.86 (Split-MNIST), **−0.37 (Split-CIFAR10)**, 0.61 (Split-MiniImagenet).
Scope: parameter-isolation methods excluded — "for methods where no changes in performance of learned tasks is allowed, a stability gap cannot be present either." Buffer sweep: "the stability gap persists for all memory sizes, even when storing all samples with M = |S| which approximates joint learning."
**Recipe:** evaluate every ρ_eval = 1 iteration on 1k held-out samples per task; WC-ACC_{t|T_k} ≤ ACC_k is guaranteed (Eq. 7). Failure mode to copy: at ρ_eval=100 on Split-CIFAR10 they "found in some runs … to completely miss the accuracy drop to zero of T4 for 2 out of the 5 seeds."

### [2] PAPER 6 — De Lange survey, own experiments (complete tables)
Tiny ImageNet, 10 tasks × 20 classes, 3.51 M-param VGG-style CNN, random order, buffer 4.5k, avg accuracy (forgetting) %: finetuning 21.30 (26.90) · joint* 55.70 · **PackNet 49.13 (0.00)** · HAT 43.57 (0.00) · SI 33.93 (15.77) · EWC 42.43 (7.51) · MAS 46.90 (1.58) · mode-IMM 36.89 (0.98) · LwF 41.91 (3.08) · EBLL 45.34 (1.44) · R-PM 4.5k 36.09 (10.96) · R-PM 9k 38.69 (7.23) · R-FM 4.5k 37.31 (9.21) · R-FM 9k 42.36 (3.94) · GEM 4.5k 45.13 (4.96) · GEM 9k 41.75 (5.18) · **iCaRL 4.5k 47.27 (−1.11)** · iCaRL 9k 48.76 (−1.76). Random guess 5 %.
Capacity sweep, best over all configs: iCaRL 9k 49.94 (WIDE) · PackNet 55.96 (WIDE+dropout) · MAS 48.98 (BASE+dropout) · LwF 48.11 (WIDE, weight decay) · EBLL 48.17 · EWC 45.13 (**SMALL**) · SI 43.74 (dropout) · HAT 44.19 · mode-IMM 42.41 · GEM 45.27. "WIDE models obtain significant better results (on average 11% better over all methods)"; "a modest 0.89% more forgetting for the SMALL model compared to the BASE model"; DEEP is worst (joint* 55.70 → 51.04).
Long Tiny ImageNet (40 tasks × 5 classes, BASE, Table 20), avg acc / then a novel SVHN task for plasticity: joint* 79.40 / 92.76 · finetuning 24.15 (49.89) / 93.68 · **PackNet 67.52 (0.00) / 26.62** · HAT 63.30 (0.00) / 81.40 · iCaRL 67.10 (1.53) / 82.27 · LwF 64.03 (3.61) / 88.54 · MAS 60.43 (8.45) / 88.41. "PackNet to severely deteriorate in plasticity, obtaining only 26.62% compared to near 90% accuracy for the other methods"; "iCaRL accuracies superseding PackNet … from task 15."
**The method-hyperparameter leak warning, §4 p.4, verbatim:** "These hyperparameters are in many cases found via a grid search, using held-out validation data from all tasks. However, this inherently violates the main assumption in continual learning, namely no access to previous task data. This may lead to overoptimistic results, that cannot be reproduced in a true continual learning setting." Their fix: Algorithm 1, Maximal Plasticity Search → Stability Decay (p = 0.2, α = 0.5).
**The consensus admission, p.2, verbatim:** "There is limited consensus in the literature on experimental setups and datasets to be used. Although papers provide evidence for at least one specific setting of model architecture, combination of tasks and hyperparameters under which the proposed method reduces forgetting and outperforms alternative approaches, there is no comprehensive experimental comparison performed to date."
**Ordering:** asserted "insignificant" (pp.2, 16, 18) — but Tables 4-5 (p.11) contain a 16.5-point swing (SI, SMALL: 23.91 random vs 40.39 hard-to-easy) and 6 points for PackNet (BASE: 49.13 vs 43.17). **The claim is weaker than the table next to it.**

### [14] PAPER 53 — spike budgeting, full local read
**Mechanism:** (1) class-balanced replay buffer, current-task CE + 0.5-weighted replay CE, replay minibatch capped at 32; (2) learnable layer-wise scalars β (decay) and V_thr; (3) the "spike budget" = a **proportional controller on a regularizer coefficient**, not a budget: `Δλrate = η(r_spike − r_target)` (eq. 8), `L_total = L_task + λrate(r_spike − r_target)²` (eq. 9). The squared penalty is **bidirectional** — below target it *releases* the constraint. λrate clipped to [0, 0.8]; r_target = 0.05; λrate init 0.15; η = 0.08; 5-batch moving average (App. A.4). Training-time only, zero inference overhead. Configs: C0 naive / C1 replay / C2 +learnable / C3 +controller / C4 all.
**Protocol:** class-IL, no task ID at inference. 5×2 tasks for MNIST, CIFAR-10, N-MNIST, CIFAR-10-DVS; 4+4+3 (11 classes) DVS-Gesture. **T = 25/50/50/50/60** respectively (verified by grep: "encoding with T = 25 (MNIST) or T = 50 (CIFAR-10)"; "using T = 50 for N-MNIST and CIFAR-10-DVS and T = 60 for the…"; "whereas DVS-Gesture requires T = 60 to cover full gesture sequences"). Poisson encoding for frames, native DVS for events; snnTorch LIF, Fast-Sigmoid surrogate k=25. Buffers 2000 / 10000 (= 20 % of train set) / 440. Architectures: Linear-784→128→10; VGG-5 spiking CNN; 4-block ConvSNN with 16,384-dim flatten. "A few GPU-hours per configuration on a single modern accelerator."
**Table 2, C1 → C4 (ACC / Forgetting / BWT / Spike %):** MNIST 93.44±1.40→95.75±0.16 / 7.17→4.40 / −2.45→−1.65 / 15.31→8.07 · N-MNIST 92.27±2.43→94.07±1.35 / 6.09→4.06 / −8.59→−5.37 / **0.25→2.70** · CIFAR-10 59.50±1.58→61.26±0.79 / **33.56→33.56** / −60.37→−60.16 / 37.50→**30.90** · CIFAR-10-DVS 45.52±4.24→49.68±0.93 / 37.11→29.51 / −38.19→−25.07 / 1.18→7.25 · DVS-Gesture 74.48±0.74→**91.93±1.85** / 25.35→5.90 / −25.35→−5.90 / 0.48→0.72.
**Negatives:** "No ANN or non-spiking baselines are used." Table 5 quotes rivals (HLOP 95.15 % ACC / BWT −1.30 %; SOR-SNN 80.12 %) then concedes "direct numerical ranking is not meaningful." Per-seed (Table C5) forgetting identical for C1 and C4 on CIFAR-10 (32.49/33.15/35.05) while accuracies differ. Table 3: C3 accuracy 92.40 vs C1 93.44, forgetting 8.39 vs 7.17 — "marginally decreases accuracy (−1.04% points)". Seed reversals: CIFAR-10 s42 60.35 vs 60.95; CIFAR-10-DVS s44 45.57 vs 46.36. Fashion-MNIST (Table C6): C1 ≈79.7 → C4 ≈80.5, forgetting 21.03 → 19.66; C3 s43 = 76.31, worse than replay by 3+. Hedged language: helps "in successful runs" / "whenever training remains stable." Acknowledged confound: different architectures per dataset, so cross-dataset spike comparisons are confounded — "claims rest on within-dataset C0–C4 only."
**Their future work = candidate experiments:** same-architecture cross-modal runs; automatic gain tuning; task-free/open-world adaptive targets; generative replay to cut memory; **buffer-size sweep (explicitly skipped)**; real chip energy.

### [15] PAPER 27 — SNN energy, full local read
Fair baseline = "QNN-SNN twin": rate-coded SNN at T timesteps ↔ same-architecture QNN with ⌈log₂(T+1)⌉-bit activations (Theorem 1). Core asymmetry: an event-driven SNN needs **T×sr weight accesses per weight** vs an ANN's 1, and data movement dominates.
**Threshold, verbatim (re-grepped):** "For example, under typical neuromorphic hardware conditions, SNNs with moderate time windows (T = 5) require an average spike rate (sr) below 5.[7]%" — with T ≤ 5 also stated. Breakeven sr: 0.297 (T=1) → 0.082 (T=3) → 0.057 (T=5), 8-bit, N_src=4096. Sparse transmission stops paying above **sr ≈ 0.12-0.17**.
**E_SNN/E_QNN measured:** VGG16/CIFAR-10 0.987 (T=3), 1.026, 1.406, 1.824, 2.243, 1.966 (T=8); CIFAR-100 1.262 already at T=3; ResNet-18 1.071 / 1.023; RepVGG/ImageNet 2.232; GLUE/SST-2 0.996 → 1.500 → 2.338 → 3.762 at T=1,3,7,15 (~1.500 across all 7 GLUE tasks at T=3); spiking Llama-2-7B (B=64, S=128, T=15) **3.793**, matching the asymptotic law E_SNN/E_QNN ≈ T/⌈log₂(T+1)⌉. Bit-serial QNN comparator: SNN loses in 10 of 12 configs. Hardware-η sweep: SNN lower in 12/12 at η=1, only 1/12 at η=12, **0/12 at η=64**. Mapping: parity at khop≈0.46 (no reuse), 0.05 with strong reuse. Sensitivity: crossover moves 3.41-10.80 % under ±50 % data-movement change. Noise: at σ=8/255 accuracy falls 9.43-16.82 points while the ratio moves only 1.406 → 1.536 — "noise hurts accuracy much earlier than it hurts energy." Cross-checked in Timeloop/Sparseloop (F_W=1, F_P=0); op constants from 22 nm synthesis. **Zero CL content.**

### [11] PAPER 22 — TTT with self-supervision, local read
ResNet-26 (CIFAR-10) / ResNet-18 (ImageNet), **GroupNorm not BatchNorm** ("BN breaks at batch size 1"), shared trunk split after group 2 / group 3, rotation-prediction auxiliary branch. Test-time SGD LR 0.001, zero weight decay/momentum, 10 gradient steps (standard) or 1 step (TTT-Online). CIFAR-10-C level 5 test error % (baseline / joint / TTT / TTT-Online): clean 8.9 / 8.1 / 7.9 / 8.2 · Gaussian noise 50.5 / 49.4 / 45.6 / **25.8** · impulse 56.1 / 53.4 / 50.0 / **30.6** · pixelate 55.8 / 51.6 / 47.2 / **18.1** · contrast 25.0 / 25.3 / 23.9 / **15.6**. VID-Robust 41.4 → 45.4; ImageNet VID-Robust 62.7 → 64.3; CIFAR-10.1 17.4 → 15.9 (a gain the authors call "small relative to the performance drop").
**Where it does not win:** TTT-Online worse than joint on clean CIFAR-10 (8.2 vs 8.1) and on ImageNet-C original (68.8 vs 69.1); **standard TTT worse than joint on 14 of 16 ImageNet-C corruption types** (Gaussian 3.1 vs 2.1 accuracy; defocus 10.1 vs 8.7; zoom 18.5 vs 16.0). VID-Robust airplane class shows **zero gain** — black margins make rotation trivially predictable → "requires the self-supervised task to be both well defined and non-trivial."
**Weak baselines flagged:** joint training is their own improved version of Hendrycks et al. (22.8 % avg error vs his 28.6 %); **UDA-SS is explicitly "an oracle instead of a baseline"** (sees the whole unlabeled test set); ALP doubles clean-set error (16.5 vs 8.9) and is catastrophic on fog (64.8) and contrast (73.6).
**On non-stationary evidence:** essentially none. The only drifting test is a monotone σ ramp level 1→5 over 8,000 samples, three noise types, where TTT-Online merely has "gentler slopes" than joint. Cost: 2 × batch size × iterations slower than plain inference (~20× standard).
**Theory:** Theorem 1 — positive gradient inner product ⇒ main-task loss improves; empirical correlation r = 0.93 / 0.89 across 75 test sets; covers only convex/smooth models.

### [6] PAPER 3 — EWC, local read (6 pages, main text only)
**Three experiments:** (a) toy — **linear** network, n = 1,000 synapses, random uncorrelated binary patterns → binary outcomes, analytic solution vs simulation; (b) supervised — fully-connected MLP on permuted MNIST, Fig. 3C x-axis to layer depth 6, Fig. 3B sweeping 2→10 tasks, baselines plain SGD / L2 / SGD+dropout; (c) RL — DQN-style agent on **10 Atari games** randomly drawn from games DQN plays at/above human level, order randomized with revisits, EWC penalty applied only after a game has seen 20M frames, FMN task-inference vs oracle labels.
**Numbers are in figures only** (main text contains zero percentage figures — verified by full-text scan). Approximate axis readings, treated as **directional**: Fig. 3B EWC ≈0.99 (2 tasks) → ≈0.97 (10), SGD+dropout ≈0.975 → **≈0.785**; Fig. 3A after B and C, task-A accuracy ≈0.86 (SGD) / ≈0.92 (L2) / ≈0.97-0.98 (EWC), while SGD learns task C *best* (≈0.96 vs EWC ≈0.95) — the plasticity tax. Fig. 4B total human-normalized score (clipped, max 10): no-penalty SGD stays ≈0-1 (text confirms "remains below one"); EWC+FMN ≈6.2-6.5; EWC+oracle ≈7. Fig. 2 (toy, fraction retained) at capacity t≈n=1,000: EWC ≈0.7 vs GD ≈0.4; **past capacity EWC falls below GD** (crossing visible just right of the capacity line).
**Fisher-overlap measurement (Fig. 3C):** 8×8-permutation tasks overlap ≈0.92-0.97 across all 6 layers; 26×26-permutation tasks ≈0.82 at layer 1 rising to ≈0.96 at the shared output layer. This is **allocation, not causal importance**.
**The self-undermining control (Fig. 4C), verbatim:** "This suggests that we are overconfident about certain parameters being unimportant: it is therefore likely that the chief limitation of the current implementation is that it underestimates parameter uncertainty."
**Other admitted failures:** "After network capacity is exceeded (Right), EWC performs worse than gradient descent." / "blackout catastrophe" when capacity is saturated. / L2 "cannot learn task B properly"; dropout "does not scale to more tasks." / EWC does not reach the score of 10 separate DQNs (Fig. S3, referenced but NOT READ). / Point-estimate Laplace posterior variance is "a significant weakness" → suggests Bayesian NNs. / EWC can only *increase* constraint over time, so it "can model only memory retention rather than forgetting."
**Not in your copy:** Tables S1/S2 (parameter counts, λ, LR, batch sizes) — the 6-page file is main text only.

### [16] PAPER 45 — CLS model, local read (complete)
Split-MNIST, 5×2 binary contexts, ~12,000 images/task. VAE (10 epochs, batch 32, lr 1e-3, BCE+KL, latent 48) + Modern Hopfield Network holding **~600 stored latent vectors (5 % of images)**; replay via noise cue → recurrent energy-minimizing settling → decode. **Table 1:** VAE offline UB 95.55 % · untrained 8.92 % · sequential control (CIL, no replay) **67.75 %** · VAE+MHN **89.71 %** (+21.96 over naive sequential).
**Negatives and caveats:** best config was 91.01 % but they reported 89.71 % "for reasons of parsimony." **Replay rate is quadratic** — "both low and high ratios impaired performance," optimum ~1:1; more replay is not monotonically better. The "hippocampal" MHN is **worse at pattern completion** — lower SSIM for all 10 classes (Bonferroni p<0.001). Classification is done by a **separate external MLP** (97.58 % on MNIST) scoring *reconstructed* images, so 89.71 % is partly a generative-fidelity measure, not a CL accuracy. **Claim/mechanism contradiction:** the intro says CL works "without the need for explicit task labels or stored replay buffers," but the MHN *is* a ~600-entry episodic buffer (latent, not raw) — do not cite as replay-free CL. Single dataset, single task order, 5 seeds. No EWC/SI/GEM/BI-R comparison anywhere, so the abstract's "close to state-of-the-art (~90 %)" is unsupported. The Spens & Burgess original never tested CL at all. Transitive inference = future work.

### [46] PAPER 42 — Online-LoRA, local read
ViT-B/16 (86.6M) and ViT-S/16 (48.6M) **pretrained on ImageNet**; LoRA rank 4 on Q,V of every attention layer; a new LoRA pair opened at each **loss-surface plateau**, previous pairs merged into the frozen weights; online Laplace/Fisher regularization on LoRA factors only (147,456 importance entries ≈ 0.17 % of ViT-B/16); **hard buffer of 4 highest-loss samples**; λ=2000; per-dataset loss-window thresholds grid-searched (mean 2.6 CIFAR-100 / 5.2 ImageNet-R / 5.6 ImageNet-S / 6.0 CORe50 / 24.0 CUB-200). Benchmarks: Split-CIFAR-100 (10×10), Split-ImageNet-R (10×20), Split-ImageNet-S, Split-CUB-200 (5×40), Si-Blurry, CORe50 DIL. All baselines re-implemented by the authors on the same ViT backbone. One A100.
**A_Final (buffer 500, ViT-B/16):** Split-CIFAR-10 — ER 44.85, MIR 48.36, PCR 48.48, DER++ 36.64, EWC++ 10.61, AGEM 12.67, GDumb 41.00±19.97, **Ours 49.40** (forgetting 41.74), UB 89.50. ImageNet-R — ER 40.99, PCR 46.11, **Ours 48.18** (forg 23.85), UB 76.78. ImageNet-S — ER 30.21, PCR 38.75, DER++ 6.47, AGEM 0.16, **Ours 47.06**, UB 63.82. CUB-200 — PCR 41.11, **Ours 41.46** (forg 13.64), UB 82.81. CORe50 DIL — L2P 87.97 (forg 0.00), PCR 87.16, DER++ 81.88, **Ours 93.71** (forg **0.00**), UB 95.60. Si-Blurry — L2P 39.86, MVP 44.49, **Ours 61.70**. Ablation (ImageNet-R): neither 28.68/53.45; incremental-LoRA only 34.74; hard-loss only 36.08; both 48.23/23.85 — **the hard buffer alone is worth 13.5 points**. Pretrained-not-the-explanation control: Random Head 0.08, Frozen FT 27.98, Continual FT 28.49.
**Audit flags:** Table 1 vs Table 11 GDumb inconsistency (8.87±1.36 vs 1.65±0.22). Appendix G at buffer 5000: "sophisticated memory retrieval strategies … do not significantly outperform GDumb's simple approach." CORe50 forgetting = 0.00 for three methods simultaneously → degenerate metric. Si-Blurry results **exclude the hard-buffer loss** "to be fair to L2P/MVP" — i.e. a different method than Table 1's. DER++ re-implemented on ViT collapses to 6.47 on ImageNet-S, inflating apparent margins. No limitations section in 25 pages.

### [75]/[76] Titans and Nested Learning — the exact hedging language
**Titans abstract, read in full:** neural long-term memory + attention, three variants; experiments are language modelling, common-sense reasoning, genomics, time series, scaling to >2M context with NIAH accuracy. **No continual-learning benchmark, no drifting stream, and no hedge language at all** in the abstract — "We show that…", "we argue that…". The title is "Learning to **Memorize** at Test Time." A NeurIPS 2025 proceedings version exists and was **not** read.
**Nested Learning abstract, read in full:** NL reframes models as nested optimization problems; introduces Expressive Optimizers, a Self-Modifying Learning Module, a Continuum Memory System, and the **Hope** module. Verbatim: NL "suggests a philosophy to design more expressive algorithms with more levels, resulting in higher-order in-context learning and **potentially unlocking effective continual learning capabilities**"; Hope shows "**promising results** in language modeling, knowledge incorporation, and few-shot generalization tasks, continual learning, and long-context reasoning tasks." **The abstract names no benchmark, no task count, no stream length and no metric.** Behrouz, Razaviyayn, Zhong, Mirrokni (Google Research), submitted 31 Dec 2025, arXiv:2512.24695.

### [80]/[81] The similarity question, primary text
**[80] Hiratani, arXiv:2405.20236** — "Disentangling and Mitigating the Impact of Task Similarity for Continual Learning", sole author, submitted 30 May 2024, also NeurIPS 2024. Abstract, verbatim: they "show analytically that high input feature similarity coupled with low readout similarity is **catastrophic for retention**"; "Task-dependent activity gating improves knowledge retention **at the expense of transfer**"; the opposite scenario (low input similarity, high readout similarity) "is relatively benign." Method: "a linear teacher-student model with latent structure," results "shown analytically," empirical component a *permuted* MNIST task with latent variables. **No CNN accuracy table, no Fashion-MNIST.** NeurIPS PDF fetch returned no extractable text.
**[81] Goldfarb et al., ICLR 2024** — two-task continual linear regression where task 2 is a random orthogonal transformation of task 1, exact closed-form expected forgetting, validated on synthetic linear regression and neural nets on permutation-task benchmarks. Headline: **direction depends on overparameterization.** Highly overparameterized → *intermediate* similarity causes the most forgetting (non-monotone peak). Near the interpolation threshold → forgetting **decreases monotonically as similarity increases.** Under both regimes, maximum similarity is not the worst case.

### [42] Bouthillier et al. — the statistics you can actually use
Variance sources separated: data sampling (bootstrap resampling), data augmentation, weight initialisation, dropout, SGD visit order, hyperparameter optimisation. **Sampling/data variability dominates.** Weight initialisation contributes "less than 50 %" of bootstrap variance; hyperparameter tuning induces "as much variance as the commonly studied weights initialization." Headline efficiency figure is about estimator design, not seed count: re-running full HPO per split costs "IdealEst(k=100) takes 1070 hours … compared to only 21 hours for each FixedHOptEst(k=100)" (~51×). **Recommendation is not a seed count:** randomise as many variance sources as possible, use out-of-bag/bootstrap estimates with a non-parametric percentile bootstrap, and claim practical superiority only when the bootstrap probability of outperformance exceeds **0.75**. Workloads: CIFAR-10+VGG11; PascalVOC segmentation (FCN+ResNet-18); GLUE SST-2 and RTE (BERT); peptide-MHC-I (shallow MLP). Also carries the line your journal quotes: "this is prohibitively expensive, and corners are cut to reach conclusions."

### [9]-adjacent housekeeping, verified locally
`03-energy-efficiency/PAPER 46.pdf` is **byte-identical** to `PAPER 36.pdf` (both 5,318,875 bytes) — one is pure redundancy, safe to delete. `PAPER 13.pdf` vs `PAPER 25.pdf` are **different versions** of Forward-Forward (16 vs 17 pages, different hashes); keep PAPER 13 (it carries the retraction and the deleted §4.1), and PAPER 25 can go once you have looked at the diff yourself. `PAPER 23.pdf` is **62 pages**, not the ~45 that `PAPERS_MANIFEST.md` implies.
