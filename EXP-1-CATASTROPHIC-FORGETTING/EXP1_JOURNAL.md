# EXP-1 · SEEING CATASTROPHIC FORGETTING
### A pre-registration journal for the first experiment — design, tasks, datasets, instruments, controls, and predictions

**Author:** MK · **Written:** 2026-09-25 · **Amended:** 2026-10-01 (citation audit — see §19) · **Status:** DESIGN FROZEN PENDING REVIEW — no results yet
**Companion PDF:** `EXP1_JOURNAL.pdf`
**Bracketed IDs** `[n]` resolve to Section 17. The ones that could not be confirmed are quarantined in Section 17.2 and must not be cited. **Correction to this line as originally written:** it claimed "every ID was opened and checked." That was not true — the 2026-10-01 audit found five entries ([16], [17], [26], [27], [28]) whose IDs resolved to different papers or whose quotes were overstated, and they were carried into load-bearing places (§2.2 and §14). Treat every `[A]` tag as "an agent reported it," not "it is right," until re-checked.

---

## 0. What this experiment is for, in one paragraph

We are going to teach a small neural network one thing, then teach it a second thing, and then check whether it still knows the first. This is not a test of any method or any idea of ours. It is a **measurement** — we build the instrument before we use it. The instrument has two dials: the obvious one is *accuracy on task A*, and the one nobody else has read properly is *the weights themselves*. Which numbers moved, by how much, in which direction, and did they move in the directions task A cared about?

**Why this is the right first experiment for the programme.** `END_GOAL.md` §3 says any acquired skill must be stored *somewhere*, and lists three places. Row 1 is "into existing weights," and that row is exactly where forgetting happens. Before we can argue about how to make weight updates "so local and so safe that retraining stops being the thing we do," we have to be able to see an unsafe update, name the layer it happened in, and measure how much damage it did. That is a measurement problem, and it is the cheapest thing on the list.

There is a second reason, and it is the more interesting one. `END_GOAL.md` §9 and the notes on machine unlearning both point at the same hole: we cannot currently distinguish **"the parameters genuinely changed"** from **"the model merely stopped reporting."** This experiment builds a crude version of that discriminator — a linear probe on frozen post-B features — and it costs about nine lines of scikit-learn. If the probe says the information is still there while accuracy says it is gone, we have found something worth a second experiment.

---

## 1. The phenomenon, stated carefully

Catastrophic forgetting: a network trained on task B loses the ability to do task A, even though nothing about A's inputs changed. The classic demonstration is dramatic and it is not subtle — sequential fine-tuning with no special treatment is the *worst* baseline in the entire continual-learning literature, so bad it is usually reported as a single negative number rather than a curve.

Two facts that will shape how we read our own results:

**Fact 1 — the size of the drop depends on how we ask the question, not just on what the network did.** Verified baselines from van de Ven & Tolias [1], for the *same* sequentially-trained networks:

| Benchmark (5 tasks, 2 classes each) | Task-IL | Domain-IL | Class-IL |
|---|---|---|---|
| Split-MNIST, "None" baseline | **87.19 ± 0.94** | **59.21 ± 2.04** | **19.90 ± 0.02** |
| Permuted-MNIST, "None" baseline | 81.79 ± 0.48 | 78.51 ± 0.24 | 17.26 ± 0.19 |
| Permuted-MNIST, with EWC | 94.74 | 94.31 | 25.04 |
| Permuted-MNIST, with SI | 94.75 | 95.33 | 29.31 |

Read the first row again: 87% versus 20% is the *same weights* doing the *same underlying work*, and the only difference is whether the experimenter tells the model which task it is being shown. Forgetting numbers are therefore meaningless without a declared evaluation protocol. That is Section 3.

**Fact 2 — forgetting is not always permanent, and a single number can hide that.** De Lange, van de Ven & Tuytelaars [2] found that after fine-tuning begins, "this forgetting is temporary and followed by a phase of performance recovery," and named the worst-case transient the **stability gap**. Consequence for us: if we only evaluate task A at the *end* of stage B, we can measure the wrong thing in either direction — we might catch the dip and call it catastrophe, or catch the recovery and call it safety. So we evaluate task A *continuously, during* stage B. Section 8.

---

## 2. Which two tasks — the actual decision

### 2.1 What we are running

**Phase 1A (primary): MNIST → Fashion-MNIST.**

Task A = the ten digits, 28×28 grayscale, 60,000 train / 10,000 test.
Task B = the ten Fashion-MNIST classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot), 28×28 grayscale, 60,000 train / 10,000 test.

Both tasks are 10-way classification over the identical input geometry. The model has one backbone and one 10-output head. Training A teaches the head "row 3 means digit 3"; training B teaches the same rows "row 3 means Dress." This is the *cleanest available* overwrite story: task B literally re-uses the coordinates task A was reading out of, while the input pixels look deceptively similar.

**Phase 1B (confirmation, deferred): Split-CIFAR-10.** CIFAR-10's 10 classes split into two halves of 5, trained A-then-B, at 32×32 with a wider CNN and a ResNet-18 replication. This exists so we have at least one number that sits next to published tables.

### 2.2 Why this pair, with receipts

- **The drop is documented as severe — but only in the scenario that is *not* ours.** *(Entry corrected 2026-10-01; the previous version of this bullet cited arXiv:2410.16154 for a quote that paper does not contain — see §17.1 [16].)* Naive sequential fine-tuning collapses to chance when the model must infer the task: 19.90 ± 0.02 % on Split-MNIST CIL and 17.26 ± 0.19 % on Permuted-MNIST CIL against joint-training ceilings of ~98 % [1]. At CIFAR scale, ResNet-18 from scratch over 10 runs: 19.62 ± 0.05 % on Split-CIFAR-10 CIL and 7.92 ± 0.26 % on Split-TinyImageNet CIL [32]. **In domain-incremental settings the same baselines are markedly milder** — 59.21 % on Split-MNIST DIL [1], and 40.70 ± 2.33 % on 20-task Permuted-MNIST and 67.66 ± 8.53 % on 20-task Rotated-MNIST [32]. Our primary protocol is Domain-IL-like (§3), so **we are in the regime where "screaming" is not established.** P1's falsification branch is the more likely one; run C4 before concluding anything about the pair.
- **It is cheap enough to repeat.** MNIST + Fashion-MNIST are ~11 MB and ~26 MB, both shipped in `torchvision`, and a 242k-parameter model does 10 epochs in about a minute on our GPU. That means **many seeds, many controls, and a mistake we can afford to make**.
- **The input distributions genuinely differ, but not by accident of geometry.** Fashion-MNIST was built as "a direct drop-in replacement for the original MNIST dataset … same image size, data format and the structure of training and testing splits" [8]. That is exactly the trap in Section 12.3 — the geometry match is a *feature of the dataset*, so any A→B difference we find is about content, not about resizing.
- **Theory does *not* say this pair is the destructive one.** *(Corrected 2026-10-01; the previous version overstated the quote and mis-described the paper — see §17.1 [17].)* Hiratani shows **analytically**, for a **linear teacher-student model with latent structure**, that "high input feature similarity coupled with low readout similarity is catastrophic for **retention**" [17]. Retention only — transfer is a separate and opposing axis in the same paper ("task-dependent activity gating improves knowledge retention **at the expense of transfer**"), and the "opposite scenario" is called relatively benign. There is no CNN, no Fashion-MNIST and no accuracy table in it, so it does not license a prediction about a 242k-parameter net. Two counterweights point the other way: Goldfarb et al. derive a closed-form similarity–forgetting relation and find either monotone *decreasing* forgetting with similarity or a peak at **intermediate** similarity — under neither branch is the most-similar pair the worst [36]; and [2] measured rotation angle as the similarity dial on Rotated-MNIST and found the damage grows with **dissimilarity** (min-ACC 94.3 % at ϕ = 10° → 69.2 % at ϕ = 80°, while ordinary accuracy fell only 6.6 %) [2, Table 2, p.6]. **So the pair is not chosen because theory predicts it will hurt.** It is chosen because it is cheap, because the head-overwrite story is legible at the weight level, and because the pair appears never to have been measured (§14, §17.2).
- **The weights stay small enough to interrogate.** 242k numbers across 14 trainable tensors (plus 8 BatchNorm buffers). We can dump them all, hash them, take their SVDs, and plot every per-layer statistic without touching a disk budget. ResNet-18 is 11.7 million parameters [18] and makes "which weights changed" a much noisier question.

### 2.3 What we rejected, and why

| Option | Why not first |
|---|---|
| **Split-MNIST** (5 tasks × 2 classes) | The canonical choice and the cheapest, and we should run it later — but with 2 classes per task, chance is 50%, the head does most of the work, and the interesting thing (feature destruction) is partly hidden behind the readout. Good for literature comparability, weak for weight forensics. |
| **Split-CIFAR-10 with ResNet-18** | Best external validity; a continually-trained ResNet-18 README reports task-1 accuracy 84.4% falling to 52.85% (class-IL) / 62.92% (task-IL) after task 2 [19]. But on 4 GB it is an hour-plus per task, so controls and seeds get expensive fast. Deferred to Phase 1B. |
| **Permuted-MNIST** | Forgetting is *milder* per task because all tasks share digit semantics — positive transfer fights the effect we want to see [1]. Wrong tool. |
| **CORe50 / CLEAR / ImageNet-R / Split-CUB / Split-AWA** | All either too large or too slow for a laptop. CLEAR is millions of images (YFCC100M-derived) [20]; AWA/CUB run at 224×224 with standard ResNet-18 [21]. Out of reach. |
| **An LLM or CLIP fine-tune** | Tempting because the END_GOAL literature is language-heavy, but we would have no visibility into θ₀, no cheap per-layer statistics, and no way to afford the seeds. Not an observation experiment — a money experiment. |

---

## 3. Declare the scenario before we run anything

Per van de Ven & Tolias [1]: Task-IL = "solve tasks so far, task-ID provided"; Domain-IL = "solve tasks so far, task-ID not provided"; Class-IL = "solve tasks so far *and* infer task-ID."

**Our primary protocol is Domain-IL-like.** Each task is evaluated inside its own 10-class label space and the model is *not* given the task identity at test time — the model must decide from pixels alone whether it is looking at a digit or a garment. Chance level for A evaluated this way is 10%.

We additionally report a Task-IL view (evaluate A with A's own head, which is the "backbone-B + head-A" number in Section 7) and we *do not* claim a Class-IL number for Phase 1A, because A and B share an output space with unrelated meanings — a 20-way joint readout would be measuring a different question. When we get to Phase 1B we *will* report Class-IL, because there the classes are disjoint and 5-way halves give a chance level of 20%.

The trap worth naming: [1] warns that "a single-headed layout … might by itself not require task identity to be known, it is still possible for the model to use task identity in other ways (e.g. in its hidden layers)." We keep no task-ID input, no task-specific BatchNorm, no task embeddings, and we will say so in the writeup. Any of those silently turn Domain-IL into Task-IL and inflate our accuracy by tens of points.

---

## 4. Models

We run two architectures on purpose. If the phenomenon appears in both, it is about learning; if it appears in only one, it is about architecture, and that is a more interesting finding than we planned for.

**M1 — small CNN (primary).** Four 3×3 conv layers with BatchNorm + ReLU, 32→64→128→128 channels, max-pool after block 1 and block 2, global average pool to a 128-d feature, then a single `Linear(128, 10)` head. **≈242,000 parameters, 14 trainable tensors** (4 conv weights + 4 pairs of BatchNorm γ/β + head weight/bias) **plus 8 non-trainable BatchNorm buffers.** Expected solo accuracy: ~98.5% on A, ~90% on B. The GAP→linear head is what makes the backbone/head split in Section 7 exact rather than approximate.

**M2 — MLP (replication).** `784 → 100 → 100 → 10`, ReLU. ≈89,600 parameters. This mirrors the classic two-hidden-layer nets used in the baseline tables of [1], so our numbers have company.

**Why from scratch, not pretrained.** Two reasons, both practical. (a) To measure "which weights changed" we need a θ₀ we own and can hash. (b) A pretrained model's forgetting is contaminated by the thousands of tasks it already knows, none of which we can enumerate or test. Pretrained fine-tuning is EXP-2's question, not EXP-1's.

**VRAM.** At batch 128 on 28×28 grayscale with 242k parameters, activation memory is tens of megabytes. The 4 GB ceiling is not a constraint in Phase 1A. In Phase 1B it is: ResNet-18 at 32×32 wants batch 32 with AMP, and that is the reason Phase 1B is deferred.

---

## 5. Preprocessing, frozen before we look at any number

1. Resize: none. Both datasets are natively 28×28.
2. Flatten/scale to float in [0, 1]; `ToTensor` only. **No augmentation for Phase 1A.** Augmentation is a regulariser and a confound; if we later want to know how much of forgetting augmentation prevents, that is a separate question.
3. **Normalisation: per-dataset standardisation, computed on that dataset's own TRAIN split, hardcoded as constants in the config.** Mean/std recorded in `manifest.json`. This is the honest realistic case (each stream arrives with its own statistics) but see Section 12.3 — we run the counterfactual control.
4. Phase 1B augmentation is declared up front to match the literature: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`, and CIFAR mean/std computed once on our train split and frozen (two competing constant sets circulate; computing ours removes the ambiguity).
5. Test sets are never touched during training, never used for any hyperparameter choice, and eval batch order is fixed by seed.

---

## 6. Protocol — the exact sequence, and what we save

This is the part the original idea asked for: train, **save everything**, reload, train again.

```
STEP 0   Seed everything. Instantiate model -> record theta_0 (state_dict + SHA256 of every tensor)
STEP 1   TRAIN on A (MNIST) for E_A epochs                -> theta_A, full checkpoint
STEP 2   VERIFY LOAD: reload theta_A, forward a fixed 512-image probe batch,
         assert logits are bit-identical to pre-save. FAIL LOUD if not.
STEP 3   EVALUATE A  -> a[A,A]   (this is the reference number every later delta uses)
STEP 4   LOAD theta_A (not theta_0), keep the SAME optimiser class, and
         TRAIN on B (Fashion) for E_B epochs, with per-step logging:
            - every K steps: evaluate A on a fixed 2,000-image subset -> FORGETTING CURVE
            - every epoch:   full eval on A and B, save checkpoint
            - every epoch:   per-tensor ||theta - theta_A||, Fisher-weighted drift
         -> theta_B, full checkpoint
STEP 5   EVALUATE B  -> a[B,B]
STEP 6   EVALUATE A again, four ways (Section 7) -> a[A,B]_full, _headA, _probe, _jointref
STEP 7   WEIGHT FORENSICS (Section 9) on {theta_0, theta_A, theta_B} + per-epoch checkpoints
STEP 8   Re-run Steps 0-7 for each control condition (Section 10) and each seed
```

**What "save everything" means concretely**, per run, in `runs/<run_id>/`:

| Artifact | Contents |
|---|---|
| `theta_0.pt`, `theta_A.pt`, `theta_B.pt` | full `state_dict` |
| `epochs/theta_B_e{1..E}.pt` | every stage-B epoch checkpoint, for the drift and probe curves |
| `optim_state_A.pt`, `optim_state_B.pt` | Adam moments + buffers — we need them to test the reset-vs-continue confound |
| `manifest.json` | every hyperparameter, seed, torch/CUDA version, dataset file checksums, normalisation constants, git commit hash, wall-clock per stage, peak VRAM |
| `tensors.csv` | per-tensor: name, shape, numel, L2, mean, std, min, max, SHA256 — for θ₀, θ_A, θ_B |
| `curve_A_during_B.csv` | step, epoch, A-accuracy, A-loss, A-mean-margin, ‖θ−θ_A‖, Fisher-weighted drift so far |
| `forensics.json` | all Section 9 metrics, per layer |
| `probe_A_on_thetaB.json` | linear-probe accuracy ± over probe seeds |

The load-verification in STEP 2 is not ceremony. Reloading a checkpoint and silently training a *different* network — because of a `map_location` mismatch, a dropped prefix in the state-dict keys, or a BatchNorm buffer that did not restore — is the single most common way a sequential-training result becomes fiction. We assert on logits, not on shapes.

---

## 7. The accuracy measurements: one drop is three drops

Sequential fine-tuning with a shared head mixes up two different failures, and separating them is the whole point of looking at weights. Let f be the backbone, h the head.

| # | Name | Model used | Reads out | What it means if it collapses |
|---|---|---|---|---|
| **A1** | `a[A,A]` | θ_A (f_A, h_A) | A at end of stage A | *reference.* Everything else is a delta from here. If this is not high, nothing downstream is interpretable. |
| **A2** | `a[A,B]_full` | θ_B (f_B, h_B) | **total forgetting** | the headline number. Task A is gone. |
| **A3** | `a[A,B]_headA` | θ_B's **f** + θ_A's **h** | **representational damage** | the features no longer encode what A's head needs. This isolates the backbone. |
| **A4** | `a[A,B]_probe` | θ_B's f **frozen** + freshly trained linear head on A's train labels | **is the information still there at all?** | A4 high while A2 collapses ⇒ the network did not lose the capability, it lost the ability to *report* it. |

**A4 is the experiment's most valuable row and it costs almost nothing.** Procedure: freeze f_B, dump features for A's 60,000 train images and 10,000 test images once, fit multinomial logistic regression (L2, C grid {0.01, 0.1, 1, 10} selected on an A-train holdout, *never* on A's test set), report test accuracy ± over 3 fit seeds. Same probe fitted on f_A's features is the upper bound for what the probe can achieve, and we report it as `a[A,A]_probe` so A4 is never read in isolation.

The interpretive ladder we will actually use:

- `A2 ≈ chance, A3 ≈ chance, A4 high` → **readout destruction with intact representation.** The pixels are still perfectly separable in feature space; the trained head stopped reading them. This is a "the model merely stopped reporting" result.
- `A2 ≈ chance, A4 ≈ chance` → **genuine representational destruction.** The information is gone from the weights.
- `A2 high` → the pair is too similar, or stage B barely moved the weights. Fix the task pair or the learning rate before concluding anything.

We also record **loss on A** at every evaluation, not just accuracy. Accuracy saturates (98% → 99% is invisible) and it hides margin collapse; A-loss and A-mean-logit-margin will move first and move smoothly.

---

## 8. The forgetting curve, not the forgetting number

During STEP 4 we evaluate task A every K = 50 optimiser steps on a fixed 2,000-image subset (same images, same order, every time), plus full A and B evaluation every epoch. We plot against **optimiser steps in stage B**, never against epochs, because epochs mean different numbers of updates for different batch sizes.

Three curves on one figure, one line each: A-accuracy, A-loss, and total ‖θ − θ_A‖. The reason is [2]: if the accuracy dip is followed by recovery, we are watching the stability gap and the endpoint number is a lie. The reason we overlay weight norm is the question this experiment exists to answer — **does the weight movement keep going after the accuracy stops falling?** If yes, weights and capability have decoupled, and that is a finding.

---

## 9. Weight forensics: the battery

Let θ₀, θ_A, θ_B be the three checkpoints. Following task arithmetic [3], define the **task vectors**

```
tau_A = theta_A - theta_0        (what learning A did to the weights)
tau_B = theta_B - theta_A        (what learning B did to the weights)
```

[3] define it verbatim as vectors "by subtracting the weights of a pre-trained model from the weights of the same model after fine-tuning on a task." Ours is the same object with a from-scratch net. Every metric below is computed **per tensor** (all 14 separately) and globally.

**Tier 1 — checkpoints only, seconds, do all of it.**

| ID | Metric | Formula | What it tells us |
|---|---|---|---|
| W1 | per-tensor drift | `d_l = ‖tau_l‖₂`, relative `r_l = ‖tau_l‖₂ / ‖theta_{0,l}‖₂` | the shape of the change; `r_l` is scale-free so layers are comparable |
| W2 | update-direction cosine | `c_l = <tau_{A,l}, tau_{B,l}> / (‖tau_{A,l}‖‖tau_{B,l}‖)` | do A and B push the same coordinates the same way? |
| W3 | change density | `p_l(tau) = #{i : |Δ_i| > tau·std_l} / numel` | how much of the update is real vs noise floor |
| W4 | support overlap | Jaccard of `{i: |tau_{A,i}|>t}` and `{i: |tau_{B,i}|>t}` | do the two tasks rewrite the *same* numbers? |
| W5 | sign-flip rate | fraction of jointly-changed coordinates where `sign(tau_A)·sign(tau_B) < 0` | direct interference measure. TIES-merging's whole method is "resolving sign conflicts … merging only the parameters that are in alignment with the final agreed-upon sign" [11] |
| W6 | head-row attribution | per output row: ‖Δh_row‖, and which digit class each row was for vs became | human-readable proof of the overwrite: "row 3 went from 'digit 3' to 'Dress'" |
| W7 | low-rank summary | singular-value spectrum of `tau_l` per weight matrix; effective rank `erank = exp(-Σ p_i ln p_i)` over normalised σ | how concentrated the update is. LoRA-vs-full-FT work reports "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations" [22] |
| W8 | loss barrier | `B = max_t L_A(θ_A + t(θ_B−θ_A)) − min(L_A(θ_A), L_A(θ_B))`, t ∈ {0,…,1} in 21 steps | is θ_B in A's basin? ~21 forward evals, no training. Barrier ≈ 0 with large forgetting means distance is not the mechanism. Concept from [23] |

**Tier 2 — cheap extra passes.**

| ID | Metric | How | Why it is the one that matters |
|---|---|---|---|
| W9 | **Diagonal Fisher of A** | `F_i = E_{x~A}[(∂ log p(y|x,θ_A)/∂θ_i)²]`, Monte-Carlo over ~100 batches of 128 | EWC "remembers old tasks by selectively slowing down learning on the weights important for those tasks" [4]. This is our map of *which coordinates A cares about* |
| W10 | **Fisher-weighted drift of B** | `Φ = Σ_i F_i^A · (tau_{B,i})²`, plus per-layer `Φ_l` and `Φ_top10 = Σ_{i in top-decile F^A} …` | the key quantity. Total drift is a weak predictor; drift *in A's important directions* is the mechanism EWC assumed. Section 13 makes this the money plot |
| W11 | gradient conflict during stage B | at 10 logged points in stage B, on one fixed A batch and one fixed B batch: `cos(g_A, g_B)`, the fraction of coordinates with opposite signs, and ‖g_A‖/‖g_B‖ | a live interference signal we can correlate with the per-step forgetting curve. PCGrad's analysis is precisely about conditions that "cause detrimental gradient interference" [10] |
| W12 | per-layer CKA between f_A and f_B | linear CKA on features of the same fixed 500 A-images through both models | representational similarity that is invariant to rotations/scaling of the feature space, unlike raw weight distance [15] |
| W13 | gradient-subspace escape (GPM-lite) | per layer, SVD of activations on ~500 A-images, keep top-d_u directions (90% energy) as Q_A; then `residual ratio = ‖(I − Q_AQ_Aᵀ)·g_B‖ / ‖g_B‖` and the principal angles between Q_A and Q_B | DIY version of GPM, which "learns new tasks by taking gradient steps in the orthogonal direction to the gradient subspaces deemed important for the past tasks … with Singular Value Decomposition" [6] and Adam-NSCL's null-space-of-input-covariance formulation [7]. Answers: how much of B's update escapes the space A lives in |

**Tier 3 — only if 1A comes back clean.** SI path-importance accumulated during stage A [5] (needs per-step bookkeeping inside the training loop); task-vector negation as a *causal* check — take θ_B and subtract λ·proj(τ_B onto A-important coordinates), or reconstruct A by θ_A − λτ_A, and see whether the accuracy loss reverses. Causal beats correlational, and it is the strongest claim EXP-1 could make.

**What we are explicitly not doing yet:** no EWC, no replay, no any CL method. Per the scope decision, EXP-1 is observational. Note also that `END_GOAL.md` C1 already records the field's own survey consensus that parameter-efficient methods "remain vulnerable to catastrophic forgetting," so the mitigation literature is not the open question — the *instrument* is.

---

## 10. Controls — five numbers that make the six main numbers mean something

| # | Control | Why without it we would be wrong | Source |
|---|---|---|---|
| **C1** | **Joint training** on A+B simultaneously | the upper target. Separates forgetting from the intrinsic difficulty of doing both. Also supplies the `a*` term needed for intransigence | "Joint training … provides a target reference performance" [2]; van de Ven's slides call it "(upper target)" [1] |
| **C2** | **A-only, stop** (identical step count to the sequential run) | A1 is only a fair reference if the A-training was not truncated differently between runs | the `a_{j,j}` term is load-bearing in every CL metric |
| **C3** | **B-only from scratch** | without it, "B reached 88%" has no meaning — is B intrinsically harder? is its optimum farther away? | intransigence `IM = a* − a_{k,k}` [9] formalises exactly this |
| **C4** | **Stage B with LR ×1/10, and stage B with 1/10 the steps** | *the most important control we have.* Measured forgetting is dominated by total updates and learning-rate schedule, not by architecture or cleverness. Skip it and we are measuring hyperparameters while believing we measured plasticity. Mirzadeh et al. show these regime choices "can outperform considerably more complex algorithms meant to deal with continual learning" [14] | [14] |
| **C5** | **Head ablation** = A3/A4 in Section 7 | the shared-head overwrite is a *protocol choice*, not a property of the world. Keeping vs discarding vs re-initialising A's head changes the answer by tens of points | multi-head "imposes a separate fully connected layer with softmax for each task" [2]; task-specific components are what define Task-IL [1] |
| **C6** | **Reversed order B → A** | order asymmetry. If A→B and B→A give different forgetting magnitudes, we must report both or we are cherry-picking a direction | [2] studies "the order in which the tasks are presented" and finds order effects small over 10 tasks — which is *not* a claim about 2 tasks, so we measure it ourselves |
| **C7** | **Adam state: continue vs reset** at the stage boundary | no source in the literature for this. It is a real mechanism (continued second moments make the first post-reload steps behave like a warm restart) and it is free to measure. Flagged in Section 17 as an open question we resolve by experiment rather than by citation | — |
| **C8** | **Normalisation counterfactual**: B standardised with **A's** mean/std | if some of our "forgetting" is only a pixel-intensity shift, this control shrinks it and tells us how much | Fashion-MNIST being a "direct drop-in replacement" [8] is what makes this test meaningful |

---

## 11. Metrics table, including the ones that break at two tasks

Notation: `a[k,j]` = accuracy on task *j*'s test set after having learned task *k*. The output space used for `a[k,j]` must be declared — [24] is explicit that "the output space to compute a_{k,j} consists of the classes in either Y_j or ∪ Y_i, corresponding to the use of multi-head evaluation (e.g., TIL) or single-head evaluation (e.g., CIL)." Ours is per-task output space (Section 3).

| Metric | Formula | Verdict for a **2-task** protocol |
|---|---|---|
| Average accuracy | `AA_k = (1/k) Σ_{j≤k} a[k,j]` | fine, but at k=2 it is just the mean of two numbers and perfectly collinear with ΔA. Report it, do not lean on it |
| Backward transfer | `BWT = (1/(T−1)) Σ_{j<T} (a[T,j] − a[j,j])` | **degenerates**: at T=2, `BWT = a[2,1] − a[1,1]` = exactly −ΔA. It is not wrong, it is *the same number we already have* |
| Forgetting measure | `f_j = max_{k<K}(a[k,j]) − a[K,j]`, averaged | **degenerates the same way**: the `max` has a single argument, so `FM = −BWT = ΔA` at T=2 [9]. The often-repeated claim that it is *undefined* for two tasks is too strong — it is defined and uninformative. It is only literally undefined at K=1 |
| Intransigence | `IM = a*_j − a[j,j]`, `a*` from joint reference | usable, needs C1. Worth computing once |
| Forward transfer | `FWT = (1/(T−1)) Σ_{j>1}(a[j,j] − ã_j)` | usable but **sign-trap**: its value flips depending on whether ã₂ is random init (chance) or the A-trained model. State which. We use A-trained, so positive FWT means "knowing digits helped dress recognition" |
| Raw ΔA | `a[1,1] − a[2,1]` | our headline. Chance-anchored: also report `(a[1,1] − a[2,1]) / (a[1,1] − chance)` |

**So what we actually report for Phase 1A:** the full triple `(a[1,1], a[2,1], a[2,2])`, plus ΔA, plus A3/A4 decomposition, plus the whole forgetting *curve*, plus intransigence against C1. Not BWT and FM as separate rows — reporting both at T=2 would be padding the table with one number written three ways, and a reviewer who notices looks at us differently than one who does not.

---

## 12. Confound register

Things that will try to fool us, and what we do about each.

**12.1 Update count masquerades as forgetting.** Covered by C4. Record the number of optimiser steps in stage B in every table row.

**12.2 Accuracy saturates; loss does not.** Report A-loss and mean logit margin at every checkpoint.

**12.3 The input-distribution confound.** Because Fashion-MNIST was *designed* to match MNIST's format [8], a naive reading is "these are the same inputs, so any damage is real." The reverse is the risk: the two datasets differ in contrast, stroke width and pixel statistics, and part of what we call forgetting could be a low-level rescaling the backbone was never given a chance to absorb. C8 handles it.

**12.4 Fisher estimation is itself a choice.** The ICLR 2025 blogpost line of work reports "significant differences between the different options also in terms of their best performance" for how practitioners compute the EWC Fisher [25]. We declare ours before running: diagonal, empirical (squared gradient of log-likelihood at θ_A), 100 batches of 128, one epoch of the A train loader, plus the alternative (expected-squared-gradient with the network's own predicted distribution) logged as a second column so W10 never rests on one estimator.

**12.5 Task-ID leakage.** No task conditioning anywhere (Section 3). Assert in code that the model's forward signature cannot accept a task index.

**12.6 BatchNorm running statistics.** Stage B's BN buffers overwrite stage A's running mean/var, and A's test-set accuracy depends on them. This is a real, physical mechanism of forgetting, not a bug — but we must be able to say how much of A2 is BN. So we log BN buffer drift separately, and run one extra variant evaluating A2 with θ_A's BN buffers restored (a 30-second ablation, and one I would otherwise never think of).

**12.7 Weight decay / dropout.** Both off in Phase 1A, recorded in the manifest. [2] studies the influence of "model capacity, weight decay and dropout regularization" on stability-plasticity; we remove them rather than control them, because EXP-1 is about the phenomenon, not the cure.

**12.8 The order we chose.** C6 reverses it.

---

## 13. The figures we will have at the end

1. **Forgetting curve.** Steps in B (x) vs A-accuracy, A-loss, ‖θ−θ_A‖ (three y-panels, shared x). With C4's low-LR curve overlaid.
2. **Per-layer update profile.** Grouped bars: for each of the 14 tensors, `r_l` for τ_A and for τ_B side by side. *This is the figure I believe is not in the literature* (Section 14).
3. **Interference heat strip.** Per layer: cos(τ_A, τ_B), sign-flip rate, support overlap. Expect strongly negative values in the head, near-zero in early conv.
4. **The money plot.** Per layer, scatter of **Fisher-weighted drift of B** (x) vs **drop in A-probe accuracy** (y), one point per layer, with the trend line and its Spearman ρ. If drift in A-important coordinates predicts loss of A-recoverable information, we have an instrument; if raw ‖Δθ‖ predicts it worse, we have a negative result worth stating loudly, because people still use raw norms.
5. **The A2/A3/A4 decomposition bar.** Four bars: `a[A,A]`, `a[A,B]_full`, `a[A,B]_headA`, `a[A,B]_probe`, with chance line at 10%.
6. **Loss barrier profile.** L_A along the θ_A→θ_B line, 21 points, annotated at the maximum.
7. **Seeds box-plot** of ΔA across all conditions and all 5 seeds. Box, not error bar — spread is the point.

---

## 14. What is ours to contribute, and what is already taken

**Narrowed 2026-10-01.** The original claim here — "no verified paper plots per-layer parameter change for task A versus task B alongside measured forgetting" — **is false as stated**, and two papers found after the design was frozen collide with it directly. The corrected claim is below, and it is materially smaller.

Two collisions, both now read in full:

- **Zhao, Zhou, Long, Jiang & Zhang, ICML 2023** [38] already measures per-module parameter change alongside forgetting in continual learning, using L1 change normalised by module size — `(1/|θ|)‖θ_{t,n} − θ_{t,n−1}‖₁` — and concludes "only a few modules are more task-specific and sensitively alter between tasks, while others can be shared across tasks as common knowledge. Hence, we attribute forgetting mainly to the former." So *localising forgetting to modules is taken*. What they do **not** do: report any correlation between drift and forgetting (full-text search of both the PMLR and arXiv versions returns zero hits for `correl*`, `spearman`, `pearson`, `rho`), compute anything between two task vectors, or probe representations.
- **Davari, Asadi, Mudur, Aljundi & Belilovsky, CVPR 2022** [37] already runs the §7 A4 measurement: a linear classifier re-fitted on frozen post-task-2 activations. On a two-task Split-CIFAR-10 sequence, network accuracy fell 85 % → 63.6 % while the top-block probe fell only **5.72 points** (85.82 → 80.10), and lower-block probes *improved*. They state our intended reading outright: "A permutation of the features leading into the classification heads leads to total catastrophic forgetting as measured by standard approaches … However, this does not correspond to a loss of knowledge about the data." Their own limitation bounds their coverage: "it currently focuses on the task-incremental setting and does not consider the important class-incremental setting," and it contains no MNIST or Fashion-MNIST.

Also corrected: the previous paragraph described [26] and [27] as work on "two sequential fine-tunes." **They are not sequential.** Both are model-merging papers on *parallel* LLM fine-tunes — Gemma-3-1B safety tunes and Qwen3-8B RL specialists — that are subsequently merged (§17.1). That mistake made our slice look closer-taken than it is, and removing it helps us.

**What is actually unclaimed, and therefore ours:** per-layer task-vector geometry computed **between τ_A and τ_B of a sequential A→B pair** on a from-scratch sub-300k-parameter CNN — cosine, support overlap, sign-flip rate and spectral/effective rank (the merging literature reports cosine and overlap, but for parallel specialists on 1-8B models) — reported *jointly with* measured forgetting and correlated against it; plus a per-layer Fisher-weighted-drift audit, per-layer CKA between the two checkpoints, and a measured `cos(g_A, g_B)` magnitude. I found no primary measurement of any of these at this scale.

That is a small, honest, free-compute instrument slice, **not a paper on its own**, and P2 in §15 is now a **replication of [37]** rather than a novel prediction. It is also a well-defined negative-result target: if our own per-layer profile shows that raw drift is a good predictor, then half the field's intuitions are more defensible than this journal currently assumes.

---

## 15. Pre-registered predictions — written down before any run

The point of these is to be *wrong* on purpose where the literature is thin. If everything below comes true, EXP-1 mostly replicated known results and taught us little about weights.

| # | Prediction | Falsified if |
|---|---|---|
| **P1** | `a[A,A] ≥ 95%` for M1, and after stage B, `a[A,B]_full ≤ 30%` (chance = 10%) | forgetting is milder than 40 absolute points → LR too small or pair too similar; fix with C4 sweep before proceeding |
| **P2** | `a[A,B]_probe ≥ 70%` — i.e. linearly recoverable A-information survives stage B | probe ≤ 40% → information is genuinely destroyed, and the "merely stopped reporting" hypothesis is dead for this pair. Either outcome is a result |
| **P3** | `a[A,B]_headA < a[A,B]_probe` — keeping A's head is *worse* than fitting a fresh one, so at least some damage is representational rather than pure readout mismatch | head-A ≥ probe → the backbone is untouched and all forgetting is in the head, which would make M1 an uninteresting testbed and push us to Phase 1B immediately |
| **P4** | Per-layer cos(τ_A, τ_B) ≈ 0 or negative in conv layers; strongly negative in the head rows (B reassigns the same 10 coordinates) | positive cos in the head → tasks are more compatible than assumed |
| **P5** | ≥70% of B's Fisher-weighted drift lands on the top-10% A-important coordinates | drift spreads evenly → EWC's premise is empirically shaky at this scale, which is itself worth writing up |
| **P6** | The forgetting curve is **not** monotone: a transient minimum followed by partial recovery inside stage B (stability gap, [2]) | strictly monotone decline → note that the stability gap did not appear at our scale, which is a legitimate observation about small nets |
| **P7** | **C4 moves the answer by more than any architectural choice:** stage-B LR ÷10 cuts forgetting by ≥20 points while costing B less than 5 points | if LR barely matters, our update-count confound is not real and the protocol is cleaner than expected |
| **P8** | A→B and B→A give different forgetting magnitudes (order asymmetry) | symmetric within noise → report the symmetry, it constrains the mechanism story |
| **P9** | M2 (MLP) forgets *more* than M1 (CNN), because it has no spatial invariance to fall back on | MLP forgets less → architecture-dependence is real and larger than assumed; chase it |
| **P10** | Restoring θ_A's BatchNorm buffers (12.6) recovers a visible but minority share of A2 (<10 points) | BN restoration recovers most of A2 → our headline would be a BN statistics story, and we must say so loudly |

**Statistical protocol.** 5 seeds, identical seed sets across every condition, mean ± **SD** (not SEM — SEM hides seed-to-seed spread at n=5), and paired per-seed comparisons between conditions rather than comparing independent means. Five is the floor; the published convention at this scale is higher: [24] uses "averages over 10 runs" for CIFAR-100 and 5 for other settings, [1]'s journal version says "Experiments were run 10 times, reported is the mean (± SEM)," and [14] reports "the average and standard deviation over five runs." Given a ~1-minute run, we can afford **10 seeds for Phase 1A** and that is what we plan; Bouthillier et al. exist for a reason — "this is prohibitively expensive, and corners are cut to reach conclusions" [29].

---

## 16. Runbook

### 16.1 Environment (verified today, not assumed)

- Python **3.14.6** at `C:\Python314\python.exe`, single interpreter on this machine.
- `torch` **2.14.0+cu126** is confirmed available for this interpreter from PyTorch's index (`pip index versions torch --index-url https://download.pytorch.org/whl/cu126` → 2.14.0+cu126). **This install has not been performed yet.**
- GPU: RTX 3050 Laptop, **4096 MiB**. Driver 616.56. Disk: 151 GB free.
- **Decision: hand-roll the training loop (~500 lines), do not install `avalanche-lib`.** PyPI metadata for `avalanche-lib` was checked and no usable `requires_python` was surfaced; its own install docs assume a managed environment, and neither Avalanche nor Mammoth documents Python 3.14 support. Mammoth additionally requires `torch >= 2.1.0` and marks GEM "_Unavailable on windows_." On a laptop with one interpreter, an unfamiliar library's dependency resolver is the fastest way to lose a day. We use Avalanche and Mammoth as **reference implementations** we read, and as the source of standard split definitions we reproduce exactly [30][31].
- If a CUDA wheel install ever fails: everything in Phase 1A runs on CPU. The M1 CNN on MNIST is CPU-viable for a single seed; for 10 seeds, budget ~4–6 h CPU against ~40 min GPU, so GPU is worth the setup time but is not a blocker.

### 16.2 Planned layout

```
EXP-1-CATASTROPHIC-FORGETTING/
├── EXP1_JOURNAL.md          <- this document (living; new dated entries appended in Section 19)
├── EXP1_JOURNAL.pdf         <- rendered copy
├── render_pdf.py            <- md -> html -> PDF (headless Chrome)
├── PRE_REGISTRATION.md      <- frozen copy of Sections 6-15, hashed before run #1
├── src/                     <- models.py, data.py, train.py, forensics.py, probes.py, plot.py
├── configs/                 <- phase1a.yaml, phase1b.yaml, one yaml per control
├── runs/<run_id>/           <- Section 6 artifact table, one dir per (condition, seed)
├── figures/                 <- Section 13 outputs
└── data/                    <- MNIST/, FashionMNIST/, cifar-10-python.tar.gz
```

### 16.3 Time budget, with the arithmetic

Phase 1A, per run: stage A ≈ 4,690 steps at 10 epochs × 60,000/128 ≈ 1 min; stage B same, plus ~940 cheap subset evals; forensics ≈ 1 min. **≈3 min per run** including Fisher estimation and probing.

- Core sequential run + C1/C2/C3/C4/C5(implicit)/C6/C7/C8 + BN variant ≈ **12 conditions × 10 seeds = 120 runs ≈ 6 h GPU.**
- Realistic, with the inevitable bug that invalidates the first 20 runs: **one working day.**
- Phase 1B: ResNet-18 at 11.7 M params [18], 2 tasks × 50 epochs × 25,000 images at batch 32 + AMP. Extrapolating from the published "approximately 2.5 hours on Seq. CIFAR-10" DER++ figure on a Titan X [32], a 3050 Laptop is slower, so **~1–2 h per task**. Core + 3 controls × 3 seeds ≈ **1.5–2 days**, mostly unattended. This is why 1B waits until 1A is written up.

### 16.4 Go / no-go checks before run #1

A checklist, because the failure mode of a first experiment is not a bad hypothesis, it is a broken harness.

- [ ] `torch.cuda.is_available()` is True and `torch.cuda.get_device_name(0)` prints the 3050
- [ ] Peak VRAM during one stage-A epoch < 3.5 GB
- [ ] Dataset files land in `data/`, and their checksums are written into `manifest.json`
- [ ] Both train loaders yield the declared shapes and dtypes; a plotted batch of 8 A-images and 8 B-images **looks like** digits and garments respectively (catches transposed axes and the label-10-is-zero style bugs)
- [ ] One seed trains M1 to ≥97% on A and M2 to ≥96% — if solo training is broken, every downstream delta is meaningless
- [ ] STEP 2 load-verification passes bit-exactly on logits
- [ ] A zero-training run reports A2 == A1 exactly (proves the eval harness and the checkpoint plumbing are the same path)
- [ ] Eval is deterministic: fixed generator for the A-subset, `model.eval()` everywhere, no dropout, `torch.use_deterministic_algorithms` where it does not cost us
- [ ] `git commit` hash recorded per run

---

## 17. Sources — verified vs. not

Verification tags: **[M]** = I opened it this session. **[A]** = a research agent dispatched for this design opened it and reported a verbatim quote; treat the quote as trustworthy but **spot-check before citing externally.**

### 17.1 Verified

1. **[A]** van de Ven & Tolias, *Three scenarios for continual learning*, arXiv:1904.07734 — https://arxiv.org/abs/1904.07734 · definitions and the 87.19/59.21/19.90 baseline table
2. **[A]** De Lange, van de Ven & Tuytelaars, *Continual evaluation for lifelong learning: Identifying the stability gap*, arXiv:2205.13452, ICLR 2023 — https://arxiv.org/abs/2205.13452 · **"forgetting is temporary and followed by a phase of performance recovery"** — I independently re-opened this abstract page [M]; it is *not* a Mirzadeh paper
3. **[A]** Ilharco et al., *Editing Models with Task Arithmetic*, arXiv:2212.04089, ICLR 2023 — "build task vectors by subtracting the weights of a pre-trained model from the weights of the same model after fine-tuning on a task"
4. **[A]** Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, arXiv:1612.00796 / PNAS 114(13):3521-3526, DOI 10.1073/pnas.1611835114 — "selectively slowing down learning on the weights important for those tasks"
5. **[A]** Zenke, Poole & Ganguli, *Continual Learning Through Synaptic Intelligence*, arXiv:1703.04200, ICML 2017 — "each synapse accumulates task relevant information over time"
6. **[A]** Saha et al., *Gradient Projection Memory for Continual Learning*, arXiv:2103.09762, ICLR 2021 — SVD subspaces, orthogonal gradient steps
7. **[A]** Wang et al., *Training Domain-Incremental Neural Networks with Half-space Parameterization* (Adam-NSCL), arXiv:2103.07113, CVPR 2021 — null space of the uncentered input covariance per linear layer
8. **[A]** Xiao et al., *Fashion-MNIST*, arXiv:1708.07747 — "a direct drop-in replacement for the original MNIST dataset"; 70,000 images, 10 categories, 7,000 each
9. **[A]** Chaudhry, Dokania, Ajanthan & Torr, *Riemannian Walk for Incremental Learning*, ECCV 2018, arXiv:1801.10112 — the forgetting and intransigence measures. **Attribution corrected:** these come from RWalk, *not* from A-GEM and *not* from Chaudhry et al. Neurocomputing 2019
10. **[A]** Mirzadeh et al., *PCGrad*, arXiv:2001.06782, NeurIPS 2020 — gradient interference conditions
11. **[A]** Yadav et al., *TIES-Merging*, arXiv:2306.01708, NeurIPS 2023 — "resolving sign conflicts … merging only the parameters that are in alignment with the final agreed-upon sign"
12. **[A]** Yu et al., *Language Models are Secretly Simple*, *DARE*, arXiv:2311.03099, ICML 2024 — "SFT delta parameter value ranges are typically small (within 0.002) with extreme redundancy, and DARE can effortlessly eliminate 90% or even 99% of them"
13. **[A]** Kornblith et al., *Similarity of Neural Network Revisited (CKA)*, arXiv:1905.00414, ICML 2019
14. **[A]** Mirzadeh et al., *Understanding the Role of Training Regimes in Continual Learning*, arXiv:2006.06958 — LR decay, batch size, dropout "can outperform considerably more complex algorithms meant to deal with continual learning"
15. **[A]** Frankle et al., *Linear Mode Connectivity and the Lottery Ticket Hypothesis*, arXiv:1912.05671, ICML 2020 — the loss barrier
16. **⚠ MISATTRIBUTED — do not cite.** arXiv:2410.16154 was carried here as "methodology study of sequential MNIST/Fashion/CIFAR tasks" with the quote "T1 was suffered from catastrophic forgetting and T1 performance reduced to zero." The ID resolves to **Bazhenov, Dewasurendra, Krishnan & Delanois, *Unsupervised Replay Strategies for Continual Learning with Limited Data*** — a biology-inspired replay method whose abstract claims *improved* accuracy, i.e. the opposite of the use it was put to. Neither the quote nor the framing is supported by the title or abstract; the body could not be read, so this is "unsupported by what was checked," not "absent from the paper." Caught 2026-10-01 by two independent searches. **No substitute source for "MNIST→Fashion forgetting is severe" has been found — see §2.2, which no longer rests on it.**
17. **[A]** *(agent-opened 2026-10-01; abstract only)* Hiratani, *Disentangling and Mitigating the Impact of Task Similarity for Continual Learning*, arXiv:2405.20236 (NeurIPS 2024). Correct abstract wording: high input similarity with low readout similarity is "catastrophic **for retention**" — *not* "for both knowledge transfer and retention," as this journal previously quoted it; transfer is treated on an opposing axis ("improves knowledge retention **at the expense of transfer**"). **Scope, which the old citation hid:** an *analytical* result for a **linear teacher-student model with latent structure**, confirmed on a permuted-MNIST task with latent variables. No CNN, no Fashion-MNIST, no accuracy table. Do not use it as an empirical prediction for a 242k-parameter net. (The NeurIPS PDF returned no extractable text; abstract-level reading only.)
18. **[A]** SE-Pruned ResNet-18 paper — ResNet-18 parameter count "from 11.7 million to 8.5 million"
19. **[A]** CFC continual-learning README (medium authority, not peer-reviewed) — 2-task Split-CIFAR-10: "Task 1: 84.4% → Task 2: 52.85% Class-IL / 62.92% Task-IL"; 5-task SGD run "Final Class-IL: 19.22% (severe forgetting)"
20. **[A]** Cai, Sener & Koltun, *Clear: Continual Learning on Real-World Distributions*, ICCV 2021, arXiv:2108.09020 — YFCC100M-derived, 11 segments, out of our reach
21. **[A]** arXiv:1812.00420 (A-GEM, ICLR 2019) — AWA/CUB task construction, 3×224×224, standard ResNet-18; also the accuracy-metric definitions we cite in Section 11
22. **[A]** arXiv:2405.09673 (TMLR 2024) — "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations"
23. **[A]** arXiv:2506.13234 (ICML 2025) — the drift-metric suite: L2, loss barrier, permutation-aligned distance, CKA
24. **[A]** Masana et al., *Class-Incremental Learning: Survey and Performance Evaluation*, arXiv:2010.15277, IEEE TPAMI — FT vs FT+ baseline, 10-run/5-run convention, the `a_{k,j}` output-space warning (also via [A] arXiv:2302.00487, Wang et al.'s comprehensive survey)
25. **[A]** ICLR 2025 blogposts, *On the Computation of the Fisher Information in Continual Learning* — https://iclr-blogposts.github.io/2025/blog/fisher/ — "significant differences between the different options also in terms of their best performance"
26. **[A]** *(metadata verified 2026-10-01 via arXiv API + abstract)* Choudhary, Rocha, Seo, Sharma & Chaudhary, *Asymmetric Collapse in Model Merging: When Refusal Overwrites Recognition*, arXiv:2607.27240 — reports per-layer task-vector magnitudes and cosine ≈ 0.011. **This journal previously described it as "two sequential fine-tunes." It is not.** These are two *parallel* Gemma-3-1B-IT fine-tunes on complementary safety objectives (CARES / WildJailbreak) that are then **merged**. A model-merging paper, not a continual-learning one.
27. **[A]** *(metadata verified 2026-10-01)* McClendon, *When Model Merging Rivals Joint Multi-Task Reinforcement Learning: A Task-Vector Geometry Analysis*, arXiv:2607.16062 — "near-orthogonal (cosine 0.06 - 0.10) despite ~65% support overlap," on Qwen3-8B RL specialists. **Same correction: parallel specialists, not a task sequence.**
28. **[A]** *(abstracts verified 2026-10-01)* arXiv:2207.02099, Gulcehre et al., *An Empirical Study of Implicit Regularization in Deep Offline RL*, carries "the rank of the penultimate feature layer, also called *effective rank*, has been observed to drastically collapse during the training" — **but the same paper then disowns it**: "a direct association exists only in restricted settings and disappears in the more extensive hyperparameter sweeps," and "studying this association under simplistic assumptions could be highly misleading." It is an offline-RL study, not supervised or continual learning. arXiv:2309.00257 (Fu) uses effective rank as a *federated-learning* metric and does not establish penultimate-layer collapse. **W7 in §9 must not lean on either as evidence that rank collapse tracks damage.** Note also that a typo of this ID — **arXiv:2607.02099** — resolves to *X-Splat*, a dental CBCT imaging paper; check the digits before quoting.
29. **[A]** Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021, arXiv:2103.03098 — "this is prohibitively expensive, and corners are cut to reach conclusions"
30. **[A]** Avalanche — https://github.com/ContinualAI/avalanche · https://avalanche.continualai.org/ · arXiv:2104.00405 and arXiv:2302.01766 (JMLR). Ships `SplitMNIST`, `PermutedMNIST`, `RotatedMNIST`, `SplitCIFAR10`, `SplitTinyImageNet`, `CORe50`, `SplitCUB200`, `cfashion_mnist`; `Naive(model, optimizer, criterion)` **is** our fine-tuning baseline; helpers `accuracy_metrics`, `forgetting_metrics`
31. **[A]** Mammoth — https://github.com/aimagelab/mammoth · "23 datasets", "more than 70 methods", `--dataset seq-cifar10`, baseline listed as "SGD - Vanilla fine-tuning (catastrophic forgetting baseline)". **Cite the repo, not an arXiv ID** (see Section 17.2)
32. **[A]** arXiv:2004.07211 (DER++, ICLR 2020) — "approximately 2.5 hours on Seq. CIFAR-10" on a Titan X; our time extrapolation. **Table 2 numbers used in §2.2 are also from this paper and are on p.6 of our local `01-continual-learning/PAPER 7.pdf`** — read three times independently (twice from the PDF text layer, once from the arXiv file) with exact agreement: Split-CIFAR-10 class-IL SGD 19.62 ± 0.05 / task-IL 61.02 ± 3.33 / JOINT 92.20 / 98.31; Split-TinyImageNet SGD 7.92 ± 0.26 / 18.31 ± 0.68 / JOINT 59.99 / 82.04; Permuted-MNIST DIL 40.70 ± 2.33 (JOINT 94.33); Rotated-MNIST DIL 67.66 ± 8.53 (JOINT 95.76). ResNet-18 **not pretrained**, 10 runs, buffer 500.
33. **[A]** GMvandeVen/continual-learning — reference code for [1]; "expected run-time on a standard desktop computer is ~6 minutes, with a GPU … ~3 minutes" per Split-MNIST run. **Read this before writing `train.py`**
34. **[M]** torchvision dataset docs + `mnist.py`/`svhn.py` source — MNIST/Fashion-MNIST/CIFAR-10/EMNIST/SVHN availability; CIFAR official page: 60,000 32×32 colour images, 50,000/10,000, 163 MB Python tarball
35. **[M]** PyPI/PyTorch index probes on this machine today: `torch 2.14.0+cu126` present for cp314; `avalanche-lib` Python support not stated
36. **[A]** *(agent-opened 2026-10-01; abstract only)* Goldfarb, Evron, Weinberger, Soudry & Hand, *The Joint Effect of Task Similarity and Overparameterization on Catastrophic Forgetting — An Analytical Model*, ICLR 2024, https://openreview.net/forum?id=u3dHl287oB — closed-form expected forgetting for two-task continual linear regression where task 2 is a random orthogonal transform of task 1. Direction depends on overparameterization: peak forgetting at **intermediate** similarity when highly overparameterized, monotone **decreasing** in similarity near the interpolation threshold. Neither branch makes maximum similarity the worst case. **Abstract read; body not read.**
37. **[A]** *(agent read IN FULL 2026-10-01 — abstract, Tables 4/5/7, Appendix 5.1)* Davari, Asadi, Mudur, Aljundi & Belilovsky, *Probing Representation Forgetting in Supervised and Unsupervised Continual Learning*, CVPR 2022, arXiv:2203.13381. **This is the A4 collision. Their protocol is ours:** a linear classifier re-fitted on the frozen activations of the base network after a new task is introduced, compared against end-to-end accuracy, per ResNet/VGG block. Two-task Split-CIFAR-10: network accuracy 85 % → 63.6 % while top-block probe fell only 5.72 points (85.82 → 80.10); lower blocks improved. Verbatim: "a model's representation can change without losing knowledge about prior tasks"; "A permutation of the features leading into the classification heads leads to total catastrophic forgetting as measured by standard approaches … However, this does not correspond to a loss of knowledge about the data." **Bounds on the collision:** task-incremental only ("does not consider the important class-incremental setting"); no MNIST, no Fashion-MNIST; ResNet-18/101/VGG scale, parameter counts not reported; their Split-CIFAR-100 fine-tuning probe at sequence end is only 64.8-70.5 %, so **our P2 ≥70 % bar is dataset-dependent, not principled.**
38. **[A]** *(agent read IN FULL 2026-10-01, both versions)* Zhao, Zhou, Long, Jiang & Zhang, *Does Continual Learning Equally Forget All Parameters?*, ICML 2023 / PMLR v202, arXiv:2304.04158 — per-module change metrics `(1/|θ|)‖θ_{t,n} − θ_{t,n−1}‖₁` (L1, normalised by module size, **not** a Frobenius relative ratio); "only a few modules are more task-specific and sensitively alter between tasks, while others can be shared across tasks as common knowledge. Hence, we attribute forgetting mainly to the former." **No correlation statistic anywhere**: full-text search returns 0 hits for `correl*`, `spearman`, `pearson`, `rho`. Module ranking is by intervention (FPF / k-FPF), not by correlating drift against forgetting. No task-vector geometry, no probes.

### 17.2 Could not verify — do not cite these

Three arXiv IDs I carried in from memory were **wrong**, caught during this design pass. I had them in my head, they sounded right, and they were not. This is the same failure `END_GOAL.md` §10 documents for this project, reproduced here on my own recall:

- **arXiv:2405.17103** — I believed this was "catastrophic forgetting with almost no weight change" / the stability-gap fine-tuning paper. I opened it: it is *Empowering Character-level Text Infilling by Eliminating Sub-Tokens* (Ren, Zhan, Wu, Li). The Mirzadeh paper I was thinking of was **not located in this session under any ID**; the term "stability gap" belongs to [2] instead. If we want the "forgetting with tiny weight change" result, find and read it properly first.
- **arXiv:2310.03540** — believed to be "Improved Fine-Tuning by Lowering the Loss Barrier Using Convex Overlays." Not that; and `all:"Convex Overlays"` returns nothing in the index. The *loss barrier* concept stands on [15], verified.
- **arXiv:2212.04023** — believed to be Task Arithmetic; it is a GaSe condensed-matter paper. Correct ID is [3].
- **Mammoth's arXiv ID (2303.15350)** — resolves to a topic-modelling paper. Cite the repository [31].
- **SPR / Average Forward Transfer closed forms** — the `SPR = sqrt((1−S)²+PL²)` formula is *not* in the arXiv versions of [2]'s sister survey; it may live only in a TPAMI camera-ready we did not open. Not used anywhere in this design.
- **"The Chaudhry forgetting measure is undefined for two tasks"** — too strong as circulated; corrected in Section 11 to "defined but degenerate."
- **Farquhar & Gal, arXiv:1805.09733** (*Towards robust evaluations of continual learning*) — reached only second-hand through [2]'s bibliography. Primary not opened. Relevant to Section 3's warning; read before leaning on it.
- **OWM** — an arXiv ID was never confirmed; the attribution I carried ("Zenke 2020") is unreliable — the citation trail points to Zeng et al. Not used in this design.
- **XAcc (Kuditipudi et al.), Pope et al.'s effective-rank paper, Adam-moment-reset as a named confound, any paper denouncing single-seed CL** — all unresolved. We instrument these ourselves (C7, Section 15) instead of citing them.

---

## 18. What EXP-1 does *not* do

- It does not test any anti-forgetting method. No EWC, no replay, no regularisation, no PEFT. We measure the disease in EXP-1; EXP-2 can compare treatments.
- It does not produce a pretrained-model or LLM result. Phase 1A's model has 242k parameters. Nothing here licenses a sentence about GPT-scale behaviour.
- It does not settle where knowledge is stored. A strong linear probe is evidence about *linear recoverability*, and only about that. A frozen-feature probe is not a mechanism proof, and I want that written in the design so it cannot drift into the results section later.
- It does not address energy at all. Consistent with `END_GOAL.md` §2.
- Two tasks is two tasks. The 5- and 10-task sequences that dominate the literature come after this, and the T=2 metric degeneracies in Section 11 are exactly why.

---

## 19. Journal entries

### 2026-09-25 — Design

Hardware and software probed before choosing anything: RTX 3050 Laptop with 4 GB VRAM, Python 3.14.6, no PyTorch installed, `torch 2.14.0+cu126` confirmed available, 151 GB free disk. This fixed the ceiling on everything else — a 4 GB laptop GPU is generous for a 242k-parameter net and hostile to ResNet-18 at CIFAR scale.

Three parallel literature threads were dispatched (task-pair choice, weight-instrumentation methods, metrics and traps) with a standing rule that no citation may be reported unless the page was actually opened and quoted. That rule paid for itself immediately: three of my remembered arXiv IDs were wrong, one of them load-bearing for my favourite idea (Section 17.2).

Decision record, and the reason each was made:

- **MNIST → Fashion-MNIST as primary, Split-CIFAR-10 deferred.** Chosen for cost-per-run and for how legible the weight story is, not for external validity. Accepted cost: our headline numbers are not directly comparable to published CL tables until Phase 1B runs.
- **From scratch, not pretrained.** So we own θ₀ and can hash it.
- **Decompose the accuracy drop three ways** (Section 7). This came out of the research, not out of the original sketch — the original was "test A again and compare." Splitting total forgetting into representational damage versus linearly recoverable-but-unreported is the single upgrade that makes EXP-1 relevant to `END_GOAL.md`'s unlearning-verification wedge.
- **Drop BWT and the forgetting measure from the metric table** after [9] made clear both are the same number as ΔA when T=2. Nearly lost a week reporting one quantity in three costumes.
- **C4 (LR and update-count matching) elevated to co-primary with the result itself,** on [14]'s evidence that training-regime knobs beat most published methods.
- **Hand-roll the loop instead of installing Avalanche** — Python 3.14 support is undocumented for the whole CL library ecosystem, and a dependency-resolution fight is not the experiment.
- **Ten seeds for Phase 1A** rather than the five I first planned, because ~3 min/run makes 10 affordable and [29] is exactly about what it costs to skip this.
- **Novelty claim scoped honestly:** per-layer A-vs-B update profile correlated with measured forgetting appears unclaimed (Section 14). It is a small, free-compute, instrument-level contribution and I do not want it oversold in the eventual writeup.

Open risks I am not confident about: (a) whether the load-verification in STEP 2 catches BatchNorm buffer drift, which is a physical forgetting mechanism and not plumbing (12.6); (b) whether `a[A,B]_probe` will really separate A2 from A4 cleanly on 28×28 grayscale, or whether the probe's capacity is the binding constraint rather than the features' — mitigation: also run a 2-layer nonlinear probe on f_B so we can tell "no linear readout" from "no information"; (c) the whole plan assumes ~3 min/run, and that estimate is arithmetic, not measurement.

**Next action:** freeze Sections 6–15 into `PRE_REGISTRATION.md`, hash it, then write `src/` and run the Section 16.4 checklist. No results before the checklist is green.

### 2026-10-01 — Citation audit, and the novelty claim does not survive it

A full scan of the continual-learning literature (all 35 library PDFs plus four retrieval waves and two adversarial verification passes) was run to establish what is already known before committing compute. Report: `../DEEP_RESEARCH_CL_KNOWLEDGE_LEDGER.md`. It invalidated five entries in §17.1 and, through them, two load-bearing passages. Old → new wording, so the reasoning is auditable rather than silently overwritten:

| Where | Was | Now | Why |
|---|---|---|---|
| §2.2 bullet 1 | "The drop is documented as severe" — quoted arXiv:2410.16154 for "T1 performance reduced to zero" | Severe forgetting is documented **only in class-incremental settings**; in our Domain-IL-like regime it is not established (59.21 % Split-MNIST DIL, 40.70 % Permuted, 67.66 % Rotated) | 2410.16154 is Bazhenov et al., *Unsupervised Replay Strategies* — a different paper claiming **improved** accuracy [16] |
| §2.2 bullet 4 | "Theory says this shape of pair is the destructive one," Hiratani quoted as catastrophic "for both knowledge transfer and retention" | Theory does **not** say this pair is destructive. Analytical linear teacher-student result, **retention only**, no CNN, no Fashion-MNIST; and two sources point the other way | The quote was overstated and the paper's scope mis-described [17]; counter-evidence from [2, Table 2] and Goldfarb et al. [36] |
| §14 | "no verified paper plots per-layer parameter change for task A versus task B alongside measured forgetting" | Claim **narrowed** to pairwise task-vector geometry + drift-vs-forgetting correlation on a from-scratch sub-300k CNN | Zhao et al. [38] already localises forgetting to modules; Davari et al. [37] already ran the A4 probe |
| §17.1 [26]/[27] | "two **sequential** fine-tunes" / "two sequential specialists" | Both are **parallel** LLM fine-tunes that are then **merged** (Gemma-3-1B safety tunes; Qwen3-8B RL specialists) | Wrong framing. **This correction helps us** — it removes the only work that looked like ours |
| §17.1 [28] | effective rank "observed to drastically collapse during the training" | Same sentence exists, but the paper **disowns it**: the association "disappears in the more extensive hyperparameter sweeps." Offline-RL, not CL | W7 was leaning on a source that argues against its own use |

**The one that actually stings is [37], Davari CVPR 2022.** P2 — "`a[A,B]_probe ≥ 70 %`, i.e. linearly recoverable A-information survives stage B" — is not an open prediction. They ran our exact protocol and got probe 85.82 → 80.10 while network accuracy fell 85 → 63.6, and they wrote our interpretation sentence first: "this does not correspond to a loss of knowledge about the data." **So P2 is now a replication, and I am keeping it** — it is nearly free, their setting is task-incremental at ResNet scale with no MNIST/Fashion anywhere, and our headline is Domain-IL-like, which they explicitly exclude. But the writeup must say "replicates [37] in a domain-incremental two-task setting" and never "we show that." Their probe on Split-CIFAR-100 lands at 64.8-70.5 %, so the ≥70 % bar was a guess about the dataset, not about the phenomenon.

**On the §14 novelty claim after narrowing:** it is smaller than I wrote on 09-25 and it is still not empty. What nobody has done is compute τ_A-vs-τ_B geometry (cosine, support overlap, sign flips, spectral rank) **between the two task vectors of a sequential pair** on a network small enough to interrogate exhaustively, *and* correlate it against measured forgetting, *and* audit it with Fisher-weighted drift and per-layer CKA. Zhao does module attribution with no correlation coefficient at all — zero hits for `correl*`/`spearman`/`pearson` in both published versions — which is precisely the gap. **Also caught: the "ρ ≈ 0.09" figure that was circulating as a Zhao result does not exist in that paper.** Do not cite it.

**Two structural problems the audit surfaced, not yet acted on — these need a decision, not a citation fix:**
1. **The shared 10-unit head confounds our own headline.** With one head and unrelated label semantics, row *k* is reassigned by construction, so A2 mixes head-reuse with representational damage. A3/A4 partially handle it; nothing handles A2. Reported in §7 as a measurement, it is arguably a protocol artifact.
2. **T=2 cannot support any claim about cumulative forgetting**, and Farquhar & Gal say so in print: "Showing that a method succeeds on a two-task transfer does not entail that it will work on a longer series." A 5-task arm on the same architecture (MNIST → Fashion → KMNIST/EMNIST → SVHN → Rotated-MNIST graded) fixes this, kills the §11 metric degeneracy, and is the only way loss of plasticity becomes observable at our scale — which is the bigger finding and one §9 does not measure at all.

**Process note:** §14 sits inside the 6-15 range destined for `PRE_REGISTRATION.md`, and that file does not exist yet, so no hash was broken — but the freeze must now be taken *after* these corrections, not before. The evaluation cadence also needs raising from every K = 50 steps to ρ_eval = 1 with min-ACC / WC-ACC / WF10 reported, on [2]'s own recommendation rather than my judgement: they missed a task's drop to zero for 2 of 5 seeds at ρ_eval = 100.

**Do-not-run list the audit produced**, none of which was in this design but all of which are plausible next questions: EWC/SI/MAS as class-incremental contenders (Split-MNIST CIL 19.90-20.01 vs chance 19.90 — [1]); GEM/A-GEM expecting an advantage; generative replay (18-34 min vs ~7 min for coresets, and no budget-matched win documented); prompt/expansion methods as robustness contenders (they "frequently collapse on unseen data despite high initial benchmarks"); Forward-Forward positive/negative-phase consolidation — Hinton retracted that experiment himself, in the version of PAPER 13 we hold.
