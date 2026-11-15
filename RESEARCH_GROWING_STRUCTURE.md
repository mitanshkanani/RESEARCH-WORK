# Growing new structure instead of overwriting: what the evidence says, and what to test

**Date:** 2026-11-12 · **Scope:** methods that add connections, neurons or modules for new knowledge, and whether they can learn a class from 2-3 examples without forgetting.

**How this was produced.** Four Sonnet readers each opened the primary papers for one thread and returned quoted findings. All 16 arXiv IDs cited here were then checked against the arXiv API and resolve to the stated titles. The readers mostly worked from extracted passages of the full text, not a page-by-page read, so numbers should be checked against the PDF before they go into anything you publish.

Evidence tags: **[read]** passages extracted from the full text · **[abstract]** abstract only · **[unverified]** not opened, or a detail that could not be confirmed.

---

## 1. The short answer

1. **Giving new knowledge its own place does stop forgetting.** This part of the idea is right and has been shown many times.
2. **Adding a class from a few examples without forgetting is nearly solved once good features exist.** A classifier that keeps one running average ("prototype") per class does it with no backprop and no stored images.
3. **The accuracy comes from the features, not from the learner.** With pretrained features a prototype classifier reaches 83.7% on CIFAR-100; with random features on raw pixels the best result found is 28.6% (the two setups are not identical).
4. **Nobody was found doing all of it at once:** few examples per class, no stored old examples, no task hint at test time, and no large pretraining. Three readers searched for this separately. That is absence of evidence from one search pass, not proof.
5. **So the open problem is narrower than "solve forgetting".** It is: how do you get features good enough for few-shot learning without a huge training run, and keep them improving without damaging old knowledge? This is where an original contribution would sit.

---

## 2. Three things every method has to answer

Using the cats-vs-dogs-then-digits example:

| Question | Why it matters |
|---|---|
| **Does it need a task hint?** | Some methods only work if someone says "this is an animal picture" before the model answers. That is called task-incremental. The hard, realistic case is no hint (class-incremental). |
| **Does it keep old examples?** | Many methods quietly store a few thousand old images and re-show them (replay). That works, but it is the thing we want to avoid. |
| **Where did its features come from?** | To tell a dog from 3 images, the system must already see edges, fur and shapes. Either it borrowed that from a pretrained network or it had to learn it. |

---

## 3. What each family of methods actually delivers

### 3.1 Networks that grow per task

| Method | What it adds | Task hint needed? | Stores old examples? | Result |
|---|---|---|---|---|
| Progressive Networks (1606.04671) | A whole new column per task, old ones frozen | Yes | No | "Immune to forgetting", tested on games only **[abstract]** |
| PackNet (1711.05769) | Nothing; prunes and freezes a slice of weights per task | Yes | No | Zero forgetting by construction; later tasks get worse as space runs out (+3% error by the third slot) **[read]** |
| CPG (1910.06562) | Prunes, reuses, grows filters when accuracy target is missed | Yes | No | Beats DEN and PackNet in its own comparison; 1.5x weights after 20 tasks **[read, partial]** |
| DEN (1708.01547) | Neurons, when loss stays high | Yes | No | Criticised by two later papers: "DEN does not ensure non-forgetting" **[read]** |
| RCL (1805.12369) | Units chosen by a reinforcement-learning controller | Yes | No | No forgetting, but 3936 s training vs 167 s for the simpler baseline **[read]** |
| DER (2103.16788) | A new feature extractor per step | **No** | **Yes, 2,000 images** | CIFAR-100, 10 steps: 74.64% vs iCaRL 65.27% **[read]** |
| FOSTER (2204.04662), MEMO (2205.13218) | Temporary or partial new blocks | **No** | **Yes** | Within about a point of DER **[read]** |
| Width Expansion (2609.37702) | Neurons on a loss criterion, from scratch | No | Combined with replay or regularisers | CIFAR-100: 25-33% vs 48.56% for training on everything at once **[read]** |

Two findings matter most here:

- **The methods with exactly zero forgetting all need the task hint.** They do not solve the cats/dogs/digits case as you described it.
- **The no-hint growth methods lean on stored examples.** A survey (2302.03648) found that when memory is counted fairly, DER's lead over plain replay shrinks from 9.07 points to 1.74, and concludes "there is no free lunch" **[read]**.

### 3.2 Brain-style mechanisms

| Method | Mechanism | Gradient? | Result | Catch |
|---|---|---|---|---|
| **FlyModel** (2107.07617) | Fixed random sparse expansion, winner-take-all, then only the weights from active units to the correct class are strengthened | **None** | No task hint, no replay. MNIST-20: 0.86 after 10 classes vs 0.77 for the best replay-style baseline, 0.58 for EWC. Sparse coding 0.64 vs dense 0.07 **[read]** | Only fixes the classifier, not the features. Its inputs were not raw pixels: CIFAR-100 used "the penultimate layer (512 hidden nodes) of ResNet18 pre-trained on ImageNet", and MNIST-20 used 84 features from a LeNet5 trained on KMNIST **[read from the PDF, 2026-11-12]**. Not tested with few examples. |
| **CLP on Loihi 2** (2511.01553) | A prototype neuron is added for each new class, local three-factor rule | **None** | No replay. 1 example per class: 55.4% vs 54.4% for the GPU baseline. 25 examples: 90.0% vs 95.7%. Energy 0.05 mJ per update vs 281 mJ **[read]** | Uses a fixed pretrained feature extractor. Needs a neuromorphic chip for the energy number. |
| FearNet (1711.10563) | Fast memory, slow memory, "sleep" consolidation | Backprop in slow memory | Beats iCaRL with 10.7 MB vs 166 MB **[read]** | Pretrained ResNet-50 features; relies on generated replay |
| Growing Dual-Memory (1805.10966) | Self-organising networks that add neurons on novelty | None in the memories | CORe50 about 87% **[read]** | First class drops to 79.53% without replay |
| Active Dendrites (2201.00042) | Dendrites gate each unit by context, sparse activity | Backprop | 94.6% on 10 permuted-MNIST tasks, 81.4% on 100 **[read]** | MNIST-scale; routing works because permuted tasks look very different |
| Neurogenesis Deep Learning (1612.03770) | Adds nodes when reconstruction error is high | Backprop | Reconstruction results only on MNIST-scale data **[read]** | Fails without replay, by its own figure |
| NISPA (2206.09117) | Rewires a sparse network, freezes stable units | Backprop | Good with task hint **[read]** | Without the hint it needs replay and reaches 38-57% on CIFAR-10 vs a 79% reference |
| ART / Fuzzy ARTMAP, SOINN, Growing Neural Gas | Add a category or node on novelty | None | **[unverified]**: not opened | Still to be read |

### 3.3 Frozen features plus a classifier that grows without backprop

| Method | Mechanism | Result |
|---|---|---|
| Deep Streaming LDA (1909.01520) | Running class means plus one shared covariance, updated one sample at a time | Beats iCaRL by 6-11%; **30 minutes on ImageNet vs 31-62 hours** for replay methods; 0.001 GB storage **[read]** |
| Nearest-class-mean baseline (2210.04428) | Class means on frozen ViT features | 83.70% on Split-CIFAR-100 with zero buffer, level with replay methods **[read]** |
| FeCAM (2309.14062) | Prototypes with per-class covariance | 85.7% vs 83.7% for plain means **[read]** |
| RanPAC (2307.02251) | Random projection to 10,000 dimensions, decorrelated prototypes | 92.2% on CIFAR-100; rated the most energy-efficient method to train (2409.18664) **[read]** |
| First Session Adaptation (2303.13199) | Adapt features once, then LDA head | Few-shot CIFAR-100: 70.9 vs 55.7 for a replay method **[read]** |
| C-FSCIL (2203.16588) | Hyperdimensional memory, one vector per class | 5 examples per class, 76.37% falling to 48.87% after 100 classes **[read]** |

Every one of these papers states the same limit in its own words. RanPAC: "completely reliant on supply of a good generic feature extractor". FeCAM: "not apt when training from scratch". Streaming LDA: "reliant on robust deep feature representations".

### 3.4 Evidence on pretrained versus from-scratch features

- Random features on raw pixels (RanDumb, 2402.08823): 28.6% on CIFAR-100, which still beat the best trained online method at 26.5% **[read]**.
- Pretrained features lose much of their value on data unlike their training data: 35.7% vs an 83.7% upper bound on one such dataset **[read]**.
- "Feature reuse is the dominant factor" in few-shot meta-learning (1909.09157) **[read]**.
- Methods called "few-shot" still pay for features: Prototypical Networks train on 4,800 Omniglot classes first; C-FSCIL runs 70,000 meta-learning iterations **[read]**.
- Human one-shot learning (Lake 2015) and the developmental evidence that it relies on prior experience were seen only as an abstract and search snippets **[unverified]**.

### 3.5 Two further findings

- **Loss of plasticity** (Dohare et al., Nature 2024): a network trained continually for a long time gradually loses the ability to learn anything new. The fix is to keep re-initialising a small fraction of little-used units. This is a separate problem from forgetting, and a complete solution probably needs both **[read]**.
- **Growth only when needed** (SEMA, 2403.18886): adds a small adapter only when new data looks different from anything seen, no replay, 86.98% on CIFAR-100, expanding at only 3 tasks on one example layer. Requires a pretrained backbone; untested without one **[read]**.

---

## 4. Ranked methods to test on the 4 GB laptop

| Rank | Method | Why test it | Brain-like? | Risk |
|---|---|---|---|---|
| 1 | **Prototype / streaming-LDA classifier** | Strongest evidence, no backprop, no replay, minutes to run. This is the baseline anything new must beat. | Partly (one unit per class) | None; it is a baseline |
| 2 | **FlyModel sparse expansion with local rule** | No gradients at all, no task hint, no replay, and closest to "strengthen only the connections that matter". Never tested with 2-3 examples. | Yes | Its published results used pretrained-network features, so its behaviour on cheap features is unknown |
| 3 | **Prototype neurons added on novelty (CLP-style)** | Literally grows a neuron per new class, with real one-shot results and measured energy. | Yes | Energy number needs special hardware; on a laptop only the algorithm can be tested |
| 4 | **Random projection plus decorrelation (RanPAC / RanDumb)** | Shows how far features with no training can go. Use about 5,000-10,000 dimensions; 25,000 would need about 2.5 GB for one matrix. | Partly (random expansion) | Low accuracy on raw pixels |
| 5 | **Unit re-initialisation (continual backprop)** | Needed later, when features are being learned continually. | Partly (neuron turnover) | Not a forgetting fix by itself |
| 6 | **Growth triggered by novelty (SEMA-style)** | The best existing "grow only when needed" rule. | Partly | Only shown on a large pretrained model |

**Not worth testing as contenders:** PackNet, CPG, Progressive Networks (need the task hint); DER, FOSTER, MEMO (replay plus a full network per step); DEN, RCL (hint needed, criticised or slow); Neurogenesis Deep Learning and NISPA (need replay without the hint).

---

## 5. Proposed EXP-2: how cheap can the features be before few-shot learning breaks?

No reader found a paper that runs the same few-shot, no-forgetting learner on random, cheaply learned, and fully pretrained features. That comparison answers the pretrained-model question with numbers and shows exactly where a new method has to do better.

**Setup.** Classes arrive one at a time with 1, 2, 3, 5 or 10 examples each. No task hint. No stored examples. Datasets: Omniglot and CIFAR-100, plus MNIST followed by Fashion-MNIST as the cats-then-digits case.

**Feature sources, cheapest first:**

| | Feature source | One-off cost |
|---|---|---|
| A | Raw pixels | none |
| B | Random sparse expansion (FlyModel style) | none |
| C | Random convolutional network, never trained | none |
| D | Small network trained without labels on a few thousand unrelated images | minutes |
| E | Small network trained on 10-20 unrelated classes | minutes to an hour |
| F | Large pretrained network, frozen | paid by someone else |

**Learners on top (all without backprop):** nearest class mean, streaming LDA, FlyModel local rule, prototype neurons added on novelty.

**What gets measured:** accuracy against number of examples; accuracy on the first classes after all classes are learned; time and energy per new class; and the one-off cost of the features.

**What it tells you:**
- If D or E gets close to F, a large pretrained model is not needed, and the project's thesis has its first hard evidence.
- If only F works, the contribution has to be a cheaper way to build features, and the gap between E and F is the size of that job.
- Either way, the four learners are compared fairly for the first time at 2-3 examples.

Everything here fits in 4 GB: the learners use no gradients, and the largest model is run only forward.

**The step after that** would be making the features themselves grow: add units when a new class is poorly represented, freeze what older classes depend on, and recycle unused units. No one was found doing this from scratch without replay.

---

## 6. What is still unverified

- ART, Fuzzy ARTMAP, SOINN, Growing Neural Gas, and dynamic sparse training (DEEP R, SET, RigL) were not opened.
- Lake et al. 2015 (human-level one-shot learning) and the cognitive-science sources were not read in full.
- O-LoRA was not opened; MoE-Adapters, CL-LoRA and the memristor energy paper were read only in part.
- Progressive Networks was read from the abstract only; accuracy tables for CPG and Learn to Grow were not retrieved.
- No independent replication was found for any brain-style method listed here.
- Step 4 in Section 1 and the gap behind EXP-2 are "not found" claims. Someone should search again before either is stated in a paper.
- Feature source D (cheap label-free feature learning) is my suggestion; no paper on it was read in this pass.

---

## 7. Papers added to the library

PAPER 54 to 67, listed in `PAPERS_MANIFEST.md`. Not downloaded because no verified open PDF link was in hand: Dohare et al. (Nature 2024), PackNet, FOSTER, MEMO, Width Expansion.
