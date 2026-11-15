# Literature log

Running notes from paper reading done by Sonnet subagents, started 2026-11-15 at the owner's
request ("keep on reading research papers" while Kaggle jobs run). Newest section last.

**How to read the tags.** The subagents read papers through a tool that returns the relevant
passages (method text, result tables), not whole PDFs.

- **passages**: method and result passages were opened and the numbers below were seen in them.
- **abstract**: only the abstract was seen. Do not rely on details.
- **secondhand**: the number comes from a table inside another paper that was opened.
- **not opened**: seen in search results only. Listed as a lead, nothing more.

Nothing here was read cover to cover. Numbers from different papers use different settings and
are not directly comparable with ours (CIFAR-100, 80 classes in 8 stages of 10, small 4-layer
conv net from scratch, no stored images).

---

## 1. Learning the feature extractor without backprop, or in one pass (2026-11-15)

Question: our classifier needs no backprop, but the feature extractor is trained with ordinary
backprop for 30 passes (1 pass gives 16.5% against 39.6%, EXP-5). Can features be learned
without end-to-end backprop, or from one pass?

### Features without end-to-end backprop

| Paper | Tag | What it does | Numbers seen |
|---|---|---|---|
| Journé et al., "Hebbian Deep Learning Without Feedback" (SoftHebb), arXiv:2209.11883 | passages | Conv layers trained by an unsupervised Hebbian rule with soft winner-take-all. No labels, no error signal. Layers trained one at a time, then frozen. Linear head trained afterwards | CIFAR-10: 80.3% (3 conv layers) against 84.0% for the same network with backprop. **One unsupervised epoch.** Random frozen weights in the same one-layer network: 63.9%. Needs very wide layers (96, 384, 1536, 6144 filters) and a special activation (80.3% against 70.7% with ReLU). No CIFAR-100 number seen |
| Nøkland and Eidnes, "Training Neural Networks with Local Error Signals", arXiv:1901.06656 | passages | Every layer has its own small loss; gradients stop at layer boundaries | CIFAR-100 error, VGG8B: 24.1% local against 26.2% global backprop. Fully backprop-free variant: 9.0% error on CIFAR-10 against 6.0%. Uses labels and 400 epochs |
| Belilovsky et al., "Greedy Layerwise Learning Can Scale to ImageNet", arXiv:1812.11446 | passages | Train one layer at a time with a small helper classifier, then freeze it | CIFAR-10: 88.3% against 89.7% end to end. 50 epochs per layer |
| Hinton, "The Forward-Forward Algorithm", arXiv:2212.13345 | abstract | Each layer has its own "goodness" objective | – |
| Scodellaro et al., Forward-Forward for CNNs, arXiv:2312.14924 | passages | Forward-Forward on a 6-layer CNN, 100 epochs | CIFAR-10: 68.6% against 85.4% backprop. CIFAR-100: 38.2% |
| Chen, DTG-FF, arXiv:2606.06539 | passages (single-author 2026 paper, not cross-checked) | Forward-Forward variant | CIFAR-100: 67.3% against 73.2% backprop, 400 epochs. Used more memory than backprop |

Secondhand figures on CIFAR-10 (from SoftHebb's and Belilovsky's tables): feedback alignment
63 to 72%, direct feedback alignment 70.5%, predictive coding about 60%, hard winner-take-all
Hebbian 64.6%, scattering features plus linear classifier 82.3%.

### Closed-form and one-pass classifiers

| Paper | Tag | What it does | Numbers seen |
|---|---|---|---|
| Prabhu et al., "Random Representations Outperform Online Continually Learned Representations" (RanDumb), arXiv:2402.08823 | passages | **No training of features at all.** A fixed random map of the raw pixels (25,000 random features), then a one-pass classifier using class means and one shared covariance. No stored images, no backprop | CIFAR-100, 10 steps of 10 classes, one pass: **28.6%**, against 24.6% for the best learned method without rehearsal in that table; joint one-pass reference 33.0%. Without the covariance step CIFAR-10 falls from 55.6% to 30.0% |
| Hayes and Kanan, "Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis", arXiv:1909.01520 | passages | One running mean per class plus one shared covariance, updated one example at a time, on top of a frozen feature extractor trained on the first classes | ImageNet, ResNet-18: 30 minutes against 31 to 62 hours for rehearsal methods |
| Zhuang et al., "ACIL: Analytic Class-Incremental Learning", arXiv:2205.14922 | passages | Recursive least squares head, one epoch per phase, same result as joint training of the head | Features come from backprop on half of CIFAR-100 (50 classes). Not from a small first task |
| McDonnell et al., "RanPAC", arXiv:2307.02251 | passages | Random projection plus ridge head | Needs a large pretrained model |

### Learning features from few passes (with backprop)

| Paper | Tag | What it does | Numbers seen |
|---|---|---|---|
| Tong et al., "EMP-SSL", arXiv:2304.03977 | passages | Label-free training with about 200 patch views per image | **One epoch**: 85.1% CIFAR-10, 58.5% CIFAR-100 (ResNet-18, linear head trained afterwards for 100 epochs). The one epoch hides about 200 times more computing per image |
| Jordan, "94% on CIFAR-10 in 3.29 Seconds on a Single GPU", arXiv:2404.00498 | passages | Many training tricks on a small network | 94% CIFAR-10 in about 10 epochs. A first layer set by a closed-form "patch whitening" (no backprop) more than halves the epochs needed |

### What this means for us (our reading, not from the papers)

- **A cheap floor we have not measured.** Random features plus a class-mean-and-covariance
  classifier got 28.6% on a CIFAR-100 split close to ours with no feature training. Our 1-pass
  backprop result is 16.5% and our frozen eyes 32.8%. Our memory cards use the class mean only;
  adding one shared covariance is one pass, no backprop and no stored pictures, and was the
  largest single ingredient in that paper.
- **Hebbian soft winner-take-all layers** are the one backprop-free, one-pass way of learning
  conv features with a real result (80.3% on CIFAR-10). They are local, so locking old units fits
  naturally. Open risks: they need very wide layers, there is no CIFAR-100 result, and nobody
  seems to have tested adding units to an already trained layer.
- **Backprop-free methods lose more as the number of classes grows** (2 to 4 points on CIFAR-10,
  about 6 on CIFAR-100 in the best case seen).
- **A closed-form whitened first layer** is a cheap, backprop-free piece that speeds training up.

Not opened (leads only): Coates and Ng "single-layer networks" (K-means features; secondhand
82% on CIFAR-10), PCANet arXiv:1404.3606, "Scaling Forward Gradient with Local Losses"
arXiv:2210.03310, FeCAM arXiv:2309.14062, DS-AL arXiv:2403.17503, F-OAL arXiv:2403.15751,
"Budgeted Training" arXiv:1905.04753.

---

## 2. Giving new units a view of old classes without old pictures; does freeze-and-grow limit features? (2026-11-15)

Question: in grow and lock, the units added at a stage see only that stage's 10 classes. What
is published on showing a network its old classes without storing old images, and on the
feature quality of networks that grow by frozen pieces?

### Ways to stand in for old data

| Paper | Tag | What it does | Numbers seen (CIFAR-100, no stored images unless stated) |
|---|---|---|---|
| Smith et al., "Always Be Dreaming" (ABD), arXiv:2106.09701 | passages | Makes synthetic old-class images by inverting the old network (a generator trained about 5,000 steps per task). Nothing stored but the model | Equal splits, last-step accuracy: 5 / 10 / 20 tasks = 43.9 / 33.7 / 20.0. The plain inversion recipe gets 18.8 / 10.9 / 5.7. Finding: the network learns "synthetic against real" more easily than "old class against new class", so the details of the loss matter a lot |
| Gao et al., "R-DFCIL", arXiv:2203.13104 | passages | Builds on ABD | 5 / 10 / 20 tasks = 50.5 / 42.4 / 30.8, small ResNet-32 |
| Zhang et al., "Deep Model Consolidation" (DMC), arXiv:1903.07864 | passages | Trains a separate model on the new classes, then merges old and new models using **unlabeled outside images** (not old data) | 8,000 to 32,000 outside images were enough. Natural images (textures, scenes) work; digits as outside images fail (about 2%) |
| Lee et al., "Overcoming Catastrophic Forgetting with Unlabeled Data in the Wild", arXiv:1903.12648 | passages | Distillation helped by sampled outside images | 65.0 to 68.1 average with outside data. **May also use a stored set of old images; not confirmed as exemplar-free** |
| Magistri et al., "Elastic Feature Consolidation" (EFC), arXiv:2402.03917 | passages | Backbone keeps training; stores a mean and covariance per class and replays samples from them at the last layer | Small first task, 10 steps: 43.6 last-step. About 30 directions per class keep 99% of the covariance |
| Petit et al., "FeTrIL", arXiv:2211.13131 | passages | Frozen extractor; makes fake old-class features by shifting new-class features | Good with a large first task (66.3), weak with a small one (34.9 in EFC's table) |
| Balaji et al., "Compressed Activation Replay", arXiv:2010.02418 | passages | Stores compressed activations of middle layers **together with real inputs** | Not exemplar-free. Shows that pooled middle-layer activations are usable |

**No paper was found that replays stored statistics of middle layers without stored inputs.**
Everything exemplar-free works at the last feature layer only.

### Networks that grow by frozen pieces

| Paper | Tag | What it says |
|---|---|---|
| Yan et al., "DER", arXiv:2103.16788 | passages | Freezes the old extractor, adds a new one per step. Uses stored images. Adds a special loss so the new extractor learns to tell "new classes" from "all old classes as one", for the stated purpose of learning "diverse and discriminative features". That loss needs old images |
| Wang et al., "FOSTER", arXiv:2204.04662 | passages | States the narrow-view problem directly: a new module trained only on what the old one gets wrong "cannot learn anything about old categories". Fix: make the new features classify all classes seen so far, which needs old images |
| Zhou et al., "MEMO", arXiv:2205.13218 | passages | Early layers of networks from different stages are similar; deep layers differ. **With fewer than about 20 first classes, freezing the early layers hurts; they should keep training.** With more first classes, freezing is better. (Uses stored images) |
| Rypeść et al., "SEED", arXiv:2401.10191 | passages | No stored images, from scratch, equal splits. Five full networks; one is fine-tuned per task; all vote through a mean and full covariance per class | Average accuracy 61.7 for 10 tasks against 46.3 for FeTrIL. With class means only: 54.1. So a **full covariance per class adds about 7.6 points** |

### What this means for us (our reading, not from the papers)

- **The narrow-view problem is known**, described in DER, FOSTER and FeTrIL. Every published fix
  uses old images. Without old images the options are: outside unlabeled images, images dreamed
  up from the old network, or statistics replay (so far only at the last layer).
- **Outside unlabeled images are the cheapest stand-in** with direct evidence, and a few
  thousand natural images are enough. It does mean bringing in a second dataset.
- **Dreamed images work but are fragile** and cost about 5,000 extra training steps per stage.
  Our old units are locked, so the "teacher" never changes; that may make it easier. Untested.
- **Our first group is 10 classes, under the roughly 20 below which freezing early layers was
  found harmful.** If EXP-6 points to locking, the first thing to try is a larger or longer
  first stage, or locking only the deep layers.
- **A covariance per class in the memory is worth points in two separate papers** (SEED +7.6,
  RanDumb about +25 on CIFAR-10). Our cards store means only. This is cheap, needs no backprop
  and no old pictures, and has not been tried here.
- Replaying stored statistics of middle layers so that new units can "see" old classes is, as
  far as this search went, not published without stored inputs. It would be a new experiment.

Not opened (leads only): DeepInversion (known only through ABD), Generative Feature Replay
arXiv:2004.09199, brain-inspired replay (van de Ven et al.), Latent Replay arXiv:1912.01100,
ACAE-REMIND arXiv:2105.08595, AdaGauss arXiv:2409.18265, "Adapt Your Teacher" arXiv:2308.09544,
Progressive Neural Networks, DEN.

---

## 3. Upgrading the memory from "average only" to "average plus spread" (2026-11-15)

Question: our memory card is one mean vector per class, compared by cosine. What is the
practical recipe for also storing a covariance (how the features of a class spread and vary
together), with no stored images and no backprop, when the feature vector grows over time?

| Paper | Tag | What matters for us |
|---|---|---|
| Goswami et al., "FeCAM", arXiv:2309.14062 | passages | Frozen backbone, Mahalanobis distance to class means. CIFAR-100 average accuracy: plain nearest-mean 64.8, one shared covariance 68.8, a covariance per class 70.9. **Raw covariance with no care is far worse than nearest-mean (29.7)**; it needs three steps: (1) square-root every feature (valid because features after ReLU are non-negative), (2) shrink the covariance by adding the average variance to the diagonal, (3) for per-class covariances, rescale each to a correlation matrix so classes learned at different times are comparable. Square root alone: 64.8 to 66.6. With a first group of 20 classes: nearest-mean 30.6 / 50.0 (last / average), FeCAM 48.1 / 62.3. The paper says it is "not apt when training from scratch, starting with small tasks", because its backbone is frozen |
| Hayes and Kanan, Deep Streaming LDA, arXiv:1909.01520 | passages | One running mean per class and **one shared covariance updated one example at a time**. Scores are linear: weight = inverse covariance times class mean. Shrinkage 0.0001 toward the identity. Updating the covariance online added only about 0.5 points over freezing it after the first classes |
| Rypeść et al., "AdaGauss", arXiv:2409.18265 | passages | From scratch, equal steps of 10 classes, backbone keeps training. Stores mean and full covariance per class in a 64-number space and moves both when features change, with a small learned map fitted on current data. CIFAR-100, 10 steps, last / average: nearest-mean 36.4 / 54.0, diagonal covariance 41.1 / 56.3, **full covariance 46.1 / 60.2**. Explains the bias: early classes have lower-rank covariances, which inflates their distances |
| Rypeść et al., "SEED", arXiv:2401.10191 | passages | Full covariance per class 61.7 against class means only 54.1 and diagonal 53.5. Removing the last ReLU (so features look more bell-shaped) is worth 3.9 points |
| Nguyen et al., "EPEM: Efficient Parameter Estimation for Multiple Class Monotone Missing Data", arXiv:2009.11360 | passages | Exact one-pass formulas for a shared covariance when all samples have the first block of variables and only some have the later blocks. **This is exactly our pattern**: old units exist for every class, new units only for newer classes. The missing blocks are filled through the regression of new on old, which is what R7 already fits |
| Chen et al., "Shrinkage Algorithms for MMSE Covariance Estimation" (OAS), arXiv:0907.4698 | passages | A closed-form choice of how much to shrink a covariance toward a scaled identity when samples are few |

### What this means for us (our reading)

- A **single shared covariance** is the safe first step: every class is scored with the same
  ruler, it needs little shrinkage, and the missing blocks for grown units have an exact
  formula. Published gain over plain nearest-mean: about 4 to 6 points on a strong frozen
  backbone, larger (12 or more) where the backbone is weak, which is closer to our case.
- **Square-rooting the features** is free and helped by about 2 points on its own in one paper.
- A covariance **per class** is worth a further 2 to 8 points in the papers, but with 500
  pictures and 576 numbers per class it needs heavy shrinkage and the rescaling step, and old
  classes would need their missing blocks filled.
- None of these papers tests a feature vector that grows. That part would be new here.

Not opened: Ledoit and Wolf 2004; Mensink et al. (nearest class mean metric learning);
EFC arXiv:2402.03917 for its covariance handling.

---

## 4. Brain-inspired mechanisms with measured results (2026-11-15)

Question: which brain-inspired mechanisms have real numbers on class-by-class image learning,
and how do they compare with ordinary methods?

| Paper | Tag | Mechanism | What was measured |
|---|---|---|---|
| Shen, Dasgupta, Navlakha, "Algorithmic insights on continual learning from fruit flies", arXiv:2107.07617 | passages | Fixed random expansion to a very wide, sparse code (only a few units active), then a readout where only the links from active units to the correct class are strengthened. One pass, no backprop in that layer, no task hint | CIFAR-100, 25 steps of 4 classes: at least 15 points above the other continual methods tested, 13 below offline. **Inputs were features from an ImageNet-pretrained ResNet-18**, so this measures the memory, not the eyes. Ablation at the end: sparse code plus this readout 0.41; sparse code with an ordinary readout 0.21 |
| Iyer et al., "Avoiding Catastrophe: Active Dendrites...", arXiv:2201.00042 | passages | Sparse activity (5% active) plus "dendrites" that switch sub-networks on by context | Permuted MNIST only, an easy case. No photos |
| Masse, Grant, Freedman, context-dependent gating, arXiv:1802.01569 | passages | 80% of units switched off per task | Needs the task to be named at test time. Cannot be used in our setting as is |
| van de Ven, Siegelmann, Tolias, "Brain-inspired replay", Nature Communications 2020 (PMC7426273) | passages | Replays hidden-layer activity made by a generative model inside the network; nothing stored | CIFAR-100, 10 steps of 10, no task hint: 0.344 average accuracy (with conv layers pre-trained on CIFAR-10). The authors call it the best they know without stored data, and still far below joint training |
| Tadros et al., "Sleep-like unsupervised replay", Nature Communications 2022 (PMC9755223) | passages | After each task, an offline "sleep" phase with noise input and a local Hebbian rule | CIFAR-10 in 5 steps (pretrained features): 19.0 without, 44.6 with sleep, 72.4 joint |
| Thapa, Ororbia, Desell, self-organising maps with synthetic replay, arXiv:2508.21240 | passages | Map units keep running statistics and generate replay samples | CIFAR-100 from scratch, 10 steps of 10: 14.4, against 48.1 offline. Strong numbers only with a pretrained encoder |
| Han et al., SCA-SNN, arXiv:2411.05802 | passages | Spiking network that grows new neurons per task and reuses similar old ones | CIFAR-100, 10 steps, from scratch: 57.1 without task hint, **but with 2,000 stored pictures**. Energy figure is an estimate from operation counts |
| Hajizada et al., CLP-SNN on Intel Loihi 2, arXiv:2511.01553 | passages | Prototype learner with a local rule; new prototypes allocated for novel inputs | **Only measured energy found**: 0.05 mJ per learning step on the neuromorphic chip against 1.9 to 3.9 mJ for a class-mean learner on a small GPU board. Pretrained frozen backbone; the energy excludes the backbone |
| Ahmad and Scheinkman, "How Can We Be So Dense?", arXiv:1903.11257 | passages | Sparse networks | Tests noise robustness only. Not evidence about forgetting |
| Magistri et al., EFC, arXiv:2402.03917 | passages | Ordinary method, used as the reference | CIFAR-100, ResNet-18 from scratch, no stored images, 10 equal steps: 43.6 last-step accuracy; older methods 30 to 35 |

### What this means for us (our reading)

- **No brain-inspired method that was opened beats the best ordinary exemplar-free method** on
  CIFAR-100 from scratch without a task hint. Where they look strong, the features came from a
  pretrained network or pictures were stored.
- The mechanisms with real support all sit in the **memory**, not the eyes: sparse wide codes
  with a simple readout (fruit fly), prototype learners with local rules. That is the half of
  our system that already works. EXP-2 tested our own version of the fruit-fly learner and it
  lost to plain class means in 18 of 18 cases.
- For reference, our 39.6% on 80 classes is in the same range as the published 43.6% for the
  best ordinary method on 100 classes in 10 steps, with a network about 30 times smaller. The
  settings differ, so this is context, not a comparison.
- Energy has been measured in one paper only, on special hardware, for the memory step only.

Not opened: Draelos et al. "Neurogenesis deep learning"; Dynamically Expandable Networks;
Growing When Required networks; self-organising incremental neural networks; FearNet;
differentiable plasticity; HebbCL arXiv:2207.04874.
