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
  best ordinary method on 100 classes in 10 steps, with a much smaller network. The
  settings differ, so this is context, not a comparison.
- Energy has been measured in one paper only, on special hardware, for the memory step only.

Not opened: Draelos et al. "Neurogenesis deep learning"; Dynamically Expandable Networks;
Growing When Required networks; self-organising incremental neural networks; FearNet;
differentiable plasticity; HebbCL arXiv:2207.04874.

---

## 5. Background for later: what a "memory card" could be for a language model (2026-11-15)

No language experiment is planned until the owner asks. This is background for the open
question in the journals: a language model predicts the next word instead of picking a class,
so what plays the role of one card per thing learned?

| Paper | Tag | What is stored as a "card" | Numbers seen |
|---|---|---|---|
| Khandelwal et al., "Nearest Neighbor Language Models" (kNN-LM), arXiv:1911.00172 | passages | One card per word of text: key = the model's description of the text so far, value = the word that came next. Added by one forward pass, no training | 247M-parameter model, Wikitext-103: perplexity 18.65 to 16.12 (lower is better) with 103M cards. A model trained on 100M words plus a 3-billion-word memory (13.73) beat a model trained on all 3 billion words (15.17) |
| He, Neubig, Berg-Kirkpatrick, "Efficient Nearest Neighbor Language Models", arXiv:2109.04212 | passages | Same | Lookup makes it about 14 times slower (277 against 3,847 words per second); tricks bring that to about 2 times. New domain (law text): base model 106.6, **memory only 12.6, fine-tuning 8.6**. **Replacing groups of cards by their average made it worse than no memory at all**; only merging near-identical cards worked |
| Alon et al., "RetoMaton", arXiv:2201.12431 | passages | Same, with links between consecutive cards | Saves 81% of lookups at equal quality. Memory on top of fine-tuning: 8.61 to 7.10, so the two add up |
| Grave et al., continuous cache, arXiv:1612.04426; unbounded cache, arXiv:1711.02604 | passages | Recent (or all past) hidden states with the word that followed | Wikitext-103, LSTM: 48.7 to 40.8. Far domain with a tiny model: 687 to 237 |
| Min et al., "Nonparametric Masked Language Modeling", arXiv:2212.01349 | passages | One card per phrase of a text collection; no fixed vocabulary | Updating the collection updates the answers (2.9 to 17.5 on changed facts) with unchanged facts kept. Cannot generate text. Very large index (about 70 GB memory for Wikipedia) |
| Lan et al., "Copy is All You Need", arXiv:2307.06962 | passages | Phrases copied from a collection while generating | New domain with no training scored better than a fine-tuned model on a text-quality measure (28.1 against 26.9) |
| Hartvigsen et al., "Aging with GRACE", arXiv:2211.11031 | passages | **Closest to our cards.** At one layer: a key vector, a radius, and a stored replacement output. An input landing inside the radius uses the card; everything else is untouched. New cards are added, old ones never change | 1,000 corrections on a 60M model needed 137 cards; old corrections kept at 0.96, original ability 0.69 against 0.56 for plain fine-tuning. The stored output is trained with a small backprop step |
| Mitchell et al., "Memory-Based Model Editing at Scale" (SERAC), arXiv:2206.06520 | passages | The correction kept as plain text, plus a small trained model deciding when it applies | Edit success 0.99 with almost no damage elsewhere (0.009). About 3 KB per card |
| Eldan and Li, "TinyStories", arXiv:2305.07759 | passages | – | Models of 1 to 35 million parameters trained on simple stories write fluent, consistent text; all trained on one GPU within 30 hours. Under 10M parameters is still fluent |
| Warstadt et al., BabyLM findings, arXiv:2504.08165 | passages | – | Training on only 10M or 100M words; winners were not text generators |
| Luo et al., forgetting in large language models, arXiv:2308.08747 | passages | – | Sequential tuning on 5 tasks: for example 36.2% to 26.1% on one knowledge test; larger models forgot more |

### What this means for us (our reading)

- The language version of our memory already exists and works: store "what the text looked
  like" together with "what came next", and look it up. New text is learned in one pass with no
  training and nothing old is overwritten. Its price is a very large memory and slower answers.
- **One average card per word does not work for language** (averaging cards made things worse).
  The working form is one card per situation, merged only when situations are nearly the same.
- Memory alone did not match fine-tuning on a new domain (12.6 against 8.6), but memory plus
  fine-tuning beat both. So here too, the memory helps and the eyes still matter.
- A laptop-sized test exists when the owner wants it: a TinyStories-size model, then a second
  kind of text added by memory against added by fine-tuning, measuring what is forgotten.
  This is the language twin of EXP-3.

Not opened: ROME, MEMIT, Progressive Prompts, AdapterFusion arXiv:2005.00247 (abstract only).

---

## 6. Eyes without backprop from picture patches (2026-11-15)

Question: is there a feature extractor that is learned without backprop, ideally in one pass,
and that can grow by adding units without changing old ones?

| Paper | Tag | Recipe | Numbers seen |
|---|---|---|---|
| Coates, Lee, Ng, "An Analysis of Single-Layer Networks in Unsupervised Feature Learning", AISTATS 2011 | passages (whole paper scraped) | Cut random 6x6 patches from the pictures. Normalise each patch, whiten the patch set, run K-means to get K typical patches ("centroids"). To describe a picture: compare every 6x6 window with every centroid ("triangle" rule: how much closer than average), sum the answers over the four quarters of the picture. No labels, no backprop. Linear classifier on top | **CIFAR-10: 77.9% with 1,600 centroids, 79.6% with 4,000.** By number of centroids (whitened): 100: 65.8, 400: 72.8, 1,600: 77.0. **Whitening is worth 7 to 8 points** (1,600 centroids: 77.0 whitened, 69.0 not). Step between windows matters: every pixel 77.0, every 2nd 73.5, every 4th 67.2. No CIFAR-100 result |
| Coates and Ng, "Learning Feature Representations with K-means", 2012 (book chapter) | passages | Practical settings: add 10 to the patch variance before dividing; whitening constant 0.01 to 0.1; about 10 K-means rounds; simpler rule max(0, match minus a threshold) | CIFAR-10: one layer 78.3%, one wider layer 80.6%, three layers 82.0%. A second layer helped only with a careful way of choosing its inputs |
| Thiry, Arbel, Belilovsky, Oyallon, "The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods", arXiv:2101.07528 | passages | **No learning of the dictionary at all**: it is made of randomly picked whitened 6x6 training patches. Only a linear classifier is trained | **CIFAR-10: 82.5% with 2,048 patches, 85.6% with 10,000, 86.7% with 60,000.** Training the dictionary end to end gave no gain over picking patches at random. Replacing patches by pure noise: 8 points worse. Using patches from a different dataset (ImageNet): only 0.4 points worse |
| Chan et al., "PCANet", arXiv:1404.3606 | passages | Filters from PCA of patches, two stages | CIFAR-10: 77.1% |
| Oyallon and Mallat, "Deep Roto-Translation Scattering", arXiv:1412.8659 | passages | Fixed wavelet filters, nothing learned | CIFAR-10: 82.3%. CIFAR-100: 56.8% (table rows inferred by the reader; check before citing) |
| Jordan, airbench, arXiv:2404.00498 | passages | A frozen whitening first layer inside an ordinary network | More than halves training time to a given accuracy |
| Chen et al., "Sparse Manifold Transform", arXiv:2209.15261 | abstract | One-layer label-free method | Abstract: 81.1% CIFAR-10, 53.2% CIFAR-100 with nearest neighbours |

**Gaps found:** no CIFAR-100 result for the K-means or random-patch recipes; no paper that
grows a patch dictionary class by class with old patches kept fixed; no test of whether
whitening computed on a few classes is good for other classes (nearest evidence: patches from a
different dataset cost only 0.4 points).

### What this means for us (our reading)

- This is the strongest lead so far for **eyes that need no backprop and one pass**. A fixed
  set of whitened picture patches gives 78 to 86% on 10-class photos, in the range of small
  backprop-trained networks, with nothing trained but the final classifier.
- It fits grow and lock naturally: **adding patches adds units; old patches never change**, so
  nothing old can be overwritten, and an old memory card only lacks the new units, which is
  the situation R7 already handles.
- EXP-2 tried "free" features (random expansion, random conv net) and they failed on photos
  (6.5 to 6.9% at 3 pictures per class). Those had no whitening, no large patch dictionary and
  no triangle rule, which are exactly the ingredients these papers say matter most.
- Unknowns that only an experiment can settle: accuracy on 80 to 100 classes with class-mean
  cards instead of a trained linear classifier, and whether patches taken from the first 10
  classes serve later classes.

---

## 7. Making backprop-free patch eyes stronger (2026-11-16)

Question: once fixed patch eyes are in place (EXP-7), what are the known ways to improve them
without backprop?

| Paper | Tag | What it gives | Numbers seen (CIFAR-10 unless stated) |
|---|---|---|---|
| Coates and Ng, "Selecting Receptive Fields in Deep Networks", NIPS 2011 | passages (whole body) | **A second and third layer without backprop.** Average the first layer's answers over 3x3 blocks. Group first-layer detectors that tend to be active together (correlation of their squared answers; 32 groups of 200). Inside each group, pick new detectors by K-means on the group's answers | 1 layer 78.3; 2 layers 81.2; 3 layers 82.0. **A second layer wired at random or to everything gives nothing** (77.4 to 77.6). With only 400 labelled pictures per class: 64.6, then 69.2, then 70.7 |
| Coates and Ng, "The Importance of Encoding Versus Training...", ICML 2011 | passages | How a detector answers matters more than how the detectors were found. Splitting each answer into a positive and a negative part "always improved" | Random patches with a threshold of 0.25: 79.1. K-means: 79.4. Pure noise as detectors: 73.2. 6,000 detectors: 81.5 |
| Jia, Huang, Darrell, "Beyond Spatial Pyramids" (workshop version of the CVPR 2012 paper) | passages | Finer regions than four quarters | With 200 detectors: quarters 70.2, a 4x4 grid 72.2 (average) or 73.0 (max), three-level pyramid 74.8. **CIFAR-100**, 1,600 detectors, 6,400 features, linear classifier: about 54.5 to 54.9 |
| Thiry et al., arXiv:2101.07528 | passages | More detectors | 512: 79, 1,024: 81, 2,048: 83, 4,096: 84 (read from figure data), 10,000: 85.6. Their way of shrinking the description is trained by backprop; no backprop-free alternative is given |
| Prabhu et al., RanDumb, arXiv:2402.08823 | passages | Mirror every picture when filling the tables | CIFAR-100: 26.9 without, 28.6 with mirrored copies. Removing the shared table: 16.6 points worse |
| Mairal, supervised convolutional kernel networks, arXiv:1605.06265 | passages | Layers built by K-means plus one matrix calculation each | Two label-free layers: 14.2% error (85.8%), details sparse |
| Boureau, Ponce, LeCun, "A Theoretical Analysis of Feature Pooling", ICML 2010 | passages | Max pooling suits very sparse answers; average suits dense ones | No CIFAR numbers |
| Guo and Hastie, regularized discriminant analysis, 2005 | passages (section 2.1) | Standard covariance shrinkage formulas | – |

### What this means for us (our reading)

Cheapest improvements, in the order we would try them after EXP-7:

1. **Mirrored copies** when filling the tables: free, about +1.7 points in the one paper that
   measured it on CIFAR-100.
2. **Positive and negative parts** of each detector's answer as separate numbers.
3. **Finer regions** (3x3 or a pyramid) at the cost of a longer description.
4. **A second layer** built by grouping detectors that fire together. This is the largest
   documented gain (+3 to +5 points, and more when labelled pictures are few), but only if the
   grouping step is done; random wiring gives nothing. Units stay independent if the grouping
   and its whitening are fixed after the first fit (our suggestion, not from the paper).

A published reference point for one layer of such features on all 100 CIFAR-100 classes with a
trained linear classifier is about 55%.

Not opened: Friedman 1989 (scanned, unreadable); Mairal et al. 2014; Recht et al. random-feature
baseline (only as cited by Thiry).

---

## 8. Learning a new class from a few pictures on top of fixed eyes (2026-11-16)

Question: EXP-7's no-backprop system learns well from 500 pictures per class but badly from 3
(17.0% against 29.7% for grow and lock). What closed-form tricks exist for few pictures on top
of fixed features, and what do the papers say about the limit?

| Paper | Tag | Recipe | Numbers seen |
|---|---|---|---|
| Wang et al., "SimpleShot", arXiv:1911.04623 | passages (whole text) | Subtract the average of all base-class features, scale each description to length 1, then nearest class average | CIFAR-100, 20 new classes, 5 at a time, 1 / 5 pictures: plain 36.4 / 52.7; length-1 scaling 38.5 / 53.3; with the average subtracted 40.1 / 53.6. Gains of 3 to 6 points at 1 picture, small at 5 |
| Yang, Liu, Xu, "Free Lunch for Few-shot Learning: Distribution Calibration", arXiv:2101.06395 | passages | (1) Square root of every (non-negative) feature. (2) Blend the new class's average with its 2 nearest base classes and borrow their spread | miniImageNet, 1 picture: nothing 56.4; **square root only 64.3**; both 68.6. Only small positive powers help (power 2: 43.5) |
| Zhang et al., MetaQDA, arXiv:2101.02833 | passages | Class averages and spreads with a strong pull toward a simple prior | 5 pictures: shared-table classifier with no pull 51.8, **with the pull 81.1**. States that un-smoothed tables are "very poor" with few pictures |
| Bateni et al., Simple CNAPS, arXiv:1912.03432 | passages | Blend each class's spread with the overall spread and the identity; weight = shots / (shots + 1) | The blending matters most when features were not adapted to the task |
| Tian et al., "Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?", arXiv:2003.11539 | passages | A small trained classifier on fixed features | Length-1 scaling adds about 2 points at 1 picture. Title and findings: the quality of the features decides |
| Mensink et al., "Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost", TPAMI 2013 (HAL preprint) | passages | Hand-made (not learned) features; a projection learned on 800 classes; new classes added by their average only | New classes with 100+ pictures each: close to seen classes (40.0 against 36.5 top-5 error). **With 1 picture: nearly useless** (about 91 error, read from a chart). Closed-form projections (whitening 46.1, Fisher 50.5) are better than none (68.0) but worse than the trained projection (37.0) |
| Chen et al., "A Closer Look at Few-shot Classification", arXiv:1904.04232 | passages | Cosine-style classifier reduces spread within a class | Differences between methods shrink as the feature network gets better |

### What this means for us (our reading)

- **Every paper puts few-picture ability in the learned features**, not in the step on top.
  The one paper with hand-made features found them fine with 100 pictures per class and
  nearly useless with 1. Our 17% against 26 to 38% for trained eyes fits that picture.
- Cheap things that may recover a few points, in order: subtract the base average and scale
  to length 1 after the square root; pull the tables hard toward the identity (EXP-7 already
  saw accuracy rise with more smoothing); blend a new class's average with its nearest known
  classes; project onto the directions that separate the 80 known classes before comparing.
- No paper read shows non-learned eyes matching learned ones from a few pictures. So the
  honest expectation is: patch eyes can be made less bad at few pictures, not good.
- This suggests a **two-part design** as the thing to test later: class-blind eyes for
  learning from plenty of pictures without forgetting, plus something learned for few-picture
  generalisation.

Not opened: Prototypical Networks with Mahalanobis; adaptive-subspace and transductive
few-shot methods; the ECCV 2012 version of Mensink et al.

---

## 9. New classes from a few pictures competing with classes from many (2026-11-17)

Question: EXP-9's new classes (3 pictures) score 33.5% among themselves but 7.1% among all 100
classes, because classes learned from 500 pictures take the answers. What closed-form ways exist
to balance the two? One Sonnet subagent, all tags "passages". Every title and arXiv ID was
confirmed on the opened page (the subagent itself caught a wrong guessed ID for CEC; the right
one is below).

| Paper | Tag | Recipe | Numbers seen |
|---|---|---|---|
| Chao, Changpinyo, Gong, Sha, "An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild", ECCV 2016, arXiv:1605.04253 | passages | **Calibrated stacking**: answer = the class with the highest score after subtracting one constant from every seen (old) class. The constant is chosen on classes held out from the old ones, pretending they are new. Sweeping it gives a curve of old against new accuracy | One area-under-curve value seen without its dataset; no per-setting numbers seen |
| Qi, Brown, Lowe, "Low-Shot Learning with Imprinted Weights", arXiv:1712.07136 | passages | A new class's weight is the length-1 average of its length-1 descriptions (our "card by angle") | CUB-200, 100 old + 100 new, no fine-tuning, new-class accuracy at 1/2/5/10/20 pictures: 21.3/28.7/39.5/45.8/49.3 |
| Gidaris, Komodakis, "Dynamic Few-Shot Visual Learning without Forgetting", arXiv:1804.09458 | passages | Cosine classifier with a learned scale; new weights from averaged descriptions (plus a learned part) | Mini-ImageNet, 5 pictures: cosine classifier 72.8 new only, 70.7 old only, **51.9 both together**. The drop when both compete is the same problem as ours |
| Bateni et al., "Improved Few-Shot Visual Classification" (Simple CNAPS), arXiv:1912.03432 | passages | Covariance per class blended with the shared one by weight n/(n+1), n = pictures of that class | Episodic test only (73.8 in-domain); no old-vs-new competition |
| Hayes, Kanan, "Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis", arXiv:1909.01520 | passages | Our memory: running class means plus one shared covariance on fixed features, no old pictures | Updating the shared table as classes arrive gave only about 0.5 points on ImageNet over freezing it |
| Zhang et al., "Few-Shot Incremental Learning with Continually Evolved Classifiers" (CEC), arXiv:2104.03047 | passages | Frozen backbone, class-mean classifier, plus a trained module that adjusts all class means | Abstract: a pretrained backbone with class means "can beat state-of-the-art methods". Reports accuracy over all classes per session, no old/new split |
| Zhou et al., "Forward Compatible Few-Shot Class-Incremental Learning" (FACT), arXiv:2203.06953 | passages (no tables seen) | Training on old classes reserves room for future ones | none seen |
| Tao et al., "Few-Shot Class-Incremental Learning" (TOPIC), arXiv:2004.10956 | abstract | Defines the few-shot class-incremental benchmark | none |
| Goswami et al., FeCAM, arXiv:2309.14062 | passages | Distances made comparable across classes (shrinkage, power transform, correlation normalisation) | CIFAR-100, 50 old + 50 new, trained on the old 50, old/new/all: class means 76.7/26.5/51.6; shared covariance 75.9/38.4/57.2; per-class covariance 77.2/47.0/62.1 |
| McDonnell et al., RanPAC, arXiv:2307.02251 | passages | Fixed random projection, then a ridge classifier from accumulated tables; setting by cross-validation | Pretrained ViT; not comparable |
| Xian et al., "Zero-Shot Learning – A Comprehensive Evaluation of the Good, the Bad and the Ugly", arXiv:1707.00600 | passages | **Harmonic mean** H = 2·old·new/(old+new) as the single score when old and new compete | E.g. ALE on CUB: new 23.7, old 62.8, H 34.4. Old classes "act as distractors" |

### What this means for us (our reading)

- **The problem is known and expected**, not a bug of ours (Gidaris: 72.8 / 70.7 alone, 51.9
  together). The standard single score is the harmonic mean of old and new accuracy.
- **Calibrated stacking is the standard fix**: one constant subtracted from the old classes'
  scores, chosen by pretending some old classes are new. Our EXP-9 rehearsal already does that
  pretending without leakage (its tables never saw the pretend-new classes).
- **Score by certainty** (the subagent's own derivation, not from a paper read): a class mean from
  n pictures is uncertain, so a picture of that class lies on average farther from it, by a factor
  of about (1 + 1/n) in squared distance. Dividing each class's squared distance by (1 + 1/n)
  needs no tuned setting. It needs comparison by distance in the learned view, not by angle.
- **Adding new classes to the view: weak evidence** (only the 0.5-point result for the shared
  table, on many-picture classes). Untested for few-picture classes.
- Learned modules (CEC, Gidaris's attention, FACT) need training on old data; not candidates.

Not opened: Bayesian few-shot papers with a full posterior-predictive (Student-t) class model;
incremental LDA with few-picture classes; Hou et al. 2019 (rebalancing); Ren et al. 2019.

---

## 10. A switch between known and new classes (2026-11-19)

Question: EXP-11's new classes score 36.2% among themselves but 17% among all 100 at the
old-class limit; a perfect "old or new?" switch would give a harmonic mean of about 45 instead
of 26 (arithmetic). Which closed-form switches exist? One Sonnet subagent. **Caveat on the tags:**
direct PDF reading failed, so the subagent read arXiv abstract pages and ar5iv HTML pages through
a fetch tool that returns a summary made by a small model. Numbers below are as relayed by that
summary, not re-read by us.

| Paper | Tag | Recipe | Numbers relayed |
|---|---|---|---|
| Socher, Ganjoo, Sridhar, Bastani, Manning, Ng, "Zero-Shot Learning Through Cross-Modal Transfer", NIPS 2013, arXiv:1301.3666 | passages (relayed) | Switch first: novelty = low likelihood under a mixture of Gaussians, one per known class; threshold swept, no rule given | No combined accuracy; at 70% known-class accuracy, new classes roughly 15-30% (CIFAR-10, chance 10%) |
| Atzmon, Chechik, "Adaptive Confidence Smoothing for Generalized Zero-Shot Learning" (COSMO), CVPR 2019, arXiv:1812.09903 | passages (relayed) | Soft switch: a small logistic regression on the sorted top scores of both experts, trained on known classes pretending to be new | Same experts, handicap (calibrated stacking) vs COSMO, new/old/H: AWA 45.4/68.2/54.5 vs 52.8/80.0/63.6; SUN 41.7/25.9/31.9 vs 44.9/37.7/41.0; CUB 43.1/53.7/47.9 vs 44.4/57.8/50.2 |
| Ren et al., "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection", 2021, arXiv:2106.09022 | passages (relayed) | Relative distance: Mahalanobis to nearest class minus Mahalanobis to one background Gaussian fitted to all training features | CIFAR-100 (known) vs CIFAR-10 (unknown), no pretraining, AUROC: plain 74.9, relative 81.0, max-softmax 80.1 |
| Kamoi, Kobayashi, "Why is the Mahalanobis Distance Effective for Anomaly Detection?", 2020, arXiv:2003.00402 | abstract + passages (relayed) | Partial Mahalanobis over chosen principal components | CIFAR-10 vs SVHN AUROC: full 93.9; components 10-512 94.1; components 1-9 88.1. The low-variance components carry the novelty signal but little class information (20.5% accuracy alone) |
| Wang et al., "ViM: Out-Of-Distribution with Virtual-logit Matching", CVPR 2022, arXiv:2203.10807 | abstract + passages (relayed) | Novelty from the norm of the feature outside the top principal subspace, turned into an extra "virtual" class score | ImageNet-1K average AUROC: 90.9 (BiT), Mahalanobis 86.6; not a CIFAR split |
| Lee, Lee, Lee, Shin, "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018, arXiv:1807.03888 | abstract | Class-conditional Gaussians with a tied covariance; Mahalanobis to the nearest class as novelty score; abstract says it extends to class-incremental learning without retraining | none seen |

Not opened (leads only): MetaQDA (arXiv:2101.02833); a Student-t predictive for few-picture
classes; Atzmon et al. arXiv:2004.02235.

### What this means for us (our reading)

- A **switch first, then name within the group** is an established design (Socher 2013; COSMO).
  COSMO's trained version beats the handicap by 2-9 harmonic-mean points; ours must be closed-form.
- The papers agree the novelty signal sits in directions that matter little for naming known
  classes (Kamoi; ViM). Our learned view keeps only the naming directions, so the part **outside
  the view** is a natural candidate.
- **Relative distance** (Ren) helps most on "near" unknowns, which is our case. With a background
  built from our own tables (within-class table plus the spread of the class averages), the part
  outside the view cancels exactly, so the relative signal lives inside the view (our derivation,
  not from the paper).
- Telling near unknowns from known classes is hard even with trained features (AUROC about 81 for
  CIFAR-100 vs CIFAR-10). Expect a real switch to land well under the perfect one.
