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
