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
