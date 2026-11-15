# Deep Research: Learning Like a Brain — Continual Learning × Energy-Efficient AI
> Generated 2026-09-19 | Depth: deep | Sources: 53 (4 retracted-flagged/corrected during verification)

## TL;DR

Nobody has yet built AI that learns like a human, but the field has split your exact ambition into three attackable problems: **continual learning** (don't forget), **brain-inspired algorithms** (learn locally, not by backprop), and **neuromorphic/efficient computing** (do it at watts, not megawatts). Each is individually mature enough to survey but their **intersection is nearly empty** — almost no papers measure the energy cost of continual learning, and almost no neuromorphic work runs continual protocols. For an independent researcher on free cloud compute, that measurement gap — plus rigorous replication of brain-inspired learners at CIFAR/Tiny-ImageNet scale — is the most defensible path to a first publication.

## Executive Summary

Your intuition is correct and well-supported by the literature: modern AI's "train once, offline, at ever-larger scale" paradigm is both environmentally costly and scientifically mismatched to biological learning. GPT-3's training is estimated at ~1,287 MWh and 552 tCO2e [2]; a Transformer tuned with neural architecture search emits roughly five times a US car's lifetime CO2 [1]; meanwhile the human brain runs lifelong learning on ~20 W [8]. Inference now rivals training in lifetime energy cost, so the problem is getting worse, not better [10].

The research landscape relevant to your goal has consolidated into six areas. **(1) Energy framing** is settled science — Green AI arguments are mainstream [1][2]. **(2) Continual learning (CL)** has three canonical method families — regularization (EWC [3]), replay (DER [7]), parameter isolation — and a well-documented evaluation crisis [5][39][40]. **(3) Brain-inspired learning algorithms** (predictive coding, Forward-Forward, PEPITA, feedback alignment, e-prop) have matured from proofs-of-concept to matching backprop at CIFAR scale [12][13][51], but a rigorous 2018 study established a "scalability wall" — these rules degrade significantly on ImageNet-class problems [50], and nothing since has demolished that wall. **(4) Neuromorphic hardware** (TrueNorth, Loihi/Loihi 2, Hala Point) delivers real, verified order-of-magnitude energy advantages — but only on sparse, event-driven workloads [31][33][52], and a sharp 2024 critique shows SNN energy claims collapse unless spike rates stay below ~5.7% [27]. **(5) The intersection** — few-shot CL, online/task-free CL, on-device training under 256KB [36], CL for foundation models, SNN-based CL — is young, sparsely populated, and has a demonstrated reproducibility problem (a 2025 SNN sleep-wake CL paper was withdrawn for data errors [38]). **(6) Open problems**: evaluation methodology, the stability gap [40], and above all the **near-total absence of energy accounting in CL research** [27][53].

The actionable conclusion: you cannot compete on scale, but this field canonically rewards diagnostic and methodological contributions done on small benchmarks — DER [7] and the stability-gap paper [40] both became standard citations with modest compute. A paper that brings rigorous energy/spike accounting to continual-learning evaluation, or that rigorously stress-tests brain-inspired learners where informal critique currently outnumbers rigorous critique, is feasible on Colab/Kaggle and would be a genuine contribution.

---

## 1. Status Quo [Confidence: High]

### 1.1 The energy problem is real, quantified, and growing

The energy and carbon costs of modern deep learning are large, measurable, and growing, and this motivates the search for fundamentally more efficient learning paradigms. Strubell, Ganesh and McCallum estimated that training a Transformer with neural architecture search emitted roughly 626,155 lbs of CO2e — about five times the lifetime emissions of an average American car — and used the result to argue for "Green AI": research that reports computational cost alongside accuracy and treats efficiency as a first-class objective [1]. Patterson et al. refined the picture for frontier-scale training, estimating GPT-3 at roughly 1,287 MWh and 552 tCO2e gross, while emphasizing that four levers — model, machine, mechanization, and map (datacenter location) — can jointly reduce energy by ~100× and carbon by ~1000×, and that sparsely-activated networks consume less than a tenth of the energy of dense equivalents [2]. More recent accounting shows the problem has shifted downstream: LLM inference, not just training, now rivals or exceeds training in lifetime energy, water, and carbon cost [10].

Against these figures stands the canonical biological benchmark: the human brain sustains perception, language, and lifelong skill acquisition on roughly 20 W, a figure widely cited as the efficiency target for artificial systems [8][9]. Popular framings push far beyond this — one claims brains beat AI by 225,000× in energy efficiency [11] — but such multipliers are unverified and should be treated as rhetorical context rather than evidence. The defensible synthesis is threefold: deep learning's energy trajectory is unsustainable at scale [1][10]; the brain is an existence proof of vastly more efficient intelligence [8]; and while engineering levers can capture large savings before any algorithmic novelty is needed [2], closing the remaining orders-of-magnitude gap plausibly requires the algorithmic and hardware innovations surveyed below. (A caveat for your own writing: the 20 W-brain vs megawatt-GPT comparison contrasts *training* energy with *running* power on different tasks — it is directionally honest but dimensionally loose. Use it as motivation, not as a quantitative claim.)

### 1.2 Continual learning: the established method families

Continual learning research has consolidated around a small number of method families: regularization-based approaches, replay approaches, and architecture/parameter-isolation approaches, with optimization- and representation-based refinements in more recent taxonomies [4][6].

- **Regularization.** The foundational method, Elastic Weight Consolidation (EWC), uses a Fisher-information quadratic penalty to selectively slow learning on weights important for prior tasks — explicitly inspired by synaptic consolidation in the brain [3]. Synaptic Intelligence (SI) and Learning without Forgetting (LwF) belong to the same family [4]. Known weakness: regularization-only methods degrade sharply when task identity is unavailable at test time [6].
- **Replay.** Replay methods interleave stored or generated past data during new-task training — the direct engineering descendant of complementary learning systems theory (fast hippocampus + slow neocortex + replay) from McClelland et al. 1995 [44]. Dark Experience Replay (DER/DER++), which stores logits rather than raw exemplars, remains a strong, cheap, reproducible baseline that any new method must beat [7]. Naive replay incurs storage and privacy overheads [4].
- **Architecture / parameter isolation.** Progressive Neural Networks, PackNet, and HAT allocate or freeze per-task capacity [4]. These control forgetting well but scale parameters with tasks.

A second consensus concerns evaluation: outcomes depend strongly on the incremental scenario. van de Ven, Tuytelaars and Tolias distinguish task-, domain-, and class-incremental learning; method rankings shift across scenarios, class-incremental learning is the hardest, and the absence of a common framework makes cross-paper comparison difficult [5].

### 1.3 Neuromorphic hardware: real, but conditional, efficiency

Neuromorphic chips deliver order-of-magnitude energy advantages on workloads matched to their sparse, event-driven computational model. IBM's TrueNorth delivered ~1M neurons and 256M synapses at 65 mW (~26 pJ per synaptic event) but no on-chip learning [31]. Intel's Loihi added on-chip learning; Davies et al. reported up to ~1000× energy-delay product improvements over CPUs on LASSO/optimization workloads, with gains confined to sparse event-driven regimes and early-stage toolchain limitations [52]. Independent benchmarking on keyword spotting showed TrueNorth and Loihi achieving orders-of-magnitude better energy per inference than CPU/GPU at comparable accuracy on streaming audio [33]. Loihi 2 scaled the platform; Intel's Hala Point system (1,152 chips) reaches 1.15B neurons and 128B synapses at 2.6 kW with a vendor-claimed >15 TOPS/W [29]. A 370M-parameter MatMul-free LLM running on Loihi 2 uses 2× less energy than transformer-based LLMs on an edge GPU per its abstract, with the paper body reporting ~405 mJ/token and at least 14× lower energy than an H100 — inference only, not learning [28].

Training SNNs today is done mostly with surrogate gradients — smoothing the spike nonlinearity so gradient methods apply [16], with recent theory clarifying that gradient bias is tolerable at low spike rates [18]. e-prop achieves online learning in recurrent SNNs approaching BPTT performance without storing activation history, but remains below ImageNet scale [17].

## 2. Emerging Trends [Confidence: High]

### 2.1 Brain-inspired alternatives to backprop are closing the gap — at small scale

The shared motivation is the set of biological implausibilities of backpropagation — weight transport, separate forward/backward phases, and non-local error signals — catalogued canonically by Whittington and Bogacz [15]. Feedback alignment showed that fixed random feedback weights suffice for effective error-driven learning, dissolving the weight-transport problem in principle [14].

Progress since then is real but bounded:

- **Predictive coding (PC)** has advanced furthest. Qi et al. (2025) combine time-dependent precision weighting with forward-only updates; their abstract claims performance comparable to backpropagation on deep models such as ResNets, and the authors report 93.27% on CIFAR-10 (VGG10), 72.02% on CIFAR-100, and 53.04% on Tiny-ImageNet — while conceding reduced biological plausibility and slower training than backprop, with ImageNet-scale left to future work [12]. Sakana AI's PC-ALM trains 1000-layer MLPs via an augmented-Lagrangian reformulation of PC inference, but only in MLP/tabular domains, not CNNs or LLMs [19].
- **Forward-only learning.** Hinton's Forward-Forward algorithm replaces the backward pass with two forward passes (positive/negative data) and per-layer goodness functions; Hinton himself describes it as working "well enough on a few small problems to be worth further investigation," with CIFAR-10 MLPs roughly 1–2% below backprop [13]. Follow-ups such as DeeperForward improve it but remain at CIFAR scale and slightly below backprop [20]. PEPITA replaces the backward pass with a second forward pass carrying error-modulated input, validated at MNIST/CIFAR scale [51].
- **Online adaptation as learning.** Test-time training and test-time adaptation — per-instance plasticity with no labels — produce large gains on corrupted benchmarks [22], and now have a mature taxonomy including continual TTA [21]. This is arguably the closest deployed analogue to "learning continuously at runtime."
- **Associative memory formalized.** Modern Hopfield networks rigorously bridge Hebbian associative memory and transformer attention [24], and LeCun's JEPA/world-model program argues for energy-based self-supervised architectures over generative scaling — an influential but contested position [23].

### 2.2 The intersection is where the field is moving

Continual learning is being actively transplanted into brain-inspired and resource-constrained settings, but the intersection is young:

- **Few-shot class-incremental learning (FSCIL)** — your "show the baby one dog" scenario formalized — has crystallized around the TOPIC framework and miniImageNet/CIFAR-100/CUB benchmarks; its own survey names closing the human–machine efficiency gap as an open problem [35].
- **Online and task-free CL** is advancing via dynamic cluster expansion (single-pass, grows capacity as new distributions arrive) [48] and Online-LoRA, which performs task-free single-pass online CL on vision transformers with no growing rehearsal buffer [42].
- **Meta-learning for CL**: OML/MRCL meta-learns sparse, interference-resistant representations so new tasks require only small updates; it spawned ANML (neuromodulated meta-learning) [43].
- **On-device CL became practically real** with MCUNetV3, which trains under 256KB SRAM / 1MB Flash — under 1/1000 the memory of PyTorch/TensorFlow while matching accuracy on Visual Wake Words — with PyTorch-based, Colab-runnable reference code [36][46].
- **CL for foundation models** is now a surveyed subfield: continual pretraining, instruction tuning, alignment, PEFT-for-CL (LoRA/adapters/weight merging) and MoE approaches [41]; a 2025 position paper argues the future of CL likely resides in decentralized ecosystems of specialized, composable models rather than monolithic retraining [37].
- **Complementary learning systems theory** — the 1995 neuroscience root of replay [44] — is being actively operationalized with modern architectures as of 2025 [45].
- **SNN-specific continual learning** is emerging through similarity-based context gating [49] and energy-aware spike budgeting, which treats spike counts as a feedback-controlled resource across five class-incremental tasks, cutting spikes by up to 47% with accuracy gains, and adding +17.45 points on DVS-Gesture via budget relaxation [53].

## 3. Critical Assessment [Confidence: High]

### 3.1 The scalability wall for brain-inspired learning

The anchor result for this entire area is negative, and it is rigorous: Bartunov et al. (NeurIPS 2018, with Hinton and Lillicrap as co-authors) systematically assessed biologically motivated algorithms — feedback alignment, target propagation, sign-symmetry — and found they match backprop on MNIST but fall **significantly** short on CIFAR-10 and ImageNet, with the gap widening in locally connected (convolutional) networks [50]. Informal stress tests corroborate accuracy collapse relative to backprop as task complexity grows [26]. Nothing published since has demolished this wall: the best 2025 predictive-coding results top out at Tiny-ImageNet [12], Forward-Forward variants remain CIFAR-scale [20], and community critique notes that FF's negative-data generation is task-specific with no evidence of large-scale competitiveness [25]. The honest read is that biological plausibility is currently *traded away* for performance — Qi et al. explicitly admit their forward updates reduce plausibility [12]. Anyone claiming their forward-only method "replaces backprop" without ImageNet-class evidence should be read with suspicion.

### 3.2 SNN energy claims are conditional — sometimes dramatically so

The sharpest contradiction in this survey concerns SNN efficiency itself. Yan, Bai and Wong (2024) argue that typical SNN-vs-ANN comparisons ignore memory access and data-movement overheads and often pit unoptimized ANNs against SNNs on specialized chips. Their abstract-level conclusion: SNNs beat matched quantized ANNs **only when the average spike rate falls below ~5.7% at T=5 timesteps**; the body reports some published SNNs consuming 27–41× *more* energy than ANN equivalents [27]. This directly qualifies the headline neuromorphic numbers [28][33][52]: the advantage is real but conditional on sparsity and workload structure — a caveat vendor materials like Hala Point's press release understandably understate [29]. Kudithipudi et al.'s Nature perspective diagnoses the systemic cause: without shared benchmarks and software stacks, efficiency claims across chips are not comparable and the field stalls [34]. The ecosystem is fragmented, toolchains immature, hardware lab-only (Loihi 1 is end-of-life [30]), and GPU inertia overwhelming [32].

### 3.3 Continual learning has an evaluation crisis

The critique of CL evaluation practice is itself now consensus. Averaged end-of-sequence accuracy masks the retention–acquisition tradeoff [39]; the "stability gap" — a transient accuracy dip on previously learned tasks while a new task is being learned — is invisible to standard end-of-training evaluation [40]; and the three incremental scenarios are routinely conflated, inflating results when task identity is implicitly assumed at test time [5]. Most methods are validated on small, artificially split benchmarks (permuted/split MNIST, CIFAR) whose relevance to real lifelong learning is questioned in the surveys themselves [4][35].

### 3.4 The reproducibility warning you must internalize

In 2025, SESLR — an online continual learning paper combining SNNs with a sleep-wake replay cycle, reporting ~30% accuracy improvements at one-third memory — was **withdrawn by its own authors due to errors in the data** [38]. This matters to you twice: first, never build on results from this young intersection without independent replication; second, it reveals that the SNN × CL niche is so new that its early results are fragile — which is precisely where a careful, rigorous worker can add value.

### 3.5 What a skeptic would say about your whole premise

Steel-manning the opposition: (a) "Learning without training" partially exists already — in-context learning lets frozen LLMs adapt from a few examples; what's missing is *persistence and consolidation* of that learning, which reframes your problem as memory/consolidation research, not training elimination. (b) Efficiency gains from the 4M levers (sparsity, better hardware, better datacenters) are available *today* without any brain-inspired novelty [2]. (c) Biological plausibility may be a distraction: airplanes don't flap. The brain's 20 W budget includes running a body and 86 billion neurons' worth of functions AI doesn't need. (d) Even brains have a massive "training" phase — evolution plus years of childhood; the few-shot infant learns on top of a heavily pretrained substrate. Your report and eventual paper should anticipate all four objections; each maps to a real section of the literature [2][24][50].

### 3.6 The gap the evidence actually supports

Synthesizing across areas reveals a genuine **measurement gap at the intersection**: CL papers report accuracy and forgetting but almost never energy; neuromorphic papers report energy but almost never continual protocols. Energy-aware spike budgeting [53] is an early, lonely exception, and the SNN × CL literature rests on a handful of papers [49][53] plus one retraction [38]. This is an inference from the corpus rather than a documented consensus — treat it as a hypothesis to re-verify with a fresh literature search before committing — but it is the strongest signal in this entire survey for where an independent researcher's paper can be novel.

## 4. Action Plan

Your constraint (free cloud compute only) is less limiting than it feels: this field's canonical baselines (DER [7]) and its most cited diagnostics (stability gap [40]) were produced at small scale. Ranked by feasibility × novelty:

- [ ] **Project A (recommended): Energy-accounted continual learning evaluation.** Take a standard class-incremental suite (Split CIFAR-10/100, per van de Ven's scenarios [5]) using Avalanche [47], run 4–6 representative methods (EWC [3], DER [7], an online method [48], a PEFT/LoRA method [42], an SNN method in simulation via snnTorch [16][49][53]), and measure joules/spikes/memory-per-task alongside accuracy and the stability gap [40]. Fills the measurement gap (§3.6) directly; entirely Colab-feasible; produces a methods-comparison paper the field demonstrably needs [34].
- [ ] **Project B: Stability-gap stress-test of the new wave.** Apply De Lange et al.'s worst-case continual evaluation [40] to method families it hasn't covered — Online-LoRA [42], spike-budgeted SNNs [53], MCUNetV3-style on-device training [36]. Diagnostic papers in this field become standard citations.
- [ ] **Project C: Rigorous replication of brain-inspired learners.** Reproduce Qi et al. PC [12], PEPITA [51], and Forward-Forward [13] at CIFAR-10/100 and Tiny-ImageNet with controlled seeds/compute, including a training-energy comparison against backprop — the subfield has a proven reproducibility problem [38] and informal critiques [25][26] outnumber rigorous ones beyond Bartunov [50].
- [ ] **Project D (later, ambitious): Sleep-phase consolidation done right.** The withdrawn SESLR [38] shows both the appetite for and the risk of CLS-inspired sleep/replay in SNNs [44][45]. A careful, small-scale, fully open replication-style contribution here is high-novelty — but only after A/B/C gives you the evaluation machinery to make it trustworthy.
- [ ] **Foundations first (weeks 1–4):** Read the backbone surveys [4][39][35]; reproduce DER on Split CIFAR-10 in Avalanche [7][47]; reproduce MCUNetV3's tiny-training example [46] to internalize on-device constraints.
- [ ] **Writing habit:** Report energy and memory alongside accuracy in every experiment you ever run [1][2] — it costs nothing and makes your work citable by the Green AI community.
- [ ] **Community positioning:** Track ContinualAI's curated lists and Avalanche [47], the Awesome-Incremental-Learning repo [47], and the LLM-CL survey companion repo [41]; cite the retraction [38] as motivation for rigor, not as a result.
- [ ] **Before committing to Project A's novelty claim:** run a fresh literature check specifically for "energy consumption continual learning benchmark" post-2025 papers — the gap is an inference, not a documented consensus (§3.6).

## 5. Open Questions & Caveats

1. **Is the measurement gap real?** It is inferred from the corpus (few sources straddle both literatures — mainly [27][53]), not stated as consensus anywhere. A dedicated search may surface exceptions post-dating this report.
2. **Body-reported numbers.** Three headline figures come from paper bodies, not abstracts, and were verified only at retrieval level: PC's 93.27% CIFAR-10 [12], Loihi 2's 405 mJ/token and 14×-vs-H100 [28], and the 27–41× SNN-over-ANN figure [27]. Pull the full texts and cite table/section before using them in your own paper.
3. **Davies et al. 2021** [52] was snippet-verified only (paywall); the ~1000× energy-delay claim is widely cited but confirm against the Nature full text.
4. **The brain comparison is loose.** 20 W running power vs one-shot training MWh compare different things (§1.1); the 225,000× popular multiplier [11] is unverified.
5. **Vendor claims.** Hala Point's TOPS/W figures are marketing [29]; independent apples-to-apples cross-chip benchmarks are exactly what the field lacks [34].
6. **Scope exclusions.** Embodied/RL continual learning, spiking-hardware access (Loihi is research-gated), and German-language literature were out of scope. Few-shot/meta-learning was included only where it intersects CL (FSCIL, OML/ANML).
7. **Contradiction left unresolved:** whether predictive coding's scalability gains [12][19] represent genuine progress past the Bartunov wall [50] or a replay of the same trade-off at one level deeper — the evidence supports both readings.

## Methodology

Depth: **deep**. Phases: clarify (2 rounds) → scope (6 key areas) → Wave 1 retrieval (4 parallel subagents, 40+ sources) → Wave 2 gap-fill (1 subagent, 4 targeted gaps: Bartunov, PEPITA, Davies, spike-budgeting) → triangulation → citation spot-check (1 verification subagent, 10 highest-impact claims) → critique → synthesis (1 synthesis subagent) → writing. Total: 7 subagents, 53 unique sources (deduplicated; Wang et al. TPAMI 2024 appeared twice and was merged).

**Citation corrections from spot-check (Phase 3.1):** 6/10 claims SUPPORTED as-is. Claim on predictive coding numbers downgraded to "authors report" phrasing [12]. Loihi 2 LLM claim split into abstract-level (2× vs edge GPU) and body-reported (405 mJ/token, 14× vs H100) [28]. Yan et al. claim corrected to abstract-level "<5.7% spike rate at T=5" [27]. **SESLR [38] found WITHDRAWN by authors (data errors) — removed as supporting evidence and retained only as a cautionary finding.** Outline changes vs. plan: added §3.4 (retraction) and §3.5 (skeptic steel-man, incl. in-context-learning reframe) — under the 50% structural-change limit. No retrieval-wave failures; degradation: none. Recency: trend sources 2019–2026; foundational sources (McClelland 1995, CLS; Lillicrap 2016; Kirkpatrick 2017; Bartunov 2018; Neftci 2019; Merolla 2015) marked `[foundational]` and retained regardless of age.

## Bibliography

[1] Strubell, Ganesh & McCallum — Energy and Policy Considerations for Deep Learning in NLP — https://arxiv.org/abs/1906.02243 (ACL 2019) — Accessed 2026-09-19 — Tier: 1 [foundational]
[2] Patterson et al. — Carbon Emissions and Large Neural Network Training — https://arxiv.org/abs/2104.10350 — 2021 — Accessed 2026-09-19 — Tier: 1
[3] Kirkpatrick et al. — Overcoming catastrophic forgetting in neural networks (EWC) — https://www.pnas.org/doi/10.1073/pnas.1611835114 (PNAS 2017) — Accessed 2026-09-19 — Tier: 1 [foundational]
[4] Wang, Zhang, Su & Zhu — A Comprehensive Survey of Continual Learning: Theory, Method and Application — https://arxiv.org/abs/2302.00487 (IEEE TPAMI 2024) — Accessed 2026-09-19 — Tier: 1
[5] van de Ven, Tuytelaars & Tolias — Three types of incremental learning — https://www.nature.com/articles/s42256-022-00568-3 (Nature Machine Intelligence 2022) — Accessed 2026-09-19 — Tier: 1
[6] De Lange et al. — A Continual Learning Survey: Defying Forgetting in Classification Tasks — https://arxiv.org/abs/1909.08383 (IEEE TPAMI 2021) — Accessed 2026-09-19 — Tier: 1
[7] Buzzega et al. — Dark Experience for General Continual Learning: a Strong, Simple Baseline — https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf (NeurIPS 2020) — Accessed 2026-09-19 — Tier: 1
[8] Human Brain Project — Learning from the brain to make AI more energy-efficient — https://www.humanbrainproject.eu/en/follow-hbp/news/2023/09/04/learning-brain-make-ai-more-energy-efficient/ — 2023 — Accessed 2026-09-19 — Tier: 2
[9] Texas A&M University Stories — AI that uses less energy by mimicking the human brain — https://stories.tamu.edu/news/2025/03/25/artificial-intelligence-that-uses-less-energy-by-mimicking-the-human-brain/ — 2025 — Accessed 2026-09-19 — Tier: 2
[10] Jegham et al. — How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference — https://arxiv.org/abs/2505.09598 — 2025 — Accessed 2026-09-19 — Tier: 1
[11] Write a Catalyst (Medium) — Human Brains Beat AI by 225,000 Times in Energy Efficiency — https://medium.com/write-a-catalyst/human-brains-beat-ai-by-225-000-times-in-energy-efficiency-762b9327e8ad — 2025 — Accessed 2026-09-19 — Tier: 3
[12] Qi, Forasassi, Lukasiewicz & Salvatori — Towards the Training of Deeper Predictive Coding Neural Networks — https://arxiv.org/abs/2506.23800 — 2025 — Accessed 2026-09-19 — Tier: 1
[13] Hinton — The Forward-Forward Algorithm: Some Preliminary Investigations — https://arxiv.org/abs/2212.13345 — 2022 — Accessed 2026-09-19 — Tier: 1
[14] Lillicrap et al. — Random synaptic feedback weights support error backpropagation for deep learning — https://www.nature.com/articles/ncomms13276 (Nature Communications 2016) — Accessed 2026-09-19 — Tier: 1 [foundational]
[15] Whittington & Bogacz — Theories of Error Back-Propagation in the Brain — https://pubmed.ncbi.nlm.nih.gov/30704969/ (Trends in Cognitive Sciences 2019) — Accessed 2026-09-19 — Tier: 1 [foundational]
[16] Neftci, Mostafa & Zenke — Surrogate Gradient Learning in Spiking Neural Networks — https://ieeexplore.ieee.org/document/8891809 (IEEE Signal Processing Magazine 2019) — Accessed 2026-09-19 — Tier: 1 [foundational]
[17] Bellec et al. — A solution to the learning dilemma for recurrent networks of spiking neurons (e-prop) — https://www.nature.com/articles/s41467-020-17236-y (Nature Communications 2020) — Accessed 2026-09-19 — Tier: 1
[18] Zenke/Neftci groups — Elucidating the Theoretical Underpinnings of Surrogate Gradient Learning in SNNs — https://direct.mit.edu/neco/article/37/5/886/128506 (Neural Computation 2025) — Accessed 2026-09-19 — Tier: 1
[19] Sakana AI — PC-ALM: training 1000-layer networks with predictive coding — https://pub.sakana.ai/pc-alm/ — 2025 — Accessed 2026-09-19 — Tier: 2
[20] Papachristodoulou et al. — DeeperForward: Enhanced Forward-Forward Training — https://openreview.net/forum?id=kOYnXVQCtA — Accessed 2026-09-19 — Tier: 1
[21] Liang, He & Tan — A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts — https://arxiv.org/abs/2303.15361 (IJCV 2024/25) — Accessed 2026-09-19 — Tier: 1
[22] Sun, Wang, Efros & Darrell — Test-Time Training with Self-Supervision — https://xiaolonw.github.io/papers/ttt.pdf (ICML 2020) — Accessed 2026-09-19 — Tier: 1
[23] LeCun — A Path Towards Autonomous Machine Intelligence — https://openreview.net/pdf?id=BZ5a1r-kVsf — 2022 — Accessed 2026-09-19 — Tier: 2
[24] Ramsauer et al. — Hopfield Networks is All You Need — https://arxiv.org/abs/2008.02217 (ICLR 2021) — Accessed 2026-09-19 — Tier: 1
[25] r/MachineLearning — Discussion: The Forward-Forward Algorithm — https://www.reddit.com/r/MachineLearning/comments/zdkpgb/ — Accessed 2026-09-19 — Tier: 3
[26] Bedi et al. — Testing the Limits of Biologically-Plausible Backpropagation (Stanford course report) — https://cs.stanford.edu/~rbedi/files/appphys293_report.pdf — Accessed 2026-09-19 — Tier: 3
[27] Yan, Bai & Wong — Reconsidering the Energy Efficiency of Spiking Neural Networks — https://arxiv.org/abs/2409.08290 — 2024 — Accessed 2026-09-19 — Tier: 1
[28] Abreu, Shrestha, Zhu & Eshraghian — Neuromorphic Principles for Efficient Large Language Models on Loihi 2 — https://arxiv.org/abs/2503.18002 — 2025 — Accessed 2026-09-19 — Tier: 1
[29] Intel Newsroom — Intel Builds World's Largest Neuromorphic System (Hala Point) — https://www.intc.com/news-events/press-releases/detail/1691/ — 2024 — Accessed 2026-09-19 — Tier: 2
[30] Open Neuromorphic — A Look at Loihi — https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-intel/ — Accessed 2026-09-19 — Tier: 2
[31] Merolla et al. — TrueNorth: Design and Tool Flow of a 65 mW 1 Million Neuron Programmable Neurosynaptic Chip — https://research.ibm.com/publications/truenorth-design-and-tool-flow-of-a-65-mw-1-million-neuron-programmable-neurosynaptic-chip (IEEE TCAD 2015; Science 2014) — Accessed 2026-09-19 — Tier: 1 [foundational]
[32] humanunsupervised.com — Neuromorphic Computing 2025: Current SotA — https://humanunsupervised.com/papers/neuromorphic_landscape.html — Accessed 2026-09-19 — Tier: 3
[33] Blouw, Choo, Hunsberger & Eliasmith — Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware — https://arxiv.org/abs/1812.01739 (NICE 2019) — Accessed 2026-09-19 — Tier: 1
[34] Kudithipudi et al. — Brain-Inspired Computing Needs a Master Plan — https://doi.org/10.1038/s41586-022-04991-4 (Nature 2022) — Accessed 2026-09-19 — Tier: 1
[35] Zhang, Hu, Liu, Silvén & Pietikäinen — Few-shot Class-incremental Learning: A Survey — https://arxiv.org/abs/2308.06764 (Neural Networks 2024) — Accessed 2026-09-19 — Tier: 1
[36] Lin et al. — On-Device Training Under 256KB Memory (MCUNetV3) — https://arxiv.org/abs/2206.15472 (NeurIPS 2022) — Accessed 2026-09-19 — Tier: 1
[37] Bell et al. — The Future of Continual Learning in the Era of Foundation Models — https://arxiv.org/abs/2506.03320 — 2025 — Accessed 2026-09-19 — Tier: 1
[38] Lin et al. — Online Continual Learning via Spiking Neural Networks with Sleep-Wake Cycle (SESLR) — https://arxiv.org/abs/2507.02901 — 2025 — Accessed 2026-09-19 — Tier: 1 — **WITHDRAWN by authors (data errors); cautionary reference only**
[39] van de Ven, Soures & Kudithipudi — Continual Learning and Catastrophic Forgetting — https://arxiv.org/abs/2403.05175 — 2024 — Accessed 2026-09-19 — Tier: 1
[40] De Lange, van de Ven & Tuytelaars — Continual Evaluation for Lifelong Learning: Identifying the Stability Gap — https://openreview.net/forum?id=Zy350cRstc6 (ICLR 2023) — Accessed 2026-09-19 — Tier: 1
[41] Yang et al. — Recent Advances of Foundation Language Models-based Continual Learning: A Survey — https://dl.acm.org/doi/10.1145/3705725 (ACM TIST; arXiv 2405.18653) — Accessed 2026-09-19 — Tier: 1
[42] Wei et al. — Online-LoRA: Task-free Online Continual Learning via Low-Rank Adaptation — https://arxiv.org/abs/2411.05663 — 2024 — Accessed 2026-09-19 — Tier: 1
[43] Javed & White — Meta-Learning Representations for Continual Learning (OML/MRCL) — https://arxiv.org/abs/1905.12588 (NeurIPS 2019; ANML: Beaulieu et al., ICLR 2020, https://openreview.net/pdf?id=IaUh7CSD3k) — Accessed 2026-09-19 — Tier: 1
[44] McClelland, McNaughton & O'Reilly — Why There Are Complementary Learning Systems in the Hippocampus and Neocortex — https://pubmed.ncbi.nlm.nih.gov/7624455/ (Psychological Review 1995) — Accessed 2026-09-19 — Tier: 1 [foundational]
[45] A Neural Network Model of Complementary Learning Systems — https://arxiv.org/abs/2507.11393 — 2025 — Accessed 2026-09-19 — Tier: 1
[46] MIT HAN Lab — MCUNetV3 blog post + tiny-training code — https://hanlab.mit.edu/blog/mcunetv3-blogpost (code: github.com/mit-han-lab/tiny-training) — 2022 — Accessed 2026-09-19 — Tier: 2
[47] ContinualAI — continual-learning-papers curated list (+ Avalanche framework; also github.com/xialeiliu/Awesome-Incremental-Learning) — https://github.com/ContinualAI/continual-learning-papers — Accessed 2026-09-19 — Tier: 2
[48] Ye & Bors — Online Task-Free Continual Generative and Discriminative Learning via Dynamic Cluster Expansion — https://openaccess.thecvf.com/content/CVPR2024/html/Ye_Online_Task-Free_Continual_Generative_and_Discriminative_Learning_via_Dynamic_Cluster_CVPR_2024_paper.html (CVPR 2024) — Accessed 2026-09-19 — Tier: 1
[49] Similarity-based Context Aware Continual Learning for Spiking Neural Networks — https://www.sciencedirect.com/science/article/abs/pii/S0893608024009663 (Neural Networks 2024) — Accessed 2026-09-19 — Tier: 1
[50] Bartunov et al. — Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures — https://arxiv.org/abs/1807.04587 (NeurIPS 2018) — Accessed 2026-09-19 — Tier: 1 [foundational]
[51] Dellaferrera & Kreiman — Error-driven Input Modulation (PEPITA) — https://proceedings.mlr.press/v162/dellaferrera22a.html (ICML 2022) — Accessed 2026-09-19 — Tier: 1
[52] Davies et al. — Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook — https://doi.org/10.1038/s42256-021-00331-y (Nature Machine Intelligence 2021) — Accessed 2026-09-19 — Tier: 1 [snippet-level verification]
[53] Meem, Nadid & Mia — Energy-Aware Spike Budgeting for Continual Learning in Spiking Neural Networks — https://iopscience.iop.org/article/10.1088/2634-4386/ae8627 (IOP Neuromorphic Computing and Engineering 2026; arXiv:2602.12236) — Accessed 2026-09-19 — Tier: 1

## Source Extracts

### [1] Strubell et al. 2019 (ACL)
- **Summary:** Foundational Green AI paper; quantifies financial/environmental cost of training/tuning NLP models; urges reporting compute cost and efficiency-first research.
- **Key quotes:** "626,155 lbs" CO2e for Transformer + NAS (≈5× US car lifetime emissions).
- **Source type:** academic — **Tier:** 1

### [2] Patterson et al. 2021
- **Summary:** GPT-3 ≈1,287 MWh / 552 tCO2e gross; "4M" levers cut energy ~100×, CO2 ~1000×; sparse DNNs <1/10th energy of dense.
- **Key quotes:** "Large but sparsely activated DNNs can consume <1/10th the energy of large, dense DNNs without sacrificing accuracy."
- **Source type:** academic/industry lab — **Tier:** 1

### [3] Kirkpatrick et al. 2017 (PNAS)
- **Summary:** EWC: Fisher-information quadratic penalty protecting weights important to old tasks; validated on sequential MNIST and Atari.
- **Key quotes:** "selectively slowing down learning on the weights important for those tasks."
- **Source type:** academic — **Tier:** 1

### [4] Wang et al. 2024 (TPAMI)
- **Summary:** Canonical CL taxonomy (regularization/replay/optimization/representation/architecture); metrics; scenarios; open limitations.
- **Key quotes:** naive approaches suffer "huge computational and storage overheads, as well as potential privacy issues."
- **Source type:** academic — **Tier:** 1

### [5] van de Ven et al. 2022 (Nature MI)
- **Summary:** Task/domain/class-incremental scenarios formalized; method rankings change by scenario; class-incremental hardest; lack of common framework hinders comparison.
- **Source type:** academic — **Tier:** 1

### [6] De Lange et al. 2021 (TPAMI)
- **Summary:** Replay/regularization/parameter-isolation families for classification; regularization-only degrades without task IDs at test.
- **Source type:** academic — **Tier:** 1

### [7] Buzzega et al. 2020 (NeurIPS)
- **Summary:** DER stores logits ("dark knowledge") in replay buffer; strong, cheap baseline for general CL.
- **Source type:** academic — **Tier:** 1

### [8] Human Brain Project 2023
- **Summary:** Frames the ~20 W brain, learning continuously for a lifetime, as the efficiency benchmark; surveys neuromorphic/spiking routes.
- **Source type:** research-infrastructure news — **Tier:** 2

### [9] Texas A&M Stories 2025
- **Summary:** Brain-inspired materials/oscillatory neurons for energy-efficient AI; dated anchor that the motivation is active in 2025.
- **Source type:** university news — **Tier:** 2

### [10] Jegham et al. 2025
- **Summary:** Benchmarks per-query energy/water/carbon of GPT-4o-class inference; deployment costs now rival training.
- **Source type:** academic — **Tier:** 1

### [11] Medium/Write a Catalyst 2025
- **Summary:** Popular "225,000× efficiency" framing; unverified multiplier — public framing only.
- **Source type:** blog — **Tier:** 3

### [12] Qi et al. 2025
- **Summary:** PC training recipe (time-dependent precision weighting, forward updates, BN freezing); abstract: comparable to backprop on deep models such as ResNets; body: 93.27% CIFAR-10 (VGG10), 72.02% CIFAR-100, 53.04% Tiny-ImageNet; admits reduced plausibility, slower than BP.
- **Key quotes:** "achieve performances comparable to those of backprop."
- **Source type:** academic — **Tier:** 1

### [13] Hinton 2022
- **Summary:** Forward-Forward: two forward passes, per-layer goodness, no activity storage; small problems only.
- **Key quotes:** "works well enough on a few small problems to be worth further investigation."
- **Source type:** academic — **Tier:** 1

### [14] Lillicrap et al. 2016 (Nat. Commun.)
- **Summary:** Fixed random feedback suffices for error-driven learning; dissolves weight-transport problem.
- **Source type:** academic — **Tier:** 1

### [15] Whittington & Bogacz 2019 (TiCS)
- **Summary:** Canonical review of how backprop-like computation could arise in cortex; PC approximations, target propagation, feedback alignment; honest evidential assessment.
- **Source type:** academic — **Tier:** 1

### [16] Neftci et al. 2019 (IEEE SPM)
- **Summary:** Standard reference for surrogate-gradient SNN training; SNNs' value is neuromorphic energy efficiency; BPTT unrolling memory-hungry.
- **Source type:** academic — **Tier:** 1

### [17] Bellec et al. 2020 (Nat. Commun.)
- **Summary:** e-prop: eligibility traces + top-down signals enable online learning in recurrent SNNs near BPTT without storing history; below ImageNet scale.
- **Source type:** academic — **Tier:** 1

### [18] Zenke/Neftci groups 2025 (Neural Computation)
- **Summary:** Theory of when/why surrogate gradients work; gradient bias tolerable at low spike rates.
- **Source type:** academic — **Tier:** 1

### [19] Sakana AI 2025
- **Summary:** PC-ALM trains 1000-layer MLPs stably; MLP/tabular only, not CNNs/LLMs.
- **Source type:** research lab report — **Tier:** 2

### [20] Papachristodoulou et al.
- **Summary:** DeeperForward improves FF (channel-wise goodness, better negative samples); still CIFAR-scale, slightly below BP.
- **Source type:** academic (OpenReview) — **Tier:** 1

### [21] Liang et al. 2024/25 (IJCV)
- **Summary:** TTA taxonomy (TTT, BN adaptation, TENT, pseudo-labeling, continual TTA); failure modes: error accumulation, catastrophic drift.
- **Source type:** academic — **Tier:** 1

### [22] Sun et al. 2020 (ICML)
- **Summary:** Original TTT: self-supervised auxiliary task at test time; large gains on CIFAR-10-C/ImageNet-C without labels.
- **Source type:** academic — **Tier:** 1

### [23] LeCun 2022
- **Summary:** JEPA/world-model position paper; argues against LLM-style scaling; contested claims.
- **Source type:** position paper — **Tier:** 2

### [24] Ramsauer et al. 2021 (ICLR)
- **Summary:** Modern Hopfield networks equivalent to transformer attention; rigorous Hebbian-memory-to-deep-learning bridge.
- **Source type:** academic — **Tier:** 1

### [25] r/MachineLearning FF thread
- **Summary:** Community critique: FF negative-data generation task-specific; no large-scale competitiveness evidence.
- **Source type:** community — **Tier:** 3

### [26] Bedi et al. (Stanford report)
- **Summary:** Stress-test of FA/target-prop/local rules; accuracy collapse vs BP as complexity grows; informal, corroborated by [50].
- **Source type:** course report — **Tier:** 3

### [27] Yan et al. 2024
- **Summary:** Strongest SNN-energy critique: comparisons ignore memory/data movement; SNNs beat matched quantized ANNs only below ~5.7% spike rate (T=5); body reports some SNNs 27–41× worse.
- **Key quotes:** "this oversight can lead to a misleading perception of efficiency."
- **Source type:** academic — **Tier:** 1

### [28] Abreu et al. 2025
- **Summary:** 370M-param MatMul-free LLM on Loihi 2; abstract: 2× less energy than edge-GPU transformers; body: ~405 mJ/token, ≥14× vs H100; inference only.
- **Key quotes:** "up to 3× higher throughput with 2× less energy, compared to transformer-based LLMs on an edge GPU."
- **Source type:** academic — **Tier:** 1

### [29] Intel Newsroom 2024 (Hala Point)
- **Summary:** 1.15B neurons, 128B synapses, 1,152 Loihi 2 chips, 2.6 kW; vendor efficiency claims >15 TOPS/W — marketing, not peer-reviewed.
- **Source type:** press release — **Tier:** 2

### [30] Open Neuromorphic — Loihi specs
- **Summary:** Loihi 1: 130k neurons, <1.5 W, on-chip learning, >5000× energy-delay on LASSO; Loihi 1 EOL; scaling/mapping challenges remain.
- **Source type:** community docs — **Tier:** 2

### [31] Merolla et al. 2015 (TrueNorth)
- **Summary:** 4,096 cores, 1M neurons, 256M synapses, 65 mW real-time video; ~26 pJ per synaptic event; no on-chip learning.
- **Source type:** academic/industry lab — **Tier:** 1

### [32] humanunsupervised.com 2025
- **Summary:** Neuromorphic landscape: fragmented ecosystem, immature toolchains, lab-only availability, GPU inertia; "efficiency alone doesn't guarantee accuracy."
- **Source type:** blog — **Tier:** 3

### [33] Blouw et al. 2019 (NICE)
- **Summary:** TrueNorth/Loihi orders-of-magnitude better energy per inference than CPU/GPU on streaming keyword spotting, comparable accuracy.
- **Source type:** academic — **Tier:** 1

### [34] Kudithipudi et al. 2022 (Nature)
- **Summary:** Field stalls without shared benchmarks, common software stacks, community infrastructure; cross-chip efficiency claims not comparable.
- **Source type:** academic perspective — **Tier:** 1

### [35] Zhang et al. 2024 (FSCIL survey)
- **Summary:** FSCIL defined; TOPIC canonical framework; miniImageNet/CIFAR-100/CUB benchmarks; open problems incl. human–machine efficiency gap.
- **Key quotes:** "FSCIL seeks to emulate human learning efficiency with minimal data."
- **Source type:** academic — **Tier:** 1

### [36] Lin et al. 2022 (MCUNetV3, NeurIPS)
- **Summary:** QAS + Sparse Update + Tiny Training Engine → training under 256KB SRAM / 1MB Flash; <1/1000 memory of PyTorch/TF; matches VWW accuracy.
- **Key quotes:** "make on-device training possible with only 256KB of memory."
- **Source type:** academic — **Tier:** 1

### [37] Bell et al. 2025
- **Summary:** CL for foundation models: continual pretraining, PEFT-for-CL, MoE; "the future of continual learning likely resides in decentralised ecosystems."
- **Source type:** academic — **Tier:** 1

### [38] Lin et al. 2025 (SESLR) — WITHDRAWN
- **Summary:** SNN sleep-wake online CL; claimed ~30% over ER/DER++ at 1/3 memory; **withdrawn by authors due to data errors — cautionary reference only**.
- **Source type:** academic (retracted) — **Tier:** 1 (void)

### [39] van de Ven et al. 2024
- **Summary:** CL evaluation critique: averaged metrics mask retention–acquisition tradeoffs; diagnostic metrics; task-free open problems.
- **Source type:** academic book chapter — **Tier:** 1

### [40] De Lange et al. 2023 (ICLR)
- **Summary:** The stability gap: transient accuracy dip during new-task learning, invisible to end-of-sequence metrics; worst-case continual evaluation proposed.
- **Source type:** academic — **Tier:** 1

### [41] Yang et al. (ACM TIST)
- **Summary:** LLM-CL survey: continual pretraining/instruction tuning/alignment; PEFT-CL; companion paper-tracking repo.
- **Source type:** academic — **Tier:** 1

### [42] Wei et al. 2024 (Online-LoRA)
- **Summary:** Task-free single-pass online CL on ViTs via LoRA + online weight regularization; no growing rehearsal buffer.
- **Source type:** academic — **Tier:** 1

### [43] Javed & White 2019 (OML/MRCL, NeurIPS)
- **Summary:** Meta-learned sparse, interference-resistant representations; new tasks need only small predictor updates; parent of ANML.
- **Source type:** academic — **Tier:** 1

### [44] McClelland et al. 1995 (CLS)
- **Summary:** Fast hippocampus + slow neocortex + interleaved replay solves stability–plasticity; theoretical root of replay-based CL.
- **Source type:** academic — **Tier:** 1

### [45] CLS neural network model 2025
- **Summary:** CLS theory operationalized with modern architectures; active 2025 research vein.
- **Source type:** academic — **Tier:** 1

### [46] MIT HAN Lab 2022
- **Summary:** MCUNetV3 accessible write-up; official PyTorch code, Colab-runnable.
- **Source type:** lab blog + code — **Tier:** 2

### [47] ContinualAI
- **Summary:** Curated CL bibliography + Avalanche framework; Awesome-Incremental-Learning for CIL/FSCIL tracking.
- **Source type:** community — **Tier:** 2

### [48] Ye & Bors 2024 (CVPR)
- **Summary:** Single-pass task-free online CL via dynamic cluster expansion; grows capacity with new distributions.
- **Source type:** academic — **Tier:** 1

### [49] Context-aware SNN CL 2024 (Neural Networks)
- **Summary:** Similarity-based context gating for SNN continual learning; peer-reviewed SNN×CL evidence.
- **Source type:** academic — **Tier:** 1

### [50] Bartunov et al. 2018 (NeurIPS)
- **Summary:** FA/TP match BP on MNIST, significantly worse on CIFAR-10/ImageNet; gap widens in locally-connected nets; the scalability-wall anchor.
- **Key quotes:** "TP and FA variants perform significantly worse than BP."
- **Source type:** academic — **Tier:** 1

### [51] Dellaferrera & Kreiman 2022 (PEPITA, ICML)
- **Summary:** Backward pass replaced by second forward pass with error-modulated input; MNIST/CIFAR-10/CIFAR-100, FC+conv.
- **Key quotes:** "replace the backward pass with a second forward pass in which the input signal is modulated based on the error of the network."
- **Source type:** academic — **Tier:** 1

### [52] Davies et al. 2021 (Nature MI)
- **Summary:** Loihi results survey; reported up to ~1000× energy-delay vs CPU on LASSO/optimization; gains confined to sparse event-driven workloads; toolchain/benchmark limitations. Snippet-verified.
- **Source type:** academic — **Tier:** 1 [snippet-level]

### [53] Meem et al. 2026 (IOP NCE)
- **Summary:** Spike budgeting as feedback control for energy-aware SNN CL; five class-incremental tasks; up to 47% spike reduction with accuracy gains; +17.45 pts DVS-Gesture via budget relaxation.
- **Key quotes:** "We formulate spike budgeting as a feedback control problem."
- **Source type:** academic — **Tier:** 1
