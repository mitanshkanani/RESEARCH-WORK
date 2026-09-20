# Papers Manifest

35 downloaded PDFs, sorted into three folders by each paper's **primary** contribution.
Files keep their original `PAPER N` names. Generated 2026-09-20.

Cross-listings are noted in the **Also relevant to** column — a paper lives in only one
folder, so check these before assuming a topic isn't covered.

## 01-continual-learning/ — 13 papers

| File | Paper | Also relevant to |
|---|---|---|
| PAPER 3.pdf | Kirkpatrick et al. 2017 (PNAS) — Overcoming catastrophic forgetting in neural networks (**EWC**) | brain-inspired (synaptic consolidation) |
| PAPER 4.pdf | Wang, Zhang, Su & Zhu 2024 (TPAMI) — A Comprehensive Survey of Continual Learning | — |
| PAPER 6.pdf | De Lange et al. 2021 (TPAMI) — A Continual Learning Survey: Defying Forgetting in Classification Tasks | — |
| PAPER 7.pdf | Buzzega et al. 2020 (NeurIPS) — Dark Experience for General Continual Learning (**DER/DER++**) | energy (cheap baseline) |
| PAPER 21.pdf | Liang, He & Tan 2024 (IJCV) — A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts | brain-inspired (online adaptation) |
| PAPER 22.pdf | Sun et al. 2020 (ICML) — Test-Time Training with Self-Supervision | brain-inspired |
| PAPER 35.pdf | Zhang et al. 2024 — Few-shot Class-incremental Learning for Classification and Object Detection: A Survey (**FSCIL**) | brain-inspired (few-shot learning) |
| PAPER 37.pdf | Bell et al. 2025 — The Future of Continual Learning in the Era of Foundation Models | — |
| PAPER 39.pdf | van de Ven, Soures & Kudithipudi 2024 — Continual Learning and Catastrophic Forgetting (book chapter) | — |
| PAPER 40.pdf | De Lange, van de Ven & Tuytelaars 2023 (ICLR) — Continual Evaluation for Lifelong Learning: Identifying the **Stability Gap** | energy (evaluation method) |
| PAPER 41.pdf | Yang et al. 2024/25 (ACM TIST) — Recent Advances of Foundation Language Models-based Continual Learning | — |
| PAPER 42.pdf | Wei, Li, Marculescu 2024 — **Online-LoRA**: Task-free Online Continual Learning via Low Rank Adaptation | energy (parameter efficiency) |
| PAPER 43.pdf | Javed & White 2019 (NeurIPS) — Meta-Learning Representations for Continual Learning (**OML**) | brain-inspired (meta-learning) |

## 02-brain-inspired-neuro/ — 13 papers

| File | Paper | Also relevant to |
|---|---|---|
| PAPER 12.pdf | Qi et al. 2025 — Towards the Training of Deeper **Predictive Coding** Neural Networks | — |
| PAPER 13.pdf | Hinton 2022 — The **Forward-Forward** Algorithm ⚠️ *duplicate of PAPER 25* | — |
| PAPER 14.pdf | Lillicrap et al. 2016 (Nature Comm.) — Random synaptic feedback weights (**Feedback Alignment**) | — |
| PAPER 17.pdf | Bellec et al. 2020 (Nature Comm.) — A solution to the learning dilemma for recurrent spiking neurons (**e-prop**) | energy (spiking) |
| PAPER 18.pdf | Gygax & Zenke 2025 (Neural Comp.) — Theoretical Underpinnings of **Surrogate Gradient** Learning in SNNs | energy (spiking) |
| PAPER 19.pdf | Seely et al. 2025 (Sakana AI) — **Augmented Lagrangian Predictive Coding** (PC-ALM) | — |
| PAPER 23.pdf | LeCun 2022 — A Path Towards Autonomous Machine Intelligence (**JEPA** / world models) | continual learning |
| PAPER 24.pdf | Ramsauer et al. 2021 (ICLR) — **Hopfield Networks** is All You Need | — |
| PAPER 25.pdf | Hinton 2022 — The Forward-Forward Algorithm ⚠️ *duplicate of PAPER 13* | — |
| PAPER 26.pdf | Bedi & Khandwala (Stanford report) — Testing the Limits of Biologically-Plausible Backpropagation | — |
| PAPER 45.pdf | Jun et al. 2025 — A Neural Network Model of **Complementary Learning Systems** (hippocampus/neocortex) | continual learning |
| PAPER 50.pdf | Bartunov et al. 2018 (NeurIPS) — Assessing the Scalability of Biologically-Motivated Deep Learning | — |
| PAPER 51.pdf | Dellaferrera & Kreiman 2022 (ICML) — Error-driven Input Modulation (**PEPITA**) | — |

## 03-energy-efficiency/ — 9 papers

| File | Paper | Also relevant to |
|---|---|---|
| PAPER 1.pdf | Strubell et al. 2019 (ACL) — Energy and Policy Considerations for Deep Learning in NLP | — |
| PAPER 2.pdf | Patterson et al. 2021 — Carbon Emissions and Large Neural Network Training | — |
| PAPER 10.pdf | Jegham et al. 2025 — How Hungry is AI? Benchmarking Energy, Water, and Carbon of LLM Inference | — |
| PAPER 27.pdf | Yan et al. 2024 — **Reconsidering** the Energy Efficiency of Spiking Neural Networks | brain-inspired (SNN critique) |
| PAPER 28.pdf | Abreu et al. 2025 (ICLR-W) — Neuromorphic Principles for Efficient LLMs on **Intel Loihi 2** | brain-inspired |
| PAPER 33.pdf | Blouw et al. 2019 — Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware | brain-inspired |
| PAPER 36.pdf | Lin et al. 2022 (NeurIPS) — On-Device Training Under 256KB Memory (**MCUNetV3**) ⚠️ *duplicate of PAPER 46* | continual learning (edge) |
| PAPER 46.pdf | Lin et al. 2022 — MCUNetV3 ⚠️ *duplicate of PAPER 36* | continual learning (edge) |
| PAPER 53.pdf | Meem et al. 2026 (IOP NCE) — Energy-aware **spike budgeting** for continual learning in SNNs | continual learning **and** brain-inspired — spans all three |

---

## Two things worth knowing

**1. Two duplicate pairs, checked by SHA-256.** PAPER 36 and PAPER 46 (MCUNetV3) are
**byte-identical** — one of them is pure redundancy and safe to delete. PAPER 13 and
PAPER 25 (Forward-Forward) are **different versions** of the same paper (16 vs 17 pages,
different hashes), so keep both until you compare them — one is likely a later revision
with extra figures. Nothing was deleted.

**2. One paper spans all three folders.** PAPER 53 (spike budgeting) is simultaneously a
continual-learning method, an SNN/brain-inspired method, and an energy-accounting method.
It sits in `03-energy-efficiency/`. If you end up working on the CL × SNN intersection,
this is the single closest piece of existing work to that idea — read it first.

## Coverage gap in the collection

The survey cites 53 sources; this folder has 35. Notably **absent** as PDFs: the
three-incremental-scenarios paper (van de Ven et al., *Nature MI* 2022) — the most-cited
benchmarking critique in the whole survey — plus the OpenNeuromorphic/Loihi hardware
spec sources and the HBP 20W-brain material. Worth downloading if you keep building this
library.
