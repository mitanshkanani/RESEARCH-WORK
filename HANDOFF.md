# Handoff: state of the project on 2026-11-15

Read `CLAUDE.md` first (project goal and working rules). This file says where things stand.

## The project in one paragraph

Continual learning without catastrophic forgetting, without big retraining runs. Learn a new class
from 2-3 examples, keep old classes, use little GPU/RAM/energy. Brain-inspired mechanisms are the
first thing to try, not a requirement. Pretrained vs from-scratch is an open question to be
researched both ways.

## What exists

| Item | Where | State |
|---|---|---|
| Project rules and goal | `CLAUDE.md` | written, not committed or pushed |
| Literature report: growing structure, ranked methods | `RESEARCH_GROWING_STRUCTURE.md` | written, not committed or pushed |
| 14 new papers (PAPER 54-67) + manifest | `01-/02-/03-` folders, `PAPERS_MANIFEST.md` | added, not committed or pushed |
| Old agent's END_GOAL and reports | `_archive-previous-agent/` | moved with `git mv` (staged), not committed. Goals and recommendations there are not to be used |
| EXP-2 (feature ladder) | `EXP-2-FEATURE-LADDER/` | complete. Pushed to GitHub branch `exp-2-feature-ladder` (commit 88ceeb5). PDF journal: `EXP-2-JOURNAL.pdf` |
| EXP-1 | `EXP-1-CATASTROPHIC-FORGETTING/` | a design only, written by the earlier agent. Never run: no code, no results. Its plain "train A, then B" baseline is reused inside EXP-3 |
| EXP-3 (growing features) | `EXP-3-GROWING-FEATURES/` | complete (2026-11-15). Run on a private Kaggle GPU job (Tesla T4, 157 min), results in `results/runs.jsonl`, `summary.json`, `meta.json`. Journal `EXP-3-JOURNAL.pdf` has Part A and Part B (14 pages). Verdict by the pre-written rule: **does not work on photos**. Not committed |
| EXP-4 (fair cards) | `EXP-4-FAIR-CARDS/` | complete (2026-11-15). Private Kaggle job (Tesla T4, 64 min), results in `results/runs.jsonl`, `summary.json`, `meta.json`, `preds_*.npz`. Journal `EXP-4-JOURNAL.pdf` has Part A and Part B (15 pages). Verdict by the pre-written rule: **does not work on photos**; cause confirmed. Branch `exp-4-fair-cards` |
| Journal rule | `CLAUDE.md`, section "Experiment journals" | 2026-11-13: every experiment gets one PDF, Part A before the run and Part B appended after. EXP-2's journal got a "why this experiment" section to match |

The local repo is on branch `exp-2-feature-ladder`, not `main`. `01-continual-learning/PAPER 4.PDF`
had a modification before this work started; it is unrelated.

## EXP-2 result (read the journal for detail)

Classes arrive one at a time, 1-10 examples each, no task hint, no stored old examples, no backprop
in the learner. 3 datasets x 6 feature sources x 4 learners x 5 shot counts x 10 seeds = 3,600 runs.

Accuracy at 3 examples per class (best learner per source; % of pretrained ResNet-18 "F" in brackets):

| Feature source | CIFAR-100 | Omniglot | MNIST then Fashion |
|---|---|---|---|
| A raw pixels | 6.8 (19%) | 13.5 (23%) | 60.0 (92%) |
| B random expansion | 6.9 (20%) | 11.4 (20%) | 58.5 (90%) |
| C random conv net | 6.5 (18%) | 8.1 (14%) | 44.8 (69%) |
| D trained, no labels, minutes | 9.7 (27%) | 55.8 (96%) | 65.8 (101%) |
| E trained, labels, minutes | 16.6 (47%) | 75.3 (130%) | 63.4 (97%) |
| F ImageNet ResNet-18 | 35.2 | 58.2 | 65.3 |

What it means:
- The learning step is not the bottleneck; the features are. With good features a class is learned
  in about a millisecond without touching old classes.
- Omniglot (fairest test): about a minute of training on other characters beat ImageNet pretraining
  by 17 points. A big pretrained model is not needed when the features match the data.
- CIFAR-100: nothing cheap comes close (16.6 vs 35.2). This gap is the open problem.
- Free features (A, B, C) only work on easy data.
- The brain-style learners (`fly`, `clp`) lost to plain nearest-class-mean (`fly` in 18 of 18 cells,
  `clp` in 17 of 18). These are our implementations with our own choices where papers were silent.
- No catastrophic forgetting occurred, by construction. Accuracy on old classes still drops when
  similar new classes arrive (CIFAR-100: 44.7% to 34.7%).
- Next question (proposed EXP-3): how to build or grow good features cheaply and continually, not
  how to grow the classifier.

Caveats: sanity check got 79.4% against the published 83.7% (cause unconfirmed); D and E were
trained on data of the same kind as the test data; D is a weak label-free method; energy was not
measured; images only.

## EXP-3 result (read the journal for detail)

Four ways of treating the feature network ("eyes") as groups of 10 classes arrive, no stored old
examples, no task hint, nearest-class-mean memory. CIFAR-100, 8 stages, 5 seeds, after the last stage:

| Way | First group | All 80 classes | 20 unseen classes, 3 shots | All 80, open-book (diagnostic) |
|---|---|---|---|---|
| Frozen after stage 1 | 55.3 | 32.3 | 25.6 | 32.3 |
| Ordinary fine-tuning | 0.0 | 14.5 | 32.0 | 40.4 |
| Grow and lock (tested idea) | 60.4 | 29.9 | 30.0 | 40.4 |
| Retrain on everything (ceiling) | 54.6 | 52.6 | 39.0 | 52.6 |

- Locking works: old units changed by exactly 0.0, first group kept.
- Features improved a little (33% of the frozen-to-retrain gap on unseen classes), no more than
  ordinary fine-tuning's features did.
- Overall accuracy fell below frozen (share of gap -12%), so the rule gives "does not work".
- Likely cause, not directly verified: prototypes of old classes cover fewer units than new ones,
  and the cosine comparison on each prototype's own units favours old classes (per-group accuracy
  falls from 60.4 for the oldest group to 13.3 for the newest; open-book is roughly level).
- MNIST then Fashion-MNIST: grow and lock 89.1 against frozen 82.4, retrain 94.0, ordinary 47.5.
- Cost per seed on CIFAR-100: grow 129 s and 1.34 Wh, ordinary 205 s, retrain 924 s and 9.69 Wh
  (GPU-only energy on the Kaggle T4).
- Proposed EXP-4: make prototypes of different sizes compete fairly without old examples, and log
  the confusions so the cause is checked directly. Not designed yet; needs a Part A first.

## EXP-4 result (read the journal for detail)

Same grown network as EXP-3, eight rules for comparing a picture with prototypes of different
sizes, no old examples. Rule picked on a practice half of the test set, judged on the exam half.
CIFAR-100, after stage 8, exam half, 5 seeds (`results/summary.json`):

| | All 80 classes | First group | Cross-group errors going to older groups |
|---|---|---|---|
| Frozen (floor) | 32.9 | 56.2 | 56% |
| R0, the EXP-3 rule (selected by the written procedure) | 30.4 | 60.8 | 84% |
| R4 stranger-calibrated | 31.0 | 33.0 | 49% |
| R7 predicted fill-in (best score, failed step 3) | 39.5 | 39.5 | 50% |
| Open-book prototypes, same network (not allowed) | 41.0 | 47.0 | 54% |
| Retrain (ceiling) | 52.7 | 54.3 | 50% |

- Cause confirmed: under R0, 84.2% of cross-group errors went to older classes.
- R7 (ridge regression from old units to new units, used to complete old prototypes) wins back
  85% of the points lost to the unfair comparison, with no old examples and no extra cost at
  answer time. It would have closed 33% of the frozen-to-retrain gap.
- Step 3 (first group within 5 points of frozen) failed for every balanced rule, and would fail
  for open-book too. So the written procedure selected R0 and the verdict is "does not work".
  The 60% first-group score in EXP-3 was partly produced by the bias (interpretation).
- MNIST then Fashion-MNIST: R6 selected, 91.5 against frozen 82.3 and retrain 94.0 (79% of the
  gap), verdict "works".
- The remaining gap (41.0 to 52.7) is in the features, not the prototypes.
- No literature search was done for the rules; R7-like ideas may exist in published work.
- Proposed EXP-5: keep grow and lock and R7, change how the added units are trained (add a
  label-free signal), and replace step 3 with a test that separates forgetting from stronger
  rivals. Needs a literature pass by a subagent and a Part A first.

## Other literature facts worth keeping (from `RESEARCH_GROWING_STRUCTURE.md`)

- Zero-forgetting growth methods mostly need a task hint; no-hint growth methods mostly rely on replay.
- No paper was found that does few-shot + no replay + no task hint + no large pretraining. This is
  absence of evidence, not proof.
- Loss of plasticity (Dohare et al., Nature 2024; continual backprop) matters for any method that
  keeps training weights.
- Rules from earlier mistakes: never cite a paper that was not opened; tag claims as full text,
  abstract, or unverified. Earlier automated research produced four fake citations.

## How to run things

- Python: `EXP-2-FEATURE-LADDER/.venv/Scripts/python.exe` (environment lives inside the experiment folder).
- Rebuild: `src/data.py`, `src/features.py`, `src/run.py --tag full --seeds 10`, `src/figures.py --tag full`,
  `src/render_pdf.py`. The venv, data, weights and cache (7.4 GB) are not in git.
- Laptop: RTX 3050 4 GB, about 63 GB free disk. A background job was once killed for low memory; close
  heavy apps before long runs.
- Rules: each experiment in its own `EXP-<n>-<NAME>/` folder with everything inside; PDF journal on
  completion with SVG figures made from real results; plain-language explanations; no extra planning docs.

## Open decisions for the owner

1. Merge the experiment branches into `main`? Everything is pushed (`exp-2-feature-ladder`,
   `exp-3-growing-features`, `exp-4-fair-cards`), nothing is merged.
2. Go ahead with EXP-5 as proposed in section 18 of `EXP-4-FAIR-CARDS/EXP-4-JOURNAL.pdf`, or
   pick a different next step?
