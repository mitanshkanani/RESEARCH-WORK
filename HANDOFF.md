# Handoff: state of the project on 2026-11-17 (after EXP-10)

Read `CLAUDE.md` first (project goal and working rules). This file says where things stand.

## RESUME HERE (written 2026-11-17, for a fresh session after /clear)

**Where we are (updated 2026-11-17 after EXP-10).** EXP-1 to EXP-10 are done, with journals.
EXP-11 is only proposed (section 17 of `EXP-10-JOURNAL.pdf`). No other branches (owner's rule).

**Standing rules from the owner (all in memory and `CLAUDE.md`):**
- Work on `main` only, no branches.
- **Ask before** pushing to GitHub, pushing/running anything on Kaggle, or downloading anything.
  The overnight "work without asking" mode (2026-11-15 night) ended 2026-11-16 08:44 IST. Do not
  assume it carries over.
- Each experiment: own folder `EXP-<n>-<NAME>/`, its own `.venv`, Part A of `JOURNAL.md` rendered to
  PDF and committed **before any code runs**, then code, tiny smoke check on the laptop (borrow
  `EXP-2-FEATURE-LADDER/.venv` and `data/`), private Kaggle T4 job, results into `results/`,
  figures from real results, Part B appended, update `CLAUDE.md` and this file.
- Explain in plain language; the owner is not an ML researcher. Paper reading goes to Sonnet
  subagents; never cite a paper that was not opened; tag read/abstract/unverified.
- Kaggle free GPU cap is about 30 h/week; about 23.7 h used since 2026-11-15 (EXP-3 to EXP-10).
  CIFAR-100 downloads slowly on Kaggle (20 of EXP-9's 28.5 min; 53.5 of EXP-10's 70 min). Proposed
  fix (needs owner approval): upload CIFAR-100 once as a private Kaggle dataset.

**The story in six lines (exam half, CIFAR-100, 80 classes in 8 stages, no stored old pictures):**
1. Plain class-mean memory + trained eyes: frozen 33%, grow-and-lock 40%, retrain 53%.
2. EXP-4 fixed a bias (R7 fill-in of old short cards); EXP-5 showed extra training jobs do not
   help; EXP-6 showed locking-and-building-in-steps costs 78% of the gap.
3. EXP-7: eyes of fixed whitened 6x6 patches (no backprop, one pass) + class means with one
   shared covariance (LDA) = 55.2%, 24 s.
4. EXP-8, same memory for all: ceiling (retrain, own softmax layer) **61.1%**; retrain+LDA 58.8%;
   **patch+LDA 55.2%**; frozen+LDA 44.1%; grow-and-lock 43.7%. Grow-and-lock no longer pays.
5. Open problem: **3-shot learning of new classes** (patch 21%, frozen 26%, grow-and-lock 30%,
   retrain 38.5%). No memory fixes it; it depends on how much the eyes learned from labels.
6. Patch eyes need 5.6 GB GPU as written (laptop has 4 GB) and 6 points to the ceiling remain.
7. **EXP-9 fixed point 5 (verdict "works")**: project onto the discriminant directions of the
   known classes (one calculation from the class means and the shared table), then plain cards by
   angle, plus borrowing from the 2 nearest known cards: **33.5% from 3 pictures** on patch eyes
   (frozen 29.6%), beating grow and lock 29.8%. Rises with known classes (16.1 / 29.9 / 33.5 at
   10 / 40 / 80). Peak GPU 2.9 GB now. But among all 100 classes the new ones score only 7.1%.
8. **EXP-10 ("does not work")**: a handicap on old classes' scores lifts the harmonic mean of old
   and new from 12.4 to 31.5 (new 26.8%), but old classes fall 52.2 -> 38.1 (limit 48.3). Score
   shifting only trades old for new; with all their pictures new classes would reach 47.6% (old
   50.3%), so the limit is the card from few pictures. Certainty correction (distance / (1+1/n),
   no setting) 28.5. New classes in the view: worse. Stream: forgetting tiny (57.7 -> 54.6).
   Lesson: choose handicaps under the old-class limit, and never judge by plain all-class accuracy.

**Next steps, in order (owner decides):**
- **Commit/push anything pending** (check `git status`; `PAPER 4.PDF` is an old unrelated change,
  leave it unstaged).
- **EXP-11 (proposed, section 17 of `EXP-10-JOURNAL.pdf`):** better cards through better eyes:
  mirrored copies of each picture, +/- parts, finer regions, more detectors (under 4 GB GPU);
  all three tests (80 classes; 20 new from 3 pictures; 100 together with handicap chosen so old
  stays within 5 points). Write Part A first.
- Later: cheap boosts to patch eyes (mirrored images, +/- parts, finer regions, more detectors,
  second layer by grouping co-active detectors); make patch eyes fit 4 GB; a laptop-sized language
  test only when the owner asks (see literature log section 5).

**Useful facts.** Kaggle status: `PYTHONUTF8=1 kaggle kernels status mitanshkanani/<name>`; download
logs with `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`. PDF render: `.venv/Scripts/python.exe src/render_pdf.py`
inside each experiment. No `gh`, no `pdftoppm`; check SVG figures by headless Chrome screenshot.
Literature notes: `RESEARCH_LITERATURE_LOG.md` (8 sections, subagent passages-only reading).

## The project in one paragraph

Continual learning without catastrophic forgetting, without big retraining runs. Learn a new class
from 2-3 examples, keep old classes, use little GPU/RAM/energy. Brain-inspired mechanisms are the
first thing to try, not a requirement. Pretrained vs from-scratch is an open question to be
researched both ways.

## What exists

| Item | Where | State |
|---|---|---|
| Project rules and goal | `CLAUDE.md` | on `main` |
| Literature report: growing structure, ranked methods | `RESEARCH_GROWING_STRUCTURE.md` | on `main` |
| 14 new papers (PAPER 54-67) + manifest | `01-/02-/03-` folders, `PAPERS_MANIFEST.md` | on `main` |
| Old agent's END_GOAL and reports | `_archive-previous-agent/` | on `main`. Goals and recommendations there are not to be used |
| EXP-2 (feature ladder) | `EXP-2-FEATURE-LADDER/` | complete. PDF journal: `EXP-2-JOURNAL.pdf` |
| EXP-1 | `EXP-1-CATASTROPHIC-FORGETTING/` | a design only, written by the earlier agent. Never run: no code, no results. Its plain "train A, then B" baseline is reused inside EXP-3 |
| EXP-3 (growing features) | `EXP-3-GROWING-FEATURES/` | complete (2026-11-15). Run on a private Kaggle GPU job (Tesla T4, 157 min), results in `results/runs.jsonl`, `summary.json`, `meta.json`. Journal `EXP-3-JOURNAL.pdf` has Part A and Part B (14 pages). Verdict by the pre-written rule: **does not work on photos**. Not committed |
| EXP-4 (fair cards) | `EXP-4-FAIR-CARDS/` | complete (2026-11-15). Private Kaggle job (Tesla T4, 64 min), results in `results/runs.jsonl`, `summary.json`, `meta.json`, `preds_*.npz`. Journal `EXP-4-JOURNAL.pdf` has Part A and Part B (15 pages). Verdict by the pre-written rule: **does not work on photos**; cause confirmed |
| EXP-5 (better eyes) | `EXP-5-BETTER-EYES/` | complete (2026-11-15). Private Kaggle job (Tesla T4, 216 min), results in `results/runs.jsonl`, `summary.json`, `meta.json`, `preds.npz`. Journal `EXP-5-JOURNAL.pdf` has Part A and Part B (15 pages). Verdict by the pre-written rule: **does not work** |
| EXP-6 (what limits the eyes) | `EXP-6-WHAT-LIMITS-THE-EYES/` | complete (2026-11-15). Private Kaggle job (Tesla T4, 173 min), results in `results/runs.jsonl`, `summary.json`, `meta.json`, `preds.npz`. Journal `EXP-6-JOURNAL.pdf` has Part A and Part B. A diagnosis: **locking and building in steps is the main cause** |
| EXP-7 (patch eyes) | `EXP-7-PATCH-EYES/` | complete (2026-11-16). Private Kaggle job (Tesla T4, 64 min), results in `results/runs.jsonl`, `summary.json`, `sanity.json`, `meta.json`, `preds.npz`. Journal `EXP-7-JOURNAL.pdf` has Part A and Part B. Verdict by the pre-written rule: **works**, with three warnings |
| EXP-8 (same memory) | `EXP-8-SAME-MEMORY/` | complete (2026-11-16). Private Kaggle job (Tesla T4, 47 min), results in `results/runs.jsonl`, `summary.json`, `meta.json`. Journal `EXP-8-JOURNAL.pdf` has Part A and Part B. Patch features with LDA are "close, at a fraction of the cost" to the ceiling |
| EXP-9 (few pictures) | `EXP-9-FEW-PICTURES/` | complete (2026-11-17). Private Kaggle job (Tesla T4, 28.5 min, 20 of them dataset download), results in `results/runs.jsonl`, `summary.json`, `meta.json`. Journal `EXP-9-JOURNAL.pdf` has Part A and Part B. Verdict by the pre-written rule: **works** (33.5% from 3 pictures) |
| EXP-10 (new among old) | `EXP-10-NEW-AMONG-OLD/` | complete (2026-11-17). Private Kaggle job (Tesla T4, 70 min, 53.5 of them dataset download), results in `results/runs.jsonl`, `summary.json`, `meta.json`. Journal `EXP-10-JOURNAL.pdf` has Part A and Part B. Verdict by the pre-written rule: **does not work** |
| Literature log | `RESEARCH_LITERATURE_LOG.md` | nine sections written 2026-11-15/17 from subagent reading (passages, not full papers) |
| Journal rule | `CLAUDE.md`, section "Experiment journals" | 2026-11-13: every experiment gets one PDF, Part A before the run and Part B appended after. EXP-2's journal got a "why this experiment" section to match |

Everything is on `main`; the owner wants no other branches (2026-11-15). `01-continual-learning/PAPER 4.PDF`
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

## EXP-5 result (read the journal for detail)

Grow and lock with R7 fixed; four ways of training the units added at each stage; each run at
30, 10, 3 and 1 passes. CIFAR-100, after stage 8, exam half, 5 seeds (`results/summary.json`):

| Way (30 passes) | All 80 classes (R7) | Open-book | 20 unseen classes, 3 shots | Time / energy per seed |
|---|---|---|---|---|
| T0 cross-entropy on the stage's 10 classes | 39.6 | 40.9 | 29.6 | 128 s / 1.34 Wh |
| T1 rotation as extra classes | 38.5 | 40.2 | 29.7 | 484 s / 5.07 Wh |
| T2 rotated copies, same label (control) | 34.3 | 36.1 | 28.8 | 483 s / 5.06 Wh |
| T3 cross-entropy + two-view contrastive (selected) | 40.1 | 41.3 | 29.8 | 259 s / 2.71 Wh |
| Frozen | 32.8 | | 25.8 | 24 s / 0.25 Wh |
| Retrain once | 53.2 | | 38.8 | 191 s / 2.02 Wh |

- Verdict "does not work": the selected way closed 4% of the gap to retrain; open-book rose 0.5.
- New forgetting test (first group scored with only its own 10 prototypes allowed): 73.4 to 70.8
  for T0 over seven stages. The fall of the first group among all classes (73 to 39) is rivalry.
- Passes, T0 with R7 / retrain: 30: 39.6 / 53.2; 10: 33.1 / 48.2; 3: 23.1 / 38.6; 1: 16.5 / 26.1.
- Literature (passages read by subagents, not full papers): R7 is known as drift compensation
  (LDC arXiv:2407.08536; a ridge fill-in for growing features in DGASA, PLoS ONE 2026; EASE
  arXiv:2403.12030). Rotation self-supervision is reported to help with a large first task
  (PASS++ arXiv:2407.14029) and not in cold start (EFC arXiv:2402.03917).
- Why grown features stay about 12 points under retrain is unknown. Candidates: 10 classes per
  stage, 16 units per stage, locking.
- Proposed EXP-6: diagnostic. Grow and lock with (a) 48 units per stage, (b) new units trained
  on all classes so far (old data, diagnosis only), (c) both; see which moves open-book and the
  3-shot test. Needs a Part A first.

## EXP-6 result (read the journal for detail)

Diagnosis of why grown features stay under retraining. Grow and lock with R7, 30 passes,
CIFAR-100, 80 classes, exam half, 5 seeds (`results/summary.json`). Open-book accuracy:

| Run | Open-book | 20 unseen classes, 3 shots | Time / energy per seed |
|---|---|---|---|
| A as before (16 units per stage, final 144) | 41.0 | 29.7 | 131 s / 1.36 Wh |
| C diagnosis: new units trained on all classes so far, old units locked | 43.7 | 31.1 | 737 s / 7.68 Wh |
| Retrain, size 144 | 53.2 | 38.5 | 202 s / 2.12 Wh |
| B double units (final 288) | 42.7 | 30.9 | 238 s / 2.48 Wh |
| Retrain, size 288 | 55.0 | 40.2 | 453 s / 4.75 Wh |

- Gap at size 144 is 12.2 points: narrow data per stage explains 2.7 (22%, "minor"); locking
  and staged building explains 9.5 (78%, "main cause").
- Gap at size 288 is 12.3 points: size is not a cause of the gap.
- Double units with R7: 41.7 against 39.6, 15% of the gap, "does not work"; first group with
  its own prototypes dropped 3.4 points.
- Not separated: "old units cannot improve" against "network built piece by piece". A larger
  first group was not tested.
- Proposed EXP-7: feature extractors that are not fitted to classes at all (fixed whitened
  image patches, no backprop, one pass; see section 6 of `RESEARCH_LITERATURE_LOG.md`), with
  the same 80-class protocol, plus a shared-covariance memory. Needs a Part A first.

## EXP-7 result (read the journal for detail)

A feature extractor with no backprop: 1,600 fixed whitened 6x6 patches taken from the first
10 classes, soft-threshold or triangle encoding, sum-pooled over 4 quadrants (6,400 numbers).
Memories filled in one pass with no stored images. Same 80-class protocol, exam half, 5 seeds
(`results/summary.json`). Backprop references are quoted from EXP-6, not rebuilt.

| | All 80 classes | 20 unseen classes, 3 shots | Time per seed |
|---|---|---|---|
| Patch features + class means and one shared covariance (LDA, shrinkage 0.01), selected | **55.2** | 8.2 (17.0 with plain means on square-rooted features) | 24 s |
| Patch features + closed-form ridge classifier | 53.9 | – | similar |
| Patch features + plain class means (cosine) | 15.4 | 15.8 | 9 s |
| Retrain, backprop, plain class means (EXP-6) | 53.2 | 38.5 | 202 s |
| Grow and lock, backprop, R7 (EXP-6) | 39.6 | 29.7 | 131 s |
| Frozen backprop features (EXP-6) | 33.2 | 25.6 | 26 s |

- CIFAR-10 implementation check: 76.3% (ridge), 76.7% (LDA); Coates et al. 2011 report 77.9%.
- No forgetting: first group with its own classes only 77.6 to 78.2. Per-group accuracy level.
- Patches from the first 10 classes are as good as patches from all 80 (53.8 against 53.9).
  K-means patches +1.0 over random. 400 patches: 47.5. Growing the dictionary gave nothing
  with class means (not tested with LDA).
- Smallest shrinkage tried (0.01) was best; smaller not tried.
- **Not a fair comparison yet**: LDA and ridge were never applied to the backprop features, and
  retraining was never scored with its own trained output layer. EXP-2's conclusion that the
  feature extractor is the bottleneck needs re-checking with this memory.
- Weak at 3 shots. Peak GPU memory 5.5 GB as written (features held on the GPU).
- Proposed EXP-8: LDA and ridge on frozen, grow-and-lock and retrained backprop features;
  retraining's own softmax accuracy as the ceiling; patch features rebuilt alongside with
  smaller shrinkage; 3-shot test for all; time and energy including the tables.

## EXP-8 result (read the journal for detail)

The same memories on four feature extractors, same 80-class protocol, exam half, 5 seeds
(`results/summary.json`). Memory chosen per extractor on the practice half.

| Feature extractor | Plain class means | Best table memory | 3 shots, best of any memory | Time / energy per seed |
|---|---|---|---|---|
| Patch features (no backprop) | 15.4 | **55.2** (LDA, shrinkage 0.01) | 21.2 | 24 s / 0.25 Wh |
| Frozen backprop features | 32.9 | 44.1 (LDA, 0.1) | 25.6 | 24 s / 0.25 Wh |
| Grow and lock | 39.7 (R7) | 43.7 (LDA with filled-in blocks, 1.0); open-book LDA 46.5 | 29.8 | 127 s / 1.32 Wh |
| Retrain (keeps all data) | 53.1 | 58.8 (LDA, 0.1) | 38.5 | 190 s / 2.00 Wh |
| Retrain with its own softmax layer (ceiling) | | **61.1** | | 190 s / 2.00 Wh |

- LDA beat ridge on every extractor. Forgetting with the selected memory is at most 2.1 points.
- The memory was a bottleneck in all earlier experiments (+11 on frozen, +6 on retrain, +40 on patch).
- Grow and lock has no advantage over frozen features under the same memory, except at 3 shots.
- 3-shot accuracy follows how much the features learned from labels; no memory changed it for
  backprop features. LDA with very strong shrinkage (10 to 100) adds about 4 points on patch features.
- Patch features used 5.6 GB of GPU memory as written.
- Proposed EXP-9: few-shot learning on patch features with closed-form steps (centering and
  normalising, projecting onto the discriminant directions of the known classes, borrowing from
  nearest known classes), at 1, 3 and 10 shots and with 10, 40 or 80 known classes.

## EXP-9 result (read the journal for detail)

20 surprise classes learned from 1/3/10 pictures with the memory after 10/40/80 known classes,
square-rooted descriptions, settings chosen in a rehearsal (memory after stage 6, classes of
stages 7-8 as pretend-new), 5 seeds (`results/summary.json`). 3 pictures, 80 known:

| Way (setting from rehearsal) | Patch eyes | Frozen backprop eyes |
|---|---|---|
| W0 plain cards | 17.0 | 25.2 |
| W1 shared table, smoothing 100 | 21.2 | 25.5 |
| W2 subtract known mean, angle | 18.0 | 25.6 |
| W3 learned view (P: smoothing 0.1, F: 1; all C-1 directions; angle) | 32.0 | 28.5 |
| **W3+B + borrow 2 nearest known cards, weight 1 picture** | **33.5** | 29.6 |

- 1 / 3 / 10 pictures, patch W3+B: 23.3 / 33.5 / 44.2 (grow and lock 19.4 / 29.8 / 41.1; retrain
  25.4 / 38.5 / 51.5, quoted from EXP-8).
- Known classes 10 / 40 / 80: 16.1 / 29.9 / 33.5. With 10 known the view is slightly worse than plain.
- Rehearsal picked the best way on the real test for both eyes (rehearsal 3-4 points optimistic).
- All 100 classes together (new from 3 pictures): new classes 7.1%, known 52.2% (53.3% without the
  new ones). EXP-8's LDA at 0.01: new 0.05%, known 54.2%.
- Cost per seed: patch eyes ~65 s on T4 including all ways, 0.69 Wh, peak GPU 2.9 GB.
- My prediction was "helps, but not enough" (about 26%); wrong.

## EXP-10 result (read the journal for detail)

20 new classes (1/3/10 pictures) named among all 100; handicap chosen in a rehearsal (memory after
stage 6, stages 7-8 pretend-new, among 80) by highest harmonic mean (`results/summary.json`).
Patch eyes, 3 pictures: old / new / harmonic mean:

| Correction | Old | New | H |
|---|---|---|---|
| C0 none (EXP-9) | 52.2 | 7.0 | 12.4 |
| **C1 handicap 0.18 on cosine scores** (selected) | 38.1 | 26.8 | **31.5** |
| C2 Mahalanobis in the view / (1 + 1/n), no setting | 37.5 | 23.0 | 28.5 |
| C3 C2 + handicap | 35.4 | 24.0 | 28.6 |
| C4 new classes added to the view + handicap | 35.5 | 22.9 | 27.8 |
| Reference: new classes from all pictures | 50.3 | 47.6 | 48.9 |

- Best on the C1 curve with old >= 48.3 (read on the test, diagnosis): handicap 0.08, new 16.5, H 24.6.
- Stream (60 + 8 x 5 classes, 5 pictures): after session 8, all-class accuracy C0 37.1 vs C1 30.8;
  H 18.0 vs 29.3. Old 60 classes without correction 57.7 -> 54.6.
- Peak GPU 2.6 GB (patch). Rehearsal picked the right correction.

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

1. Go ahead with EXP-11 as proposed in section 17 of `EXP-10-NEW-AMONG-OLD/EXP-10-JOURNAL.pdf`, or
   pick a different next step?
2. Upload CIFAR-100 as a private Kaggle dataset to skip the slow download?
