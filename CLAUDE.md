# Research project: learning without forgetting, without brute-force training

This file is the source of truth for what the project is. It was written on 2026-11-12 from the
owner's own description. Where it disagrees with anything in `_archive-previous-agent/`, this file wins.

## The problem we are attacking

Today's AI gets better by adding more data and more GPUs. We think that is the wrong loop: it is
like burning more coal to keep a train running instead of building a better engine. The problems
below are connected, and we treat them as one problem:

1. **Catastrophic forgetting.** Train a network on cats vs dogs, then on digits, and it loses cats
   vs dogs, because the same weights get overwritten.
2. **Data hunger.** A model should recognise a dog after 2-3 dog images, not thousands.
3. **Compute and energy.** Models need huge GPU, RAM and electricity because of how they are
   trained. A method that learns from few examples without retraining needs far less of all three.

## The goal

Build a learning system that:

- learns a new class or skill from a handful of examples,
- keeps what it already learned when it learns something new,
- does this without a large retraining run.

Solving continual learning and catastrophic forgetting is the main aim. Lower energy use is
expected to follow from it; if it does not, the main aim still stands.

**Where this is heading (owner, 2026-11-15).** The end goal is a general model in the style of
GPT that can be trained and run at low cost. Images come first: get a method that works on
image tasks, then move to language. So a method is only interesting if its core idea does not
depend on pictures. When designing an experiment, keep the image-specific parts (for example
image filters) separate from the parts meant to carry over (how knowledge is added, protected
and looked up), and say in the journal which is which. Do not add language experiments until
the owner asks for them.

## How to think about "brain-inspired"

The brain is the motivating example: it forms new connections when it learns, it does not wipe
old knowledge, and it runs on very little power. Try brain-like mechanisms first (new
connections or neurons for new knowledge, local learning, sparse activity).

It is not a requirement. A solution that is 50-60% brain-like plus something else, or not
brain-like at all, counts as success if it solves the problems above. Do not reject a method for
being conventional, and do not keep one only because it is biological.

## Open question: pretrained models

The owner is sceptical of starting from a big pretrained model: its knowledge is locked in
ordinary weights, and it is unclear how that could be converted into a brain-like network.
Current stance: research both from-scratch and pretrained-feature approaches and report how much
of any few-shot ability comes from the pretraining rather than from the learning mechanism.

## People and folders

- The owner (MK) does research in this repo.
- A collaborator runs experiments in a separate folder (a sandbox teaching a small language
  model new skills). Do not assume that work is visible from here.
- `01-continual-learning/`, `02-brain-inspired-neuro/`, `03-energy-efficiency/`: paper PDFs,
  indexed by `PAPERS_MANIFEST.md`.
- `EXP-1-CATASTROPHIC-FORGETTING/`: a design only, written by an earlier agent. **It was never
  run: there is no code and there are no results.** Its one useful piece (train on A, then on B,
  watch A collapse) is the "ordinary training" baseline inside EXP-3.
- `RESEARCH_GROWING_STRUCTURE.md`: 2026-11-12 literature pass on growing structure and few-shot
  learning without forgetting, with the ranked methods and the design of EXP-2.
- `EXP-2-FEATURE-LADDER/`: tests how cheap the features can be before few-shot,
  no-forgetting learning breaks. Complete; journal `EXP-2-JOURNAL.pdf`.
- `EXP-3-GROWING-FEATURES/`: tests whether the features themselves can keep learning as new
  classes arrive without damaging old classes, by adding new units and freezing old ones.
  Complete (2026-11-15); journal `EXP-3-JOURNAL.pdf`. Old classes were kept and the features
  improved a little, but overall accuracy on photos fell below frozen features, so the verdict
  by its own rule was "does not work". Section 18 of the journal proposes EXP-4.
- `EXP-4-FAIR-CARDS/`: tests eight ways of comparing a picture with class prototypes of
  different sizes on the EXP-3 network, without old examples. Complete (2026-11-15); journal
  `EXP-4-JOURNAL.pdf`. The suspected cause was confirmed (84% of cross-group errors went to
  older classes). One rule (R7, filling in the missing part of old prototypes by regression)
  reached 39.5% against 32.9% frozen and 41.0% open-book, but dropped the first group by 17
  points, so the verdict by the written rule was again "does not work". Section 18 proposes
  EXP-5 (better features at each stage, R7 fixed).
- `EXP-5-BETTER-EYES/`: tests whether the units added at each stage can be trained to give
  more general features (three extra training signals, R7 fixed), and how the score falls
  with fewer passes over the data. Complete (2026-11-15); journal `EXP-5-JOURNAL.pdf`. None
  of the extra signals improved the features (40.1% against 39.6% without, retrain 53.2%), so
  the verdict was "does not work". Real forgetting measured under 3 points; one pass over
  the data gives 16.5% against 39.6% with 30 passes. Section 18 proposes EXP-6 (diagnose
  what limits the grown features: units per stage, or seeing only 10 classes per stage).
- `EXP-6-WHAT-LIMITS-THE-EYES/`: a diagnosis of why grown features stay under retraining:
  narrow data per stage, too few units per stage, or locking. Complete (2026-11-15); journal
  `EXP-6-JOURNAL.pdf`. Locking and building in steps is the main cause (78% of the 12-point
  gap), narrow data is minor (22%), and doubling the units leaves the gap unchanged.
  Section 17 proposes EXP-7 (feature extractors made of whitened image patches, no backprop).
- `EXP-7-PATCH-EYES/`: tests a feature extractor made of fixed whitened image patches (no
  backprop, one pass) with class-mean, shared-covariance and ridge memories on the same
  80-class protocol. Complete (2026-11-16); journal `EXP-7-JOURNAL.pdf`. Patch features
  with class means plus one shared covariance (LDA) scored 55.2% on 80 classes with no
  backprop, one pass and no forgetting, in 24 s, against 53.2% for the retraining reference
  and 39.6% for grow and lock. With plain class means the same features score 15.4%.
  Warnings: the references used plain class means (comparison not yet fair), 3-shot accuracy
  is poor (17.0% against 29.7%), and peak GPU memory was 5.5 GB. Section 17 proposes EXP-8
  (same memory on every feature extractor, and the true ceiling).
- `EXP-8-SAME-MEMORY/`: applies the same memories (class means, shared covariance, ridge)
  to every feature extractor (patch, frozen, grow and lock, retrain) and measures the
  retrained network with its own output layer as the ceiling. Complete (2026-11-16); journal
  `EXP-8-JOURNAL.pdf`. Ceiling 61.1%. With the same memory (class means plus shared
  covariance): retrain 58.8%, patch features 55.2%, frozen 44.1%, grow and lock 43.7%. So
  the no-backprop system is 6 points under the ceiling at about an eighth of the cost, and
  grow and lock no longer beats frozen features. 3-shot accuracy is not fixed by any memory
  (patch 21.2%, grow and lock 29.8%, retrain 38.5%). Section 17 proposes EXP-9 (few-shot
  learning on patch features with closed-form steps only).
- `EXP-9-FEW-PICTURES/`: tests whether the memory's knowledge of known classes can make a
  new class learnable from 1-10 pictures on patch features, by single calculations. Complete
  (2026-11-17); journal `EXP-9-JOURNAL.pdf`. Verdict "works": a "learned view" (the
  directions that separate the known classes, computed from the class means and the shared
  covariance) plus borrowing from the 2 nearest known classes gives 33.5% from 3 pictures
  among 20 new classes, against 17.0% before, 29.8% for grow and lock and 38.5% for retrain.
  Knowing more helps (16.1% with 10 known classes, 33.5% with 80). Patch eyes now fit in
  2.9 GB of GPU memory. Open: among all 100 classes the new ones score only 7.1%. Section 17
  proposes EXP-10 (new classes among old ones).
- `EXP-10-NEW-AMONG-OLD/`: tests closed-form corrections that let new classes (1-10 pictures)
  compete with old ones (about 500 pictures): a handicap on old classes' scores, a certainty
  correction (distance divided by 1 + 1/n), both, and new classes added to the learned view;
  plus a stream (60 classes, then 8 sessions of 5 new classes with 5 pictures). Complete
  (2026-11-17); journal `EXP-10-JOURNAL.pdf`. Verdict "does not work": the handicap lifts the
  harmonic mean of old and new from 12.4 to 31.5, but old classes fall from 52.2% to 38.1%
  (limit 48.3%). Shifting scores only trades old against new; with all their pictures the new
  classes would reach 47.6% (old 50.3%), so the limit is the card from few pictures. Forgetting
  stays tiny (stream: 57.7% to 54.6% for the first 60 classes). Section 17 proposes EXP-11
  (better cards through better eyes: mirrored pictures, cheap patch-eye upgrades).
- `EXP-11-SHARPER-EYES/`: tests cheap upgrades to the patch eyes (mirrored pictures, two-sided
  stencil answers, 3 x 3 regions, more stencils; descriptions up to 14,400 numbers in compact
  tables) on all three earlier tests at once. Complete (2026-11-19); journal `EXP-11-JOURNAL.pdf`.
  Verdict "helps, but not enough": the selected eyes (mirror + two-sided, 12,800 numbers, 2.2 GB
  GPU) reach 59.3% on 80 classes (from 54.1%; about level with EXP-8's retrained network with the
  same memory, 58.8%, not on the same test pictures) and 36.2% for 20 new classes from 3 pictures
  (from 33.3%), but among all 100 the new classes stay at 17% at the old-class limit (harmonic
  mean 24.8 to 26.0; 3 needed). Mirroring alone gives +3.0 on 80 classes. The gap to "all
  pictures" grows. Section 17 proposes EXP-12 (a closed-form "old or new?" switch; a perfect
  switch would give a harmonic mean of about 45, by arithmetic).
- `EXP-12-OLD-OR-NEW/`: tests a switch that first decides "old class or new class?" from what
  the memory stores (four familiarity signals from the class averages and the shared table, alone
  or with the handicap), then names the picture within that group, on EXP-11's eyes. Complete
  (2026-11-19); journal `EXP-12-JOURNAL.pdf`. Verdict "does not work": a perfect switch would give
  a harmonic mean of 45.1 at 3 pictures (confirmed), but distance to the known cards cannot tell a
  new class's picture from an old one (AUC 0.47 to 0.51, guessing); only the relative signal
  separates partly (0.68), less than the handicap's own score (0.73). Best real switch 26.6 against
  26.0, and that half point comes from where the old-class limit falls, not from the signal.
  EXP-10 to EXP-12 all say the limit is the new card from few pictures. Section 17 proposes EXP-13
  (how many pictures a new class needs; more from each picture by shifted copies).
- `EXP-13-HOW-MANY-PICTURES/`: measures how the score among all 100 classes grows with pictures
  per new class (1 to 500), and whether moved or enlarged copies of each new picture make it count
  for more (no backprop). Complete (2026-11-19); journal `EXP-13-JOURNAL.pdf`. Verdict on copies
  "does not work": moved copies +0.35 at 3 pictures (worth 3.1 pictures), enlarged copies hurt. The
  curve with the handicap: 18 / 26 / 36 / 41 / 44 / 45 at 1 / 3 / 10 / 30 / 100 / 500 pictures; 90%
  is reached at about 27 pictures. Even with all 500 pictures, new classes left out of the learned
  view score only 26.8% among 100 without correction (harmonic mean 36.5), against 52.9 when they are
  in the tables and view (EXP-11's reference). Two limits: the card up to about 30 pictures, then
  the view. Section 17 proposes EXP-14 (when a new class should join the view; a stream with
  consolidation).
- `EXP-14-WHEN-TO-JOIN/`: tests after how many pictures a new class should join the learned view
  (one extra direction per class, computed from its card and the old shared table), and a stream
  where new classes are held apart and join once they have N pictures. Complete (2026-11-19);
  journal `EXP-14-JOURNAL.pdf`. Verdict "helps, but not enough": joining hurts a little below 30
  pictures (-1.1 at 3) and helps a lot above (harmonic mean 45.1 to 58.0 with all pictures; new
  classes 62.8% among 100 with old at 54.0%); rebuilding the shared table too adds only 0.5. In a
  4-session stream, joining at 30 pictures gives 38.8 against 36.5 for never (3.0 needed), and the
  gain needs a handicap re-chosen for joined classes. Section 17 proposes EXP-15 (gradual joining,
  a separate handicap for joined classes, a longer stream).
- `EXP-15-GRADUAL-JOIN/`: tests gradual joining: a new class's question counts with a weight that
  grows with the evidence (by its number of pictures, or measured as novelty against the chance the
  shared table predicts), with a second handicap and an 8-session stream (classes up to 500 pictures).
  Complete (2026-11-19); journal `EXP-15-JOURNAL.pdf`. Verdict "works": the best rule asks the new
  question of every picture in full and turns down only the card's novelty (measured weight, beta 0.5).
  It beats the better of apart and full joining by 2.7 to 3.9 points from 3 to 30 pictures (harmonic
  mean at 3 pictures 28.7, from 26.0), though it is 0.56 under full joining at 1 picture. Stream score
  47.6 against 45.4 for EXP-14's rule and 39.2 for never joining; the second handicap does not matter.
  Section 9 holds a roadmap; section 18 proposes EXP-16 (the whole memory as one stream, from 10
  classes, against rebuilding from scratch).
- `RESEARCH_LITERATURE_LOG.md`: running notes from paper reading by subagents (started
  2026-11-15), with honest tags for how each paper was read.
- `_archive-previous-agent/`: reports and an END_GOAL written by an earlier agent that
  misread the project (it treated brain-inspired and energy as side issues and steered toward
  benchmark-building). Their sourced facts about papers are still usable as reference. Their
  goals, recommendations and action plans are not.

## How to work here

- **Delegate paper reading to Sonnet subagents.** Have them read a paper and return the
  important findings, so the main session does not spend tokens on full papers.
- **Never cite a paper that was not actually opened.** Earlier automated research in this repo
  produced four citations to papers that do not exist and several wrong arXiv IDs. Every claim
  in a report must say whether it was read in full, read from the abstract, or is unverified.
- Experiments must fit the owner's laptop: RTX 3050 with 4 GB VRAM.
- Long runs may go to a free Kaggle GPU instead of the laptop (owner approved 2026-11-13 and
  again 2026-11-15). The Kaggle CLI is already set up on the laptop. Keep the Kaggle job
  private, download its result files into the experiment's `results/` folder, and say in
  Part B that the run was on Kaggle. The method must still be small enough for the laptop.
- **Each experiment lives in its own folder** named `EXP-<number>-<SHORT-NAME>/`. Everything the
  experiment creates or downloads stays inside that folder: the Python environment, packages,
  datasets, pretrained weights, saved features, results and figures. Nothing is installed
  globally or written elsewhere on the machine. Tell the owner the expected disk use before
  downloading, and the measured disk use afterwards.
- Explain in plain language. The owner thinks in concrete examples (cats vs dogs, then digits).
- Work directly on `main`. Do not create git branches (owner, 2026-11-15).
- Do not create extra planning documents unless asked. The experiment journal below is the one
  document every experiment must have.

## Experiment journals (the lab notebook rule)

The experiments form a chain. Someone who reads the journals in order, EXP-1, EXP-2, EXP-3 and
so on, must be able to follow the whole research story without opening any code. Every
experiment has exactly one journal: `EXP-<number>-JOURNAL.pdf` in its folder, rendered from
`JOURNAL.md` in the same folder. A markdown file alone does not count; the PDF must exist.

**The journal is written in two parts, at two different times, in the same PDF.**

1. **Part A is written BEFORE any code is run.** Render the PDF and show it to the owner before
   starting the experiment. Part A contains, in this order:
   - **Page 1: why this experiment.** What the previous experiment found, what failed or stayed
     open in it, and how that leads to this one. Quote the previous experiment's actual numbers.
     If the previous experiment was never run or was abandoned, say so and say why.
   - The question, in one or two plain sentences.
   - What will be done, step by step, with a concrete example (cats vs dogs, then digits).
   - What will be measured and why each measurement matters.
   - How the result will be judged: the pass/fail rule and the expected outcome, written down
     now so they cannot be bent later to fit the numbers.
   - What could go wrong, and the expected time and disk use.
   - A short glossary of every technical word used.
2. **Part B is APPENDED after the experiment finishes.** Do not start a new document. The PDF
   that was 10 pages before the run becomes 15 pages after it. Part B contains: what was
   actually run (including anything that differed from Part A and why), the readings (tables
   and figures), the verdict against the rule written in Part A, what it means for the project,
   what is still unverified, and **what the next experiment should be and why**. That last item
   becomes page 1 of the next journal.

**Rules for both parts.**

- Do not rewrite Part A after results exist. If the plan had to change, record the change in
  Part B. A wrong prediction stays in the journal; it is a result.
- Failed and abandoned experiments are documented too. A negative result gets the same
  write-up as a positive one. An experiment that was designed and not run gets a note saying so.
- Write for a reader who is not an ML researcher. Explain each technical word the first time it
  appears, use everyday comparisons, and put the plain-language meaning before the number.
- Every number in Part B comes from a result file in the experiment folder, and the journal
  names that file.
- Figures are for visual analysis of what happened in the experiment (error and accuracy curves
  and the like). Generate them from the real result files, save each one as an SVG in the
  experiment's `figures/` folder, and then add them to the PDF. No illustrative or made-up
  figures.
- State what is still unverified.
