# EXP-7 · PATCH EYES

**Experiment ID:** EXP-7 · **Part A written:** 2026-11-16, before any code was run · **Hardware:** free Kaggle GPU (the method must still fit the RTX 3050 laptop, 4 GB) · **Status:** PLANNED, NOT RUN YET

This journal has two parts. **Part A** (this part) explains the experiment before it starts.
**Part B** will be added to the end of this same PDF when the experiment finishes. Part A
will not be edited after results exist, except for the status word in the line above.

## 1. Why this experiment

### What happened before

The project thinks of a learning system as two parts: the **eyes**, which turn a picture into
a list of numbers describing it, and the **memory**, which keeps one **memory card** per class
and answers with the best-matching card.

- **EXP-1** was a plan only. It was never run.
- **EXP-2**: the memory works when the eyes are good. The eyes are the bottleneck.
- **EXP-3 to EXP-5**: **grow and lock** (add units for each new group of 10 classes, lock the
  old ones) keeps old knowledge, and with the card fix R7 reaches 39.6% on 80 classes. Extra
  training jobs for the new units did not help.
- **EXP-6** asked why grown eyes stay about 12 points under retraining. Its readings, exam
  half, 5 seeds (from `EXP-6-WHAT-LIMITS-THE-EYES/results/summary.json`):

| | Open-book score, all 80 classes | 20 never-seen classes from 3 pictures | Training time and GPU energy per run |
|---|---|---|---|
| Frozen eyes (trained on the first 10 classes only) | 33.2% | 25.6% | 26 s, 0.27 Wh |
| Grow and lock (run A) | 41.0% (39.6% on the real test with R7) | 29.7% | 131 s, 1.36 Wh |
| Grow and lock, new units may see all old pictures (diagnosis) | 43.7% | 31.1% | 737 s, 7.68 Wh |
| Grow and lock, double units | 42.7% (41.7% with R7) | 30.9% | 238 s, 2.48 Wh |
| Retrain everything (ceiling) | 53.2% | 38.5% | 202 s, 2.12 Wh |

### What EXP-6 found

Of the 12.2 points between grow and lock and retraining:

- only **22%** comes from the new units seeing just 10 classes ("narrow view");
- **78%** comes from **locking and building in steps**: units are trained on very little,
  then frozen for good, and everything later is built on top of them;
- doubling the units changes nothing about the gap.

So the thing that protects old knowledge (locking) is also what costs the accuracy, as long
as the eyes are trained by backpropagation on a few classes at a time.

### What stayed open, and what the owner asked

The owner asked two things on 2026-11-15: are we still using backpropagation (yes, for the
eyes), and why must each picture be looked at 30 times (EXP-5: one look gives 16.5%).

Reading done during EXP-6 (`RESEARCH_LITERATURE_LOG.md`, section 6) found a kind of eyes
that has neither problem: **fixed picture patches**. Reported results on 10-class photos
(CIFAR-10) with only a final classifier trained: 77.9% with 1,600 patch detectors (Coates,
Lee, Ng, AISTATS 2011, whole paper read as scraped text) and 82.5% with 2,048 randomly picked
patches (Thiry et al., arXiv:2101.07528, passages read). No result was found for 100 classes,
for class-by-class learning, or with memory cards instead of a trained classifier.

### So EXP-7 asks

**Can eyes built from picture patches, with no backpropagation and one look at each
picture, match or beat grow and lock on the same 80-class test?** Because such eyes are
never fitted to particular classes, locking them should cost nothing.

# Part A: the plan (written before the run)

## 2. The idea in plain words

Take the pictures of the first 10 classes. Cut out thousands of small 6-by-6-pixel squares
at random: a bit of fur, an edge, a patch of sky. Keep 1,600 of them. These are the
**detectors**. Nothing is trained; they are simply kept.

To describe any picture, slide a 6-by-6 window over it and ask, for each detector, "how
much does this spot look like you?" Add up the answers over each quarter of the picture.
The result is a list of 6,400 numbers (1,600 detectors times 4 quarters). That list is the
picture's description, and it goes to the memory as before.

**Cats and digits example.** Round 1 is cats against dogs. The detectors cut from those
pictures are bits of fur, ears, eyes, grass, carpet. Round 2 brings the digits. No detector
was made for digits, but a digit is still made of edges, corners and curves, and many
detectors respond to those. Nothing in the eyes has to change for the digits, so nothing
about cats and dogs can be damaged.

One preparation step matters a lot according to the papers: **whitening**. Neighbouring
pixels are nearly the same, so raw patches all look alike. Whitening removes what is common
to all patches and keeps what makes each one different. It is a single calculation from the
first group's patches (worth 7 to 8 points in Coates 2011).

## 3. What is run

### The eyes

| | Name | Where the detectors come from | Purpose |
|---|---|---|---|
| E1 | **Fixed patch eyes** (the tested idea) | 1,600 patches from the first group's pictures only; whitening from the same pictures; never changed | The real method: one look, no backpropagation, nothing ever changes |
| E1-small | Same with 400 detectors | as E1 | Does size matter? |
| E1-km | Same, but the 1,600 detectors are found by grouping similar patches (K-means, 10 rounds) | as E1 | The original 2011 recipe; is random picking as good? |
| E2 | **Growing patch eyes** | starts with 200 detectors from the first group, adds 200 from each new group, old ones locked; ends with 1,600 | Does it help to add detectors made from the new classes? Old cards lack the new units, so R7 fills them in |
| E3 | **All-classes patch eyes** (diagnosis only) | 1,600 patches and the whitening from the pictures of all 80 classes | Does it matter which classes the patches came from? Not allowed in the real method |

**How a detector answers.** Two published rules are tried on E1:

- **Triangle** (Coates 2011): "how much closer is this spot to me than to the average
  detector". Each answer depends on all detectors, so it cannot be used when detectors are
  added later.
- **Threshold**: "how well does this spot match me, above a fixed level". Each detector
  answers alone, so adding detectors never changes old answers. Three levels are tried:
  0, 0.1 and 0.25. Growing eyes (E2) use the threshold rule at 0.1.

### The memory

All of these need one look per picture, no backpropagation, and no stored pictures.

| | Name | What is kept | From |
|---|---|---|---|
| M-cos | Class cards, compared by angle (as in all earlier experiments) | one average per class | EXP-2 to EXP-6 |
| M-sqrt | The same after taking the square root of every number in the description | one average per class | FeCAM, arXiv:2309.14062 (passages read) |
| M-lda | Cards plus **one shared table of how the numbers vary together** (a covariance). Each picture is judged with that table as the ruler. The table is a running sum, updated as groups arrive | one average per class + one shared table | Deep Streaming LDA, arXiv:1909.01520 (passages read) |
| M-ridge | A straight-line classifier solved in one calculation from two running sums. Gives exactly the same answer as if all pictures were present at once | two running tables | ACIL, arXiv:2205.14922 (passages read) |

M-lda and M-ridge each have one setting (how strongly the table is smoothed); three values
are tried (0.01, 0.1, 1 times the average spread). Both use square-rooted descriptions.

Because fixed eyes never change, cards written at stage 1 are still exactly right at stage
8. There is no "open-book" gap for E1: the real test and the open-book test are the same thing.

### The test

The same as EXP-3 to EXP-6: photos (CIFAR-100), 80 classes arriving in 8 stages of 10, old
pictures thrown away after each stage, 20 surprise classes learned from 3 pictures each, 5
seeds with the same class orders and the same practice and exam halves as before. So the
numbers can be put directly next to EXP-6's. The backprop-trained reference points are not
rebuilt; EXP-6's are used (they came out the same three times in a row).

**Settings fixed now.**

| Setting | Value |
|---|---|
| Patch size, step | 6 by 6 pixels, every position (the papers' best) |
| Patch preparation | subtract the patch's mean, divide by its spread (constant 10 on the 0 to 255 scale), then whitening with constant 0.1 |
| Patches used to compute the whitening and to pick detectors | 200,000 random patches from the allowed pictures |
| Pooling | sum over the four quarters of the picture |
| Selecting one combination for the verdict | among E1 with 1,600 random detectors: the answer rule and the memory with the highest score on all 80 classes on the practice half, among those that pass step 2 on the practice half |
| Seeds | 5 |

## 4. What is measured

| | Measurement | Plain meaning | Why it matters |
|---|---|---|---|
| P1 | Accuracy on **all 80 classes** after stage 8 | The overall score | To compare with 39.6% (grow and lock), 33.2% (frozen backprop eyes), 53.2% (retrain) |
| P2 | First group with only its own 10 cards allowed, after every stage | Real forgetting | With M-cos it cannot change at all. With M-lda and M-ridge the shared table changes, so it must be checked |
| P3 | 3-picture test on the 20 surprise classes | How general the eyes are | Compare with 29.7% (grow and lock) and 38.5% (retrain) |
| P4 | Score after every stage, score per group | The usual curves | To see whether early or late classes are favoured |
| P5 | Time and GPU energy to build the eyes, describe all training pictures once, and fill the memory | The whole cost of learning 80 classes | Compare with 131 s and 1.36 Wh for grow and lock |
| P6 | Implementation check on CIFAR-10 | Does our code reproduce the published level? | Coates 2011 reports 77.9% with 1,600 detectors and a trained classifier |

## 5. How the result will be judged (written before the run)

The verdict uses E1 with 1,600 random detectors, the one combination selected on the
practice half, scored on the exam half after stage 8, mean of 5 seeds.

**Step 0: is the code right?** On CIFAR-10 (all 50,000 training pictures, 1,600 detectors
found by K-means, triangle rule, M-ridge as the classifier) the test accuracy must be at
least **70%**. The paper has 77.9% with a different final classifier; under 70% means a
mistake, and it is fixed before any verdict.

**Step 1: the score, against EXP-6's numbers.**

| Score on all 80 classes | Verdict |
|---|---|
| under 33.2% | **Fails**: worse than backprop eyes that stopped after the first group |
| 33.2% to 37.6% | **Weaker** than grow and lock |
| 37.6% to 43.0% | **Matches grow and lock** (within 2 points of 39.6% or better) with no backpropagation and one look |
| 43.0% to 46.4% | **Partly closes the gap** to retraining (25% to 50% of 53.2 − 39.6) |
| 46.4% or more | **Works**: closes at least half of the gap |

**Step 2: is old knowledge kept?** P2 for the first group after stage 8 must be no more
than 3 points below its value after stage 1. A combination that fails this cannot be selected.

**Also reported, without a verdict:** each memory on its own; 400 against 1,600 detectors;
random picking against K-means; growing eyes against fixed eyes; whether patches from all
classes (E3) beat patches from the first group by 2 points or more; the surprise test; cost.

**What I expect, written down now.**

- Step 0 passes, at about 74%.
- With plain cards (M-cos) patch eyes will be weak: about **27%**. Square root lifts that to
  about 30%. These descriptions are long and their numbers move together, which plain
  averages cannot use.
- M-lda or M-ridge will be selected and reach about **44%**: "partly closes the gap", near
  its lower border. Most of the gain comes from the memory upgrade, not from the eyes alone.
- Forgetting (step 2) will pass, under 2 points.
- Patches from all classes (E3) will be within 1 point of patches from the first group, as
  the Thiry paper suggests. Growing eyes with R7 will be about 2 points below fixed eyes.
- K-means and random detectors within 1 point. 400 detectors about 5 points below 1,600.
- Surprise test about 24% with plain cards, below grow and lock's 29.7%.
- Cost: under 60 s for everything, less than half of grow and lock's 131 s.

My main predictions were wrong in EXP-4, EXP-5 and EXP-6. If any of this is wrong, it stays
in the journal as written.

## 6. What carries over beyond pictures

| Part | Depends on pictures? |
|---|---|
| Cutting 6-by-6 patches, sliding windows, quarters of a picture | **yes** |
| The idea behind them: detectors that are samples of the data, not trained on labels | no. For text the samples would be short stretches of words in context |
| Whitening (remove what is common to everything) | no. It works on any lists of numbers |
| M-lda and M-ridge: running tables that give the all-at-once answer without old data | no. They work on any descriptions |
| The finding from EXP-6 that motivates this: do not fit units to a few classes and then freeze them | to be measured for language |

## 7. What could go wrong

- **Plain cards may be the wrong memory for these eyes.** That is expected, and is why
  M-lda and M-ridge are in the run. If only they work, the result is as much about the
  memory as about the eyes, and Part B will say so.
- **Fair comparison.** M-lda and M-ridge were never tried on the backprop eyes. If they win
  here, the same memory on grow-and-lock eyes might also gain. That needs a follow-up before
  claiming patch eyes are better eyes.
- **The description is long**: 6,400 numbers per picture against 576. The shared table has
  6,400 by 6,400 entries (about 160 MB). It fits, but answering costs more than with cards.
- **The settings come from 10-class papers** and are not tuned for 80 classes. A different
  patch size or whitening constant might do better. Three threshold levels and three
  smoothing values are tried; nothing else.
- **Detectors from 10 classes may miss things** that later classes need. E3 checks this.
- **Not a class-by-class method in the papers.** Using it this way is new here; no
  published number says what to expect.
- **5 seeds, one dataset, images only.** The reference numbers are quoted from EXP-6, not rebuilt.

## 8. Time and disk

| | Expected |
|---|---|
| Run time | under 1.5 hours on a free Kaggle GPU for all 5 seeds and the CIFAR-10 check. There is no training loop; the time goes into describing pictures and solving the tables. This is an estimate with no earlier run to base it on |
| Disk in this folder | 0.15 GB measured for the Python environment (figures and PDF), plus under 0.05 GB of results and figures |
| Downloads to the laptop | none. CIFAR-100 and CIFAR-10 are downloaded on Kaggle's machine. A short code check on the laptop borrows EXP-2's environment and CIFAR-100 without changing them |
| GPU memory | expected under 6 GB on Kaggle as written (descriptions of all pictures are held at once). On the 4 GB laptop the same code would need to hold them in ordinary memory instead; peak use will be reported |

## 9. Words used in this journal

| Word | Meaning here |
|---|---|
| Class | One kind of thing to recognise |
| Eyes (features) | The part that turns a picture into a list of numbers |
| Patch | A small 6-by-6-pixel square cut from a picture |
| Detector | A kept patch. It answers "how much does this spot look like me?" |
| Whitening | A one-off calculation that removes what all patches have in common |
| Backpropagation | The usual way to train a network: measure the error and push corrections back through all layers, many times |
| Memory card | The average description of one class |
| Covariance (shared table) | A table recording which numbers of the description tend to rise and fall together |
| R7 | Guess the missing part of an old, short card from the part it has |
| Open-book | Rewriting all cards from old pictures; shows how good the eyes are |
| Practice half, exam half | The two halves of the test pictures: one to choose, one to grade |
| Surprise classes | 20 classes never used to build anything, learned from 3 pictures each |
| Seed | One repeat with a different random shuffle |

<div class="pagebreak"></div>

# Part B: the readings (to be added after the run)

**Not run yet.** When the experiment finishes, this part will be filled in, in the same PDF:

1. What was actually run, and anything that differed from Part A
2. The readings: tables and figures made from the result files
3. The verdict, using the rule in section 5
4. Was the expectation in section 5 right or wrong
5. What it means for the project
6. What is still unverified
7. What the next experiment should be, and why
