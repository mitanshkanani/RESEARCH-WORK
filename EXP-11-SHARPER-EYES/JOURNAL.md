# EXP-11 · SHARPER EYES FOR BETTER CARDS

**Experiment ID:** EXP-11 · **Part A written:** 2026-11-19, before any code was run · **Hardware:** free Kaggle GPU (the method must still fit the RTX 3050 laptop, 4 GB) · **Status:** PLANNED. Part A only; nothing has been run

This journal has two parts. **Part A** (this part) explains the experiment before it starts.
**Part B** will be added to the end of this same PDF when the experiment finishes. Part A
will not be edited after results exist, except for the status line above.

## 1. Why this experiment

### What happened before

The project thinks of a learning system as two parts: the **eyes**, which turn a picture into
a list of numbers describing it, and the **memory**, which stores what was learned about
each class and gives the answer.

- **EXP-1** was a plan only. It was never run.
- **EXP-2 to EXP-6** used eyes trained by backpropagation. The best system that keeps no old
  pictures (grow and lock) reached 39.6% on the 80-class photo test.
- **EXP-7 and EXP-8** built eyes with no training at all (1,600 fixed picture patches) and a
  memory of class averages (**cards**) plus one **shared table**. On 80 classes they reach
  **55.2%**, 6 points under the best network retrained on all the pictures (**61.1%**), and
  forget nothing.
- **EXP-9** computed a **learned view** from the cards and the shared table (the directions
  along which known classes differ most) and let new cards **borrow** from their 2 nearest
  known cards. A new class learned from **3 pictures** then scores **33.5%** among 20 new
  classes (retrained network: 38.5%).
- **EXP-10** put those new classes among the 80 old ones. Without a correction they are
  drowned out (new 7.0%, old 52.2%, harmonic mean 12.4). Every correction tried (a handicap
  on old classes, a certainty correction) only moved answers from old classes to new ones.
  Readings, 5 seeds, from `EXP-10-NEW-AMONG-OLD/results/summary.json`:

| Patch eyes, 100 classes, 3 pictures per new class | Old classes | New classes | Harmonic mean |
|---|---|---|---|
| No correction | 52.2 | 7.0 | 12.4 |
| Handicap at the rehearsal's choice (old classes fell too far) | 38.1 | 26.8 | 31.5 |
| Handicap at the strongest value that keeps old within 5 points (read off the test, a diagnosis only) | 48.3 | 16.5 | 24.6 |
| **New classes learned from all their 500 pictures** (reference) | **50.3** | **47.6** | **48.9** |

### What stayed open

The last row is the key. The same memory and view hold 100 classes well when every class is
learned from all its pictures. From 3 pictures, the new classes are too badly placed, and no
shift of scores fixes that. EXP-10's verdict was "does not work", and its section 17 says
what is left goes through the **eyes**, and through **how much each picture gives**:

- the gap on 80 classes (55.2% against 61.1%);
- the gap on 20 new classes from 3 pictures (33.5% against 38.5%);
- the gap between cards from 3 pictures and cards from all pictures (harmonic mean 24.6 within
  the old-class limit, against 48.9).

EXP-10 also found a flaw in its own rule: it chose the handicap by the best harmonic mean
but judged it with a limit on how far old classes may fall. This time the limit is part of
the choosing rule, written down below.

### What the papers say

The literature log, section 7 (all read as passages), lists cheap ways to make this kind of
eyes stronger, with no backpropagation:

| Upgrade | Source and what it reports |
|---|---|
| **Mirrored copies** of each picture | Prabhu et al., RanDumb, arXiv:2402.08823: CIFAR-100, 26.9% without, 28.6% with mirrored copies when filling the tables |
| **Positive and negative parts** of each detector's answer | Coates and Ng, ICML 2011: splitting the answer "always improved" |
| **Finer regions** than 4 quarters | Jia, Huang, Darrell (CVPR 2012 workshop version): with 200 detectors, 70.2% with quarters, 72.2% with a 4 × 4 grid (CIFAR-10) |
| **More detectors** | Thiry et al., arXiv:2101.07528: 79% with 512, 85.6% with 10,000 (CIFAR-10, a different pipeline) |

None of these numbers was measured with our memory, our class order or our few-picture tests.

### So EXP-11 asks

**Do sharper eyes, still with no backpropagation, make better cards: on 80 classes, on new
classes from 3 pictures, and on new classes among old ones? And do they fit the 4 GB laptop?**

# Part A: the plan (written before the run)

## 2. The idea in plain words

**Cats and digits example.** The eyes look at a picture through 1,600 small stencils (6 × 6
pixel patterns cut from real photos), slid over every place in the picture. For each stencil
they write down how strongly it matched in each quarter of the picture. That list is the
description.

The four upgrades change only how the description is written:

1. **Mirror.** A cat looking left is still a cat looking right. Every picture is also
   described flipped left to right. When learning, both copies go into the tables (the memory
   sees twice as many examples). When answering, and when making a new card, the picture's
   description is the average of itself and its mirror image.
2. **Two-sided answers.** Today a stencil only reports "this matches". It can also report "this
   is the exact opposite" (a dark edge where the stencil has a light one), as a separate number.
3. **Finer regions.** Instead of 4 quarters, the picture is cut into a 3 × 3 grid of 9 regions,
   so the description records more precisely where things are (ears at the top, paws at the
   bottom).
4. **More stencils.** 3,200 instead of 1,600.

Each upgrade makes the description longer, and the shared table grows with the square of the
length. A table for a 14,400-number description takes 1.6 GB in full precision. So the
longest description allowed here is **14,400 numbers**, and the tables are written more
compactly (section 3).

## 3. What is run

### The six versions of the eyes

| | Eyes | Stencils | Sides | Regions | Mirror | Description length |
|---|---|---|---|---|---|---|
| A0 | **EXP-10's eyes, unchanged** | 1,600 | one | 4 | no | 6,400 |
| A1 | + mirror | 1,600 | one | 4 | yes | 6,400 |
| A2 | + mirror + two-sided | 1,600 | two | 4 | yes | 12,800 |
| A3 | + mirror + finer regions | 1,600 | one | 9 | yes | 14,400 |
| A4 | + mirror + more stencils | 3,200 | one | 4 | yes | 12,800 |
| A5 | + mirror + all three, fewer stencils | 800 | two | 9 | yes | 14,400 |

All stencils are cut from stage 1's pictures only, as before, and the same stencils are
shared by all versions (A5 uses the first 800 of A0's 1,600; A4 adds 1,600 more). Each
stencil's answer is "how well it matches, minus 0.1, or zero" as in EXP-7; for two-sided
eyes the opposite match gets the same rule. Matches are added up per region, then square-rooted.

### The memory: EXP-10's, with two changes for size

The memory is the one from EXP-9 and EXP-10: running tables (class sums and one shared table,
filled stage by stage, no old pictures kept), the learned view (all directions), and
borrowing from the 2 nearest known cards with a weight of 1 picture. Two changes are needed
to fit the longer descriptions in 4 GB:

- **The shared table is kept in single precision** (half the memory), as sums around a fixed
  reference point (stage 1's average description), so that the small differences that matter
  are not lost in rounding. It is turned into full precision only while the view is computed.
- **The view is computed in place**, inside one full-precision working table, instead of
  making several copies.

A0 checks these changes: it must reproduce EXP-10's numbers (section 5, step 0).

The **smoothing** of the shared table is tried at 0.01, 0.03, 0.1, 0.3 and 1 (times the
average of the table's diagonal), because longer descriptions may need more of it. EXP-8's
choice for A0 was 0.01, EXP-9's was 0.1.

### The three tests, all in one run

The same 80 known classes in 8 stages of 10, the same 20 surprise classes and 5 seeds as
EXP-8 to EXP-10. All test pictures of a class are used.

| | Test | Same as | How a picture is named |
|---|---|---|---|
| T1 | **80 classes**, learned from all their pictures | EXP-8 (55.2%) | nearest card by distance through the shared table (the EXP-8 memory) |
| T2 | **20 new classes from 3 pictures, among themselves** | EXP-9 (33.5%) | learned view, by angle, with borrowing (EXP-9's way) |
| T3 | **the 20 new classes (3 pictures) among all 100** | EXP-10 | as T2, plus a correction: C1 handicap, or C3 certainty plus handicap |

T2 and T3 are also run with 1 and 10 pictures (20 random draws each), as in EXP-10. Also run:
C0 (no correction) and C2 (certainty, no setting), for reference, and the **reference with
all pictures** (the new classes learned from all their 500 pictures, no correction).
EXP-10's correction C4 (new classes reshape the view) is dropped, since it lost 3.6 points
there; so is EXP-10's stream and the frozen backprop eyes, which are not part of this question.

### How settings and the selected version are chosen: the rehearsal

As in EXP-9 and EXP-10: the memory after stage 6 knows 60 classes, and the 20 classes of
stages 7 and 8 play the new classes, learned from the same number of pictures. No surprise
class is used to choose anything. For each version of the eyes, from the rehearsal only:

- **T1 smoothing:** the one with the best 60-class accuracy.
- **T2 smoothing:** the one with the best accuracy of the pretend-new classes among themselves.
- **T3 setting:** among all smoothings, both corrections (C1, C3) and every handicap value,
  keep only those where the old classes stay within 5 points of no correction (C0 at the same
  smoothing) in the rehearsal; of those, take the highest harmonic mean.
- **The selected version** of the eyes is the one with the highest T3 harmonic mean in the
  rehearsal at 3 pictures, among the versions whose peak GPU memory is 3.5 GB or less.

All values of every setting are run and logged; the choosing is done afterwards from the
rehearsal rows, by these rules, in `src/figures.py`.

| Setting | Value |
|---|---|
| Harmonic mean | H = 2 × old × new / (old + new), old and new each the accuracy among all 100 |
| Handicap grids | as EXP-10: C1 from −0.2 to 0.5 in steps of 0.01; C3 from −0.5 to 1.0 times the number of directions, steps of 0.02 |
| Pictures per new class | 1, 3, 10 (20 draws each) |
| Seeds | 5 |

## 4. What is measured

| | Measurement | Why it matters |
|---|---|---|
| M1 | T1: 80-class accuracy, each version, each smoothing | Does sharper eyes close the 6-point gap to retraining? |
| M2 | T2: new among themselves at 1, 3, 10 pictures | Does it close the 5-point gap from 3 pictures? |
| M3 | T3: old, new and harmonic mean with each correction; and the full old-against-new curve | The EXP-10 problem |
| M4 | The reference with all pictures, each version | Is the gap between 3 pictures and all pictures getting smaller, or are both rising together? |
| M5 | Rehearsal against test, for the choice of version and settings | Can the choosing be trusted? |
| M6 | Time, GPU energy, peak GPU memory, peak ordinary memory | Cost, and whether it fits the laptop |

## 5. How the result will be judged (written before the run)

All scores are means of 5 seeds, on the test, at the settings chosen in the rehearsal.
**"Gain" means the selected version minus A0, both measured in this run with the same rules.**

**Step 0: are the earlier numbers rebuilt?** A0 at smoothing 0.1, no correction, 3 pictures:
old within 0.5 of 52.2% and new within 0.5 of 7.0% (EXP-10); new among themselves within 0.5
of 33.5% (EXP-9). A0's T1 at smoothing 0.01 within 1 point of 55.2% (EXP-8 used only half of
the test pictures, so a small difference is expected). If step 0 fails, the compact tables
changed the system, and every comparison with earlier experiments is marked unreliable.

**Step 1: the main verdict.** The selected version must reach:

| | Condition |
|---|---|
| (a) | T1 gain of 2.0 points or more |
| (b) | T2 gain of 2.0 points or more (3 pictures) |
| (c) | T3 harmonic-mean gain of 3.0 or more (3 pictures), with the test's old-class accuracy no more than 6 points under the same version's no-correction old-class accuracy (the rule allows 5 in the rehearsal; 1 point of slack for the move from rehearsal to test) |
| (d) | peak GPU memory 3.5 GB or less |

| Reading | Verdict |
|---|---|
| (a), (b), (c) and (d) all hold | **works** |
| (d) holds, and one or two of (a), (b), (c) | **helps, but not enough** |
| anything else | **does not work** |

**Step 2: which upgrade does the work?** On each test: A1 − A0 (mirror), A2 − A1
(two-sided), A3 − A1 (finer regions), A4 − A1 (more stencils), A5 − A1 (all three within the
same length). An upgrade counts on a test if it adds 1.0 point or more.

**Step 3: does the gap to "all pictures" shrink?** The ratio of the T3 harmonic mean (3
pictures) to the reference's harmonic mean (all pictures), for the selected version against
A0. A rise of 0.05 or more counts as **the gap shrinks**; otherwise the eyes lift both
together and the few-picture problem stays.

**Step 4: can the rehearsal be trusted?** If the selected version's test harmonic mean is
within 1.0 of the best version's test harmonic mean: **yes**.

**What I expect, written down now.**

- Step 0 passes.
- Mirror (A1) adds about 1 point on T1 and 1.5 on T2. Each of the other upgrades adds 1 to 1.5
  on T1 and under 1 on T2. A5 (all three) is the best version and is selected.
- Selected against A0: T1 +3 (to about 58%), T2 +2 (about 35.5%), T3 harmonic mean +2.
- So the verdict would be **"helps, but not enough"**: sharper eyes help on all pictures more
  than on few, and the gap to "all pictures" does not shrink (step 3: no).
- Peak GPU memory about 3 GB for the longest descriptions.
- The rehearsal picks well.

My main prediction was wrong in EXP-9 and in EXP-10. If any of this is wrong, it stays in
the journal as written.

## 6. What carries over beyond pictures

| Part | Depends on pictures? |
|---|---|
| Mirror, finer regions, two-sided stencils, more stencils | **yes**. These are all image-specific |
| The idea behind mirroring: give the memory more than one version of each example | no, but the language version (rephrasing) is not part of this plan |
| The compact tables (single precision around a fixed reference, view computed in place) | **no**. They halve the memory cost of the shared table for any list of numbers |
| The choosing rule with an old-class limit | no |

## 7. What could go wrong

- **Longer descriptions need a bigger table, learned from the same pictures.** With 14,400
  numbers and 80,000 examples (40,000 pictures and their mirrors), the table is less certain.
  Smoothing is the guard; the rehearsal chooses it.
- **Mirror copies are not new pictures.** They count twice in the tables but carry less than
  twice the information. The certainty correction (C3) counts real pictures only.
- **The compact tables may round away something.** Step 0 checks it.
- **Finer regions hurt when the object moves:** a cat in the top-left region in one picture
  and the centre in another looks less alike. More smoothing, or fewer stencils, may be needed.
- **A5 has half the stencils of the others.** If it loses, it may be the fewer stencils, not
  the combination.
- **The rehearsal picks smoothing on 60 classes and the test uses 80 or 100.**
- **5 seeds, one dataset, images only. The surprise classes are fixed.**

## 8. Time and disk

| | Expected |
|---|---|
| Run time | about 2 hours on a free Kaggle GPU: the shared table of the longest descriptions takes about 11 seconds to turn into a view on the laptop's GPU (measured once in a test of that step), done 15 times per version and seed. The dataset is read from the private Kaggle copy made on 2026-11-17, so no download wait. About 24 of the roughly 30 free weekly hours are used |
| Disk in this folder | about 0.15 GB for this experiment's own Python environment (figures and PDF), under 0.05 GB of results and figures |
| Downloads to the laptop | the small Python environment only (approved by the owner). A short code check on the laptop borrows EXP-2's environment and CIFAR-100 without changing them |
| GPU memory | about 3 GB for the longest descriptions (1.6 GB working table + 0.8 GB compact table + pictures). The peak is measured for every version |

## 9. Words used in this journal

| Word | Meaning here |
|---|---|
| Class | One kind of thing to recognise |
| Eyes (features) | The part that turns a picture into a list of numbers (the description) |
| Stencil (detector) | A small 6 × 6 pattern cut from a real photo; the eyes record how well it matches each place in a picture |
| Region | A part of the picture over which the matches of one stencil are added up (4 quarters, or a 3 × 3 grid) |
| Two-sided | Each stencil also reports how well its opposite (light and dark swapped) matches, as a separate number |
| Mirror | The picture flipped left to right |
| Memory | The part that stores what was learned about each class and gives the answer |
| Card | The average description of one class |
| Shared table (covariance) | A table of which numbers of a description rise and fall together inside a class, kept as a running sum |
| Single / full precision | Numbers kept with about 7 or about 16 correct digits; single takes half the memory |
| Smoothing | How strongly the shared table is evened out before use. More smoothing trusts the table less |
| Learned view | The directions along which the known classes differ most, computed in one step from the cards and the shared table (EXP-9) |
| Borrowing | Mixing the cards of the 2 nearest known classes into a new class's card (EXP-9) |
| Handicap | One fixed amount subtracted from every old class's score before the answer is picked (EXP-10) |
| Certainty correction | Dividing a class's distance by (1 + 1/n), n the number of pictures it was learned from (EXP-10) |
| Harmonic mean | One score from two: 2 × a × b / (a + b). It is high only if both are high |
| Rehearsal | Choosing settings by pretending the most recent known classes are new. Uses no test class |
| Backpropagation | The usual way to train a network: measure the error and push corrections back through all layers, many times |
| Seed | One repeat with a different random shuffle |

<div class="pagebreak"></div>
