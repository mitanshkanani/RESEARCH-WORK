# EXP-13 · HOW MANY PICTURES DOES A NEW CLASS NEED?

**Experiment ID:** EXP-13 · **Part A written:** 2026-11-19, before any code was run · **Hardware:** free Kaggle GPU (the method must still fit the RTX 3050 laptop, 4 GB) · **Status:** PLANNED. Part A only; nothing has been run

This journal has two parts. **Part A** (this part) explains the experiment before it starts.
**Part B** will be added to the end of this same PDF when the experiment finishes. Part A
will not be edited after results exist, except for the status line above.

## 1. Why this experiment

### What happened before

The project thinks of a learning system as two parts: the **eyes**, which turn a picture into
a list of numbers describing it, and the **memory**, which stores what was learned about each
class and gives the answer.

- **EXP-1** was a plan only. It was never run.
- **EXP-2 to EXP-6** used eyes trained by backpropagation (best without old pictures: 39.6%
  on 80 classes).
- **EXP-7 and EXP-8**: eyes with no training (fixed picture patches) and a memory of class
  averages (**cards**) plus one **shared table**: 55.2% on 80 classes, no forgetting.
- **EXP-9**: a **learned view** and **borrowing** from the 2 nearest known cards: a new class
  from 3 pictures scores 33.5% among 20 new classes.
- **EXP-10**: among all 100 classes the new ones are drowned out; a **handicap** on old classes
  only trades old answers for new ones.
- **EXP-11**: sharper eyes (mirror, two-sided stencils): 59.3% on 80 classes, 36.2% for new
  classes among themselves, but among all 100 the harmonic mean of old and new rises only from
  24.8 to 26.0.
- **EXP-12**: a switch that first decides "old class or new class?" from how familiar a picture
  looks. Readings from `EXP-12-OLD-OR-NEW/results/summary.json`, test, 5 seeds:

| EXP-12, 3 pictures per new class | Old | New | Harmonic mean |
|---|---|---|---|
| Handicap (EXP-11's way) | 52.5 | 17.3 | 26.0 |
| Best familiarity switch (selected by the rule) | 52.0 | 17.9 | 26.6 |
| **Perfect switch** (true group given; a ceiling) | **59.4** | **36.4** | **45.1** |

Distance to the known cards told new pictures from old ones no better than guessing (0.47 to
0.51 on a scale where 0.5 is guessing and 1 is perfect). Only the handicap's own score, which
compares the picture with the **new cards**, separates the groups (0.69, 0.73, 0.76 at 1, 3, 10
pictures). Verdict "does not work".

### What stayed open

EXP-10, EXP-11 and EXP-12 all point to the same limit: **a card made from 3 pictures says too
little.** Shifting scores (EXP-10), sharper eyes (EXP-11) and a familiarity switch (EXP-12) each
failed for that reason. Two things are not known:

1. **How many pictures are enough?** At 3 pictures the harmonic mean is 26.0; EXP-11's reference
   with all 500 pictures (new classes also added to the tables and the view) was 52.9. Nobody has
   measured the curve in between with this memory, nor whether the handicap is still needed once
   cards are good.
2. **Can each picture count for more?** EXP-11 found the mirror image worth 3 points on known
   classes. Shifted and enlarged copies of each new picture might make a 3-picture card behave
   like a card from more pictures.

### What the papers say

One Sonnet subagent, 2026-11-19 (literature log, section 11).

| Paper | Tag | What it reports |
|---|---|---|
| Yang, Liu, Xu, "Free Lunch for Few-shot Learning: Distribution Calibration", ICLR 2021, arXiv:2101.06395 | passages | A new class's average is mixed with the averages of its 2 nearest known classes, **as our borrowing already does**, and its spread is taken from them. A square-root transform of the features helps as much as the mixing at 1 picture (miniImageNet 5-way: 56.4 without either, 64.3 with the transform only, 63.7 with the mixing only, 68.6 with both). Our eyes already take a square root of every number (since EXP-7) |
| Bai et al., "Can One-Shot Test-Time Data Augmentation Help with Generalization?", arXiv:2602.00114 | model-written summary of the page only; unverified | Copies of a single picture (moved, noised, then cleaned by a **trained** model) gave +4.8 to +8.2 points for a class-average classifier at 1 picture (miniImageNet). The trained cleaning step makes it a different method |

No paper was found that tests plain shifted or enlarged copies averaged into a class average on
fixed eyes, or that reports a curve from 1 to 500 pictures for new classes among old ones.

### So EXP-13 asks

**How quickly do new classes catch up with old ones as they get more pictures? And can copies
of each new picture (moved, enlarged), made with no training at all, make 3 pictures count like
more?**

# Part A: the plan (written before the run)

## 2. The idea in plain words

**Cats and digits example.** The memory knows 80 classes from 500 pictures each. A fox arrives
with 3 pictures. Its card (the average of the 3 descriptions) is shaky: one picture shows a
fox's head close up, another a whole fox far away. The "dog" card, from 500 pictures, is steady.

**Part 1, the curve.** Give the fox 1, 3, 10, 30, 100 pictures, and then all 500, and see how
the score among all 100 classes grows, with and without the handicap. This says how good a card
must be before new and old classes compete fairly.

**Part 2, copies.** A person who has seen one photo of a fox can still recognise a fox a little
to the left, or a little closer. The eyes cannot: their description changes when the picture
moves. So each of the fox's pictures is also described:

- **moved**: 8 copies shifted by 3 pixels (left, right, up, down and the diagonals);
- **enlarged**: 5 copies in which a 28 × 28 part (each corner and the centre) is enlarged back to
  the full 32 × 32, like stepping closer.

The fox's card then uses, for each picture, the average of its description and its copies'
descriptions. Only the new pictures are copied; the old classes, the tables and the test
pictures stay as they were. Nothing is trained.

## 3. What is run

**Eyes and memory: EXP-11's version A2 and EXP-12's memory, unchanged.** 1,600 stencils,
two-sided, 4 quarters, mirrored (12,800 numbers per picture); compact tables; learned view of
the old classes with all directions; new cards by angle with borrowing from the 2 nearest known
cards (weight 1 picture). The same 80 known classes in 8 stages of 10, the same 20 surprise
classes, the same 5 seeds. The draws of 1, 3 and 10 pictures are the same as in EXP-11 and
EXP-12. Smoothing: 0.01, 0.03, 0.1, 0.3, 1.

**Naming:** every test picture among all 100 classes by angle in the learned view, with the
handicap h subtracted from old classes' scores (EXP-11's C1; h = 0 means no correction). h from
−0.2 to 0.5 in steps of 0.01, every value logged.

| Part | Pictures per new class | Draws | Copies |
|---|---|---|---|
| 1, the curve | 1, 3, 10, 30, 100, all (500) | 20, 20, 20, 20, 10, 1 | none |
| 2, copies | 1, 3, 10 | 20 each | none / moved (8) / enlarged (5) / both (13) |

The new classes are never added to the tables or to the view (they were in EXP-11's reference),
so "all 500 pictures" here measures only the card, not a rebuilt memory.

**Choosing settings: the rehearsal**, as in EXP-9 to EXP-12. The memory after stage 6 knows 60
classes; the 20 classes of stages 7 and 8 play the new ones. For each copy setting and each
number of pictures, among all smoothings and handicaps, keep those where the old classes stay
within 5 points of no correction (h = 0, same smoothing) in the rehearsal; of those, take the
highest harmonic mean. The same rule gives the "no correction" readings (h = 0, smoothing with
the best rehearsal harmonic mean). **The selected copy setting** is the one, among moved,
enlarged and both, with the highest rehearsal harmonic mean at 3 pictures. All choosing is done
afterwards, from the rehearsal rows only, in `src/figures.py`.

## 4. What is measured

| | Measurement | Why it matters |
|---|---|---|
| M1 | The curve: old, new, harmonic mean and new-among-themselves against pictures per new class, with the handicap and without | How good a card must be; whether the handicap is still needed with good cards |
| M2 | Copies at 1, 3, 10 pictures: harmonic mean and new-among-themselves | Does each picture count for more? |
| M3 | **Picture-equivalent**: where 3 pictures with copies land on the curve of part 1 (read between the measured points, on a log scale) | Copies in the same unit as real pictures |
| M4 | Rehearsal against test | Can the choosing be trusted? |
| M5 | Time, GPU energy, peak GPU memory | Cost, and whether it fits the laptop |

## 5. How the result will be judged (written before the run)

All scores are means of 5 seeds, on the test, at the settings chosen in the rehearsal.
**"Gain" means the selected copy setting minus no copies, both chosen by the same rule in this
run, at the same number of pictures.**

**Step 0: is EXP-12 rebuilt?** No copies, smoothing 0.1, h = 0.08, 3 pictures: old within 0.5 of
52.47, new within 0.5 of 17.28 (EXP-11 and EXP-12).

**Step 1: the main verdict on copies.** The selected copy setting must reach:

| | Condition |
|---|---|
| (a) | harmonic-mean gain of 3.0 or more at 3 pictures, with the test's old-class accuracy no more than 6 points under no correction (5 in the rehearsal, 1 of slack) |
| (b) | harmonic-mean gain of 1.5 or more at 1 picture and at 10 pictures |
| (c) | new-among-themselves gain of 2.0 or more at 3 pictures (the cards themselves are better) |
| (d) | peak GPU memory 3.5 GB or less |

| Reading | Verdict |
|---|---|
| (a) to (d) all hold | **works** |
| (d) holds, and the harmonic-mean gain at 3 pictures is 1.0 or more, but (a), (b) or (c) fails | **helps, but not enough** |
| anything else | **does not work** |

**Step 2: the picture-equivalent.** 3 pictures with the selected copies count as how many
pictures, on the handicap curve of part 1? **1.5 times or more (4.5 pictures or more) counts as
"each picture counts for more".**

**Step 3: how many pictures are needed?** The smallest number of pictures on the handicap curve
at which the harmonic mean reaches 90% of its value with all pictures (read between the measured
points). A description, not a pass or fail.

**Step 4: is the handicap still needed with good cards?** With all pictures, the harmonic mean
with the handicap minus without. Under 1.0: **not needed**.

**Step 5: can the rehearsal be trusted?** If the selected copy setting's test harmonic mean at 3
pictures is within 1.0 of the best copy setting's on the test: **yes**.

**What I expect, written down now.**

- Step 0 passes exactly.
- The curve with the handicap, harmonic mean at 1 / 3 / 10 / 30 / 100 / all pictures: about
  18 / 26 / 36 / 43 / 47 / 50. Without correction: about 9 / 13 / 21 / 32 / 42 / 49. So step 3:
  about 60 pictures are needed for 90%, and step 4: the handicap is **not needed** with all
  pictures (under 1.0).
- Copies at 3 pictures: moved +1 on new-among-themselves, enlarged +0.5, both +1; harmonic mean
  +0.5 to +1. Selected: both. Picture-equivalent about 3.5 (step 2: no).
- So the verdict would be **"does not work"**: copies help a little but far from enough.
- Peak GPU memory about 2.3 GB. The rehearsal picks well.

I expect little from the copies because the eyes add up each stencil's matches over each quarter
of the picture, which already ignores small moves inside a quarter. My main prediction was wrong
in EXP-9, EXP-10 and EXP-12 and right in EXP-11. If it is wrong again, it stays in the journal.

## 6. What carries over beyond pictures

| Part | Depends on pictures? |
|---|---|
| The curve of score against examples per new class, and the measure "picture-equivalent" | **no**. The same curve can be measured for new words or topics in a language memory |
| Whether the handicap is still needed when new classes have as many examples as old ones | **no** |
| Moved and enlarged copies | **yes**. The language version would be rephrasing a sentence, which is not part of this plan |

## 7. What could go wrong

- **Copies are not new pictures.** A moved fox is the same fox: copies can steady the card
  against position but cannot show a different fox. The gain may be small for that reason.
- **Copies change what the card looks like.** An average of moved copies is blurrier than any
  single picture, and the test pictures are not blurred. This could even lower the score.
- **Enlarging** uses a smooth resize, which softens fine detail the stencils respond to.
- **The view does not include the new classes.** With all 500 pictures, the new classes may be
  held back by a view built without them, so "all" here can be lower than EXP-11's 52.9.
- **100 pictures and all pictures** use fewer draws (10 and 1), so those points are a little
  less certain.
- **5 seeds, one dataset, images only. The surprise classes are fixed.**

## 8. Time and disk

| | Expected |
|---|---|
| Run time | about 25 minutes on a free Kaggle GPU (EXP-12 took 18; describing the copies adds about 2 minutes per seed). About 25 of the roughly 30 free weekly hours are used, so about 0.5 is needed |
| Disk in this folder | 0.15 GB for this experiment's own Python environment (copied from EXP-12's, so nothing is downloaded), under 0.05 GB of results and figures |
| Downloads | none. A short code check on the laptop borrows EXP-2's environment and CIFAR-100 without changing them |
| GPU memory | about 2.3 GB (as EXP-11's A2) |

## 9. Words used in this journal

| Word | Meaning here |
|---|---|
| Class | One kind of thing to recognise |
| Eyes (features) | The part that turns a picture into a list of numbers (the description) |
| Memory | The part that stores what was learned about each class and gives the answer |
| Card | The average description of one class |
| Shared table (covariance) | A table of which numbers of a description rise and fall together inside a class |
| Smoothing | How strongly the shared table is evened out before use |
| Learned view | The directions along which the known classes differ most, computed in one step from the cards and the shared table (EXP-9) |
| Borrowing | Mixing the cards of the 2 nearest known classes into a new class's card (EXP-9) |
| Handicap | One amount subtracted from every old class's score before the answer is picked (EXP-10) |
| Copy | The same picture moved by a few pixels, or a part of it enlarged |
| Picture-equivalent | How many real pictures would give the same score as the pictures with copies |
| Harmonic mean | One score from two: 2 × a × b / (a + b). It is high only if both are high |
| Rehearsal | Choosing settings by pretending the most recent known classes are new. Uses no test class |
| Backpropagation | The usual way to train a network: push corrections back through all layers, many times |
| Seed | One repeat with a different random shuffle |

<div class="pagebreak"></div>
