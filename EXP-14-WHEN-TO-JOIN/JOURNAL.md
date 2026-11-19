# EXP-14 · WHEN SHOULD A NEW CLASS JOIN THE VIEW?

**Experiment ID:** EXP-14 · **Part A written:** 2026-11-19, before any code was run · **Hardware:** free Kaggle GPU (the method must still fit the RTX 3050 laptop, 4 GB) · **Status:** PLANNED (Part A only; nothing has been run)

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
- **EXP-9**: a **learned view** (the directions along which the known classes differ most) and
  **borrowing** from the 2 nearest known cards: a new class from 3 pictures scores 33.5% among
  20 new classes.
- **EXP-10**: among all 100 classes the new ones are drowned out. A **handicap** on old classes
  only trades old answers for new ones. Putting the new classes into the tables and the view
  when they have 3 pictures **made things worse** (harmonic mean −3.6, `EXP-10-NEW-AMONG-OLD`).
- **EXP-11**: sharper eyes: 59.3% on 80 classes. A reference with the new classes' **all 500
  pictures put into the tables and the view** reached a harmonic mean of **52.9** with no
  correction (old 54.1, new 51.7).
- **EXP-12**: a switch "old class or new class?" from familiarity did not work.
- **EXP-13** measured how the score among all 100 classes grows with the pictures per new class,
  with the new classes **kept out of the tables and the view**. Readings from
  `EXP-13-HOW-MANY-PICTURES/results/summary.json`, test, 5 seeds:

| Pictures per new class | 1 | 3 | 10 | 30 | 100 | all (500) |
|---|---|---|---|---|---|---|
| Harmonic mean, with the handicap | 18.4 | 26.0 | 35.8 | 41.1 | 44.2 | 45.1 |
| Harmonic mean, no correction | 9.7 | 14.3 | 22.7 | 29.2 | 34.3 | **36.5** |
| New classes among themselves | 24.9 | 36.2 | 46.9 | 53.8 | 58.1 | 59.4 |

Moved or enlarged copies of a picture did not help (verdict "does not work").

### What stayed open

EXP-13 found **two limits**. Up to about 30 pictures, the limit is the card. Beyond that, the limit
is the **view**: with all 500 pictures, new classes kept out of the view reach a harmonic mean of
36.5 with no correction, against 52.9 when they are inside it (EXP-11). The view keeps the directions
that tell the **old** classes apart; a fox that is not part of it is seen only through
"dog-or-cat-or-wolf" directions.

So a new class should join the view at some point. EXP-10 says 3 pictures is too early; EXP-11 says
500 is fine. **Where the crossover lies, and whether the shared table matters as well as the view,
is unknown.** EXP-13's section 15 lists both as unverified.

### What the papers say

One Sonnet subagent, 2026-11-19 (literature log, section 12). Five papers were opened; none
measures after how many examples a new class should join a shared view.

| Paper | Tag | What it reports |
|---|---|---|
| McClelland, McNaughton, Lampinen, "Integration of new information in memory: new insights from a complementary learning systems perspective", Phil. Trans. R. Soc. B 375:20190637 (2020) | passages | Brain theory with small simulations. A fast store holds a new item at once; a slow "cortex" absorbs it later. The abstract says "the projection onto the known dimensions can be learned rapidly", but "learning the new dimension requires gradual interleaved learning". **This is the same split as our "card now, join the view later"** |
| Zhou et al., "Forward Compatible Few-Shot Class-Incremental Learning" (FACT), arXiv:2203.06953 | passages | Keeps the shared features fixed when new classes arrive; new classes get only an average. CIFAR-100, last session: 49.1%. The text says gains from more pictures per class become "trivial for K > 20" (figure only, no numbers read) |
| Akyürek et al., "Subspace Regularizers for Few-Shot Class Incremental Learning", ICLR 2022, arXiv:2110.07059 | passages | New classes are kept close to the directions of the old classes, which are not changed. miniImageNet, 5 pictures per class, last session: 46.8 against 26.5 for changing the representation |
| Chumachenko et al., "Incremental Fast Subclass Discriminant Analysis", arXiv:2002.04348 | passages (method; tables did not extract) | A closed-form update of a discriminant view as data arrive, "equal or close" to rebuilding it. About how to update, not when |
| Khawand et al., "Continual Learning with Deep Streaming Regularized Discriminant Analysis", arXiv:2309.08353 | passages (text; table did not extract) | Each class's own table is blended with the shared table (best blend 0.55 by search); a class's own table alone fails when it has little data |

The few-shot methods all keep new classes **out** of the shared directions when they have 1 to 5
pictures; none tests letting them in later. The brain theory says new directions should be added
slowly. No paper gives the crossover.

### So EXP-14 asks

**After how many pictures should a new class join the memory's learned view? And does a memory
that holds new classes apart first and lets them join once they have enough pictures (like
consolidation in the brain) beat one that never lets them join, or one that lets them join at
once?**

# Part A: the plan (written before the run)

## 2. The idea in plain words

**Cats and digits example.** The memory knows 80 classes from 500 pictures each. Its learned view
is like a set of questions it asks about every picture: "pointy ears or round?", "fur or
feathers?", "four legs on the ground or wings?". These questions were chosen because they tell the
80 known classes apart. A fox arrives. The memory makes a fox card, but it still asks only its old
questions, and a fox answers them much like a dog. The memory never asks "bushy tail and orange
coat?", because no known class needed that question.

**Joining the view** means adding the fox's question: one new direction, the one in which the fox's
card differs from everything the memory already distinguishes. It is computed in one step from the
fox's card and the existing shared table; nothing is trained. With 3 fox pictures, that direction is
partly about those 3 particular photos (a fox in snow, a fox at night), so the new question may be
a bad one. With 100 pictures it is about foxes.

**Three ways to treat a new class**, each from the same pictures:

| Name | What happens to the new class | Cost |
|---|---|---|
| **Apart** | a card only; the view is the old classes' view (EXP-13) | none |
| **View** | its card also adds new directions to the view; the shared table stays as it was | a few small calculations per class |
| **Table and view** | its pictures are also added to the shared table, and the view is rebuilt | rebuilding the view (about 4 seconds) |

**Part 1, the crossover.** Give each of the 20 new classes 1, 3, 10, 30, 100 or all 500 pictures,
and compare the three ways. This finds the number of pictures from which joining helps.

**Part 2, a stream with consolidation.** New classes arrive a few at a time and get more pictures
later, as a child sees a fox once and then more often. A class is held apart until it has at least
N pictures, then it joins the view. Compared: never joining, joining at once, and joining at 10, 30
or 100 pictures. N is chosen in the rehearsal (below), not on the test.

## 3. What is run

**Eyes and memory: EXP-11's version A2 and EXP-13's memory, unchanged.** 1,600 stencils,
two-sided, 4 quarters, mirrored (12,800 numbers per picture); compact tables; learned view of the
old classes with all directions; new cards by angle with borrowing from the 2 nearest known cards
(weight 1 picture). The same 80 known classes in 8 stages of 10, the same 20 surprise classes, the
same 5 seeds. Smoothing: 0.01, 0.03, 0.1, 0.3, 1. No copies.

**Naming:** every test picture among all classes by angle in the view, with the handicap h
subtracted from the **80 old classes'** scores (EXP-11's C1; h = 0 means no correction). h from
−0.2 to 0.5 in steps of 0.01, every value logged. New classes that joined the view still count as
new, and the handicap is never subtracted from them.

**How a class joins the view ("View").** The learned view is built from the cards of the known
classes, as seen through the shared table. A joining class adds its own card: the view then covers
the directions of all the old cards and all the joining cards. In practice, for each joining
class, the part of its card that the old view cannot express becomes one extra direction. The
old directions are unchanged, so the old classes' cards do not move. The new class's card is
then made exactly as before (average of its pictures in the view, borrowing from the 2 nearest
old cards), only now in the larger view.

**Table and view.** As in EXP-10's C4 and EXP-11's reference: the new pictures (and their mirror
images) are added to the running tables, the new classes become classes of the tables, and the
view is rebuilt from scratch from all classes. This costs a full rebuild per setting, so it is
measured on fewer draws (below).

| Part | Pictures per new class | Draws | Ways |
|---|---|---|---|
| 1, the crossover | 1, 3, 10, 30, 100, all (500) | 20, 20, 20, 20, 10, 1 (the same draws as EXP-13) | apart, view |
| 1, table and view | 3, 10, 30, 100, all | the first 2, 2, 1, 1, 1 of those draws | table and view |
| 2, the stream | see below | 10 runs of the stream | 5 rules |

**The stream.** The 20 new classes arrive in 4 sessions of 5 (in a random order for each run). A
class has 3 pictures in the session it arrives, then 10, 30 and 100 pictures in the next sessions
(more pictures of the same class keep arriving; earlier pictures are kept). After each session,
every test picture of the 80 old classes and of the classes that have arrived is named among all of
them. Joining rules: **never**; **at once** (3 pictures); **at 10**, **at 30**, **at 100**
pictures. At each session, every class that has reached N pictures is in the view, through its card
from all its pictures so far. The **stream score** is the harmonic mean of old and new, averaged
over the 4 sessions.

| Session | Classes that have arrived (pictures each) |
|---|---|
| 1 | 5 (3) |
| 2 | 5 (10), 5 (3) |
| 3 | 5 (30), 5 (10), 5 (3) |
| 4 | 5 (100), 5 (30), 5 (10), 5 (3) |

**Choosing settings: the rehearsal**, as in EXP-9 to EXP-13. The memory after stage 6 knows 60
classes; the 20 classes of stages 7 and 8 play the new ones, in the same three ways and the same
stream. For each way (or joining rule) and each number of pictures, among all smoothings and
handicaps, keep those where the old classes stay within 5 points of no correction (h = 0, same
smoothing) in the rehearsal; of those, take the highest harmonic mean (for the stream: the highest
stream score, with the old-class accuracy averaged over the 4 sessions). The same rule gives the "no
correction" readings (h = 0, the smoothing with the best rehearsal harmonic mean). **The selected
joining rule N\*** is the one with the highest rehearsal stream score. All choosing is done
afterwards, from the rehearsal rows only, in `src/figures.py`.

## 4. What is measured

| | Measurement | Why it matters |
|---|---|---|
| M1 | The crossover: old, new, harmonic mean and new-among-themselves against pictures per new class, apart and in the view, with the handicap and without | From how many pictures joining helps |
| M2 | Table and view against view only, on the same draws | Whether the shared table must change too, or only the view |
| M3 | The stream: stream score and the score at each session, for each joining rule | Whether "hold apart, then join" works as a running system |
| M4 | The handicap with all pictures, in the view: with minus without | Whether joining removes the need for the handicap |
| M5 | Rehearsal against test | Can the choosing be trusted? |
| M6 | Time, GPU energy, peak GPU memory | Cost, and whether it fits the laptop |

## 5. How the result will be judged (written before the run)

All scores are means of 5 seeds, on the test, at the settings chosen in the rehearsal.

**Step 0: are the earlier results rebuilt?**

- (a) Apart, smoothing 0.1, h = 0.08, 3 pictures: old within 0.5 of 52.47 and new within 0.5 of
  17.28 (EXP-11 to EXP-13).
- (b) Table and view with all pictures, EXP-11's cards (class average, no borrowing), no
  correction, smoothing 0.1: old within 0.5 of 54.09 and new within 0.5 of 51.67 (EXP-11's
  reference, `EXP-11-SHARPER-EYES/results/summary.json`).

**Step 1: the crossover (a description, not a pass or fail).** For each number of pictures, the
gain from joining = view minus apart, harmonic mean, with the handicap and with no correction.
The **crossover** is the smallest measured number of pictures from which the gain with the
handicap is +1.0 or more, at that number and at every larger one.

**Step 2: does the table matter?** With all pictures, table and view minus view only, harmonic mean
with the handicap, same draw. Under 1.0: **the view is enough**; 1.0 or more: **the table matters
too**.

**Step 3: the main verdict, the stream.** The selected joining rule N\* against never joining:

| | Condition |
|---|---|
| (a) | stream score gain of 3.0 or more over never joining |
| (b) | N\*'s old-class accuracy (averaged over the 4 sessions) no more than 6 points under its own no correction (5 in the rehearsal, 1 of slack) |
| (c) | peak GPU memory 3.5 GB or less |

| Reading | Verdict |
|---|---|
| (a) to (c) all hold | **works** |
| (c) holds, and the stream score gain is 1.0 or more, but (a) or (b) fails | **helps, but not enough** |
| anything else | **does not work** |

**Step 4: does holding apart matter?** If N\* is not "at once" and its stream score is 1.0 or more
above "at once": **yes, holding apart matters**. Otherwise: **no** (joining at once is as good).

**Step 5: is the handicap still needed once new classes are in the view?** With all pictures, in
the view, the harmonic mean with the handicap minus with no correction. Under 1.0: **not needed**.

**Step 6: can the rehearsal be trusted?** If N\*'s test stream score is within 1.0 of the best
joining rule's on the test: **yes**.

**What I expect, written down now.**

- Step 0 passes exactly, both parts.
- In the view, harmonic mean with the handicap at 1 / 3 / 10 / 30 / 100 / all pictures: about
  17 / 25 / 37 / 44 / 49 / 52 (apart: 18.4 / 26.0 / 35.8 / 41.1 / 44.2 / 45.1). With no correction:
  about 9 / 14 / 25 / 35 / 45 / 50 (apart: 9.7 / 14.3 / 22.7 / 29.2 / 34.3 / 36.5). So joining
  hurts a little at 1 and 3 pictures and helps from 10 on: **crossover at 10 pictures**.
- Step 2: table and view adds under 1.0 over the view alone: **the view is enough**.
- Step 3: N\* is "at 10" or "at 30"; its stream score is about 2 points above never joining. Verdict
  **"helps, but not enough"**, because in most sessions most new classes have few pictures, and few
  pictures is where joining helps least.
- Step 4: **yes**, joining at once is about 1 point worse than N\*.
- Step 5: **not needed** (under 1.0) once the new classes are in the view with all their pictures.
- Step 6: yes. Peak GPU memory about 2.5 GB.

My main prediction was wrong in EXP-9, EXP-10, EXP-12 and EXP-13 and right in EXP-11. If it is
wrong again, it stays in the journal.

## 6. What carries over beyond pictures

| Part | Depends on pictures? |
|---|---|
| A memory with a learned view, and the step that adds a new class's direction to it | **no**. It needs only the class averages and the shared table, which a language memory would have too |
| The crossover: from how many examples a new class should reshape the view | **no**. The number itself may differ for words or topics |
| Holding new knowledge apart, then consolidating it (the stream) | **no**. This is the part most likely to matter for a language model that keeps learning |
| The eyes (stencils, mirror images) | **yes**, unchanged from EXP-11 |

## 7. What could go wrong

- **A new direction from few pictures fits those pictures.** The fox's direction from 3 pictures
  points partly at "snow" and "night". Old pictures with snow in them may then look like foxes. This
  is why joining at once may hurt.
- **The new directions are measured with the old shared table.** In "view", the table does not
  know how a fox varies from fox to fox. Table and view fixes this, but only on a few draws.
- **The handicap and the joining interact.** A joined class is still "new" for the handicap, so the
  handicap may now push too many answers to new classes. The rehearsal chooses h for each rule
  separately to allow for this.
- **The stream is short.** 4 sessions, at most 100 pictures per class, 10 runs. A class never
  reaches 500 pictures in the stream, so "at 100" is joined only in the last session.
- **Table and view uses 1 or 2 draws per point.** Its readings are less certain.
- **5 seeds, one dataset, images only. The surprise classes are fixed.**

## 8. Time and disk

| | Expected |
|---|---|
| Run time | about 60 minutes on a free Kaggle GPU (EXP-13 took 16; rebuilding the view for table and view adds about 6 minutes per seed). About 25.3 of the roughly 30 free weekly hours are used, so about 1 is needed |
| Disk in this folder | 0.15 GB for this experiment's own Python environment (copied from EXP-13's, so nothing is downloaded), under 0.05 GB of results and figures |
| Downloads | none. A short code check on the laptop borrows EXP-2's environment and CIFAR-100 without changing them |
| GPU memory | about 2.5 GB (EXP-13: 2.2 GB) |

## 9. Words used in this journal

| Word | Meaning here |
|---|---|
| Class | One kind of thing to recognise |
| Eyes (features) | The part that turns a picture into a list of numbers (the description) |
| Memory | The part that stores what was learned about each class and gives the answer |
| Card | The average description of one class |
| Shared table (covariance) | A table of which numbers of a description rise and fall together inside a class |
| Smoothing | How strongly the shared table is evened out before use |
| Learned view | The directions along which the known classes differ most, computed in one step from the cards and the shared table (EXP-9). Like a set of questions the memory asks about every picture |
| Joining the view | Adding a new class's own direction to the view, computed in one step from its card |
| Consolidation | Holding a new memory apart first and folding it into the general structure later. The brain is thought to do this between the hippocampus and the cortex |
| Borrowing | Mixing the cards of the 2 nearest known classes into a new class's card (EXP-9) |
| Handicap | One amount subtracted from every old class's score before the answer is picked (EXP-10) |
| Harmonic mean | One score from two: 2 × a × b / (a + b). It is high only if both are high |
| Stream score | The harmonic mean of old and new, averaged over the 4 sessions of the stream |
| Rehearsal | Choosing settings by pretending the most recent known classes are new. Uses no test class |
| Backpropagation | The usual way to train a network: push corrections back through all layers, many times |
| Seed | One repeat with a different random shuffle |
