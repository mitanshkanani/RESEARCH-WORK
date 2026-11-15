# EXP-8 · SAME MEMORY FOR EVERY EYES

**Experiment ID:** EXP-8 · **Part A written:** 2026-11-16, before any code was run · **Hardware:** free Kaggle GPU (the method must still fit the RTX 3050 laptop, 4 GB) · **Status:** PLANNED, NOT RUN YET

This journal has two parts. **Part A** (this part) explains the experiment before it starts.
**Part B** will be added to the end of this same PDF when the experiment finishes. Part A
will not be edited after results exist, except for the status word in the line above.

## 1. Why this experiment

### What happened before

The project thinks of a learning system as two parts: the **eyes**, which turn a picture into
a list of numbers describing it, and the **memory**, which stores what was learned about
each class and gives the answer.

- **EXP-1** was a plan only. It was never run.
- **EXP-2 to EXP-6** all used the same simple memory: one **card** per class (the average
  description), compared by angle. With it, eyes trained by backpropagation reached 33.2%
  (frozen after the first group), 39.6% (grow and lock) and 53.2% (retrain everything) on the
  80-class photo test. EXP-6 found that locking units trained on few classes is what holds
  grow and lock back.
- **EXP-7** tried eyes that need no training at all (1,600 fixed picture patches) and, for
  the first time, a richer memory: cards plus **one shared table** of how the numbers of a
  description vary together. Its readings, exam half, 5 seeds (from
  `EXP-7-PATCH-EYES/results/summary.json`; backprop rows quoted from EXP-6):

| | All 80 classes | 20 never-seen classes from 3 pictures | Time to learn |
|---|---|---|---|
| Patch eyes + plain cards | 15.4% | 15.8% | 9 s |
| **Patch eyes + cards and shared table** | **55.2%** | 8.2% (17.0% with plain cards after a square root) | 24 s |
| Frozen backprop eyes + plain cards | 33.2% | 25.6% | 26 s |
| Grow and lock + plain cards (R7) | 39.6% | 29.7% | 131 s |
| Retrain everything + plain cards | 53.2% | 38.5% | 202 s |

### What stayed open in EXP-7

By its written rule the verdict was "works": no backpropagation, one look at each picture,
nothing forgotten, and a higher score than the retraining reference. But the journal
attached three warnings, and the first one makes the headline unreliable:

1. **The comparison was not fair.** The same eyes went from 15.4% to 55.2% just by
   changing the memory. The backprop eyes were only ever measured with the weak memory.
   Nobody knows what they score with the strong one. The retrained network was also never
   allowed to answer with its own trained answer layer, which is how such a network is
   normally used.
2. **From 3 pictures per class the new system is poor** (17.0% against 29.7%).
3. It used 5.5 GB of GPU memory as written.

EXP-2 concluded "the memory works, the eyes are the bottleneck". After EXP-7 that
conclusion is in doubt.

### So EXP-8 asks

**With the same memory on every kind of eyes, which eyes are best, and how far is the
cheap no-backprop system from the best a trained network can do?** And: **does any memory
make learning from 3 pictures work?**

# Part A: the plan (written before the run)

## 2. The idea in plain words

So far we compared a runner in good shoes with runners in sandals and said the first one
was faster. EXP-8 gives everyone the same shoes.

**Cats and digits example.** Four students have learned cats, dogs and the digits, each
with different eyes. Until now three of them had to answer by "which average picture is
this closest to?", and the fourth was also told which features tend to go together (fur
with whiskers; a loop on top with a stem below). The fourth scored best. Was it the better
eyes, or the extra knowledge? EXP-8 gives all four the same extra knowledge and asks again.

## 3. What is run

### Four kinds of eyes, all on the same test

| | Eyes | Built how | Allowed in the real method? |
|---|---|---|---|
| P | **Patch eyes** | 1,600 fixed patches from the first group, threshold 0.1 (EXP-7's selected) | yes; no backprop |
| F | **Frozen backprop eyes** | network trained on the first 10 classes, then never again | yes |
| G | **Grow and lock** | units added and locked at each stage (EXP-6 run A) | yes |
| R | **Retrain everything** | a fresh network trained on all 80 classes at once | no (keeps all pictures); it is the reference |

### The same memories on each

| Memory | What it is |
|---|---|
| M-cos | plain cards, compared by angle (as in EXP-2 to EXP-6) |
| M-sqrt | the same after taking the square root of every number |
| M-lda | cards plus one shared table, with five smoothing values: 0.0001, 0.001, 0.01, 0.1, 1 |
| M-ridge | one-step straight-line classifier from running tables, same five values |
| Own answer layer (R only) | the answer layer the retrained network was trained with. **This is the honest ceiling** |

For grow and lock, old classes never had the newer units, so their part of the shared
table is missing. Two versions are run:

- **G with filled-in tables** (a real method): the missing parts of old cards are guessed
  as in R7, and the missing parts of the shared table are filled by the same kind of
  guess: learn from the current pictures how the new units follow from the old ones, and
  assume old classes behave the same way (an exact formula for this pattern of missing
  data is in arXiv:2009.11360, read as passages).
- **G open-book** (diagnosis only): tables written from all old pictures.

### The 3-picture surprise test

For every kind of eyes: plain cards, cards after square root, and cards judged with the
shared table built from the 80 learned classes, with smoothing from 0.01 up to 100 (strong
smoothing is what one paper used for few examples; FeCAM, arXiv:2309.14062, passages read).

### The test

Unchanged: photos (CIFAR-100), 80 classes in 8 stages of 10, old pictures thrown away
after each stage, 20 surprise classes, 5 seeds with the same class orders and the same
practice and exam halves as EXP-4 to EXP-7. Backprop networks as in EXP-6 (30 passes).

**Settings fixed now.**

| Setting | Value |
|---|---|
| Choosing the memory for each kind of eyes | the memory and smoothing with the highest score on all 80 classes on the practice half, among those that keep the first group (drop of 3 points or less with only its own cards, on the practice half). Scores are then read on the exam half |
| Tables are solved after stages | 1, 4 and 8 |
| Seeds | 5 |

## 4. What is measured

| | Measurement | Why it matters |
|---|---|---|
| S1 | Accuracy on all 80 classes after stage 8, for every eyes with every memory | The fair comparison |
| S2 | The retrained network with its own answer layer | The real ceiling for this size of network |
| S3 | First group with only its own cards, stage 1 against stage 8 | Forgetting, for every eyes and memory |
| S4 | 3-picture surprise test, every eyes and memory | The project's few-example goal |
| S5 | Time and GPU energy: training (backprop eyes), describing pictures, filling and solving tables | Cost of each full system, this time including the tables |
| S6 | Peak GPU memory | Laptop fit |

## 5. How the result will be judged (written before the run)

Each kind of eyes gets its best memory, chosen on the practice half. All scores below are
exam half, after stage 8, mean of 5 seeds.

**Step 0: are the earlier numbers rebuilt?** Within 3 points each: patch eyes with M-lda at
0.01: 55.2%; frozen with M-cos: 33.2%; grow and lock with R7 cards: 39.6%; retrain with
M-cos: 53.2%.

**Step 1: the ceiling.** Ceiling = the higher of "retrain with its own answer layer" and
"retrain with its best table memory".

**Step 2: where does the no-backprop system stand?** Patch eyes with their best memory,
against the ceiling:

| Distance below the ceiling | Verdict |
|---|---|
| 3 points or less (or above) | **matches the trained network** |
| 3 to 10 points | **close, at a fraction of the cost** |
| more than 10 points | **clearly behind** |

**Step 3: was grow and lock held back by its memory?** Grow and lock with filled-in tables
against 39.6%: a gain of 5 points or more counts as **yes**.

**Step 4: was EXP-2 right that the eyes are the bottleneck?** For frozen backprop eyes:
if the best table memory lifts them by 5 points or more over plain cards, the memory was
also a bottleneck.

**Step 5: few examples.** The best 3-picture score of each eyes with any memory. The
target to beat is 29.7% (grow and lock with plain cards); the retrain reference is 38.5%.

**What I expect, written down now.**

- Ceiling: retrain with its own answer layer about **60%**; with the table memory about 60% too.
- Patch eyes with a smaller smoothing value: about **56%**. So the verdict would be
  **"close, at a fraction of the cost"**, about 4 points under the ceiling.
- Frozen backprop eyes with tables: about 42% (a gain of about 9, so the memory was also a
  bottleneck in EXP-2).
- Grow and lock with filled-in tables: about 46% (a gain of about 6); open-book about 49%.
- Under the same memory the order will be: retrain, then patch eyes, then grow and lock,
  then frozen.
- Few examples: no memory will rescue patch eyes (best about 18%). The table memory with
  strong smoothing will help the backprop eyes a little (retrain about 40%, grow and lock
  about 31%).

My main predictions were wrong in EXP-4, EXP-5 and EXP-6, and my score prediction was 11
points too low in EXP-7. If any of this is wrong, it stays in the journal as written.

## 6. What carries over beyond pictures

| Part | Depends on pictures? |
|---|---|
| The four kinds of eyes | yes, all four are built for pictures |
| The memories (cards, shared table, straight-line classifier, filled-in tables) | no. They work on lists of numbers |
| The question "is it the eyes or the memory?" | no. Any system with these two parts should ask it before changing the eyes |

## 7. What could go wrong

- **Filled-in tables rest on a guess**: that old classes relate new units to old units the
  way current classes do. R7 made the same guess for cards and it worked (39.6% against
  41.0% open-book). For the table it is untested; the open-book version shows what is lost.
- **The ceiling is for this small network and 30 passes.** A bigger or longer-trained
  network would be higher. "Matches the trained network" would mean this network, nothing more.
- **Choosing among ten memory settings per eyes** on the practice half guards against a
  lucky pick only partly; differences under 1 point between settings should not be read into.
- **The smallest smoothing may again be the best**, in which case the best value still
  lies outside the tried range. Part B will say so.
- **Backprop features are short** (576 numbers) and patch features long (6,400). The table
  memory may suit long descriptions better. That is part of what is being measured, not an error.
- **5 seeds, one dataset, images only.**
- **Training is not perfectly repeatable on a GPU.** Step 0 checks the rebuild.

## 8. Time and disk

| | Expected |
|---|---|
| Run time | about 1 hour on a free Kaggle GPU for all 5 seeds. From measured times per seed: frozen 26 s, grow and lock 131 s, retrain 202 s (EXP-6), patch eyes about 1.5 minutes with five smoothing values (EXP-7), plus scoring |
| Disk in this folder | 0.15 GB measured for the Python environment (figures and PDF), plus under 0.05 GB of results and figures |
| Downloads to the laptop | none. A short code check on the laptop borrows EXP-2's environment and CIFAR-100 without changing them |
| GPU memory | about 5.5 GB for the patch eyes as in EXP-7 (not laptop-sized as written); under 1.2 GB for the backprop parts. Peak use will be reported per kind of eyes |

## 9. Words used in this journal

| Word | Meaning here |
|---|---|
| Class | One kind of thing to recognise |
| Eyes (features) | The part that turns a picture into a list of numbers |
| Memory | The part that stores what was learned about each class and gives the answer |
| Card | The average description of one class |
| Shared table (covariance) | A table recording which numbers of a description tend to rise and fall together, kept as a running sum |
| Smoothing | How strongly the shared table is evened out before use. More smoothing trusts the table less |
| Answer layer | The last layer of an ordinary trained network, which turns the description into one score per class. It is trained by backpropagation together with the eyes |
| Backpropagation | The usual way to train a network: measure the error and push corrections back through all layers, many times |
| Grow and lock | Add units for each new group of classes; never change old units |
| R7 | Guess the missing part of an old, short card from the part it has |
| Open-book | Using old pictures to rewrite what the memory holds. Forbidden in the real test; used as a diagnosis |
| Ceiling | The best score a trained network of this size reaches when it may keep all pictures |
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
