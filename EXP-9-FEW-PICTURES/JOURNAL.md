# EXP-9 · A NEW CLASS FROM A FEW PICTURES

**Experiment ID:** EXP-9 · **Part A written:** 2026-11-17, before any code was run · **Hardware:** free Kaggle GPU (the method must still fit the RTX 3050 laptop, 4 GB) · **Status:** PLANNED. Nothing has been run yet

This journal has two parts. **Part A** (this part) explains the experiment before it starts.
**Part B** will be added to the end of this same PDF when the experiment finishes. Part A
will not be edited after results exist, except for the status word in the line above.

## 1. Why this experiment

### What happened before

The project thinks of a learning system as two parts: the **eyes**, which turn a picture into
a list of numbers describing it, and the **memory**, which stores what was learned about
each class and gives the answer.

- **EXP-1** was a plan only. It was never run.
- **EXP-2 to EXP-6** used eyes trained by backpropagation and a simple memory: one **card**
  per class (the average description). The best system that keeps no old pictures (grow
  and lock) reached 39.6% on the 80-class photo test.
- **EXP-7** built eyes that need no training (1,600 fixed picture patches) and a richer
  memory: cards plus **one shared table** of how the numbers of a description vary together.
- **EXP-8** gave every kind of eyes the same memories. Its readings, 5 seeds (from
  `EXP-8-SAME-MEMORY/results/summary.json`):

| Eyes | All 80 classes, best memory | 20 never-seen classes from 3 pictures, best memory | Time to learn |
|---|---|---|---|
| **Patch eyes (no backprop)** | **55.2%** | **21.2%** | 24 s |
| Frozen backprop eyes | 44.1% | 25.6% | 24 s |
| Grow and lock | 43.7% | 29.8% | 127 s |
| Retrain everything (keeps all pictures) | 58.8% | 38.5% | 190 s |
| Retrain with its own answer layer (the ceiling) | 61.1% | | 190 s |

### What stayed open in EXP-8

The no-backprop system now meets two of the project's three goals: it forgets nothing, and
it is cheap (one look at each picture, 24 seconds). The third goal, **a new class from a
handful of pictures**, is not met. The same test at 1, 3 and 10 pictures per class:

| Eyes, best memory at each count | 1 picture | 3 pictures | 10 pictures |
|---|---|---|---|
| Patch eyes | 12.5% | 21.2% | 35.7% |
| Frozen backprop eyes | 17.2% | 25.6% | 37.8% |
| Grow and lock | 19.4% | 29.8% | 41.1% |
| Retrain everything | 25.4% | 38.5% | 51.5% |

No memory tried in EXP-8 changed this order. The papers read during EXP-8 (literature log,
section 8, read as passages) all say the same: learning from a few pictures depends on what
the eyes learned from labelled classes. Patch eyes learn nothing from labels.

But the **memory** of the patch system does hold labelled knowledge: the cards of 80
classes and the shared table. EXP-8 used that knowledge only to answer questions about
those 80 classes. It was never used to change how a **new** class is described.

### So EXP-9 asks

**Can what the memory already knows about 80 classes be turned, by single calculations
with no backpropagation, into a better way of learning a new class from 1 to 10 pictures?
And does knowing more classes make the next class easier to learn?**

# Part A: the plan (written before the run)

## 2. The idea in plain words

**Cats and digits example.** A child has learned cats, dogs and the ten digits from many
pictures. Now the child sees three pictures of a fox. A child with no experience would
compare every detail of the fox pictures: the grass behind it, the light, the exact pose.
Most of those details are accidents of those three pictures. A child with experience
knows, from cats and dogs, **which details usually tell animals apart** (ear shape, snout,
tail) and which do not (the background). It looks at the fox through that knowledge and
needs far fewer pictures.

The patch eyes are the child with no experience: 6,400 numbers per picture, all treated as
equally important. The memory, though, already contains the experience: from 80 learned
classes it can work out which combinations of numbers separate classes from one another.
EXP-9 tests three ways of using that, each a single calculation:

1. **Tidy the descriptions.** Take the square root of every number, subtract the average
   description of everything known so far, and scale each description to the same length.
   This removes what all pictures have in common, so that what is left is what differs.
2. **A learned view.** From the cards and the shared table, compute the handful of
   directions (at most 79 for 80 classes) along which the known classes differ most
   compared with how much pictures of one class vary. Describe every new picture by those
   directions only, instead of all 6,400 numbers. This is the "which details matter"
   knowledge. Nothing is trained; it is one calculation from tables the memory already keeps.
3. **Borrowing.** A card made from 3 pictures is shaky. Steady it by mixing in the cards
   of the two known classes it looks most like (a fox card leans a little on dog and cat).

## 3. What is run

### The eyes

| | Eyes | Built how |
|---|---|---|
| P | **Patch eyes** (the system under test) | exactly EXP-8's: 1,600 fixed patches from the first group, threshold 0.1, 6,400 numbers per picture, no backprop |
| F | **Frozen backprop eyes** (comparison) | exactly EXP-8's: a network trained on the first 10 classes, then never again, 576 numbers per picture |

Grow and lock (29.8%) and retraining (38.5%) are not rebuilt. Their EXP-8 scores are the
targets quoted in section 5.

### The ways of learning a new class

In every case a new class is stored as one card made from its few pictures, and a test
picture is given to the class whose card is nearest. What differs is how descriptions and
cards are prepared. Square roots are taken first in all of them, as in EXP-8.

| | Way | What it uses from the known classes | Settings tried |
|---|---|---|---|
| W0 | Plain cards, compared by angle (EXP-8's "cards after square root") | nothing | none |
| W1 | Shared table with heavy smoothing (EXP-8's best for patch eyes) | the shared table | smoothing 1, 10, 100, 1000 |
| W2 | Tidy: subtract the known average, scale to length 1 | the average description | none |
| W3 | **Learned view**, then tidy | cards and shared table | smoothing of the table 0.01, 0.1, 1, 10, 100; keep all directions or the stronger half; compare by angle or by straight distance |
| W2+B, W3+B | The same with **borrowing** from the 2 nearest known cards | also the known cards | weight of the borrowed part: 0.5, 1 or 2 pictures' worth |

### How settings are chosen without touching the test: a rehearsal

Several ways have settings, and choosing them on the surprise classes would be cheating.
So settings are chosen in a **rehearsal that uses no surprise class**:

- After stage 6 the memory knows 60 classes. The 20 classes of stages 7 and 8 have not
  been seen yet. They play the part of new classes: each is learned from 3 pictures and
  tested, with the memory as it stood after stage 6.
- For each way, the setting with the highest rehearsal score (mean of 5 seeds) is fixed.
  The **selected way** for each kind of eyes is the one with the highest rehearsal score.
- Those fixed settings are then used, unchanged, on the real surprise classes, for every
  number of pictures and every number of known classes.

A real system could do the same: it needs only the pictures of the classes arriving now.

### The test

Unchanged from EXP-3 to EXP-8: photos (CIFAR-100), 80 classes learned in 8 stages of 10,
old pictures thrown away after each stage, the same 20 surprise classes, 5 seeds with the
same class orders. Each surprise class is learned from 1, 3 or 10 pictures, drawn 20 times
at random; all 2,000 test pictures of the surprise classes are then sorted among the 20
surprise classes.

**New in EXP-9:**

- The surprise test is repeated with the memory as it stood after **stage 1 (10 known
  classes), stage 4 (40) and stage 8 (80)**.
- **One extra check, all 100 classes together.** After stage 8, the 20 surprise classes
  (3 pictures each) are added to the 80 learned classes and every test picture must be
  sorted among all 100. This shows whether a class learned from 3 pictures can hold its
  place next to classes learned from 500, and whether adding it harms them. It is run for
  the selected way and for EXP-8's shared-table memory (smoothing 0.01). It is reported,
  not judged.

**Settings fixed now.**

| Setting | Value |
|---|---|
| Rehearsal | memory after stage 6; stages 7 and 8 as pretend-new classes; 3 pictures each; 20 draws; mean of 5 seeds |
| Borrowing | 2 nearest known cards (nearest by the same comparison the way uses) |
| Smoothing | as in EXP-8: the value times the average of the table's diagonal |
| Pictures per new class | 1, 3, 10; 20 random draws each |
| Seeds | 5 |

## 4. What is measured

| | Measurement | Why it matters |
|---|---|---|
| S1 | 3-picture score on the 20 surprise classes, 80 known, for every way, on both eyes | The project's few-example goal, and which piece does the work |
| S2 | The same at 1 and 10 pictures | Whether a fix that helps at 3 pictures also helps at 1, and stops mattering at 10 |
| S3 | The same with 10, 40 and 80 known classes | Whether knowing more makes the next class easier. This is the property a growing system needs |
| S4 | Rehearsal score against real surprise score | Whether a rehearsal is a trustworthy way to pick settings |
| S5 | All 100 classes together: score on new-class pictures, on old-class pictures, and old classes with and without the new ones present | Whether few-picture classes live alongside well-learned ones |
| S6 | Time, GPU energy and peak GPU memory | Cost; the extra calculations should add seconds, not minutes |

## 5. How the result will be judged (written before the run)

All scores are means of 5 seeds on the 20 surprise classes. The fixed targets come from
EXP-8: patch eyes so far **21.2%**, frozen backprop eyes **25.6%**, grow and lock
**29.8%**, retrain **38.5%**, all from 3 pictures with 80 known classes.

**Step 0: are the earlier numbers rebuilt?** Within 2 points each, from 3 pictures: patch
eyes W0: 17.0%; patch eyes W1 at smoothing 100: 21.2%; frozen eyes W0: 25.2%.

**Step 1: the main verdict.** Patch eyes, selected way, 3 pictures, 80 known classes:

| Score | Verdict |
|---|---|
| 29.8% or more | **works**: no-backprop eyes learn a new class as well as the grown, trained eyes |
| 24.2% to 29.8% (a gain of 3 points or more) | **helps, but not enough** |
| under 24.2% | **does not work** |

**Step 2: does knowing more help?** Patch eyes, selected way, 3 pictures: the score with
80 known classes minus the score with 10 known classes. A rise of 3 points or more counts
as **yes**. (W0 uses nothing from the known classes, so it cannot rise.)

**Step 3: which piece does the work?** The ladder W0 → W2 → W3 → W3+B on patch eyes, each
at its rehearsal setting. A step counts if it adds 1.5 points or more.

**Step 4: do the same fixes help trained eyes?** Frozen backprop eyes, selected way,
against 25.6%: a gain of 3 points or more counts as **yes**.

**Step 5: can the rehearsal be trusted?** If the way selected in the rehearsal is also
the best, or within 1 point of the best, on the real surprise classes: **yes**.

**What I expect, written down now.**

- Tidy cards (W2) on patch eyes: about **19%**, a little better than plain cards (17.0%)
  and below the heavily smoothed table (21.2%).
- Learned view (W3): about **25%**. With borrowing about **26%**. So the verdict would be
  **"helps, but not enough"**: level with frozen backprop eyes, under grow and lock.
- 1 picture about 15%, 10 pictures about 38%.
- Knowing more helps: about 19% with 10 known classes, 25% with 80. **Yes.**
- Frozen backprop eyes: about 27% at best, a gain under 3. **No.**
- All 100 together: new classes from 3 pictures score low, about 12%; old classes lose
  about 2 points.
- The rehearsal will pick the right way.

My score predictions were right in EXP-8 for the main lines and wrong for grow and lock
and for the 3-picture memories. If any of this is wrong, it stays in the journal as written.

## 6. What carries over beyond pictures

| Part | Depends on pictures? |
|---|---|
| The two kinds of eyes | yes |
| Tidying, the learned view, borrowing | no. They work on any lists of numbers with class labels, whatever produced them |
| The rehearsal (pretend the newest classes are unknown, to choose settings) | no |
| The question "does knowing more classes make the next one easier?" | no. It is the question any system that keeps learning must answer, for words as much as for pictures |

## 7. What could go wrong

- **The papers suggest a limit.** No paper read shows eyes that learned nothing from labels
  matching trained eyes from a few pictures. The honest expectation is "less bad", and the
  rule in section 5 is written so that this counts as "helps, but not enough", not as success.
- **The learned view may fit the known classes too closely.** The directions that separate
  80 known classes need not separate a new one (ears and tails do not help with a
  bicycle). Stronger smoothing and keeping fewer directions are the guards; the rehearsal
  chooses among them.
- **With 10 known classes the learned view has only 9 directions.** It may then be worse
  than plain cards. That would be a finding about how much must be known before this pays.
- **Rehearsal and test may disagree.** The rehearsal has 60 known classes and other new
  classes. Step 5 checks this.
- **The test is 20 classes among themselves**, as in every earlier experiment. The
  100-class check (S5) is harder and is the more realistic one; it is reported but the
  verdict does not rest on it.
- **Many settings are tried** (20 for the learned view). They are chosen on classes that
  are not the test classes, which is the protection; small differences between settings
  should still not be read into.
- **The comparison targets are quoted, not rebuilt** (grow and lock 29.8%, retrain 38.5%).
  Step 0 checks that the parts that are rebuilt match EXP-8.
- **5 seeds, one dataset, images only.**

## 8. Time and disk

| | Expected |
|---|---|
| Run time | about 30 minutes on a free Kaggle GPU for all 5 seeds. From measured times per seed in EXP-8: patch eyes with tables 24 s, frozen eyes 24 s; the new calculations reuse the same tables. The weekly free allowance is about 30 hours, of which about 22 are used |
| Disk in this folder | about 0.15 GB for the Python environment (figures and PDF), plus under 0.05 GB of results and figures |
| Downloads to the laptop | the small Python environment only (about 0.15 GB, asked for before it is made). A short code check on the laptop borrows EXP-2's environment and CIFAR-100 without changing them |
| GPU memory | EXP-8 used 5.6 GB for patch eyes because all descriptions were held on the GPU. This code will hold them in ordinary memory and send them to the GPU in pieces, aiming to stay under 4 GB. The peak will be reported; if it is over 4 GB, Part B will say so |

## 9. Words used in this journal

| Word | Meaning here |
|---|---|
| Class | One kind of thing to recognise |
| Eyes (features) | The part that turns a picture into a list of numbers |
| Description | The list of numbers the eyes give for one picture |
| Memory | The part that stores what was learned about each class and gives the answer |
| Card | The average description of one class |
| Shared table (covariance) | A table recording which numbers of a description tend to rise and fall together inside a class, kept as a running sum |
| Smoothing | How strongly the shared table is evened out before use. More smoothing trusts the table less |
| Known classes | The classes the memory has already learned from many pictures (10, 40 or 80 of them) |
| Surprise classes | 20 classes never used to build anything, learned from 1, 3 or 10 pictures each |
| Tidy | Square root, subtract the average description of the known classes, scale to length 1 |
| Direction | A fixed recipe for combining the numbers of a description into one new number |
| Learned view (discriminant directions) | The few directions along which known classes differ most, relative to how much pictures of one class vary. Computed in one step from the cards and the shared table |
| Borrowing | Mixing the cards of the nearest known classes into a new class's card |
| Rehearsal | Choosing settings by pretending the most recent known classes are new. Uses no surprise class |
| Backpropagation | The usual way to train a network: measure the error and push corrections back through all layers, many times |
| Grow and lock | Add units for each new group of classes; never change old units (EXP-3 to EXP-6) |
| Seed | One repeat with a different random shuffle |
