# EXP-6 · WHAT LIMITS THE EYES

**Experiment ID:** EXP-6 · **Part A written:** 2026-11-15, before any code was run · **Hardware:** free Kaggle GPU (the method must still fit the RTX 3050 laptop, 4 GB) · **Status:** PLANNED, NOT RUN YET

This journal has two parts. **Part A** (this part) explains the experiment before it starts.
**Part B** will be added to the end of this same PDF when the experiment finishes. Part A
will not be edited after results exist, except for the status word in the line above.

## 1. Why this experiment

### What happened before

The project thinks of a learning system as two parts: the **eyes**, which turn a picture into
a list of numbers describing it, and the **memory**, which keeps one **memory card** per class
(the average description of that class) and answers with the nearest card.

- **EXP-1** was a plan only. It was never run.
- **EXP-2**: the memory part works when the eyes are good (a new class from 3 pictures in
  about a millisecond). The eyes are the bottleneck.
- **EXP-3**: **grow and lock**. For every new group of 10 classes, add new units and lock
  the old ones. Nothing old was overwritten, but the overall score (29.9%) was below eyes
  that never learn again (32.3%).
- **EXP-4**: found why. Short old cards won too often. Rule **R7** (guess the missing part
  of each old card) lifted the score from 30.4% to 39.5%, close to the 41.0% that the best
  possible cards could reach on those eyes.
- **EXP-5**: tried to improve the eyes by giving the newly added units an extra training
  job. Readings on photos after 8 groups of 10 classes, exam half, 5 seeds, 30 passes
  (from `EXP-5-BETTER-EYES/results/summary.json`):

| | All 80 classes | Best possible cards on the same eyes (open-book) | 20 never-seen classes from 3 pictures |
|---|---|---|---|
| Frozen eyes (never learn again) | 32.8% | | 25.8% |
| Grow and lock, as before | 39.6% | 40.9% | 29.6% |
| Grow and lock, best extra job | 40.1% | 41.3% | 29.8% |
| Retrain everything (ceiling) | 53.2% | | 38.8% |

### What failed in EXP-5

The extra jobs changed nothing: 40.1% against 39.6%, and no movement at all on never-seen
classes. EXP-5 was a fix for a cause that had only been guessed ("the new units' job is too
narrow"). The guess was wrong, or at least that kind of fix does not reach it.

EXP-5 did settle one thing. Old knowledge is kept: the first 10 classes, graded with only
their own 10 cards, went from 73.4% to 70.8% over seven more stages. So forgetting is not
where the points go.

### What is still open

The eyes of grow and lock are stuck about 12 points under retraining (40.9% open-book
against 53.2%). Three suspects remain, and none has been tested:

1. **Narrow view.** The units added at a stage only ever see that stage's 10 classes.
2. **Too few units.** Only 16 units are added per stage. One clue: on the same first 10
   classes, a 144-unit network scored 79.8% and grow and lock's 32-unit start 73.4%.
3. **Locking and building in steps.** Old units can never improve, and the network is
   built piece by piece instead of all at once.

### So EXP-6 asks

**Which of these three holds the eyes back, and by how much?** This is a diagnosis. It does
not try a new method. It switches the suspects off one at a time and reads what moves.

# Part A: the plan (written before the run)

## 2. The question in plain words

Think of a school that hires one new teacher every term and never retrains the old ones.
The school is weaker than one where all teachers train together. Why?

- Maybe because each new teacher only meets this term's 10 subjects (narrow view).
- Maybe because one teacher per term is too few (too few units).
- Maybe because the old teachers are never allowed to learn anything new (locking).

To find out, we run the school again with one thing changed each time.

**Cats and digits example.** Round 1 is cats against dogs, round 2 is the digits. Under
"narrow view" switched off, the units added in round 2 are trained on cats, dogs *and*
digits together. That needs the old cat and dog pictures, which our real method is not
allowed to keep. Here it is allowed as a test only, like a doctor's scan: it is not the
treatment, it tells us where the problem is.

## 3. The three runs

All three use grow and lock, the same training as EXP-4 (plain "tell the classes apart"),
30 passes, 8 stages of 10 classes.

| | Name | New units per stage | What the new units are trained on | Allowed in the real method? |
|---|---|---|---|---|
| A | **As now** | 16 (start with 32). Final size 144 | the 10 new classes only | yes |
| B | **Double units** | 32 (start with 64). Final size 288 | the 10 new classes only | yes |
| C | **Wide view** (diagnosis only) | 16 (start with 32). Final size 144 | all classes so far, using the old pictures. Old units stay locked | **no**, it keeps old pictures |

Reference points, built fresh: **retrain everything** at size 144 and at size 288, and
**frozen eyes** at size 144 and at size 288.

**How the three suspects are read from these runs.**

- **Narrow view** = C minus A. Same size, same locking; the only difference is what the
  new units get to see.
- **Locking and building in steps** = retrain (size 144) minus C. Both see all pictures
  and have the same size; the only difference is that C is built in locked steps.
- These two add up to the whole gap between A and retrain.
- **Too few units** = does doubling the units shrink the gap? Compare the gap at size 144
  (retrain 144 minus A) with the gap at size 288 (retrain 288 minus B). If both are about
  the same, bigger is simply better for everyone and size is not what causes the gap.

A fourth run, "double units and wide view together", is left out. It would roughly double
the run time and is not needed to tell the three suspects apart.

**Settings fixed now.**

| Setting | Value |
|---|---|
| Dataset | photos (CIFAR-100): 80 classes in 8 stages of 10, plus the same 20 surprise classes |
| Network, training | as EXP-3 to EXP-5: 4 layers, 30 passes, same learning settings |
| Card rule in the real test | R7, as in EXP-4 and EXP-5 (runs A and B) |
| Seeds | 5, the same seed numbers as before |
| Numbers used | exam half of the test pictures, the same half as in EXP-4 and EXP-5, so the numbers can be compared directly |

## 4. What is measured

| | Measurement | Plain meaning | Why it matters |
|---|---|---|---|
| M1 | **Open-book score** on all 80 classes | How good the eyes are, with perfect cards | The main reading. It measures the eyes without mixing in the card problem, and it is the only fair way to compare run C with the others |
| M2 | **3-picture test on 20 surprise classes** | How good the eyes are on things never trained on | A second, independent reading of the eyes |
| M3 | Real test with R7, all 80 classes (runs A and B) | The score the real method gets | Shows whether double units is a usable gain |
| M4 | First group with only its own 10 cards, after every stage (A and B) | Real forgetting | Checks that more units does not bring forgetting back |
| M5 | Training time, GPU energy, peak GPU memory | Cost | Double units must fit the laptop and must earn its cost |

## 5. How the result will be judged (written before the run)

This experiment names causes. It does not give a "works" verdict for a method, except for
double units, which is an allowed change.

**Step 0: is EXP-5 rebuilt correctly?** Run A must give open-book within 3 points of 40.9%
and R7 within 3 of 39.6%. Retrain at size 144 must be within 3 of 53.2%, frozen at size
144 within 3 of 32.8%.

**Step 1: split the gap at size 144.** Gap = retrain 144 − A, on the open-book score
(about 12 points in EXP-5).

> narrow view share = (C − A) ÷ gap  locking share = (retrain 144 − C) ÷ gap

| Share of the gap | Name given to that suspect |
|---|---|
| 50% or more | **main cause** |
| 25% to 50% | **contributing cause** |
| under 25% | **minor** |

**Step 2: does size cause the gap?** Gap at 288 = retrain 288 − B (open-book).

| Reading | Conclusion |
|---|---|
| Gap at 288 is at least 25% smaller than the gap at 144 | too few units is a cause of the gap |
| Otherwise | size is not a cause of the gap. Bigger helps everyone equally |

**Step 3: is double units a usable gain?** On the real test with R7:

> share = (B − A) ÷ (retrain 144 − A)

50% or more: **works**. 25% to 50%: **partly works**. Under 25%: **does not work**. It
must also keep the first group (M4 drop of 3 points or less) and stay under 4 GB of GPU
memory. Note that B is being compared with a ceiling of the smaller size on purpose: the
question is whether doubling the units buys back the points lost to retraining.

**What I expect, written down now.**

- Run C (wide view): open-book about **47%**. That would put narrow view at about half the
  gap and locking at about half. So I expect **both to be real causes, neither one alone**.
- Run B (double units): open-book about **44.5%**, R7 about **43%**. Retrain at size 288
  about **57%**. So the gap at 288 would be about 12.5 points, the same as at 144: **size
  is not a cause of the gap**. As a usable gain, B would close about 25% of the gap:
  on the border between "does not work" and "partly works".
- 3-picture surprise test: C about 34%, B about 31.5% (A is 29.6%, retrain 38.8%).
- Forgetting in B stays under 3 points.

My last two main predictions were wrong (EXP-4 and EXP-5), so these should be read as
guesses. If any of this is wrong, it stays in the journal as written.

## 6. What carries over beyond pictures

| Part | Depends on pictures? |
|---|---|
| Image filters inside the network | yes |
| The three suspects (narrow view, too few units, locking) | no. Any network that grows by adding locked pieces faces the same three questions |
| The way of telling them apart (change one thing, use old data as a scan) | no |

Whichever cause this run names, it names it for image filters on small photos. The split
could come out differently for language. The method of asking carries over; the answer
has to be measured again.

## 7. What could go wrong

- **The suspects may not add up neatly.** Narrow view and size could depend on each other
  (more units might only help when the view is wide). The run that would show this was left
  out to save time. If the readings look odd, Part B will say so and propose that run.
- **Run C is not a method.** It keeps every old picture. A good score for C tells us what
  is missing, not how to supply it without old pictures.
- **Run C trains more.** Its new units see more pictures per stage (all classes so far, 30
  passes), so it also gets more training steps. Part of its gain could come from that. The
  locking share is not affected by this, because retrain also sees everything.
- **"Locking" lumps two things together**: old units cannot improve, and the network is
  built in steps. This run cannot separate those two.
- **Double units costs about four times more** computing than run A. Its time, energy and
  memory are recorded.
- **One dataset, one kind of network, 5 seeds.**
- **Training is not perfectly repeatable on a GPU.** Step 0 checks the rebuild is close.

## 8. Time and disk

| | Expected |
|---|---|
| Run time | about 3.5 hours on a free Kaggle GPU for all 5 seeds. Estimate from EXP-5's measured times per seed (A 128 s, retrain 191 s, frozen 24 s), with size 288 taken as 4 times size 144 and run C as 4.5 times run A |
| Disk in this folder | 0.15 GB measured for the Python environment (figures and PDF), plus under 0.05 GB of results and figures |
| Downloads to the laptop | none. The dataset is downloaded on Kaggle's machine. A short code check on the laptop borrows EXP-2's environment and dataset without changing them |
| GPU memory | A used 0.9 GB in EXP-5. Size 288 is expected under 2.5 GB, inside the laptop's 4 GB. It will be measured |

## 9. Words used in this journal

| Word | Meaning here |
|---|---|
| Class | One kind of thing to recognise: "dog", "bus", "7" |
| Unit | One small detector inside the network |
| Size | The number of units in each layer of the network |
| Eyes (features) | The part of the network that turns a picture into a list of numbers |
| Memory card (prototype) | The average description of all pictures of one class |
| Grow and lock | Add new units for new classes; forbid the old units from ever changing |
| Stage, group | The moment 10 new classes arrive; those 10 classes |
| Pass | One time through all the pictures of a stage |
| R7 | Guess the missing part of an old, short card from the part it has |
| Open-book | Rewriting all cards from old pictures. Forbidden in the real test; shows how good the eyes are |
| Diagnosis run | A run that breaks the rules on purpose (here: keeps old pictures) to locate a problem. It is not a candidate method |
| Gap | The distance in points between grow and lock and retraining |
| Surprise classes | 20 classes the eyes are never trained on, learned from 3 pictures each |
| Floor and ceiling | Frozen eyes and retrain everything, the two reference points |
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
