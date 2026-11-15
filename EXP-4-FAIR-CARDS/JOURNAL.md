# EXP-4 · FAIR CARDS

**Experiment ID:** EXP-4 · **Part A written:** 2026-11-15, before any code was run · **Hardware:** free Kaggle GPU (the method must still fit the RTX 3050 laptop, 4 GB) · **Status:** PLANNED, NOT RUN YET

This journal has two parts. **Part A** (this part) explains the experiment before it starts.
**Part B** will be added to the end of this same PDF when the experiment finishes. Part A
will not be edited after results exist, except for the status word in the line above.

## 1. Why this experiment

### What happened before

The project thinks of a learning system as two parts: the **eyes**, which turn a picture into
a list of numbers describing it, and the **memory**, which keeps one **memory card** per class
(the average description of that class) and answers with the nearest card.

- **EXP-1** was a plan only. It was never run and has no readings.
- **EXP-2** showed that the memory part works when the eyes are good, and that the eyes are
  the bottleneck. Its eyes were frozen, so they could never improve.
- **EXP-3** let the eyes keep learning by **grow and lock**: for every new group of classes,
  add new units and lock the old ones. Its readings on photos, after 8 groups of 10 classes
  (from `EXP-3-GROWING-FEATURES/results/summary.json`):

| Way | Still knows first group | All 80 classes | New classes from 3 pictures | All 80, open-book |
|---|---|---|---|---|
| Frozen eyes (never learn again) | 55.3% | 32.3% | 25.6% | 32.3% |
| Ordinary training | 0.0% | 14.5% | 32.0% | 40.4% |
| Grow and lock | 60.4% | 29.9% | 30.0% | 40.4% |
| Retrain everything (ceiling) | 54.6% | 52.6% | 39.0% | 52.6% |

### What failed in EXP-3

Grow and lock did two things right: it did not forget (60.4% on the first group, against 0.0%
for ordinary training), and its eyes got somewhat better at new things (30.0% against 25.6%).

But its overall score, 29.9%, was **below frozen eyes at 32.3%**. A system that keeps learning
ended up worse than one that stopped on day one. By the rule written before that run, the
verdict was "does not work".

### The clue

In the **open-book check**, every card was rewritten using the old pictures. This is not
allowed in the real test; it only shows what the eyes could do with perfect cards. The same
eyes then scored **40.4%**. So about 10 points are being lost in the memory step, not in the eyes.

The suspected reason: the cards have different sizes. A card written in stage 1 describes the
32 units that existed then. A card written in stage 8 describes 144 units. EXP-3 compared a
picture with each card using only the units that card has. It looks as if short, old cards
win too often. Score per group in EXP-3, oldest group first:

| Group | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Real test | 60.4 | 38.6 | 34.1 | 28.2 | 25.1 | 19.7 | 19.4 | 13.3 |
| Open-book | 44.4 | 38.8 | 41.9 | 42.7 | 43.1 | 37.4 | 41.8 | 32.8 |

This is a reading of a pattern. EXP-3 did not record which wrong class was chosen, so the
cause was never checked directly.

### So EXP-4 asks

**Can cards of different sizes be compared fairly, without keeping any old pictures, so that
the improvement in the eyes shows up in the overall score?** And first: **is the suspected
cause the real one?**

# Part A: the plan (written before the run)

## 2. The question in plain words

Imagine a quiz with 80 answer cards. The oldest cards were written when the system could only
notice 32 things about a picture. The newest were written when it could notice 144 things. A
picture comes in, and each card says how well it matches. If short cards tend to say "good
match" more easily than long cards, the old classes win arguments they should lose.

EXP-4 keeps the eyes exactly as in EXP-3 and changes only **the rule for comparing a picture
with the cards**. Eight rules are tried on the same eyes.

## 3. The eight rules

None of the rules keeps any old picture. Some keep a few extra numbers per card. Rules 2 to 8
were designed by us for this experiment; no literature search was done for them (see section 8).

| | Name | In plain words | Extra things stored |
|---|---|---|---|
| R0 | **As in EXP-3** | Compare the picture with each card using only the units that card has | nothing |
| R1 | **Blank fill** | Treat the units an old card does not have as zeros, and compare at full length | nothing |
| R2 | **Group-centred** | For each picture, look at the scores of the 10 cards of one group and subtract that group's average. A group whose cards all score high gets no advantage | nothing |
| R3 | **Group-standardised** | Like R2, and also divide by how spread out that group's scores are. Asks "does one card in this group stand out?" | nothing |
| R4 | **Stranger-calibrated (running)** | Each card remembers how high it usually scores on pictures that are *not* its class ("strangers"), measured on the training pictures of every stage as they pass. A match counts only by how far it rises above that usual level | 2 numbers per card |
| R5 | **Stranger-calibrated (same crowd)** | Like R4, but every card is measured against the same crowd: the newest stage's training pictures | 2 numbers per card |
| R6 | **Spread-aware card** | The card also stores how much each unit varies within the class. The score is the average closeness per unit, so the number of units does not matter | one more list per card |
| R7 | **Predicted fill-in** | When new units are added, learn from the current pictures how the new units' values follow from the old units' values. Use that to guess the missing part of every old card, so all cards have full length | nothing after the guess |

**Example for R4 with cats and digits.** The "cat" card is short. Suppose it gives a match
score of 0.80 to almost any picture, including digits. The "7" card is long and gives 0.50 to
most pictures. A picture of a 7 scores 0.82 on "cat" and 0.75 on "7". Under R0, "cat" wins.
Under R4, "cat" is only 0.02 above its usual level while "7" is 0.25 above its usual level,
so "7" wins.

## 4. What is run, step by step

1. **Rebuild the eyes of EXP-3**, with the same settings: grow and lock on photos (CIFAR-100,
   80 classes in 8 stages of 10, starting at 32 units per layer and adding 16 per stage) and
   on digits then clothes (2 stages). Old pictures are thrown away after each stage, as before.
2. **Rebuild the two reference points**: frozen eyes (trained on the first group only) and
   retrain everything (a fresh network trained on all classes at once). Retrain is only built
   for the final stage, because the verdict only uses the final stage. Ordinary training is
   not repeated; EXP-3 already showed it collapses.
3. **Apply all eight rules to the same grown eyes** and record the score of each.
4. **Record every answer.** For each test picture and each rule, the chosen class is saved, so
   we can count where the wrong answers go.
5. Repeat with 5 seeds (5 different class orders and random starts), as in EXP-3.

**Practice half and exam half.** Trying eight rules and then reporting the best one on the
same pictures would flatter the result. So the test pictures are split once, at random, into
two halves. The **practice half** is used to pick the best rule. The **exam half** is used
for the verdict. The split is the same for every rule. Picking a rule this way is something
the researcher does; the system itself still never sees an old picture.

**Settings fixed now.**

| Setting | Value |
|---|---|
| Network, sizes, training | exactly as EXP-3: 4 layers, 30 passes per stage, same learning settings |
| Seeds | 5 (the same seed numbers as EXP-3) |
| Test split | half practice, half exam, per class, fixed in advance |
| Rule selection | highest average score on all 80 classes on the practice half of the photos run, among rules that pass step 3 of section 6 on the practice half |
| Digits then clothes | reported for every rule, not used to select |

## 5. What is measured

| | Measurement | Plain meaning | Why it matters |
|---|---|---|---|
| N1 | Accuracy on **all classes**, for each rule | Overall score | This is the number that fell below frozen eyes in EXP-3 |
| N2 | Accuracy **per group**, oldest to newest, for each rule | Is any age of class favoured? | A fair rule should give a fairly level line |
| N3 | Accuracy on the **first group** | Is old knowledge still kept? | A rule could "fix" the bias by sacrificing old classes |
| N4 | **Where the wrong answers go**: to an older group, a newer group, or the same group | Do old cards steal answers? | This checks the suspected cause directly |
| N5 | **Open-book score** on the same eyes (not counted) | What perfect cards would give | The most any rule could hope to recover |
| N6 | Time and extra storage of each rule | What the rule costs | A rule that needs heavy computing is not a fix |

The 3-picture test on surprise classes is not repeated. It does not use the old cards, so no
rule here can change it. It stays at the EXP-3 value of 30.0%.

## 6. How the result will be judged (written before the run)

The verdict uses photos, after all 8 stages, on the **exam half**, averaged over 5 seeds, for
the one rule that was selected on the practice half.

**Step 0: did we rebuild EXP-3 correctly?** On the full test set, rule R0 must land within 3
points of EXP-3's 29.9%, frozen eyes within 3 points of 32.3%, and retrain within 3 points of
52.6%. If not, the cause is found before any verdict is given.

**Step 1: is the suspected cause real?** Under rule R0, take the wrong answers where the
chosen class belongs to a different group than the true class. Count the share that went to
an **older** group. With no bias this would be about 50%.

| Share going to an older group | Conclusion |
|---|---|
| 65% or more | Cause confirmed: old cards win too often |
| 55% to 65% | Unclear |
| under 55% | Cause not supported; the explanation in EXP-3 was wrong |

**Step 2: how much of the gap does the selected rule close?** Same formula as EXP-3:

> share of the gap closed = (grow and lock with the selected rule − frozen) ÷ (retrain − frozen)

**Step 3: is the old knowledge still kept?** On the first group, the selected rule must be no
more than 5 points below frozen eyes.

| Verdict | Condition |
|---|---|
| **Works** | closes at least 50% of the gap and passes step 3 |
| **Partly works** | closes 25% to 50% and passes step 3 |
| **Does not work** | closes under 25%, or fails step 3 |

The thresholds are the same as in EXP-3 on purpose. With EXP-3's numbers, "partly works"
needs about 37.4% and "works" needs about 42.5%. The open-book score was 40.4%, so "works" is
probably out of reach for any card rule. Two plainer numbers will also be reported: whether
the selected rule **beats frozen eyes at all** (the minimum for the idea to be worth keeping),
and how many of the roughly 10 lost points it wins back.

**What I expect, written down now.**

- Step 1: the cause will be confirmed, with 75% or more of those wrong answers going to older groups.
- The selected rule will be one of the two stranger-calibrated rules (R4 or R5).
- It will score about **36%** on all 80 classes: above frozen eyes (32.3%), but just under
  the 37.4% needed for "partly works". So my guess for the verdict is **"does not work", by
  a small margin, while beating the floor**. Reason: a fair comparison removes the bias, but
  an old card still describes its class with only the few units that existed when it was
  written. No rule can add information the card never had.
- R1 (blank fill) will swing the bias the other way and favour the newest classes.

If any of this is wrong, it stays in the journal as written.

## 7. What carries over beyond pictures

The owner's end goal is a general model that also handles language. So it matters which parts
of this work depend on pictures.

| Part | Depends on pictures? |
|---|---|
| The image filters inside the network | Yes. A language model would use different building blocks |
| Lock old units, add new units | No. Any network made of layers can be grown this way |
| Memory cards (average description per class) | No. A card is just a list of numbers |
| All eight comparison rules | No. They work on lists of numbers of different lengths, wherever those come from |

An open point for later, not solved here: a language model does not pick one class out of a
small set of cards. It predicts the next word. How "one card per thing learned" should look
for language is a design question for a later experiment.

## 8. What could go wrong

- **Eight rules, one dataset.** The best rule on these photos may not be the best elsewhere.
  The practice/exam split guards against lucky picks, not against this. Digits then clothes is
  a small second check.
- **No literature search was done for these rules.** Ideas like score calibration and
  estimating how old averages shift very likely exist in published work. We have not opened
  those papers, so none is cited. A rule that looks new here may be known.
- **R7 guesses from the wrong pictures.** It learns how new units relate to old units from
  the current classes, then applies that to old classes. The relation may not hold for them.
- **R4 and R5 measure "strangers" unevenly.** A new card's strangers are its 9 sister classes
  in the same stage; an old card's strangers are classes from later stages. That difference
  could itself tilt the result.
- **Training is not perfectly repeatable on a GPU.** The rebuilt eyes will not be identical to
  EXP-3's. Step 0 checks that they are close.
- **Half the test pictures means more noise.** The exam half has 50 pictures per class.
- **The ceiling for this experiment is low.** Even perfect cards gave 40.4%, far from 52.6%.
  Better eyes are a separate problem.

## 9. Time and disk

| | Expected |
|---|---|
| Run time | about 1 to 1.5 hours on a free Kaggle GPU for all 5 seeds (estimate from EXP-3's measured times) |
| Disk in this folder | about 0.2 GB: Python environment for figures and the PDF 0.15 GB (measured), results and figures under 0.05 GB |
| Downloads to the laptop | none. Datasets are downloaded on Kaggle's machine. A short code check on the laptop borrows EXP-2's environment and datasets without changing them |
| GPU memory | under 1.1 GB in EXP-3, well inside the laptop's 4 GB |
| Free disk now | 63 GB |

## 10. Words used in this journal

| Word | Meaning here |
|---|---|
| Class | One kind of thing to recognise: "dog", "7", "sandal" |
| Unit | One small detector inside the network |
| Eyes (features) | The part of the network that turns a picture into a list of numbers |
| Memory card (prototype) | The average description of all pictures of one class |
| Grow and lock | Add new units for new classes; forbid the old units from ever changing |
| Group, stage | 10 classes that arrive together; the moment they arrive |
| Match score | A number saying how well a picture fits a card. Highest score wins |
| Stranger | A picture that does not belong to the card's class |
| Calibrate | Adjust scores so that different cards can be compared on equal terms |
| Open-book | Rewriting all cards from old pictures. Forbidden in the real test, used only to see the best the eyes could do |
| Floor and ceiling | Frozen eyes and retrain everything, the two reference points |
| Practice half, exam half | The two halves of the test pictures: one to choose the rule, one to grade it |
| Seed | One repeat with a different random shuffle |

<div class="pagebreak"></div>

# Part B: the readings (to be added after the run)

**Not run yet.** When the experiment finishes, this part will be filled in, in the same PDF:

1. What was actually run, and anything that differed from Part A
2. The readings: tables and figures made from the result files
3. The verdict, using the rule in section 6
4. Was the expectation in section 6 right or wrong
5. What it means for the project
6. What is still unverified
7. What the next experiment should be, and why
