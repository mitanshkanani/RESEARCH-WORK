# END GOAL
### What we are trying to build, how we plan to get there, and what needs human verification

**Author:** MK · **Date:** 2026-09-20 · **Status:** for review by Aadit
**Bracketed citation numbers `[N]` refer to `DEEP_RESEARCH_NO_RETRAINING.md`** (77 sources, generated 2026-09-20). Anything not bracketed is named inline.

**How to read this document.** Section 8 is a claims ledger: every load-bearing statement, its evidence status, and its source. Section 9 lists the specific things I need you to check with your own eyes, and Section 10 lists citations that are already known to be fake. Please try to break the argument, not to agree with it.

---

## 1. The end goal, stated as a capability

We want a model that **learns new things without a retraining phase**. Concretely:

- Give it a reference — documentation, a tutorial, a public repository — and it becomes able to *do* the thing, not just recite it.
- Show it five labelled examples of a category and it knows that category, including knowing when something is *not* it, without being shown negatives.
- It keeps everything it has learned. Adding skill B does not damage skill A.
- It can eventually notice its own gaps, seek the information that closes them, and improve the process by which it does so.

The working formulation I want us to use, because it is the one that can actually be tested:

> **Make parameter updates so cheap, so local, and so safe that retraining stops being the thing we do.**

## 2. What the goal is explicitly *not*

Three disclaimers, so nobody over-reads section 1.

- **It is not "brain-like" as an objective.** Backprop-free, Hebbian, spiking, predictive-coding — those are candidate *means*. If the capability is achieved with ordinary gradient descent on a GPU, that counts as success. I said this plainly on 2026-09-20 and it should stay in writing.
- **It is not AGI, consciousness, self-awareness, or unrestricted self-improvement.** Your `PROJECT_STATE.md` already says this and it is the right call: an unfalsifiable claim gets our real claim rejected with it.
- **Energy efficiency is a downstream consequence, not the target.** It is worth reporting, not worth chasing as the contribution.

## 3. The constraint that does most of the work: where can new knowledge live?

Any acquired skill must be stored *somewhere*, and there are only three places. This trilemma is the skeleton of the whole project:

| Where knowledge goes | Mechanism | What it costs |
|---|---|---|
| **Into existing weights** | gradient updates — backprop, local rules, PEFT | this *is* training; the question is only how local and how safe |
| **Into new structure** | module growth, per-task experts, skill libraries | linear growth; skills stay isolated unless composition is demonstrated |
| **Outside the model** | retrieval, prototypes, episodic memory | zero weight change — but it yields recital, not competence |

The second and third rows are where our hopes live. The evidence says both are real and both have hard ceilings.

## 4. What the research says about each route

**Cheap local weight updates.** The field's own 2026 survey states the consensus: parameter-efficient methods "remain vulnerable to catastrophic forgetting" [1]. Low rank limits how much an update *can* damage, not whether it does. Null-space projection (GPM, Adam-NSCL) has genuinely clean theory — project new gradients away from stored input directions and old-task outputs survive to first order [2][4] — but it is validated on Split-CIFAR-scale CNNs with roughly 100 stored directions, and the field already had to soften hard orthogonality into a scaled version because the hard constraint destroyed plasticity [3]. Architecture growth (Progressive Networks, PackNet) achieves zero forgetting by construction and pays in linear capacity [10][13].

Two numbers I want us to hold onto. LoRA does forget less than full fine-tuning — but the measurement is on *out-of-target-domain* tasks, and it simultaneously learns less on the target [14]. And full fine-tuning gains ~2% in-distribution while losing ~7% out-of-distribution versus linear probing, because "the lower layers of the neural network change simultaneously and distort the pretrained features"; the repair is to learn the head *first*, then fine-tune, which buys ~10% OOD [15]. **How we update may matter more than how much.**

**Fast weights and test-time learning.** This is the closest published architecture to our thesis. Titans updates a memory module's weights *during inference* by online gradient descent driven by a surprise loss [20]. TTT layers make the hidden state itself a small model trained on the incoming stream [21]. Nested Learning reframes the whole field as optimisation problems running at different update frequencies [18][19][24].

The catch is the one that matters: every one of these layers has an inner objective, inner learning rate, chunk size and update rule that were themselves learned by backprop-through-updates during a full pretraining run. **Retraining is relocated, not removed.** Nested Learning's own abstract says continual learning is only "**potentially** unlocked" and reports no quantitative results in its abstract [18]; its continual-learning evidence is class-incremental text classification on CLINC/Banking77/DBpedia, not a long non-stationary stream [18]. The 2026 wave still calls itself "a promising step towards" continual learning [22][25][26], and a 2026 survey asserts the connection without introducing a benchmark that would test it [27].

**Weight-free memory, prototypes and skill libraries.** Real verified wins with zero weight updates: RETRO matches models 25× its size by retrieving from 2 trillion tokens [44]; kNN-LM adapts to a new domain by swapping a datastore [43]; Voyager discovers 3.3× more items and unlocks milestones up to 15.3× faster with no fine-tuning, storing skills as executable code [53]; FunSearch produced new mathematics with no backprop at all [72].

The boundary is precise. Models reliably *answer* questions whose evidence is in context yet fail to *copy or transform* that same content [47]; multi-hop accuracy is U-shaped in evidence position [48]; knowledge lives as sparse circuits, so context can only reactivate circuits pretraining already built [39]; capacity is capped at ~2 bits of knowledge per parameter [41]. And the five-dog case works **because a prior training run already made dogs separable** — few-shot learning is a property of the representation, not of the learner [49][50][51].

**Self-improvement from feedback.** It works inside a narrow envelope, and the envelope is the verifier. Absolute Zero needs a code executor [57]; SWE-Gym keeps only issues with reproducible sandbox test runs [59]; R-Zero substitutes consensus agreement for ground truth, which is the configuration the collapse literature flags as unsafe [58]. The theory: self-improvement is *sharpening* — concentrating probability mass on answers the base model already produces — so it cannot exceed the base model's support [63], and SPIN's fixed point is the human data distribution [71]. Model collapse eats distribution tails first, which is exactly what an agent needs in order to notice its own gaps [60].

## 5. The wall, in one paragraph

All four routes meet the same constraint in costume. Route 2 needs a slow outer loop. Route 3 needs structure someone already paid for. Route 4 needs a cheap, deterministic, ungameable verifier. Nobody has shown a system acquiring a genuinely new function after deployment with none of the three. That is not proof of impossibility — it is the honest map of where the difficulty actually is, and it tells us what to build instruments for.

## 6. What your Phase 1 already told us

Your Phase 1 result is the most relevant evidence either of us has, and I think we should read it as evidence *about the goal*, not just as a protocol failure. A frozen model given full trusted documentation scored **0/24**, and failed identically with and without the docs (Go output instead of GOCO). That is our scenario, run honestly, and the answer at frozen weights was no.

Two implications. First, the "read the docs, no weight change" version of section 1 is now empirically disconfirmed at least for a 1.5B coder model on an unfamiliar language. Second — and this is the useful part — the failure was a **floor effect, not an equivalence**, which means the instrument cannot yet measure the thing we care about. Phase 1R is therefore not a detour from the research question; it *is* the research question at its first step.

One genuine piece of good news I found while researching: **the reversal curse does not apply in context.** Models trained on "A is B" fail "B is A" catastrophically — 96.7% forward versus 0.1% reverse — but the same paper states that when the fact appears *in context*, models can deduce the reverse [54]. So that particular doom is about gradient-trained knowledge, not about documentation-based acquisition. It is the strongest empirical argument I have found *for* our reading-then-doing hypothesis, and it deserves a careful look.

## 7. Where I think the opening is

**Nobody has a long-horizon, non-stationary benchmark that tests whether test-time learning delivers continual learning.** The 2024-2026 literature evaluates long *contexts*, not drifting *tasks* [18][22][25][26][27]; the single 2026 result pairing test-time training with a real stream is video domain-incremental, not language [28]. That is a measurable, small-compute, high-leverage hole, and it is the kind of contribution the field cites: the stability-gap paper and the three-scenarios paper (van de Ven et al., *Nature MI* 2022) both became standard by measuring other people's claims properly, not by building bigger models.

A second, cheaper opening that uses your infrastructure directly: **the compositionality probe.** Does a stored skill survive inversion, paraphrase, and composition? Voyager-style libraries store and retrieve; no paper I found demonstrates that composed skills transfer to a neighbouring problem, and the acknowledged failure mode is silent-but-wrong programs entering the library because they executed [53]. GOCO is close to an ideal testbed for this because the compiler is an objective referee.

## 8. Claims ledger

| # | Claim | Status | Source |
|---|---|---|---|
| C1 | PEFT methods remain vulnerable to catastrophic forgetting | **Verified** — fetched, quoted | [1] |
| C2 | LoRA forgets less than full FT *on out-of-target tasks* while learning less on target | **Verified** — abstract read | [14] |
| C3 | Full FT +2% ID / −7% OOD vs linear probing; LP-FT +1% / +10% | **Verified** — abstract read | [15] |
| C4 | Titans updates memory weights during inference via online gradient descent | **Verified** — abstract read | [20] |
| C5 | Nested Learning claims continual learning only "potentially"; abstract has no quantitative results | **Verified** — abstract read | [18] |
| C6 | Hinton is not an author of Titans / MIRAS / Nested Learning | **Verified** — author lists checked | [18][20][24] |
| C7 | No long-horizon non-stationary benchmark validates TTT as continual learning (as of 2026-09) | **Inferred** — absence of evidence across searched corpus; *needs adversarial check* | [18][22][25][26][27][28] |
| C8 | RETRO matches 25×-larger models with a 2T-token datastore (~4% of params, not 2%) | **Verified + corrected** | [44] |
| C9 | Voyager: 3.3× items, 2.3× distance, up to 15.3× faster milestones, no fine-tuning | **Verified** — abstract verbatim | [53] |
| C10 | ~2 bits of knowledge per parameter, even at int8 | **Verified** — abstract verbatim | [41] |
| C11 | Onboarding gains ~two orders of magnitude *between presentation formats*, only for composable knowledge | **Verified + corrected** (commonly misquoted as vs pretraining) | [42] |
| C12 | Reversal curse: 96.7% forward vs 0.1% reverse; **does not apply in-context** | **Verified** — incl. the caveat | [54] |
| C13 | Self-improvement is sharpening bounded by base-model support | **Verified** — preprint, not peer-reviewed | [63] |
| C14 | Model collapse loses distribution tails; possibly unavoidable | **Verified** — Nature 2024 + note [60][62] |
| C15 | ReST^EM does *not* claim one round + merging beats many (it says multiple iterations help) | **Verified + corrected** — a widely repeated error | [67] |
| C16 | Your Phase 1: frozen model + full docs = 0/24, floor effect | **Verified** — your own repo record | `PROJECT_STATE.md` |
| C17 | RANK shows retrieval degrades with reasoning depth | **UNVERIFIED** — paper exists, no numbers retrievable, venue unconfirmed | [40] |
| C18 | Absolute Zero ≈ +20 points over base | **UNVERIFIED** — qualitative claim confirmed, number not retrievable | [57] |

## 9. What I need you to check by hand

1. **C7 is the load-bearing claim for section 7 and it is an *absence* claim.** Absences are the easiest thing for a literature search to miss. Search specifically for long-horizon / non-stationary / task-drift evaluations of Titans, TTT layers, LaCT, or any test-time-memory architecture. If one exists, my recommended direction changes.
2. **C17 and C18 need the PDFs opened in a browser.** OpenReview blocked automated access. If RANK's numbers are real and strong, they either support or undercut section 4's Route 3 argument and I want to know which.
3. **C13's peer-review status** — I have it as a preprint. If it was published, cite the venue.
4. **Whether you agree the goal statement in section 1 is the right one**, and specifically whether "so cheap and local that retraining stops being the thing we do" is a formulation your Phase 1R protocol can actually falsify. If it cannot be falsified with your current instrument, tell me and we sharpen it again.
5. **Whether the compositionality probe (section 7) belongs to your protocol or mine.** If it uses GOCO's sealed suites it is yours; if it uses a separate toy domain it is mine. I do not want to accidentally contaminate your pre-registration.

## 10. Citations that are known to be fake — do not use these

During production of the source report, four citations were generated by an automated research agent with plausible authors and venues, and were later confirmed **not to exist**. They are listed so that neither of us reintroduces them:

- "Keeping Neural Networks Current with Test-Time Training" — attributed to Google / Alvarez-Melis / Hinton, 2024. Not on arXiv, not on DBLP, not on the named author's publication list. The nearest real 2024 item is about reweighting *training* data under concept drift [30].
- Everaert et al., "Is Future Learning Affected by Parameter-Space Constraints for Previous Tasks?" — attributed NeurIPS 2021. Absent from NeurIPS 2021/2022/2023 proceedings, arXiv, OpenReview, Crossref.
- Ghorbani, Krishnan & Xiao, "Gradient Descent Performs Projections onto Orthogonal Gradient Subspaces" — attributed NeurIPS 2021. Absent everywhere checked.
- Kumar et al., "Fine-Tuning, LoRA and Stochastic Freezing Differentially Affect LLM Robustness" — attributed NeurIPS 2023. Absent. The real Kumar et al. paper [15] does not study LoRA or LLMs.

Also flagged: **SESLR** (arXiv 2507.02901), an SNN sleep-wake continual-learning paper, was **retracted by its own authors for data errors**. And three arXiv IDs in circulation resolve to unrelated papers (RETRO is 2112.04426, not 2112.09114; Kumar et al. is 2202.10054, not 2106.09685 which is the LoRA paper; FunSearch has no arXiv preprint — cite Nature 625(7995):468-475).

**Why this section exists in our own document:** four fabricated citations in one careful, verified pipeline is the clearest possible demonstration that the failure mode we are positioning against is real. It is also the reason I trust your pre-registration discipline more than I trust any literature summary, including this one.

## 11. Provenance

This document was assembled from: my stated goal across this conversation; your `PROJECT_STATE.md` and Phase 1 record; and two automated deep-research passes (2026-09-19, 53 sources; 2026-09-20, 77 sources) run through a pipeline of parallel retrieval agents, targeted gap-fill agents, and a citation-verification agent, with all corrections logged in the source report's Methodology section. Every `[N]` resolves to a URL in that report's bibliography. Nothing here was written from memory without a fetched source, except the four items marked UNVERIFIED and the inference flagged as C7.

**Full source report:** `DEEP_RESEARCH_NO_RETRAINING.md` / `.pdf` · **First survey:** `DEEP_RESEARCH_HUMAN_LIKE_LEARNING.md` / `.pdf` · **Paper library:** three topic folders indexed by `PAPERS_MANIFEST.md`
