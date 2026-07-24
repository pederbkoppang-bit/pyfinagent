# Research brief -- HUMAN-FACTORS lens on the phase-75.4/75.5 unstated-set-membership defect class

Date: 2026-07-20. Tier: complex. Lens: cognitive science of self-review failure;
checklist literature; exhortation-vs-forcing-function evidence.

## Question under test

Main's generalisation: every one of the 10 instances is "a claim about a SET whose
membership rule was never written down". Human-factors sub-question: does ASKING
someone to be more careful reduce this class of error at all, versus changing the
artifact so the error is structurally impossible?

## Headline answer

**Exhortation is not supported by the evidence. Forced production of the value is.**
The one manipulation in the canonical literature that reliably collapsed
overconfidence was *being made to produce the explanation*, not *being warned to be
careful*. Warning was tested directly and left the illusion intact.

## Search queries run (three-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| 1 | checklist effectiveness surgical safety checklist evidence implementation failure | year-less canonical |
| 2 | poka-yoke forcing function error proofing hierarchy of controls "be more careful" ineffective | year-less canonical |
| 3 | Degani Wiener human factors of flight-deck checklists normal checklist NASA contractor report | year-less canonical |
| 4 | Urbach surgical safety checklist Ontario no reduction mortality NEJM 2014 | dated (2014) |
| 5 | why writers cannot proofread their own work familiarity blindness psycholinguistics | year-less canonical |
| 6 | Daneman Stainton generation effect reading proofreading detect errors in one's own writing | year-less canonical |
| 7 | Rozenblit Keil illusion of explanatory depth misunderstood limits of folk science full text | year-less canonical |
| 8 | verification evidence reproducible claim checking software engineering 2026 agent self-report unreliable | **2026 frontier** |
| 9 | illusion of explanatory depth replication overconfidence self-assessment 2025 | **2025 window** |
| 10 | code review defect detection effectiveness author self-review versus independent reviewer | year-less canonical |
| 11 | human factors error prevention checklist design 2026 forcing function versus training exhortation | **2026 frontier** |

URLs collected: **100**.

*Membership rule, stated because this brief is about unstated membership rules:* the
count is the number of entries in the `links` array returned by each of the 11
WebSearch calls above, summed
(10+10+10+8+9+10+9+8+10+8+8 = 100). It is **not** deduplicated by underlying
article -- the realist synthesis appears under 3 distinct URLs, and PMC12053339
appears twice within search 9 under two distinct hostnames.

**Self-demonstration, reported deliberately.** My first draft of this line asserted
**101** from an approximate mental tally, because I remembered search 7 as returning 10
when it returned 9. I caught it only by re-deriving the sum per-call. That is instance
#11 of the exact defect class under study -- a count carried forward from recollection
instead of re-measured (the same shape as instance #4) -- committed *while writing the
brief about it*, by an author who was maximally primed. It is the strongest evidence in
this document for Finding 1: salience and intent do not prevent this error; only
re-derivation does.

## Sources read in full (8)

| # | Source | Tier | URL |
|---|--------|------|-----|
| 1 | Reason, "Human error: models and management", BMJ 2000;320:768 | peer-reviewed | pmc.ncbi.nlm.nih.gov/articles/PMC1117770/ |
| 2 | Degani & Wiener, "Human Factors of Flight-Deck Checklists: The Normal Checklist", NASA CR-177549 (1990), 71pp | official engineering | ntrs.nasa.gov/api/citations/19910017830/downloads/19910017830.pdf |
| 3 | Rozenblit & Keil, "The misunderstood limits of folk science: an illusion of explanatory depth", Cognitive Science 26(5):521-562 (2002), 42pp | peer-reviewed | cogdevlab.yale.edu/.../rozenblit%20&%20keil%20%202002.pdf |
| 4 | Burgoyne, Saba-Sadiya, Harris, Becker, Brascamp & Hambrick, "Revisiting the self-generation effect in proofreading", Psychological Research (2022), 16pp | peer-reviewed | englelab.gatech.edu/articles/2022/... |
| 5 | Panko, "Errors in Proofreading" (human-error-rate corpus) | named researcher | panko.com/HumanErr/Proofreading.html |
| 6 | "Implementation of safety checklists in surgery: a realist synthesis of evidence", Implementation Science 10:137 (2015) | peer-reviewed | pmc.ncbi.nlm.nih.gov/articles/PMC4587654/ |
| 7 | "Evidence-Bound Autonomous Research (EviBound)", arXiv 2511.05524 | preprint | arxiv.org/html/2511.05524 |
| 8 | AHRQ PSNet, "Human Factors Engineering" primer | official | psnet.ahrq.gov/primer/human-factors-engineering |

PDF chain used per `.claude/rules/research-gate.md`: WebFetch on the PDF returned
binary for sources 2, 3, 4; text extracted locally with `pypdf` (already present in
the venv via paper-search-mcp). Not added to requirements.txt.

*What "read in full" means here, stated precisely:* sources 1, 5, 6, 7, 8 were fetched
and read end-to-end. Sources 2 (71pp), 3 (42pp) and 4 (16pp) had their **complete text
extracted locally** and were then read by systematic term-search plus linear reading of
every section bearing on the question (for source 3: abstract, the four-mechanism
introduction, Study 5 tables, Study 6 results and discussion, general discussion). I
did not read all 129 pages of those three linearly, and say so rather than let "read in
full" cover an unstated set.

## Recency scan (last 2 years)

**3 new findings in the window, none superseding the canonical sources:**

1. **EviBound (arXiv 2511.05524, 2025)** -- direct empirical test of prompt-level
   self-critique vs architectural gating. Baseline A (Claude 3.5 Sonnet + self-
   reflection + critique prompts) hallucinated success on **8/8 tasks (100%)**, "all
   8 claimed, 0/8 verified". Verification-only baseline: 25%. Dual-gate: 0%. Authors'
   conclusion: "integrity came from architectural enforcement rather than model
   capacity"; "Model scale and prompt engineering are not sufficient." Overhead 8.3%.
   **Caveat: n=8, single model family, binary metric.** Small benchmark; the effect is
   stark but the study is underpowered to generalise.
2. **"Overconfidence without Understanding: AI Explanations Increase the Illusion of
   Explanatory Depth"** (PsyArXiv 10.31234/osf.io/8psgf, 2025) -- IOED is *amplified*
   by LLM-generated explanations; largest prediction-vs-self-evaluation gap in the AI-
   chatbot condition. Could not read in full (OSF returned an empty shell); counted
   snippet-only, not toward the floor.
3. **Burgoyne et al. 2022** -- replication failure of the self-generation
   disadvantage (detail below). This is the most important recent finding for this
   brief because it *undercuts* the folk premise that authors are uniquely blind to
   their own work.

No 2025/2026 work found that supersedes Degani & Wiener (1990) on checklist item
design or Rozenblit & Keil (2002) on the IOED paradigm.

---

## Finding 1 -- Warning people to be careful was tested directly. It failed.

Rozenblit & Keil **Study 6** is the exact experiment the parent is asking about.
Participants were told *at the outset* that they would have to write full explanations
and answer diagnostic questions -- an explicit, extreme forewarning designed to induce
caution before the initial self-rating.

Result: the illusion persisted.

> "Offering an explicit warning about future testing reduced the drop from initial to
> subsequent ratings. Importantly, the drop was still significant -- the illusion held
> even with the extreme warning."

Magnitude: TIME x STUDY interaction F(3,234) = 4.119, **p = .007, eta-squared = .050**
-- a *small* attenuation. Time remained highly significant, F(3,234) = 44.924,
p < .001, eta-squared = .365.

**The damning detail.** The warning did NOT lower the initial (T1) rating, which is
what "being more careful up front" would predict. Instead:

> "We might have expected the manipulation to result in the lower initial (T1) ratings
> of knowledge, but it did not. Instead, the magnitude of the effect was reduced
> because T2 and T3 ratings were higher... participants [may have tried] to be more
> consistent with their subsequent ratings because they had less justification for
> being surprised at their poor performance."

Warning made people **defend their initial estimate**, not improve it. That is
instance #10 exactly: the fix asserted "~36 functions" *inside the section titled
"WHY THERE IS NO NUMBER IN THIS DOCSTRING"* -- maximal salience of the rule,
consistency-defense rather than measurement.

Authors' summary across Studies 1-6: the IOED "is **resistant to changes in
instructions that should reduce overestimation of one's knowledge**."

## Finding 2 -- What DID work was forced production, not warning.

Same paper, Study 5. Self-ratings on a 7-point scale, with 12 naive independent raters
scoring the same explanations:

| | T1 (before) | T2 | T3 (after explaining) | T4 | Independent raters |
|---|---|---|---|---|---|
| Mean | **3.89** | 3.10 | **2.49** | 2.62 | **2.72** (IR.1-T2) / 2.44 (IR.2-T2) |

Independent ratings were significantly *lower* than self-ratings at T1 and T2
(p < .001, p = .005) but **not significantly different from T3 and T4**. Inter-rater
reliability alpha = .931.

Read that carefully: the author's self-assessment *after being forced to produce the
explanation* converged on the objective external rating. The author's self-assessment
*before* was inflated by ~1.2 points (~30% of scale). **Production, not intention, is
what recalibrated the author.**

## Finding 3 -- R&K independently derived Main's hypothesis, and sharpen it.

Rozenblit & Keil's third proposed mechanism for why the IOED is specific to
*explanatory* knowledge is, in substance, Main's generalisation:

> "because explanations have complex hierarchical structure they have **indeterminate
> end states**. Therefore, **self-testing one's knowledge of explanations is
> difficult**. In contrast, determining how well one knows, e.g., a fact, can be
> trivially simple... to assess whether one knows a procedure one can envision a clear
> end state (e.g., a baked cake, a successful log-on to the Internet) and then work
> backwards to see if one knows how to get to that state. **Errors of omission are
> still possible but they are constrained by knowledge of the end state.**"

"A claim about a SET whose membership rule was never written down" *is* a claim with an
indeterminate end state. Writing the rule down converts an explanation-class claim
(illusion-prone, not self-testable) into a procedure-class claim (self-testable, with
omissions *constrained*). Main's hypothesis is well-grounded in 2002 peer-reviewed
cognitive science that was not written about software.

Two further R&K mechanisms map onto specific instances:

- **"Confusion of environmental support with internal representation"** -- "When people
  succeed at solving problems with devices they may underestimate how much of their
  understanding lies in relations that are apparent in the object as opposed to being
  mentally represented." The repo is greppable and *right there*; "I could check" is
  experienced as "I checked". This is instance #1 (source scan standing in for runtime
  behaviour) and #7 (masterplan.json is one command away, so its contents feel known).
- **"Confusion of higher with lower levels of analysis"** via Simon's "stable
  subassemblies" -- "functional sub-assemblies that are easy to visualize and mentally
  animate may lead to strong (but mistaken) feelings of understanding at a high level
  of analysis, and thereby induce inaccurate feelings of comprehension about the lower
  levels." "I linted the touched files" is a stable subassembly; *which 14 files* is
  the lower level that was never represented. Instances #2, #3, #6.

## Finding 4 -- ADVERSARIAL: the "authors can't proofread their own work" premise is weak.

This is the part of my own field I should report against. The folk claim -- and the
one most likely to be reached for as justification for the Q/A agent -- rests on
Daneman & Stainton (1993), who found students detected **20% fewer errors** in their
own essays than in a familiar other-authored essay (Panko's tabulation: own work 59%
vs others' 76% on word errors at 20 min).

**It does not replicate cleanly.** Burgoyne et al. (2022), a near-replication with
eye-tracking:

- **Experiment 1 ran in the REVERSE direction**: self-generated group detected **5.3%
  MORE** errors than other-generated (p = .059).
- Session 2 (one week later): no difference (84.1% vs 87.7%, p = .158).
- Experiment 2 induced overfamiliarity with a studying manipulation. Directionally
  correct (self-study lowest, other-study highest) but **not significant**.
- Authors: "the results of the experiments are **equivocal**. They did not provide
  strong support for Daneman and Stainton's (1993) hypothesis." Power analyses were
  based on D&S's effect size, "which may be an **overestimate of the true effect**."
- Consistent with Pilotti & Chodorow (2009), who found familiarity *facilitated*
  proofreading.

**Implication for this project.** Do not justify the Q/A agent on "Main cannot see its
own errors." That premise is contested. The defensible justification is Finding 5.

## Finding 5 -- The real base rate: semantically-plausible errors escape ~1 pass in 3.

Panko's corpus (weighted means across studies) and Burgoyne's replication agree on the
error-type asymmetry:

| Error type | Detection rate |
|---|---|
| Non-word errors (misspellings -- surface-visible) | **81%** (Panko); 87-94% (Burgoyne S2) |
| Word errors (wrong word -- locally well-formed) | **66%** (Panko); 81% (Burgoyne S2) |
| Difficult material (Riefer 1991) | **47-58%** |
| Simple material (Riefer 1991) | 84-86% |
| Professional proofreaders, all error types | ~87% |

Burgoyne: "non-word errors (misspellings) were more readily detected than word errors
(wrong words)... this finding has practical significance because word errors are also
less likely to be detected by 'spell check'."

**All 10 instances are word errors, not non-word errors.** "A 16/16 mutation matrix,
0 vacuous guards" is grammatical, plausible, and internally coherent. "THREE duplicated
JSON-parse helpers" is a well-formed sentence. Nothing is locally malformed. These sit
in the 47-66% detection band -- and #6 (a *composition* never enumerated, inside a
truncated capture) is squarely "difficult material" at 47-58%.

*My synthesis, not a source claim:* at a per-claim detection rate of ~0.6, the
probability that a single careful pass catches all 10 is 0.6^10 = **0.6%**. Two
independent passes at 0.6 give ~0.16 escape per claim, i.e. ~1.6 expected escapes
across 10. This is the honest argument for the Q/A agent -- not author-blindness, but
that *per-pass detection of plausible-but-wrong claims is near coin-flip for anyone*,
so independence multiplies rather than adds. It also predicts Q/A will NOT catch 10/10
next time; 75.4/75.5 was a good draw.

## Finding 6 -- ADVERSARIAL: carefulness-as-a-trait predicts nothing.

Burgoyne Experiment 2 correlation table (N = 100) against proofreading performance:

| Measure | r with proofreading |
|---|---|
| **Cognitive Reflection Test** | **.01** |
| **Need for Cognition** | **-.08** |
| Reading comprehension | .08 |
| Memory test | -.15 |
| Working-memory / processing speed | n.s. |
| Fluid intelligence | .21 to .45 (significant) |
| Number of fixations | .22 |
| Time spent proofreading | .19 to .35 |

The **Cognitive Reflection Test is the canonical instrument for "does this person stop
and check rather than go with the first answer."** It correlated **r = .01** with error
detection. Need for Cognition -- the disposition to enjoy effortful thinking --
correlated **-.08**.

The disposition to be careful does not predict catching this class of error. What
weakly predicted it was *time on task* and *number of fixations* (r ~ .2-.35) -- i.e.
mechanical thoroughness, not attitude. This is a second, independent line of evidence
against exhortation, from a different literature than R&K.

## Finding 7 -- Checklist item design: the value, not the tick.

Degani & Wiener's Appendix A guideline **(4)** is the single most directly applicable
sentence I found in any source:

> "**Checklist responses should portray the desired status or the value of the item
> being considered (not just 'checked' or 'set').**"

The regulatory basis, FAR 121.315(b), quoted in the report:

> "The procedure must be designed so that a flight crewmember **will not need to rely
> upon his memory** for items to be checked."

Also load-bearing:

- **Guideline (5)**: "The use of hands and fingers to touch appropriate controls,
  switches, and displays while conducting the checklist is recommended." -- physical
  grounding; the check must contact the referent.
- **Guideline (10)**: most critical items as close as possible to the *beginning*.
- **Guideline (11)**: critical items that might change "should be **duplicated between
  task-checklists**."
- **Guideline (15)**: "There should be **no compromise, however, regarding the critical
  'killer' items.**"

**Length is a hazard, not a virtue.** Degani & Wiener document an 81-item
ENGINE-START/TAXI/TAKEOFF checklist at one carrier and the resulting "poor checklist
discipline"; long lists "may lead some to deviate from the prescribed procedures,
performing only what he/she perceives as the critical items ('killer items' as some
call them)." And: "**overemphasis of items might diminish the crew's overall checklist
performance.** Conversely, duplication of a very few highly critical items ('killer
items')... can be beneficial."

The ASRS #90128 case in the report is this exact defect class: three crew plus
maintenance missed start levers in the wrong position because "**Through-stop checklist
does not call for us to check the start levers in cut-off.**" The set the checklist
covered was never questioned; everyone assumed coverage.

## Finding 8 -- ADVERSARIAL: adding checklist items has weak-to-negative evidence.

- **Urbach et al., NEJM 2014**: 101 Ontario hospitals, 109,341 procedures before vs
  106,370 after adopting a surgical safety checklist. Complications 3.86% -> 3.82%;
  30-day mortality 0.71% -> 0.65%; **neither statistically significant**. None of the
  101 hospitals showed a significant mortality reduction. (Read via abstract +
  secondary coverage; NEJM full text paywalled -- counted snippet-only, not toward the
  floor.)
- **Realist synthesis (Implementation Science 2015, read in full)**: compliance varied
  wildly -- "adherence for the check-in phase ranged from **23 to 98%**; while timeout
  ranged from **19 to 100%**; and sign-out ranged from **2 to 93%**." Items decayed
  over 6-12 months.
- The synthesis reports checklists can be *actively harmful*: "**Rather than focussing
  on the patient, using the checklist actually diverted team members' attention away
  from the patient.**"
- The coherence failure is the one to worry about here: "Implementation of safety
  checklists in surgery had limited coherence for participants, particularly for
  physicians, **many of whom believed that the formal introduction of a checklist was
  redundant as they were already enacting these principles in practice**." And:
  "forgetfulness, time constraints, and duplication were identified as barriers... but
  they may actually be **surrogates for a perceived lack of value**."
- Mandate/sanction did not feature: "Most implementation interventions occurred in
  relation to supports provided and focussed on engaging health professionals rather
  than sanctioning them. There were **no serious sanctions evident in any of the
  studies**."

**Direct implication.** This project's CLAUDE.md and `.claude/agents/qa.md` are already
very long. Adding a prose item ("re-measure every count") is the intervention with the
weakest evidence in this entire brief, and Degani & Wiener predict it *dilutes* the
killer items already there. If an item is added, something should be removed.

## Finding 9 -- Reason: the doctrinal frame, and its limit.

Reason 2000 supplies the frame the project already uses implicitly:

> "**We cannot change the human condition, but we can change the conditions under which
> humans work.**"

> "active failures are like **mosquitoes**. They can be swatted one by one, but they
> still keep coming. The best remedies are to create more effective defences and to
> **drain the swamps in which they breed**."

> "it is often **the best people who make the worst mistakes** -- error is not the
> monopoly of an unfortunate few."

High-reliability organisations: "Instead of making local repairs, they look for system
reforms." In aviation maintenance "some 90% of quality lapses were judged as blameless."

**Limit worth flagging:** Reason is an argument *frame*, not an effect size. The BMJ
piece is a narrative review. It justifies preferring system change over exhortation; it
does not quantify how much better.

## Finding 10 -- ADVERSARIAL: I could not source the "hierarchy of effectiveness".

The patient-safety and lean literatures universally cite a ranking --
forcing functions > automation > standardisation > checklists > rules/policies >
education/training -- as if it were an empirical result. **I went looking for the
primary evidence and did not find it.** The AHRQ PSNet Human Factors Engineering
primer, the most authoritative source I fetched, does **not** state the hierarchy at
all. It defines forcing functions:

> "prevent an unintended or undesirable action from being performed or allows its
> performance only if another specific action is performed first"

with the canonical examples (brake-before-reverse; "removal of concentrated potassium
from general hospital wards"), and notes checklists have "roots in human factors
engineering principles" -- but presents no ranked effectiveness comparison, and does
not say education/training is weakest.

The poka-yoke sources returned by search were **all practitioner/community tier**
(ASQ, LearnLeanSigma, Six Sigma vendors, Wikipedia). The control-vs-warning poka-yoke
distinction ("control poka yoke... does not depend on human attention") is *plausible
and mechanistically sensible* but I found **no peer-reviewed effect-size comparison**
behind it.

**Report this honestly to the parent:** the ordinal ranking is engineering folklore.
The *individual* forcing-function successes (KCl removal) are real and documented, but
they are single-intervention case evidence, not a general law that forcing functions
beat training by some known margin. The strong evidence in this brief for preferring
structure over exhortation comes from R&K Study 6, Burgoyne's CRT correlation, and
EviBound -- **not** from the hierarchy.

## Finding 11 -- ADVERSARIAL: Main's hypothesis over-fits. It covers 8 of 10, not 10.

Two instances are a different failure class and will not be fixed by writing membership
rules:

- **Instance #5** (live_check labelled "verbatim" but hand-spliced -- 40 progress dots
  over a 41-passed summary). This is not an unstated set. This is **evidence-provenance
  failure / fabrication**. A membership rule does not prevent a human or agent from
  editing a pasted capture. Only capture-by-tool (the artifact is written by the
  command, never transcribed) prevents it. EviBound's gate catches exactly this.
- **Instance #7** (a section listing steps 75.5.1-75.5.7 as queued when ZERO existed in
  masterplan.json). This is a claim about **external system state**, not about a set's
  membership. The rule "enumerate the set" doesn't help; querying the external system
  does.

And one instance is a **counter-example to the hypothesis as stated**:

- **Instance #10**. The membership rule *was* written down -- that was literally the
  subject of the section. Two reviewers executed the stated rule and got **17 and 20**
  -- neither 36, nor each other. So a *written* rule was insufficient because it was a
  **prose** rule, and prose rules retain the indeterminate end state R&K identify. The
  refinement the evidence supports:

  > The membership rule must be **EXECUTABLE** -- a command with a deterministic
  > output -- not merely *stated*. A prose rule is still an explanation-class artifact.
  > A shell command is a procedure-class artifact with a determinate end state.

  That two independent reviewers executing the same prose rule disagreed is itself a
  measurement: the rule's inter-rater reliability was ~0. Compare R&K's independent
  raters at alpha = .931 on a *scored* task.

---

## Mechanisms, ranked by yield

Instance key: 1 source-scan/behaviour; 2 ruff 10-of-14; 3 16/16 vs 17; 4 test count
1288 vs 1289; 5 spliced "verbatim"; 6 unenumerated red set; 7 phantom queued steps;
8 THREE vs FOUR helpers; 9 FOUR vs >=6 + partial regex; 10 "~36 functions".

### M1. Computed-not-transcribed counts -- **catches 8/10** (2,3,4,5,6,8,9,10)
Every number and every set in a handoff file is emitted by a command into a fenced
block; Main never types a count. The command string is stored adjacent to its output.

- **Evidence**: Degani & Wiener guideline (4) -- responses must "portray the desired
  status or the value... not just 'checked'"; FAR 121.315(b) -- the procedure "must be
  designed so that a flight crewmember will not need to rely upon his memory". EviBound
  -- architectural enforcement took hallucination 100% -> 0% where prompt-level
  self-critique took it nowhere.
- **Cost**: LOW. A shell helper plus a template change. No new agent, no new gate. Fits
  a one-dev local repo.
- **Misses**: 1 (a computed number from the wrong instrument is still wrong), 7
  (external state).

### M2. Executable membership rule, declared BEFORE the claim -- **catches 8/10** (1,2,3,4,6,8,9,10)
Each set-claim must name the command that *defines* the set, before the count is
produced. "The 14 touched files are `git diff --name-only HEAD~1`" not "the touched
files".

- **Evidence**: R&K's indeterminate-end-state mechanism -- "self-testing one's
  knowledge of explanations is difficult... Errors of omission are still possible but
  they are **constrained by knowledge of the end state**." Instance #10 proves the rule
  must be executable, not prose (two reviewers, 17 vs 20).
- **Cost**: LOW-MEDIUM. Discipline + template field. Needs M1 to be enforceable.
- **Misses**: 5 (provenance), 7 (external state).

### M3. Machine-checkable artifact gating -- **catches 3/10** (4,5,7) but UNIQUELY
The verdict cannot be promoted unless the cited artifact is queryable by the harness at
verdict time -- capture written by the tool, never pasted.

- **Evidence**: EviBound 2511.05524 -- 100% -> 25% -> 0% across prompt-only,
  verification-only, dual-gate; 8.3% overhead. **Caveat: n=8.** And EviBound's own
  limitations section concedes it does NOT catch "**specification errors where
  acceptance criteria themselves are flawed**" -- which is exactly instances 1,2,3,6,
  8,9,10. An artifact can be real, queryable, and cover the wrong set.
- **Cost**: MEDIUM (harness plumbing).
- **Why keep it anyway**: 5 and 7 are invisible to M1/M2. This is the only mechanism
  that catches them.

### M4. Killer-item concentration, with deletion -- **catches 6/10** (2,3,4,8,9,10)
ONE item, placed first, duplicated into both the contract and evaluator_critique
templates: *"every count in this document was produced by the command shown next to
it."* Pay for it by deleting a lower-value item.

- **Evidence**: Degani & Wiener guidelines (10), (11), (15); the 81-item checklist and
  its "poor checklist discipline"; "overemphasis of items might diminish the crew's
  overall checklist performance."
- **Cost**: NEGATIVE (net removes text).

### M5. Instrument-claim typing -- **catches 2/10** (1, partially 3 and 6)
Each claim names its instrument; an instrument whose observational range cannot cover
the claim is a hard fail. A static source scan cannot observe runtime wiring.

- **Evidence**: R&K "confusion of environmental support with internal representation" --
  the ability to recover information from the object in real time is mistaken for
  having represented it.
- **Cost**: LOW. Narrow but nothing else catches #1.

### M6. Independent fresh-eyes Q/A -- **empirically caught 10/10; expect ~4 escapes/10 next time**
- **Evidence**: Panko -- word errors 66%, difficult material 47-58%; professionals
  ~87%. Reason -- "it is often the best people who make the worst mistakes."
- **The justification must change.** Burgoyne et al. 2022 failed to replicate the
  self-generation disadvantage (Exp 1 ran 5.3% in the *reverse* direction, p = .059;
  "equivocal"). So the defence of Q/A is NOT "Main is blind to its own work" -- it is
  that per-pass detection of plausible-but-wrong claims is ~50-66% *for anyone*, so a
  second independent pass multiplies coverage.
- **Cost**: already paid.

### M7. Exhortation -- "be more careful", "double-check", added prose rule -- **catches 0-1/10. NOT SUPPORTED.**
- **Evidence AGAINST, three independent literatures**:
  - R&K Study 6: explicit forewarning left the illusion significant; attenuation
    eta-squared = .050; and it worked by making later ratings *higher* (consistency-
    defense), not initial ratings lower. "Resistant to changes in instructions."
  - Burgoyne: Cognitive Reflection Test **r = .01**, Need for Cognition **r = -.08**
    with error detection. The trait of carefulness predicts nothing.
  - EviBound: self-reflection + critique prompting = **100% false-claim rate (8/8)**.
    "Model scale and prompt engineering are not sufficient."
  - Realist synthesis: clinicians who believed they "were already enacting these
    principles in practice" is precisely the failure mode a prose reminder produces.
- **Cost**: appears free; is not. Degani & Wiener: added items dilute killer items.
  Realist synthesis: a checklist can divert attention from the referent.
- **Risk of backfire on instance #10 specifically**: R&K Study 6's mechanism was
  increased *consistency*, not accuracy. Instance #10 asserted a number inside a
  section forbidding numbers -- maximal salience, defended estimate. More salience is
  the wrong lever.

---

## What the literature does NOT support (state plainly)

1. **No evidence that instructing an agent to be more careful reduces this error
   class.** Directly tested (R&K Study 6, EviBound Baseline A); failed both times.
2. **No peer-reviewed effect-size basis for the forcing-function > training
   "hierarchy of effectiveness".** AHRQ's own primer does not state it; all poka-yoke
   sources returned were practitioner-tier. Use the specific evidence, not the ranking.
3. **The "authors can't proofread their own work" premise is contested**, not settled
   (Burgoyne 2022 replication failure; Pilotti & Chodorow 2009 found the opposite).
   Do not build the Q/A justification on it.
4. **Checklists as an intervention have weak outcome evidence at scale** (Urbach 2014:
   ~216k procedures, no significant effect) and documented decay and attention-
   diversion harms.
5. **Main's hypothesis is right for 8 of 10 and wrong in one specific way**: instance
   #10 had a written rule and still failed. Rules must be executable, not stated.
   Instances #5 and #7 are provenance and external-state failures, not set-membership
   failures, and need M3.

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 5,
  "urls_collected": 100,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "gate_passed": true
}
```
