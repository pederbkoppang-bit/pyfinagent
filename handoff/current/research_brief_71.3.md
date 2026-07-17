# Research Brief -- Step 71.3: Harden the single Q/A role

**Step:** 71.3 (Layer-3 harness MAS upgrade) | **Tier:** moderate | **Date:** 2026-07-17
**Author:** Researcher subagent (research gate) | **Status:** COMPLETE -- gate PASSED

## Objective

Harden the SINGLE Q/A role (NO re-split, NO 4th agent) with three additions:
- (a) CONTRACT-COMPLETENESS rubric dimension: does `experiment_results.md` cover EVERY contract criterion?
- (b) ADVERSARIAL worst-of-N / N-LENS verdict for P0/P1 money-path steps (single Q/A, worst verdict across distinct lenses).
- (c) Machine-readable `handoff/current/evaluator_critique.json` alongside the `.md` so the status-flip gate reads JSON, not prose.

Preserve: single-Q/A-per-step, file-based handoffs, exactly-3-agents.

---

## 1. CRITICAL RECONCILIATION -- #8a (dropped) vs the masterplan 71.3 criterion  [RESOLVED]

### 1.1 The apparent conflict

**Masterplan 71.3 criterion (verbatim, immutable):**
> "For P0/P1 money-path steps, add an adversarial worst-of-N self-consistency check (the single Q/A samples its own
> verdict N times / from N lenses and takes the worst) instead of a single-shot judgment."

**But the phase-71.0 design pack DROPPED the worst-of-N self-consistency proposal (#8a)** while KEEPING the adversarial
red-team leg (#8b). Exact 71.0 wording:

- `handoff/current/design_harness_mas_71.md:63` header: **"### 71.3 (P2) — Q/A rubric hardening (WITHIN the single Q/A role)  [kept #6 + #8b; DROP #8a]"**
- `design_harness_mas_71.md:66-69` (verbatim): "add a **contract-completeness** dimension (does `experiment_results.md`
  cover every contract criterion?) + an **adversarial red-team leg** to `qa.md`. **DROP the worst-of-N self-consistency
  (#8a):** the grounding was oversold — the 345,968-NAV bug was a *deterministic* catch, and a `money_path` tag is
  invented; **N identical samples add cost without independent signal.** Keep it inside the single Q/A role (no re-split)."
- `handoff/archive/phase-71.0/contract.md:25`: "**71.3 (P2)** — Q/A rubric: contract-completeness dimension + adversarial
  red-team leg. DROP the worst-of-N (#8a, grounding oversold)."
- `handoff/archive/phase-71.0/evaluator_critique.md:47-49` (the fresh Q/A's PASS-verdict scope-honesty note, verbatim):
  "SCOPE HONESTY: the two refinements (drop #8a worst-of-N; descope #12 to report-only) are explicitly justified, not
  silent, and **#8a is independently corroborated by the recency scan (ensembling judges carries correlated bias).**"

**The register** (`harness_proposals.json`) shows proposal #8 is a TWO-PART bundle whose halves score differently:
- **Part (a) = worst-of-N N-IDENTICAL self-consistency** ("run the qa agentType 3x via the Workflow path and take the
  WORST verdict") -- reviewer verdict: **DROP or strictly defer.** Reasons quoted from the register: the headline
  worst-of-N grounding is "overstated" (Anthropic docs do NOT recommend judge ensembling; in-repo self-consistency
  evidence is *majority-vote accuracy*, a different aggregation than worst-of-N recall); the flagship 345,968-NAV benefit
  is "misattributed" because that was a DETERMINISTIC live-UI/runtime-smoke catch already closed by qa.md §1c/§1d --
  "Three stochastic LLM samples all lacking that capture would PASS identically."
- **Part (b) = the adversarial red-team leg** in qa.md's LLM-judgment section ("actively construct an input/state that
  would break the claimed PASS; if you can, the verdict is not PASS") -- reviewer verdict: **KEEP (P2)** -- "a genuine,
  cheap, always-on, well-grounded win ... ports the researcher's already-blessed adversarial pass ... to the
  load-bearing false-PASS gate."

So the criterion's phrasing `"N times / from N lenses"` encodes BOTH readings, joined by `/`:
- **"N times"** = N-IDENTICAL samples = **#8a (DROPPED)**.
- **"from N lenses"** = N perspective-diverse adversarial samples = the **kept adversarial red-team leg #8b**, generalized.

### 1.2 The resolution (safe framing GENERATE MUST use)

**Implement the "FROM N LENSES" reading only.** The single Q/A evaluates the claimed PASS from **N DISTINCT adversarial
lenses** -- e.g. a **correctness** lens, a **does-it-reproduce** lens, and a **scope-honesty** lens -- and takes the
**WORST verdict across lenses** (any lens FAIL -> FAIL; any lens CONDITIONAL dominates PASS). This is the adversarial
red-team leg (#8b) structured as a worst-of-N over PERSPECTIVES. It does NOT re-introduce N-identical resampling (#8a).

This resolution simultaneously:

**(i) SATISFIES the immutable criterion + the verification grep.** It IS "an adversarial worst-of-N ... from N lenses ...
takes the worst," gated to P0/P1 money-path steps, kept inside the single Q/A role (the SAME agent applies the lenses in
one pass -- or via same-role forks, which the repo already blesses: per-step-protocol.md:296-298 "forks are more
instances of the SAME role, never new roles"). No fourth agent, no re-split. The masterplan verification command is:
> `bash -c 'grep -Eqi "contract completeness|completeness" .claude/agents/qa.md && grep -Eqi "worst-of-n|self-consistency|adversarial" .claude/agents/qa.md docs/runbooks/per-step-protocol.md'`

The tokens **"completeness"**, **"adversarial"**, and **"worst-of-N"** all land in qa.md -> BOTH greps pass. (Second grep
is satisfied by ANY of the three alternation tokens appearing in EITHER file; "adversarial"+"worst-of-N" in qa.md alone
clears it. Confirmed the tokens are currently ABSENT in both files, so GENERATE must add them -- see §4.)

**(ii) Is CONSISTENT with the 71.0 design.** 71.0 objected to **N-IDENTICAL sampling specifically** ("N identical samples
add cost without independent signal"; "ensembling judges carries correlated bias"), NOT to perspective-diverse adversarial
checks -- it explicitly KEPT the adversarial red-team leg (#8b). Distinct-lens verification catches failure modes that
N-identical redundancy cannot: each lens asks a DIFFERENT question, so the samples are not correlated repetitions of one
biased draw. GENERATE must frame the leg as **"worst-of-N over N distinct adversarial LENSES, NOT N-identical
self-consistency resampling (#8a, dropped in 71.0)"** -- an explicit negation that prevents re-introducing the rejected idea
and reads cleanly for the grep.

### 1.3 Research basis for "distinct lenses > N-identical redundancy"

- **arXiv:2505.19477 "Judging with Many Minds" (2025, read in full):** homogeneous iterative repetition (Multi-Agent
  Debate) AMPLIFIES bias -- "introducing Multi-Agent Debate into LLM evaluation leads to a sharp increase in bias
  immediately after the first round" and "this elevated level of bias persists ... with no significant further increase or
  signs of recovery" (§5.1); whereas a perspective-diverse Meta-Judge RESISTS it -- "the LLM-as-Meta-Judge approach
  demonstrates better consistency, and sometimes improvement, against these biases" (Conclusion). Crucially "bias
  amplification ... is a phenomenon observed across all judge-critic pairings" (§5.1) -> the problem is FRAMEWORK
  HOMOGENEITY, not model weakness. Direct support: repeating the same judgment does not help; changing the PERSPECTIVE does.
- **LLM-juries practitioner finding (orq.ai, snippet):** "three correlated judges are one judge with 3× more requests if
  they share similar biases"; "base learners must be diverse with different models and error profiles." This is exactly the
  "no independent signal" objection to #8a stated in information-theoretic terms.
- **arXiv:2508.06709 "Play Favorites" (2025, read in full):** self-bias is "systematic ... even after controlling for the
  quality of the completions" (§5.2) and is measurable/persistent -> a judge sampled N times carries its OWN systematic
  bias in every draw (correlated), so N-identical resampling cannot surface a bias the judge shares with itself.
  Recommended mitigation is "a diverse panel ... drawn from multiple model families" (§7) -- diversity, not repetition.
- **Anthropic "Building Effective Agents" (read in full):** the evaluator-optimizer LOOP ("one LLM call generates a
  response while another provides evaluation and feedback in a loop") is ALREADY implemented in the harness as the
  retry-on-FAIL ladder (per-step-protocol.md:162-196). So #8a's loop framing adds nothing; the incremental value is the
  adversarial PERSPECTIVE inside the single judge, not another loop.
- **Anthropic "Demystifying evals" (read in full):** `pass^k` "measures the probability that all k trials succeed" -- it
  is an AGENT-reliability metric, NOT a judge-aggregation recipe; it does not license worst-of-N judge ensembling. The doc
  only name-drops "multi-judge consensus" without any protocol -- confirming the register's "grounding oversold" finding.

**Net:** the N-DISTINCT-LENSES adversarial worst-of is the correct, grounded, criterion-satisfying implementation; the
rejected N-IDENTICAL #8a stays rejected.

---

## 2. External findings (Anthropic judge rubric + doer/judge + self-consistency vs multi-lens)

- **Completeness IS a first-class Anthropic judge-rubric dimension.** "How we built our multi-agent research system"
  (read in full): the LLM judge scores "factual accuracy ... citation accuracy ... **completeness (are all requested
  aspects covered?)** ... source quality ... tool efficiency." This is the exact grounding for adding a
  contract-completeness dimension -- map each contract criterion to its evidence and flag any uncovered one.
- **Anthropic prefers a SINGLE judge call, not many specialized judges.** Same source: "a single LLM call with a single
  prompt outputting scores from 0.0-1.0 and a pass-fail grade was the most consistent and aligned with human judgements."
  -> reinforces single-Q/A-per-step; the N lenses are ONE judge reasoning across perspectives, not N separate judge agents.
- **Doer/judge separation is the strong lever.** Harness-design (read in full): "When asked to evaluate work they've
  produced, agents tend to respond by confidently praising the work"; "Separating the agent doing the work from the agent
  judging it proves to be a strong lever"; "**Tuning a standalone evaluator to be skeptical turns out to be far more
  tractable than making a generator critical of its own work.**" The adversarial red-team leg operationalizes that
  skepticism INSIDE the already-separate Q/A -- it does not require a new agent.
- **Hard threshold per criterion / detailed feedback on fail.** Harness-design: "Each criterion had a hard threshold, and
  if any one fell below it, the sprint failed and the generator got detailed feedback." Maps to worst-of-lenses (any lens
  below bar -> not PASS) and to the existing violation_details feedback contract.
- **File-based handoff + structured artifacts are the documented pattern.** Harness-design: "Communication was handled via
  files: one agent would write a file, another agent would read it"; "Using structured artifacts to hand off context." ->
  grounds emitting `evaluator_critique.json` as a durable, machine-readable handoff artifact.
- **Rubric decomposition & correlation-pruning (recency, 2026):** Recursive Rubric Decomposition (arXiv:2602.05125) and
  Autorubric (arXiv:2603.00077) decompose high-level rubric items into finer subpoints for "more comprehensive and
  discriminative evaluations," and aggregation should be "removing redundant rubrics while down-weighting highly correlated
  ones to prevent overlapping perspectives from being overrepresented." Supports distinct-lens design AND warns against
  redundant/correlated lenses (keep the N lenses genuinely orthogonal: correctness / reproduce / scope-honesty).

## 3. Recency scan (last 2 years)

**Performed.** Query variants run (3-variant discipline satisfied):
1. Current-frontier (2026): "LLM-as-judge self-consistency correlated bias ensemble same model perspective-diverse
   multi-lens evaluation 2026".
2. Anthropic canonical: "Anthropic multi-agent research system evaluator rubric completeness ...".
3. Year-less canonical: "worst-of-N verifier high recall LLM judge multi-perspective rubric completeness factual accuracy
   dimensions".

**New findings in the window that COMPLEMENT the older Anthropic canon (do not supersede it):**
- arXiv:2505.19477 (May 2025) -- perspective-diverse Meta-Judge resists bias; homogeneous debate amplifies it. NEW,
  directly settles the #8a-vs-N-lens question in favor of distinct lenses.
- arXiv:2508.06709 (Aug 2025) -- self-bias is systematic and correlated across a judge's own outputs; mitigation is
  cross-family diversity. NEW, grounds "no independent signal" for N-identical resampling.
- arXiv:2602.05125 / arXiv:2603.00077 (2026) -- rubric decomposition + correlation-pruning for multi-dimension judges.
  NEW, refines HOW to keep the lenses orthogonal.
- arXiv:2508.02994 "Agent-as-a-Judge" (2025, snippet) -- agentic evaluators; consistent with doer/judge separation.
The older Anthropic canon (multi-agent-research rubric, harness-design doer/judge, building-effective-agents
evaluator-optimizer) remains the authoritative baseline; the 2025-2026 work sharpens the aggregation choice.

---

## 4. qa.md + per-step-protocol.md insertion anchors (precise; role NOT re-split)

Current `qa.md` structure (verified by Read; tokens completeness/adversarial/worst-of-N/self-consistency ALL absent today):
- `## Verification order (deterministic FIRST)` L115; `### 4. LLM judgment (last resort)` **L235**, bullets **L237-247**
  (Contract alignment / Anti-rubber-stamp / Scope honesty / Research-gate compliance).
- `## Output format (single JSON)` **L255**; success schema **L257-267**; failure schema **L276-293**.
- `## Quality criteria (from agent_definitions.py)` **L302**; table rows **L304-311**; "Score below 6 ... = FAIL" **L311**.
- `## Constraints` **L313-349** (NEVER Edit/Write L315-318; verification budget L320-326; 3rd-CONDITIONAL L342-349).

**(a) Contract-completeness dimension -- insert at TWO anchors (both cheap; the first lands the grep token):**
1. Add a bullet to the §4 LLM-judgment list **after L247** (after "Research-gate compliance"):
   > "- **Contract completeness (are all requested aspects covered?):** map EVERY immutable success criterion /
   >   contract criterion in `handoff/current/contract.md` to the specific evidence in `experiment_results.md` that
   >   covers it. Any criterion with no covering evidence is `Missing_Assumption` and caps the verdict at CONDITIONAL
   >   (or FAIL if it is a hard/immutable criterion). Grounding: Anthropic multi-agent-research judge rubric."
2. Add a row to the Quality-criteria table **at L304-311** (or a one-line note beneath it) naming
   **"Contract completeness"** so the string is unambiguously present for the `"completeness"` grep and for readers who
   scan the table. (Either anchor satisfies the grep; do both for clarity.)

**(b) Adversarial N-lens worst-of leg -- insert as a NEW subsection right after §4 (after L247), e.g. `### 4a.`:**
> "### 4a. Adversarial worst-of-N-LENSES (P0/P1 money-path steps only)
> For steps tagged P0/P1 or touching the money-path, do NOT issue a single-shot verdict. As the SAME single Q/A agent
> (no re-split, no fourth agent), evaluate the claimed PASS from N DISTINCT adversarial LENSES and take the WORST
> verdict across them (any lens FAIL -> FAIL; any lens CONDITIONAL dominates PASS):
>  - **correctness lens** -- actively construct an input/state that breaks the claimed PASS; if you can, it is not PASS;
>  - **does-it-reproduce lens** -- re-run the immutable verification command and the runtime/live checks independently;
>  - **scope-honesty lens** -- did experiment_results overclaim, or silently touch out-of-scope production code?
> This is worst-of-N over N distinct LENSES -- NOT N-identical self-consistency resampling (#8a, dropped in phase-71.0
> because N identical samples share correlated bias and add cost without independent signal;
> `design_harness_mas_71.md:63-69`). Cheaper P2/P3 steps stay single-shot (cost discipline)."

This lands "adversarial", "worst-of-N", and (via the negation) "self-consistency" in qa.md.

**(c) evaluator_critique.json emission -- note in the Output-format section (L255) + Constraints (L313):**
- After the success schema (**~L267**): "Main persists this returned verdict object VERBATIM to
  `handoff/current/evaluator_critique.json` (adding only `step_id` and `cycle_num`) alongside the human-readable
  `evaluator_critique.md`, so the status-flip gate reads JSON, not prose. **Q/A itself never writes the file** (Q/A is
  read-only) -- Main is the scribe, exactly as it already transcribes the `.md` (no-self-eval preserved: Main records the
  verdict, never authors it)."
- Constraints (L315-318 already say NEVER Edit/Write) -- add one clause that the JSON emission is Main's job, not Q/A's.

**per-step-protocol.md §4 EVALUATE anchors** (`### 4. EVALUATE` L111; "Q/A runs deterministic-first" list L137-144;
"Returns the JSON schema ..." **L145-148**):
- After **L148**, add a short paragraph documenting (1) the contract-completeness dimension, (2) the P0/P1 adversarial
  worst-of-N-LENSES leg (explicitly "distinct lenses, not N-identical resampling"), and (3) that Main persists the returned
  verdict as `handoff/current/evaluator_critique.json` (+ step_id + cycle_num) so the status-flip gate reads JSON. Also
  add the `evaluator_critique.json` filename to the LOG/flip step (**L236-239**) so the artifact is part of the archived
  five-file set. (Putting a token in per-step-protocol.md is optional for the grep -- qa.md alone satisfies it -- but the
  runbook SHOULD carry it for completeness.)

**Roster-snapshot / separation-of-duties note (binding):** 71.3 edits `.claude/agents/qa.md`. On the **Workflow
structured-output path** the qa role reads `qa.md` from disk at runtime (qa-verdict.js:13-15; qa.md:73-93), so the new
dimension + leg are LIVE immediately -- NO roster snapshot needed there. Only the **Agent-tool `qa` fallback** snapshots at
session start, so a same-session self-eval of 71.3-dependent work on THAT path is stale until restart. Per CLAUDE.md
separation-of-duties: the session that AUTHORS this qa.md edit must not also self-evaluate work depending on it; leave a
`handoff/harness_log.md` note requesting Peder review, and verify with `scripts/qa/verify_qa_roster_live.sh` after restart.

---

## 5. Recommended `evaluator_critique.json` shape + where the gate reads it

### 5.1 The schema ALREADY exists -- the return value IS the machine-readable verdict

`.claude/workflows/qa-verdict.js::VERDICT_SCHEMA` (L80-108) is the checked-in constrained-decoding schema the Q/A
`agent()` call returns (L111-119). It **already carries** every field the masterplan wants EXCEPT two:
- Present in VERDICT_SCHEMA: `ok` (bool), `verdict` (enum PASS|CONDITIONAL|FAIL), `reason`, `violated_criteria[]`,
  `violation_details[]` (`{violation_type, action, state, constraint}`), `certified_fallback`, **`checks_run[]`**,
  `harness_compliance_ok`, `notes`.
- **MISSING (Main must ADD when persisting):** `step_id` and `cycle_num`. Both are already in scope for Main:
  `step_id` = `args.step_id` passed into qa-verdict.js (L35 `const stepId = a.step_id`); `cycle_num` = the Cycle-N counter
  Main maintains in `handoff/harness_log.md` (the `## Cycle N` header). Neither belongs IN the Q/A return schema (they are
  harness bookkeeping, not judge output) -- Main injects them at persist time.

**One shape note on `checks_run`:** the masterplan writes `checks_run{}` (object) but the existing schema is
`checks_run` **array-of-strings** (qa-verdict.js:104; qa.md:265 `["syntax","verification_command",...]`). Do **NOT** edit
the 71.1-owned VERDICT_SCHEMA to change the return type. Instead, when Main PERSISTS the JSON it may store `checks_run`
either (a) as the array verbatim from the return value (simplest; the gate only needs `ok`/`verdict`), or (b) transform to
an object map `{"verification_command_exit":0,"syntax":"ok","lint_F821":"ok",...}` to literally match the `{}` notation and
give the gate richer deterministic assertions. **Recommendation: (b) object map** -- it satisfies the `{}` spec and lets a
gate assert `checks_run.verification_command_exit == 0`; keep the array available under a second key if convenient. The
transform is a few lines in Main's persist step; the Q/A schema stays untouched.

### 5.2 Recommended persisted `handoff/current/evaluator_critique.json`

```json
{
  "step_id": "71.3",
  "cycle_num": 1,
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met; deterministic checks exit=0; adversarial lenses all PASS.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": {
    "verification_command_exit": 0,
    "syntax": "ok",
    "lint_F821_F401_F811": "ok",
    "contract_completeness": "3/3 criteria covered",
    "adversarial_lenses": { "correctness": "PASS", "reproduce": "PASS", "scope_honesty": "PASS" }
  },
  "harness_compliance_ok": true,
  "notes": "..."
}
```

- `step_id` + `cycle_num`: Main-injected (args + harness_log counter).
- `ok`, `verdict`, `violated_criteria`, `violation_details`, `certified_fallback`, `harness_compliance_ok`, `reason`,
  `notes`: verbatim from the captured VERDICT_SCHEMA return value.
- `checks_run`: object map (superset of the schema's array); for P0/P1 steps include the per-lens `adversarial_lenses`
  sub-object so worst-of-lenses is auditable.
- Gate contract: **`verdict == "PASS"` AND `ok == true`** is the machine-readable go/no-go.

### 5.3 Where the status-flip gate reads it (this JSON is NEW; nothing reads it today)

Today's status-flip pipeline: Main writes `status:"done"` to `.claude/masterplan.json` -> PostToolUse
`auto-commit-and-push.sh` fires -> it calls `lib/live_check_gate.py` (reads `verification.live_check` in masterplan.json +
checks `handoff/current/live_check_<id>.md`) and a default-OFF harness_log gate (checks for `phase=<id>` token). **No
gate reads any evaluator JSON** -- `evaluator_critique.json` does not exist yet.

Two complementary wirings (recommend BOTH; escalate as needed):

1. **Main-discipline read (mandatory, zero hook change) -- the minimum to satisfy criterion #2.** Before flipping
   `status:"done"`, Main MUST read `handoff/current/evaluator_critique.json` and confirm `verdict=="PASS"` (`ok==true`).
   Codify in per-step-protocol.md §5 + qa.md next to the log-last rule. This is the same discipline class as the existing
   "log-last / no-batched-done" rules.

2. **Fail-open hook gate (recommended, literally makes "the gate reads JSON").** Add a `verdict_gate.py` under
   `.claude/hooks/lib/` that MIRRORS `live_check_gate.py` exactly: given masterplan path + step_id + handoff dir, return
   `proceed` (no JSON / unreadable / parse error -> **fail-open**, matching live_check_gate's discipline), `passed`
   (`evaluator_critique.json` exists AND `verdict=="PASS"`/`ok==true`), or `skip` (JSON exists but verdict != PASS -> log
   WARN, hold the auto-push; commit + changelog still happen, exactly like the live_check gate). Wire it into
   `auto-commit-and-push.sh` **immediately after the harness_log gate block (~L180)**, same shape as the two gates already
   there (L142-165 live_check, L167-180 harness_log). This converts "the agent claimed PASS" into "a machine-readable PASS
   artifact exists that the push gate verified" -- a direct extension of the R-1 VERIFICATION_DEFECT remediation the
   live_check gate already implements. Keep it **fail-open** so it never breaks the masterplan Write.

**Preservation check:** none of this adds an agent or a loop. Q/A stays single, read-only, 3rd-CONDITIONAL + no-auto-PASS +
single-shot-return rules intact (the worst-of-lenses happens INSIDE the one Q/A pass; it does not spawn parallel judges and
it does not loop fix->re-grade -- Main still owns fixes via the cycle-2 respawn on changed evidence).

---

## 6. Search queries run

1. (Anthropic canonical) "Anthropic multi-agent research system evaluator rubric completeness are all requested aspects covered"
2. (2026 frontier) "LLM-as-judge self-consistency correlated bias ensemble same model perspective-diverse multi-lens evaluation 2026"
3. (year-less canonical) "worst-of-N verifier high recall LLM judge multi-perspective rubric completeness factual accuracy dimensions"

## 7. Sources read IN FULL (>=5 required; 6 read)

| # | Source | Tier | Key load-bearing finding |
|---|---|---|---|
| 1 | Anthropic -- How we built our multi-agent research system (anthropic.com/engineering/multi-agent-research-system) | Official | judge rubric incl. "completeness (are all requested aspects covered?)"; single-LLM-call judge > many judges; LeadResearcher spawns more subagents; human eval catches edge cases |
| 2 | Anthropic -- Building Effective Agents (anthropic.com/engineering/building-effective-agents) | Official | evaluator-optimizer loop (already = retry-on-FAIL); generator/evaluator role separation; "stopping conditions (max iterations)" |
| 3 | Anthropic -- Harness Design for Long-Running Apps (anthropic.com/engineering/harness-design-long-running-apps) | Official | "agents ... confidently praising the work"; "Separating the agent doing the work from the agent judging it ... a strong lever"; "far more tractable" to tune a skeptical standalone evaluator; file-based handoff; per-criterion hard threshold |
| 4 | Anthropic -- Demystifying evals for AI agents (anthropic.com/engineering/demystifying-evals-for-ai-agents) | Official | pass^k = AGENT reliability (not judge aggregation); "vague rubrics produce inconsistent judgments"; "structured rubrics to grade each dimension"; multi-judge consensus only name-dropped (no protocol) |
| 5 | arXiv:2505.19477 -- Judging with Many Minds (arxiv.org/html/2505.19477) | Peer-reviewed | Multi-Agent Debate AMPLIFIES bias (§5.1); perspective-diverse Meta-Judge RESISTS it (Conclusion); "observed across all judge-critic pairings" -> framework homogeneity is the cause |
| 6 | arXiv:2508.06709 -- Play Favorites: Self-Bias in LLM-as-a-Judge (arxiv.org/html/2508.06709) | Peer-reviewed | self-bias is systematic even controlling for quality (§5.2); mitigation = diverse cross-family panel (§7) -> repetition of one judge preserves correlated bias |

## 8. Snippet-only sources (evaluated, not read in full)

| URL | Kind | Why snippet |
|---|---|---|
| orq.ai/blog/llm-juries-in-practice | practitioner | "three correlated judges are one judge with 3× more requests if they share similar biases"; "base learners must be diverse" -- snippet sufficed |
| arxiv.org/html/2508.02994 (Agent-as-a-Judge) | peer-reviewed | agentic evaluators; consistent w/ doer-judge separation |
| arxiv.org/pdf/2603.00077 (Autorubric) | peer-reviewed | rubric decomposition/unification |
| arxiv.org/html/2602.05125 (Rethinking Rubric Generation / RRD) | peer-reviewed | Recursive Rubric Decomposition; correlation-pruning of redundant rubrics |
| arxiv.org/pdf/2406.11939 (Arena-Hard/BenchBuilder) | peer-reviewed | benchmark judge separability |
| arxiv.org/pdf/2507.01352 (Skywork-Reward-V2) | peer-reviewed | preference-data judge scaling |
| medium.com/@adnanmasood rubric-based evals | practitioner | rubric methodologies/biases overview |
| galtea.ai/blog/llm-as-a-judge | practitioner | LLM-judge guide |
| verifywise.ai demystifying-evals mirror | secondary | pass^k mirror |
| medium/substack/bytebytego multi-agent-research explainers (x4) | secondary | restate Anthropic canon |

## 9. Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: 4 Anthropic official + 2 arXiv peer-reviewed)
- [x] 10+ unique candidate URLs (~24 across 3 searches)
- [x] Recency scan performed + reported (2025-2026 window; 3 query variants incl. year-less canonical)
- [x] Source-quality hierarchy honored (official + peer-reviewed in the read-in-full set)
- [x] file:line anchors for every internal claim (qa.md, qa-verdict.js, per-step-protocol.md, design_harness_mas_71.md, live_check_gate.py, auto-commit-and-push.sh, masterplan 71.3)
- [x] #8a-vs-N-lens reconciliation resolved with the safe framing for GENERATE
- [x] evaluator_critique.json shape + gate-read location specified (schema exists; Main adds step_id+cycle_num)

## 10. JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 24,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "gate_passed": true
}
```
