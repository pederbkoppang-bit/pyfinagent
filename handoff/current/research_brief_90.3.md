# Research Brief -- step 90.3

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Topic:** Progress-gated retry for autonomous agent loops -- why a content-digest of changed
evidence measures CHANGE not PROGRESS; oscillation/cycle detection in iterative repair;
transient-infra vs genuine-no-progress discrimination; idempotency keys + retry-budget design;
fail-open vs fail-closed for a gate that can crash; and published cautions against reusing a
liveness signal as evidence of convergence or quality.

<!-- ENVELOPE: born inert per phase-86.37; flipped to COMPLETE as the final act. -->
```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 24,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {"audit_class": false, "rounds": 3, "dry_rounds": 0, "K_required": 2,
               "new_findings_last_round": 1, "dry": false},
  "brief_path": "handoff/current/research_brief_90.3.md",
  "gate_passed": true
}
```

## Status log (append-only, write-first)
- Round 0: brief created, envelope born inert. Beginning internal exploration + external search.
- Round 1: 2 sources read in full (arXiv 2607.01641, arXiv 2605.01471); internal read of attempt_gate.py + attempt_budget.py.

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://arxiv.org/html/2607.01641v1 | 2026-08-21 | preprint (arXiv, "When Agents Do Not Stop: Uncovering Infinite Agentic Loops in LLM Agents") | WebFetch, native arXiv HTML | Defines an Infinite Agentic Loop as *"a structural execution failure where an agentic feedback path repeatedly triggers costly or state-growing actions without an effective stopping bound."* Detection (IAL-Scan, Alg. 1 / IV-C1) is **static SCC analysis over a cycle-relevant call graph**: *"computes strongly connected components and retains nontrivial SCCs, including singleton SCCs with explicit self-loop edges."* Explicitly separates STATE GROWTH from PROGRESS (III-C): the motivating loop grows *"the enlarged message history"* every iteration yet has no deterministic progress, since continuation *"may still be controlled by model outputs, tool observations, external state, exceptions, or delegation decisions."* Top failure pattern in the taxonomy is **"Retry feedback without bounds" (25.0%)**, then unbounded tool-call iteration (23.5%); "Runner/delegation/evaluator feedback" is 7.4%. Empirics: 68 confirmed IALs across 47 projects from 6,549 repos, precision 91.9%, and **95.6% cause API-cost exhaustion or model DoS**. Notably it does **NOT** use duplicate-state/visited-set detection -- it asks only whether a feedback path *can* re-reach a costly operation without an effective bound. |
| 2 | https://arxiv.org/html/2605.01471 | 2026-08-21 | preprint (arXiv, "Practical Limits of Autonomous Test Repair") | WebFetch, native arXiv HTML | The single most on-point empirical statement for this step (S5.2): *"Without a formal correctness signal, the agent cannot distinguish genuine progress from lateral variation."* Repairs modified *"selectors, assertions, or interaction sequences without resolving the underlying failure."* Worst case: *"113 consecutive reports spanning multiple execution cycles, each exhausting the maximum retry depth of 16, without ever producing a single executable test artifact"* -- i.e. maximal output CHANGE with zero progress. Convergence: *"3 of 10 scenario families (30%) failed to converge within the observed window."* S6.8: *"Retry alone is not recovery: retries helped 7 of 10 families converge, but the same mechanism sustained 113 sequential fruitless reports."* Their remedy is NOT a smarter progress signal but a **hard cumulative cap + human escalation** (S9, Rule 2): *"terminates after a configurable number of attempts. Failures that do not converge are escalated to human review ... a limit of 6-7 retries captures all converging families while preventing the 113-report stagnation."* |
| 3 | https://ar5iv.labs.arxiv.org/html/2310.01798 | 2026-08-21 | peer-reviewed (ICLR 2024, Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet") | WebFetch, ar5iv (pre-Dec-2023 paper) | The canonical caution that a CHANGED answer is not an IMPROVED answer. S1: *"LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction."* S3.2: *"After self-correction, the accuracies of all models drop across all benchmarks"* (GPT-3.5 GSM8K 75.9->75.1; Llama-2 GSM8K 62.0->43.5). The reported gains in prior work came from the STOPPING RULE, not the correction: they *"used the correct label to determine when to stop the self-correction loop."* S3.3: *"determining how to prevent such mischanges is, in fact, the key to ensuring the success of self-correction"* -- and *"the model is more likely to modify a correct answer to an incorrect one than to revise an incorrect answer to a correct one."* Directly relevant: an oracle-free loop that only observes THAT the answer changed cannot tell correct->incorrect from incorrect->correct. |
| 4 | https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/ | 2026-08-21 | official docs (Kubernetes, Dynamic Admission Control) | WebFetch | The canonical fail-open/fail-closed vocabulary for a GATE THAT CAN CRASH. `failurePolicy: Fail` -- *"If a webhook fails or times out, the request is rejected"*; `failurePolicy: Ignore` -- *"the request is allowed to proceed as if the webhook was not called."* A timeout is routed through the SAME policy: *"If the webhook call times out, the request is handled according to the webhook's failure policy"*, default 10s and *"it is encouraged to use a short timeout."* Two more directly applicable rules: (a) *"Admission webhooks should be idempotent"*, because a webhook can be retried and can be reinvoked (`reinvocationPolicy: IfNeeded`) after another webhook mutates the object; (b) the self-referential deadlock warning -- a webhook that intercepts the resources running its own infrastructure wedges the cluster, mitigated by excluding its own namespace. |
| 5 | https://sre.google/sre-book/handling-overload/ | 2026-08-21 | official docs / book (Google SRE, ch. "Handling Overload") | WebFetch | The work-accounting retry bound this repo's `attempt_budget.py` docstring already cites, in its own words. TWO budgets with different jobs: a *"per-request retry budget of up to three attempts. If a request has already failed three times, we let the failure bubble up to the caller"*, AND a *"per-client retry budget. Each client keeps track of the ratio of requests that correspond to retries. A request will only be retried as long as this ratio is below 10%."* Retry amplification is bounded by locating the retry at exactly ONE layer: *"a failed request from the DB Frontend should only be retried by Backend B, the layer immediately above it. If multiple layers retried, we'd have a combinatorial explosion."* And the classification is carried ON THE ERROR, not inferred by the caller: return *"an 'overloaded; don't retry' error and thus avoid a combinatorial retry explosion."* |
| 6 | https://arxiv.org/html/2605.08455 | 2026-08-21 | preprint (arXiv, "CUDABeaver: Benchmarking LLM-Based Automated CUDA Debugging") | WebFetch, native arXiv HTML | **The single most transferable design for 90.3.** Four named stagnation signals (S3.5 / App. D.1), each with its computation and measured firing rate (Table 6): `duplicate_code` = *"SHA-256 of the current solution matches the previous iteration's"* (0.0%-50.8%); `code_cycle` = *"SHA-256 ... matches ANY earlier iteration in the same task trajectory"* -- a visited-set/Floyd-Brent cycle check (0.7%-3.8%); `category_oscillation` = *"at least three category transitions in the window of the last five iterations"* (0.0%-12.4%); `no_progress` = *"the tuple (category, primary error signature) is unchanged across three consecutive iterations"* (**44.6%-84.6%, dominant across most models**). Rationale: *"pass@k captures whether the task passes within k iterations but not whether each iteration does real work"*, so the signals *"decompose 'budget burned without progress' into interpretable failure modes."* Note the ordering: the two HASH-based signals are the weak ones; the SEMANTIC invariant (error signature unchanged) is the one that fires on the overwhelming majority of stalled runs. |
| 7 | https://docs.temporal.io/encyclopedia/retry-policies | 2026-08-21 | official docs (Temporal, Retry Policies) | WebFetch | The cleanest published statement of the transient/permanent split, and it maps exactly onto the rail-drop exemption: *"Permanent failures, by definition, require you to make some change to your logic or your input. Therefore, it is better to surface them than to retry them."* Contrapositive: a TRANSIENT failure requires NO change to input, so a byte-identical retry is the CORRECT response, not an evasion. Classification is carried on the failure itself (`Non-Retryable Error Types`, matched against the `type` field of Application Failures) rather than re-derived by the caller. Also: *"Setting the value to 0 also means unlimited. Setting the value to 1 means a single execution attempt and no retries."* Workflow Executions do NOT retry by default because *"retrying an entire Workflow Execution ... would repeat the same logic without resolving the underlying issue"* -- i.e. an unchanged retry of a deterministic unit is explicitly recognised as useless, while an unchanged retry of a transient one is explicitly recognised as useful. Activities are expected to be idempotent precisely because they re-execute on failure. |
| 8 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-21 | official docs (Anthropic, canonical harness reference for this project) | WebFetch | Confirms the gate shape this repo already uses and, importantly, confirms a GAP: *"Each criterion had a hard threshold, and if any one fell below it, the sprint failed and the generator got detailed feedback on what went wrong"*; *"the generator and evaluator negotiated a sprint contract"*; *"Communication was handled via files: one agent would write a file, another agent would read it and respond either within that file or a new file."* On the specific question 90.3 asks -- when to STOP retrying -- the article provides **no explicit guidance** on maximum retry attempts, escalation procedure, or human handoff. So the attempt/progress bound is genuinely this project's own design problem and must not be attributed to Anthropic (consistent with `attempt_budget.py:22-24`, which already says the 5/1.2M ceilings are sourced internally). |
| 9 | https://arxiv.org/html/2607.00038 | 2026-08-21 | preprint (arXiv, "Stop Hand-Holding Your Coding Agent: Engineering the Loops that Replace Step-by-Step Prompting") | WebFetch, native arXiv HTML | Names the terminal-state vocabulary this repo half-has. S4 "Stopping rule and named terminal states": *"a well-formed loop specification distinguishes its terminal states by name: success, a clean no-op, blocked, stalled, exhausted. Crucially, an error or an exhausted budget never counts as success."* Stopping is triggered by *"the goal being met, a stagnation detector firing after rounds without progress, or a budget ceiling"* -- THREE separate mechanisms, not one. Anti-pattern "Unattended Runaway": *"A loop with no task-related stopping rule, no stagnation detector, and no budget ceiling circles a problem it cannot solve until cost is exhausted."* The no-progress detector is drawn as a distinct component of loop anatomy (Fig. 2), separate from the budget ceiling. |
| 10 | https://arxiv.org/html/2608.10729v1 | 2026-08-21 | preprint (arXiv 2026-08, "Optimal Stopping of Self-Refining Foundation Models") | WebFetch, native arXiv HTML | The strongest theoretical backing for criterion 5. The stopping policy is a function of the ABSOLUTE quality score, never of inter-iteration change: S3, *"a stopping policy mu uses the current score to decide whether to stop"*; S4 objective *"maximize the expected payoff E{g(x_tau)} while minimizing refinement costs"*. Explicitly *"stopping depends on absolute score, not score change between iterations -- a structurally different approach that sidesteps the measurement problem."* Empirics (S9): *"all three models improve output quality through self-refinement, but with progressively smaller gains per iteration"* -- diminishing returns, not regression. Observation 1 / Proposition 4: monotonicity of the identified model yields a **stage-independent threshold** -- one score threshold stops regardless of iteration number. |

## Search queries run (three-variant discipline, per `.claude/rules/research-gate.md`)

| Variant | Query |
|---|---|
| year-less canonical | `LLM agent iterative repair loop stagnation detection termination criterion no-progress` |
| year-less canonical | `retry budget token bucket transient vs permanent failure idempotency key distributed systems best practice` |
| current-year frontier (2026) | `agent loop detection repeated state tabu visited set oscillation revert cycle coding agent 2026` |
| last-2-year window (2025) | `"self-refine" OR "iterative refinement" LLM stopping criterion when to stop 2025 diminishing returns oscillation` |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/pdf/2511.00592 | preprint | Agentic auto-scheduling; loop-optimization domain, off-topic for the gate |
| https://arxiv.org/pdf/2605.27716 | preprint | Web-accessibility repair; cost/remediation angle only |
| https://arxiv.org/pdf/2606.06741 | preprint | OpenSkill self-evolution; adjacent, budget spent elsewhere |
| https://arxiv.org/pdf/2606.24820 | preprint | SHERLOC diagnostic localization; about WHERE to fix, not WHEN to stop |
| https://arxiv.org/pdf/2606.24937 | preprint | Hitchhiker's Guide to Agentic AI; survey-tier overlap with #9 |
| https://www.emergentmind.com/topics/repairagent | secondary summary | Summary site, not a primary source |
| https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/ | official docs | ATTEMPTED, 301 -> builder.aws.com; body not retrievable |
| https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter | official docs | ATTEMPTED, page returned header only (JS-rendered). Substituted Google SRE (#5) + Temporal (#7) |
| https://docs.aws.amazon.com/whitepapers/latest/aws-fault-isolation-boundaries/retries-with-backoff-and-jitter.html | official docs | ATTEMPTED, nav-only render (same class as the known GCloud-docs trap) |
| https://cacm.acm.org/research/proving-program-termination/ | peer-reviewed (CACM) | ATTEMPTED, HTTP 403. Wanted for the well-founded-ranking-function argument; substituted by #10's absolute-score formulation |
| https://www.alphaxiv.org/abs/2303.17651 | preprint mirror (Self-Refine) | Mirror of the canonical Self-Refine paper; its stop rule is subsumed by #10 |
| https://arxiv.org/pdf/2510.02919 | preprint | Self-reflective generation at test time |
| https://arxiv.org/pdf/2310.05035 | preprint | Self-Convinced Prompting / repeated introspection |
| https://www.emergentmind.com/topics/iterative-self-refinement | secondary summary | Aggregator |
| https://www.myrobertson.com/blog/retry-strategies-and-idempotency | community blog | Community tier; superseded by #5/#7 |
| https://aloknecessary.in/blogs/idempotency-distributed-systems/ | community blog | Community tier |
| https://bhavishyapandit9.substack.com/p/idempotency-and-retry-semantics-for | community blog | Community tier |
| https://www.geeksforgeeks.org/system-design/retries-strategies-in-distributed-systems/ | community | Lowest tier |
| https://hokstadconsulting.com/blog/retry-backoff-in-distributed-systems | industry blog | Restates #5 |
| https://atlan.com/know/ai-agent/what-is-an-agent-loop/ | vendor blog | Vendor explainer |
| https://blogs.oracle.com/developers/the-agent-loop-decoded-three-levels-every-agent-engineer-must-know | vendor blog | Vendor explainer |
| https://datasciencedojo.com/blog/agentic-loops-explained-from-react-to-loop-engineering-2026-guide/ | vendor blog | Vendor explainer; source of the "hash each (tool,args) pair; if the same call appears k times, break" idiom |
| https://markaicode.com/fix-ai-agent-looping-autonomous-coding/ | community blog | Community tier |
| https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11429435 | patent | "Distributed execution budget management system"; prior art, not needed for the design |

**URL accounting:** 10 read in full + 24 snippet-only = **34 unique URLs**, de-duplicated (the `arxiv.org/abs/2607.01641` and `arxiv.org/pdf/2607.01641` variants of source #1 are NOT counted again).

## Recency scan (2024-2026) -- MANDATORY SECTION

Searched the 2024-2026 window explicitly (queries 3 and 4 above, plus the year-less
queries which returned overwhelmingly 2026 preprints). Result: **found 5 new findings that
supersede or complement the canonical sources.** Seven of the ten read-in-full sources are
inside the window, five of them 2026 preprints: 2605.01471, 2605.08455, 2607.00038,
2607.01641 and 2608.10729 (the last dated 2026-08, ~3 weeks old). What is NEW in the window
and did not exist when this repo's attempt-budget doctrine was written:

1. **A named, measured stagnation-signal vocabulary exists now** (CUDABeaver, #6). Before
   2026 the literature offered iteration caps; it now offers four signals with per-model
   firing rates, and the measurement inverts the naive intuition -- the hash signals are the
   weak ones and the semantic-invariant signal dominates.
2. **A named terminal-state vocabulary** (#9): success / clean no-op / blocked / stalled /
   exhausted, with "an error or an exhausted budget never counts as success".
3. **A structural taxonomy of unbounded agent loops with population statistics** (#1):
   "Retry feedback without bounds" is the LARGEST class at 25.0% of 68 confirmed cases.
4. **Optimal-stopping theory applied to self-refining models** (#10), which stops on an
   absolute score and explicitly declines to use inter-iteration change.
5. **A field measurement of retry-without-progress cost** (#2): 113 consecutive fruitless
   reports, 30% of families never converging.

The older canonical sources are NOT superseded: Huang et al. (#3, ICLR 2024) remains the
citation for "changed != improved", and Google SRE (#5) remains the citation for
work-accounting retry budgets. Anthropic's harness-design article (#8) is current and,
notably, still says nothing about when to stop retrying.

## Key findings (each cited per claim)

1. **A content digest is the WEAK stagnation signal; the strong one is a semantic
   invariant.** CUDABeaver's four signals are measured side by side: the SHA-256 signals
   `duplicate_code` (0.0-50.8%) and `code_cycle` (0.7-3.8%) versus `no_progress` --
   *"the tuple (category, primary error signature) is unchanged across three consecutive
   iterations"* -- at **44.6-84.6%, dominant across most models**
   (https://arxiv.org/html/2605.08455, accessed 2026-08-21). This is the external
   confirmation of 90.3's audit_basis: a hash catches only the degenerate case; the loops
   that actually burn budget change bytes while the *failure* stays the same.
2. **"Cannot distinguish genuine progress from lateral variation" is the literature's own
   phrasing.** *"Without a formal correctness signal, the agent cannot distinguish genuine
   progress from lateral variation"* -- repairs altered *"selectors, assertions, or
   interaction sequences without resolving the underlying failure"*, producing *"113
   consecutive reports ... without ever producing a single executable test artifact"*
   (https://arxiv.org/html/2605.01471, S5.2/S5.2, accessed 2026-08-21). Maximum change,
   zero progress -- the exact regime 89.1's digest would score as healthy.
3. **Changed does not mean improved, and the gains in the self-correction literature came
   from the STOPPING RULE, not the correction.** *"After self-correction, the accuracies of
   all models drop across all benchmarks"*; prior positive results *"used the correct label
   to determine when to stop the self-correction loop"*; and *"the model is more likely to
   modify a correct answer to an incorrect one than to revise an incorrect answer to a
   correct one"* (Huang et al., ICLR 2024,
   https://ar5iv.labs.arxiv.org/html/2310.01798, S3.2/S3.3, accessed 2026-08-21).
4. **The frontier stopping formulation deliberately avoids inter-iteration change.**
   *"a stopping policy mu uses the current score to decide whether to stop"* -- absolute
   score, *"not score change between iterations -- a structurally different approach that
   sidesteps the measurement problem"* (https://arxiv.org/html/2608.10729v1, S3/S5-C,
   accessed 2026-08-21). Criterion 5's prohibition on reading a changed digest as
   convergence evidence is therefore aligned with the current literature, not a local
   quirk.
5. **Stopping needs THREE separate mechanisms and NAMED terminal states.** *"a well-formed
   loop specification distinguishes its terminal states by name: success, a clean no-op,
   blocked, stalled, exhausted. Crucially, an error or an exhausted budget never counts as
   success"*; stopping fires on *"the goal being met, a stagnation detector firing after
   rounds without progress, or a budget ceiling"*
   (https://arxiv.org/html/2607.00038, S4, accessed 2026-08-21). pyfinagent has the budget
   ceiling and the goal; **`stalled` is the name it lacks**, and criterion 4's demand that
   inputs-incomplete carry *its own machine reason, distinct from a gate crash* is this
   vocabulary in miniature.
6. **Unbounded retry feedback is the single largest published class of runaway agent
   loops.** An Infinite Agentic Loop is *"a structural execution failure where an agentic
   feedback path repeatedly triggers costly or state-growing actions without an effective
   stopping bound"*; *"Retry feedback without bounds"* is 25.0% of 68 confirmed cases across
   47 projects, and *"95.6% cause API cost exhaustion or model DoS"*
   (https://arxiv.org/html/2607.01641v1, S3-A / Table II, accessed 2026-08-21). The same
   paper separates state growth from progress: the motivating loop grows *"the enlarged
   message history"* every iteration with no termination guarantee.
7. **Transient vs permanent has a crisp published definition that decides criterion 2 on
   principle, not just on the 14/16 count.** *"Permanent failures, by definition, require you
   to make some change to your logic or your input. Therefore, it is better to surface them
   than to retry them"* (https://docs.temporal.io/encyclopedia/retry-policies, accessed
   2026-08-21). A rail drop requires no change to the input -- so a byte-identical relaunch
   after a drop is the textbook-correct retry, and denying it (89.1 criterion 2) inverts the
   rule. Temporal also carries the classification **on the failure** (Non-Retryable Error
   Types matched against the failure's `type`), not re-derived by the caller -- which argues
   for keying the exemption on the recorded outcome/reason field rather than on a heuristic.
8. **Fail-open vs fail-closed: the canonical vocabulary routes TIMEOUTS through the same
   policy, and warns about self-reference.** `failurePolicy: Fail` -> *"the request is
   rejected"*; `Ignore` -> *"the request is allowed to proceed as if the webhook was not
   called"*; *"If the webhook call times out, the request is handled according to the
   webhook's failure policy"*; *"Admission webhooks should be idempotent"*; and a webhook
   that intercepts the resources running its own infrastructure deadlocks
   (https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/,
   accessed 2026-08-21). Kubernetes defaults CLOSED; the Claude Code hook contract forces
   this gate OPEN (only exit 2 blocks), which `attempt_gate.py:35-42` already states. The
   useful distinction the k8s model adds: a gate that RESPONDS with a denial is not the same
   event as a gate that FAILS -- which is precisely criterion 4's split.
9. **Retry budgets are work-accounting, and the error should carry the retry decision.**
   Google SRE runs two at once: a *"per-request retry budget of up to three attempts"* and a
   *"per-client retry budget ... A request will only be retried as long as this ratio is
   below 10%"*, plus *"an 'overloaded; don't retry' error"* and retry at exactly one layer
   because *"if multiple layers retried, we'd have a combinatorial explosion"*
   (https://sre.google/sre-book/handling-overload/, accessed 2026-08-21). This corroborates
   `attempt_budget.py:13-24` verbatim and is the correct external anchor -- **not** Anthropic.
10. **Anthropic's harness-design article does not answer this question.** It specifies hard
    per-criterion thresholds, the sprint contract, and file-based handoffs, but provides no
    guidance on maximum retry attempts, escalation, or when to stop
    (https://www.anthropic.com/engineering/harness-design-long-running-apps, accessed
    2026-08-21). Do not cite it as the source of any ceiling.

## Internal code inventory (every claim carries a file:line anchor)

| File / artifact | Anchor | Role | Status |
|---|---|---|---|
| `scripts/harness/attempt_gate.py` | 824 lines total | The PreToolUse gate 90.3 modifies | LIVE |
| ^ hook entry / decision | `:382-470` `handle_hook()` | reads stdin, attributes, allows or denies | LIVE |
| ^ the attempt row write | `:445-463` `append_row({...})` | **the natural insertion point for a digest field**; already writes explicit-null `outcome` / `total_tokens` / `run_id` at launch | LIVE |
| ^ fail-open handler | `:465-470` | `except Exception` -> stderr + `return 0`; docstring rationale at `:35-42` | LIVE |
| ^ decision + state | `:276-281` `decide()`, `:224-273` `build_state()` | where a digest comparison would have to bind | LIVE |
| ^ escalation writer | `:340-379` `write_escalation()`, reasons at `:331-337` | already reason-named (`attempt_budget` / `unknown_step_id`) -- a third reason for no-progress fits the existing shape | LIVE (90.1) |
| ^ self-test sandbox | `:518-800`, containment refusal `:554-561` | ~40 checks; refuses to run if any output path resolves inside the repo | LIVE |
| ^ digest/hash concept | **grep `sha256\|hashlib\|digest` over `attempt_gate.py`, `attempt_budget.py`, `attempt_outcomes.py` returns ZERO hits** | confirms 89.1's audit_basis: the gate has no evidence concept at all | ABSENT |
| `.claude/settings.json` | `:39` | `"command": "python3 .../attempt_gate.py"` under `"matcher": "Workflow"`, `timeout: 45` | WIRED (confirms CLAUDE.md) |
| `scripts/harness/attempt_budget.py` | `:59` `DEFAULT_MAX_ATTEMPTS=5`, `:64` `DEFAULT_MAX_TOKENS=1_200_000` | the ceilings | LIVE |
| ^ outcome vocabulary | `:67-78` `Outcome{PASS,CONDITIONAL,FAIL,NO_VERDICT}`, `:81-84` `Disposition{CONTINUE,CLOSED_PASS,ESCALATE}` | **no `STALLED` disposition exists** -- see finding 5 | GAP |
| ^ health-check vs work-accounting | `:13-24` | already cites Google SRE correctly and forbids attributing the ceilings to Anthropic | LIVE |
| `scripts/harness/attempt_outcomes.py` | `:227-241` | NO_VERDICT **reason** vocabulary: `killed`, `args_unparseable`, `structured_output_drop`, `not_an_evaluation`, `completed_without_result`, `no_verdict_other`; `:253` `UNKNOWN`/`no_run_record` | LIVE (90.1) |
| `handoff/verdict_ledger.jsonl` | measured 2026-08-21 | **146 rows** (89.1 said 134; 90.3's audit_basis says 138) -- CONDITIONAL 79 / FAIL 32 / PASS 19 / **NO_VERDICT 16 (11.0%)** | DENOMINATOR MOVED |
| ^ the 75.11.4 fixture pair | ledger rows cycle 3 + 4 | CONFIRMED verbatim: NO_VERDICT `recorded_at 19:39:57.905629Z`, note *"RAIL DROP: API Error connection lost mid-response; result null, agents_error=1, 188930 subagent tokens spent, 33 tool uses"*, then CONDITIONAL `19:58:38.558362Z`; step later **PASS** (cycle 5, backfilled 2026-08-18) | REPRODUCIBLE |
| `handoff/audit/attempt_budget_audit.jsonl` | measured 2026-08-21 | **118 rows** (90.1 says 92; CLAUDE.md says 93) = 114 `attempt` + 4 `operator_extension`; outcome mix CONDITIONAL 45 / null 26 / **NO_VERDICT 20 (17.5% of attempts)** / FAIL 11 / PASS 11 / UNKNOWN 5; 92 rows carry an int `total_tokens` | **90.1's backfill HAS ALREADY RUN** |
| ^ 90.1 prerequisite | `attempt_gate.py:87-89` imports `resolved_rows`, called at `:243` | 90.3's note "LAND AFTER 90.1" is **already satisfied on disk** even though step 90.1 is `status: pending` (parked) | SATISFIED |
| `.claude/agent-memory/qa/verdicts/` | **194 files, NOT gitignored**, contains `verdict_wip_*.md` written DURING a Q/A run | this is WHY criterion 1 excludes it: the Q/A's own WIP markers would advance the digest every launch | EXCLUSION CORROBORATED |
| `scripts/qa/mutation_matrix_90_3.py` | -- | required by 90.3's immutable command | **DOES NOT EXIST YET** (86_71, 90_1, 90_9 do) |
| masterplan step schema | union of step-level keys across every step | **no `files` / `paths` key exists on any step** | criterion 1's "declared masterplan paths" has NO existing source |

### The self-reference hazard (new, measured 2026-08-21 -- not named in 90.3's audit_basis)

Criterion 1 derives the file set from `git diff --name-only HEAD` union
`git ls-files --others --exclude-standard`. Run right now, that is **exactly**:

```
handoff/audit/attempt_budget_audit.jsonl      <- written BY THIS GATE, and TRACKED
handoff/audit/pre_tool_use_audit.jsonl        <- written by the PreToolUse danger guard on every tool call
handoff/current/research_brief_90.3.md        <- (this brief)
```

`git ls-files --error-unmatch handoff/audit/attempt_budget_audit.jsonl` -> TRACKED. So unless
the "checked-in root allowlist" **excludes `handoff/audit/`**, the gate's own append mutates
its own next input and the digest advances by construction on every launch -- reproducing
89.1's exact defect through a different door, and vacuously. This is the k8s self-referential
webhook deadlock (finding 8) in miniature. The allowlist choice is therefore load-bearing and
is not stated in the criteria; it must be decided in the contract.

Two corollaries: (a) after Main commits a cycle's fixes, `git diff --name-only HEAD` can be
**empty**, so the digest set collapses to the (currently non-existent) declared-paths member
-- and an empty set hashes identically for two different evidence states, which is exactly the
inputs-incomplete DENY of criterion 4; (b) the auto-commit hook commits on masterplan flip, so
whether evidence is committed at launch time varies by cycle.

### Criterion-5 grep noise (measured)

`grep -rIn 'digest' .claude/ scripts/` -> **1086 lines across 111 files**, overwhelmingly
`.claude/.masterplan.json.bak.*` snapshots that contain 89.1's and 90.3's own criteria text.
A naive grep-based "no consumer treats the digest as progress" test therefore matches its own
specification and is ~all noise (the same over-report pattern measured at 91.9). The test must
use a targeted consumer pattern and exclude `.claude/.masterplan.json.bak.*`.

## Consensus vs debate (external)

**Consensus:** every source that bounds a loop uses a CUMULATIVE budget plus escalation, and
none of them treats "the output changed" as evidence of progress (#2 S9, #5, #7, #9). All agree
that terminal states must be distinguishable and that exhaustion is never success (#9).

**Debate / genuine divergence:** *whether hashing belongs in the design at all.* CUDABeaver (#6)
ships two hash-based signals and measures them; IAL-Scan (#1) explicitly declines duplicate-state
detection and reasons over the call graph instead (SCCs); the optimal-stopping formulation (#10)
uses neither and stops on an absolute score. The resolution consistent with all three is the one
90.3 already encodes: **use the hash for the CYCLE case only (a visited SET, not a
previous-attempt comparison), and never promote it to a progress or convergence signal.**

## Pitfalls (from the literature and from this repo's own history)

- **Exact-match hashing fails open under trivial perturbation** -- CUDABeaver's `duplicate_code`
  fires on 0.0% of runs for some models and 50.8% for others (#6); deepseek's repeat-tool-reminder
  README (quoted in 90.3's own audit_basis) names *"a tweaked path, extra whitespace inside a
  value"* as known evasions.
- **A monotonic counter beats a content hash against reverts** -- deepseek's `replaceGeneration`
  is compared `<=` so a revert cannot restore an admissible state (90.3 audit_basis). A digest SET
  (criterion 3) gets the same property a different way; a previous-attempt comparison does not.
- **Retry is not recovery** -- *"retries helped 7 of 10 families converge, but the same mechanism
  sustained 113 sequential fruitless reports"* (#2 S6.8).
- **The stopping rule can silently be the thing doing the work** (#3): an oracle-free loop that
  observes only change will convert correct answers to incorrect ones more often than the reverse.
- **Denominators in this step's own audit_basis have drifted**: verdict ledger 134 -> 138 -> **146**
  live; attempt ledger 92/93 -> **118** live. Re-derive before quoting any of them in the contract.
- **A grep-based negative test can match its own documentation** (measured above, 111 files).

## Application to pyfinagent (external findings -> file:line anchors)

1. **Criterion 2's exemption should key on the recorded reason, not just the value.**
   `attempt_outcomes.py:227-241` already distinguishes `structured_output_drop`,
   `completed_without_result` and `killed` (genuine transients) from `not_an_evaluation` and
   `args_unparseable` (NOT rail drops). Temporal's rule (#7) -- permanent failures are the ones
   requiring an input change -- says the exemption belongs to the first group. Exempting every
   `NO_VERDICT` is broader than the criterion's own stated fixture. 20 of 114 attempt rows (17.5%)
   carry `NO_VERDICT`, so this branch is exercised roughly 1 launch in 6.
2. **Add the missing terminal-state NAME** (#9). `attempt_budget.py:81-84` has
   CONTINUE / CLOSED_PASS / ESCALATE. A no-progress denial is neither exhaustion nor a verdict;
   `write_escalation`'s reason-named path (`attempt_gate.py:331-337`, added by 90.1 for exactly
   this reason) is the existing idiom to extend -- and 90.1 already proved that reusing the
   `attempt_budget` name forges an exhaustion record.
3. **Set the allowlist so `handoff/audit/` is excluded** -- see the self-reference hazard above.
   Otherwise criterion 1 is satisfiable on paper and vacuous in fact.
4. **Keep the fail-open handler at `attempt_gate.py:465-470` exactly as it is** (criterion 8 of
   89.1, criterion 4 of 90.3), and make the crash path append a row with `digest: null` + an
   explicit unavailable status so the NEXT launch has no comparable baseline. Kubernetes (#8)
   supports this split: a gate that responds with a denial is a different event from a gate that
   fails, and the timeout must route through the declared policy rather than a second rule.
5. **Criterion 5 is defensible on the literature, not just on the 86.118/86.116/86.108 evidence**
   (#3, #10). Re-measured 2026-08-21, those three steps' artifacts were committed on essentially
   every cycle -- 86.118: 4 critique / 4 experiment commits; 86.116: 5 / 6; 86.108: 2 / 3 -- so
   digest-advanced was TRUE while all three ended FAIL at 5/5. NOTE the "perfect 1:1" phrasing in
   89.1's audit_basis reproduces exactly for 86.118 only; the direction of the claim holds for all
   three, the exact pairing does not. Re-derive before restating.
6. **`scripts/qa/mutation_matrix_90_3.py` must be authored** -- the immutable command depends on it,
   and the existing `mutation_matrix_90_1.py` / `86_71.py` are the shape to copy. The self-test's
   repo-containment refusal (`attempt_gate.py:554-561`) must be extended to any NEW output channel
   the digest work introduces; that lesson is written into the file at `:536-561` after two real
   leaks (the 9.4 extension row and `escalation_unknown_step_id_9.9.md`).

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **10** (4 arXiv preprints,
      1 peer-reviewed ICLR paper via ar5iv, 4 official docs, 1 vendor engineering article)
- [x] 10+ unique URLs total (incl. snippet-only) -- **34** (10 full + 24 snippet-only, de-duped)
- [x] Recency scan (last 2 years) performed + reported -- dedicated section above, 5 new findings
- [x] Full papers / pages read (not abstracts) for the read-in-full set -- arXiv native HTML used
      for every 2026 preprint, ar5iv for the pre-Dec-2023 paper; no `/pdf/` URL was WebFetched
- [x] file:line anchors for every internal claim -- inventory table above

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE
      (`attempt_gate.py` read end to end; `attempt_budget.py` + `attempt_outcomes.py` read at the
      docstring/vocabulary level; both ledgers parsed; masterplan 90.3/90.1/89.1 read verbatim;
      `.claude/settings.json:39` confirmed)
- [x] Contradictions / consensus noted -- see "Consensus vs debate"; the hash-vs-no-hash divergence
      between sources #1, #6 and #10 is reported rather than smoothed over
- [x] All claims cited per-claim with URL + access date, not only in a footer
- [ ] GAP, disclosed: the AWS Builders' Library retry article and the CACM termination paper both
      failed to fetch (301-to-empty and HTTP 403). Google SRE (#5) and Temporal (#7) cover the retry
      half; the well-founded-ranking-function argument is carried only by #10's absolute-score
      formulation, which is weaker than a formal termination-measure citation would have been.

*(Tier: moderate. Rounds: 3. Not audit-class -- `coverage` is informational and does not gate.)*
