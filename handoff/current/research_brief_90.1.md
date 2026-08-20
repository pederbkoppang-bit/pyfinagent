# Research Brief -- step 90.1

**Tier:** moderate (caller-stated). Audit-class: NO (coverage reported for information only).

**Objective:** Attempt/outcome accounting for bounded retry loops in long-running LLM agent
harnesses -- terminal outcome vocabularies (complete / blocked / budget-limited), token
accounting on attempt records, distinguishing a GRADED attempt from an INFRASTRUCTURE DROP,
and validating identifiers against a plan of record so a budget ceiling cannot be reset by an
unrecognized key.

## Envelope (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 15,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {"audit_class": false, "rounds": 3, "dry_rounds": 0, "K_required": 2,
               "new_findings_last_round": 2, "dry": false},
  "gate_passed": true
}
```

---
_(sections appended incrementally below as sources land)_

## Internal code inventory (Explore half)

| File | Lines | Role | Status |
|---|---|---|---|
| `scripts/harness/attempt_gate.py` | 441 | PreToolUse hook on `Workflow`; attributes a launch to a step, appends an attempt row, denies at the ceiling | WIRED (`.claude/settings.json` PreToolUse matcher `Workflow`, timeout 45) |
| `scripts/harness/attempt_budget.py` | 337 | `Outcome` / `Disposition` enums, `BudgetState`, escalation summary, 86.28 replay fixture | Library; imported by the gate |
| `handoff/audit/attempt_budget_audit.jsonl` | 93 rows | append-only attempt ledger (89 `attempt` + 4 `operator_extension`) | LIVE since 2026-08-17 |
| `handoff/verdict_ledger.jsonl` | 138 rows | verdict stream (`CONDITIONAL` 75, `FAIL` 28, `PASS` 19, `NO_VERDICT` 16) over 30 step-ids | LIVE, predates the gate |
| `scripts/qa/mutation_matrix_86_71.py` | 354 | mutation matrix for the gate | see below |
| `~/.claude/projects/<proj>/*/workflows/wf_*.json` | 617 records | Workflow run records | see below |

### I-1. There is NO `outcome` key on a Workflow run record -- 617/617

Measured over all 617 run records under
`~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/*/workflows/wf_*.json`:
the top-level key set is
`{runId, timestamp, taskId, script, scriptPath, result, agentCount, logs, durationMs,
summary, workflowName, status, startTime, defaultModel, workflowProgress, totalTokens,
totalToolCalls, phases(614), args(542), error(53)}`.
**`outcome` appears on ZERO records.** The terminal-state field is `status`.
Any design that says "resolve outcome + total_tokens from the run record" must read
`status`, not `outcome`, or it reads `None` on every record.

### I-2. `totalTokens` IS present and populated -- 617/617 present, 608 non-zero

So token accounting is *available* at the run-record seam. It is simply never carried
onto an attempt row (see I-3).

### I-3. The token ceiling is structurally unreachable in the wired path

`attempt_budget.py:64` sets `DEFAULT_MAX_TOKENS = 1_200_000` and `:128-130` makes
`exhausted` true when `attempts_used >= max_attempts` **or** `tokens_used >= max_tokens`.
But the only wired producer of attempts, `attempt_gate.py:190-193`, calls
`state.record(outcome)` with **no `tokens=` argument**, and `Attempt.tokens` defaults to
`0` (`attempt_budget.py:91`). Therefore `tokens_used` is **always 0** and the token half
of the ceiling can never bind. Confirmed against the ledger: **no attempt row in
`handoff/audit/attempt_budget_audit.jsonl` carries any token-shaped key** -- the row key
set is exactly `{ts, type, step_id, workflow, tool_use_id, session_id,
attempt_number_inclusive, note}`. The 1.2M ceiling is, today, decorative.

### I-4. Outcomes are paired to attempts POSITIONALLY across two different populations

`attempt_gate.py:190-191`:
`outcome = verdicts[i] if i < len(verdicts) else Outcome.NO_VERDICT` -- the i-th attempt
row is paired with the i-th verdict-ledger row. The two ledgers are different populations
with different start dates (the attempt ledger begins 2026-08-17; the verdict ledger goes
back to cycle 190). This is the same positional-pairing failure the module's own docstring
records as the reason the first 86.28 fixture was wrong
(`attempt_budget.py:240-256`: "built by parsing ... in DOCUMENT ORDER ... a positional
parse cannot tell them apart", 3 spurious / 2 omitted of 8). The two streams share no join
key today: attempt rows carry `tool_use_id` (`toolu_*`) + `session_id`; verdict rows carry
`run_id` (`wf_*`).

### I-5. Step-ids are validated for SHAPE, never against the plan of record

`attempt_gate.py:97` `_STEP_ID_RE = re.compile(r"\A[0-9]+(?:\.[0-9]+)*\Z")` accepts any
dotted-numeric string. It is a *syntactic* filter (it correctly refuses `../evil`), not a
*membership* check. Measured: of the 27 distinct step-ids in the live attempt ledger,
**`999.2` is ABSENT from `.claude/masterplan.json`** (1457 step-shaped `{id,status}`
objects) and holds 5 attempt rows plus a written escalation file
(`handoff/current/escalation_attempt_budget_999.2.md`). A single unrecognised key
therefore already owns its own private 5-attempt allowance. The mechanism is exactly the
unbounded-key problem OTel's cardinality limit exists to contain (see external E-2).

### I-6. A self-test id COLLIDED with a real masterplan step

`attempt_gate.py:379-388`'s self-test uses synthetic ids `9.1`/`9.4`; `read_ledger`'s
docstring (`:132-137`) already discloses that one run "appended its synthetic 9.4 extension
row to the real audit stream". Both `9.1` and `9.4` are **real masterplan steps**
(`status: done`). So the leaked row did not merely pollute the ledger -- it raised the
allowance of a genuine step. Membership validation alone would NOT have caught this one;
it needs a namespace separation between test and production ids as well.


### I-7. The 5-attempt allowance is shared by TWO different budgets

`attempt_gate.build_state()` (`attempt_gate.py:182-184`) counts **every** row with
`type == "attempt"`, ignoring the `workflow` field. But the ledger holds launches of
*both* Layer-3 rails, and `scripts/harness/research_router.py:30-35` reads the **same**
stream and uses the `workflow` name to count *re-research rounds* against its own Tmax=2.
So a `research-gate.js` launch today consumes one of the step's five **Q/A** attempts.
Two budgets, one counter, no discriminator.

### I-8. The token denominator is undefined, and the choice moves the answer by 5 steps

Re-derived independently from the 617 run records:

| population | step-attributed runs | steps whose cumulative tokens exceed 1.2M | max step |
|---|---|---|---|
| `qa-verdict` only | 441 | **13** | 86.85 @ 2,506,619 |
| every step-attributed Workflow (`qa-verdict` 435 + `research-gate` 104 + 1 probe) | 540 | **18** | 86.85 @ **2,677,199** |

The masterplan's `audit_basis` for 90.1 says *"MEASURED ... over 441 `qa-verdict` Workflow
run records"* and then quotes *"18 steps exceeded 1.2M tokens ... (max 2,677,199 on
86.85)"*. Both figures are correct, but **for different populations**: 18/2,677,199
reproduces only on the 540-run superset. Restricted to the 441 qa-verdict runs the answer
is 13 / 2,506,619. The design must therefore STATE which population the token ceiling
sums over; it is not a detail, it changes the bound's bite by 5 steps.
(Verdict re-derivation matched the audit_basis exactly: 441 qa-verdict runs, CONDITIONAL
221 / PASS 109 / FAIL 67 = 397 with a verdict, **44 (10.0%) with none**.)

### I-9. What a rail drop actually looks like in the run record -- measured

Over all 617 records, `status` takes exactly three values and there is a clean 1:1 mapping
to the error class:

| status | n | error class | mean tokens |
|---|---|---|---|
| `completed` | 564 | none | 242,997 |
| `failed` | 46 | `subagent completed without calling StructuredOutput (after in-conversation nudge)` | **191,796** |
| `failed` | 2 | `qa-verdict: args are PRESENT but not parseable as JSON` | 0 |
| `killed` | 5 | `Workflow aborted` | 105,889 |

Every one of the 46 StructuredOutput drops carries `status: "failed"`, so the drop IS
mechanically separable from a graded attempt **at the run record** -- the information the
attempt row throws away is available, and free. And the drops are not cheap: 8,822,653
tokens total, median 180,539, max 385,807 -- i.e. a drop costs about the same as a
successful run (191,796 vs 242,997 mean). That is the quantitative case for recording
`total_tokens` per attempt rather than counting bare attempts.

Note the 2 zero-token `args-unparseable` failures are a genuinely *different* class: they
cost nothing and are a caller bug, not a rail drop. A binary `completed`/`not-completed`
vocabulary would merge them with the 46 expensive drops.

### I-10. `tool-ralph` has no in-repo corroboration

The step's `audit_basis` names *"deepseek-harness `tool-ralph`"* as the MECHANISM SOURCE
for the closed vocabulary `complete | blocked | budget-limited`. A repo-wide search for
`tool-ralph`/`tool_ralph` returns exactly ONE hit -- the `audit_basis` string itself in
`.claude/masterplan.json:27087`. The 450-line internal audit
`docs/audits/deepseek-harness-2026-08-18.md` (which lists the exact deepseek-harness files
read, at commit `99f6f02`) contains **zero** occurrences of "ralph". The vocabulary may
still be right; it is simply not corroborated by the artifact the project cites, so the
contract should not present it as an established external precedent without a fresh fetch.


---

## External research

### Search-query composition (three-variant discipline)

| variant | query |
|---|---|
| year-less canonical | `retry budget bounded retries distributed systems terminal outcome status vocabulary` |
| year-less canonical | `Kubernetes Job podFailurePolicy backoffLimit disruption not counted toward retry limit infrastructure failure` |
| year-less canonical | `OpenTelemetry metrics cardinality limit overflow attribute unbounded label new time series` |
| current-year (2026) | `LLM agent harness attempt accounting infrastructure error vs task failure 2026` |
| last-2-year (2025) | `agent evaluation harness "infrastructure error" excluded from score attempt budget termination 2025` |
| last-2-year (2025/26) | `idempotency key validation unknown key creates new record allowlist known identifiers 2025 2026` |
| targeted | `deepseek-harness tool-ralph terminal outcome complete blocked budget-limited report` |

### Read in full (WebFetch; counts toward the gate)

| # | URL | Accessed | Kind | Key finding |
|---|---|---|---|---|
| 1 | https://kubernetes.io/docs/concepts/workloads/controllers/job/ | 2026-08-20 | official doc | Terminal Job condition vocabulary is a CLOSED set -- `Complete`, `Failed`, `FailureTarget`, `SuccessCriteriaMet`, `Suspended` -- and a `Failed` condition carries a machine REASON (`BackoffLimitExceeded`, `DeadlineExceeded`, `PodFailurePolicy`, `MaxFailedIndexesExceeded`). The page is long and the fetch truncated; the retry-counting detail was taken from source #4 instead. |
| 2 | https://kubernetes.io/blog/2024/08/19/kubernetes-1-31-pod-failure-policy-for-jobs-goes-ga/ | 2026-08-20 | official blog (2024) | The exact mechanism 90.1 needs: `"Ignore: Do not count the failure towards the backoffLimit or backoffLimitPerIndex."` vs `"Count: Count the failure towards the backoffLimit... This is the default behavior."` The discriminator is a first-class condition on the record: `"To allow matching Pod failure policy rules against failures caused by disruptions initiated by Kubernetes, this feature introduces the DisruptionTarget Pod condition."` -- Kubernetes adds it `"to any Pod... that fails because of a retriable disruption scenario"`. |
| 3 | https://sre.google/sre-book/handling-overload/ | 2026-08-20 | canonical practitioner (Google SRE) | Two budgets, not one: `"a per-request retry budget of up to three attempts"` AND `"Each client keeps track of the ratio of requests that correspond to retries. A request will only be retried as long as this ratio is below 10%."` Also the amplification rule -- `"If multiple layers retried, we'd have a combinatorial explosion"` and `"a failed request from the DB Frontend should only be retried by Backend B, the layer immediately above it."` |
| 4 | https://docs.temporal.io/workflow-execution | 2026-08-20 | official doc | A production terminal vocabulary with SIX closed values: `Completed`, `Failed`, `Timed Out`, `Cancelled`, `Terminated`, `Continued-As-New` (open: `Running`, `Paused`). Note it separates `Failed` (`"returned an error and failed"`) from `Timed Out` (`"reached a timeout limit"`) from `Terminated` (forced) -- three distinct non-success ends, not one. |
| 5 | https://opentelemetry.io/blog/2026/cardinality-limits-in-opentelemetry/ | 2026-08-20 | official blog (2026) | The unknown-key countermeasure. `"The default aggregation cardinality limit is 2000 combinations per metric stream."` Past it, new combinations are NOT dropped: `"their values are folded into a single overflow data point marked with otel.metric.overflow=true"` and `"the original measurement attributes are removed from the overflow data point."` Guidance: `"Raw URLs, user input, request IDs, session IDs, and unbounded error messages usually should not be metric attributes"`; prefer bounded dimensions. |
| 6 | https://arxiv.org/html/2607.07946v1 | 2026-08-20 | preprint (DeepSWE) | §5.6 "Exclusions and failure scoring": infrastructure errors -- `"model-provider error, a verifier or grading error, or a transient network error"` -- are `"exclude[d] from both the numerator and the denominator of every metric rather than scoring them as failures."` `"Excluded rollouts are not resampled."` Measured `"excluded fraction ranges from 0%... to 5.3%"`. Terminal modes: submit a patch, exhaust the context window, or hit the timeout -- `"the latter two are scored as failures."` |
| 7 | https://arxiv.org/html/2607.12227v1 | 2026-08-20 | preprint | **[DISAGREES with #6]** §A.5: `"Rollouts that terminate on an infrastructure exception, such as a sandbox crash or an API timeout, are scored as r=0 rather than excluded from the average."` Reiterated: `"rollouts lost to infrastructure exceptions count as failures."` Budgets are declared (`"Max model turns: 300"`, `"Max generation per turn: 128,000 tokens"`) but no exhaustion vocabulary is defined. |
| 8 | https://arxiv.org/html/2606.06324v2 | 2026-08-20 | preprint (HarnessFix) | Failures `"cannot usually be mapped directly to specific locations in the harness implementation... Instead, they emerge in execution trajectories"` (§I). Offers a taxonomy of harness LAYERS, not of failure causes; `"Lifecycle, Tooling, and Observability are the most frequent sources of flaws, with nearly all LLM agents (29 out of 30) exhibiting flaws"` (§II-B2). Explicitly does NOT separate harness errors from task failures when computing task completion rate -- an absence that is itself the finding. |
| 9 | https://deepseek.com/harness/en/ | 2026-08-20 | vendor doc | Fetched specifically to corroborate the step's cited MECHANISM SOURCE. The page describes the plugin architecture and an append-only session log (`"Everything the model sees is recorded in an append-only session log"`) but contains **no** Ralph-loop outcome vocabulary, no budget/cap exhaustion semantics and no per-attempt token accounting. The `complete \| blocked \| budget-limited` triple is NOT corroborated here. |

### Attempted but NOT read in full (recorded honestly; do NOT count toward the gate)

| URL | Why not |
|---|---|
| https://github.com/deepseek-ai/deepseek-harness/blob/main/docs/agent-lifecycle.md | HTTP 404 -- path from the internal audit's source list does not resolve unauthenticated |
| https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/ | redirect stub only ("moved to the GenAI semconv repository"); no content |
| https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/ | same redirect stub |

### Identified, snippet-only (context)

| URL | Kind |
|---|---|
| https://arxiv.org/html/2606.20683v1 | survey: agent system + harness design |
| https://arxiv.org/html/2605.14271v1 | Auditing Agent Harness Safety |
| https://picrew.github.io/LLM-Harness/main.pdf | Agent Harness Engineering survey (PDF; project rule: PDF summaries fabricate quotes) |
| https://opentelemetry.io/docs/specs/otel/metrics/sdk/ | normative overflow spec |
| https://github.com/open-telemetry/opentelemetry-specification/issues/3904 | stabilising the overflow attribute |
| https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12181972 | patent: transaction/workflow-level retry budgets |
| https://dzone.com/articles/retry-budget-pattern | retry-budget pattern, practitioner tier |
| https://increase.com/documentation/idempotency-keys | `idempotency_key_already_used_error` / 409 |
| https://httptoolkit.com/blog/idempotency-keys/ | IETF Idempotency-Key draft |
| https://atlan.com/know/agent-harness-failures-anti-patterns/ | harness anti-patterns |
| https://oneuptime.com/blog/post/2026-02-09-pod-failure-policy-retriable-errors/view | retriable vs non-retriable |
| https://www.theregister.com/ai-and-ml/2026/08/14/deepseeks-innovative-harness-treats-everything-as-a-plug-in/5288095 | deepseek-harness coverage |

### Recency scan (2024-2026) -- performed

Searched the 2025 and 2026 windows explicitly (rows 4-6 of the query table). Result:
**four new findings that supersede or sharpen the canonical prior art.**
(a) The 2024 Kubernetes 1.31 GA of pod failure policy is now the reference implementation
of "do not count an infrastructure failure toward the retry ceiling" -- newer and more
directly applicable than the SRE-book retry budget (which bounds LOAD, not per-item work).
(b) DeepSWE (2026, #6) and #7 (2026) are both current and **directly contradict each
other** on whether infra-terminated attempts are excluded or scored as failures -- see
"Consensus vs debate".
(c) The 2026 OTel cardinality-limits post is the freshest statement of the unknown-key
containment pattern.
(d) HarnessFix (2026, #8) confirms that current agent-harness research still does NOT
separate harness faults from task failures in its headline metric -- so pyfinagent's
requirement here is ahead of the published practice, not behind it.
No source in the window supersedes Google SRE's retry-amplification rule.

## Consensus vs debate

**Consensus (3 of 3 independent domains).** A terminal outcome must be a CLOSED,
enumerated vocabulary with a separate machine REASON, not a boolean: Kubernetes
(`Complete/Failed/FailureTarget/SuccessCriteriaMet/Suspended` + reason
`BackoffLimitExceeded|DeadlineExceeded|PodFailurePolicy|...`), Temporal (six closed
statuses), and the local run record (`completed|failed|killed` + `error`). All three keep
"why it ended" separate from "how it ended".

**Consensus.** An infrastructure-caused failure should be distinguishable at the record
level and treated differently from a graded failure. Kubernetes makes this a first-class
field (`DisruptionTarget`); DeepSWE makes it an exclusion class.

**Live debate -- and it is directly on 90.1's question.** Given an attempt that died of
infrastructure, DeepSWE §5.6 **excludes** it from numerator and denominator; #7 §A.5
**scores it 0** and says so explicitly. Neither is wrong: they optimise different things
(capability estimate vs end-to-end reliability estimate). The pyfinagent-relevant
resolution is that these are two different questions and 90.1 only needs the RECORD to
support both -- which is precisely why the field must be an outcome value, not a policy
decision baked into the counter. Note also that 90.1 is a *cost* budget, where Kubernetes'
"a disruption would not reoccur on retry" logic does NOT transfer cleanly: a rail drop here
costs a measured 191,796 tokens whether or not it was the agent's fault.

## Pitfalls (from the literature + the local record)

1. **Do not silently drop an unrecognised key.** OTel's answer is an explicit overflow
   bucket with a visible marker, not rejection and not unbounded growth (#5). Applied here:
   an unrecognised `step_id` should land in a visible, non-budget-bearing bucket, not
   silently mint a fresh 5-attempt allowance (I-5) and not hard-fail the launch (the gate
   is fail-open by design, `attempt_gate.py:35-42`).
2. **A key's existence is not a key's validity.** Stripe/Increase both 409 when a known key
   arrives with different parameters; validating shape only is the known hole.
3. **Two budgets, one counter** is the SRE amplification failure in miniature (#3): the
   local ledger already merges Q/A attempts with research rounds (I-7, 9 steps affected).
4. **A "read the outcome from the run record" design must read `status`, not `outcome`** --
   the latter does not exist on any of the 617 records (I-1).
5. **State the population before quoting a token figure** (I-8): 13 vs 18 steps over the
   same 1.2M ceiling.
6. **Don't collapse zero-cost caller errors into expensive rail drops** (I-9).

### Read in full -- source #10 (added after the tables above)

| # | URL | Accessed | Kind | Key finding |
|---|---|---|---|---|
| 10 | https://docs.stripe.com/api/idempotent_requests | 2026-08-20 | official doc | The unknown-key reset, stated by a system that lives with it. `"We generate a new request if a key is reused after the original is pruned."` -- i.e. key EXPIRY silently converts a repeat into a fresh unit of work. And existence is not validity: `"The idempotency layer compares incoming parameters to those of the original request and errors if they're not the same to prevent accidental misuse."` Also `"We save results only after the execution of an endpoint begins. If incoming parameters fail validation... we don't save the idempotent result"` -- a validation failure is deliberately NOT recorded as an attempt. |

## Key findings

1. **The discriminator belongs on the record, not in the policy.** Kubernetes puts it on
   the Pod (`DisruptionTarget`) and lets policy choose `Ignore` vs `Count`:
   *"Ignore: Do not count the failure towards the backoffLimit"* / *"Count: ... This is the
   default behavior"* (k8s 1.31 GA blog, 2024-08-19,
   https://kubernetes.io/blog/2024/08/19/kubernetes-1-31-pod-failure-policy-for-jobs-goes-ga/).
   90.1 is the record half; 90.3's rail-drop exemption is the policy half. Splitting them
   this way is the mainstream design, and it is why 90.1 can be *"pure accounting"*.
2. **Terminal outcome = closed enum + separate machine reason.** Three independent systems
   agree (k8s conditions+reasons; Temporal's six closed statuses,
   https://docs.temporal.io/workflow-execution; the local `status`+`error` pair). A single
   `complete|blocked|budget-limited` triple is *narrower* than all three: it has no slot
   for "graded FAIL" vs "graded PASS", which the step's own criterion needs
   (`PASS|CONDITIONAL|FAIL|NO_VERDICT|UNKNOWN`). Treat the deepseek triple as a *reason*
   vocabulary layered under an *outcome* field, not as the outcome field itself.
3. **The exclude-vs-score-zero question is genuinely contested in 2026 literature** --
   DeepSWE §5.6 excludes (https://arxiv.org/html/2607.07946v1), arXiv:2607.12227v1 §A.5
   scores r=0. Recording the outcome keeps both computable; deciding it in the counter does
   not.
4. **Unknown keys should overflow visibly, never mint a fresh allowance.** OTel folds
   over-limit attribute sets into one marked bucket (`otel.metric.overflow=true`) rather
   than growing unboundedly (https://opentelemetry.io/blog/2026/cardinality-limits-in-opentelemetry/);
   Stripe validates parameters against the original rather than trusting key existence
   (https://docs.stripe.com/api/idempotent_requests).
5. **Bound the WORK, and bound the RATE, separately.** Google SRE runs both a per-request
   attempt cap (3) and a cumulative client-side ratio budget (10%)
   (https://sre.google/sre-book/handling-overload/). pyfinagent's attempts-ceiling is the
   first; the (inert) token ceiling is the second. They are not redundant.
6. **Published agent-harness work still does not separate harness faults from task
   failures** in its headline metric (HarnessFix, https://arxiv.org/html/2606.06324v2), so
   there is no off-the-shelf vocabulary to copy from that literature -- the borrowings have
   to come from workflow/batch systems.

## Application to pyfinagent (external finding -> internal anchor)

| External precedent | pyfinagent anchor | Implication for 90.1 |
|---|---|---|
| `DisruptionTarget` + `Ignore`/`Count` | `attempt_gate.py:267-274` writes the attempt row with `note: "outcome unknown at this seam"` | The outcome is unknown at **PreToolUse** by construction. The record must be *completed later* from the run record (a backfill/PostToolUse pass), which is exactly what criterion 1 asks for. Do not try to resolve it at launch. |
| k8s `Failed` + machine reason | `attempt_gate.py:212` writes a FIXED path `escalation_attempt_budget_<sid>.md` | Criterion 2's `escalation_<reason>_<sid>.md` matches the k8s reason pattern; today a non-exhaustion denial writes a false "BUDGET EXHAUSTED" body (`attempt_gate.py:213-214` fallback) at the same path, overwriting a real one. |
| Temporal 6-value closed status | `attempt_budget.py:67-78` `Outcome` = PASS/CONDITIONAL/FAIL/NO_VERDICT | `NO_VERDICT` currently conflates the 46 expensive rail drops with the 2 zero-cost `args-unparseable` failures and the 5 `killed` aborts (I-9). `UNKNOWN` (criterion 1) is a *fourth* thing again: no run record found. Keep all four distinct. |
| run-record `status`/`totalTokens` | `~/.claude/projects/.../workflows/wf_*.json` -- `outcome` absent on 617/617, `totalTokens` present on 617/617 (I-1, I-2) | Resolve outcome from `status` + `error` + the `result` payload's `verdict` key; take tokens from `totalTokens`. A design that reads `outcome` reads `None` every time. |
| SRE dual budget | `attempt_budget.py:64,128-130` vs `attempt_gate.py:190-193` | The token ceiling is inert because `record()` is called without `tokens=` (I-3). Criterion 1's `total_tokens` field is what makes `DEFAULT_MAX_TOKENS` live for the first time. |
| Stripe key validation | `attempt_gate.py:97` `_STEP_ID_RE` | Shape-only validation; `999.2` already holds 5 rows + an escalation file though it is absent from all 1457 masterplan steps (I-5). Membership check against `.claude/masterplan.json` is the fix, with an OTel-style visible bucket rather than a hard refusal (the hook is fail-open by design, `attempt_gate.py:35-42`). |
| SRE "only one layer retries" | `research_router.py:30-35` vs `attempt_gate.py:182-184` | Two budgets share one counter; 9 of 27 step-ids have research-gate launches inside their 5 Q/A attempts (I-7). Any outcome field should also carry enough to separate them, or the ceiling means different things for different steps. |

**Two things to watch when the contract is written** (research-gate observations, not plan):
- The `audit_basis` token figures (18 steps / 2,677,199) reproduce on the 540-run
  ALL-workflows population, not the 441 qa-verdict population it names (I-8). Criterion
  wording that says "1.2M" should say over what.
- `tool-ralph` is uncorroborated in-repo and was not confirmed by the vendor page (I-10,
  source #9). The vocabulary is defensible on Temporal/k8s precedent alone; it should not
  be justified by a citation that cannot be produced.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **10**
- [x] 10+ unique URLs total -- **25** (10 read in full + 3 attempted-but-unreadable + 12 snippet-only)
- [x] Recency scan (2024-2026) performed + reported -- 4 findings, plus a live contradiction
- [x] Full pages read, not abstracts (source #1's fetch truncated; disclosed, and the
      missing mechanism was taken from #2 instead)
- [x] file:line anchors for every internal claim (I-1..I-10)

Soft checks:
- [x] Internal exploration covered every module in the stated scope
- [x] Contradictions noted (#6 vs #7; audit_basis population; uncorroborated `tool-ralph`)
- [x] Claims cited per-claim
- Gap: the deepseek-harness source repo could not be fetched (404), so the cited mechanism
  source remains unverified from the primary artifact.

---

## Envelope (FINAL)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 15,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {"audit_class": false, "rounds": 3, "dry_rounds": 0, "K_required": 2,
               "new_findings_last_round": 2, "dry": false},
  "brief_path": "handoff/current/research_brief_90.1.md",
  "gate_passed": true
}
```
