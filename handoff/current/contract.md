# Contract -- step 90.1

**Step:** 90.1 -- "an attempt row cannot tell a graded attempt from a rail drop, and the
token half of the budget has never been able to fire"
**Phase:** phase-90 -- Terminate the Q/A re-cycle loop by ROUTING findings, not re-grading them
**Priority:** P0. **Filed:** 2026-08-20. **Contract written:** 2026-08-20.
**Order:** LANDS FIRST -- 90.3's rail-drop exemption cannot be expressed without the
`outcome` field this step adds.

---

## 1. Research gate -- PASSED (enforced, not self-reported)

Launched `Workflow({scriptPath: '.claude/workflows/research-gate.js'})` (scriptPath, never
name). Run `wf_db313c3d-b75`, 2 agents, 200,572 tokens, 641s.
Brief: `handoff/current/research_brief_90.1.md` (28,744 chars).

Enforced return: `gate_passed: true`, `agent_self_reported_gate_passed: true`,
`self_report_disagreed: false`, `violations: []`. Checks: sources 10 >= 5, URLs 25 >= 10,
recency scan performed, brief on disk non-empty and independently read,
`brief_status: COMPLETE`, **all 10 claimed read-in-full URLs present in the brief**,
`urls_collected` corroborated (25 <= 26 distinct URLs found). Not audit-class, so
`coverage.dry` is informational only.

Independently re-checked by Main before writing this contract: all 10 claimed URLs are
literally present in the brief on disk (10/10).

### The three brief findings that CHANGED this plan

**(a) A run record has NO `outcome` key -- 617/617.** The terminal field is `status`
(`completed` 564, `failed` 48, `killed` 5). Criterion 1 says "resolved from the
corresponding Workflow run record"; a resolver that reads `outcome` reads `None` every
time. **Resolution therefore reads `status` + `error` + `result.verdict`.**

**(b) `NO_VERDICT` today conflates FOUR different things.** Measured: 46 StructuredOutput
rail drops (`status: failed`, mean **191,796** tokens -- nearly as expensive as a
completed run's 242,997, 8.8M tokens burned in total), 2 `args-unparseable` failures
(**zero** tokens, a caller bug not a rail drop), 5 `killed` aborts (105,889 mean), and --
new with this step -- `UNKNOWN`, meaning no run record was found at all. Criterion 1 fixes
the `outcome` vocabulary at five values, so the finer distinction goes in a **separate**
`outcome_reason` field rather than being crushed into the five. Precedent: Kubernetes
carries a machine `reason` beside a closed terminal condition
(`BackoffLimitExceeded` / `DeadlineExceeded` / `PodFailurePolicy`).

**(c) POPULATION TRAP in this step's own audit_basis.** It names "441 qa-verdict run
records" and then quotes "18 steps exceeded 1.2M (max 2,677,199 on 86.85)". Both numbers
are real but **for different populations**: 18 / 2,677,199 reproduces only on the 540-run
all-workflows superset; restricted to the 441 qa-verdict runs it is **13 / 2,506,619**.
Criterion 3 sums "the step's rows", and the ledger holds BOTH rails, so the enforced
denominator here is the **all-workflows** one. This contract states that explicitly
because it moves the bound's bite by 5 steps.

### One citation this step must NOT lean on

The audit_basis names deepseek-harness **`tool-ralph`** as MECHANISM SOURCE for the closed
vocabulary. The researcher fetched the vendor page (`https://deepseek.com/harness/en/`) in
full and found **no** Ralph-loop outcome vocabulary, no budget/exhaustion semantics and no
per-attempt token accounting; a repo-wide search returns exactly one hit -- the
audit_basis string itself. **The design is justified on corroborated precedent instead**:
Kubernetes pod failure policy (`Ignore` vs `Count` + the `DisruptionTarget` condition,
1.31 GA), Temporal's six closed workflow statuses, Google SRE's two-budgets rule, and
OpenTelemetry's overflow-bucket answer to unbounded keys. `tool-ralph` is cited nowhere in
the shipped code or docs.

---

## 2. Hypothesis

An attempt row records only *that* a launch happened. Because it carries neither what the
launch produced nor what it cost, three separate defects follow mechanically, and all
three are accounting defects rather than policy defects:

1. the gate spends budget identically on a graded attempt and on a rail drop;
2. `DEFAULT_MAX_TOKENS` cannot bind, because `attempt_gate.py` calls `state.record(outcome)`
   with no `tokens=` and `Attempt.tokens` defaults to 0 -- so `tokens_used` is **always 0**;
3. a step id that no plan of record contains mints itself a private 5-attempt allowance.

Adding a resolved `outcome` + `total_tokens` to every attempt row, giving each denial its
own reason-named artifact, and requiring a step id to resolve against the masterplan makes
all three expressible. **No verdict changes. No new gate concept is introduced.** The
token ceiling and the id check are existing intentions that were inert; this step makes
them execute.

---

## 3. Immutable success criteria (copied VERBATIM from .claude/masterplan.json)

1. every attempt row gains an `outcome` field resolved from the corresponding Workflow run record (PASS|CONDITIONAL|FAIL|NO_VERDICT|UNKNOWN) and an integer `total_tokens`; a re-runnable backfill reconstructs both for all 92 existing rows and prints the per-value counts, with UNKNOWN used only where no run record exists and that count stated

2. a denial writes handoff/current/escalation_<reason>_<sid>.md where <reason> is the machine reason for that denial, and a test asserts that triggering a non-exhaustion denial leaves any pre-existing escalation_attempt_budget_<sid>.md BYTE-IDENTICAL, proven by sha256 before and after

3. the token ceiling either fires or is deleted: EITHER DEFAULT_MAX_TOKENS is enforced against the summed total_tokens of the step's rows and a fixture at 1,200,001 tokens is DENIED by execution, OR the constant and the docstring claim that it binds are both removed -- decided and shown by running it, never by reading the source

4. extract_step_id rejects any id absent from .claude/masterplan.json as a loud DENY rather than the current 'not a step attempt; allowed and not counted'; cells prove '86.118.1', '86.1180' and '999.99' are each DENIED while '86.118' is ADMITTED, and the self-test's own synthetic ids are updated or explicitly exempted rather than silently passing

5. mutation-tested with the control observed GREEN first: a mutant that records a NO_VERDICT attempt as a graded outcome must be KILLED, a mutant that turns the unresolvable-step-id DENY into exit 0 must be KILLED, and a mutant that fails to run scores ERROR and never counts as a kill

6. verdict semantics unchanged and shown: no code path in the changed files can write a verdict value, asserted by sha256 of handoff/verdict_ledger.jsonl taken before and after the whole cell run

**Immutable verification command** (NOT amended):

```
python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify
```

**live_check:** `live_check_90.1.md`: the verbatim backfill output over the real 92 rows
with per-outcome counts, the sha256 pair proving a pre-existing exhaustion escalation was
untouched by a non-exhaustion denial, and the four step-id cells run against the real module.

---

## 4. Plan

### 4.1 The resolver (criterion 1)

New `scripts/harness/attempt_outcomes.py`:

- Reads run records from `~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/*/workflows/*.json`
  (overridable via `ATTEMPT_GATE_RUN_RECORDS` for tests).
- **Join key: `startTime`, never `timestamp`.** `timestamp` is written at COMPLETION and a
  run can last 15+ minutes, so it is useless as a launch key. Measured over the real 89
  attempt rows: joining on `(args.step_id, |startTime - row.ts| <= tol)` resolves
  **83 of 89 uniquely with 0 ambiguous** at every tolerance from 30s to 300s; nearest-match
  |delta| is min 0.021s / p50 0.464s / **max 1.007s**. The same join on `timestamp`
  resolves **9 of 89** -- the field is the bug, not the key. Default tolerance **30s**
  (30x headroom over the observed max); ambiguity first appears at 900s and any ambiguous
  or absent match resolves to `UNKNOWN`, never a guess.
- `outcome` in the closed five: `result.verdict` when the return carries one, else
  `NO_VERDICT` when a record exists without one, else `UNKNOWN` when no record exists.
- `outcome_reason` carries the finer class from (b): `structured_output_drop`,
  `args_unparseable`, `killed`, `no_run_record`, `ambiguous_match`, or `graded`.
- `total_tokens` from `totalTokens`; `run_id` from `runId` so the row finally shares a key
  with `handoff/verdict_ledger.jsonl` (which stores `run_id`) -- closing the "two streams
  share no join key" gap the brief records as I-4.

### 4.2 Persisting it (criterion 1: rows GAIN the fields)

`attempt_outcomes.py --backfill` enriches rows **in place**, and the enrichment is
constrained to be **additive-only**, proven rather than asserted: the projection of every
post-write row onto its pre-write key set must be byte-identical to the pre-write row, and
row count and order must be unchanged. A `.bak` is written first, and the ledger is
git-tracked so the diff is itself the audit. `--backfill --dry-run` prints the per-value
counts without writing.

New attempt rows are written by the hook with `outcome: null`, `total_tokens: null` and a
note saying the outcome is unknown at the PreToolUse seam -- the brief's k8s anchor is
exactly this: the record is completed later, not at launch.

### 4.3 The token ceiling FIRES (criterion 3)

`build_state()` passes `tokens=` into `state.record(...)` from the row's resolved
`total_tokens`, so `tokens_used` stops being a constant 0. For rows not yet persisted-resolved,
the gate resolves the step's OWN prior rows lazily in memory at decision time (measured
cost to scan+parse all 617 records: **0.244s**, against a 45s hook timeout). Any failure in
that path falls back to unresolved and says so on stderr -- the fail-open direction, which
can only ever allow more.

**Measured consequence on the live ledger, stated before shipping:** five steps have
cumulative tokens at or above 1.2M (75.11.4, 86.108, 86.59, 86.78, 86.116). Every one of
them is **already** denied by the 5-attempt ceiling or already `CLOSED_PASS`, so switching
the token ceiling on **denies nothing that is not already denied today**. It is a real
bound going forward, not a retroactive one.

### 4.4 Reason-named escalations (criterion 2)

`write_escalation(state, reason)` writes `escalation_<reason>_<sid>.md`. Exhaustion keeps
`reason="attempt_budget"`, so today's four files keep their exact paths and nothing is
orphaned. New reasons get new paths: `unknown_step_id`, `token_budget`. The
`# BUDGET EXHAUSTED` fallback body is **removed** -- a non-exhaustion denial writes its own
body and can no longer forge an exhaustion record.

Fixture for the sha256 test is a REAL pre-existing file:
`handoff/current/escalation_attempt_budget_86.85.md` (sha256
`1d8a53e5...1779ab`), which is hand-authored by the operator and sits at exactly the path
the fallback would clobber.

### 4.5 Step ids must resolve (criterion 4)

Rule, stated as a split so the escape hatch survives:

- `args.step_id` **absent entirely** -> not a step attempt -> allowed, not counted.
  UNCHANGED. This is the documented hatch for self-audit and ad-hoc workflows and covers
  **81 of 617** historical launches.
- `args.step_id` **present but not resolvable** against `.claude/masterplan.json` ->
  **loud DENY** (exit 2) + `escalation_unknown_step_id_<sid>.md`.

Membership accepts both the raw id and the `phase-` prefixed form. Duplicate-id resolution
is explicitly **out of scope** -- that is step 90.7.

**Blast radius, measured over all 617 historical launches before choosing this:** 531
resolve, 81 carry no `step_id`, and only **5 do not resolve** -- `82.3+82.4`,
`PLAYWRIGHT-SUBAGENT-PROBE`, `86.90-PROBE`, and two `86.28-LIVETEST-*` -- every one a
one-off probe, and every one already refused today by the shape regex (so none was ever
counted). **Zero production Q/A evaluations are affected.**

The researcher proposed an OTel-style *visible overflow bucket* instead of a hard refusal,
noting the hook is fail-open by design. **Criterion 4 is immutable and demands a loud
DENY, so DENY is what ships**; the measurement above is why that costs nothing. The
alternative is recorded here, not silently dropped.

**The self-test's ids (criterion 4's last clause).** `9.1`..`9.5` are NOT synthetic -- all
five are REAL masterplan steps, so membership validation would let them pass **silently**,
which is exactly what the criterion forbids, and a leaked self-test row has already raised
a real step's allowance once (`read_ledger`'s own docstring records the `9.4` incident).
Fix: add an `ATTEMPT_GATE_MASTERPLAN` override (testing only, same shape as the existing
`ATTEMPT_GATE_LEDGER`) and point the self-test at a synthetic plan holding its own ids, so
test ids are exempt **by construction** and can never again touch a real step's allowance.
Note the masterplan's own notes field asserts `9.1` "is not a masterplan id" -- that is
false on today's plan, and the correction is recorded in `experiment_results_90.1.md`.

### 4.6 Mutation matrix (criterion 5)

`scripts/qa/mutation_matrix_90_1.py --verify`, following the house pattern of
`mutation_matrix_86_71.py`: CONTROL observed GREEN before any cell; mutants run from a temp
copy via subprocess so the CALL SITE is what is tested; a mutant that fails to run scores
**ERROR**, never a kill; the real tree is proven untouched by md5 before/after. Required
cells include the two the criterion names by name -- a mutant recording a `NO_VERDICT`
attempt as graded, and a mutant turning the unresolvable-id DENY into exit 0 -- plus cells
for the additive-only backfill invariant, the escalation-reason path, and the token ceiling.

### 4.7 Verdict semantics untouched (criterion 6)

`handoff/verdict_ledger.jsonl` sha256 is captured before and after the whole cell run and
must match. Baseline at contract time: `fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2`.
Nothing in the changed files may write a verdict value.

---

## 5. Out of scope (queued, not silently dropped)

- **90.5** -- the verdict ledger's own accuracy. Found while validating the join: of 120
  `run_id`s shared between `handoff/verdict_ledger.jsonl` and the run records, **7
  disagree** -- 2 rows record `NO_VERDICT` where the run record carries a real verdict
  (86.84 FAIL, 86.85 CONDITIONAL), and 5 record `FAIL` where the rail returned
  `CONDITIONAL` (the documented 3rd-CONDITIONAL conversion). This step's `outcome` is the
  **rail's raw return**, deliberately NOT the doctrine-adjusted ledger verdict; the two
  are not reconciled here.
- **90.6** -- research-gate launches sharing the Q/A ceiling. Confirmed live: this step's
  own research gate consumed 90.1 attempt 1 of 5. The row already carries `workflow`, so
  the discriminator exists; acting on it is 90.6's.
- **90.7** -- duplicate step ids under `phase-` normalization.
- **87.11** -- the four non-functional metrics, and the remaining stale-doc sweep.

## 6. References

- `handoff/current/research_brief_90.1.md` (gate PASSED, enforced) -- findings I-1..I-10.
- Kubernetes pod failure policy, 1.31 GA: `Ignore` vs `Count` + the `DisruptionTarget`
  condition -- https://kubernetes.io/blog/2024/08/19/kubernetes-1-31-pod-failure-policy-for-jobs-goes-ga/
- Temporal's six closed workflow statuses -- https://docs.temporal.io/workflow-execution
- Google SRE, two budgets (per-request cap AND a cumulative ratio) -- https://sre.google/sre-book/handling-overload/
- OpenTelemetry cardinality limits / overflow bucket -- https://opentelemetry.io/blog/2026/cardinality-limits-in-opentelemetry/
- Live contradiction, recorded not resolved: DeepSWE excludes infra-terminated rollouts
  (https://arxiv.org/html/2607.07946v1 §5.6) while https://arxiv.org/html/2607.12227v1 §A.5
  scores them r=0. Recording the outcome keeps BOTH computable; deciding it inside the
  counter does not. This step records and does not decide.
