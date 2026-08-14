# Research Brief — step 86.79

**Topic:** Counter correctness in autonomous-agent retry/escalation harnesses —
off-by-one / inclusive-vs-exclusive in a self-counting judge; implicit temporal
coupling (correctness enforced in a different file); retention/pruning windows
that SATURATE a derived count and silently disable a threshold; fail-closed
design for an uncomputable counter; cross-process attempt-budget persistence in
agent/workflow frameworks; LLM-as-judge self-attempt-number derivation.

**Tier:** moderate (caller-specified). **Audit-class:** NO (`coverage` reported
for information; `coverage.dry` not required).
**Researcher:** Layer-3 Workflow rail. **Date accessed (all sources):** 2026-08-14.

---

## ENVELOPE (born inert — flipped to COMPLETE as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 15,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.79.md",
  "gate_passed": true
}
```

---

## Search queries run (three-variant discipline)

| # | Variant | Query |
|---|---|---|
| 1 | year-less canonical | `counter saturation truncated log retention disables alerting threshold escalation` |
| 2 | year-less canonical | `Airflow try_number off-by-one retry attempt counter semantics` |
| 3 | year-less canonical | `temporal coupling implicit ordering dependency fail-closed API design` |
| 4 | current-year frontier (2026) | `LLM-as-judge self-evaluation bias iteration count 2026 agent evaluator` |
| 5 | last-2-year window (2025-2026) | `agent retry budget exhaustion escalate human 2025 2026 autonomous agent loop termination` |
| 6 | last-2-year window (2025-2026) | `monotonic sequence number high-water mark detect lost dropped records ring buffer 2025 2026` |
| 7 | year-less canonical | `"retention policy" OR "keep last N" silently caps count metric undercounts incidents postmortem` |

---

## Read in full (WebFetch; counts toward the gate) — 10

| # | URL | Kind | Tier | Key finding (quoted) |
|---|-----|------|------|----------------------|
| 1 | https://arxiv.org/html/2410.21819v2 | paper (arXiv HTML) | 1 | Self-preference bias as an Equal-Opportunity gap; GPT-4 = **0.520**. Mechanism is perplexity/familiarity: the preference held "regardless of whether the outputs were self-generated". |
| 2 | https://docs.temporal.io/encyclopedia/retry-policies | official docs | 2 | "Setting the value to 0 also means unlimited. **Setting the value to 1 means a single execution attempt and no retries.**" At exhaustion: "If this limit is exceeded, the execution fails without retrying again." Pending attempt count is NOT in Event History — "Use the Describe API to get a pending Activity Execution's attempt count." |
| 3 | https://docs.aws.amazon.com/step-functions/latest/dg/concepts-error-handling.html | official docs | 2 | "`MaxAttempts` ... represents the **maximum number of retry attempts** (`3` by default) ... A value of `0` specifies that the error is never retried." On redrive: "the retry attempt count for these states is **reset to 0**". Example 4: with `MaxAttempts: 2` the 3rd occurrence "already reached its maximum of two retries ... that retrier fails". |
| 4 | https://sre.google/sre-book/handling-overload/ | industry canonical | 4/2 | "a *per-request retry budget* of up to three attempts"; "A request will only be retried as long as this ratio is below **10%**"; and the mechanism: "**clients include a counter of how many times the request has already been tried in the request metadata.** For instance, the counter starts at 0 in the first attempt and is incremented on every retry until it reaches 2." |
| 5 | https://prometheus.io/docs/concepts/metric_types/ | official docs | 2 | Counter = "a cumulative metric that represents a single monotonically increasing counter whose value can only increase or be reset to zero on restart." Gauge = "can arbitrarily go up and down." **"Do not use a counter to expose a value that can decrease."** |
| 6 | https://docs.kernel.org/userspace-api/perf_ring_buffer.html | official docs | 2 | "When the consumer doesn't keep up with the producer, it would lose some data, **the kernel keeps how many records it lost and generates the `PERF_RECORD_LOST` records**". The retained window and the loss count are separate records. |
| 7 | https://openai.github.io/openai-agents-python/running_agents/ | official docs | 2 | "If we exceed the `max_turns` passed, we raise a `MaxTurnsExceeded` exception." "**The turn count does not persist across separate runs.**" |
| 8 | https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html | official docs | 2 | Retry policy receives "the exception, try number, max tries"; a policy "can fail earlier but cannot extend past the configured maximum". `up_for_retry` = "failed, but has retry attempts left". Doc does **not** state increment/reset mechanics. |
| 9 | https://github.com/apache/airflow/issues/38304 | upstream issue | 5 | Attempt≠outcome, upstream: "**if the same task is killed for being stuck in queued, the task never started**, so the lack of idempotency does not matter and the task should definitely be re-attempted." Infrastructure-killed attempts consume the same counter as real failures. |
| 10 | https://blog.ploeh.dk/2011/05/24/DesignSmellTemporalCoupling/ | named-researcher blog | 3 | Temporal Coupling = "an implicit relationship between two, or more, members of a class requiring clients to invoke one member before the other." It is a smell because the type "doesn't properly protect its invariants ... encapsulation is broken" and misuse "cannot be caught during compilation—only at runtime". Remedy = make the invalid state unrepresentable. |

## Identified but snippet-only (does NOT count toward the gate) — 15

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://langchain-ai.github.io/langgraph/troubleshooting/errors/GRAPH_RECURSION_LIMIT/ | official docs | **Fetch ATTEMPTED and failed** — returned only "Redirecting..." with no content. Recorded as an attempt, not a read. |
| https://github.com/apache/airflow/pull/1230 | upstream PR | Corroborates the classic off-by-one ("Attempt 2 out of 1"); superseded by #38304 which is the sharper attempt/outcome case |
| https://github.com/apache/airflow/issues/47971 | upstream issue | Backoff float overflow, adjacent not central |
| https://arxiv.org/pdf/2604.23178 | paper | Judge-bias mitigation — sibling step 86.78's scope, deliberately not duplicated |
| https://deepeval.com/blog/llm-as-a-judge | vendor blog | 2026 practice summary; tier-3/4, superseded by source 1 |
| https://github.com/CSHaitao/Awesome-LLMs-as-Judges | survey index | Index, not a source |
| https://www.pluralsight.com/tech-blog/forms-of-temporal-coupling/ | industry blog | Taxonomy of temporal-coupling forms; source 10 is the canonical statement |
| https://hackernoon.com/api-design-temporal-coupling-2c1687173c7c | community blog | Lower tier, same content as 10 |
| https://www.javacodegeeks.com/2026/03/temporal-couplingthe-hidden-dependency-that-breaks-systems.html | community blog | 2026 restatement; recency-scan evidence only |
| https://matrixtrak.com/blog/agents-loop-forever-how-to-stop | community blog | 2026 agent stop-rules; recency-scan evidence |
| https://www.prismocode.io/ai-agent-retry-policies/ | community blog | 2026 agent retry practice; recency-scan evidence |
| https://dpdk.readthedocs.io/en/v16.07/prog_guide/ring_lib.html | official docs | High-water-mark precedent; source 6 covers the loss-count remedy better |
| https://oneuptime.com/blog/post/2026-02-06-data-retention-policies-opentelemetry/view | vendor blog | Retention/observability 2026; no counter-saturation content |
| https://owasp.org/Top10/2025/A09_2025-Security_Logging_and_Alerting_Failures/ | standards body | Alerting-failure class, too general |
| https://docs.datadoghq.com/monitors/guide/troubleshooting-monitor-alerts/ | official docs | Alert-threshold troubleshooting, not counter truncation |

---

## Recency scan (last 2 years, 2024-2026)

Performed — queries 4, 5, 6 above, plus 7. **Result: 3 new findings that
complement (none that supersede) the canonical sources.**

1. **(2026)** The agentic-error-budget pattern has converged on rules that match
   this step's thesis: budgets "must be set in config, not chosen adaptively by
   the agent" (an agent under pressure rationalises a higher budget), and
   exhaustion must "checkpoint state, stop cleanly, flag for human" — "Looping on
   a budget-exhausted task is the failure mode, not the fix." This is 2026
   restatement of Google SRE (source 4), not a supersession.
2. **(2026)** LLM-as-judge practice has standardised on cross-family judges and
   position-swap double-scoring; the *self*-preference literature (source 1,
   2024, v2) is still the mechanistic reference and has not been overturned.
3. **(2024-2026)** No source found — in any of the 4 recency queries — that
   names "counter saturation by retention window disables an escalation
   threshold" as a *documented defect class* under that description. The nearest
   named prior art is the kernel's `PERF_RECORD_LOST` remedy (source 6) and
   Prometheus's counter-vs-gauge type rule (source 5). **Treat this as a finding:
   the hazard is well-covered by its remedies but poorly named in the
   literature**, so pyfinagent should not expect to find an off-the-shelf
   citation for it and should cite the remedies instead.

---

## Key findings — external

**E1. The same field name means opposite things in two major engines, so a
counter's name never carries its unit.** Temporal's `Maximum Attempts` counts
TOTAL executions — "Setting the value to 1 means a single execution attempt and
no retries" (source 2). Step Functions' `MaxAttempts` counts RETRIES, excluding
the initial attempt — default 3, and "A value of `0` specifies that the error is
never retried" (source 3). Two tier-2 official docs, same word, off by exactly
one. The remedy the pair implies: write the unit next to the number.

**E2. Attempt ≠ outcome is a live upstream defect class, in both directions.**
Airflow #38304 (source 9) complains that an attempt which *never started* burns
the retry counter. pyfinagent has the mirror: an attempt that *ran and cost
tokens* returns no verdict. One counter cannot serve both; the engines that get
this right keep the ATTEMPT count and the OUTCOME count separate.

**E3. Cumulative-over-a-window, and carried WITH the work item.** Google SRE
(source 4) bounds retries at 3 per request AND 10% per client, and — the part
that matters for cross-process persistence — "clients include a counter of how
many times the request has already been tried **in the request metadata**". The
count travels with the unit of work, not in the caller's memory. That is the
canonical answer to "how do you persist an attempt count across a process
boundary".

**E4. The saturation bug has a name at the type level.** Prometheus (source 5):
a counter "can only increase or be reset to zero on restart"; **"Do not use a
counter to expose a value that can decrease."** A count derived from a pruned
retained set is a GAUGE. Reading it as an attempt number is the documented type
error, stated by the canonical monitoring doc.

**E5. The remedy for a lossy retained window is to EMIT the loss, not hide it.**
Linux perf (source 6): "the kernel keeps how many records it lost and generates
the `PERF_RECORD_LOST` records". Retained data and the count of dropped data are
separate records. Generalised: persist the monotonic count separately from the
retained records; a pruner that deletes without recording how many it deleted
converts a known quantity into an unknown one and reports the unknown as a
number.

**E6. Temporal coupling's remedy is structural, not documentary.** Seemann
(source 10): the smell is that misuse "cannot be caught during compilation—only
at runtime", because the type "doesn't properly protect its invariants". You fix
it by making the invalid state unrepresentable — you do not fix it by writing
the required order in a comment or a second file.

**E7. Every surveyed framework's attempt counter is run-scoped or explicitly
reset — none gives a durable cross-session count.** OpenAI Agents SDK: "The turn
count does not persist across separate runs" (source 7). Step Functions: on
redrive "the retry attempt count for these states is reset to 0" (source 3).
Temporal: a pending activity's attempt count is not in Event History and needs a
separate Describe call (source 2). **Implication:** pyfinagent's Layer-3 loop —
which spans sessions — is precisely the case none of them covers, so its count
MUST be derived from a durable artifact. That is what makes the retention window
load-bearing rather than cosmetic.

**E8. A self-counting judge is not de-risked by being unaware it is self-
judging.** Source 1 measures GPT-4 self-preference at 0.520 EO but attributes it
to perplexity/familiarity, holding "regardless of whether the outputs were
self-generated". So "the judge does not know it is counting itself" is not a
safety argument. (Bias mitigation proper is sibling step 86.78's scope; recorded
here only as the boundary condition.)

---

## Internal code inventory

| File:line | Role | Status |
|---|---|---|
| `scripts/qa/qa_wip.py:122-124` | `DEFAULT_KEEP = 3`, comment "Current record + this many prior attempts" | **Off-by-one vs the code** (F-I1) |
| `scripts/qa/qa_wip.py:206-227` | `prune_wip_records(...)`, `records[keep:]` deleted at `:221` | **No automatic caller** (F-I2) |
| `scripts/qa/qa_wip.py:179-203` | `list_wip_records` — counts every file, incl. the current run's | Feeds the counter |
| `scripts/qa/qa_wip.py:316` | `"records_retained": len(records)` | THE attempt counter |
| `scripts/qa/qa_wip.py:161-176` | `source_present()` — fail-closed on a MISSING SINK | LIVE, scope-limited (F-I6) |
| `.claude/agents/qa.md:98-151` | Write-first: WIP file in the first few tool calls | The ordering rule (F-I4) |
| `.claude/agents/qa.md:619-623` | `records_retained` = "count of prior Q/A spawns ... the attempt number" | **The two halves differ by 1** (F-I3) |
| `.claude/agents/qa.md:706-717` | F1b 5-attempt note + the stated pruning limit | Threshold that saturation kills (F-I5/F-I6) |
| `scripts/qa/verdict_history_86_21.py:70-113` | Verdict-keyed counter, returns `None` not 0 | Correctly fail-closed (F-I7) |
| `scripts/harness/attempt_budget.py:~92-135` | Cumulative monotonic budget, `DEFAULT_MAX_ATTEMPTS=5` | **Unwired, unpersisted** (F-I8) |
| `handoff/verdict_ledger.jsonl` | 35 rows, last row dated **2026-08-11** | Stale, hand-appended (F-I7) |
| `.claude/agent-memory/qa/verdicts/` | 53 records, 22 step-ids | 5 step-ids exceed `DEFAULT_KEEP` (F-I5) |

### F-I1. Off-by-one between `DEFAULT_KEEP`'s comment and `prune`'s arithmetic — MEASURED

`qa_wip.py:122-124` documents `DEFAULT_KEEP = 3` as "Current record + this many
prior attempts" (i.e. 4 retained). `prune_wip_records` at `:221` does
`for p in records[keep:]`, retaining `keep` records TOTAL. Measured on a temp
sink seeded with 6 records: `removed 3, RETAINED 3 = current + 2 priors`. The
comment promises 4, the code delivers 3. This is E1's hazard inside the module
itself — the word "keep" does not carry whether the current record is inside or
outside the window.

### F-I2. `prune_wip_records` has NO automatic caller — enumerated

Search command used (as required by the caller):

```
grep -rn "prune_wip_records" --include='*.py' --include='*.sh' \
  --include='*.js' --include='*.mjs' --include='*.ts' . | grep -v '^\./\.git/'
```

7 hits in 4 files, **zero of them production or scheduled**:
`scripts/qa/qa_wip.py:206` (its own definition);
`scripts/qa/verify_wip_retention_86_36.py:129,135` (its checker);
`scripts/qa/mutation_matrix_86_36.py:45,47` (its mutation matrix);
`scripts/qa/mutate_counter_source_86_21.py:56,93` (comments only). This
independently confirms `handoff/archive/phase-86.36/live_check.md:200`
("`prune_wip_records()` has no production caller"). **The saturation hazard is
therefore LATENT — armed the moment anyone wires it, not firing today.** That
distinction should survive into the contract; overstating it as a live defect
would be wrong.

### F-I3. The off-by-one inside qa.md's own sentence

`qa.md:621-623`: "`records_retained` is the count of **prior** Q/A spawns on this
step — the **attempt number**, and it is authoritative." Those two descriptions
differ by exactly one. Because `qa.md:110-116` makes the Q/A write its own WIP
record in its first tool calls, and `list_wip_records` counts every file
including that one, the true semantics is `records_retained = priors + 1 = the
current attempt number`. Measured 2026-08-14: step 86.79 (no Q/A yet) → `0`;
step 86.32 (5 spawns) → `5`. So "attempt number" is the correct half and "count
of prior spawns" is the wrong half, sitting in the same sentence.

### F-I4. That correctness is enforced in a DIFFERENT file — textbook temporal coupling

`records_retained` equals the attempt number ONLY IF the current spawn already
wrote its record. The ordering is mandated in `.claude/agents/qa.md:98-151`;
the number is produced by `scripts/qa/qa_wip.py`, which cannot observe whether
the write happened. A Q/A that queried the counter before its first write reads
`N-1` — and it fails **OPEN**, because a lower attempt number *suppresses*
escalation. This is exactly Seemann's smell (E6): an implicit call-order
relationship whose violation "cannot be caught ... only at runtime". His remedy
maps directly: make it self-evident instead of documented — e.g. have `report()`
accept the current run stamp and return `attempt_number` and `prior_attempts` as
two separately-named fields, so no reader must know the ordering rule to read
the number correctly.

### F-I5. Saturation would disable F1b — and 5 of 22 step-ids are ALREADY over the window

`qa.md:706-711` keys operator escalation off the attempt number: "at 5+, say so
in `notes` and recommend operator escalation". That number is `records_retained`.
With prune wired at `keep=3` it can never exceed 3, so **the 5-attempt
escalation becomes unreachable** — the precise "counter saturation disables
escalation" class, and the precise type error Prometheus names in E4.

Live measurement, 2026-08-14 (`python scripts/qa/qa_wip.py <sid>`):

| step | `records_retained` | `source_present` |
|---|---|---|
| 86.32 | **5** | true |
| 86.9 | 4 | true |
| 86.62 | 4 | true |
| 86.44 | 4 | true |
| 86.38 | 4 | true |

Five step-ids whose true attempt count already exceeds `DEFAULT_KEEP`, and
**86.32 sits exactly ON the F1b threshold of 5** — it is the one live case that
would flip from "escalate" to "attempt 3, carry on". Temp-sink proof: 6 true
attempts → `records_retained: 3`.

### F-I6. `source_present` is fail-closed but explicitly does NOT cover pruning

`qa_wip.py:161-176` makes a MISSING SINK legible, and `qa.md:713-717` already
states the residual limit verbatim: "loss of records **inside** an existing sink
is still not self-detectable, because `prune_wip_records` deletes old records by
design." So the existing guard cannot be extended to cover saturation — the
remedy has to be a separately-persisted monotonic count (E3/E5), not a wider
`source_present`.

### F-I7. The verdict-keyed counter is the fail-closed model to copy — but its source is stale

`verdict_history_86_21.py:98-113` returns `None`, never 0, for `UNPARSEABLE` /
`LEDGER_EMPTY` / `LEDGER_MISSING`, and `would_auto_fail` propagates the `None`.
That is precisely objective (d): a counter that cannot be computed refuses to
report a number. Its input, however, is stale: `handoff/verdict_ledger.jsonl` is
**35 rows, last row dated 2026-08-11** (3 days before this brief), it is
hand-appended, and `qa.md:648` records the measured divergence (86.62:
`qa_wip`=4 vs ledger `no_rows_for_step`). Note the ledger already models
`NO_VERDICT` as a first-class row — its last row is one — which is the E2
attempt/outcome split done correctly.

### F-I8. `attempt_budget.py` is monotonic-by-construction but has no durable state

`BudgetState` is documented "Cumulative, monotonic. NOTHING in this class ever
decrements a counter", carries `DEFAULT_MAX_ATTEMPTS = 5`, and makes
`NO_VERDICT` a first-class `Outcome` — the E2 split, correctly designed. But per
CLAUDE.md F1b it has no runtime caller and no persistence. **So the only durable
attempt count in the live system is the one that can saturate**, and the
non-saturating design is the one that is not running.

---

## Consensus vs debate

- **Consensus (strong).** Bound work cumulatively over a window; never reset on
  success (E3). Separate ATTEMPT from OUTCOME (E2, F-I8). Escalate to a human at
  exhaustion, never auto-pass (E3 + 2026 sources). Make ordering requirements
  structural, not documentary (E6).
- **Debate / genuine divergence.** Inclusive-vs-exclusive attempt counting is
  *not* settled: Temporal and Step Functions ship opposite conventions under the
  same field name (E1). There is no "right" convention to adopt — only a
  requirement to state the unit at the point of use.
- **Contradiction with pyfinagent's current shape.** Every framework surveyed
  scopes its counter to one run or resets it explicitly (E7); pyfinagent needs a
  cross-session count. The frameworks therefore offer no drop-in answer, only the
  metadata-carrying pattern from E3 and the loss-record pattern from E5.

## Pitfalls (from the literature, mapped)

1. **A count derived from a bounded retained set is a gauge, not a counter** — do
   not compare it to a threshold above the window size (E4 → F-I5).
2. **A pruner that deletes without recording the deletion destroys the
   denominator silently** (E5 → F-I1/F-I6).
3. **A field whose correctness depends on call order fails open at runtime with
   no signal** (E6 → F-I4) — and here the open direction is the unsafe one.
4. **A name is not a unit**; `keep`, `attempts`, `MaxAttempts` all shipped
   off-by-one in the wild (E1 → F-I1/F-I3).
5. **A dropped run is an attempt** — a verdict-keyed counter cannot see it (E2 →
   F-I7/F-I8).

## Application to pyfinagent (evidence for PLAN; Main owns the contract)

The literature supports four structural options, in the order the evidence
favours them. These are findings, not a plan:

- **(i) Split the field.** Return `attempt_number` and `prior_attempts` as two
  named integers from `qa_wip.report()` (`scripts/qa/qa_wip.py:299-320`), so the
  E1/F-I3 ambiguity and the F-I4 ordering dependency both stop being readable
  errors. Structural remedy per E6.
- **(ii) Persist the count separately from the retained records.** A monotonic
  per-step high-water mark that pruning cannot lower (E3/E5). This is the only
  option that keeps F1b reachable if `prune_wip_records` is ever wired.
- **(iii) If (ii) is not taken, record what was pruned.** `prune_wip_records`
  should return/record the removed count into a durable place, the
  `PERF_RECORD_LOST` shape (E5) — the minimum that keeps the count *knowable*.
- **(iv) Fail closed when the count is not knowable.** Copy
  `verdict_history_86_21.py:98-113` — return `None`, never 0 — and extend it to
  the "records were pruned" case that `qa.md:713-717` currently concedes is
  undetectable (objective (d)).

**Scoping note for the contract:** F-I2 establishes that the saturation defect is
**latent, not live** — `prune_wip_records` has no automatic caller today, so no
current step's escalation has actually been suppressed by it. F-I1, F-I3 and
F-I4 are live today and independent of pruning.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **10**
- [x] 10+ unique URLs total — **25** (10 full + 15 snippet-only)
- [x] Recency scan (last 2 years) performed + reported — 4 queries, 3 findings
- [x] Full pages read (not abstracts) for the read-in-full set — arXiv source
      fetched via `arxiv.org/html/` per the chain; no `/pdf/` fetch attempted
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
      (`qa_wip.py`, `qa.md`, `attempt_budget.py`, `verdict_history_86_21.py`,
      the verdicts dir, CLAUDE.md F1/F1b)
- [x] Contradictions noted (E1 Temporal vs Step Functions; F-I1 comment vs code;
      F-I3 the two halves of one qa.md sentence)
- [x] All claims cited per-claim with URL or file:line
- [ ] **Gap, disclosed:** the LangGraph `GRAPH_RECURSION_LIMIT` page returned a
      bare redirect and was NOT read; LangGraph is covered only by the objective's
      framework list, not by a fetched source. Temporal, Step Functions, Airflow
      and the OpenAI Agents SDK were all read in full, so the framework question
      is answered but not exhaustively.
- [ ] **Gap, disclosed:** no source names "counter saturation disables
      escalation" as a documented defect class (recency finding 3); the argument
      rests on Prometheus's counter/gauge type rule and the kernel's lost-record
      pattern rather than on a paper about this exact failure.
</content>
