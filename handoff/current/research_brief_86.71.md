# Research Brief -- step 86.71: cumulative attempt/retry budgets for long-running autonomous loops

Tier: **moderate** (caller-stated). Audit-class: **NO** (`coverage` reported for information only).
Accessed 2026-08-17. Researcher = Layer-3 combined external-literature + internal-code explorer.

## Envelope (born inert -- phase-86.37; flipped to COMPLETE as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 21,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "summary": "PreToolUse DOES fire on Workflow (655 rows, 2026-05-28..2026-08-17) -- the origin seam is real and measurable. But it is fail-OPEN on everything except exit 2, and the existing danger hook persists only {ts,tool,verdict,reason}, never tool_input. attempt_budget.py has ZERO file I/O and ZERO runtime callers. The module docstring's claim that no SRE bound resets on success is REFUTED by Fowler's canonical breaker. Anthropic harness-design, the 2607.01641 loop survey and LoopTrap all decline to recommend cumulative budgets or human escalation -- the step's premise is under-supported externally and must stand on internal measurement.",
  "brief_path": "handoff/current/research_brief_86.71.md",
  "gate_passed": true
}
```

## Objective

Cumulative attempt/retry budgets for long-running autonomous loops: SRE retry-budget and
circuit-breaker literature (cumulative-over-window vs reset-on-success), attempt-vs-outcome
accounting for spawns that return nothing, deterministic pre-execution enforcement via Claude Code
PreToolUse hooks, append-only ledgers as cross-session persistent counters, and
escalation-to-operator design where exhaustion can never auto-pass.

## Search queries run (three-variant discipline)

| # | Variant | Query |
|---|---------|-------|
| 1 | year-less canonical | `retry budget SRE client-side retry amplification cumulative window vs consecutive failures` |
| 2 | current-year frontier (2026) | `agent retry budget cumulative attempt accounting long-running autonomous loop 2026` |
| 3 | year-less canonical | `resilience4j circuit breaker sliding window COUNT_BASED versus consecutive failures reset` |
| 4 | last-2-year window (2025) | `LLM agent loop termination budget exhaustion escalate to human arXiv 2025` |

---

## Read in full (9; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://code.claude.com/docs/en/hooks | 2026-08-17 | Official docs (Anthropic) | WebFetch (301 from `docs.claude.com/en/docs/claude-code/hooks`; refetched at target) | PreToolUse runs "Before a tool call executes. Can block it". Exit 2 "blocks the tool call" and "exit 2 blocks whether or not you print JSON: even a JSON `permissionDecision` of `\"allow\"` can't override it." Input carries `tool_name`, `tool_input`, **`tool_use_id`**, `agent_id`, `agent_type`, `session_id`, `cwd`, `permission_mode`, `effort.level`. Matchers of letters/digits/`_`/`-`/space/`,`/`\|` are **exact-string**, anything else is an unanchored JS regex. **Everything but exit 2 fails OPEN**: "A timed-out ... hook doesn't block the tool call ... so don't count on a stalled hook to act as a gate"; a schema-invalid exit-0 object "is a non-blocking error: the action proceeds"; "A hook that can't start lands in the same non-blocking bucket". Default `command` timeout 600s. |
| 2 | https://gateway-api.sigs.k8s.io/geps/gep-3388/ | 2026-08-17 | Official design doc (k8s SIG-Network GEP) | WebFetch | Budget = **percentage of active requests over an interval**, never a streak: "Interval defines the duration in which requests will be considered for calculating the budget for retries" (default 10s; 1s..1h). "Retrying the same original request multiple times within the retry budget interval will lead to each retry being counted towards calculating the budget." `MinRetryRate` floor "ensures that requests can still be retried during periods of low traffic". **Does not discuss consecutive-failure counters at all.** |
| 3 | https://sre.google/sre-book/addressing-cascading-failures/ | 2026-08-17 | Canonical book (Google SRE ch. 22) | WebFetch | "Consider having a server-wide retry budget. For example, only allow 60 retries per minute in a process, and if the retry budget is exceeded, don't retry; just fail the request." Also "Limit retries per request. Don't retry a given request indefinitely." Amplification is **multiplicative**: "a single request at the highest layer may produce a number of attempts as large as the _product_ of the number of attempts at each layer"; 3 layers x 4 attempts = 64. |
| 4 | https://docs.temporal.io/encyclopedia/retry-policies | 2026-08-17 | Official docs (Temporal) | WebFetch | Maximum Attempts is **INCLUSIVE of the first attempt** -- 1 means "a single execution attempt and no retries". "If this limit is exceeded, the execution fails without retrying again. When this happens an error is returned." Non-retryable errors "will not be retried, regardless of a Retry Policy". **The page does NOT state whether the attempt counter survives a worker restart** -- do not cite it for durability. |
| 5 | https://martinfowler.com/bliki/CircuitBreaker.html | 2026-08-17 | Authoritative blog (named practitioner, canonical) | WebFetch | **[ADVERSARIAL -- refutes an internal claim.]** The canonical breaker *does* reset on success: "successful calls reset it back to zero" (`@failure_count = 0`); half-open "will either reset the breaker if successful or restart the timeout if not." Fowler "does **not** discuss trade-offs between consecutive-failure counters and rolling-window approaches. It only presents the simple consecutive-counter model." On escalation: "Usually you'll also want some kind of monitor alert if the circuit breaker trips" (`@monitor.alert(:open_circuit)`). |
| 6 | https://resilience4j.readme.io/docs/circuitbreaker | 2026-08-17 | Official docs (resilience4j) | WebFetch | The rate-based alternative Fowler omits. "The count-based sliding window aggregrates the outcome of the last N calls" (circular array of N); TIME_BASED "aggregrates the outcome of the calls of the last N seconds". Trips "when the failure rate is equal or greater than a configurable threshold", gated by `minimumNumberOfCalls`: "if only 9 calls have been evaluated the CircuitBreaker will not trip open even if all 9 calls have failed." A success occupies a slot; it does **not** zero the window. |
| 7 | https://arxiv.org/html/2607.01641v1 | 2026-08-17 | Preprint (arXiv; "When Agents Do Not Stop") | WebFetch (`/html/` per the arXiv chain -- never `/pdf/`) | Measured taxonomy (Table II, §V-A) of 68 confirmed infinite-agentic-loop cases: **retry feedback without bound 17 (25.0%)**, tool-call iteration 16 (23.5%), multi-agent chat 14 (20.6%), workflow loop 9 (13.2%), message reentry 7 (10.3%), runner/delegation/evaluator feedback 5 (7.4%). Impacts: **API cost exhaustion 65 (95.6%)** and model DoS 65 (95.6%). §VI-A: "bounds should be enforced at the runtime scope where feedback is created, rather than exposed only as optional local parameters." Why per-iteration limits fail (§I): "developers may omit them, misuse them, configure them with ineffective bounds, or place them outside the actual feedback path"; "termination conditions are semantically fragile". **[Qualifying]** The paper does **not** recommend cumulative/global budgets, persistent counters, or human escalation. |
| 8 | https://arxiv.org/html/2605.05846v1 | 2026-08-17 | Preprint (arXiv; "LoopTrap") | WebFetch (`/html/`) | **[ADVERSARIAL.]** Defines Termination Poisoning (§1): "an adversary injects malicious content into the agent's operational context to corrupt the progress signals the agent uses to assess task completion, thereby preventing termination and inducing unbounded execution loops." Core vulnerability (§4.3): "Progress evaluation is entrusted to the same LLM reasoning engine that processes potentially untrusted external content." Recommended defense (§7) is "an independent, sandboxed module [that] validates the agent's self-assessed progress against objective completion criteria" -- they do **not** name an external deterministic budget as the primary defense. |
| 9 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-17 | Official engineering blog (Anthropic; the project's canonical harness reference) | WebFetch | **[ADVERSARIAL -- under-supports the citation CLAUDE.md rests on.]** Confirms the file-handoff and hard-threshold doctrine: "Communication was handled via files: one agent would write a file, another agent would read it and respond..."; "Each criterion had a hard threshold, and if any one fell below it, the sprint failed and the generator got detailed feedback on what went wrong." Iteration counts are **empirical, not prescriptive**: "I ran 5 to 15 iterations per generation". But **"The article does not explicitly discuss retry counts, failure thresholds, or escalation procedures for handing work back to humans"** and "does not address what happens when a step repeatedly fails or how many failures trigger human intervention." |

### Attempted but NOT read in full (honest record; does NOT count toward the gate)

| URL | Why not counted |
|-----|-----------------|
| https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/ | 301 -> `builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/...`; the redirect target returned **only the "AWS Builder Center" header, no body** (JS-rendered shell). Refetched once at the target; same result. The token-bucket retry-quota claim is therefore **NOT sourced** in this brief -- do not cite AWS for it. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://tianpan.co/blog/2026-04-16-retry-budget-llm-agent-cost-amplification | blog | 2026 recency hit; community tier, superseded by #7 |
| https://waxell.ai/blog/ai-agent-token-budget-enforcement | vendor blog | vendor marketing; claim ($47k runaway) already carried by #7 |
| https://aisecurityguard.io/reports/secrets-of-llm-whisperer/8_retry_cost | blog | community tier |
| https://techwithsyl.substack.com/p/the-agent-loop-is-now-infrastructure | blog | community tier |
| https://www.developersdigest.tech/blog/loop-engineering-designing-agent-loops | blog | community tier |
| https://claudeskills.info/loop-engineering/ | blog | unofficial, name-adjacent to Claude; deliberately not cited |
| https://levelup.gitconnected.com/engineering-reliable-coding-agent-loops-control-flow-verification-retries-and-stop-conditions-f002d2dc168c | blog | community tier |
| https://arxiv.org/html/2607.06503v1 | preprint ("Doomed from the Start": early abort via probe cascade) | adjacent (predictive abort, not budget accounting); flagged for a future step |
| https://www.researchgate.net/publication/405474946_Budget-Aware_LLM_Agents | preprint (Budget-Aware LLM Agents) | paywalled aggregator; the concept (agent maintains live budget estimate) conflicts with LoopTrap's finding that self-assessment is untrustworthy -- noted as a debate, below |
| https://arxiv.org/html/2604.11378v1 | preprint (scheduler-theoretic agent execution) | adjacent framing, not budget accounting |
| https://arxiv.org/pdf/2603.19896 | preprint (utility-guided orchestration) | `/pdf/` only; not fetched per the arXiv chain rule |
| https://arxiv.org/pdf/2602.15112 | preprint (ResearchGym) | off-topic |
| https://deepwiki.com/resilience4j/resilience4j/2.2-circuit-breaker-configuration | third-party wiki | superseded by official #6 |
| https://github.com/resilience4j/resilience4j/issues/1731 | issue thread | community tier |
| https://github.com/resilience4j/resilience4j/issues/1362 | issue thread | community tier |
| https://reflectoring.io/circuitbreaker-with-resilience4j/ | blog | superseded by official #6 |
| https://dzone.com/articles/why-retries-are-more-dangerous-than-failures-2 | blog | superseded by #3 |
| https://thinhdanggroup.github.io/retry-without-thundering-herds/ | blog | superseded by #3 |
| https://sreschool.com/blog/retry/ | blog | community tier |
| https://codelit.io/blog/retry-exponential-backoff | blog | backoff, not budget |
| https://sujeet.pro/articles/exponential-backoff-retry-strategy | blog | backoff, not budget |

**URLs collected: 30** (9 read in full + 21 snippet-only). Read-in-full mix by tier: 2 preprints,
5 official docs/design docs, 1 canonical book, 1 named-practitioner blog. No community-tier source
counts toward the floor.

---

## Recency scan (last 2 years, 2024-2026)

**Performed.** Queries 2 and 4 above were scoped to 2026 and 2025 respectively.

**Result: 4 new findings that COMPLEMENT and one that QUALIFIES the canonical sources.**

1. **arXiv 2607.01641 (2026)** supplies the first measured base rates for this failure class:
   retry-feedback-without-bound is the single largest cause at 25.0%, and API cost exhaustion is
   the impact in 95.6% of cases. The classical SRE sources (#2, #3) are about *throughput
   amplification*; this is about *cost exhaustion in a single logical task*, which is exactly
   86.71's shape. **New, not superseding.**
2. **arXiv 2607.01641 §VI-A qualifies the plan's placement**: "bounds should be enforced at the
   runtime scope where feedback is created, rather than exposed only as optional local
   parameters." A library that must be imported *is* an optional local parameter. This is direct
   external support for enforcing at the **hook/tool-call seam** rather than inside
   `attempt_budget.py`.
3. **arXiv 2605.05846 (2026, LoopTrap)** adds an adversarial reason the counter must live outside
   the agent: an agent's own progress signal is corruptible, so a budget the agent maintains about
   itself is not a bound.
4. **Budget-Aware LLM Agents (2026, snippet-only)** argues the opposite -- that the agent should
   maintain a live estimate of its remaining budget. Recorded as an open debate, below.
5. **No 2024-2026 source supersedes** Google SRE ch.22 or GEP-3388 on the cumulative-window
   mechanism; those remain the canonical statements.

---

## Key findings

1. **PreToolUse DOES fire on the `Workflow` tool -- measured, not assumed.** The origin seam the
   step needs is real. `handoff/audit/pre_tool_use_audit.jsonl` (185,014 parseable rows) contains
   **655 `Workflow` rows**, earliest `2026-05-28T20:42:05Z`, latest `2026-08-17T09:40:29Z`
   (i.e. today, this session). 99 distinct tool names; `Agent` = 1,226 rows; `Task` = **0**.
   *(Source: measured this session against the hook's own audit stream.)*

2. **But the hook's own audit CANNOT tell you which step a run was for.** `pre-tool-use-danger.sh:53`
   emits exactly one shape: `printf '{"ts":"%s","tool":"%s","verdict":"%s","reason":"%s"}\n'`.
   **185,020 of 185,020 rows carry the key-set `(reason, tool, ts, verdict)` and nothing else** --
   `tool_input` is received (parsed into `$INPUT` at `pre-tool-use-danger.sh:38-44`) and then
   **discarded**. So the existing stream is a *positive control that the seam fires*, and is
   **useless as a retrospective attempt ledger**. A new writer is required; the history cannot be
   backfilled from it.

3. **A PreToolUse gate is deterministic only in the deny direction; it fails OPEN everywhere else.**
   Per the official docs (#1): a timed-out hook "doesn't block the tool call ... so don't count on
   a stalled hook to act as a gate"; a schema-invalid exit-0 object is "a non-blocking error: the
   action proceeds"; "A hook that can't start lands in the same non-blocking bucket." Only exit 2
   (or a well-formed `permissionDecision: "deny"`) blocks. **This is the correct discipline for
   this repo** (`auto-commit-and-push.sh:29` is a blanket `trap 'exit 0' EXIT` with the comment
   "never blocking the masterplan Write that triggered it") **but it means the hook can never be
   the sole bound** -- a broken budget hook is silently no budget at all. Loud-on-failure
   (`systemMessage`, an append to the audit log) is what converts fail-open into fail-open-**but-loud**.

4. **`attempt_budget.py` cannot persist anything -- confirmed by a direct scan, not inference.**
   Grepping the module for `open(` / `json.load` / `write_text` / `Path(` returns **nothing at
   all**. `to_json()` (`attempt_budget.py:216-227`) returns a *string*. Cross-session persistence,
   which criterion 1 demands, is not a wiring gap in this module -- it is absent by construction.

5. **Zero runtime callers, positive-controlled.** `grep -rl attempt_budget` (excluding `handoff/`,
   `node_modules`, `.git`) returns only: `.claude/masterplan.json`, `CLAUDE.md`,
   `backend/tests/test_phase_86_32_attempt_budget.py`, `scripts/qa/mutation_matrix_86_32.py`,
   `scripts/qa/verify_counter_86_79.py`, and Q/A memory files. **Positive control**: the same grep
   for `qa_wip` returns `.claude/agents/qa.md`, `.claude/workflows/qa-verdict.js` and eight
   `scripts/qa/*` files -- so the pattern finds live wiring when live wiring exists. This
   reproduces the caller's banked measurement independently.

6. **The module's own docstring overstates the literature, and Fowler refutes it.**
   `attempt_budget.py:16-18` asserts: *"Every bound in the SRE literature is cumulative over a
   window, never a consecutive streak ... None of them reset on one success."* The canonical
   circuit breaker (#5) is **exactly** a consecutive counter that resets on one success --
   "successful calls reset it back to zero." The accurate claim is narrower and still sufficient:
   *rate-limiting* bounds (Google SRE's 60/min, GEP-3388's percentage-of-interval, resilience4j's
   sliding window) are cumulative; the *availability* breaker is the streak, and it is a
   health-check on a dependency, not work accounting. **The docstring's conclusion survives; its
   supporting sentence does not.** Fix the sentence before the contract quotes it.

7. **Inclusive-vs-exclusive attempt counting is a live cross-vendor trap.** Temporal (#4):
   `MaximumAttempts = 1` means "a single execution attempt and no retries" -- inclusive. This
   corroborates the warning already written at `qa_wip.py:508-519` (Temporal inclusive vs Step
   Functions exclusive, "same word, off by one"). `attempt_budget.py:122-124` uses
   `attempts_used >= max_attempts` with `attempts_used = len(self.attempts)`, i.e. **inclusive** --
   consistent with Temporal, and the unit must travel with the number in any persisted record.

8. **The house append-only-ledger pattern is proven and directly reusable, with one mismatch.**
   `scripts/qa/verdict_ledger_write.py` is the model: one JSONL, producer-assigned dedup key,
   never rewrites, loud on every failure path (exit 2/3/4), separate `date` (event) and
   `recorded_at` (write) fields. `handoff/verdict_ledger.jsonl` currently holds **47 rows**.
   **The mismatch**: its key is `(step_id, run_id)` -- and a *PreToolUse* hook has **no `run_id`**,
   because the run does not exist yet. Per phase-86.85's measured finding, `runId` first appears in
   the **PostToolUse launch receipt** `{runId, scriptPath, status, summary, taskId, taskType,
   transcriptDir, workflowName}`. The natural producer-assigned key available *before* execution is
   **`tool_use_id`**, which the docs (#1) list in the PreToolUse input and describe as "Unique ID
   for this tool call". That is precisely an ATTEMPT identifier.

9. **An attempt ledger and a verdict ledger are different ledgers, and merging them re-creates the
   bug.** `verdict_ledger_write.py` records OUTCOMES (its vocabulary is
   `{PASS, CONDITIONAL, FAIL, NO_VERDICT}`); a dropped spawn only ever gets a row if a human writes
   one. An ATTEMPT ledger written at PreToolUse is by construction complete, because it is written
   *before* the thing that might drop. This is the attempt-vs-outcome distinction the module
   docstring names (`attempt_budget.py:28-38`) expressed as two files rather than one.

10. **`qa-write-guard.sh` is the working precedent for a PreToolUse deny in this repo**, including
    the identity read (`qa-write-guard.sh:102` `agent_type = d.get("agent_type") or ""`; deny at
    `:196` `exit 2`, allow at `:199` `exit 0`) and the caveat that matters here: its own comment at
    `:106-108` records that **`agent_type` is chosen by the SPAWNER** (RFC 9700 §4.15), so it is an
    identity claim, not an authenticated one. A budget keyed on `step_id` read from
    `tool_input.args` inherits the same property: the caller supplies it.

11. **Matcher syntax makes `Workflow` exactly targetable.** Per #1, a matcher of letters/digits/
    `_`/`-`/spaces/`,`/`|` is compared as an exact string; `Workflow` qualifies. **No PreToolUse
    matcher in `.claude/settings.json` currently mentions `Workflow` or `Task`** -- the two
    registered PreToolUse hooks are `pre-tool-use-danger.sh` (no matcher = all tools) and
    `qa-write-guard.sh` (`Write|Edit`). A third entry with `matcher: "Workflow"` is additive and
    touches neither.

12. **Escalation-never-auto-passes is already proven in the module and must not be re-litigated.**
    `disposition()` (`attempt_budget.py:134-147`) checks PASS first, then exhaustion, with no third
    branch; `escalation_summary()` (`:172-214`) emits "## THIS IS NOT A PASS AND NOT A FAIL" and
    "No verdict is implied by exhaustion, and none may be inferred from it." Guarded by
    `test_exhaustion_cannot_auto_pass` (`:102`), `test_a_fail_stays_a_fail_under_every_flag_combination`
    (`:148`), `test_residuals_door_requires_an_actual_pass` (`:169`), and mutation cells
    `M3-exhaustion-auto-passes` / `M4-residuals-door-opens-for-a-fail`. **The safety property is
    done; only persistence and wiring are missing.**

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/harness/attempt_budget.py` | 332 | The cumulative budget: `BudgetState`, `Outcome` (incl. `NO_VERDICT`), `disposition()`, `close_kind()`, `escalation_summary()`, 86.28 replay fixture | **UNWIRED + NON-PERSISTENT.** Zero runtime callers (finding 5); zero file I/O (finding 4). Docstring `:16-18` overstates the literature (finding 6). |
| `backend/tests/test_phase_86_32_attempt_budget.py` | 288 | 15 tests, incl. `test_exhaustion_cannot_auto_pass:102`, `test_dropped_spawns_count_against_the_budget:66`, `test_86_28_replay_terminates_where_the_legacy_rule_never_would:271`, `test_fixture_matches_the_recorded_ledger:220` | Green + genuinely discriminating. Covers **semantics only** -- no test touches persistence or a process boundary, because neither exists. |
| `scripts/qa/mutation_matrix_86_32.py` | 225 | 8 cells M1-M8; M3/M4 named "the safety-critical ones" | Existing guard set. **No cell covers persistence or cross-process reads** -- necessarily, since there is nothing to mutate. New cells are needed for criterion 1. |
| `scripts/qa/qa_wip.py` | 636 | The LIVE per-step attempt counter. `report():448`, `_attempt_counts():378`, `source_present():182`, `read_loss():235`, `prune_wip_records():276` | **Wired** (read by `.claude/agents/qa.md` + `.claude/workflows/qa-verdict.js`). The fail-closed doctrine to copy: every uncomputable path returns `None`, never `0` (`:401`, `:390-399`), because "a threshold compared against 0 silently suppresses escalation". Also the `PERF_RECORD_LOST` loss ledger (`:137-145`, `:253-273`) -- monotonic, written **before** the unlink so a crash over-counts (escalate early = safe). |
| `scripts/qa/verdict_ledger_write.py` | 577 | The house append-only JSONL writer: `_dedup_key():126`, `append_row():244`, `emit_sequence():263`, `_self_test():298` | **Wired**, the pattern to reuse. Key `(step_id, run_id)` is unavailable pre-execution (finding 8). Records outcomes, not attempts (finding 9). |
| `handoff/verdict_ledger.jsonl` | 47 rows | Verdict history consumed by `enforceEscalation` | Live. Outcome-keyed. |
| `.claude/settings.json` (hooks) | -- | 2 PreToolUse entries: `pre-tool-use-danger.sh` (no matcher), `qa-write-guard.sh` (`Write\|Edit`); 8 PostToolUse; `Stop` has `timeout: 55` | **No matcher mentions `Workflow` or `Task`.** Adding one is additive. |
| `.claude/hooks/pre-tool-use-danger.sh` | -- | The all-tools PreToolUse guard. Parses `tool_name` + `tool_input` from stdin (`:29-44`); logs at `:53`; blocks via exit 2 | Fires on `Workflow` 655 times (finding 1) but **persists no `tool_input`** (finding 2). Header states the discipline verbatim: "Designed to FAIL OPEN on any internal error -- a broken guard must not brick the session." |
| `.claude/hooks/qa-write-guard.sh` | -- | PreToolUse deny precedent: `agent_type` read `:102`, `is_qa_role():153`, deny `:196`, allow `:199` | Working model for a step-scoped deny. Notes `agent_type` is spawner-chosen (`:106-108`). |
| `.claude/hooks/auto-commit-and-push.sh` | -- | Single-writer lock at `.git/pyfinagent-auto-commit.lock.d` (`:299`); explicit staleness policy (`:286-292`); blanket `trap 'exit 0' EXIT` (`:29`) | The concurrency precedent. macOS has **no `flock(1)`**, and `shlock` was measured not to break a stale lock -- so the house pattern is `mkdir` + dead-pid/age staleness, **fail-open on timeout** (`:296-298`). Any ledger appended from two concurrent sessions needs this. |
| `handoff/audit/pre_tool_use_audit.jsonl` | 185,014 rows | The hook's append-only stream | Proves the seam; unusable as an attempt ledger (finding 2). |
| `CLAUDE.md` (F1/F1b block) | -- | Documents the cumulative budget and says plainly it is "NOT YET WIRED", with "no runtime caller" and "no persistence" | Accurate. It also already warns: "do not read the paragraphs below as a termination guarantee this harness actually has". |

---

## Consensus vs debate (external)

**Consensus.** (a) Bounds on retry must be *cumulative over a window*, not per-request only --
Google SRE (#3), GEP-3388 (#2), resilience4j (#6) agree. (b) Retry amplification is multiplicative
across layers, so bound at one layer (#3). (c) Exhaustion means *fail*, not *proceed*: "if the
retry budget is exceeded, don't retry; just fail the request" (#3); "the execution fails without
retrying again" (#4). (d) Attempt limits are inclusive of the first attempt in the durable-execution
world (#4).

**Debate 1 -- does a success reset the counter?** Fowler's canonical breaker: **yes** (#5).
resilience4j's sliding window: **no**, a success merely occupies a slot (#6). GEP-3388: the question
does not arise (#2). This is not a contradiction once the *purpose* is separated: reset-on-success
is right for **liveness of a dependency**, wrong for **cumulative cost of one work item**. That
distinction -- not a blanket "nothing resets" -- is the defensible claim for the contract.

**Debate 2 -- should the agent own its own budget?** Budget-Aware LLM Agents (snippet-only) says the
agent should maintain a live estimate of remaining budget. LoopTrap (#8) shows self-assessed
progress is adversarially corruptible and recommends "an independent, sandboxed module". For 86.71
the internal evidence settles it: the loop being bounded is the *Q/A grading loop*, and letting the
graded party own the counter is the confused-deputy shape `qa_wip.py:16-26` already rejected.

**Where the literature is SILENT, and this must be said plainly.** Three sources the step's premise
would naturally lean on decline to support it: Anthropic harness-design (#9) "does not explicitly
discuss retry counts, failure thresholds, or escalation procedures for handing work back to humans";
arXiv 2607.01641 (#7) does not recommend cumulative budgets, persistent counters or human
escalation; LoopTrap (#8) does not name external budgets as a defense. **The 5-attempt / 1.2M-token
ceiling is justified by pyfinagent's own measured run distribution, not by external authority** --
the contract should cite the internal measurement and must not attribute the number to Anthropic.

---

## Pitfalls (from literature + measured internally)

1. **A fail-open gate that breaks is indistinguishable from no gate.** #1 is explicit ("don't count
   on a stalled hook to act as a gate"). Mitigation: on any internal error the hook must still
   append a loud row and emit `systemMessage`, so absence-of-enforcement is *visible*.
2. **Bounds placed outside the feedback path do nothing** (#7 §I). A budget importable but never
   imported is the measured instance of this (findings 4-5).
3. **A counter that saturates stops binding exactly where it matters.** `qa_wip.py:286-291` records
   this from phase-86.79: pruning made the retained-record count saturate at `keep`, and "F1b's
   5-attempt escalation becomes unreachable the moment anyone schedules pruning." Any new counter
   must be monotonic and must record its own losses.
4. **Zero is a claim; absence is not zero.** `qa_wip.py:390-399` -- a missing source returns `None`.
   A budget reading "0 attempts" from a missing ledger fails OPEN and silently.
5. **A name is not a unit** (#4 vs Step Functions; `qa_wip.py:508-519`). Persist the inclusivity of
   `attempt_number` next to the number.
6. **Dedup keys designed for outcomes drop attempts.** `verdict_ledger_write.py` refuses a duplicate
   `(step_id, run_id)` with exit 2 -- "the benign 'already recorded' code a caller ignores"
   (`:449-455`). An attempt ledger keyed on something unavailable at PreToolUse would silently lose
   rows in exactly that way.
7. **Two concurrent Claude sessions are routine here** (`qa_wip.py:112-115`; `auto-commit-and-push.sh:280-299`).
   An unlocked append from two sessions interleaves. Use the house `mkdir`-lock + staleness policy;
   `flock(1)` does not exist on macOS.
8. **Cardinality agreement is not identity.** `attempt_budget.py:230-256` documents the 86.28
   fixture that matched on count while differing on 5 of 8 members. Any replay evidence for 86.71
   must re-derive from the record, not assert a constant.

---

## Application to pyfinagent (external findings -> file:line anchors)

- **Criterion "wiring at the path where runs actually originate"** -> the origin is the `Workflow`
  tool call, and it is measurably observable (655 rows). Register a third PreToolUse entry in
  `.claude/settings.json` with `matcher: "Workflow"` (exact-string per #1) alongside the existing
  `pre-tool-use-danger.sh` / `qa-write-guard.sh` entries. External support: #7 §VI-A, "bounds should
  be enforced at the runtime scope where feedback is created".
- **Criterion "cross-session persistence across separate process invocations"** -> `attempt_budget.py`
  has no I/O (finding 4), so persistence must be added as an append-only JSONL modelled on
  `verdict_ledger_write.py:244` (`append_row`), keyed on `tool_use_id` (available pre-execution per
  #1) rather than `run_id` (only in the PostToolUse receipt). A hook process is a *separate process
  invocation by construction*, so hook-writes + a CLI read is the natural demonstration.
- **Criterion "exhaustion escalates and never auto-passes"** -> already satisfied in-module at
  `attempt_budget.py:134-147` + `:172-214`, guarded by `test_exhaustion_cannot_auto_pass:102` and
  cells M3/M4. Deny surface: exit 2, or `hookSpecificOutput.permissionDecision: "deny"` with
  `permissionDecisionReason` carrying `escalation_summary()` (#1). Note `"ask"` also exists and maps
  cleanly to "operator decides" -- but in an unattended/cron session `ask` has no one to ask, so
  `deny` + a written escalation artifact is the safer default.
- **Criterion "mutation-tested guards"** -> extend `scripts/qa/mutation_matrix_86_32.py` (8 cells
  today) with cells that (a) make the ledger write non-monotonic, (b) make a missing ledger read as
  0 instead of `None` (the `qa_wip.py:390-399` fail-closed property), (c) make the hook exit 0 on
  exhaustion, and (d) make the dedup key drop `step_id` -- the exact class
  `verdict_ledger_write.py:417-433` already proves is fail-open.
- **Correct the docstring before the contract quotes it** -> `attempt_budget.py:16-18` (finding 6).
  The contract should state the narrower, true claim: *rate-limiting* bounds are cumulative;
  reset-on-success belongs to dependency health checks, not work accounting.
- **Do not attribute the 5/1.2M ceiling to external authority** -> #9 does not supply it. Cite the
  internal distribution recorded at `attempt_budget.py:47-58`.

---

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **9**
- [x] 10+ unique URLs total (incl. snippet-only) -- **30**
- [x] Recency scan (last 2 years) performed + reported -- 4 complementing findings + 1 debate
- [x] Full papers / pages read (not abstracts) for the read-in-full set -- both arXiv papers via
      `/html/`, never `/pdf/`; the one failed fetch (AWS) is disclosed and excluded
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope (12 files/artifacts)
- [x] Contradictions / consensus noted -- two live debates + three silences recorded, including one
      that refutes an internal docstring and one that under-supports a CLAUDE.md citation
- [x] All claims cited per-claim (URL + access date, or file:line)

*Coverage note: this step is NOT audit-class, so `coverage.dry` is informational and not required.
Three search/fetch rounds were run; the third still produced new findings, so `dry: false` is
reported honestly rather than asserted.*
