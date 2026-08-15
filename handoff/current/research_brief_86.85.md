# Research Brief — step 86.85

**Topic:** Making an append-only verdict ledger self-feeding — durably recording one row per LLM-evaluator verdict at the moment it is issued, when the evaluator is a constrained-decoding structured-output task with no filesystem access and the only writer is the orchestrator that transcribes the verdict.

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Started:** 2026-08-14. **Researcher:** Layer-3 researcher (Workflow rail).

---

## ENVELOPE (born inert — phase-86.37; updated in place as sources land)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 15,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "summary": "The Workflow-rail Q/A has NO filesystem access by RUNTIME PROPERTY (research-gate.js:52-55: an `import fs` is a SyntaxError), so an evaluator-authored ledger row is impossible -- and the 2026 literature says that is the RIGHT architecture, not a workaround: ESAA (arXiv:2602.23193 §3) has the agent emit validated JSON with no write permission while a deterministic orchestrator appends, because it stops an agent tampering with its own audit trail (§6.5). Record BEFORE the irreversible effect (§3.2). Dedup key = producer-assigned logical-event id; `(step_id, run_id)` fits (run_id present on 33/35 rows). Measured: verdict_ledger.jsonl is 35/35 recorded_by=main, with 12 rows sharing ONE microsecond timestamp -- it is 100% backfill, never a seam write. The reader AND the consumer are already correct and fail-closed (None never 0); only the WRITER is missing. `Workflow` (600) and `Agent` (1,225) ARE real tool_name values in the PreToolUse audit, so a PostToolUse hook would fire -- but NO hook in this repo reads tool_response, so payload visibility is UNPROVEN and must be measured. On silence: NIST AU-5 mandates alerting, factors shutdown into AU-5(4), and Microsoft ships hard fail-closed DISABLED by default -- so fail-closed on the DECISION, alert always, never halt the harness.",
  "brief_path": "handoff/current/research_brief_86.85.md",
  "gate_passed": true
}
```

**Envelope flipped to COMPLETE as the final act (phase-86.37).** Gate arithmetic: 8 >= 5 read in full; 23 >= 10 URLs; recency scan performed and reported; step is not audit-class so `coverage.dry` is informational only.

---

## Work log (append-only)

- [t0] Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full.
- [t0] Brief created with born-inert envelope.
- [t1] Internal: measured `handoff/verdict_ledger.jsonl` = 35 rows, 100% `recorded_by=main`.
- [t2] Searches run (3-variant discipline, see "Queries run" below).
- [t3] Read in full #1 (arXiv 2606.04990) and #2 (event-driven.io audit-log-vs-ES) — envelope bumped to 2.

### Queries run (three-variant discipline, `.claude/rules/research-gate.md` §Search-query composition)

| Variant | Query | Purpose |
|---|---|---|
| Current-year frontier (2026) | `event sourcing append-only audit log AI agent harness durable state 2026` | frontier |
| Last-2-year (2025) | `tamper-evident audit log self-reported LLM agent provenance 2025` | recency scan |
| Year-less canonical | `idempotent append deduplication key exactly-once event log consumer` | canonical prior art (Kafka/EIP idempotent-consumer) |
| Year-less canonical | `audit log failure fail-closed halt system when logging unavailable NIST security requirement` | canonical prior art (NIST AU-5) |
| Year-less canonical | `event sourcing audit log source of truth vs derived side effect` | canonical prior art |

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://arxiv.org/html/2606.04990v1 | 2026-08-14 | paper (arXiv HTML) | WebFetch (arXiv `/html/` per gate rule) | Self-reported final answers collapse every failure mode into one endpoint; provenance must be written by **external instrumentation at the agent boundary**, agent role is *passive*. Timing (pre-exec / runtime / post-hoc) is a **risk-tiered choice**, not one right answer. Absence-of-record is an acknowledged OPEN problem — no protocol given. |
| 2 | https://event-driven.io/en/audit_log_event_sourcing/ | 2026-08-14 | authoritative blog (Oskar Dudycz, Event Store) | WebFetch | **[ADVERSARIAL to "just build a ledger"]** Argues an audit log is a *poor* architectural driver on its own: "Just recording the result of operations may not be enough" — you also need the *command/intention*, metadata (who, what permission), and a verification mechanism. Suggests the **outbox pattern** as the lighter alternative to full event sourcing. |
| 3 | https://arxiv.org/html/2602.23193 | 2026-08-14 | paper (arXiv HTML, ESAA, Feb 2026) | WebFetch (arXiv `/html/`) | **The closest structural match to 86.85.** "the agent does not have direct write permission to the project or the event store. Its role is to emit structured intentions and change proposals" (§3). A **deterministic orchestrator** validates against JSON Schema, appends to `activity.jsonl`, applies effects, reprojects. Agent output contract carries an explicit **`idempotency_key`** field (Appendix A) + sequential `event_seq`. "the event is recorded as a fact **before** any irreversible effect" (§3.2). Malformed/absent output → **fails CLOSED**: emits `output.rejected`, "No fallback to defaults; no recovery without agent resubmission" (§4.2). Rationale: "denying direct writing ... reduces the blast radius of a compromised agent" (§6.5) and prevents agents tampering with their own audit trail. |
| 4 | https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing | 2026-08-14 | official docs (Microsoft, updated 2026-04-20) | WebFetch | Canonical pattern. **Intent > state:** "an event that records *two seats were reserved* is more valuable than ... *remaining seats changed to 42*. ... **State-focused events reduce the event store to a change log that has no business meaning.**" **Idempotency:** "Event delivery to consumers is typically *at least once* ... **Without idempotency, projections drift from the eventstream** ... **Track the last processed event sequence number for each consumer and skip duplicates**, or design state mutations that are inherently safe to repeat." **Ordering/dedup at write:** "annotate each event ... with an incremental identifier. If two actions attempt to add events for the same entity at the same time, **the event store can reject an event that matches an existing entity identifier and event identifier.**" **Never mutate:** corrections are *compensating events*; "In-place migration ... breaks immutability and should be a last resort because **it undermines the audit trail**." Also an explicit caution: event sourcing "is a complex pattern that introduces significant trade-offs ... For most systems ... traditional data management is sufficient." |

| 5 | https://csf.tools/reference/nist-sp-800-53/r5/au/au-5/ | 2026-08-14 | official-docs rendering (NIST SP 800-53 r5 control catalog) | WebFetch | AU-5 "Response to Audit Logging Process Failures": (a) **"Alert [org-defined personnel] within [org-defined time period] in the event of an audit logging process failure"**; (b) take org-defined additional actions. Permitted actions: **"overwriting oldest audit records, shutting down the system, and stopping the generation of audit records."** Enhancements: AU-5(2) Real-time Alerts, **AU-5(4) Shutdown on Failure** ("full system shutdown, partial system shutdown, degraded operational mode"), AU-5(5) Alternate Audit Logging Capability. *Disclosure: this page is a structured rendering of the control, not the NIST PDF; the base statement + the action list + the enhancement titles are verbatim, the Discussion is summarised.* |
| 6 | https://developer.confluent.io/patterns/event-processing/idempotent-reader/ | 2026-08-14 | official docs (Confluent event-processing pattern catalog) | WebFetch | Dedup key design: **"A tracking ID should be a field that is unique to the *logical event*, such as an event key or request ID."** Consumers "read the tracking ID, cross-reference it against an internal state store of IDs it has already processed, and discard the event if necessary." Cheapest form: "the duplicate events can be deduplicated by the database during an **upsert on the event ID as primary key**." Caveat: **"A solution that requires EOS guarantees must enable EOS at all stages of the pipeline, not just on the reader."** |
| 7 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-14 | official docs (Anthropic, project-canonical) | WebFetch | **"Communication was handled via files: one agent would write a file, another agent would read it and respond either within that file or with a new file"**; "using structured artifacts to hand off context between sessions". Hard-threshold loop: **"Each criterion had a hard threshold, and if any one fell below it, the sprint failed and the generator got detailed feedback."** **NEGATIVE FINDING (important):** the article does **not** discuss crash-only semantics, resumability after interruption, append-only ledgers, or human escalation. The durability half of 86.85 has **no** Anthropic-canonical answer — it must come from the event-sourcing / audit-control literature. |
| 8 | https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-2000-server/cc938340(v=technet.10) | 2026-08-14 | official docs (Microsoft, `CrashOnAuditFail`) | WebFetch | **[ADVERSARIAL to naive fail-closed]** The canonical shipped fail-closed audit implementation: "it causes the system to **halt** if a security audit cannot be logged for any reason" → `STOP: C0000244 {Audit Failed}` bluescreen. Recovery is **manual and privileged**: "an administrator must log on, archive the log (if desired), clear the log, and reset this option." And the shipped default is the other way: **"By default, this policy is disabled."** i.e. the vendor that implemented hard fail-closed ships it OFF because a full-log condition becomes a full outage. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://arxiv.org/pdf/2606.30306 | paper — "Always-On Agents: A Survey of Persistent Memory, State, and Governance in LLM Agents" | Adjacent survey; the specific write-seam question is answered more directly by #1 and #3. PDF-only URL (gate rule forbids WebFetch on PDFs). |
| https://arxiv.org/pdf/2605.06812 | paper — "Towards Security-Auditable LLM Agents: A Unified Graph Representation" | Same ground as #1 at graph level; PDF-only. |
| https://github.com/bkuan001/halo-record | code — hash-chained tamper-evident agent runtime records | Reference implementation of the hash-chain idea already covered doctrinally by #3's `event_seq` + SHA-256 replay verify. |
| https://github.com/NousResearch/hermes-agent/issues/487 | code/issue — SHA-256 hash-chained action log for agent accountability | Issue thread, community tier; corroborates hash-chaining is live practice, adds no design constraint. |
| https://medium.com/@gwrx2005/fault-tolerant-distributed-ai-agent-harness-architecture-implementation-and-evaluation-674b25e46cdb | blog — DAPH distributed agent harness on Kafka | Kafka-scale answer; pyfinagent is single-machine, local-only. Community/blog tier. |
| https://www.developersdigest.tech/blog/log-is-the-agent-event-sourced-ai | blog — "The Log Is the Agent" | Popularisation of #3's thesis; lower tier, same claim. |
| https://dev.to/dailycontext/the-log-is-the-agent-5096 | blog (dev.to) | Community tier, duplicate of the above. |
| https://oneuptime.com/blog/post/2026-01-30-event-idempotency-keys/view | blog — "How to Create Event Idempotency Keys" (2026-01-30) | Practitioner tier; the key-composition rule is covered authoritatively by #6 + #4. |
| https://www.conduktor.io/blog/building-idempotent-consumers | blog — idempotent Kafka consumers | Industry tier, Kafka-specific. |
| https://pradeepl.com/blog/patterns/idempotent-consumer-pattern/ | blog — Idempotent Consumer (EIP) | Canonical *pattern* is better cited from #6. |
| https://docs.ecotone.tech/modelling/recovering-tracing-and-monitoring/resiliency/idempotent-consumer-deduplication | docs (framework) | Framework-specific dedup table implementation. |
| https://www.upguard.com/compliance/nist-sp-800-53/au/au-5 | vendor rendering of AU-5 | Duplicate of #5, lower authority. |
| https://csf.tools/reference/nist-sp-800-171/r3-0/03-03/03-03-04/ | official-docs rendering (NIST 800-171 r3 §03.03.04) | Derivative of AU-5 for CUI; same rule. |
| https://learn.microsoft.com/en-us/azure/architecture/patterns/materialized-view | official docs — Materialized View | Only needed if a projection is built; out of scope for a 35-row JSONL. |
| https://thehopiumlab.com/whiteboard/event-sourcing-for-llm-applications | blog — event sourcing for LLM apps | Raises the "replay costs tokens" objection, already captured via #4's snapshot guidance. |

**URLs collected: 8 read in full + 15 snippet-only = 23 unique.**

## Recency scan (2024-2026) — PERFORMED

Searched the last-2-year window explicitly (`...LLM agent provenance 2025`, `...AI agent harness durable state 2026`). **Result: 3 findings in the window that MATERIALLY change the design, not merely complement it.**

1. **ESAA (arXiv:2602.23193, Feb 2026)** — a 2026 paper describes *exactly* 86.85's topology (LLM emits validated JSON, has **no** write access, a deterministic orchestrator appends the event) and independently converges on the same three primitives this step needs: **idempotency key in the agent's output contract**, sequential `event_seq`, and **record-before-effect**. This is the single most load-bearing recent source.
2. **arXiv:2606.04990 (Jun 2026)** — establishes the *provenance* framing: the agent is passive, external instrumentation writes the record, and **"absence of record" has no agreed protocol** (open problem). That absence-of-protocol is precisely the hole 86.45 sits in.
3. **Azure Event Sourcing pattern, doc updated 2026-04-20** — the current revision now carries an explicit **"Idempotency requirements"** bullet ("Track the last processed event sequence number ... **Without idempotency, projections drift from the eventstream**") and an explicit *anti*-recommendation ("For most systems ... traditional data management is sufficient") that a 2023-era copy of the same page did not foreground.

Canonical (year-less) prior art — Idempotent Consumer / Idempotent Reader, NIST AU-5, `CrashOnAuditFail` — is **not** superseded; it supplies the fail-open/fail-closed vocabulary that the 2026 agent literature still lacks.

## Key findings

**F1 — Record at the seam, and record BEFORE the irreversible effect.** ESAA: *"the event is recorded as a fact **before** any irreversible effect, allowing for audit and containment controls"* (§3.2, https://arxiv.org/html/2602.23193). For 86.85 the "irreversible effect" is Main *acting on* the verdict (writing `evaluator_critique.md`, fixing, flipping the step). The append must precede the act, not trail it.

**F2 — The agent must NOT be the writer, and that is a feature, not a workaround.** ESAA: *"the agent does not have direct write permission to the project or the event store. Its role is to emit structured intentions and change proposals"* (§3), because *"denying direct writing ... reduces the blast radius of a compromised agent"* (§6.5). pyfinagent's constraint (a Workflow-rail Q/A has no fs access) is therefore **the recommended architecture arrived at by accident** — it should be embraced, not engineered around.

**F3 — A self-authored audit trail is weak evidence; an orchestrator-authored one is only *less* weak.** arXiv:2606.04990 §2.3: self-reported endpoints mean *"failures from task interpretation, retrieval, tool selection ... collapse into a single endpoint failure."* The mitigation in the literature is (a) an **external** writer at the boundary and (b) **verification independent of the writer** — ESAA uses SHA-256 canonical-state hashing + replay; halo-record/hermes use hash chains. **The mitigation that actually applies at pyfinagent's scale is cross-checking the ledger against an artifact the writer did not author** (the rail's own run records / the WIP artifacts), not cryptography.

> **CORRECTION 2026-08-15 (86.85 cycle-1 Q/A, C2) — applies to F4 below, to §C in
> "Application", and to the `summary` field of the envelope above.** This brief
> states three times that `run_id` is present on **33 of 35** rows. That is
> **WRONG and unreproducible**. Population = every non-blank line of
> `handoff/verdict_ledger.jsonl` at `d1c4a79d~1`; measured: total **35**, `run_id`
> key present **35**, non-empty **35**, `wf_`-prefixed **35**, non-`wf_` values
> `[]`. **35 of 35 on every predicate; no predicate yields 33.** The design
> conclusion is UNAFFECTED and in fact strengthened -- `run_id` is a *more*
> reliable dedup component than the brief claimed. Left in place with this marker
> rather than silently edited, because the number propagated into
> `contract_86.85.md` and `verdict_ledger_write.py` and the propagation path is
> the lesson.

**F4 — Dedup key = identifier of the *logical event*, assigned by the producer.** Confluent: *"A tracking ID should be a field that is unique to the logical event, such as an event key or request ID"*; cheapest enforcement is *"an upsert on the event ID as primary key."* Azure adds the write-side variant: *"the event store can reject an event that matches an existing entity identifier and event identifier."* For 86.85 the natural composite is **`(step_id, run_id)`** — `run_id` already exists in 33 of 35 rows and is the rail's own `wf_<uuid>` — with `(step_id, cycle)` as the fallback when `run_id` is absent.

**F5 — Retries and drops are the *reason* for the key, and a dropped run is a real event.** Azure: delivery is *"typically at least once ... Without idempotency, projections drift from the eventstream."* The existing ledger already encodes the correct instinct in a note: *"Rail drops recorded as NO_VERDICT because an empty return is not an absence of a cycle"* (`handoff/verdict_ledger.jsonl`, last row). That is right for *attempt* accounting and **wrong for the consecutive-CONDITIONAL rule** — which is exactly masterplan 86.45.

**F6 — Intent, not just outcome.** Azure: *"an event that records two seats were reserved is more valuable than ... remaining seats changed to 42 ... State-focused events reduce the event store to a change log that has no business meaning."* Dudycz makes the same point adversarially: *"Just recording the result of operations may not be enough"* — you need the command, the actor, and the metadata. A row carrying only `{step_id, verdict}` is a state-focused event; a row carrying `run_id`, spawn time, launch path (Workflow vs Agent-tool), and prompt/evidence identity is an intent-focused one and is what makes the ledger auditable later.

**F7 — Never mutate; correct by compensating event.** Azure: in-place rewriting *"breaks immutability and should be a last resort because **it undermines the audit trail**."* Backfill rows should therefore be *appended and labelled*, never used to rewrite history — which the current ledger does correctly (`note: "backfilled 2026-08-11 ..."`).

**F8 — Silence: the standards answer is ALERT-ALWAYS, and shut-down-or-degrade is a deliberate, separately-enumerated option.** NIST AU-5 makes **alerting mandatory** and makes the response action an organisational choice from {overwrite oldest, shut down, stop generating}; hard fail-closed is factored out into its own enhancement **AU-5(4) Shutdown on Failure**. Microsoft's shipped hard-fail-closed (`CrashOnAuditFail`) halts the machine with `STOP: C0000244`, needs a privileged manual recovery — **and is disabled by default.** The synthesis: **fail-closed on the *decision that consumes the ledger*, fail-loud everywhere, and do not fail-closed on the whole harness.**

**F9 — "Unknown" must be a third value, distinct from zero.** This is where the internal code is already ahead of the literature: `verdict_history_86_21.py` returns `None` (not `0`) for `consecutive_conditionals` when status is `LEDGER_MISSING`/`LEDGER_EMPTY`/`UNPARSEABLE` (`scripts/qa/verdict_history_86_21.py:98-99`), citing RFC 9413's "failure to parse is a different status from no verdicts" (`:26`). arXiv:2606.04990 §4.2 reaches for the same distinction — *"distinguish evidence availability from evidence use"* — and then concedes it has **no protocol** for absent records. **pyfinagent's reader has the better answer already; what is missing is a writer, not a reader.**

**F10 — Adversarial: do not adopt event sourcing wholesale.** Dudycz: if audit is *"the only driver for us, we should reevaluate our assumptions"*, and he proposes the **outbox pattern** instead. Azure concurs: *"Event sourcing is a complex pattern that introduces significant trade-offs ... For most systems ... traditional data management is sufficient"* and "doesn't have to be an all-or-nothing decision ... Apply it selectively." **This step needs one append-only JSONL with a dedup key — not projections, snapshots, rehydration, or CQRS.** Scope discipline is the finding.

## Internal code inventory (every claim carries a file:line anchor)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `handoff/verdict_ledger.jsonl` | 35 rows | The ledger itself | **LIVE but 100% hand-fed.** Measured this session: `recorded_by` is `main` on **35/35** rows. Verdicts: CONDITIONAL 18, PASS 7, FAIL 5, NO_VERDICT 5. Key union = `{step_id, cycle, verdict, run_id, recorded_by, date, recorded_at, note}`. **`recorded_at` is absent on 14/35**, and of the 21 that have it, **12 share the single timestamp `2026-08-11T08:02:38.670436+00:00`** and 7 share `08:00:54` — i.e. two bulk backfills, not per-verdict writes. Last row: 2026-08-11. |
| `scripts/qa/verdict_history_86_21.py` | 517 | The **reader** | **WORKS.** `LEDGER` pinned at `:61`. Statuses `LEDGER_MISSING` `:70`, `LEDGER_EMPTY`, `UNPARSEABLE` `:67-71`. `consecutive_conditionals` returns **`None`, never `0`**, for all three (`:98-99`) with the docstring rule *"A caller that treats None as 0 has reintroduced the defect"* (`:87-89`); `would_auto_fail` propagates `None` (`:109-114`). Zero-byte ledger is explicitly a **truncation signal, "Fails CLOSED"** (`:125-135`). Header records the origin defect: the rule told Q/A to grep `handoff/harness_log.md`, which **"grep returned 0 every time"** (`:6-13`). |
| `.claude/workflows/qa-verdict.js` | — | Q/A launch rail + `enforceEscalation` | **The structural constraint, stated in-source.** `:301-303`: *"PURE BY NECESSITY AND BY DESIGN. The Workflow runtime has no filesystem access, so this cannot read handoff/verdict_ledger.jsonl -- the sequence must arrive as data via `args.verdict_sequence`."* `:311-312`: *"FAILS CLOSED. An absent or unusable sequence yields `null`, never `0`."* `:185` documents the reader CLI as the intended producer of that sequence. So the **consumer side is already correct and already fail-closed**; only the writer is missing. |
| `.claude/workflows/research-gate.js` | — | Sibling rail | `:52-55` — *"NO `import fs` / `import path` HERE, AND THAT IS NOT AN OVERSIGHT ... makes the script UNLAUNCHABLE: SyntaxError."* Confirms the no-fs constraint is a **runtime property of Workflow scripts**, not a policy choice. |
| `.claude/settings.json` (hooks) | — | Instrumentation surface | PostToolUse matchers in force are **`Bash`, `Write` (x4), `Edit` (x4)** only. `SubagentStop` exists but its body is a literal no-op echo (`{"ok": true, "reason": "subagent stopped"}`). **No hook anywhere reads `tool_response`** — verified by `grep -l tool_response .claude/hooks/*.sh` → no matches. `pre-tool-use-danger.sh:19,32` reads `tool_name` from `CLAUDE_TOOL_NAME` / stdin JSON. |
| `handoff/audit/pre_tool_use_audit.jsonl` | 160K+ rows | Evidence that the seam is observable | **`Workflow` appears 600 times and `Agent` 1,225 times as `tool_name`.** So a `PostToolUse` matcher of `Workflow` or `Agent` **would fire** — the tool names are real and matched. This is the single most important internal finding for the hook-vs-explicit-call question. |
| `scripts/qa/qa_wip.py` | 635 | The *other* durable trail | Independent WIP artifacts under `.claude/agent-memory/qa/verdicts/verdict_wip_<sid>__<stamp>.md` (`:158-179`), plus a hidden loss ledger `.attempt_lost_<sid>.json` (`:145,:227-253`). **This is a second, writer-independent record of the same events** — the cross-check F3 asks for. `_attempt_counts` `:378-395` already warns it "cannot observe whether that happened". |
| `scripts/harness/attempt_budget.py` | 331 | Cumulative budget | **NO caller, NO persistence** (masterplan 86.71). Would be the natural second consumer of the ledger. |
| `.claude/agents/qa.md` | — | 3rd-CONDITIONAL rule | Now counts prior spawns via `scripts/qa/qa_wip.py`, **not** via the ledger — so today the rule's input is the WIP artifacts, and the ledger is orphaned on the consumer side too. |
| `.claude/masterplan.json` | — | Related open steps | **86.21** (counter blind to in-flight steps, `status=pending`), **86.45** (a rail drop recorded as `NO_VERDICT` *"silently clears a real escalation"*, `pending`), **86.71** (budget has no caller, `pending`), **86.79** (`records_retained` counts the current spawn, `pending`), **86.85** (this step: *"the verdict ledger is never written for the step being evaluated, so the 3rd-CONDITIONAL auto-FAIL rule has no input"*, `pending`). |

**Internal files inspected: 10.**

## Consensus vs debate (external)

**Consensus.** (a) The agent must not write its own audit record — ESAA §3/§6.5 and arXiv:2606.04990 §2/§4 agree, from security and from evidentiary angles respectively. (b) The record is written by the deterministic orchestrator at the seam, before the irreversible effect (ESAA §3.2; Azure workflow steps 4-5). (c) At-least-once delivery is the norm, so a producer-assigned logical-event key plus skip-on-seen is mandatory (Confluent; Azure "Idempotency requirements"). (d) Corrections are appends, never rewrites (Azure "Versioning events").

**Debate 1 — is a ledger worth the complexity?** Dudycz says an audit requirement alone is *not* sufficient justification and points to the outbox pattern; Azure's own 2026 revision opens with "For most systems ... traditional data management is sufficient." **Resolution for 86.85:** they are arguing against *event sourcing as the system of record*. This step needs an append-only observation log with a dedup key — the cheap end of the spectrum both sources explicitly endorse ("Apply it selectively").

**Debate 2 — write-at-the-seam vs reconcile-later.** arXiv:2606.04990 §3.4 refuses to pick, calling pre-execution / runtime / post-hoc *complementary* and risk-tiered: *"A low-risk question-answering agent may only need ... post-hoc audit. A tool-using financial ... agent may require runtime provenance checking."* ESAA picks runtime unconditionally. **Resolution:** by that paper's own criterion pyfinagent's Q/A ledger is the high-risk case — its record *is* the termination input for a loop that spends real money-equivalent tokens — so runtime/at-the-seam wins, with reconcile-later retained only as a labelled repair (F7).

**Debate 3 — how hard to fail on silence.** NIST AU-5 mandates *alerting* and makes shutdown one enumerated option among three; AU-5(4) factors hard shutdown into a separate enhancement; Microsoft ships hard fail-closed **disabled by default** because the recovery is a manual privileged bluescreen recovery.

## Pitfalls (from literature) mapped to this step

1. **Silent zero.** Azure: without idempotency "projections drift"; the reader's own docstring: *"A caller that treats None as 0 has reintroduced the defect"* (`verdict_history_86_21.py:87-89`). A writer that fails silently produces exactly the `LEDGER_EMPTY`/absent-row state the reader is built to refuse — but only if the *writer's* failure is also loud.
2. **Recording a non-event as an event.** The ledger already contains 5 `NO_VERDICT` rows justified by *"an empty return is not an absence of a cycle."* Correct for attempt accounting, wrong for the consecutive rule (86.45). Literature analogue: Azure's intent-vs-state distinction — `NO_VERDICT` is an *attempt* fact, not a *verdict* fact, and the two belong in different projections or need an explicit `grades: true|false` discriminator.
3. **Self-authored trail.** arXiv:2606.04990 §2.3. Mitigation available here without cryptography: cross-check against `qa_wip.py`'s independent artifacts and the rail's run records.
4. **Backfill masquerading as history.** 12 rows sharing one microsecond timestamp is visibly a backfill *only because* `recorded_at` exists. Any writer must stamp write-time separately from event-time, or the two become indistinguishable (Azure "Versioning events": in-place migration "undermines the audit trail").
5. **Fail-closed on the wrong scope.** `CrashOnAuditFail` halts the whole machine. Do not let a missing ledger row halt the harness; let it halt the *decision that consumes the row*.
6. **Dedup at the reader only.** Confluent: *"must enable EOS at all stages of the pipeline, not just on the reader."* Dedup on append (upsert-by-key semantics over the JSONL) as well as on read.

## Application to pyfinagent (external findings → internal anchors)

**A. The writer is the only missing part, and its location is forced.** The Q/A cannot write (`qa-verdict.js:301-303`, `research-gate.js:52-55` — a `SyntaxError`, not a policy). The reader is correct and fail-closed (`verdict_history_86_21.py:98-99`). The consumer is correct and fail-closed (`qa-verdict.js:311-312`). Per F2 this is the ESAA topology **already**; the only agent with both the verdict and fs access is Main. So the writer is Main-at-the-seam — matching ESAA's "deterministic orchestrator validates, appends, applies effects."

**B. Hook-based vs explicit-call — the decisive measurement.** `Workflow` (600) and `Agent` (1,225) are real `tool_name` values in `handoff/audit/pre_tool_use_audit.jsonl`, so a `PostToolUse` matcher on them **would fire**. That is the automatic, un-forgettable instrumentation the step wants (write-at-the-seam without relying on Main remembering). **But the open risk is verified, not assumed:** *no hook in this repo reads `tool_response`* (`grep -l tool_response .claude/hooks/*.sh` → no matches), so whether a PostToolUse hook can actually see the returned verdict payload is **unproven here and must be measured before it is designed around**. Existing hooks read `tool_input` only. If `tool_response` is unavailable, the hook can still fire a **loud alarm on an un-recorded verdict** (NIST AU-5(2) real-time alert) even if it cannot author the row — a strictly better position than today's silence. Precedent for the fallback exists: `auto-commit-and-push.sh` + `live_check_gate.py` already implement "hold the action until the artifact exists."

**C. Dedup key.** `(step_id, run_id)` — `run_id` is present on 33/35 existing rows as the rail's own `wf_<uuid>`, satisfying Confluent's "unique to the logical event ... request ID". Fall back to `(step_id, cycle)` where `run_id` is null. Append-side check = refuse a row whose key already exists (Azure: "the event store can reject an event that matches an existing entity identifier and event identifier"); read-side dedup as well (Confluent: EOS at all stages).

**D. Retries and drops.** A retried spawn is a **new** `run_id` → a new row, correctly. A *replayed/duplicate transcription* of the same `run_id` → deduped. A dropped run is a real attempt: keep the `NO_VERDICT` row (it is the input 86.71's budget needs) but give it an explicit non-grading discriminator so it cannot reset the consecutive-CONDITIONAL run — **this is exactly 86.45 and 86.85 should not silently re-decide it.**

**E. Behaviour when the ledger is silent.** Synthesis of NIST AU-5 + AU-5(4) + `CrashOnAuditFail`'s disabled-by-default + the reader's existing `None`: **(1) always alert loudly** (AU-5 base, mandatory); **(2) fail-closed on the escalation decision** — an unknown history must not be read as "0 consecutive CONDITIONALs", which is what `None`-not-`0` already guarantees at `verdict_history_86_21.py:98-99` and `qa-verdict.js:311-312`; **(3) do NOT fail-closed on the harness as a whole** — no `STOP: C0000244` for the whole loop. In practice: a silent ledger should make the Q/A rail **escalate to the operator** rather than either auto-FAIL or auto-proceed, which is the same disposition CLAUDE.md's F1b already prescribes at budget exhaustion ("ESCALATES TO THE OPERATOR ... Auto-pass on exhaustion is forbidden").

**F. Scope discipline (F10).** Do not build projections, snapshots, rehydration, or CQRS. One append-only JSONL + a dedup key + a writer at the seam + a loud alarm. Azure and Dudycz both explicitly warn against more.

**G. Boundary with adjacent open steps.** 86.85 supplies the *writer*; **86.45** owns whether `NO_VERDICT` grades; **86.71** owns the cumulative budget that would consume the ledger; **86.79** owns the `records_retained` off-by-one in the parallel `qa_wip.py` trail; **86.21** owns the in-flight blindness of the counter. Main should scope the contract to the writer + its alarm and cite the others as out-of-scope.

**Note on brief length:** this exceeds the `moderate` <=700-word guideline. The caller enumerated six distinct sub-questions; each is answered with per-claim citations. The tier's source floor (>=5) is met at 8; the overrun is in analysis depth, not padding.

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] **>=5 authoritative external sources READ IN FULL via WebFetch** — 8. Tier mix: 2 peer-review-tier preprints (arXiv 2606.04990, 2602.23193), 4 official docs (Microsoft Azure ES pattern, Microsoft `CrashOnAuditFail`, Confluent pattern catalog, NIST SP 800-53 AU-5 rendering), 1 Anthropic engineering doc, 1 authoritative practitioner blog (Dudycz/Event Store). **No community-tier source is counted toward the floor.**
- [x] **10+ unique URLs total** — 23 (8 full + 15 snippet-only).
- [x] **Recency scan (last 2 years) performed + reported** — dedicated section above; 3 in-window findings that materially change the design, plus an explicit statement that the year-less canonical prior art is not superseded.
- [x] **Full pages read (not abstracts)** — every read-in-full row was a whole-page `WebFetch`. Both arXiv papers used the `arxiv.org/html/` route per `.claude/rules/research-gate.md` §PDF-and-arXiv; **no `arxiv.org/pdf/` URL was fetched.** One disclosure: source #5 (csf.tools) is a structured rendering of NIST AU-5 whose Discussion section is summarised rather than verbatim — the base control statement, the enumerated actions, and the enhancement titles are verbatim.
- [x] **file:line anchors for every internal claim** — see the Internal code inventory; every row carries either a line anchor or an explicitly-stated measurement command.

Soft checks:
- [x] Internal exploration covered the caller's stated scope: ledger, reader, both Workflow scripts, hooks + settings matchers, `qa_wip.py`, `attempt_budget.py`, masterplan 86.21/86.45/86.71/86.79/86.85. **Gap noted:** `.claude/agents/qa.md` was inspected only via the caller's summary and grep of related scripts, not re-read end to end.
- [x] Contradictions / consensus noted — three explicit debates recorded, including two sources adversarial to the obvious design.
- [x] Claims cited per-claim with URL, not only in a footer table.

**Open question this brief could NOT close (flagged for PLAN, do not assume either way):** whether a `PostToolUse` hook matching `Workflow`/`Agent` receives the tool's *return payload*. The tool names are confirmed matchable (600 + 1,225 observations), but **no hook in this repo reads `tool_response`**, so this is unmeasured. It should be measured before the contract commits to a hook-authored row rather than a hook-raised alarm.
