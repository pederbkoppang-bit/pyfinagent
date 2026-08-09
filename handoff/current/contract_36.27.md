# Contract — phase-36.27

**Step:** `36.27` — the Researcher gate has no Workflow rail, so half the
doctrine has no mechanism. **Cycle 187**, 2026-08-09.

> Note: the step's `name` opens `[P2 --]` but its `priority` field is **`P1`**.
> The field is authoritative; the title text is stale. Corrected in this step
> (a `name` edit, not a `verification` edit).

---

## 1. Research-gate summary

**Brief:** `handoff/current/research_brief_36.27.md` · **`gate_passed: true`**
(8 read in full, 25 URLs, recency scan performed, 10 internal files).

**Self-referential disclosure, again on the fallback:** this gate ALSO ran via
the Agent-tool fallback, because the mechanism this step builds does not exist
yet. That is the defect, observed one last time. After this step it stops.

### The four findings that decide the design

1. **The numeric floors are NOT schema-enforceable.** Anthropic structured
   outputs **strips `minimum`/`maximum`/`minLength`** from the wire schema and
   caps `minItems` at 1. So `>=5 sources` and `>=10 URLs` **cannot** be
   expressed in the schema and **must be asserted in JS**.
   *Main's note: I do not need this claim to be true in order to be safe — I
   assert in JS unconditionally. If the schema also happens to enforce it, that
   is redundancy, not reliance. Designing the other way round would have been
   the risk.*
2. **`const: true` on a gate field is a trap.** It makes an honest `false`
   *unrepresentable*, so the agent must either lie or fail to return. Never
   constrain `gate_passed` to `true`.
3. **Schema conformance is structural only.** *The Constraint Tax* (2026-05)
   measured wrong-but-schema-valid output rising **49.5% → 88.9%** under
   constraint. **EviBound** measured **100%** false completion claims from
   prompt-level self-reflection alone, dropping to **0%** only with a *post-hoc
   gate that queries the artifact store*. A schema can force
   `external_sources_read_in_full` to be an integer; it cannot make it true.
4. ⇒ **The script must RECOMPUTE `gate_passed`, never trust the returned
   value**, and must cross-check the self-report against the artifact on disk.

---

## 2. Immutable success criteria — verbatim from `.claude/masterplan.json`

1. `.claude/workflows/research-gate.js` exists, passes `node --check`, and declares a schema covering the full research envelope
2. A live spawn through it returns a schema-valid envelope AND leaves the incremental brief on disk -- both, not either
3. Every research-gate floor is enforced (>=5 read-in-full, >=10 URLs, recency scan, audit-class coverage.dry) -- a test or the script itself proves a short-of-floor return is rejected, not rounded up
4. An EMPTY return is treated as a failed gate, never as gate_passed -- proved deliberately
5. CLAUDE.md and .claude/rules/research-gate.md are updated so the documented launch matches the mechanism that exists; the Agent-tool path stays documented as the fallback
6. MUTATION-TEST: weakening any floor in the script must fail the check that enforces it

**Verification command (immutable):**
```
source .venv/bin/activate && node --check .claude/workflows/research-gate.js && ls .claude/workflows/research-gate.js
```

> **Stated plainly: that command is weak.** It proves the file parses and
> exists — criterion 1 only. Criteria 2, 3, 4 and 6 are *not* reachable by it.
> I will not amend it (immutable), and I will not let it stand as the evidence.
> A separate re-runnable checker carries 3/4/6, and criterion 2 needs a real
> spawn. Both go in `experiment_results` with verbatim output.

---

## 3. Design

### 3.1 Schema (what it CAN carry)

`tier` (enum), `external_sources_read_in_full`, `snippet_only_sources`,
`urls_collected`, `internal_files_inspected` (integers),
`recency_scan_performed` (boolean), `gate_passed` (boolean — **never** `const`),
`brief_path`, `sources_read_in_full` (array of URL strings — needed for the
cross-check), `coverage` object (`audit_class`, `rounds`, `dry_rounds`,
`K_required`, `new_findings_last_round`, `dry`), `summary`.
`additionalProperties: false`, everything `required`.

### 3.2 Floors asserted in JS (the load-bearing half)

- `external_sources_read_in_full >= 5`
- `urls_collected >= 10`
- `recency_scan_performed === true`
- `coverage.audit_class === true` ⇒ `coverage.dry === true`
- **Artifact cross-check (the EviBound lesson):** the brief must exist on disk,
  be non-empty, and **every URL** in `sources_read_in_full` must actually appear
  in the brief text. A self-reported count that the artifact does not
  corroborate fails the gate.
- `sources_read_in_full.length` must be **>= the claimed count** — a claim of 8
  backed by 3 listed URLs is an over-claim.
- **`gate_passed` is RECOMPUTED** from the above. The agent's own value is
  recorded as `agent_self_reported_gate_passed` and, if the two disagree, that
  disagreement is itself reported.

### 3.3 Empty / errored return (criterion 4)

`agent()` returns `null` on a terminal error. Any null/undefined/non-object
return, or a missing envelope, yields `gate_passed: false` with reason
`empty_or_errored_return`. **Never `true`.** Proven deliberately, not asserted.

### 3.4 Carry-overs from `qa-verdict.js` — the rider traps

- **R1** — no internal fix→re-grade loop. Return the envelope and STOP.
- **R4** — `model: 'opus'`; do not route off Opus.
- **R11** — no Monitor / transcript-mtime watchdog.
- Three-shape `args` parse (object / JSON string / absent).
- STEP 0 binding runtime read of `.claude/agents/researcher.md` from disk, so a
  `researcher.md` edit is live immediately on this rail.
- `agentType: 'researcher'` — it gets `Write` via its `memory: project`
  injection (measured), and the `qa-write-guard` hook matches
  `agent_type == 'qa'` only, so it does not block. **The researcher legitimately
  NEEDS Write** — it must write the brief. This is the one place where the Q/A
  precedent (tool-surface *restriction*) deliberately does not transfer.
- The prompt must state the **exact `brief_path` the script will later verify**,
  so write-first and the artifact cross-check refer to the same file.

## 4. Non-goals

- Not changing `qa-verdict.js`.
- Not weakening any floor to simplify the schema (banned by the step).
- Not making the Agent-tool path unavailable — it stays the documented fallback.
- No production/backend code. No flag, threshold or gate touched.

## 5. References

`handoff/current/research_brief_36.27.md` · `.claude/workflows/qa-verdict.js` ·
`.claude/agents/researcher.md` · `.claude/rules/research-gate.md` ·
`CLAUDE.md` (harness protocol) · masterplan `36.27`
