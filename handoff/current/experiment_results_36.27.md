# Experiment results — phase-36.27

**Step:** `36.27` — the Researcher gate had no Workflow rail. **Cycle 187**, 2026-08-09.

---

## 1. What was built

| File | Change |
|---|---|
| `.claude/workflows/research-gate.js` | **NEW.** Two-stage rail: (1) researcher returns a schema-constrained envelope; (2) an independent cheap agent reads the brief on disk and reports which claimed URLs are actually in it. `enforceGate()` then RECOMPUTES `gate_passed` from both. |
| `scripts/qa/verify_research_gate_workflow.mjs` | **NEW.** 40 checks incl. a 6-mutant matrix. Drives the REAL exported `enforceGate`, not a copy. |
| `CLAUDE.md` | Launch section now a two-row table naming both scripts, with the schema-vs-JS-enforcement rationale. |
| `.claude/rules/research-gate.md` | New "Launch: the Workflow rail is FIRST-CLASS" section. |
| `.claude/masterplan.json` | `36.27` title `[P2 --]` → `[P1 --]` (its `priority` field was already `P1`). **Name only — no `verification` block touched.** |

No production/backend code. No flag, threshold or gate.

---

## 2. The design decision, and the measurements that forced it

**The floors are asserted in JS, not declared in the schema.** Two reasons, both
from the research gate:

1. **Anthropic structured outputs strips `minimum`/`maximum`/`minLength`** from
   the wire schema and caps `minItems` at 1. `>=5 sources` and `>=10 URLs` are
   therefore *not expressible* as schema constraints.
2. **Schema conformance is structural only.** *The Constraint Tax* (2026-05)
   measured wrong-but-schema-valid output rising **49.5% → 88.9%** under
   constrained decoding. **EviBound** measured **100%** false completion claims
   from prompt-level self-reflection alone, falling to **0%** only with a
   post-hoc gate that queries the artifact store. A schema can force
   `external_sources_read_in_full` to be an integer; it cannot make it **true**.

⇒ `gate_passed` is **recomputed**, never trusted; the agent's value is kept as
`agent_self_reported_gate_passed` and any disagreement is reported, with the
enforced value governing. `gate_passed` is a **plain boolean** — `const: true`
would make honest failure *unrepresentable*, forcing the agent to lie or hang.

---

## 3. Three ways the immutable verification command reports green on a dead script

The step's command is `node --check … && ls …`. **All three of these passed it.**

1. It reaches **criterion 1 only** (parses + exists) — known from the contract.
2. `import fs from 'node:fs'` — valid ESM, `node --check` green, and the
   Workflow runtime **rejects it**:
   `SyntaxError: Unexpected identifier 'fs'. import call expects one or two arguments.`
3. A trailing `export { … }` list — valid ESM, `node --check` green, runtime
   **rejects it**: `SyntaxError: Unexpected keyword 'export'` (only the leading
   `export const meta` is accepted).

**Both were found by trying to LAUNCH, not by reading.** Only **criterion 2 —
the live spawn** — catches this class. Without it I would have shipped a script
that passed its own verification command and could never run. That is the
strongest argument for criterion 2 being written the way it is, and I am
recording it because I did not anticipate it.

**Both breakages improved the design:**

- No filesystem access ⇒ the artifact cross-check moved **out** to an
  independent stage-2 agent. **The researcher no longer attests to its own
  brief** — which is exactly the EviBound finding, and stronger than what I
  originally wrote.
- No export list ⇒ `enforceGate` had to become **pure** (envelope + verification
  in, verdict out). Pure means it is mutation-testable **without spawning
  anything**. The checker appends its own export to a stripped copy.

---

## 4. Verification — verbatim

### 4.1 Immutable command (criterion 1)

```
$ source .venv/bin/activate && node --check .claude/workflows/research-gate.js && ls .claude/workflows/research-gate.js
.claude/workflows/research-gate.js
exit=0
```

### 4.2 Re-runnable checker (criteria 3, 4, 6)

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 40 passed, 0 failed
```

Sections: `[2]` every floor rejects a short-of-floor return (incl. **audit-class
+ dry but 4 sources STILL rejected** — the floor is never lowered); `[3]`
null/undefined/`{}`/string/array all ⇒ `gate_passed:false`; `[4]` artifact
cross-check (missing brief, empty brief, a claimed-but-absent URL, an
over-claim); `[5]` **absent stage-2 verification FAILS CLOSED**; `[6]` the agent
does not grade itself; `[7]` the 6-mutant matrix; `[8]` structural.

**Mutation matrix (criterion 6) — 6/6 killed:**

| Mutant | Killed by |
|---|---|
| `FLOOR_SOURCES 5 → 1` | let a bad envelope through |
| `FLOOR_URLS 10 → 1` | let a bad envelope through |
| recency check removed | let a bad envelope through |
| audit-class dry check removed | let a bad envelope through |
| over-claim check removed | let a bad envelope through |
| fail-closed on absent verification removed | **threw** — `Cannot read properties of null (reading 'brief_exists')` |

The last one is counted as a kill and the reason is stated in the checker: a
throw means the removed guard was the only thing standing between production
and a crash on that input.

### 4.3 LIVE SPAWN (criterion 2) — run `wf_9880694c-d30`

Not a synthetic exercise: this **is** step 86.1's real research gate.
2 agents, 40 tool uses, 191,253 tokens, 687s.

```json
{"step_id":"86.1","gate_passed":true,"agent_self_reported_gate_passed":true,
 "self_report_disagreed":false,"violations":[],
 "checks":["sources_floor_ok: 8 >= 5","urls_floor_ok: 44 >= 10","recency_scan_ok",
           "not_audit_class: coverage.dry informational only",
           "listed_sources_consistent: 8 >= 8",
           "brief_on_disk_ok: handoff/current/research_brief_86.1.md (36790 chars, independently read)",
           "all_8_claimed_sources_present_in_brief"],
 "brief_verification":{"brief_exists":true,"brief_non_empty":true,"char_count":36790,
                       "urls_checked":8,"urls_present":8,"urls_missing":[]}}
```

**Criterion 2 requires BOTH, and both hold:** a schema-valid envelope was
returned **AND** the brief is on disk (36,998 bytes at rest). **Write-first was
observed live**, not inferred — the brief measured 54 lines mid-run while stage 1
was still working.

**Main re-verified stage 2 independently** rather than trusting it:

```
claimed=8 listed=8 missing_from_brief=0
MAIN'S INDEPENDENT RE-CHECK AGREES WITH STAGE 2
```

And spot-checked the researcher's headline measurement:

```
{'pause': 44, 'resume': 10, 'sod_snapshot': 8}   # 44+10+8 = 62 = the whole file
peak rows in LIVE journal: 0
```

Which corroborates its finding that a `peak_reset` row written today would win
the `ts` merge-sort outright — there is nothing later to override it.

### 4.4 Criterion 5 — docs match the mechanism

`CLAUDE.md` launch section is now a two-row table naming `qa-verdict.js` and
`research-gate.js`, carrying the strip/`const`-trap/recompute rationale and the
checker command. `.claude/rules/research-gate.md` gains a "Launch" section
stating the floors are script-enforced and why. **The Agent-tool path remains
documented as the fallback in both.**

---

## 5. Two defects the checker found — one in itself, one in production

1. **A JS default parameter fires on an explicitly-passed `undefined`.** My
   wrapper used `verificationOverride = NOT_SUPPLIED` to distinguish "omitted"
   from "explicitly undefined" — it cannot. The `verification undefined` case
   was silently computing a *real* verification and passing. Fixed by calling
   `enforceGate` directly for those cases. **Harness bug, caught by the harness.**
2. **`typeof [] === 'object'`**, so an array slipped the stage-2 guard and was
   read as a verification object with all-undefined fields — still failing
   closed, but via the wrong branch with a misleading message. **Production
   fix:** the guard now mirrors the envelope's `Array.isArray` check.

---

## 6. Honest limits

- **A newly added workflow is NOT dispatchable by name until session restart.**
  `Workflow({name:'research-gate'})` returned *"not found. Available:
  deep-research, code-review, harness-self-audit, probe-qa-tool-surface,
  qa-verdict"*. The registry snapshots at session start — the same class as the
  `.claude/agents/*.md` roster caveat already in CLAUDE.md. In-session, use
  `{scriptPath: ...}`. **The next session must verify the name resolves**, and
  until then a caller following the docs verbatim will get "not found". This is
  a real usability gap in what shipped today.
- The stage-2 verifier is an **LLM agent**, not a deterministic file read — the
  runtime gives the script no filesystem access, so this is the strongest
  available in-rail check. Main re-verified it by hand here (§4.3) and agreed,
  but that hand-check is not automatic on every future run.
- The URL cross-check is a **substring** test. A brief that lists a URL without
  having read it in full would still pass. It detects *fabricated* sources, not
  *shallow* ones.
- `coverage.dry` for audit-class steps is enforced, but **no audit-class step
  was run live** — that path is covered by the checker only.
- The floors are enforced **after** the agent returns, so a short-of-floor run
  still costs its tokens. This gate catches over-claims; it does not prevent
  wasted work.
