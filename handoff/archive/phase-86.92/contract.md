# Contract — phase-86.92

**Written BEFORE any GENERATE work.** Research gate cleared first; see below.

---

## Step

`86.92` (P1) — *"the args-boundary checker has been RED since phase-86.37 for an
unrelated reason — `verify_workflow_args_boundary.mjs` asserts a 'healthy run'
against a phase-86.17-era brief that predates the born-inert `brief_status`
marker, so a gate covering BOTH Layer-3 scripts has been failing unnoticed"*

**The step's own title is factually wrong on both counts, and this contract says
so up front** (see Hypothesis). The title is not editable — only the
verification criteria are immutable and they are quoted verbatim below — but a
plan that inherited the title's premise would fix the wrong artifact.

---

## Research gate — PASSED (enforced, not self-reported)

Launched by `scriptPath` per rail R7: `.claude/workflows/research-gate.js`,
run `wf_2ee79ffe-d4f`, 2 agents, 190,482 tokens, 540s.

```
gate_passed: true          agent_self_reported_gate_passed: true
self_report_disagreed: false               violations: []
sources_floor_ok: 7 >= 5                   urls_floor_ok: 22 >= 10
recency_scan_ok                            listed_sources_consistent: 7 >= 7
brief_on_disk_ok: handoff/current/research_brief_86.92.md (28140 chars, independently read)
brief_status_in_brief: COMPLETE            all_7_claimed_sources_present_in_brief
urls_collected_corroborated: 22 <= 22 distinct URLs in the brief
```

Brief: `handoff/current/research_brief_86.92.md`. Sources read in full: Fowler
*Eradicating Non-Determinism in Tests*; Jest snapshot-testing docs; Google SRE
Book *Monitoring Distributed Systems*; arXiv:2605.06125v1; arXiv:2511.21382v1;
PMC2821100; Eftimov *Testing in Go: Golden Files*.

**Disclosed gaps** (carried forward from the brief, not hidden): three sources
could not be fetched — Wiley (402), xunitpatterns.com (ECONNREFUSED), Google
Testing Blog (body not served). The canonical *Fragile Fixture* reference is
therefore snippet-only and **no claim in this contract rests on it**.

The researcher reached the same cause I did **independently**, from a different
starting point, and its summary is quoted in `experiment_results_86.92.md`.

---

## Hypothesis (localised by execution, before any code was written)

The RED has **nothing to do with `research_brief_86.17.md`**, and nothing to do
with phase-86.37.

1. `enforceGate` is a **pure** function of `(env, verification, opts)`. It never
   opens the brief; `env.brief_path` is a string it interpolates into the
   violation message. Measured: 0 `fs`/`process` uses in its body, and
   `verify_research_gate_workflow.mjs` already asserts this independently.
   **Control:** pointing `brief_path` at a nonexistent file produces the
   *byte-identical* three violations.
2. The actual stale fixture is the checker's **own hand-written `verification`
   object literal** at `verify_workflow_args_boundary.mjs:179`, cloned at `:319`.
   It supplies 4 of the 9 fields `BRIEF_VERIFICATION_SCHEMA.required` has
   declared since 86.6/86.37, and 4 of the 7 that `enforceGate` actually reads.
   (An earlier revision of this line said `86.28/86.37`. Re-derived per field:
   `recency_section_present` and `distinct_urls_in_brief` both land in `cad38647`,
   subject *phase-86.6*; `brief_status_in_brief` in `d3bb1dfb`, *phase-86.37*. The
   `86.28` came from `research-gate.js:715`, whose in-code comment labels that block
   `phase-86.28` — a discrepancy inside the gate's own source, recorded rather than
   silently resolved. Nothing load-bearing depends on it.)
   Missing: `brief_status_in_brief`, `recency_section_present`,
   `distinct_urls_in_brief`.
3. Breaking commit is **`cad38647` (phase-86.6, 2026-08-10 08:51:11 +0200)**, not
   `d3bb1dfb` (phase-86.37). Bisected by running the checker in real worktrees.

All three failing assertions share this single cause.

---

## Immutable success criteria — copied VERBATIM from `.claude/masterplan.json`

1. "the RED is REPRODUCED first and its cause is localised by execution -- which assertion, driving which fixture, failing on which enforceGate rule -- rather than inferred from the message text"
2. "the `-1 distinct URLs` figure is explained: a count of -1 is not a plausible corroboration result, so either the sentinel is deliberate and documented, or it is a second defect and is filed as such"
3. "the fix does NOT weaken enforceGate to accommodate a stale fixture: if the fixture is the problem, the fixture is replaced or pinned; loosening the brief_status / recency / URL rules to make an old artifact pass is a scope breach and fails this criterion"
4. "the fixture choice is made durable: state why the chosen brief cannot rot the same way when the next envelope rule lands, or make the fixture synthetic and owned by the checker"
5. "after the fix the checker exits 0, AND its mutation cells are shown still to KILL -- a green checker whose mutants now survive is worse than a red one"
6. "the blast radius is stated: how long the gate has been red (from git history of the two files), and whether any step closed in that window relied on it as evidence"
7. "verdict semantics are UNCHANGED: nothing here may turn a non-PASS into a PASS"

Immutable command:
`bash -c 'node --check scripts/qa/verify_workflow_args_boundary.mjs && echo parses'`

**Disclosed weakness of the immutable command:** it is a *parse* check. It
reaches criterion 1 only and **cannot fail on this defect** — it was green
throughout the six days the gate was dead. The real evidence therefore lives in
`live_check_86.92.md`, which carries the actual `node scripts/qa/verify_workflow_args_boundary.mjs`
run. This is disclosed, not worked around.

---

## Plan

**P0 — reproduce and diagnose (criterion 1).** DONE before this contract; the
measured output is already committed in `live_check_86.92.md` §A–D (commit
`687109bb`).

**P1 — replace the fixture, do not touch the gate (criteria 3, 7).**
`.claude/workflows/research-gate.js`, `.claude/workflows/qa-verdict.js` and
`.claude/agents/qa.md` are **not edited** — both by rail R5 and by criterion 3.
The only file changed is the checker. Method: the checker already appends its
own `export` line to a stripped copy of the script before importing it, so it
can reach `BRIEF_VERIFICATION_SCHEMA` **without modifying the script**. Verified
reachable.

**P2 — make the fixture synthetic and owned by the checker (criterion 4).**
Replace both hand-written literals with one `healthyVerification()` factory
built from a declared `HEALTHY_VERIFICATION_VALUES` map, plus **two** new
assertions that make the next rot self-announcing in ONE named place:
- every field in `BRIEF_VERIFICATION_SCHEMA.required` has a value in the map
  (the *declared* contract);
- every `verification.<field>` that `enforceGate` actually reads is supplied
  (the *consumed* set), scanned with comments stripped.

The two directions catch different regressions, and a field read but never
declared is itself a finding. The extractor gets a **positive control** proving
it is live (a bogus field injected into a comment must be rejected, and the
un-stripped scan must pick it up — otherwise the control is dead).

**P3 — prove the mutants still kill (criterion 5).** Show `[4]
drop-blind-violation` KILLS again — the rot had disabled it, and it currently
reports FAIL whether the guard is present or absent, i.e. it cannot
discriminate. Add a mutation cell for the NEW guard too: a guard that cannot
fail does not count.

**P4 — file the `-1` message defect as its own step (criterion 2).** The
sentinel is deliberate and documented (`research-gate.js:632`); the *rendered
sentence* states a measurement of the brief that was never taken. That is a
separate defect in a file rail R5 forbids editing tonight, so it is FILED, in
the same turn, not fixed here.

**P5 — state the blast radius (criterion 6).** Already measured: red
~6d12h; no step closed on a false green; 86.23 blocked; 86.17's `done` no longer
reproduces.

---

## Non-goals

- Fixing `research_brief_86.17.md`. It is not the cause; editing it would leave
  the checker RED (measured) and would be cargo-cult.
- Any edit to `enforceGate`, `qa-verdict.js`, or `qa.md` (rail R5, criterion 3).
- Fixing the `-1` message text (filed, not fixed — see P4).
- Closing 86.23. It is unblocked by this work but is its own step.

---

## References

- `handoff/current/research_brief_86.92.md` — research gate brief (gate PASSED)
- `handoff/current/live_check_86.92.md` §A–D — reproduction, falsification, bisect
- `.claude/rules/research-gate.md` — the floors this gate enforced
- Bisect commits: `089726f9` (green) → `cad38647` (red, phase-86.6)
