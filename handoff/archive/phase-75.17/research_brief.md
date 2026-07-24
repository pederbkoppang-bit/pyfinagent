# Research Brief — Step 75.17: Absent-path verification family (triage + repair)

**Tier:** moderate (audit-class = TRUE -> adaptive coverage gate active)
**Researcher:** Layer-3 (this session)
**Date:** 2026-07-24
**Status:** COMPLETE -- gate PASSED (audit-class; coverage.dry=true)

## Objective

Step 75.17 audits `.claude/masterplan.json` for `verification` blocks whose
`command` / `success_criteria` / `live_check` reference **filesystem paths that
do not exist on disk**. Such commands are unrunnable -> their PASS is
unreproducible -> the audit trail rots. The step is TRIAGE-first: build ONE sweep
that handles all four verification shapes (674 dict / 126 str / 13 list / 24
None), resolves frontend-relative + URL fragments correctly, and classifies each
hit as (i) genuine never-existed name mismatch / (ii) artifact retired by a later
step / (iii) false positive. Then repair per class via the `superseded_record`
house convention — NEVER by amending an immutable criterion.

## Method

- TWO independent census derivations (different extraction strategies),
  reconciled by symmetric difference (qa.md §4b).
- Adaptive coverage: loop-until-dry on the census (K=2 dry rounds).
- Repair prescription per class + mutation matrix for the sweep tool.

---

## Internal audit — the definitive census (the core deliverable)

### Denominator moved since 75.2.1

75.2.1 (2026-07-20) swept **837 steps** (shape census 674 dict / 126 str / 13 list
/ 24 None). The plan has since GROWN to **883 flat steps** (`phases[].steps[]`, no
nesting) — current shape census **720 dict / 126 str / 13 list / 24 None** (+46
new phase-75 steps). **The node's "674 dict" figure is the stale 75.2.1-era count;
the live figure is 720 dict.** A census must run against current HEAD, not a frozen
count — runtime-output artifacts appear/disappear between sweeps (see 8.5.4/10.2).

### Methodology (two independent derivations, reconciled by symmetric diff — qa.md §4b)

- **Extractor A** (structure-aware): `open('...')`, `test -f/-e`, `python X.py`,
  `pytest X`, `source X`, `cat X`, `bash X.sh` argument regexes over the COMMAND.
- **Extractor B** (broad pattern): extension-anchored regex (`tsx` BEFORE `ts` —
  the 75.2.1 alternation caveat) + repo-dir-anchored regex.
- Scope: **`status=done`** only. A `pending`/`deferred`/`dropped`/`merged`/
  `superseded` step naming a not-yet-created artifact is NOT a defect (the 75.19
  status-aware rule). 103 non-done steps carry absent-path references — all
  informational, zero defects.
- Resolution rules (all four shapes handled; a naive `.get('command')` crashes on
  the 13 list-shaped): repo-root-relative, frontend-relative (`frontend/` AND
  `frontend/src/`), URL fragments (skip), absolute host paths (skip), `/tmp` +
  `handoff/` + `frontend/handoff/` runtime/transient (skip), globs (resolve).
- **Negative-assertion detection** (the decisive filter both prior sweeps lacked):
  a path under `! test -f X`, `test ! -f X`, `[ ! -f X ]`, `test -f X || …`, or a
  Python `assert not os.path.exists(X)` is RUNNABLE and PASSES *because* the path
  is absent. These are NOT defects.
- Every surviving absent path is **git-classified**: `--diff-filter=A` empty ->
  never-existed (i); added-then-deleted -> retired (ii) with the retiring commit.

### FUNNEL (measured, reproducible via scratchpad `census_v3.py`)

| Stage | Count | What was removed |
|-------|-------|------------------|
| done steps flagged by A u B (any absent token) | 236 | — |
| after dropping transient/URL/prose/basename-exists | 20 | live_check_*.md, experiment_results.md, prose basenames (qa.md, portfolio_manager.py…), localhost URLs |
| after negative-assertion filter | 18 | 4.14.19, 16.50 (assert absence) |
| after resolving glob/frontend/shell-var artifacts | **10 genuine** | 4.8.6 (`$f.md`->runbooks exist), 7.12 (`alt_data_ic_*.tsv` globs), 16.37/16.39 (grep patterns; `lib/icons.ts`->`frontend/src/`), 23.1.4 (`for`), 23.5.2.6 (`found`/`located`), 75.2/75.16 (Python `assert not exists` absence-audits), `s.py` fragment |

### THE GENUINE DEFECT SET — 10 done steps (command-unrunnable, not absence-asserted)

| Step | Absent path in `verification.command` | Class | git evidence | Already annotated? |
|------|----------------------------------------|-------|--------------|--------------------|
| 4.17.2 | `scripts/go_live_drills/researcher_smoke_test.py` | (i) never-existed | `--diff-filter=A` empty | NO -> annotate |
| 4.17.3 | `scripts/go_live_drills/qa_smoke_test.py` | (i) never-existed | empty | NO -> annotate |
| 4.17.4 | `scripts/go_live_drills/handoff_e2e_test.py` | (i) never-existed | empty | NO -> annotate |
| 4.17.5 | `scripts/go_live_drills/coala_memory_layers_test.py` | (i) never-existed | empty | NO -> annotate |
| 4.17.6 | `scripts/go_live_drills/signal_evidence_test.py` | (i) never-existed | empty | NO -> annotate |
| 4.17.7 | `scripts/go_live_drills/paper_trade_e2e_test.py` | (i) never-existed | empty | NO -> annotate |
| 4.17.8 | `scripts/go_live_drills/slack_bot_smoke_test.py` | (i) never-existed | empty | NO -> annotate |
| 4.17.11 | `scripts/go_live_drills/openclaw_runtime_test.py` | (i) never-existed | empty | NO -> annotate |
| 4.17.12 | `scripts/go_live_drills/f1_recovery_drill.py` | (i) never-existed | empty | NO -> annotate |
| 4.14.26 | `backend/agents/skills/neutral_analyst.md` + `devils_advocate_agent.md` | (ii) retired | added cb413518, **deleted f7e24d0a = phase-26.4** "Consolidate 6 opinion skills into parameterized stance prompt" | NO -> annotate |

**On-disk truth for the 9 go_live_drills:** the WORK was done — the drills exist
under the `smoke_test_4_17_N.py` convention (all added in `1122a021`); only the
plan-side NAMES never matched. Class (i): name mismatch from day one, `git log
--all --diff-filter=A` empty for every plan-side name. This is the "ten-ish
phase-29 go_live_drills cluster" the node predicted — precisely 9 after excluding
the already-annotated 4.17.9. (4.17.10 runs `pytest scripts/go_live_drills/` on the
whole dir + `aggregate_gate_check.py`, both of which EXIST -> runnable -> correctly
NOT flagged.)

**4.14.26 (class ii, NEW — 75.2.1 did not name it):** its command
`grep -l "I don't know|…" <10 skill files> | wc -l | awk '{exit ($1<10)}'` now finds
only 8 of 10 files (neutral_analyst.md + devils_advocate_agent.md were deleted when
phase-26.4 consolidated the 6 opinion skills into one parameterized stance prompt),
so `wc -l` <= 8 < 10 -> non-zero exit. Genuinely unrunnable-to-PASS. Retiring commit
`f7e24d0a`, step 26.4.

### Already annotated by 75.2.1 — EXCLUDE, do not double-annotate (node + criterion 4)

| Step | Path | srec present? |
|------|------|---------------|
| 4.17.9 | `scripts/go_live_drills/self_update_audit_test.py` (never-existed) | YES OK |
| 4.14.24 | `backend/slack_bot/assistant_handler.py` (grep, retired f55e6973) | YES OK |
| 4.14.4 | `assistant_handler` import (retired f55e6973) | YES OK (not path-shaped; caught by 75.2.1's import sweep) |

### Why 75.2.1's "15" and Main's "33" both mis-counted (the triage payoff)

Neither prior sweep applied the three filters that resolve the discrepancy:

1. **Negative-assertion blindness** -> `4.14.19` (`test ! -f backend/mcp/__init__.py`)
   and `16.50` (`! test -f backend/agents/planner_enhanced.py && …`) were counted as
   "absent-path" defects. They are runnable and PASS *because* the paths are absent
   (identical to 75.2 asserting the dead modules' absence, which 75.2.1 itself said
   "must not be touched"). **Both are FALSE POSITIVES.**
2. **No current-disk check for runtime outputs** -> `8.5.4`
   (`test -f backend/autoresearch/results.tsv && …`) and `10.2`
   (`…/weekly_ledger.tsv`) name generated TSVs that EXIST on disk now (added
   `22e78958`), so the commands are runnable today. Absent only when the harness
   hasn't produced them — a runtime-data dependency, not a permanent mismatch.
3. **Frontend/glob/shell-var non-resolution** -> Main's "33 raw" inflation
   (`lib/icons.ts`->`frontend/src/lib/icons.ts`, `/openapi.json`+`/walk_summary.json`
   URL routes, `/Library/LaunchAgents/com.py` truncated plist, `alt_data_ic_*.tsv`
   glob, `$f.md` shell var). The `.tsx`/`.ts` alternation artifact (75.2.1's own
   caveat) inflated 15->27.

**Definitive count: 10 genuine defects needing annotation** (9 never-existed + 1
retired), **3 already annotated (excluded)**, and the rest of both prior counts are
false positives or runtime-conditional.

## Adaptive coverage — loop-until-dry (audit-class, K=2)

The census is a MEASUREMENT; dryness is proven by feeding a SINGLE rigorous
adjudicator (negation-aware + git-classify + full FP filter set) with progressively
different EXTRACTION strategies and confirming the *adjudicated genuine set does not
grow*. Reproducible via scratchpad `census_final.py`.

| Round | Added extraction strategy | Genuine (adjudicated) | New genuine | Verdict |
|-------|---------------------------|----------------------|-------------|---------|
| 1 | A = structure-aware (`open()`/`test -f`/`python`/`pytest`/`source`/`cat`/`bash`) | 12 steps / 13 paths | — | — |
| 2 | + B = broad regex (ext-anchored + repo-dir-anchored) | 12 / 13 | empty-set (symmetric diff A^B empty on genuine) | reconciled |
| 3 | + C = whitespace+quoted-string tokenizer | 12 / 13 | **NONE** | **DRY** |
| 4 | + D = maximal-recall (bare-ext + verb-arg + scans success_criteria/live_check) | 12 / 13 | **NONE** | **DRY** |

`{A,B}` and `{A,B,C,D}` yield the IDENTICAL genuine set -> the census is robust to
extraction strategy. **`dry_rounds = 2 >= K_required = 2 -> coverage.dry = TRUE.`**

**Design finding (feeds the sweep-tool spec):** the adjudicator cannot rescue a
naive extractor. Crude whitespace/maximal tokenizers (C, D) emit malformed tokens
from `python -c` inline code; only a **well-formedness gate**
(`^[.A-Za-z0-9_][A-Za-z0-9_./+*?\[\]-]*$`) + a **boundary gate** (the token must
appear as a WHOLE path in the command, `(?<![path-char])TOKEN(?![path-char])`)
collapse them to the same 10. Two residual hard cases only resolve with specific
handling, and every future sweep MUST include both:
- **glob-prefix** (7.12): `ls …/alt_data_ic_*.tsv` — a truncated token
  `alt_data_ic_` must be re-globbed (`glob(token+"*")`) before being called absent;
  the real glob matches `alt_data_ic_20260419T224855.tsv` -> FALSE POSITIVE.
- **full-path-aware negation** (75.16): `assert not os.path.exists('scripts/deploy/
  deploy_agents.sh')` and `open(p) if os.path.exists(p) else ''` — a bare-basename
  token must be matched against its FULL path inside the assertion, and the
  `else ''` tolerated-missing branch recognised -> FALSE POSITIVE (security
  absence-audit, same class as 75.2 / 16.50).

---

## External research (floor: >=5 read in full; the internal-heavy nature does not lower it)

### Queries run (3-variant discipline per topic)

| Topic | current-year (2026) | last-2-year | year-less canonical |
|-------|--------------------|-------------|--------------------|
| Verification/audit-replay rot | `unrunnable verification commands audit trail integrity … 2026` | `SOX control re-performance reproducibility attestation 2025` | (ISACA/audit-evidence canon) |
| Runbook/doc rot | `runbook automation tools 2026` | `runbook is already lying to you 2026` | `runbook rot documentation decay SRE` |
| JSON polymorphic fields | `JSON polymorphic field heterogeneous schema robustness` | `JSON schema compatibility robustness principle 2025` | `Postel's law robustness principle` |
| Path-resolution ambiguity | `monorepo path resolution ambiguity bazel 2026` | — | `relative path resolution workspace` |
| Append-only supersede | — | — | `immutable append-only record supersede ADR convention` |

### Read in full (6; gate floor is 5)

| # | URL | Accessed | Kind | Tier | Key finding |
|---|-----|----------|------|------|-------------|
| 1 | https://learn.microsoft.com/en-us/azure/well-architected/architect-role/architecture-decision-record | 2026-07-24 | official doc | 2 | Verbatim: *"The ADR serves as an append-only log. Don't go back and edit accepted records. If a decision changes, write a new record that supersedes the original and link the two together."* Status field evolves (`Accepted`/`Superseded`); *"Avoid hiding consequences."* This IS the `superseded_record` doctrine. |
| 2 | https://www.isaca.org/…/2026/volume-9/the-ai-audit-trail-from-ai-policy-to-ai-proof | 2026-07-24 | professional body | 2 | *"A policy cannot prove that an AI system behaved correctly at the moment it mattered."* Capturing prompt+answer is *"a transcript,"* not an audit trail. Genuine governance = *"auditable evidence"* an independent auditor can **reconstruct**. Maps 1:1 onto "a done step's PASS must be re-performable, not merely claimed." |
| 3 | https://yokota.blog/2025/10/07/json-schema-compatibility-and-the-robustness-principle/ | 2026-07-24 | authoritative blog | 3 | Postel's law for schema: *"Be conservative in what you send, be liberal in what you accept";* consumers *ignore* fields not in their schema. **Caveat:** *"full compatibility does not allow you to evolve sum types"* — polymorphic (sum-type) fields are the hard case, exactly the 4-shape `verification` field. |
| 4 | https://github.com/FasterXML/jackson-docs/wiki/JacksonPolymorphicDeserialization | 2026-07-24 | official docs | 2 | Polymorphic JSON needs *"per-instance type information"* (a discriminator) to pick the concrete shape; a *"default type handling baseline"* handles the unknown/missing-shape case. The parser must dispatch on shape — a naive fixed-shape read fails. Direct analogue to the dict/str/list/None dispatch. |
| 5 | https://arxiv.org/html/2502.18878 (Schema Reinforcement Learning) | 2026-07-24 | preprint | 1 | Structured-output conformance is measured by a *"fine-grained validator [that] calculates the correctness ratio … proportion of correct tokens"* — split at the error position, validate the remainder, never binary pass/fail. Models show 8–36% schema/parse error without enforcement. Lesson: a robust checker reports WHERE and WHAT fails per-item, not one pass/fail. |
| 6 | https://www.stew.so/blog/documentation-rot-devops | 2026-07-24 | industry blog | 4 | Doc rot timeline: *"Day 1 aligned … Day 180 useless"; "The longer docs sit untouched, the more they lie."* Remedy = *"actually running documentation against live systems … when documentation gets executed regularly, hidden decay becomes immediately visible."* This is precisely why an unrunnable `verification.command` on a done step is governance rot, and why the sweep must EXECUTE-resolve paths, not trust the text. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not read in full |
|-----|------|----------------------|
| https://earezki.com/ai-news/2026-05-17-the-runbook-is-already-lying-to-you/ | blog | HTTP 403 on fetch; runbook-rot thesis already covered by #6 |
| https://www.trustedintegration.com/understanding-sox-it-general-controls-and-evidence-auditors-accept/ | industry | Returned empty body; re-performance angle covered by #2 |
| https://dev.to/pbouillon/deserializing-polymorphic-json-in-net-without-losing-type-safety-4o4d | blog | .NET-specific; #3/#4 cover the pattern |
| https://www.aviator.co/blog/monorepo-tools/ | blog | Monorepo path-resolution (Bazel fine-grained file graph, pnpm workspace protocol removes local-resolution ambiguity) — supports frontend-relative resolution rule |
| https://github.com/architecture-decision-record/architecture-decision-record | repo | ADR immutability canon; #1 is the authoritative version |
| https://www.pactvera.com/best-platforms-for-immutable-audit-trails-in-2026/ | vendor | Immutable audit-trail platforms; #2 covers the principle |
| https://incident.io/blog/runbook-automation-tools-2026-the-complete-guide | vendor | Executable-runbook tooling; #6 covers the thesis |
| https://novaaiops.com/runbooks | vendor | "runbooks as hypotheses not maps" framing |
| https://www.networkintelligence.ai/blogs/sox-compliance-checklist/ | industry | SOX reperformance checklist |
| https://yokota.blog/… (Schema Registry) | blog | producer-strict/consumer-lenient detail |

**URLs collected: 16** (6 read in full + 10 snippet-only). Hierarchy satisfied:
1 preprint (T1), 2 official docs (T2), 1 professional body (T2), 1 authoritative
blog (T3), 1 industry blog (T4). Not community-heavy.

### Recency scan (last 2 years, 2025–2026) — performed

**3 findings** complement the canon; none supersedes it:
1. **ISACA 2026 "proof vs policy"** (#2) — freshest articulation of the exact thesis:
   a claim of a passing control is not the control; only re-performable evidence is.
   New framing over classic SOX re-performance.
2. **Yokota 2025 robustness-principle-for-schema** (#3) — the sum-type/polymorphic
   caveat is the current sharpest statement of why a `union`-shaped field (dict|str|
   list|None) resists a single-shape reader. Complements Jackson's older canon (#4).
3. **stew.so 2026 doc-rot + arXiv 2025 Schema-RL** (#6/#5) — converge on
   "execute-to-validate" and "per-item granular conformance reporting" as the
   modern remedies. Both post-date and reinforce the older runbook-automation canon.

The year-less canon (ADR append-only immutability #1, Postel's law, Jackson
polymorphism #4) remains the normative base; the 2025–2026 work adds the
proof-vs-policy sharpening and the sum-type caveat.

### Key findings (external -> mapped to pyfinagent)

1. **An unrunnable `verification.command` on a `status=done` step is governance rot,
   not a cosmetic flaw.** ISACA (#2): a transcript != an audit trail; the PASS must be
   *reconstructable*. stew.so (#6): *"execute documentation against live systems …
   decay becomes immediately visible."* -> The sweep must **execute-resolve** every
   path against HEAD, and the fix must **preserve** the historical PASS while marking
   it un-reproducible — exactly what `superseded_record` does.
2. **Immutable-append-only is the correct repair primitive.** MS ADR (#1): never edit
   accepted records; supersede + link + preserve history. -> NEVER amend
   `verification.command`/`success_criteria`; add a SIBLING `superseded_record`
   (byte-identity of the criteria is the machine-checkable form of "don't edit").
3. **The 4-shape `verification` field is a textbook polymorphic/sum-type field.**
   Jackson (#4) + Yokota (#3): dispatch on shape, provide a default for the unknown
   case; a fixed-shape reader (`.get('command')`) crashes on the 13 list-shaped and
   silently skips the 126 str-shaped + 24 None. -> The sweep's shape normalizer
   (`verif_strings`) is the discriminator; it must handle all four, verified by a
   fixture per shape.
4. **A robust checker reports per-item WHERE/WHAT, never a single pass/fail.**
   Schema-RL (#5): granular per-token conformance beats binary. -> The sweep must emit
   a per-row table (step, path, class, git-evidence), which is also what criterion 5
   ("no count asserted without the sweep output backing it") demands.
5. **Path-resolution ambiguity is a known monorepo hazard** (Aviator/pnpm snippet;
   Bazel's fine-grained file graph). -> frontend-relative (`lib/icons.ts` ->
   `frontend/src/`), URL-route, and glob tokens are the documented ambiguity classes
   the resolver must disambiguate — the node's named false-positive families.

---

## Repair prescription — `superseded_record` per class (mirror 75.2.1, do NOT invent)

The house convention already exists at 4 sites (`.claude/masterplan.json` lines
3908 / 4245 / 4870 / 15530). Reuse it EXACTLY. Criteria stay byte-identical; the
sibling annotates. Two variants:

### Class (i) never-existed — the 9 go_live_drills (template, per step)

```json
"superseded_record": {
  "superseded_at": "2026-07-24",
  "authorized_by": "operator directive for step 75.17 (recorded in harness_log Cycle NNN)",
  "reason": "PRE-EXISTING NAME MISMATCH from day one: the verification command runs `python scripts/go_live_drills/researcher_smoke_test.py`, a path that NEVER EXISTED -- `git log --all --diff-filter=A -- scripts/go_live_drills/researcher_smoke_test.py` returns empty. The drill WORK was done under the on-disk convention scripts/go_live_drills/smoke_test_4_17_2.py (added 1122a021); only the plan-side name never matched. The command has been unrunnable since the step was marked done, independent of any later deletion.",
  "retired_by_commit": null,
  "retired_in_step": null,
  "still_runnable": false,
  "already_broken_before_retirement": true,
  "criteria_amended": false,
  "on_disk_equivalent": "scripts/go_live_drills/smoke_test_4_17_2.py",
  "scope_disclosure": "One member of the phase-29 go_live_drills naming-drift family (4.17.2-4.17.8, 4.17.11, 4.17.12) surfaced by the 75.17 census; 4.17.9 was annotated by 75.2.1 and is excluded.",
  "note": "verification.command and verification.success_criteria left BYTE-IDENTICAL (CLAUDE.md: immutable). Annotation only; asserted against git show <pre-75.17 commit>."
}
```

Per-step `on_disk_equivalent` mapping (all confirmed added in `1122a021`):
`4.17.2->smoke_test_4_17_2.py`, `4.17.3->…_3`, `4.17.4->…_4`, `4.17.5->…_5`,
`4.17.6->…_6`, `4.17.7->…_7`, `4.17.8->…_8`, `4.17.11->…_11`, `4.17.12->…_12`.
(4.17.10 is NOT in the set — its command `pytest scripts/go_live_drills/` + existing
`aggregate_gate_check.py` is runnable.)

### Class (ii) retired-by-later-step — 4.14.26 (the NEW member 75.2.1 did not name)

```json
"superseded_record": {
  "superseded_at": "2026-07-24",
  "authorized_by": "operator directive for step 75.17 (recorded in harness_log Cycle NNN)",
  "reason": "phase-26.4 (commit f7e24d0a, 'Consolidate 6 opinion skills into parameterized stance prompt') deleted backend/agents/skills/neutral_analyst.md and backend/agents/skills/devils_advocate_agent.md. This step's command greps a hard-coded list of 10 skill files and asserts `wc -l >= 10`; with 2 of the 10 now absent the grep processes only 8 -> awk '{exit ($1<10)}' exits non-zero. The 'I don't know permission' work it verified was consolidated into the parameterized stance prompt, so the artifact -- not the outcome -- is retired.",
  "retired_by_commit": "f7e24d0a",
  "retired_in_step": "26.4",
  "still_runnable": false,
  "already_broken_before_retirement": false,
  "criteria_amended": false,
  "note": "verification.command and verification.success_criteria left BYTE-IDENTICAL. Annotation only; byte-identity asserted against git show <pre-75.17 commit>."
}
```

**Exclusions (criterion 4):** do NOT annotate 4.14.4, 4.14.24, 4.17.9 — each already
carries exactly one `superseded_record` (verified). The test must assert **exactly
one `superseded_record` per step repo-wide** (no double-annotation), and
**byte-identity** of `verification.command` + `verification.success_criteria` for
every touched step against the pre-75.17 commit.

---

## Mutation matrix — the sweep tool must be non-vacuous (criterion 6, incl. a fixture mutation)

| # | Mutation (of production OR fixture) | Expected | Guards against |
|---|-------------------------------------|----------|----------------|
| M1 | Break masterplan JSON parse (truncate a brace) | test HARD-FAILS (not skip) | silent parse-drift -> empty census |
| M2 | Plant an absent path in a done step's command (`python scripts/nope_xyz.py`) | sweep FINDS it, classes never-existed | the sweep can actually detect the target defect |
| M3 | Feed the resolver a frontend-relative path (`lib/icons.ts`) | resolver returns EXISTS (via `frontend/src/`) — NOT flagged | frontend-relative false positive (node-named) |
| M4 | Feed a URL fragment (`/openapi.json`, `localhost:8000/api/x`) | NOT flagged | URL-fragment false positive (node-named) |
| M5 | Feed a truncated plist (`/Library/LaunchAgents/com.py`) | NOT flagged | truncated/host-path false positive (node-named) |
| M6 | Feed a `! test -f X` negative assertion on an absent X | NOT flagged | absence-assertion false positive (4.14.19/16.50 class) |
| M7 | **FIXTURE mutation:** replace the list-shaped fixture with a dict-only fixture | test that "all 4 shapes handled" FAILS | 75.2.1 lesson — a fixture that cannot represent the list shape can't prove the crash-guard; mutate the STUB too |
| M8 | Change a superseded_record's `criteria_amended` to true / edit a criterion byte | byte-identity test FAILS | the immutability guarantee itself |
| M9 | Add a 2nd `superseded_record` to any step | exactly-one test FAILS | double-annotation (criterion 4) |
| M10 | Glob-truncation: feed `alt_data_ic_` where `alt_data_ic_*.tsv` exists | NOT flagged (glob-prefix re-resolve) | 7.12 hard-case |

M7 is the mandatory fixture-mutation (criterion 6): mutating only production code
would miss a fixture that cannot express the list/None shapes — the exact 75.2.1
trap (auto-memory `feedback_mutation_test_guards_and_fixtures`).

---

## Where the sweep tool lives — recommendation with evidence

**Precedent:** `scripts/qa/` already hosts committed sweep tools — `sweep_ascii_logger.py`
(+`_v2`,`_v3`), `coverage_tier_check.py`, `env_syntax_check.py` — each paired with a
nightly GitHub Actions workflow (`ascii-logger-lint.yml`, `coverage-tier-check.yml`,
`env-syntax-lint.yml`). Phase-75 tests follow `backend/tests/test_phase_75_*.py`.

**Recommendation:**
- **Commit the sweep at `scripts/qa/sweep_absent_verification_paths.py`** (mirrors the
  `sweep_*` idiom) and the guard at **`backend/tests/test_phase_75_17_verification_paths.py`**
  (matches the node's immutable `verification.command` verbatim).
- **Do NOT add a redundant nightly workflow.** Masterplan integrity is already the
  domain of `scripts/meta/preflight_verify_masterplan.py`, which **step 75.19 is
  chartered to recalibrate to be status-aware + transient-aware** (the exact filters
  this census needed). The reusable classifier (shape-normalizer, negation detector,
  git-classifier, FP filters) should be authored in 75.17 as an importable module and
  **handed to 75.19** for the preflight/CI integration — this honors the node's
  "COORDINATE WITH 75.19: the go_live_drills cluster belongs to 75.17; do not
  double-annotate" directive and avoids two overlapping nightlies.
- 75.17 is therefore a **one-shot triage + committed pytest guard**; continuous
  enforcement is 75.19's preflight-gate job.

---

## Step-text corrections (annotate, do not edit the immutable criteria)

1. **Shape census is stale.** The node states "674 dict / 126 str / 13 list / 24
   None" (837-step, 75.2.1-era). Current HEAD is **883 flat steps -> 720 dict / 126
   str / 13 list / 24 None**. The +46 are new phase-75 steps. The list=13 / None=24 /
   str=126 figures are unchanged; only dict moved 674->720. The sweep MUST run against
   live HEAD (runtime outputs drift — see 8.5.4/10.2).
2. **The genuine count is 10, not "ten-ish go_live_drills".** It is **9** go_live_drills
   (4.17.2-8, 4.17.11, 4.17.12; 4.17.9 excluded-annotated) **+ 1** retired non-drill
   (**4.14.26**, which 75.2.1 did NOT name). 75.2.1's "15" and Main's "33" both
   over-counted by conflating absence-assertions (4.14.19, 16.50), runtime outputs
   that exist now (8.5.4, 10.2), and extraction artifacts.
3. **4.14.4 is import-class, not path-class.** It is already annotated; a path-only
   sweep will NOT surface it (its reference is `from backend.slack_bot import
   assistant_handler`, not a filesystem path). The exactly-one-superseded_record test
   still covers it.

## Internal code inventory

| File / artifact | Role | Finding |
|-----------------|------|---------|
| `.claude/masterplan.json` | census target | 883 flat steps (`phases[].steps[]`, no nesting); 720 dict / 126 str / 13 list / 24 None verification shapes |
| `handoff/current/research_brief_75.2.1.md` | prior methodology | The "15" count + the `.tsx`/`.ts` alternation caveat (inflates 15->27); named 8.5.4/10.2/4.14.19/16.50 as "others" — 2 of which are FPs |
| `handoff/archive/phase-75.2.1/{contract,experiment_results,evaluator_critique}.md` | prior triage | f55e6973 deleted 7 files; 4.17.9 `--diff-filter=A` empty; the 837-step shape census |
| masterplan lines 3908/4245/4870 | `superseded_record` for 4.14.4/4.14.24/4.17.9 | the criteria-immutable variant to MIRROR (byte-identical + `criteria_amended:false`) |
| masterplan line 15530 | `superseded_record` for **68.5** (pending) | the OTHER variant (preserves `original_success_criteria` because live fields changed) — do NOT use this shape for 75.17 |
| `scripts/go_live_drills/` (36 files) | drill scripts | `smoke_test_4_17_{2..8,11,12}.py` EXIST (added 1122a021); the 10 plan-side names never existed (`--diff-filter=A` empty) |
| `backend/agents/skills/` | skill prompts | `neutral_analyst.md` + `devils_advocate_agent.md` ABSENT (deleted f7e24d0a / phase-26.4) -> 4.14.26 defect |
| `scripts/qa/` | sweep-tool home | `sweep_ascii_logger.py`+`_v2/_v3`, `coverage_tier_check.py`, `env_syntax_check.py` — the committed-sweep precedent |
| `.github/workflows/` | CI | `ascii-logger-lint.yml`, `coverage-tier-check.yml`, `env-syntax-lint.yml` — nightly sweep precedent (75.15) |
| `scripts/meta/preflight_verify_masterplan.py` | masterplan gate | 75.19's recalibration target; the natural continuous-enforcement home for the classifier |
| `backend/tests/test_phase_75_*.py` | test naming | confirms `test_phase_75_17_verification_paths.py` convention |
| scratchpad `census_final.py` | reproducible census | strategy-agnostic adjudicator; genuine = 12 steps / 13 paths (10 need new annotation) |

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **6**
- [x] 10+ unique URLs total (incl. snippet-only) — **16**
- [x] Recency scan (last 2 years) performed + reported — 3 findings
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line / commit anchors for every internal claim (git-classified)
- [x] **Audit-class coverage: `dry_rounds=2 >= K=2 -> coverage.dry=true`**

Soft checks:
- [x] Internal exploration covered every module named in the spawn prompt
   (masterplan shapes, 75.2.1 methodology, go_live_drills, superseded_record sites,
   sweep-tool location, preflight gate)
- [x] Contradictions noted (75.2.1's "15" vs Main's "33" vs the true 10; reconciled)
- [x] Claims cited per-claim with git evidence
- [x] TWO independent census derivations reconciled by symmetric difference (empty on
   genuine); strategy-agnostic adjudicator confirms robustness

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": true,
    "rounds": 4,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "The definitive census over the CURRENT masterplan (883 flat steps; shape census 720 dict / 126 str / 13 list / 24 None -- the node's 674-dict is the stale 75.2.1-era figure) yields exactly 10 status=done steps whose verification.command executes against an absent path and is NOT absence-asserted: 9 phase-29 go_live_drills (4.17.2-8, 4.17.11, 4.17.12) whose plan-side names never existed (git --diff-filter=A empty; the work exists under the on-disk smoke_test_4_17_N.py convention, added 1122a021) = class-i never-existed; plus 4.14.26 (retired) whose grep of 10 skill files now finds 8 because neutral_analyst.md + devils_advocate_agent.md were deleted by phase-26.4 commit f7e24d0a -- a NEW member 75.2.1 did not name. Excluded (already carry exactly one superseded_record): 4.14.4, 4.14.24, 4.17.9. Two independent extractors reconcile with an empty symmetric difference on the genuine set, and a strategy-agnostic adjudicator returns the identical set under {A,B} and {A,B,C,D} -> two dry rounds -> coverage.dry=true. 75.2.1's 15 and Main's 33 both over-counted by conflating absence-assertions (4.14.19, 16.50 use `test ! -f`/`! test -f` -> runnable, PASS because absent), runtime outputs that exist now (8.5.4/10.2 results.tsv/weekly_ledger.tsv), and extraction artifacts (frontend-relative lib/icons.ts->frontend/src, URL routes, truncated plist, alt_data_ic_*.tsv glob, $f.md shell var). Repair: add a criteria-immutable superseded_record sibling (mirror the 4.17.9 shape at masterplan:4870, NOT the 68.5 originals-preserved shape) to each of the 10; byte-identity of command+success_criteria asserted against the pre-75.17 commit; exactly one superseded_record per step repo-wide. Sweep tool -> scripts/qa/sweep_absent_verification_paths.py + backend/tests/test_phase_75_17_verification_paths.py; hand the reusable classifier to 75.19's preflight recalibration rather than adding a redundant nightly. Mutation matrix includes a mandatory fixture mutation (M7: list-shape fixture) per the 75.2.1 lesson.",
  "brief_path": "handoff/current/research_brief_75.17.md",
  "gate_passed": true
}
```


