# Live check — step 86.75 (harness best-practice audit)

**Date:** 2026-08-14 ~02:20 CEST
**Subject commit:** `9a59a4fa` (+ the deletion in `ab0659fe`)
**Backend:** pid 93024 — irrelevant; every measurement below is against files on disk.

> **STATUS: criteria 2, 3, 4, 5, 6, 8 MEASURED. Criterion 1 requires DRIVING a Q/A and is
> not yet done. Criterion 7 is OPERATOR-OWED (separation of duties) and cannot be
> discharged by me — I authored the agent-file change.**
>
> Two defects in my own audit were found by these checks and fixed here; both are recorded
> below rather than quietly corrected.

---

## Criterion 2 — the divergence, re-derived, and a probe that indicted itself

**First attempt returned the wrong answer, and the reason is the finding.**

```
grep -cE 'phase=86\.33 +result=CONDITIONAL' handoff/harness_log.md   ->  1
```

The audit_basis says this returns **0**. I was one keystroke from reporting the
audit_basis as false. It is not — **my probe matched its own documentation.** The single
hit was line 34409, the audit's *prose* quoting the grep pattern. The only real 86.33
cycle row is line 34032, `result=PASS`.

Anchored to actual cycle headers:

```
grep -cE '^## Cycle .*phase=86\.33 +result=CONDITIONAL'   ->  0    [audit_basis CORRECT]
POSITIVE CONTROL  '^## Cycle .*phase=36\.17 .*CONDITIONAL' ->  3    [probe is live]
NEGATIVE CONTROL  '^## Cycle .*phase=99\.99'               ->  0    [no spurious fire]
ledger: python scripts/qa/qa_wip.py 86.33  -> records_retained=3, 2 prior spawn paths
```

**Divergence confirmed: ledger 3, log 0.**

### A SECOND, INDEPENDENT reason the log-grep counter was unsound

Corpus-wide, the contamination is systematic:

| | |
|---|---:|
| lines mentioning `result=` | **806** |
| real cycle headers with `result=` | **685** |
| **prose lines that contaminate any unanchored grep** | **121** |
| `result=CONDITIONAL` unanchored | 36 |
| `result=CONDITIONAL` anchored | **26** |

`handoff/harness_log.md` now contains extensive prose *about* the grep patterns, so the
file **self-matches**. The retired rule's counter was not merely reading a file written
too late — **it was reading a file that discusses its own search terms.** This is an
additional failure mode the audit did not identify, and it strengthens the audit's
conclusion by a route the audit did not use.

### Disagreements with the audit_basis, reported not adopted

| Figure | audit_basis | this tree | note |
|---|---:|---:|---|
| `## Cycle` headers | 1,227 | **1,229** | file grows; consistent with `live_check_86.44`'s 1,229 |
| `result=CONDITIONAL` rows | 35 | **26** anchored / 36 unanchored | **35 sits between them — the audit_basis figure was itself very likely unanchored, hence contaminated** |
| WIP records on disk | 38 | **46** | ledger grew by 8 |
| step-ids with >=2 CONDITIONAL | 3 | **3** | agrees |

---

## Criterion 3 — the gate row the audit finding would have deleted is LIVE

```
.claude/agents/qa.md:570:| Contract completeness | gate | EVERY immutable criterion mapped
  to covering evidence in experiment_results.md (uncovered = Missing_Assumption, caps verdict) |
```

A live table row with `| gate |`, plus its prose at `:355`. **Kept deliberately** — the
finding that proposed deleting the weighted rubric would have taken this with it, and this
one is real phase-71.3 machinery.

---

## Criterion 4 — floors unchanged, and a LOWER floor found and closed

```
.claude/workflows/research-gate.js:213: const FLOOR_SOURCES = 5
.claude/workflows/research-gate.js:214: const FLOOR_URLS    = 10
```

Unchanged, and `verify_research_gate_workflow.mjs:641,643` mutation-tests both.

**My first scan of the live-doctrine population returned "NONE" — a false zero.** zsh does
not word-split unquoted variables, so `grep -nE PAT $LIVE` passed the whole list as one
non-existent filename; `2>/dev/null` swallowed the error and `|| echo NONE` printed a
clean result. Re-run with a **shell array** (and an assertion that all 31 files exist):

**`scripts/mas_harness/cycle_prompt.md` stated a 3-source floor, in two places** —
`:23` *"minimum 3 external sources"* and `:52` *"for at least 3 sources"*. Line 23 also
carried *"Pure-doc items … can skip research"*, a carve-out **the operator overruled on
2026-05-22**.

**The 86.75 commit edited this very file** (`:90`, repointing the deleted path) **and left
both contradictions in it — one seam short.** Fixed here: both raised to >=5 in full /
>=10 URLs, both carve-outs removed.

*Scope note, honestly:* `scripts/mas_harness/` is **dormant** — both launchd plists are
disabled (`.bak`, `disabled.` prefixes), no crontab entry, last log write **2026-06-01**.
So this was a latent trap, not a live rail. It is still a file an executor is told to read.

Remaining hits are both legitimate and verified by reading them:
- `ARCHITECTURE.md:502` — records the **raise** *from* 3 *to* 5. History, not a rule.
- `cycle_prompt.md:28` — my own correction note **quoting** the removed text.

Zero-survivor check on the two carve-outs: `can skip research` **0**,
`If the item is non-trivial` **0**, `for at least 3 sources` **0**.
Negative control (new text must be findable): `no pure-doc exemption` **1**,
`5 sources read IN FULL` **1**.

---

## Criterion 5 — verifier green at the baseline, not below it

```
node scripts/qa/verify_research_gate_workflow.mjs
  ALL GREEN: 121 passed, 0 failed
```

**121 = the cited baseline exactly.** Nothing was made green by deleting assertions.

---

## Criterion 6 — every mention of the deleted file is a deletion NOTE

`.claude/context/research-gate.md` — **confirmed absent** (`test -e` fails).

The two mentions that could have been live pointers, both inspected:

- **`scripts/autoresearch/run_memo.py`** — a docstring. `grep` for `open(`/`read_text`/
  `Path(` against a research-gate path returns **nothing**: no code reads it.
  **But the docstring had the 86.62 defect** — my correction sat *beside* the claim it
  corrected. Line 10 still opened *"The memo is a valid research-gate source"* and lines
  16–18 restated *"the gate included memos in its accepted-source list."* **Rewritten so
  the correction REPLACES it.** Zero survivors on all three superseded phrases;
  negative control finds the new text (1, 1); `ast.parse` OK.
- **`.claude/rules/research-gate.md:16`** — *"moved verbatim from the deleted …"*. A note.

Remaining hits are archives, memory files, logs, `masterplan.json` audit_basis, and a
`.patch` — records, not pointers.

---

## Criterion 8 — verdict semantics unchanged, DEMONSTRATED

Four mechanical checks on `.claude/workflows/qa-verdict.js`:

1. **`verdict` has exactly one assignment**: `:233 const verdict = await agent(PROMPT,…)`.
   The only other source is `:205 verdict: null` (blind run). **No code path synthesises a
   verdict.**
2. **No literal `'PASS'` is ever assigned** to a verdict field — grep returns nothing.
3. **The enum `['PASS','CONDITIONAL','FAIL']` at `:161` was NOT touched** by 86.75; the
   diff changed only prompt prose at `:145–152`.
4. **`NEVER return PASS on a loop-prevention / errored exit` survives** at `:152`, and the
   blind path returns `verdict: null, ok: false` — NO VERDICT, never PASS.

**The directional argument, which is the real point:** the counter was repointed from a
source that reads **systematically low** (log: 0 for 86.33) to one that reads **true**
(ledger: 3). A counter that reads *higher* makes the 3rd-attempt auto-FAIL fire **earlier
and more often**. Its only possible effect on a verdict is CONDITIONAL → FAIL. **There is
no mechanism by which it reaches PASS.**

Deleting the anchoring clause is the one change that could plausibly bear on this, so
stated precisely: it removes an instruction about how to treat a **prior** verdict; it
changes **no criterion** by which current evidence is graded. On unchanged evidence the
same criteria yield the same verdict. What it stops is a judge respawned to grade
**changed** evidence being told to defer to a stale verdict — which is the documented
fresh-respawn rule.

---

## The backend file in a docs-only commit

`git show --stat 9a59a4fa` lists **`backend/config/model_tiers.py`**. Checked, because a
commit touching backend under a doc-audit subject is exactly how scope creep hides:

```diff
 References:
-  - .claude/context/research-gate.md (citations baked into the plan at
+  - .claude/rules/research-gate.md (citations baked into the plan at
```

One line, inside the **module docstring**, repointing the deleted path. No executable
change. Consistent with the commit's own claim that all changes are prompt text, doc lines
or deletions.

---

## What this artifact does NOT license

- **Criterion 1 is NOT done.** It requires *driving* a Q/A and showing the returned notes
  carry the derived attempt number. Reading the prompt is exactly what the criterion
  forbids as evidence.
- **Criterion 7 is NOT dischargeable by me.** I authored the `qa.md` change; CLAUDE.md's
  separation-of-duties rule requires operator review before a step depends on it, plus a
  roster confirmation after restart (`scripts/qa/verify_qa_roster_live.sh`).
- **Therefore this step is NOT ready to close**, and no Q/A should be spawned claiming it
  is. Two of eight criteria are open, one of them permanently outside my authority.


---

## Cycle-2 captures (2026-08-17; exits unpiped)

```
$ git grep -l "context/research-gate" -- . | wc -l
21
$ git grep -l "context/research-gate" -- . | sort
.claude/agent-memory/qa/verdicts/verdict_wip_86.75__20260814T025732Z.md
.claude/agent-memory/researcher/project_cron_maintenance_jobs.md
.claude/agent-memory/researcher/project_research_gate_discipline.md
.claude/masterplan.json
.claude/rules/research-gate.md
handoff/archive/phase-4.16.3/phase-4.16.3-contract.md
handoff/archive/phase-4.16.3/phase-4.16.3-experiment-results.md
handoff/archive/phase-4.16.3/phase-4.16.3-research-brief.md
handoff/archive/phase-51.4/research_brief.md
handoff/archive/step-2.13-evaluator_critique.md
handoff/archive/step-2.13-experiment_results.md
handoff/audit/phase-4.11/tool_use_primitives.md
handoff/current/audit_phase75/confirmed_findings.json
handoff/current/contract_86.75.md
handoff/current/evaluator_critique_86.75.md
handoff/current/experiment_results_86.75.md
handoff/current/live_check_86.75.md
handoff/data/02aed8f.patch
handoff/harness_log.md
handoff/phase-proposals/phase-5.5-data-audit.md
scripts/autoresearch/run_memo.py
$ bash scripts/qa/verify_qa_roster_live.sh > /tmp/roster.txt 2>&1; echo ROSTER_EXIT=$?
ROSTER_EXIT=0
$ tail -2 /tmp/roster.txt
 On-disk + git checks PASSED. Behavioral check is operator-driven.
================================================================
$ grep -n "records_retained" .claude/agents/qa.md | head -2
(current wording -- the gauge correction; quoted in the GENERATE)
```

Operator approval record: AskUserQuestion 2026-08-17, answer "Approve all"
(the six qa.md commits + the maxTurns removal, itemised in the question
text; transcribed in experiment_results cycle-2 item 1).
