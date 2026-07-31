# Research Brief -- phase-81.1: handoff artifact naming-convention reconciliation

Tier: **complex**. `coverage.audit_class: true` (the caller asked for "EVERY code path
that parses/globs a handoff artifact filename" -- unknown denominator -> loop-until-dry,
K=2). Written 2026-07-31. REPORT-ONLY: no repo file edited except this brief.

> ## HEADLINE -- three verdicts up front
>
> 1. **The single-root-cause claim is CONFIRMED for (a) and (b), and CONFIRMED-WITH-A-
>    CORRECTION for (c).** One regex/glob convention mismatch does explain all three
>    reported symptoms. But (c) has a *second, independent* root cause the caller's
>    framing misses: `archive-handoff.sh` doesn't merely fail to move files, it
>    **COPIES four bare rolling filenames into the closing step's archive dir**, which
>    fabricates false archives. Widening the glob does not fix that half.
> 2. **STOP -- pending P1 step 75.11.4 already owns this entire fix**, by name, with
>    the same file:line anchors (`archive-handoff.sh:160`, `verify_handoff_layout.py:64`)
>    and the same preferred remedy ("teach both the hook and the verifier the
>    `*_<sid>.(md|json)` suffix form"). Shipping 81.1 as scoped duplicates a pending
>    P1 step and will collide with its immutable criteria.
> 3. **Two BLOCKERS would break the fix as planned.** (B1) Widening
>    `archive-handoff.sh`'s MOVE glob to `*_<sid>.md` / `.json` disarms the live_check
>    gate and re-darkens the verdict gate that phase-81.0 was written to fix -- and per
>    Anthropic's own hooks doc the hooks run **in parallel**, so it is a *race*, not a
>    deterministic break. (B2) 81.0's verification command already ends in
>    `verify_handoff_layout.py`, which **exits 1 right now** -- widening the regex
>    leaves it exiting 1 (109 violations instead of 128). A verification command that
>    is red before the step starts, exactly as the caller feared.

---

## 0. Reproduction of the caller's MEASURED FACT -- CONFIRMED

Run 2026-07-31 from repo root.

```
$ python3 -c "import os,re; ..."
total entries: 162
md files: 128            # 127 before this brief was created -- matches the caller
MATCHED by STEP_ID_RE: 0 []
```

`STEP_ID_RE` is defined **twice**, byte-identically:
- `scripts/housekeeping/verify_handoff_layout.py:51`
- `scripts/housekeeping/backfill_handoff_archive.py:64`

```python
STEP_ID_RE = re.compile(r"^(?:phase-)?([0-9]+(?:\.[0-9]+)*)[-.].*\.md$")
```

Live exit states, measured:

```
$ python3 scripts/housekeeping/verify_handoff_layout.py
handoff layout FAIL -- 128 invariant violation(s):    EXIT=1

$ python3 scripts/housekeeping/backfill_handoff_archive.py --dry-run
Summary: done-moved=0 misc-moved=125 audit-moved=1 log-moved=2 root-kept=1 ambiguous=0
```

`ls handoff/current/ | grep -c '^phase-'` = **0**. The prefix convention is extinct in
`current/`; the `phase-*-research-brief.md` convention assumed by
`scripts/go_live_drills/smoke_test_4_17_2.py:60` is a **third** convention and also
matches zero files.

`handoff/archive/misc/` now holds **748** files, **488** of which match the suffix
convention. 75.11.4's text recorded 428 on 2026-07-25 -- **+60 in six days**. The sweep
is active, not historical.

---

## 1. Consumer inventory -- EVERY code path that parses/globs a handoff artifact filename

Ordered by danger. **A miss that MOVES a file is far worse than a miss that skips** --
this is the Google SRE data-integrity thesis (Section 7) and it partitions the table.

### DESTRUCTIVE on a miss (relocates data)

| # | file:line | Pattern | What it does on a MISS |
|---|-----------|---------|------------------------|
| 1 | `scripts/housekeeping/backfill_handoff_archive.py:64` + `:154` | `STEP_ID_RE` | `sid=None` -> `_move(p, MISC)` at `:157` = **`shutil.move` to `handoff/archive/misc/`**. Silent except a stdout line. This is the mechanism that swept `evaluator_critique.json` on 2026-07-24 and disarmed the verdict gate for 13 step closes. |
| 1b | same, `:171-176` | sid matched but status unknown / not in `{pending,in-progress,blocked,done}` | Appends to `ambiguous[]` **and moves to MISC anyway**. **BUG:** `:211` then prints `"Ambiguous (left in current/ for manual review):"` -- the file was already moved at `:174`. The output text contradicts the action. Any operator trusting that line will not go looking in `misc/`. |
| 1c | same, `:152` | `_is_rolling_keep()` escape hatch | The ONLY protection. Covers `ROLLING_KEEP` (8 bare names incl. `evaluator_critique.json`) + prefix `evaluator_critique_*` **when and only when it ends `.json`**. `evaluator_critique_<sid>.md` is NOT protected. |
| 2 | `.claude/hooks/archive-handoff.sh:160` | `"$CURRENT_DIR/${sid}-"*.md` and `"$CURRENT_DIR/phase-${sid}-"*.md` | **Silent no-op.** No `nullglob`, so an unmatched glob expands to the literal pattern and is filtered by `[ -f "$f" ]` at `:161`. `moved=0` is echoed to stderr, which is not surfaced. Result: a closed step's suffix-named evidence stays in `current/` forever. |
| 2b | `.claude/hooks/archive-handoff.sh:146` | literal list `contract.md experiment_results.md evaluator_critique.md research.md research_brief.md` | **COPIES the four bare rolling files into the closing step's archive dir regardless of which step last wrote them.** This is not a miss -- it is a false hit, and it is the *second* root cause of symptom (c). 75.11.4 measured 129/747 (17%) archive dirs containing at least one .md whose own title names a different step. |
| 2c | `.claude/hooks/archive-handoff.sh:160` | uses `${sid}`, not `${short_sid}` | Latent: `short_sid="${sid#phase-}"` is computed at `:127` and used for the target dir at `:128`, but the glob at `:160` uses the raw `${sid}`. For a masterplan id already spelled `phase-6.1` the second glob becomes `phase-phase-6.1-*.md`. Dead pattern. |

### BLOCKING on a miss (holds the commit/push)

| # | file:line | Pattern | What it does on a MISS |
|---|-----------|---------|------------------------|
| 3 | `.claude/hooks/lib/live_check_gate.py:71` | literal `Path(handoff_current) / f"live_check_{step_id}.md"` | Returns `"skip"`. `auto-commit-and-push.sh:153-155` logs a WARN and **`exit 0` BEFORE `git add -A` at `:270`** -- so it skips the **commit AND changelog AND push**, matching the CLAUDE.md correction. Not a glob; a literal path in `handoff/current/` only. Never looks in the archive. |

### FAIL-OPEN but audibility-critical on a miss

| # | file:line | Pattern | What it does on a MISS |
|---|-----------|---------|------------------------|
| 4 | `auto-commit-and-push.sh:216-219` -> `.claude/hooks/lib/verdict_gate.py:59` | literal `handoff/current/evaluator_critique_${STEP_ID}.json`, falling back to rolling `evaluator_critique.json` | Returns `"no_input"` -> WARN + `systemMessage` at `:238-239`, push proceeds. This is precisely the phase-81.0 fix; moving these files re-creates the dark gate. |
| 5 | `.claude/hooks/lib/harness_log_gate.py:94` | `phase=<step_id>` token in the last `TAIL_LINES=2000` of `handoff/harness_log.md` | `warn` (default) or `skip` (block mode). Gate ships DISABLED. Not a filename parser, but same gate family and same `exit 0`-before-`git add` blast radius if ever set to block. |

### READ-ONLY on a miss (silently degrades or shows the WRONG document)

| # | file:line | Pattern | What it does on a MISS |
|---|-----------|---------|------------------------|
| 6 | `backend/api/backtest.py:1381` | `glob.glob(handoff/archive/**/<filename>, recursive=True)` | Returns `None` -> `/api/harness/*` endpoint returns `{"cycles": []}` / null. **This is the /backtest Harness tab parser.** Consumed by `frontend/src/components/HarnessDashboard.tsx:193-194` (`critique`, `contract` state). |
| 6b | `backend/api/backtest.py:1386` | `glob.glob(handoff/archive/**/*<base>*.md)` -- **substring** glob | Not a miss but a **false hit**: `*contract*.md` matches `contract_80.46.md`, `contract_36.7.md`, ... Candidates are then sorted by `getmtime` at `:1393` and the NEWEST wins over `handoff/current/contract.md`. Every per-step file moved into `archive/` becomes a competitor for the Harness tab's "contract"/"critique" panel. |
| 7 | `backend/agents/harness_state_reader.py:71` | `archive.rglob(filename)` (exact name), newest mtime | Returns `None`. **LATENT DEFECT:** `_resolve_handoff_file` is annotated `-> Path \| None` but every caller does `if not path.exists()` (`:82`, `:99`, `:112`, `:125`, `:138`) with no `None` guard -> `AttributeError` on a full miss. Masked today only because `handoff/research_plan.md` and the rolling files still exist. Directly in the blast radius of moving handoff files. |
| 8 | `scripts/go_live_drills/smoke_test_4_17_2.py:60` | `CURRENT_DIR.glob("phase-*-research-brief.md")` | **A THIRD convention.** Matches 0 files -> `assert briefs` raises. Test is red today. |
| 9 | `scripts/housekeeping/quarantine_phantom_archives.py:36-37` | `PHANTOM_RE=^phase-(?:phase-)?\d+(\.\d+)+-v\d+$`, `CANONICAL_RE=^phase-\d+(\.\d+)+$` | Operates on archive **directory** names, not artifact filenames. Non-match -> left alone (safe). Note it hard-codes the `phase-<sid>` dir shape -- any change to archive dir naming must keep this in sync. |
| 10 | `scripts/qa/sweep_absent_verification_paths.py:223,260` | globs path tokens extracted from masterplan `verification.command` | Reports a path as absent. Any handoff file referenced by a verification command and then moved shows up here. |

### NON-CODE writers/readers that encode a convention (drift sources)

| # | file:line | Convention it encodes |
|---|-----------|----------------------|
| 11 | `.claude/skills/masterplan/SKILL.md:222` | "Write `handoff/contract_{step_id}.md`" -- **`handoff/` root, not `handoff/current/`**. A fourth drift; nothing reads that path. |
| 12 | `.claude/workflows/qa-verdict.js:38` | Default evidence string names the bare rolling trio `handoff/current/{contract.md, experiment_results.md, evaluator_critique.md}`, while Main writes per-step names. |
| 13 | `.claude/rules/research-gate.md` "Handoff folder convention" + `.claude/agents/researcher.md:80` | `research_brief_<step>.md` -- the suffix convention, in the agent prompt. |

### Tests that pin any of this

| # | file:line | What it pins |
|---|-----------|--------------|
| 14 | `backend/tests/test_phase_36_8_...py:552` | AST-parses **both** housekeeping scripts and fails if `HANDOFF_ROOT_KEEP` / `AUDIT_KEEP_GLOBS` drift between them. Does **not** pin `STEP_ID_RE`. |
| 15 | `backend/tests/test_phase_36_7_...py:974,1013,1047-1048` | Loads both scripts by path; asserts kill-switch allowlist behaviour. Does **not** pin `STEP_ID_RE`. |
| 16 | `tests/verify_phase_23_8_4.py:289` | Asserts `.claude/hooks/archive-handoff.sh` **exists**. Does not pin the glob. |
| 17 | `backend/tests/test_phase_75_17_verification_paths.py:69` | Maps step `4.17.2` -> `scripts/go_live_drills/smoke_test_4_17_2.py`. |

**CI:** no `.github/workflows/*.yml` references `handoff/current` or either housekeeping
script (only `seed-stability-check.yml` touches `handoff/seed_stability_results.json`).
No CI breakage risk.

---

## 2. Q1 -- WHICH FILES CHANGE CLASSIFICATION, and which get MIS-ATTRIBUTED

Simulated against the live tree + live masterplan with the candidate widened regex
`^.*?_([0-9]+(?:\.[0-9]+)*)(?:[_-][^/]*)?\.(?:md|json)$` ORed with the legacy form:

| Bucket | Today (legacy) | Widened | Delta |
|---|---|---|---|
| done -> `archive/phase-<sid>/` | 0 | **86** | correctly archived instead of dumped in misc |
| pending -> **STAY in current/** | 0 | **16** | **16 in-flight artifacts stop being destroyed** |
| ambiguous -> misc (flagged) | 0 | 18 | now visible |
| no-match -> misc (silent) | 125 | 5 | |
| **total moved out of `current/`** | **125** | **109** | **-16** |

The 16 rescued files are the whole point: `contract_81.0.md`, `experiment_results_81.0.md`,
`research_brief_81.0.md`, `contract_80.46.md`, `experiment_results_80.46.md`,
`live_check_80.46.md`, `research_brief_80.46.md`, `contract_78.1.md`,
`research_brief_78.1.md`, `contract_80.11.md`, `research_brief_80.11.md`,
`research_brief_80.36.md`, `contract_36.20.md`, `experiment_results_36.20.md`,
`live_check_36.20.md`, `research_brief_36.20.md`.

All 19 target archive dirs (`phase-36.7 … phase-80.40`) **already exist** with 4 files
each; `_safe_target()` at `:114-123` appends `-vN` rather than clobbering. No overwrite risk.

### MIS-ATTRIBUTION -- the specific names the caller asked about

| File | Parsed sid | Correct? | Verdict |
|---|---|---|---|
| `tier_ledger_2026-07-26.md` | **`2026`** | NO | **Date parses as a step id.** Nine such files: `count_reconciliation_2026-07-26.md`, `done_definition_evidence_2026-07-26.md`, `goal_masterplan_drain_2026-07-25_DRAFT.md`, `goal_masterplan_drain_2026-07-26.md`, `goal_paste_2026-07-25.md`, `goal_paste_2026-07-26_evening.md`, `operator_ask_2026-07-26.md`, `p0_triage_2026-07-26.md`, `tier_ledger_2026-07-26.md`. |
| `p0_triage_2026-07-26.md` | `2026` | NO | same |
| `census_78.md` / `.json` / `census_78_decision_draft.md` | `78` | NO | `78` is a **phase** id, not a step id. `statuses` (`:102-111`) only collects `phases[].steps[].id`, so lookup returns `None`. |
| `goal_phase80_ui_defect_burndown_DRAFT.md` | *(none)* | n/a | `_phase80_` -- the char after `_` is `p`, not a digit. Correctly unmatched. |
| `research_brief_80.4b_death_detection.md` | *(none)* | n/a | Correctly unmatched by my candidate. **But see below.** |
| `research_brief_36.7_80.40.md` | `36.7` | **partly** | This brief legitimately covers **two** steps (36.7 and 80.40). Attributing it to 36.7 hides it from anyone auditing 80.40. |
| `evaluator_critique_78.2_cycle5.md` | `78.2` | YES | trailing `_cycle5` correctly ignored |

**The `2026` blast radius is worse than "moved to the wrong place".** Today those nine
files are correctly reported by `verify_handoff_layout.py` as strays. Under the widened
regex they parse to sid `2026`, `statuses.get("2026")` returns `None`, the status is not
`"done"`, so **the verifier appends no failure and goes silent on them**. A genuinely
unclassified file becomes invisible. **A date guard is mandatory, not cosmetic** --
reject any candidate sid matching `(?:19|20)\d{2}` when the following character is `-`,
or (cleaner) require the resolved sid to exist in the masterplan before treating it as one.

**On `80.4b`:** the caller asked for a trailing-suffix form that accepts
`research_brief_80.4b_death_detection.md`. Measured: `grep -c '"80.4b' .claude/masterplan.json`
= **0** -- there is no step `80.4b`. Reading the file's header, it *is* a genuine
follow-up to step 80.4 ("Caller: Main (phase-80 step 80.4, criterion 4 reported FAILED)"),
and 80.4 is `done`. So attributing it to `80.4` happens to be correct **here**. But the
general rule "strip a trailing letter and attribute to the numeric parent" is an
unverifiable heuristic that would silently re-home any future `<sid><letter>` artifact.
Recommendation: **do not** add letter-suffix stripping to the regex. Leave `80.4b` in the
no-match bucket and, if it should be archived, rename the file once. Rename-one beats
teach-the-parser-to-guess (RFC 9413, Section 7).

---

## 3. Q2 -- Blast radius of the backfill, and the safe sequencing

**Reframe the premise.** The widened regex does **not** "suddenly move ~100 files out of
`handoff/current/`". Measured: the backfill moves **125 files out of `current/` TODAY**,
right now, on its next unflagged run. Widening reduces that to **109** and redirects 86
of them from the junk drawer to their correct per-step archive dirs. On raw counts the
fix is **strictly less destructive** than the status quo.

That does not make an 86-file move safe. Four consequences, measured:

1. **The Harness tab's document identity changes.** `backend/api/backtest.py:1386` globs
   `archive/**/*contract*.md` / `*evaluator_critique*.md` and picks the newest mtime
   (`:1393`). Adding 86 per-step files to the archive adds 86 competitors against
   `handoff/current/contract.md`. `shutil.move` within one APFS volume is `rename(2)` and
   **preserves mtime**, so the ordering is inherited, not reset -- but the newest
   `contract_<sid>.md` will outrank a staler rolling `contract.md`. The tab can begin
   rendering a different step's document with no marker. (Same class as the false-archive
   defect 75.11.4 escalated to P1.)
2. **A flipped-but-uncommitted step becomes unpushable.** `FLIPPED_STEP`
   (`auto-commit-and-push.sh:63-127`) diffs the working tree against **`HEAD`**, so a
   step stays "newly done" on every subsequent masterplan edit until it is actually
   committed. If a verdict gate `hold` (or any failure) blocks the commit, and the
   backfill is run in between, `live_check_<sid>.md` moves to the archive and the
   live_check gate flips to `skip` permanently -- commit, changelog and push all held,
   with a WARN telling the operator to create a file that already exists somewhere else.
3. **Verification-command reproducibility.** 75.11.4 already recorded this: `census_78.json`
   is opened by step 78.0's own immutable verification command
   (`json.load(open('handoff/current/census_78.json'))`). Under the widened regex it parses
   to sid `78` -> unknown -> **still swept to misc**. The fix as scoped does NOT protect it.
4. **Idempotency is preserved** (`_safe_target` `-vN`), but idempotent-and-wrong is still wrong.

### Safe sequencing -- recommended, in order

1. **Do not couple the parser fix to a bulk move.** Ship the widened classifier with the
   backfill still in dry-run. This is Fowler's **expand** phase (Section 7): accept both
   conventions, migrate nothing yet.
2. **Invert the default: `--dry-run` becomes the default; executing requires `--execute`.**
   75.11.4 already pins this as an immutable criterion, and it is the 2025-26 industry
   consensus (Section 7). The script's own docstring calling itself "Idempotent: safe to
   re-run" is what invited both unreviewed runs.
3. **Add a masterplan-reference guard before any move** -- refuse to relocate a file whose
   basename appears in any step's `verification.command` or `verification.live_check`.
   Also pinned by 75.11.4.
4. **Add the gate-artifact allowlist**: `live_check_*.md` and `evaluator_critique_*.json`
   never leave `handoff/current/` via the backfill, because a hook reads them by literal
   path. `evaluator_critique_*.json` is already covered by `ROLLING_KEEP_PREFIXES`;
   `live_check_*` is **not**.
5. **Require a clean `git status` before `--execute`**, so a flipped-but-uncommitted step
   cannot be swept mid-cycle (mitigates consequence 2).
6. **Only then** run the migrate phase, once, with the printed plan reviewed -- and
   verify the dry-run output *programmatically* (no empty sids, no duplicate targets, no
   sid absent from the masterplan), which is the specific lesson of claude-code #31034
   where a visually-correct dry-run masked the bug that destroyed 38 files.

**Leaving `current/` alone entirely is not viable** if any verification command for this
phase invokes `verify_handoff_layout.py` -- see Blocker B2.

---

## 4. Q3 -- Masterplan verification commands that depend on current behaviour

**B2 (BLOCKER).** Pending step **81.0**'s verification command ends with:

```
... && python3 scripts/housekeeping/verify_handoff_layout.py
```

That command **exits 1 today** (128 violations). Under the widened regex it still exits 1
(86 done-step violations + 23 misc). **It only reaches exit 0 after the files are actually
moved.** So 81.0 -- the step immediately before this one, still `pending` -- shipped a
verification command that was red before it started. The caller predicted exactly this.
If 81.1 reuses `verify_handoff_layout.py` as its verification command, it inherits the
same red-before-start defect unless the migrate phase runs in the same step.

**Step 4.16.2** (`status: done`) has `"command": "python scripts/housekeeping/verify_handoff_layout.py"`
-- a done step whose immutable verification command has been failing for months.

**Step 78.0** (`done`) opens `handoff/current/census_78.json` in its verification command.
`census_78.json` is in `handoff/current/` today and is swept to misc by both the current
and the widened classifier (sid `78` is a phase, not a step). Any move breaks that step's
reproducibility -- already documented verbatim in 75.11.4.

**Step 4.17.2** (`done`) has command `scripts/go_live_drills/researcher_smoke_test.py`, a
path that never existed; it carries a `superseded_record` pointing at the on-disk
equivalent `smoke_test_4_17_2.py`, whose `CURRENT_DIR.glob("phase-*-research-brief.md")`
matches zero files. Both the plan-side and the disk-side commands are red.

Older `done` steps assert specific `handoff/current/` files that are long gone
(`4.5.0` -> `phase-4.5-contract.md`, `5.5.6` -> `phase-5.5-contract.md`,
`8.4` -> `phase-8-decision.md`). These are already broken and are not made worse.

---

## 5. Q4 -- Is there a test pinning `STEP_ID_RE` or the archive glob?

**No.** Measured by grepping `backend/tests/`, `tests/`, `scripts/`, `.github/`:

- `STEP_ID_RE` appears **only** in the two housekeeping scripts. Zero test references.
- `archive-handoff.sh` is referenced by exactly one test, `tests/verify_phase_23_8_4.py:289`,
  which asserts the file **exists** -- not what it globs.
- `backend/tests/test_phase_36_8_...py:552` AST-parses both housekeeping scripts, but only
  for `HANDOFF_ROOT_KEEP` / `AUDIT_KEEP_GLOBS` drift.
- `backend/tests/test_phase_36_7_...py` loads both scripts and pins only the
  kill-switch allowlist behaviour.

**Consequence:** the widened regex has **zero existing test coverage to break** -- and
zero existing safety net. Whatever tests this step writes are the first. Per the
mutation-test memory, a fixture asserting "a pending-step file is NOT moved" is only
load-bearing if flipping that step to `done` flips the assertion; assert both directions
in one run (75.11.4 already pins this wording).

---

## 6. Q5 -- Does pending step 75.11.4 already own part of this? YES -- essentially all of it

**Status: `pending`, priority `P1`, BOUNDARY: `scripts/housekeeping/** + its tests only`.**

Quoted verbatim from `75.11.4.name`:

> "ROOT CAUSE, MEASURED 2026-07-25 (this is WHY the misclassification happens, and it
> makes the fix bigger than 'add a status check'): the tooling and the artifacts use
> OPPOSITE naming conventions. `.claude/hooks/archive-handoff.sh:160` moves only
> `${sid}-*.md` and `phase-${sid}-*.md` -- a step-id PREFIX joined by a DASH -- and
> `scripts/housekeeping/verify_handoff_layout.py:64` likewise reports 'has no step-id
> prefix'. But every artifact this project actually writes uses a step-id SUFFIX joined
> by an UNDERSCORE (`live_check_78.0.md`, `research_brief_76.9.3.md`, `census_78.json`)."

> "THE FIX MUST RECONCILE THE CONVENTION: either teach both the hook and the verifier the
> `*_<sid>.(md|json)` suffix form (preferred -- it matches ~all existing artifacts and
> needs no renames), or rename the convention repo-wide and update every writer. Decide
> explicitly and state which. Re-measure both counts before starting; they move every cycle."

That is phase-81.1's entire planned fix, already written down, already P1, already pending.
Its immutable success criteria additionally pin:

> - "verify_handoff_layout.py and archive-handoff.sh agree on ONE convention; a test asserts the same filename is classified identically by both"
> - "The archive hook actually archives a modern suffix-named artifact on a status flip -- proven by flipping a scratch step and observing the files land in handoff/archive/phase-<sid>/, not by reading the glob"
> - "**The closing step's OWN artifacts (including suffix-named ones like `live_check_<sid>.md`) land in its archive directory**"
> - "Default invocation is a DRY-RUN printing the plan; executing requires an explicit flag"
> - "The script never moves a file referenced by any masterplan step's verification.command or verification.live_check"
> - "No archive directory contains a document belonging to a different step -- proven by a checker over the whole handoff/archive tree, not a spot check"

Two structural problems follow:

1. **Duplication.** Shipping 81.1 as scoped means two masterplan steps modifying the same
   two files with different criteria. Whichever runs second either invalidates the first's
   tests or has to redo them. The operator's standing rule is that discovered defects get
   their own step -- but 75.11.4 *is* that step, filed 2026-07-24.
2. **Direct conflict (see B1).** The bolded criterion above requires `live_check_<sid>.md`
   to land in the archive dir on a status flip. The live_check gate requires it to be in
   `handoff/current/` at that same moment. Those two cannot both hold under the current
   hook wiring. **75.11.4's own immutable criteria contain a latent contradiction with
   phase-23.8.1's gate**, and nothing has caught it because the glob has never matched.

**Note a stale anchor:** 75.11.4 cites `verify_handoff_layout.py:64`. `STEP_ID_RE` is now
at **`:51`** -- phase-81.0 inserted `ROLLING_KEEP_PREFIXES` above it. Re-derive before citing.

---

## 7. External research

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://www.rfc-editor.org/rfc/rfc9413.html | 2026-07-31 | IETF standards doc (tier 2) | WebFetch | Tolerating unexpected input creates a "pathological feedback cycle" where errors "become entrenched as de facto standards" and later implementations must be "bug-for-bug compatible". Recommends **"virtuous intolerance"**: "Choosing to generate fatal errors for unspecified conditions instead of attempting error recovery can ensure that faults receive attention." |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-07-31 | Official vendor doc (tier 2) | WebFetch | "Communication was handled via files: one agent would write a file, another agent would read it and respond either within that file or a new file that the previous agent would read in turn." Notably **silent on artifact naming and on how a harness validates its own artifacts** -- the gap this step lives in. |
| https://code.claude.com/docs/en/hooks | 2026-07-31 | Official vendor doc (tier 2) | WebFetch | **"All matching hooks run in parallel, and identical handlers are deduplicated automatically."** PostToolUse: "Can block? No"; "Any other exit code is a non-blocking error." |
| https://sre.google/sre-book/data-integrity/ | 2026-07-31 | Google SRE Book ch.26 (tier 2) | WebFetch | "It's advantageous to design your interfaces to hinder developers unfamiliar with your code from circumventing soft deletion features with new code." Google Music case: a refactored deletion pipeline removed "approximately 600,000 audio references that shouldn't have been removed". "Check the correctness of the most critical elements of your data using out-of-band data validators, even if API semantics suggest that you need not do so." |
| https://martinfowler.com/bliki/ParallelChange.html | 2026-07-31 | Authoritative blog (tier 3) | WebFetch | Expand: "augment the interface to support both the old and the new versions". Migrate: "update all clients ... incrementally". Contract: "remove the old version". Risk: "If the contract phase is not executed you might end up in a worse state than you started." |
| https://arxiv.org/html/2606.20631v1 | 2026-07-31 | arXiv preprint (tier 1) | WebFetch (`/html/` chain per research-gate.md) | "Evidence & Feedback observes and acts on the run, producing records, verifications..." and the design principle that "Unavailable or unauthorised capabilities become **visible runtime mismatches** rather than implicit authority expansion." **[ADVERSARIAL-adjacent]**: treats identifiers as a deferred "synthesis role", i.e. the literature does NOT prescribe an artifact-naming scheme -- so no external authority endorses either convention here. |
| https://github.com/anthropics/claude-code/issues/31034 | 2026-07-31 | First-party incident report (tier 5) | WebFetch | "All 36 .md files were sequentially renamed to .md, each overwriting the last." "Claude first ran a dry-run preview using echo -- the preview output appeared correct (though it was also affected by the same bug ... which went unnoticed)." Recommends: "verify that all destination filenames are unique. Abort if collisions are detected"; "programmatically verify the output makes sense ... rather than just displaying it". |
| https://danieljamesglover.com/blog/2026-02-01-dry-run-engineering-practice/ | 2026-07-31 | Practitioner blog (tier 5) | WebFetch | "I have started making `--dry-run` the default for destructive scripts. You have to explicitly pass `--execute` or `--no-dry-run` to make changes." "A misconfigured job that runs in dry-run mode by default produces logs instead of damage." |

**8 read in full.** Tier mix: 1 preprint, 3 official docs/standards, 1 authoritative
engineering book, 1 authoritative blog, 2 practitioner/community -- the hierarchy is
satisfied (not 5 community-tier URLs).

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://cacm.acm.org/practice/the-robustness-principle-reconsidered/ | peer-reviewed | HTTP 403 |
| https://queue.acm.org/detail.cfm?id=1999945 | peer-reviewed (mirror of above) | HTTP 403 |
| https://en.wikipedia.org/wiki/Robustness_principle | encyclopedia | superseded by RFC 9413 read in full |
| https://lobste.rs/s/enszyj/harmful_consequences_robustness | forum | community tier, discussion of RFC 9413 |
| https://www.laws-of-software.com/laws/postel/ | community | low weight |
| https://devopedia.org/postel-s-law | community | low weight |
| https://helio.app/ux-research/laws-of-ux/postels-law/ | community | UX framing, off-topic |
| https://www.sciencedirect.com/topics/computer-science/robustness-principle | aggregator | paywalled |
| https://dev.to/sankalp_haritash/whats-the-postels-law-... | community | low weight |
| https://medium.com/@mesw1/understanding-the-robustness-principle-... | community | low weight |
| https://arxiv.org/pdf/1912.03962 | preprint | protocol-detection attacks; adjacent not central |
| https://docs.pimcore.com/platform/next/Pimcore/Upgrade_Notes/ | vendor doc | precedent: 2026.1 changed email-log folder structure and **shipped a migrate command** |
| https://github.com/knex/knex/issues/3504 | issue tracker | migration filename conventions |
| https://technori.com/2026/03/25054-how-to-implement-api-versioning-and-backward-compatibility/ | blog | generic API versioning |
| https://catchdiff.com/blog/document-version-control-best-practices | blog | ISO-8601 date naming |
| https://mapsoft.com/posts/file-naming-best-practices.html | blog | naming hygiene |
| https://rename.click/blog/file-naming-conventions | blog | naming hygiene |
| https://paiml.github.io/bashrs/concepts/idempotency.html | doc | shell idempotency definition |
| https://gist.github.com/amazingvince/52158d00fb8b3ba1b8476bc62bb562e3 | gist | "Modern Agent Harness Blueprint 2026" |
| https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-at-build-2026-announce/ | vendor blog | agent harness category |
| https://deepwiki.com/rohitg00/agentbrain/4.3-handoff-report-and-memory-artifacts | wiki | handoff reports under "a strict JSON schema" |
| https://github.com/Priivacy-ai/spec-kitty/issues/692 | issue | ADR-6: "canonical generated-artifact directory layout per artifact kind and exact identity rules including filenames" |
| https://github.com/ai-boost/awesome-harness-engineering | list | harness patterns index |
| https://zylos.ai/research/2026-03-31-agent-harness-design-patterns/ | blog | harness infra layer |
| https://arxiv.org/pdf/2604.21003 | preprint | "The Last Harness You'll Ever Build" |
| https://github.com/tw93/Mole/blob/main/SECURITY_AUDIT.md | repo doc | destructive-cleanup audit checklist |
| https://dev.to/danieljglover/dry-run-engineering-... | blog | duplicate of the read-in-full post |
| https://medium.com/@racholsanraj/idempotent-data-pipelines-... | blog | idempotency in pipelines |
| https://www.datacamp.com/tutorial/git-clean | tutorial | destructive git cleanup |

**Total unique URLs collected: 37.**

### Search-query composition (three variants, per research-gate.md)

1. **Current-year frontier** -- `"filename convention migration dual-format parser backward compatibility 2026"`; `"agent harness file-based handoff artifact naming convention drift 2026"`
2. **Last-2-year window** -- `"destructive cleanup script dry-run default safety 2025 idempotent file mover data loss"`
3. **Year-less canonical** -- `"robustness principle Postel's law parser leniency security risk"` (surfaced RFC 9413 + the CACM prior art a year-locked query buries)

### Recency scan (2024-2026)

Performed. **Four new findings in the window, none of which supersede the canonical
prior art, two of which sharpen it:**

1. **(2026, first-party) `anthropics/claude-code` #31034** -- a Claude Code bulk-rename
   destroyed 38 files because a regex silently produced empty capture groups and the
   dry-run preview *displayed the bug without anyone noticing*. This is the single most
   on-point precedent for this step: same tool, same year, same class (filename regex +
   bulk move), and it proves that a dry-run which is only *read* is not a control.
2. **(2026-02) dry-run-by-default** is now the stated practitioner default for destructive
   scripts, inverting the safety model so that damage requires an explicit flag. Confirms
   75.11.4's already-pinned criterion.
3. **(2026) agent-harness literature** (arXiv:2606.20631, spec-kitty ADR-6, agentbrain)
   converges on *schema-enforced* handoff artifacts with "exact identity rules including
   filenames" -- i.e. the field is moving toward pinning identity in a schema rather than
   inferring it from a filename regex. Directionally this argues the long-term fix is a
   front-matter/`step_id` field inside the artifact, not a smarter filename parser.
4. **(2023, still current) RFC 9413** remains the authoritative counterweight: liberal
   parsers entrench their own bugs. It does **not** forbid a transition window; it forbids
   an *indefinite* one. That maps exactly onto Fowler's contract phase.

No finding in the window supersedes Fowler (2011) or the SRE book (2016); both are cited
as current practice by the 2026 sources.

### Consensus vs debate

- **Consensus:** accept both forms *during a bounded migration*; make the destructive
  half opt-in; validate the plan programmatically before executing it.
- **Debate:** RFC 9413 / Thomson & Schinazi would say permanent dual acceptance is itself
  the bug -- the legacy `<sid>-name.md` form should be given an explicit end date, or the
  128 legacy-form files should simply be renamed once. Fowler agrees ("if the contract
  phase is not executed you might end up in a worse state than you started"). **Applied
  here:** zero files in `handoff/current/` use the legacy form. The legacy branch of the
  widened regex is therefore needed only for `handoff/archive/` history and for any
  future re-run over old trees -- worth keeping, worth commenting as terminal, not worth
  extending with letter-suffix guessing.

### Pitfalls (from literature) mapped to this step

| Pitfall | Source | Where it bites here |
|---|---|---|
| A visually-correct dry-run hides the bug | claude-code #31034 | `backfill --dry-run` prints 109 lines nobody will read line-by-line -> **assert** properties of the plan (no sid absent from the masterplan, no `2026`-shaped sid, no duplicate targets) |
| Silent regex failure produces empty/garbage keys | claude-code #31034 (BASH_REMATCH on CJK) | `STEP_ID_RE` returning `None` is the *entire* defect here; the widened form must fail LOUDLY on a partial match, not fall through to misc |
| Automation that relocates data without soft-delete | SRE ch.26 | `shutil.move` to `misc/` is already the soft-delete tier -- keep it, never add a delete path |
| Liberal parser entrenches its own bug | RFC 9413 | Do NOT add `80.4b`-style letter stripping; rename the one file instead |
| Skipping the contract phase leaves you worse off | Fowler | Widening the regex without ever running the migrate phase leaves `verify_handoff_layout.py` red forever |
| Out-of-band validation | SRE ch.26 | The "same filename classified identically by both scripts" test 75.11.4 pins IS the out-of-band validator |

---

## 8. BLOCKERS -- things that would break this fix

**B1 (HARD BLOCKER) -- widening `archive-handoff.sh`'s MOVE glob disarms the gates, as a RACE.**
`archive-handoff.sh` and `auto-commit-and-push.sh` are both `PostToolUse` hooks under the
same `Write` and `Edit` matchers (`.claude/settings.json`). `auto-commit-and-push.sh:264`
comments that archive-handoff "runs ahead of us in the chain" -- but Anthropic's hooks doc
states **"All matching hooks run in parallel"**. So if the glob is widened to `*_${sid}.md`:
- `live_check_<sid>.md` may be moved out of `handoff/current/` *while* `live_check_gate.py:71`
  is looking for it -> `skip` -> `exit 0` at `:155`, **before `git add -A` at `:270`**.
  The step's commit, changelog and push are all silently dropped.
- If `.json` is added, `evaluator_critique_<sid>.json` moves -> verdict gate returns
  `no_input` -> exactly the dark gate phase-81.0 exists to fix.
- Because it is a race, it will pass a single manual test and fail intermittently later.
**Mitigation (all three):** (i) exclude `live_check_*` and `evaluator_critique_*.json` from
the hook's MOVE list; (ii) teach `live_check_gate.py` / `verdict_gate.py` to fall back to
`handoff/archive/phase-<sid>/` so they are order-independent; (iii) do not rely on hook
ordering anywhere. **And note this contradicts 75.11.4's immutable criterion** that the
closing step's `live_check_<sid>.md` must land in the archive dir -- that conflict has to
be resolved explicitly, in writing, before either step ships.

**B2 (HARD BLOCKER) -- the verification command is red before the step starts.**
`verify_handoff_layout.py` exits **1** today (128 violations) and exits **1** after the
regex widening (109 violations). It only reaches 0 once the migrate phase actually runs.
Pending step **81.0** already ends its verification command with this exact script, so
81.0 cannot pass as written. If 81.1 adopts the same command, it inherits the same defect.
**Mitigation:** either (a) scope 81.1's verification to unit tests over the classifier
(pending-vs-done fixture, date-guard fixture, both-scripts-agree fixture) and leave the
tree untouched, or (b) include the reviewed migrate run in the step and make the exit-0
assertion the *last* thing the step does. Decide explicitly; do not assume.

**B3 (SCOPE BLOCKER) -- pending P1 step 75.11.4 already owns this fix**, with the same
anchors and the preferred remedy already chosen. Shipping 81.1 as scoped creates two steps
editing `scripts/housekeeping/**` with divergent criteria. **Mitigation:** either execute
75.11.4 (and let 81.1 cover only the parts 75.11.4's BOUNDARY excludes -- notably
`.claude/hooks/archive-handoff.sh`, which is *outside* 75.11.4's "scripts/housekeeping/**
+ its tests only" boundary), or explicitly supersede 75.11.4 with a recorded rationale.
Note the boundary gap is real and useful: 75.11.4 cannot legally touch the hook it
diagnoses.

**B4 -- a date must not parse as a step id.** Nine files in `handoff/current/` contain
`2026-07-2x`. A naive `_(\d+(\.\d+)*)` widening parses `2026` as a sid, and because `2026`
resolves to no status, `verify_handoff_layout.py` **stops reporting them entirely** --
turning a correct "stray" report into silence. Guard explicitly, or require the parsed sid
to exist in the masterplan before honouring it.

**B5 -- the ambiguous branch lies in its own output.**
`backfill_handoff_archive.py:174` moves ambiguous files to `misc/`, then `:211` prints
`"Ambiguous (left in current/ for manual review):"`. Fix the message or the behaviour;
today an operator reading the summary will not know the file left `current/`.

**B6 -- `census_78.json` is still swept after the fix.** sid `78` is a phase, not a step
(`_step_statuses()` only reads `phases[].steps[].id`), so it lands in the ambiguous bucket
and still moves -- while step 78.0's immutable verification command opens it from
`handoff/current/`. The masterplan-reference guard (75.11.4's criterion) is what actually
fixes this; the regex widening alone does not.

**B7 -- `_read_handoff_file`'s substring glob will re-point the Harness tab.**
`backend/api/backtest.py:1386` globs `archive/**/*contract*.md` and picks newest-mtime.
Moving 86 per-step artifacts into the archive adds 86 competitors to the `/backtest`
Harness tab's contract/critique panels. Any UI claim about this step needs a Playwright
capture per the Critical Rules, not a code read.

**B8 -- `harness_state_reader.py` will raise, not degrade, on a full miss.**
`_resolve_handoff_file` is typed `-> Path | None`; callers at `:82`, `:99`, `:112`, `:125`,
`:138` call `path.exists()` with no `None` guard -> `AttributeError`. Masked today only
because the rolling files still exist. Queue as its own defect (do not fix inline).

**B9 -- `smoke_test_4_17_2.py:60` encodes a THIRD convention** (`phase-*-research-brief.md`)
and is red today. Any "reconcile the convention" claim that does not name this file is
incomplete. Its owning step 4.17.2 is `done` with a `superseded_record`.

**B10 -- `.claude/skills/masterplan/SKILL.md:222` instructs writing to `handoff/`, not
`handoff/current/`.** A fourth drift, in the boot-path skill that tells Main where to write.
Nothing reads that path.

**B11 -- no test pins `STEP_ID_RE` or the archive glob.** Zero safety net; also zero
existing test to break. Whatever this step writes is the first, so mutation-test it in both
directions (memory: "a guard that can't fail doesn't count").

---

## 9. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**8**)
- [x] 10+ unique URLs total (**37**)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set -- arXiv fetched via the
      `/html/` chain per `.claude/rules/research-gate.md`, never `/pdf/`
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (2 dry rounds, see below)
- [x] Contradictions / consensus noted (RFC 9413 vs Fowler on how long dual acceptance lives)
- [x] All claims cited per-claim

**Coverage (audit-class, K=2):** Round 1 = `STEP_ID_RE` grep + `glob|iterdir|listdir|scandir|rglob|fnmatch|os.walk` over py/sh/js/ts/mjs + hooks-dir + workflows enumeration + `live_check_|contract_|research_brief_|evaluator_critique_|experiment_results_` grep + CI + backend api/services/slack_bot/meta_evolution + frontend + tests. Round 2 (dry) = `re.compile` with dotted-number patterns; string surgery (`rsplit`/`split`/`removeprefix`/`stem`/`basename`) in `scripts/housekeeping` + `.claude/hooks`; f-string `handoff/...` construction repo-wide. Round 3 (dry) = frontend Harness tab data path, `scripts/qa/**`, `scripts/go_live_drills/**`, `.claude/skills|agents|rules`. Rounds 2 and 3 surfaced **zero new consumers** beyond `scripts/qa/sweep_absent_verification_paths.py` (found in round 2 and already tabled). `dry_rounds = 2` -> `coverage.dry = true`.

## 10. Envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 29,
  "urls_collected": 37,
  "recency_scan_performed": true,
  "internal_files_inspected": 24,
  "coverage": {
    "audit_class": true,
    "rounds": 3,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "Root-cause claim CONFIRMED for (a) verifier false strays and (b) backfill sweep-to-misc; CONFIRMED-WITH-CORRECTION for (c): archive-handoff.sh also COPIES four bare rolling filenames into the closing step's dir, fabricating false archives, which glob-widening does not fix. Measured: 128 .md in handoff/current, STEP_ID_RE matches 0, verifier exits 1 with 128 violations, backfill dry-run would move 125 files to misc TODAY. Under a widened regex that drops to 109 moved, 86 correctly routed to archive/phase-<sid>/ and 16 in-flight pending artifacts rescued -- strictly less destructive than the status quo. Blockers: widening archive-handoff.sh's MOVE glob disarms the live_check gate and re-darkens the verdict gate as a RACE (Anthropic hooks doc: all matching hooks run in parallel); 81.0's verification command already ends in verify_handoff_layout.py which exits 1 today and still exits 1 after the regex fix; pending P1 step 75.11.4 already owns this entire fix by name and its criteria conflict with the live_check gate; nine 2026-07-2x dated files parse sid=2026 and would go SILENT in the verifier; census_78.json is still swept and breaks step 78.0's immutable verification command. No test pins STEP_ID_RE or the archive glob.",
  "brief_path": "handoff/current/research_brief_81.1.md",
  "gate_passed": true
}
```
