# Experiment results — Step 84.1: memory-link auditor reconciliation (tooling only)

Date: 2026-08-07 (autonomous drain, cycle 176). Contract: `contract_84.1.md`.

## What was built

1. **`scripts/housekeeping/audit_memory.py` rewritten** (resolution only — the NO POINTER / DANGLING POINTER / MALFORMED FRONTMATTER checks are logic-unchanged): the ordered resolution LADDER (exact → same-dir normalized via stem OR frontmatter-`name:` alias → cross-corpus → unresolvable, so the reported class is deterministic); normalization = casefold + '-'→'_' folded BEFORE stripping AT MOST ONE type prefix, with the implied consequence (feedback_x ≡ project_x) stated in the docstring; frontmatter `name:` via a LINE-ANCHORED REGEX (yaml.safe_load raises on 2 real main-corpus files — the yaml-hostile fixture proves no crash); the AMBIGUOUS class reports all candidates and PASSES (0 colliding keys measured today — a guard, not a behaviour change); bounded link regex + fenced/inline code stripped (kills the `[[nodiscard]]` and multi-line-greedy false positives); `--sibling` (repeatable) for tests, production default = the known three corpora minus the audited; the `:30` dead `is_dir()` guard fixed; the `memory files: N   pointers: M` header kept VERBATIM (external consumer); "BROKEN WIKILINK" retired (verified un-grepped); closing census line added.
2. **Exit semantics per contract D5**: UNRESOLVABLE or NO POINTER → 1; NORMALIZED/CROSS-CORPUS/AMBIGUOUS report-only; DANGLING POINTER and MALFORMED FRONTMATTER **keep failing** — criterion 3 names neither, and a literal reading would disarm the founding 2026-07-26 check; every criterion-3 fixture carries zero of both so BOTH readings agree, and the dangling check has its own regression test.
3. **`backend/tests/test_phase_84_1_memory_link_resolution.py`** (new; 15 tests at cycle 1 — 17 after the cycle-2/3 additions in the Follow-ups below): the six criterion-5 cases; the criterion-3 truth table BOTH directions; the D4 ambiguous case; the D5 dangling regression; the yaml-hostile-frontmatter no-crash case; the D6 code-span case; criterion-6 structural-only over the three REAL corpora (exit∈{0,1}, no Traceback, header + census present — counts deliberately unasserted, the corpora drift daily); criterion-8 read-only proof (bytes + st_mtime_ns of every file unchanged across a FAILING run, file-set unchanged).

## Verification (cycle-1 capture, 15 tests — SUPERSEDED; the current capture is regenerated in the cycle-3 Follow-up below)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_84_1_memory_link_resolution.py -q
...............
15 passed in 0.57s
```

Lint gate over the git-derived scope (run BEFORE the Q/A this time — the twice-today lesson applied): one F401 (`os`) found and removed pre-spawn; final: **"All checks passed!"**

## Mutation matrix — cycle-1 run, 5/5 KILLED (SUPERSEDED by the cycle-3 re-run on the final tree, below; anchors count==1, restores hash-verified)

| id | mutation | result |
|---|---|---|
| m1 | drop the normalization fold (exact-filename-only) | KILLED (6 failed — incl. ≥2 named fixture cases, exact-match case green: criterion 7's exact requirement) |
| m2 | drop the frontmatter-name alias lookup | KILLED (1 failed — the named alias case) |
| m3 | unresolvable reported but not failing | KILLED (3 failed — C3 direction B) |
| m4 | normalized findings made failing | KILLED (7 failed — C3 direction A) |
| m5 | auditor writes a marker file | KILLED (1 failed — the C8 read-only proof) |

## Live behaviour over the real corpora (criterion 6's subject; counts recorded as of TODAY, not asserted — they drift)

- main: `links: 165 exact, 7 normalized, 1 cross-corpus, 3 unresolvable` → exit 1 on **3 GENUINE unresolvables** ([[link]]/[[wikilinks]] inside masterplan_state.md's machine-generated prose and a [[4000.10]] step-id link) — down from 13 reported problems pre-fix, with the false-breakage class eliminated.
- qa: `10 exact, 55 normalized, 4 cross-corpus, 2 unresolvable` → the 54-problem false wall is now 2 genuine findings.
- researcher: `56 exact, 65 normalized, 32 cross-corpus, 4 unresolvable` + 3 NO POINTER → exit 1 for real reasons.
- All three corpora STILL exit 1 — correct and expected (the research projection said so); the remaining findings are genuine and belong to the corpus owners (84.2 covers the researcher NO POINTERs).

## Disclosures

- Criteria 8/9 honoured on their subject (no .md memory file authored/modified/deleted by this step) — [CORRECTED cycle 2, Q/A finding B1: my cycle-1 disclosure was WRONG twice over: the m5 write-mutant run had leaked three 1-byte `audit_marker.tmp` files into ALL THREE LIVE corpora via `test_c6_real_corpora_structural` (which runs against the real directories), and the `git status` citation offered as proof structurally could not see the main-corpus marker (outside the repo). All three markers deleted; the mutation runner now deselects the real-corpora test (`-k 'not real_corpora'`) during write-mutants, and m5 re-ran isolated with zero leaks verified.]
- The step's 2026-08-06 figures (7/54/71) were stale by measurement time and are quoted only as history; today's taxonomy (N=406: 230 exact / 127 normalized / 37 cross / 12 unresolvable, of which 3 were regex artifacts now excluded by D6) carries its denominator rule.
- The criterion-3 dangling-pointer reading (contract D5) is the one deliberate interpretation in this step — recorded prominently for the Q/A.

## Files changed

`scripts/housekeeping/audit_memory.py` (rewritten), `backend/tests/test_phase_84_1_memory_link_resolution.py` (new). Handoff: contract, research brief, this file. No masterplan changes this cycle.


## Follow-up — cycle 2 (2026-08-07, after Q/A CONDITIONAL wf_bc013c91-e82)

Both findings closed with executed proofs: **B1** — the three leaked `audit_marker.tmp` files (my m5 write-mutant escaping through the real-corpora test) deleted from all three live corpora; the disclosure corrected above; m5 re-run with `-k 'not real_corpora'` → KILLED with **zero live-corpus leaks verified**. The class lesson (a mutation runner must isolate any test that touches real state) noted for the runner pattern. **B2** — the case-folding leg now has a failing guard: new `test_c5_case_folding_same_directory` ([[Feedback-Beta-Thing]] → feedback_beta_thing.md); the Q/A's X4 mutant (`.casefold()` deleted) EXECUTED and **KILLED** (was an equivalent-on-today's-data survivor). Suite after cycle 2: **16 passed in 0.60s**; lint gate re-run: clean. The N1 note (the prefix-fixture resolving via the alias path) is acknowledged; X1 kills prefix-stripping via other tests, so coverage is real — carried as a NOTE, not reworked mid-close.


## Follow-up — cycle 3 (2026-08-07, after Q/A CONDITIONAL wf_6f91ded3-f1a; streak 2 — a third CONDITIONAL auto-FAILs)

The cycle-2 Q/A verified both cycle-1 blockers CLOSED by its own execution, then found two new gaps. Both closed:

**F1 (C7's documentation clause)** — the test file now carries the required comment above the criterion-5 block naming which fixture cases go red under each C7 mutation. The names were RE-MEASURED on the final 17-test tree (not copied from the Q/A's 16-test run, since F2's new fixture changes the red-set): fold-drop turns 4 C5 cases red (`test_c5_hyphen_underscore_same_directory`, `test_c5_case_folding_same_directory`, `test_c5_cross_corpus_match`, `test_c5_stem_resolution_without_frontmatter_name`) with `test_c5_exact_same_directory_match` GREEN; alias-drop turns exactly `test_c5_frontmatter_name_alias` red.

**F2 (Y14 stem-leg survivor)** — new `test_c5_stem_resolution_without_frontmatter_name`: the target carries NO frontmatter `name:` line (valid — MALFORMED requires only `type:`), so `[[beta-thing]]` → `feedback_beta_thing.md` can only resolve through the prefix-stripped STEM index key. The Q/A's Y14 mutant (stem key keeps casefold+hyphen fold but loses prefix-stripping) EXECUTED on the final tree: **KILLED — 1 failed, exactly the new test** (was: survived all 16 while mis-classing 13 real researcher links).

### Verification (verbatim, regenerated cycle 3 after the final edit)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_84_1_memory_link_resolution.py -q
.................
17 passed in 0.63s
```

Lint gate (git-derived scope, tracked ∪ untracked, 2 files non-empty): **"All checks passed!"**; `ast.parse` OK.

### Mutation matrix — cycle-3 re-run on the FINAL tree, 6/6 KILLED (anchors count==1, restores hash-verified, md5 70052207c9e95e174b71ab03792d5577 unchanged)

| id | mutation | result (final tree) |
|---|---|---|
| m1 | drop the normalization fold (exact-filename-only) | KILLED (8 failed — the 4 C5 cases named in the test-file comment + 4 non-C5; exact-match case GREEN) |
| m2 | drop the frontmatter-name alias lookup | KILLED (1 failed — test_c5_frontmatter_name_alias) |
| m3 | unresolvable reported but not failing | KILLED (3 failed) |
| m4 | normalized findings made failing | KILLED (8 failed) |
| m5 | auditor writes a marker file (real-corpora test deselected) | KILLED (1 failed — C8) + **zero live-corpus leaks verified in-script** |
| m6 | Y14: index stem key loses prefix-stripping | KILLED (1 failed — the new F2 test) |

### Live census re-derived on the final tree (auditor byte-identical to the graded hash)

All three corpora reproduce the figures in the section above exactly (main `165 exact, 7 normalized, 1 cross-corpus, 3 unresolvable`; qa `10/55/4/2`; researcher `56/65/32/4` + 3 NO POINTER) and all three exit 1 — no drift since cycle 1; the superseded cycle-1 capture blocks above are preserved unedited per append-only discipline.
