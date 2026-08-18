STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.91
WRITTEN: 2026-08-16T09:16:37Z

# Q/A cycle-3 evaluation record for step 86.91

Spawn: Workflow rail, Opus 5 (1M). Read .claude/agents/qa.md in full at runtime.

## Prior-attempt / sequence EVIDENCE (reported, not aggregated)
- `qa_wip.py 86.91 --spawned-at 2026-08-16T09:16:37Z` -> source_present=true,
  attempt_number=3, attempt_number_status=ok, attempt_number_is_lower_bound=true,
  prior_attempts=2, records_retained=3 (gauge), identity_checked=true.
- `verdict_history_86_21.py --step 86.91 --evidence-only` -> status=no_rows_for_step,
  verdicts=(none).
- CROSS-CHECK: attempt_number (3) > ledger verdict count (0) -> THE LEDGER IS STALE;
  the sequence from it is unreliable. Main's advisory [CONDITIONAL, CONDITIONAL] is
  consistent with attempt_number=3 but is advisory only. Prior critique bodies carry
  two `"verdict": "CONDITIONAL"` blocks; I did NOT word-scan bodies.
- `grep -cF "phase=86.91" handoff/harness_log.md` = 0 (correct at EVALUATE time).

## A. HARNESS COMPLIANCE -- 5/5 CLEAN
1. research-gate-before-contract: research_brief_86.91.md 21,062 B, brief_status
   COMPLETE, gate_passed true, external_sources_read_in_full=8 (floor 5),
   urls_collected=28 (floor 10), recency_scan_performed=true. Contract §1 cites run
   wf_6f758470-f84 and §4 uses the findings (I3, I5, ISSTA 2408.01760).
2. contract-before-generate (mtime, LOCAL CEST): research 09:58:08 < contract
   10:14:17 < hook 10:14:54 < checker 11:10:51. Criterion-1 reproduction is quoted
   IN the contract, i.e. before the hook edit.
3. experiment_results_86.91.md present, with cycle-2 and cycle-3 Follow-up sections.
4. log-last: 0 harness_log rows for phase=86.91; masterplan 86.90/86.91 both pending.
5. no-verdict-shopping: 468c7908 changed experiment_results (+57), live_check (+62),
   verify_changelog_flip_86_91.py (+134). Evidence CHANGED -> documented cycle-3 flow.

## B. DETERMINISTIC
- IMMUTABLE CMD `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
  -> `parses`, EXIT 0.
- `verify_changelog_flip_86_91.py` -> ALL GREEN: 34 passed, 0 failed, bare exit 0.
- `replay_changelog_rule_86_68.py` -> corpus 707 in [2026-08-11T00:00:00 .. 8dc70502],
  OLD 251 / SHIPPED 9 / FIXED 11, exit gate control_green=True all_cells_killed=True.
  707/251/9/11 REPRODUCE EXACTLY in my environment (Main's question D: YES).
- ruff F821,F401,F811 over the STEP-derived scope
  (`git diff --name-only c627a810..HEAD -- '*.py'` = replay + verify, non-empty guard
  asserted, xargs -0 so zsh cannot collapse the list): All checks passed, exit 0.
- No backend/** or frontend/** in the step diff -> gates 1b/1c/1d N/A. Uncommitted tree
  edits (sovereign_api.py + 5 frontend files) are dated 2026-08-14, present in NONE of
  the 4 step commits -> peer-session work, not attributable to 86.91.
- Criterion 1 INDEPENDENTLY REPRODUCED by me: e4f2e844 `86.86 before: None -> after:
  done`, OLD [] / NEW ['86.86']; and 8b520f6c `86.81 before: None -> after: done`,
  OLD [] / NEW created ['86.81']. Both steps are `done` today and both commits shipped
  real work (autonomous_loop.py+tests; qa-verdict.js+research-gate.js).
- Main's claim "the hook is unchanged since cycle 1": VERIFIED TRUE. post-commit-
  changelog.sh appears in 8dc70502 only; 0 hits in 952ed521 / 98c5b6ab / 468c7908.

## C. MY OWN MUTATION MATRIX (run in-memory via runpy + patched Path.read_text; no
## repo file written; CONTROL observed GREEN first: 34/0 exit 0)

| Cell | Mutation | Outcome |
|---|---|---|
| Q1 | delete the `open(...)/_fh.write(...)` in `_log_decision` | **SURVIVED** 34/0 |
| Q2b | hook `created_done` whitelisted to the 5 fixture ids (anchors preserved) | **SURVIVED** 34/0 |
| Q3 | `newly_done = list(transitioned_done)` (created_done still recorded) | KILLED, 4 red |
| Q4 | replay `created` whitelisted to the 2 fixture ids ("86.86","12.7") | **SURVIVED** 34/0 |
| Q5 | replay `created` whitelisted to "86.86" only | KILLED (1 red) |
| Q2 (discarded) | whitelist injected INTO the M1 anchor | mis-attributed kill -- rebuilt as Q2b |

corpus_head() probe, driven directly:
- CONTROL head = 8dc705022fe7a7a0ade7cc1303f57aa04b1f5e61 == resolve("8dc70502").
- QA-C2-1 mutant head = 821f256902d6d3a52422d31a1577d14a1700ce33 (current HEAD).
  => the cell IS discriminating; a genuine behavioural differential, NOT an artifact
  kill. (Main's question B: YES.)
- Refactor-robustness (Main's question A): helper hoisted above the start anchor /
  start anchor reworded `CORPUS_SINCE: str =` / end anchor reworded
  `sh(*list(_log_args))` / sliced block made to NameError -> ALL return head=None,
  which turns the CONTROL check `[5] corpus UPPER bound pinned BEHAVIOURALLY` RED.
  => it FAILS CLOSED. It genuinely DRIVES; it is not a re-implementation.

## D. FINDINGS

F1 [WARN] QA-C2-6 was MOVED, not closed, and the residual is UNDISCLOSED.
  Q4/Q2b measured: a predicate whitelisted to exactly the fixture ids survives all 34
  assertions on BOTH the replay and the hook. Q5 confirms the 1-id shape now dies, so
  the cycle-2 fix works for its stated shape. Artifact says the class is covered
  without stating that a whitelist superset of the fixture still passes -- against this
  step's own twice-stated doctrine ("closed against the shapes I enumerated"; the
  5th-branch bound). FIX: state the bound, or drive with a RUNTIME-GENERATED id that
  appears in no source literal, which closes it rather than moving it.

F2 [WARN] criterion 4's OUTPUT MECHANISM is unguarded -- MEASURED SURVIVOR (Q1).
  Every [2] assertion inspects the in-memory `_FLIP_DECISION` dict; nothing asserts a
  line reaches handoff/logs/changelog-decisions.log. Deleting the write leaves the
  checker ALL GREEN 34/0. Vacuity shapes #1/#3. MITIGATION (why WARN not BLOCK): the
  log exists with 4 real production lines from this step's own 4 commits, so the class
  is demonstrated end-to-end TODAY; and criterion 6's mutation mandate is scoped to the
  None exclusion, which M1 covers. FIX: one cell driving `_log_decision` into a temp
  dir and reading the line back.

F3 [NOTE] the THIRD instance of "I fixed it there and left it here" (Main asked).
  Cycle 3 states `[6]` now scores DETECTED/SURVIVED/UNSCORABLE and that UNSCORABLE
  FAILS. The QA-C2-1 cell takes the `probe is None` branch, scored
  `DETECTED if (mh is None or mh != resolve(_pin))`, and corpus_head swallows its own
  exceptions (`except Exception: return None`, :382-383) -- so a mutant that cannot
  BUILD returns None and scores DETECTED, never UNSCORABLE. MEASURED: a NameError
  mutant of the sliced block scores DETECTED. The three-outcome fix was applied to one
  branch and not the other. Harmless today (the real cell is discriminating, above),
  but it is the exact shape this cycle claims to have closed.

F4 [NOTE] stale counts, in the cycle whose sibling finding was stale captures.
  experiment_results §1 says the checker has "**31** assertions ... **6** mutation
  cells" and §7's heading says "Mutation matrix (6 cells, all KILLED...)". MEASURED at
  HEAD: 34 assertions, 8 cells (4 in [4] + 4 in [6]); §7's own quoted block says 34,
  so §1 and §7 contradict each other inside one artifact. The two cycle-3 cells
  (QA-C2-1, QA-C2-6) are described in the Follow-up but never added to the §7 table.
  Under-claims rather than over-claims.

F5 [NOTE] the bash early-exit paths are still silent and still undisclosed (raised
  cycle 1, un-remediated at cycles 2 and 3). MEASURED: 8 hook invocations since the fix
  commit, 4 decision-log lines; the 4 missing are the `chore: auto-changelog` commits
  hitting `exit 0` at :28 before the python block. That skip is benign, but :32-38
  (CHANGELOG.md absent / "### Recent Activity" renamed) is the silent-swallow class one
  layer up -- rename that heading and every commit stops bumping with zero output.
  §5's "(bounded -- see below)" points at the return-"none" scoping, which is thin but
  is a scoping.

F6 [NOTE] criterion 1's "quoting the command": live_check §1 shows the OUTPUT but the
  command body is elided (`$ python - <<'EOF'  # statuses() at e4f2e844 ...`). I
  re-derived it independently and it reproduces exactly, so the substance holds.

## E. CRITERION MAP (8/8 MET on the product; residuals are in guards + prose)
1 MET  reproduced before the change, 2 real commits, independently re-derived. F6 NOTE.
2 MET  predicate change stated verbatim; grep over the detector body finds NO step-id
       literal. F1 is about the GUARD, not the fix.
3 MET  251 / 9 / 11 over a both-ends-pinned 707-commit corpus, reproduce exactly here;
       +2 accounted for commit by commit; both gained commits closed real steps that
       shipped real work. "348" non-reproducibility disclosed; class filed as 86.94.
4 MET-with-residual  4 real log lines, closed reason set, all 4 return-"none" branches
       driven with a source-derived denominator. Residuals F2 + F5.
5 MET  no CHANGELOG.md edit in any of the 4 step commits; the only CHANGELOG changes
       are hook-produced. Version-line half deferred to the flip and disclosed
       precisely in live_check §6 -- and its prediction (flip_transitioned, NOT
       flip_created) is CORRECT: 86.90 and 86.91 both exist at HEAD~1.
6 MET  control [0] observed GREEN before every cell; M1 restores the None exclusion and
       KILLS; 8 cells total. My Q3 independently kills 4 assertions.
7 MET  fault injected into subprocess.run: does not propagate, bumps nothing, marker
       reaches stderr. Residual: `_log_decision` is a NEW raise surface not covered by
       the checker's injection (cycle-1 N3, still open).
8 MET  86.90 and 86.91 both still `pending`; no verdict altered. masterplan.json WAS
       edited (86.92-86.95 filed pending; 86.94 criterion 1 rewritten) but that is
       86.90's scope, flips nothing and alters no verdict.

## F. VERDICT REASONING
Product code correct on all 8 criteria, as Main predicted. Worst severity across the
findings is WARN (F1, F2), both MEASURED SURVIVING MUTANTS, both with named cheap
fixes, neither sole-coverage on a money path -> per qa.md 4c wiring, CONDITIONAL.
Lenses: correctness PASS, does-it-reproduce PASS, scope-honesty CONDITIONAL (F1's
undisclosed bound, F4's stale counts, F5 undisclosed) -> min = CONDITIONAL.

COMPLETED: 2026-08-16T09:28:34Z
