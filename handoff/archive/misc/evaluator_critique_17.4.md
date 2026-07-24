# Evaluator Critique -- phase-17.4 (stale-step closure) -- Q/A spawn 1

Date: 2026-06-12. Q/A (merged qa-evaluator + harness-verifier), first spawn for this
step. Verdict: **CONDITIONAL** -- criteria 1, 2, 4 verified PASS with independently
reproduced evidence; criterion 3 ("log committed") is blocked by a concrete, fixable
defect found during deterministic checks (gitignore collision, details below).

## 5-item harness-compliance audit (run FIRST)

1. **Researcher before contract: PASS.** research_brief_17.4.md mtime 2026-06-12
   10:35:57 < contract_17.4.md 10:37:23. Brief: gate_passed=true, 5 sources read in
   full (official docs x3, anthropic repo issue, practitioner blog), recency scan with
   3 supersessions, 3-variant query discipline visible, file:line anchors throughout.
2. **Contract before generate: PASS.** contract_17.4.md 10:37:23 predates the
   verification run (harness start 10:37:34 per log line 3; log final write 10:38:06).
3. **Results artifact honest: PASS (with accepted deviation).** No per-step
   experiment_results_17.4.md exists; the dryrun log + EVIDENCE ADDENDUM + contract
   serve that role. RULING: acceptable for a verification-only closure step -- the
   GENERATE artifact's required content ("what changed" = nothing, code-wise +
   "verbatim verification output" = the log itself) is fully present in the log;
   the deviation is disclosed, not hidden. The honest-framing paragraph in the
   addendum was scrutinized and is ACCURATE (see criterion-2 ruling).
4. **Log-last: PASS so far.** No 17.4 step-closure cycle entry in harness_log.md yet
   (correctly queued behind this verdict). The two automated "DRY_RUN (composite
   0/10)" entries (harness_log.md:26990, 27007) are the harness's own per-run logging
   (one from Main's 10:37 run, one from my 10:43 rerun), not step-closure entries.
5. **No verdict-shopping: PASS.** Zero prior 17.4 Q/A entries; zero prior
   CONDITIONALs for step-id 17.4 (`grep -c 'phase=17.4.*result=CONDITIONAL'` = 0).
   This is the FIRST CONDITIONAL -- 3rd-CONDITIONAL auto-FAIL not in play.

## Deterministic checks (verbatim)

- **Immutable command re-run** (tee redirected to /tmp to preserve the committed
  artifact -- instructed, disclosed deviation from verbatim):
  `source .venv/bin/activate && python3 scripts/harness/run_harness.py --cycles 1
  --iterations-per-cycle 1 --dry-run 2>&1 | tee /tmp/qa_17_4_rerun.log | grep -c
  'mcp__alpaca' || true` -> output `0`, `pipeline_exit=0`.
  `/tmp/qa_17_4_rerun.log:52` = `HARNESS COMPLETE -- 1 cycles finished` (Sharpe=1.1705,
  DSR=0.9526, matching baseline). Reproduces the committed log exactly, including the
  grep count 0 over harness stdout (dry-run spawns no subagents -- corroborated by the
  step's own 2026-04 notes field in .claude/masterplan.json).
- **Committed log evidence:** `grep -c 'mcp__alpaca'
  handoff/current/alpaca-researcher-dryrun.log` = **3** (>=1 required). Log contains
  `HARNESS COMPLETE` (line 52 region) + EVIDENCE ADDENDUM. Account id `PA3VQZZLAKE2`
  appears in BOTH the log addendum and research_brief_17.4.md (section 3) -- matches the
  17.3-era paper account (created 2026-04-24, ACTIVE).
- **MCP plausibility:** `.mcp.json:6` pins `alpaca-mcp-server==2.0.1`; `.mcp.json:10`
  hardcodes `ALPACA_PAPER_TRADE: "true"`. `scripts/mcp_servers/smoke_test_alpaca_mcp.py`
  exists (alongside reconcile_alpaca_deny_list.py). Optional live smoke rerun SKIPPED
  (budget); not load-bearing given pin + script + account-id cross-match + the archived
  17.3 smoketest precedent.
- **Structural claim verified:** `.claude/agents/researcher.md:4` allowlist =
  `tools: Read, Grep, Glob, Bash, WebSearch, WebFetch, SendMessage` -- no mcp__ tools.
  The literal in-harness mcp__ dispatch path is configuration-closed, as the brief and
  addendum state.
- **Hazard citation verified:** `scripts/harness/run_harness.py:355`
  `(HANDOFF_DIR / "contract.md").write_text(...)` -- the rolling-contract overwrite is
  real; rolling contract.md now contains the harness-generated "Sprint Contract --
  Cycle 1" as disclosed. Per-step files (contract_17.4.md etc.) correctly used.
- **no_regressions:** post-rerun `git status --porcelain` (tracked changes) = handoff
  artifacts + audit append-only streams + researcher agent-memory only. Zero Python /
  TypeScript / masterplan-criteria mutations. Matches phase-17 vocabulary
  (phase-17.1 evaluator_critique.md:37 precedent).
- **Masterplan state:** step 17.4 (phases[35].steps[3]) `status: "in-progress"`,
  `retry_count: 0`, no `live_check` field. Criteria in contract_17.4.md match
  masterplan verbatim. Frontend gate N/A (no frontend/** in diff). Live-UI gate N/A
  (no UI claims).

## Criterion-by-criterion

1. **"harness dry-run exits 0" -- PASS.** Evidenced by `HARNESS COMPLETE` inside the
   committed log AND independently reproduced (above). Note the command's
   `grep -c ... || true` masks shell exit by construction -- the completion line is
   the correct evidence, as the contract itself discloses.
2. **"at least one mcp__alpaca* tool call recorded in the research brief or dryrun
   log" -- PASS. Explicit ruling: criterion-SATISFYING, not criterion-gaming.**
   - The binding text is the immutable criterion, not the step name. Both the brief
     (research_brief_17.4.md section 3) and the log addendum record
     `mcp__alpaca__get_account_info` + `mcp__alpaca__get_clock` with live response
     data (paper account PA3VQZZLAKE2 ACTIVE, $0 cost, deny-list untouched).
   - Authorial intent is documented IN THE STEP: the 2026-04 notes field states
     dry-run "does not spawn subagents, so 0 mcp__alpaca* calls in log as expected"
     and pre-accepts the 17.3 "equivalent tool surface" framing. The OR-arm
     ("research brief or dryrun log") exists precisely because in-process capture is
     impossible by design; a reading that demands it would make the criterion
     unsatisfiable-as-written, contradicting its author's contemporaneous notes.
   - The substantive capability 17.4 exists to prove -- a researcher subagent
     reaching the pinned Alpaca MCP server and invoking its tools -- WAS exercised:
     live MCP stdio JSON-RPC `tools/call` against `alpaca-mcp-server==2.0.1` from the
     researcher session, same account as 17.3. What was NOT used is Claude Code's
     mcp__ dispatch plumbing, excluded by researcher.md:4 -- and BOTH artifacts
     disclose this distinction unprompted. Honest provenance disclosure is the
     anti-gaming marker; gaming would be planting the string while implying
     in-process calls happened. The addendum does the opposite (explicitly reports
     harness-stdout grep = 0 and why).
   - The literal alternative (editing researcher.md's allowlist) requires operator
     review per CLAUDE.md separation-of-duties and is correctly declared out of
     scope / queued as a return-day candidate rather than smuggled in.
3. **"handoff/current/alpaca-researcher-dryrun.log committed" -- BLOCKED (the
   CONDITIONAL).** The file is NOT committed, NOT staged, NOT tracked, and -- the
   defect -- NOT COMMITTABLE by the planned mechanism:
   - `git check-ignore -v` -> `.gitignore:24:*.log handoff/current/alpaca-researcher-dryrun.log`
   - `git ls-files --error-unmatch <path>` -> "did not match any file(s) known to git"
   - The auto-commit hook's `git add -A` respects .gitignore and will SILENTLY skip
     this file at the flip; criterion 3 would ship unmet with no error.
   - Precedent confirms the failure mode is real and historic: the only tracked
     dryrun artifact is `handoff/archive/misc/alpaca-researcher-dryrun.log.test`
     (suffix evades `*.log`); the three prior attempt logs
     (`alpaca-researcher-dryrun{,-v2,-verify}.log`) sit on disk in
     handoff/archive/misc/ UNTRACKED -- three prior cycles already silently failed
     this criterion, and the 2026-04 notes-field claim "Log committed" is falsified
     by `git ls-files`. This is the VERIFICATION_DEFECT pattern the live_check gate
     exists to kill; caught here pre-flip.
   - **Required fix (deterministic, small):** `git add -f
     handoff/current/alpaca-researcher-dryrun.log` and commit it AT THE LITERAL PATH
     named by the criterion BEFORE (or in the same commit as) the status flip;
     verify with `git ls-files --error-unmatch` exit 0. CAUTION on ordering: if the
     archive-handoff hook relocates the file to handoff/archive/phase-17.4/ at flip,
     `*.log` still ignores the NEW path -- `git add -A` would then stage the
     DELETION of the tracked current/ path and drop the archived copy, turning the
     move into a delete. Commit at handoff/current first; force-add the archived
     path too if the hook moves it. Optional (audit continuity, not required by
     17.4): force-add the three prior logs in archive/misc in the same commit.
4. **"no_regressions" -- PASS.** Handoff-artifacts-only diff verified post-rerun
   (above); zero source mutations; verification criteria untouched.

## Code-review heuristics

No code diff (handoff-only step) -- dimensions 1-3 largely N/A. Checked anyway: no
secrets in brief/log/contract (account id is an identifier, not a credential; brief
explicitly avoided reading backend/.env and verified key presence via env-prefix check
only); no trading-path, kill-switch, or agent-file changes; separation-of-duties
respected. No heuristic findings. anti-rubber-stamp: the closure does NOT
rubber-stamp -- the addendum self-reports the adverse fact (harness-stdout grep = 0)
and frames the evidence provenance honestly.

## Non-blocking notes (do not gate)

- N1: Criteria that name `*.log` paths as "committed" collide with `.gitignore:24` --
  masterplan authoring hazard; prefer .md/.txt evidence filenames in future criteria.
- N2: Echoing the researcher's side observation for operator awareness (out of 17.4
  scope): paper account shows short_market_value = -$13,842.89 on a nominally
  long-only system (research_brief_17.4.md, Risks item 7).
- N3: Missing per-step experiment_results file ruled acceptable for this
  verification-only closure (audit item 3); Main should say so in the 17.4
  harness_log entry for future archaeology.

## Verdict

**CONDITIONAL** (first for this step-id). Single blocker: criterion 3 commit
mechanism (gitignore collision). On fix: force-add + `git ls-files` proof appended to
the evidence (one line in the addendum or harness_log entry suffices), then fresh Q/A
per the canonical cycle-2 flow. Criteria 1, 2, 4 need no further work.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Criteria 1/2/4 PASS with reproduced evidence (HARNESS COMPLETE reproduced at /tmp/qa_17_4_rerun.log:52; grep -c mcp__alpaca on committed log = 3; account id cross-matched; handoff-only diff). Criterion 2 ruled criterion-satisfying, not gaming: immutable OR-arm + 2026-04 notes-field intent + honest provenance disclosure + separation-of-duties bar on the literal path. BLOCKER: criterion 3 -- the log is gitignored (.gitignore:24 *.log), untracked, and git add -A will silently skip it at the flip; three prior attempt logs are likewise untracked on disk, falsifying the 2026-04 'Log committed' note. Fix: git add -f at the literal path before/with the flip, verify via git ls-files, then fresh Q/A.",
  "violated_criteria": ["handoff/current/alpaca-researcher-dryrun.log committed"],
  "violation_details": [
    {
      "violation_type": "Invalid_Precondition",
      "action": "closure plan step 4 'Commit the log (criterion 3)' relying on the status-flip auto-commit (git add -A)",
      "state": "git check-ignore -v -> .gitignore:24:*.log matches handoff/current/alpaca-researcher-dryrun.log; git ls-files --error-unmatch -> not known to git; auto-commit hook stages via git add -A which skips ignored paths",
      "constraint": "success_criteria[2]: 'handoff/current/alpaca-researcher-dryrun.log committed' (immutable, masterplan 17.4)"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "inheriting the 2026-04 notes-field claim 'Log committed at handoff/current/alpaca-researcher-dryrun.log' as established fact",
      "state": "only tracked artifact is handoff/archive/misc/alpaca-researcher-dryrun.log.test (suffix evades *.log); alpaca-researcher-dryrun{,-v2,-verify}.log exist on disk but are untracked",
      "constraint": "evidence cited for an immutable criterion must be verifiable in git history (per-step-protocol section 4; VERIFICATION_DEFECT class)"
    }
  ],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit_5item", "verification_command_rerun", "log_grep_and_account_id_crossmatch", "mcp_pin_and_smoke_script_existence", "researcher_allowlist_structural_check", "run_harness_355_hazard_citation", "git_ignore_and_tracking_state", "no_regressions_git_status", "masterplan_state_and_criteria_verbatim", "harness_log_conditional_count", "code_review_heuristics"]
}
```

## Delta re-evaluation (cycle 2) -- Q/A spawn 2, 2026-06-12 (appended verbatim by Main per the read-only Q/A's instruction)

Verdict: PASS. All six delta checks deterministic-verified: (1) git ls-files exit 0
(tracked; check-ignore now inert); (2) commit 6684c9c7 creates the log (create mode
100644, on origin/main); (3) worktree clean at the path; (4) committed blob: mcp__alpaca
x3, CRITERION-3 PROOF x1, HARNESS COMPLETE x2; (5) range ae07a48c..6684c9c7 = the log +
hook-generated CHANGELOG chore only (disclosed NOTE, non-gating); no_regressions holds;
(6) 17.4 still in-progress, zero phase=17.4 harness_log entries (log-last respected).
Spawn-1 violations RESOLVED (explicit add -f replaces the invalid add -A precondition;
the 2026-04 'Log committed' note is now actually true and its historical falsity is
documented in the committed proof). Sycophancy guard: verdict flip grounded in changed
evidence (new commit at the literal criterion path) -- canonical cycle-2, not shopping.
AUTHORIZED: harness_log entry -> flip -> post-flip force-add of the archived log copy IN
THE SAME COMMIT (the now-tracked current/ path's hook-move would stage a deletion while
the *.log-ignored archive destination stays invisible). Full envelope in the session
transcript; spawn-2 agent a320e80fb49e0ce82.
