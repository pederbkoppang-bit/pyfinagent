# Evaluator critique -- 66.4 Credential-expiry resilience (Cycle 69, 2026-07-07)

Q/A agent (merged qa-evaluator + harness-verifier), single spawn, operator present.
Prior-CONDITIONAL count for 66.4 = 0 (verified: no `phase=66.4` result lines in
handoff/harness_log.md; no prior evaluator_critique_66.4.md existed before this file).

## VERDICT: PASS

All three immutable criteria met with live, server-side-verifiable evidence.
Deterministic checks reproduced by this agent, not taken from the results file.

---

## 1. Harness-compliance audit (5-item, ran first)

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher before contract | PASS | research_brief_66.4.md mtime 00:49 precedes contract_66.4.md 00:52; envelope `gate_passed: true`, tier moderate, 5 read-in-full / 29 URLs / recency scan true / 8 internal files (brief lines 290-304). Contract's research-gate summary reflects load-bearing findings (401-envelope signature incl. `subtype:"success"` trap; keychain-only storage; latch-not-tail-1 dedupe). |
| 2 | Contract before generate | PASS | mtime chain: contract 00:52 -> scripts/away_ops/*.sh 00:57 -> experiment_results 00:59; all first-committed together in fa2ef0f0 (2026-07-07 00:59:46 +0200). Drill timestamps (22:56:48Z-22:58:16Z = 00:56-00:58 local) fall after the contract write, consistent. |
| 3 | experiment_results present, verbatim, honest | PASS | experiment_results_66.4.md has verbatim immutable-command output, an Honest-disclosures section disclosing a REAL bug the drill caught (shell `true`/`false` interpolated into Python latch writes -> `NameError`), the pre-fix extra page (p1783378639651029), drill artifacts left in the real session namespace, and 3 real channel pages. Fix verified in committed code by this agent: `$([ ... ] && echo True || echo False)` present at healthcheck.sh:151 AND run_away_session.sh:196 (grep run directly). |
| 4 | Log-last | PASS | No `phase=66.4` / Cycle 69 entry in harness_log.md at evaluation time. |
| 5 | No verdict-shopping | PASS | No prior evaluator_critique_66.4.md; first Q/A spawn for this step. |

## 2. Deterministic checks (run by this agent)

**(a) Immutable command, verbatim** --
`bash scripts/away_ops/healthcheck.sh --help 2>/dev/null || grep -n 'auth' scripts/away_ops/healthcheck.sh | head -5`
-> **exit 0** (healthcheck ignores argv, ran fully; the `|| grep` leg never executed).
My run appended a fresh line to handoff/away_ops/health.jsonl:
`{"ts":"2026-07-06T23:03:21Z","ok":true,...,"auth_ok":"true","auth_detail":"ok","auth_p1":false,...}`
-- the real credential is healthy; the probe passes with three 401 session files
still on disk (all older than the latch `cleared_at`), which is itself the
no-re-page guard working. NOTE: `HEALTHCHECK_TEST_AUTH_P1` was NOT set (per drill
safety; that flag sends a real page).

**(b) Criterion 1 -- healthcheck auth probe.** Code inspection, all five sub-points:
- (i) Two layers: `claude auth status` (healthcheck.sh:96) + newest `session_*.json`
  401 scan (:110-120). Confirmed.
- (ii) Auth-vs-network: incident opens ONLY on `"api_error_status": 401` in the
  newest session JSON (:117); ECONNRESET-class failures carry no such field.
  `claude auth status` is local-only and cannot network-fail. Confirmed.
- (iii) Page-once via latch, NOT tail-1: pages only when `incident_open` is not true
  (:125-127), then writes the latch (:151); the pre-existing tail-1 dedupe (:205-211)
  is untouched and not used for auth. Confirmed.
- (iv) `cleared_at` mtime guard: 401 session flagged only if `mt > cleared_at`
  (:118-119) -- stale 401s cannot re-open a cleared incident. Proven LIVE by my own
  run in (a). Confirmed.
- (v) Drill isolation: `HEALTHCHECK_TEST_AUTH_P1=1` branch (:144-148) does real
  delivery, writes NO latch, sets no `auth_p1` (62.5 doctrine). Confirmed.
- Server-side read-back (python, read-only `conversations.history` on C0ANTGNNK8D,
  run by this agent): `[DRILL 66.4]` page at **ts 1783378608.623569** (exact match);
  wrapper AUTH-DEAD pages at **1783378639.651029** (pre-fix) and **1783378676.110199**
  (post-fix); **zero** auth/credential/drill messages after 22:58:00Z cutoff
  (1783378680) -- run 2 provably did not page. Permalinks in live_check_66.4.md
  match these ts values. `auth_ok=false` fails the healthcheck run (:263). "Daily
  probe" is satisfied by the 30-min away-watchdog cadence (exceeds daily; disclosed
  in contract). **Criterion 1 MET.**

**(c) Criterion 2 -- wrapper page-once-and-skip.** Code inspection:
401 branch keys on `rc -ne 0` AND `grep '"api_error_status": *401'` on the session
JSON (run_away_session.sh:181), never on subtype (comment :176-180 codifies the
research finding). Latch-probe gate (:137-153): latch open -> single
`gtimeout -k 5 20`-capped `claude -p ping` probe; success clears latch + proceeds
(auto-recovery, :145-147); failure logs and exits 0 with NO page (:149-151).
Committed session.log excerpt verified by this agent (grep, lines 792-826):
run 1 `AUTH-DEAD paged (delivered=true); latch OPEN` (22:57:56Z), run 2
`AUTH-DEAD latch active + probe still failing (rc=1) -- session skipped` ->
`END session result=auth-dead-skip` (22:58:00Z), no page (read-back in (b) confirms).
No retry-forever burn: run 2 cost one 20s probe, zero full launches. Live latch
state was NOT mutated by this evaluation (skip leg verified via committed log +
code, per spawn instructions); current latch read-only:
`{"incident_open": false, "cleared_at": "2026-07-06T22:58:16...", "cleared_by": "drill_cleanup_66.4..."}`.
Drill stub envelope verified byte-level: `session_pm_20260706T225718Z.json` contains
`"subtype":"success","is_error":true,"api_error_status":401` labeled
`[DRILL 66.4 stub]` -- the exact researched signature. **Criterion 2 MET.**

**(d) Criterion 3 -- OR-arm (infeasibility).** docs/runbooks/
credential-expiry-monitoring.md argues infeasibility FROM the credential storage
format, as the criterion requires: keychain item `Claude Code-credentials` is the
only store (`~/.claude/.credentials.json` ABSENT on this machine); the sole exposed
expiry is `claudeAiOauth.expiresAt` = the 8-hour ACCESS-token expiry (predicts
nothing >=24h out); the refresh token's expiry is unexposed and its lifetime
unpublished; the observed failure matched the #61912 corruption class no timer
predicts. A lagging staleness proxy is documented and reasoned-rejected. The
structural mitigation (`claude setup-token`, 1-year) is correctly routed as an
operator decision: SETUP-TOKEN ask present in handoff/away_ops/pending_tokens.json
(raised_by "66.4 (Cycle 69)"), not auto-applied. **Criterion 3 MET via the OR-arm.**

**(e) Scope** -- `git show --stat fa2ef0f0`: 13 files = 2 scripts
(healthcheck.sh +88, run_away_session.sh +54), runbook (NEW, 74 lines),
handoff/away_ops/* (latch, probe output, health.jsonl, pending_tokens, 2 drill
stubs), handoff/current/* (contract, results, live_check, research brief).
NO backend python, NO sentinel.sh, NO plists. Matches the contract's scope
boundary exactly.

**(f) Criteria integrity** -- all 3 contract criteria byte-identical to
masterplan phase-66/66.4 (whitespace-normalized for md hard-wraps; programmatic
check 3/3 True) + verification command identical.

**(g) Syntax** -- `bash -n` clean on both scripts.

## 3. Code-review heuristics (5 dimensions evaluated; no BLOCK, no WARN)

- secret-in-diff: none -- both scripts extract SLACK_BOT_TOKEN from backend/.env at
  runtime (healthcheck.sh:128, run_away_session.sh:185); no literals in diff.
- command-injection / insecure-output: interpolated `python3 -c` one-liners use
  controlled values only (`$ts`, `$auth_detail` from a fixed value set, basename of
  a self-generated filename). Fragile-but-safe; flagged as NOTE below.
- Trading-domain invariants: no financial logic, kill-switch, stop-loss, or
  perf-metrics surface touched (scope check (e)).
- Anti-rubber-stamp: the drill is a REAL planted-violation test -- a stub `claude`
  emitting the exact production 401 envelope, detection confirmed, state restored,
  and it CAUGHT a real bug whose pre-fix page is disclosed with permalink rather
  than hidden. This is the strongest form of mutation-resistance evidence this
  harness produces. financial-logic-without-behavioral-test: N/A (no financial
  logic). tautological-assertion: none.
- LLM-evaluator anti-patterns: first spawn, no prior verdict to flip; every claim
  above carries file:line or ts-level citation.

## 4. Sequencing ruling (requested): ACCEPTABLE INTERLEAVING -- not a breach

Rule (goal_phase66_reactivation.md:127): "Priorities (strict order; do not start
N+1 while N has an unmet P0 criterion)". Facts: 66.1 (P0) is CONDITIONAL solely on
a wall-clock-gated criterion (scheduled 18:00 UTC cycle writes the BQ rows; the
39.1 lesson forbids manual substitution); 66.2 (P0, depends_on 66.1) has NOT been
started; 66.4 (P1) carries masterplan edge `depends_on_step: "66.0"` (done).

Ruling grounds:
1. The masterplan's per-step dependency edges are the machine-readable expression
   of the same goal author's intent (installed in the same commit, 68909af1). A
   strict-linear reading would render 66.4's deliberate `depends_on 66.0` edge
   meaningless; the more specific provision governs.
2. The rule's protective purpose -- never abandon a P0, never build on an
   unvalidated P0 -- is intact: 66.1's remaining evidence arrives autonomously at
   18:00 UTC (no work exists that could advance it now), and the P0 advance point
   (66.2) is untouched.
3. Proactive disclosure: the interleave was flagged for Q/A review in the Cycle-68
   log entry (harness_log.md:27120) and in the contract header -- not silent drift.
4. Precedent: Cycle 58 interleaved stale-step 17.4 during phase-62.x
   (harness_log.md:27018), operator present, PASS.
5. 66.4 remediates the same audit-basis incident as the P0s (17-day credential
   silence); the detection layer being live sooner reduces recurrence risk during
   the very wait it fills.

**Boundary of this ruling**: it does NOT license starting 66.2 (or anything
depending on 66.1) before 66.1 closes -- that would breach the rule's core. It also
does not generalize to interleaves lacking (a) an independent masterplan edge,
(b) a wall-clock-blocked P0 with no available work, and (c) proactive disclosure.

## 5. Notes (PASS-with-flag; no verdict impact)

1. **Undelivered-page edge**: if Slack delivery fails at incident open, the latch
   still opens with `paged: false` and neither detector retries delivery
   (healthcheck.sh:151, run_away_session.sh:196). Compensating controls: every
   30-min healthcheck run keeps failing (`auth_ok=false` -> exit 1) and errors land
   in healthcheck_err.log; the bot-token path is live-proven. Residual risk is
   Slack-down-at-open. Acceptable; consider a `paged:false`-retry in a future step.
2. **Channel noise observed during read-back**: unrelated `[P1] Data freshness
   critical` messages (paper_trades last_tick_age ~3.2 days; historical_macro)
   repeat ~every 60s in C0ANTGNNK8D -- a pre-existing cycle_health alerting-dedupe
   gap, and the paper_trades staleness is precisely the 66.2 subject. Out of 66.4
   scope; surfaced for the operator.
3. Probe cap shipped 20s vs the contract plan's 10s -- non-criterion implementation
   detail, consistently documented in results + runbook.
4. `auth_ok="unknown"` (probe_error) deliberately does not fail the run or page
   (fail-open, commented at healthcheck.sh:263). Reasonable to avoid false P1s;
   a persistent-unknown detector could be a future hardening.

## JSON verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met. C1: two-layer auth probe with 401-only incident opening, latch dedupe, cleared_at guard proven live by this agent's own healthcheck run (exit 0, auth_ok=true with three stale 401 files on disk); drill pages verified server-side at exact ts. C2: wrapper rc!=0+401 branch pages once then probe-and-skips; run-2 skip with zero pages verified via committed session.log and channel read-back. C3 OR-arm: infeasibility argued from the credential storage format (8h access-token expiresAt only; refresh expiry unexposed/unpublished; corruption class untimeable) + SETUP-TOKEN operator ask filed. The drill caught and disclosed a real latch-write bug (fix verified at healthcheck.sh:151, run_away_session.sh:196). Sequencing ruled acceptable interleaving (independent masterplan edge, wall-clock-blocked P0 with no available work, 66.2 untouched, disclosed, Cycle-58 precedent); boundary: 66.2 must not start before 66.1 closes.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit_5item", "syntax_bash_n", "verification_command_verbatim_exit0", "health_jsonl_tail", "criterion1_code_inspection_5subpoints", "slack_server_side_readback", "session_log_forced401_excerpt", "runbook_or_arm_review", "pending_tokens_setup_token_ask", "git_show_scope_fa2ef0f0", "criteria_byte_identity_3of3", "research_gate_envelope", "code_review_heuristics", "sequencing_ruling", "mutation_test_drill_review"]
}
```
