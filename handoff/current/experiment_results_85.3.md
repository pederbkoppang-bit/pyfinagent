# Experiment results — Step 85.3: away-ops auth alarm unstuck (latch freshness + reachable clear)

Date: 2026-08-07 (autonomous drain, cycle 175). Contract: `contract_85.3.md`.

## What was built

1. **`scripts/away_ops/auth_state.py`** (new, criterion 1) — the auth-derivation seam extracted from the healthcheck.sh python heredoc, identical two-token stdout contract. Three legs (contract D3): **L1** evidence max-age — a 401 older than `--window-s` (default 129600s = 36h; derivation in the module docstring: 2× the worst-case ~18.5h normal mtime gap, tolerating one fully missed slot; the 26h Nagios worked example recorded and rejected for ~7.5h headroom) no longer holds the latch, with the honest detail `stale_401_ignored_<file>`; **L2** the active `claude auth status` re-check stays an independent un-aged failure leg; **L3** the latch stays explicit (never self-resolving), and the CLEAR transition (`--apply`) fires ONLY on strict-true — **DD-5 fixed in-scope, STRUCTURALLY** (the clear moved inside the seam where a probe error cannot reach it; the strict-true guard is defence-in-depth, now ALSO directly test-driven after the cycle-1 Q/A showed the subprocess test crashed before reaching it). Fail-open: any seam error prints `unknown probe_error`, exit 0.
2. **`scripts/away_ops/healthcheck.sh`** — the heredoc replaced by the seam call (call site quoted below), keeping the `|| echo "unknown probe_error"` wrapper verbatim; the now-redundant elif clear arm neutralized (ONE writer for the clear, in the seam). The health.jsonl printf path is untouched (do-no-harm D6: rotate_logs' second watchdog pages on 2h staleness — records keep flowing).

```
APPLY_FLAG="--apply"; [ "${HEALTHCHECK_TEST_AUTH_P1:-0}" = "1" ] && APPLY_FLAG=""
read -r auth_ok auth_detail <<< "$(python3 "$REPO/scripts/away_ops/auth_state.py" --ops "$OPS" --state "$AUTH_STATE" --status-ok "$auth_status_ok" $APPLY_FLAG 2>/dev/null || echo "unknown probe_error")"
```

3. **`backend/tests/test_phase_85_3_auth_latch_freshness.py`** (new; 11 tests in cycle 1, 13 after the cycle-2 additions) — subprocess-driven over tmp_path only (C2, with the no-repo-fallback hygiene test); the four C3 cases; the C4 regression built NOW-RELATIVE with injected `--now` (the research brief's "single most important fixture decision" — a hardcoded mtime would drift) over ≥3 distinct-mtime session files and both JSON spellings; C5 positive + the DD-5 negative + the fresh-401 latch-exit guard; C6 missing/malformed/chmod-000; the L2 leg test.
4. **DD-6 runbook line corrected** (`docs/runbooks/credential-expiry-monitoring.md:73` falsely promised the session probe clears the latch — it never could while probe rc was never 0).

## Verification (cycle-1 capture — superseded; current suite = 13 passed, see the cycle-2 Follow-up; labelled per the cycle-2 verdict's N1)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_85_3_auth_latch_freshness.py -q
...........
11 passed in 0.41s
```

`bash -n scripts/away_ops/healthcheck.sh` → syntax OK.

## Mutation matrix — 7/7 KILLED (M1-M7 from the contract; runner in scratchpad; anchors count==1; restore hash-verified)

| id | mutation | result |
|---|---|---|
| M1 | remove the freshness bound (`fresh = True`) | KILLED (3 failed — stale-401 + regression red, fresh-401 green: criterion 7's exact requirement) |
| M2 | invert the age comparison | KILLED (5 failed) |
| M3 | window → ~100y | KILLED (3 failed) |
| M4 | window → 0 | KILLED (2 failed — fresh-401 wrongly stale) |
| M5 | drop the active re-check leg | KILLED (1 failed — L2 is load-bearing, not decorative) |
| M6 | clear unconditional | KILLED (1 failed — the latch-exit guard: no clear while a FRESH 401 stands) |
| M7 | fail-open → raise | KILLED (2 failed) |

Criterion 7's named mutation is M1; its verbatim output shape at cycle 1: `3 failed, 8 passed` (current 13-test suite: `3 failed, 10 passed` — same red/green membership, re-derived by the cycle-2 Q/A's own MQ-1; N2) with the stale-401 and regression cases red and the fresh-401 case green.

## The live outcome (criterion 8 — see `live_check_85.3.md` for verbatim captures)

After **474 consecutive false alarms over 9.87 days** [figures corrected cycle 2 per the Q/A's re-derivation; my cycle-1 473/9.85 and 34/509 were off-by-one window errors] (the step's "every record for 27 days" premise was REFUTED and re-measured: 34 of 510 parseable records were ok:true; the watchdog itself was dead 07-06..07-28; auth was the sole failing leg in all 474), a real invocation now emits `ok:true, auth_ok:"true", auth_detail:"stale_401_ignored_session_am_20260710T053005Z.json"`, exit 0; the 28-day-old latch cleared (`cleared_by: healthcheck_healthy`); the sanctioned 66.4 drill delivered a real page with the latch untouched. The paging path for a REAL 401 is intact three independent ways (only leg (a)'s INPUT is age-bounded).

## Criterion 9 — why session production stopped (recorded + filed)

The away sessions never stopped being SCHEDULED — they skipped on `incident_open` twice daily and exited 0. The probe that could have recovered them conflates "CLI exited nonzero" with "credential dead": today's probe authenticated (billed 50,373 cache-creation tokens on claude-opus-4-8) yet exited rc=1 (`error_max_turns` — a 1-turn `ping` makes the model reach for a tool) or rc=124 (20s gtimeout vs the cache rebuild); rc was NEVER 0 in 28 days of session.log. **Filed as step 85.3.1 (P1)** — and it stays armed: the next real 401 would reproduce this outage until it lands (stated in the live_check, not hidden).

## Discovered defects (all queued/filed per the standing rule)

- **85.3.1** (P1): the probe predicate — criterion-9's cause (above).
- **85.3.2** (P2): DD-2 (health.jsonl line 37 is invalid JSON `"api_health":000000` + scheduler.py:513 bare-except silently deletes the digest health section) + DD-3 (formatters reads `last_cycle_age_h`, healthcheck emits `cycle_age_h` — the digest always prints '?').
- **85.3.3** (P1, SECURITY): DD-4 — the watchdog plist embeds a literal CLAUDE_CODE_OAUTH_TOKEN; ask #12 filed (rotation decision pending the git-history exposure check).
- DD-5 fixed in-scope (strict-true clear); DD-6 fixed in-scope (runbook line).

## Files changed

`scripts/away_ops/auth_state.py` (new), `scripts/away_ops/healthcheck.sh` (call-site swap + elif neutralized), `backend/tests/test_phase_85_3_auth_latch_freshness.py` (new), `docs/runbooks/credential-expiry-monitoring.md` (DD-6 line). Live state files mutated BY the fix as intended: `handoff/away_ops/auth_page_state.json` (latch cleared), `handoff/away_ops/health.jsonl` (healthy records appended). Masterplan: +3 pending steps (85.3.1/85.3.2/85.3.3). Handoff: contract, research brief, live_check, this file, ask #12.


## Follow-up — cycle 2 (2026-08-07, after Q/A CONDITIONAL wf_9bdd4eb6-03d)

All three findings closed: (1) BLOCK — the F401 (`sys` leftover from the heredoc) removed and the lint gate run over the git-derived scope: **"All checks passed!"** (recorded; the gate had not been run in cycle 1 — same miss as 61.2's cycle, now twice today, banked to memory). (2) WARN — DD-5's "test-proven" misattribution corrected in both artifacts AND made true: `test_c5_dd5_apply_transition_guard_driven_directly` drives `apply_transition` with 'unknown'/'false'/'true' directly; the Q/A's MQ-A mutant (guard relaxed to clear on 'unknown') now **KILLED**. (3) WARN — drill isolation restored STRUCTURALLY: `--apply` is withheld in drill mode (`APPLY_FLAG` gated on HEALTHCHECK_TEST_AUTH_P1 before the seam call), so the 62.5 contract holds by construction rather than by comment. Plus: the MQ-C survivor (after_clear re-page-suppression uncovered) now has a direct test and the mutant **KILLED**; the NOTE-level streak figures corrected (474/9.87; 34 of 510 parseable). Suite after cycle 2: **13 passed in 0.43s**; matrix M1-M7 re-run WHOLE: **7/7 killed**; `bash -n` clean.
