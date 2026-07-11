# Evaluator Critique — Step 69.1 (P0 book-safety: FX, kill-switch, pkill, locks)

## Cycle 1 — Q/A verdict: PASS with 2 non-degrading NOTEs (independent Q/A, workflow structured-output on Opus)

**Verdict: PASS**, `violated_criteria: []`. All 5 immutable criteria met on independent verification.

### Harness compliance (5/5 PASS)
- research_gate: research_brief_69.1.md gate_passed=true, 5 external sources read in full (OWASP/CWE-78, Fowler,
  Python contextlib, Modern Treasury), provenance disclosed.
- contract_before_generate: PASS — **mtime-proven** (research 18:59 → contract 19:00 → code 19:02-07 →
  results 19:12); the contract genuinely preceded GENERATE (no ordering slip this cycle).
- results_present / log_last / no_verdict_shopping: PASS.

### Immutable criteria (all met)
- **C1 FX**: `return val` moved inside the truthy block; `_last_known_usd_value` queries historical_fx_rates
  DIRECTLY (no `_usd_value_asof` recursion); `execute_sell:392` returns None (BLOCK+PAGE) instead of `_l2u=1.0`;
  USD returns 1.0 unaffected; fail-safe confirmed intended + unreachable for any traded market (execute_buy
  write-throughs a rate).
- **C2 kill-switch**: `current_nav<=0` guard → no-breach + `nav_invalid` (fail-safe); a valid 20%-down NAV STILL
  breaches (guard doesn't suppress real breaches); reset_peak DARK-by-default; thresholds are caller-args.
- **C3 op-safety**: pkill + `import subprocess` removed; grep + comment-stripped test prove no process-kill sink.
- **C4 locks**: cycle_lock `acquired`-flag guards the finally (live contention raises before `acquired=True` →
  live pidfile survives); autonomous_loop guarded init releases `_lock_cm` + resets `_running`.
- **C5 do-no-harm**: single paper_trader edit at execute_sell (distinct from 68.5); fail-safe additions only;
  thresholds byte-untouched (4.0/10.0/8.0/30.0 verified live); no trading-domain BLOCK/WARN.

Deterministic gates: 12 passed, ruff exit 0, 7 modules import OK, thresholds intact. Pre-existing
ma_preannounce F821 (confirmed in HEAD, not a 69.1 regression) cleared by a disclosed do-no-harm hoist.

### The 2 NOTEs (non-degrading)
1. `reset_peak` had NO call site (defined/gated/tested but not invoked) → on activation it would never fire.
2. Two out-of-scope 69.3 flags in settings.py (additive default-OFF, inert) — a commit-hygiene note.

---

## Cycle 2 — Main remediation + fresh Q/A verdict: PASS

**Main addressed both NOTEs:**
1. **reset_peak wired into `resume(nav=...)`** (kill_switch.py:208) — an operator resume re-anchors the trailing
   peak to the current NAV, called OUTSIDE the state lock (non-reentrant), still DARK-gated. +2 tests
   (DARK-off → peak unchanged; token-on → re-anchored). Re-verified: **14 passed**, ruff **exit 0**.
2. **69.3 scaffolding disclosed** in experiment_results.md (inert, default-OFF, imported nowhere).

**Fresh cycle-2 Q/A (workflow, Opus): PASS**, `violated_criteria: []`. Independently verified:
- pytest **14 passed** (the +2 are the resume DARK-off + token-on tests); ruff F821/F401/F811 **exit 0**.
- reset_peak invoked at :208 **outside** the `with self._lock` block (lines 191-200 close before :208) → the
  non-reentrant lock **cannot deadlock** — empirically confirmed (token-ON test drove both lock acquisitions in
  <1s, not hanging to the 120s timeout).
- DARK-by-default holds (flag OFF → peak 1000.0 unchanged, still un-pauses); ON → re-anchors to 800.0.
- All 5 criteria still hold; 4/10/8/30 byte-untouched; sod-nav-anchor WARN does not fire (peak_reset ships with
  its audit-log invariant: new event + `_load_from_audit` replay).
- NOTE (non-degrading, disclosed): the resume ENDPOINT (paper_trading.py:569) still calls resume() without `nav`
  — full operator-path activation needs the endpoint update, a disclosed KS-PEAK-RESET activation follow-on;
  criterion 2 only requires the DARK-gated implementation + dark-by-default test, which is satisfied.

PASS→PASS re-affirm on genuinely improved code, verified not rubber-stamped. Log-append precedes the status flip.
