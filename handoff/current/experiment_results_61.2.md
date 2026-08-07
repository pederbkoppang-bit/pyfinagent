# Experiment results — Step 61.2: decision-input integrity (dark build → evidence + promotion ask)

Date: 2026-08-07 (autonomous drain, cycle 173). Contract: `contract_61.2.md`. Prior Q/A: CONDITIONAL ×1 (commit 354eb6b4).

## Six-row triage (research-gate measured; the step was BUILT 2026-07-08 in commit 6186784c — 20 files + a 459-line test module — and sat dark awaiting evidence)

| # | Sub-item | Verdict | Evidence |
|---|---|---|---|
| A | never persist 0.00/HOLD | **LIVE DEFECT, fix BUILT-DARK** | Firing daily: my own derivation — **142 of 170 rows in 40d (83.5%)** are `0.0 + HOLD + $.final_synthesis.error='Failed to parse final report.'`, last 2026-08-06 19:20:56Z (live_check §B). Laundering line `orchestrator.py:2280`; guard gated by `paper_synthesis_integrity_enabled` (default False) |
| B | timeout ≥150s | **DONE, LIVE, UNGATED** | `settings.py:186-191` (`claude_code_timeout_s = Field(150, ...)`, description cites this criterion); `claude_code_client.py:591/:593/:600`. Deliberately inverts SRE deadline-propagation so the inner timeout fires first and stays retryable — a considered deviation, stated |
| C | company_name fallback | **FIXED + LIVE-PROVEN** | NULLs 6/6/5/2/5/4 per day 07-01..07-08 → **exactly 0 every day 07-09..08-06** (live_check §A). Retires prior-Q/A blocker #1 |
| D | meta-scorer PIT + WARN streak | **BUILT-DARK; 72.0.1's premise DEAD** | `meta_scorer.py:170-177/:296-300` (PIT over full head+tail); the rail bypass 72.0.1 exists to fix was already rewired by phase-78.1 (`make_client` at meta_scorer.py:242; no ClaudeClient construction remains) — 72.0.1 disposition queued separately |
| E | signal_downgrade | **DARK, strictly downstream of A** | `paper_trader.py:443-450` flag-gated; `portfolio_manager.py:114-121` WARNs on the unsafe E-without-A combination (promoting E alone would SELL healthy positions on fabricated HOLDs) |
| F | RiskJudge advisory ctx | **DARK, defect live** | `autonomous_loop.py:1045-1052` builds ctx only under either flag; both OFF → judge receives `''` today |

## What was built THIS cycle (D1 scope — no flag flips; promotion is the operator's)

1. **The two kill-switch-coupled 61.2 tests repaired** via the documented `kill_switch_state` injection seam (`_HealthyKillSwitch` stub mirroring the 36.13 idiom). Both contract-named mutations EXECUTED: flip the injected state to paused → **both tests FAIL** (the injection is live, not decorative); revert `paper_trader.py:443-450` to store the trade reason → **the criterion-5 flag-on test FAILS** (the test guards the criterion, not merely the kill switch). Module: **33 passed**; restores hash-verified.
2. **Same-class blast-radius repair**: `test_phase_50_2_multicurrency.py` (2 red tests, same uninjected-seam class, one matched by the immutable command's 'persist') — repaired identically. The derived population [CORRECTED post-verdict per the cycle-2 Q/A, which re-derived it with a forced-uninjected simulation]: **9 kill-switch-coupled failures across 5 files** (61.2: 2, 50_2: 2, 64_3: 3, trade_idempotency: 1, adjust_cash_and_mtm: 1 — of which the last is MIS-classified: its failure is a stale test-stub signature TypeError, not the kill-switch class). My earlier '15 failures' counted every red in the 14-file run without classifying — the exact unclassified-set mistake the memory warns about. 4 same-class failures remain in 2 files; **13 FILES** (3 red + 10 currently-green uninjected) queued as **step 36.28**, whose scope now explicitly includes the stub-signature repair (the suite's greenness must not be coupled to the operator's live pause state; production fallback-to-singleton stays correct).
3. **A test-isolation defect in MY OWN 83.0 test repaired** (`test_c5_no_alphavantage_import_chain` scanned global `sys.modules`, so any earlier `-k`-matched test importing alphavantage made it fail order-dependently; now snapshots/pops/restores and asserts the DELTA of the two target imports).
4. **Criterion-1 error-string binding VERIFIED already present** (the fixture at test line 56 carries the exact live string `Failed to parse final report.` — verified, not changed).
5. **live_check §A + §B captured** (read-only BQ; derivation rules stated; the researcher's lite-split predicate could not be reproduced against the real schema and is cited as theirs, with my own 142/170 rule stated).
6. **Promotion ask filed** (operator_ask #10) with the measured evidence and the AWS REL05-BP01 fallback-share risk stated.

## Verification (verbatim, re-derived after the final edit)

Immutable command (previously 4 failed for out-of-step reasons — now green after the seam repairs):

```
$ python -m pytest backend/tests -k 'synthesis or persist or downgrade or meta_scorer or 61_2' -q
71 passed, 2829 deselected, 1 warning in 18.99s
```

`test -f handoff/current/live_check_61.2.md` → exists. Scope-bound signal: `pytest backend/tests/test_phase_61_2_decision_integrity.py -q` → **33 passed**. Adjacent: `test_phase_50_2_multicurrency.py` → **10 passed**; 83.0 suite unaffected (43 passed combined run).

## Criterion status honestly stated (D2)

- C2 (timeout): **MET, live**. C3 (company_name): **MET, live-proven** (§A). C5 (position stores verdict): **test-covered, mutation-proven**, dark. C4 (PIT + WARN + root-cause): built-dark; the root-cause clause is documented below. C1 and the live_check's zero-new-fabricated-rows + non-constant-conviction signals: **BLOCKED on the operator's promotion** (Sections C/D) — the fix exists, the regression test simulates the timeout with the live error string, but the immutable live_check demands post-fix LIVE rows that cannot exist while the flag is dark.
- **Expected verdict this cycle: CONDITIONAL #2** (prior count 1). A third would auto-FAIL — which is the argument for the operator deciding on THIS evidence packet rather than deferring.

## Criterion-4 root-cause documentation (the "diagnosed and documented" clause)

The 06-03..06-10 meta-scorer LLM unavailability: the direct-API Anthropic rail was credit-dead (dead since 2026-05-17 per the phase-72 money reconciliation) AND `meta_scorer.py` then constructed `ClaudeClient` directly with `anthropic_api_key`, bypassing `make_client()`/`paper_use_claude_code_route` — so the scorer could not reach ANY healthy rail and fell back to the legacy constant conviction=10 (the saturated pathology criterion 4 names). The bypass was fixed by phase-78.1 (`meta_scorer.py:226-229` comment + `:242 make_client(...)`); the credit decision remains the operator's (ask #2). The dark PIT normalization completes the criterion once promoted.

## Disclosures

- The immutable command's breadth (`-k 'synthesis or persist or ...'`) collects 71 tests today and grows as later phases add matching names; per `feedback_immutable_criteria_must_be_green_able` it was run VERBATIM and recorded, never amended.
- Registered test-debt carried forward, not quietly fixed (D4): three source-grep-style 61.2 tests (`test_degraded_marker_never_enters_analyses`, `test_streak_warn_wired_at_two`, `test_ctx_gate_references_both_flags`) and the untested `make_client → claude_code_timeout_s` threading.
- 72.0.1's dead premise → its disposition is queued as its own action (not silently closed here).
- The phase-76 audit_note's manual-path claim is half-wrong (measured: all 142 rows carry UPPER 'HOLD', which only `autonomous_loop.py:1942` emits; the manual `save_report` gap is real but LATENT — zero rows in 40d).

## Files changed

`backend/tests/test_phase_61_2_decision_integrity.py` (seam injection + stub), `backend/tests/test_phase_50_2_multicurrency.py` (same-class repair), `backend/tests/test_phase_83_0_news_corpus_persistence.py` (isolation repair). Zero production-code changes. Masterplan: +1 pending step (36.28). Handoff: contract, research brief, live_check, this file, operator_ask update.
