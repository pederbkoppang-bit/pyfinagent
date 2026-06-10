# Evaluator Critique — Step 56.2 "Ops fixes"

**Q/A agent (merged qa-evaluator + harness-verifier). Single fresh spawn (first for 56.2).**
**Date:** 2026-06-10. **Verdict: PASS** (`ok: true`).

---

## 0. Harness-compliance audit (5/5 PASS — runs first, unchanged)

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher gate | PASS | `handoff/current/research_brief.md` is the 56.2 brief: tier=complex, envelope `gate_passed: true`, 7 external sources read in full + 16 URLs + recency scan (Confidence-Gate arXiv:2603.09947 2026, OneUptime heartbeat, Silent-Failure, Slack Bolt ack, pytest skipif, GX WAP, index.dev). 16 internal files inspected. |
| 2 | Contract pre-commit | PASS | `contract.md` mtime 20:16:30 precedes ALL code edits (earliest `api_call_log.py` 20:23:14). Its 4 criteria match `.claude/masterplan.json` step 56.2 VERBATIM (diffed). |
| 3 | Results artifact present | PASS | `experiment_results.md` lists 11 files + verbatim verification-command output (`749 passed, 12 skipped, 6 xfailed`). |
| 4 | Log-last | PASS | No 56.2 Cycle entry in `harness_log.md`; masterplan 56.2 `status="pending"`. |
| 5 | No verdict-shopping | PASS | First spawn for 56.2. No prior 56.2 verdict to overturn. |

---

## 1. Deterministic checks (cannot hallucinate)

| Check | Result | Evidence |
|-------|--------|----------|
| Immutable verification command (full backend suite) | **exit 0** | `python -m pytest backend/tests -q` → `749 passed, 12 skipped, 6 xfailed, 1 warning in 71.11s`. Re-run independently; matches experiment_results counts exactly. |
| `test -f handoff/current/live_check_56.2.md` | PASS | livecheck-ok (file present, fully populated §A-F). |
| New test file alone | **18/18 pass** | `test_phase_56_2_ops_fixes.py` → `18 passed in 1.97s`. |
| Syntax parse (6 changed prod files) | PASS | `ast.parse` OK on all. |
| Import-cycle smoke | PASS | `python -c "import backend.services.autonomous_loop"` OK — lazy `from backend.services.alerting import raise_cron_alert` (inside functions) introduces no module-load cycle. |
| Quarantine tests PASS (not skip) | PASS | lock-count `5 passed`; shortlist-doc `7 passed`; agent-map `7 passed`. Rainbow-canary + watchdog-pollution pass in full-suite order (0 failures in the 749). |

## 1b. Do-no-harm (decision math untouched — verified by empty diff)

`git diff --stat` is **EMPTY** for every decision-math file:
- `backend/services/meta_scorer.py` — 0 diff. `_fallback_all` (`:249-256`) VALUE (`_fallback_conviction`=`round(composite)` clamped, `:138`) and ORDERING (`sorted(...reverse=True)` by conviction, `:256`) byte-identical.
- `backend/services/portfolio_manager.py` — 0 diff.
- `backend/services/kill_switch.py` — 0 diff (F-9 is proposal-only).
- `backend/services/paper_trader.py` — 0 diff (kill-switch region included).
- screener / optimizer (glob) — 0 diff.

The conviction-fallback detection (`_all_conviction_fallback`, `autonomous_loop.py`) is **read-only** over `conviction_reason` strings; greps the exact producer string `"fallback (LLM unavailable)"` emitted by `meta_scorer.py:254`. Sets only `summary["meta_scorer_degraded"]`; never mutates a score or order.

## 1c. Crash-isolation (an alerting bug can never break a trading cycle)

Each new observability hook is wrapped in its OWN `try/except Exception`:
- **Rail probe (F-4)**: `except Exception as _probe_exc: logger.warning(... "non-fatal" ...)` — `autonomous_loop.py` cycle-start block.
- **Degraded guard (F-5)**: `except Exception as _guard_exc: logger.warning(... "non-fatal" ...)`.
- **Metering (F-6)**: `_log_claude_code_call` body wrapped in `try/except` (`logger.debug ... non-fatal`).
- **Probe internals**: `claude_code_health_probe` catches `TimeoutExpired` / `FileNotFoundError` / `Exception` and returns `(False, detail)` — NEVER raises. (These broad-excepts are in an OBSERVABILITY probe, NOT a risk-guard path → not the `broad-except-silences-risk-guard` BLOCK case; they correctly fail-soft.)

## 1d. Criterion-2 ticket-processor branch fidelity

`ticket_queue_processor.py`: the CLI branch (`if getattr(settings, "paper_use_claude_code_route", False)`) preserves the system prompt (`system=system`) and a 60s timeout (`timeout_s=60`). The direct-SDK branch is byte-identical to pre-fix when the flag is off (the original `anthropic.Anthropic(api_key=...)` block is moved below the new branch verbatim; the only other change is removal of a `[wrench-emoji]` from a comment — improvement). On the flag the keyless direct call (root cause of "Missing API key for provider anthropic") is never reached.

## 1e. generation_config whitelist (no `_role`/`_ticker` leak to Gemini)

`llm_client.py:941-957` assembles `gc_kwargs` from an explicit whitelist (`temperature, top_k, top_p, max_output_tokens` + named `response_mime_type/response_schema/thinking_config/tools`) then `GenerateContentConfig(**gc_kwargs)`. `_role`/`_ticker` are read only via `.get()` at `:1070/:1079` for `log_llm_call` and are NEVER added to `gc_kwargs` → they do not reach the Gemini API call. Confirmed by reading the assembly block.

## 1f. F-9 is proposal-only (no code, no threshold change)

`kill_switch.py` + `paper_trader.py` kill-switch region: 0 diff. The F-9 SOD-anchor proposal text is in `live_check §D`, explicitly framed as an OPERATOR DECISION ("Reply 'F-9: APPROVED' ... or leave it parked"), with the 4%/10% limits stated UNCHANGED. 55.1's audited verdict was **CORRECTLY-DID-NOT-TRIP** (postmortem line 161: worst-day -2.82% < 4%, trailing < 3.4% vs 10%) → criterion 4's IFF condition (`SHOULD-HAVE-TRIPPED`) is FALSE → no kill-switch unit-test fix required. Correctly handled.

## 1g. Quarantine honesty (the watermelon check)

Root-cause classification, NOT blanket skip:
- **2 STALE assertions UPDATED**: agent-map `claude-opus-4-8` (`test_agent_map_live_model.py:58`; the `:90` `4-7` is a `_StubSettings` *input* for a locked-node test, correctly untouched); `EXPECTED_LOCK_COUNT = 15` with re-audit note citing `alerting.py:64` (AlertDeduper). Both tests PASS (`5 passed`/`7 passed`) — skipping would have hidden real drift.
- **7 moved-doc REPOINTED** to `handoff/archive/phase-23.2.16/...` (the doc still exists; archive-handoff hook moved it). `7 passed`.
- **5 live-probe `requires_live` skipifs** (NEW `pytest.ini` registers the marker) with per-test reasons naming the EXACT dependency (MAX(ts) SLA window; `analysis_results total_cost_usd>0.05`; 5-20 `drawdown_breach` rows; live HTTP :8000). Note: 5 live-probes (incl. a 5th live-HTTP `ticker_meta_latency`), one more than the contract's "2 live-BQ" sketch — over-disclosed, not under.
- **1-2 pollution ROOT-CAUSE FIXED**: `reset_buffer_for_test()` now re-arms `_last_flush_ts` inside the lock (`api_call_log.py`) — the time-based flush was draining injected rows mid-test in full-suite order. Real fix, not a skip.

## 1h. Security + ASCII

- **secret-in-diff**: clean (only `sk-test-not-real`/`sk-unused` test fixtures — negation-listed).
- **command-injection**: `subprocess.run([resolved_binary, "auth", "status"], ...)` — LIST arg, `shell=False`, literal `binary="claude"` default — negation-listed (safe).
- **ASCII logger strings**: all NEW `logger.*` strings in the 6 prod files are pure ASCII (verified by `^\+`-line non-ASCII grep → none).

---

## 2. Mutation-resistance (anti-rubber-stamp — the 3 planted scenarios)

| Planted violation | Caught by | How |
|---|---|---|
| (i) Probe NOT scrubbing the API key | `test_rail_probe_scrubs_api_key_from_env` (`:57-68`) | Sets `ANTHROPIC_API_KEY`, spies `subprocess.run` env, asserts `"ANTHROPIC_API_KEY" not in captured["env"]`. Removing the scrub dict-comp → FAIL. |
| (ii) Guard over-alerting at 2 zeros | `test_degraded_guard_quiet_at_two_zeros_of_six` (`:87-90`) | Asserts `fire is False, n_deg == 2`. Changing `>= 3` → `>= 2` → FAIL. |
| (iii) Ticket processor uses SDK despite the flag | `test_ticket_agent_uses_cli_rail_when_route_flag_set` (`:184-198`) | Patches both rails; asserts `cli_spy.assert_called_once()` AND `sdk_spy.assert_not_called()` (paired with `out == "Approved..."`). Ignoring the flag → trips `assert_not_called`. |

Bonus real bug caught WHILE writing the tests: the **falsy-zero trap** — `test_degraded_guard_counts_confidence_zero_uppercase_tell` (`:93-99`) asserts `confidence=0` still counts as degraded; the predicate explicitly checks `_conf_raw is not None` before `float()==0.0` rather than `or`-defaulting (which would mask a real 0). Tests are NOT tautological (no `assert x==x`/`is not None`-only) and NOT over-mocked (`TicketQueueProcessor.__new__` + real `_spawn_real_agent` under test; only rails patched).

---

## 3. Code-review heuristics (5 dimensions) — 0 BLOCK / 0 WARN

- **D1 Security**: secret-in-diff clean; command-injection negation-listed; no prompt-injection/insecure-output/training-code/unbounded-loop introduced.
- **D2 Trading-domain**: kill-switch reachable (0 diff); stop-loss/backfill/max-position/crypto untouched (0 diff); no perf-metrics bypass; no LLM-output-to-execution path added.
- **D3 Code quality**: broad-excepts are observability fail-soft (return value, not silent risk-swallow); no print() in prod; new helpers carry type hints.
- **D4 Anti-rubber-stamp**: financial-logic files have 0 diff → `financial-logic-without-behavioral-test` N/A; behavioral tests present for every new path.
- **D5 LLM-evaluator**: first spawn, full file:line chain-of-thought, no criteria erosion.

**NOTE (does not degrade verdict)**: `test_phase_56_2_ops_fixes.py:22,71,107,141,177` use box-drawing chars (`--` dividers) in SECTION-DIVIDER COMMENTS (not logger calls, not runtime strings). This matches an established repo convention — 7 other committed test files (`test_phase_50_2_multicurrency.py`, `test_phase_32_*`, etc.) and multiple committed `backend/services/*.py` use the same dividers. Python-3 UTF-8 source; no cp1252 crash surface (the security.md ASCII rule targets `logger.*` calls, which are clean here). Cosmetic only.

---

## 4. LLM judgment against the 4 immutable criteria

**C1 — every P0/P1 FIXED+test or ESCALATED operator-gated, finding-ID map in live_check** → **MET.** `live_check §A` dispositions every CRITICAL+HIGH+MED-HIGH finding in the 55.3 §1 table: F-1/F-2 (fixed in 56.1), F-4/F-5/F-6/F-7/F-14/watchdog/criterion-2 (FIXED with regression tests in `test_phase_56_2_ops_fixes.py`), F-3/F-8/F-18 (ESCALATED to phase-57 — behavior-changing, already in the 55.3 §2.6 spec), F-9 (operator-proposal). The MED/LOW tier (F-10/11/13/15/16/17 "56.x", F-19 informational) is legitimately out of P0/P1 scope. Nothing changed without a finding ID.

**C2 — approve path exercised e2e OR residual escalated with one-line operator action** → **MET (OR-branch, honestly).** The fix (route `_spawn_real_agent` through the CLI rail when the flag is set) is applied AND unit-tested (both flag states). A true e2e transcript needs the operator (bot-message filtering + slack-bot restart); `live_check §B` escalates with the exact one-line action: "restart the slack bot and type `Approve` in #ford-approvals — expect an agent reply via the claude-code rail instead of the missing-key error." This is the criterion's explicit OR-branch, disclosed as a limitation (experiment_results §Honest limitations), not overclaimed.

**C3 — degraded-scoring guard exists + alerts Slack + unit-tested; watchdog bounded fix per 55.2** → **MET.** `_degraded_scoring_check` (cycle-level, post-gather) fires on ALL-degraded or ≥3 zeros → P1 `raise_cron_alert(source="autonomous_loop", error_type="degraded_scoring")`; unit-tested across all-zero/3-of-6/2-of-6/confidence-0-UPPERCASE/empty boundaries. Watchdog probe timeout 10s→30s (`scheduler.py`) per the 55.2 event-loop-starvation root cause (backend never down; mirrors digest 30s).

**C4 — kill-switch unit-test IFF SHOULD-HAVE-TRIPPED; threshold change = OPERATOR DECISION; 16 failures quarantined (skip-markers + reasons); pytest green** → **MET.** 55.1 ruled CORRECTLY-DID-NOT-TRIP → IFF false → no unit-test required (correct). F-9 SOD re-anchor presented as an OPERATOR DECISION with thresholds UNCHANGED, never auto-applied (0 code diff). Quarantine done with skip-markers + per-dependency reason strings (and honest UPDATE/REPOINT/ROOT-CAUSE-FIX for the non-env failures — superior to blanket skip). Backend pytest exit 0.

---

## 5. Verdict

**PASS** — `ok: true`. All 5 harness-compliance items pass; the immutable verification command exits 0 (`749 passed`); all 4 immutable criteria are met (C2 honestly via the criterion's OR-branch, escalation disclosed); do-no-harm is proven by empty diffs on every decision-math file; the conviction-fallback VALUE+ordering are byte-identical (unit-tested); mutation-resistance is real on all 3 planted scenarios plus the falsy-zero trap; the quarantine is root-cause-classified (no watermelon); F-9 is a code-free operator proposal with thresholds unchanged. Zero code-review heuristics fire at BLOCK or WARN; the only finding is a cosmetic NOTE (box-drawing chars in test comments, matching existing repo convention).

`violated_criteria: []`. `certified_fallback: false` (retry_count=0 < max_retries=3).
