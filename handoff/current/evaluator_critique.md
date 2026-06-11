# Evaluator Critique — Step 57.1 (Binding RiskJudge gate + concentration-aware prompt context)

**Verdict: PASS** (`ok: true`, zero violated_criteria). Single merged Q/A; FIRST spawn for 57.1.
**Date:** 2026-06-11. **Phase:** phase-57 FEATURE (config-gated default-OFF; NO live flip).

---

## 5-item harness-compliance audit (ran FIRST, all green)

1. **Researcher gate — PASS.** `handoff/current/research_brief.md` is the 57.1 brief: tier=complex, envelope `gate_passed: true`, 6 sources read in full (17 CFR 240.15c3-5 primary CFR + ESMA Feb-2026 supervisory briefing + GuardAgent arXiv:2406.09187 + Ahern 2006 + arXiv:2604.01483 + arXiv:2511.15123), 16 URLs, recency scan B5 present (ESMA Feb-2026 + 2 arXiv 2025-2026). Cites F-3 (advisory-only REJECT) + F-8 (phantom 10% cap). Contract references section names the researcher output.
2. **Contract pre-commit — PASS.** `contract.md` mtime `05:19:48` PRECEDES every code edit (settings.py 05:21:11, portfolio_manager.py 05:21:50, autonomous_loop.py 05:26:21, test 05:28:46). Programmatic verbatim compare: all **6 criteria match `.claude/masterplan.json` step 57.1 `verification.success_criteria` byte-for-byte** (whitespace-normalized equality, 6/6 True); the immutable verification command matches verbatim. Install commit verified: `git log --grep "PHASE-57: FEATURE"` hits `af4aa8d6` — subject records the operator's verbatim reply `'PHASE-57: FEATURE'`.
3. **Results artifact — PASS.** `experiment_results.md` present with the 4-file change table, verbatim verification-command output (`7 passed, 767 deselected`), and full-suite line (`756 passed, 12 skipped, 6 xfailed`).
4. **Log-last — PASS.** No 57.1 Cycle entry in `handoff/harness_log.md` (last entries are Cycle 47 phase-56.2); masterplan 57.1 `status: "pending"`. Correct order — log + flip happen AFTER this PASS.
5. **No verdict-shopping — PASS.** First Q/A spawn for 57.1; no prior critique to overturn. `retry_count: 0`.

---

## Deterministic checks (cannot hallucinate)

- **syntax** — `ast.parse` OK on all 4 files (settings.py, portfolio_manager.py, autonomous_loop.py, test_phase_57_1_reject_binding.py).
- **verification_command** (immutable) — `python -m pytest backend/tests -k 'reject_binding or risk_judge_binding' -q` → **7 passed, 767 deselected, exit 0**; `test -f handoff/current/live_check_57.1.md` → present (`livecheck-ok`). GREEN.
- **full_suite** (criterion-6 family + do-no-harm) — `python -m pytest backend/tests -q` → **756 passed, 12 skipped, 6 xfailed, exit 0** in 69.6s. Matches the contract claim EXACTLY; zero breakage from the default-OFF flag.
- **no_live_flip** — `get_settings.cache_clear(); get_settings().paper_risk_judge_reject_binding` → **False**. settings.py Field default `False`; .env untouched. No live flip, per criterion 2/6.
- **bq_event_study_reproduction** (live BQ, 2026-06-11):
  - Query 1 (`paper_trades WHERE action='BUY' AND risk_judge_decision='REJECT'`) → **exactly 3 rows**, ALL `reason=swap_buy`: HPE 2026-06-02T19:18:58, DELL 2026-06-03T19:05:19, 066570.KS 2026-06-09T18:12:39. Independently confirms the count AND the decisive topology claim (every real REJECT execution went via the swap path).
  - Query 2 (`paper_round_trips`) → HPE **−$0.81** (−0.33%, swap_for_higher_conviction), DELL **+$0.54** (+0.22%, swap_for_higher_conviction), 066570.KS **−$23.18** (−9.68%, **stop_loss_trigger**). **Net of the 3 = −$23.45**, matching the live_check verbatim. (A 4th unrelated DELL round-trip with entry 06-09 +$14.73 exists but is NOT one of the 3 REJECT BUYs — the REJECT DELL BUY was 06-03; the live_check's ticker+entry-day join correctly excludes it.)
- **other_buy_path_grep** (anti-rubber-stamp) — `grep action="BUY"` across `backend/**` (ex-tests): the only LIVE-path `TradeOrder(action="BUY")` emission sites are `portfolio_manager.py:372` (main BUY loop) and `portfolio_manager.py:589` (swap path) — **both downstream of the candidate-build chokepoint** where the gate sits. `backtest_trader.py:178` is the offline backtest engine (separate system, out of scope). `paper_trader.py:225` is a dedup-query filter arg, not an emission. `pm:105/118/128` are all `action="SELL"`. There is NO third live BUY path that bypasses the gate.
- **emoji/ascii** — no emojis/arrows in added backend diff lines; all non-ASCII in the test file is box-drawing in `#` comments only (NOT in any logger or code string). New logger strings in portfolio_manager.py:199-203 and autonomous_loop.py are ASCII-only.

---

## Code-review heuristics (5 dimensions evaluated; zero BLOCK, zero WARN)

- **Dim 1 Security** — no secret-in-diff (settings flag is a `bool`, no literal); no prompt-injection-path (the injected `portfolio_context` is system-generated from BQ positions, NOT user-supplied; and it is brace-escaped before `.format`); no command-injection; no dep-pin removal. CLEAN.
- **Dim 2 Trading-domain** — gate is REJECT-only (`_rj_decision == "REJECT"`, pm:196), so APPROVE_REDUCED/HEDGED sizing is untouched (no over-bind). kill_switch / stop-loss / max_positions / perf_metrics paths are NOT touched by this diff. No crypto re-enable. The gate ADDS a risk control (binding a previously-advisory REJECT), the regulation-aligned direction (SEC 15c3-5(c)(1) "by rejecting orders"). CLEAN.
- **Dim 3 Code quality** — the new `try/except` at autonomous_loop.py:783-786 around `_build_portfolio_sector_context` is a NON-fatal context-build guard that logs and degrades to `""` (NOT an execution-path silent-swallow; it protects byte-identity by failing open to the OFF behavior). `_build_portfolio_sector_context`'s `try/except (TypeError, ValueError): continue` is a per-row numeric-coercion guard, not a broad risk-guard swallow. Helpers are typed. No print(). CLEAN.
- **Dim 4 Anti-rubber-stamp** — financial-logic change (a BUY-path gate) ships WITH a behavioral test file (`test_phase_57_1_reject_binding.py`, 7 tests). No tautological assertions (the assertions check real order presence/absence, object identity, and rendered-string content). No over-mocking (decide_trades runs for real with fixtures; no `@patch` of the module under test). CLEAN.
- **Dim 5 LLM-evaluator anti-patterns** — first spawn, no rebuttal context, no prior CONDITIONAL to escalate. This critique carries file:line citations throughout. CLEAN.

---

## Per-criterion LLM judgment (against the 6 immutable criteria)

**C1 — Binding-gate regression fixture, both BUY paths — PASS.**
`test_reject_binding_main_path_off_emits_on_blocks` (main path): flag-OFF emits the REJECT BUY (line 102), flag-ON it is absent + `blocked_out` records it (lines 110-114). `test_reject_binding_swap_path_off_emits_on_blocks` (swap path — the away-week vulnerability): the scenario stacks 8 Technology holdings at `max_per_sector=2` so TECH_NEW1/2 get sector-COUNT-blocked into `sector_blocked` → the swap path; flag-OFF asserts `"TECH_NEW1" in swap_buys_off` AND `rejected_order.risk_judge_decision == "REJECT"` (lines 148-152) — **this proves the swap path actually ran and emitted a `swap_buy` for the REJECT candidate**, reproducing HPE/DELL/LG exactly; flag-ON asserts `"TECH_NEW1" not in on_tickers` and `"TECH_NEW2" in on_tickers` (the next-ranked survivor takes the freed slot; lines 164-170). I verified by reading `_compute_swap_candidates` (pm:439-585) that `sector_blocked` is populated ONLY from `buy_candidates` entries (pm:319, iterating pm:290) — so a candidate `continue`d out at the gate (pm:212, BEFORE `buy_candidates.append` at pm:214) can reach NEITHER the main loop NOR the swap path. The gate site is the genuine common ancestor.

**C2 — Default-OFF byte-identity (orders + prompts) + no live flip — PASS.**
`test_off_identity_orders_no_reject_set`: flag-ON == flag-OFF order lists on a REJECT-free set (line 184). `test_off_identity_prompts_are_verbatim_constants`: `_build_risk_judge_system(s_off) is al._LITE_RISK_JUDGE_SYSTEM` and `_build_risk_judge_template(s_off, "anything") is al._LITE_RISK_JUDGE_TEMPLATE` (lines 189-190) — **object-`is`-identity**, the strongest byte-identity proof, plus a rendered-equality check (lines 195-196). I confirmed the builders short-circuit on `not getattr(settings, "paper_risk_judge_reject_binding", False)` returning the bare constant (autonomous_loop.py:1591, :1606), and the per-cycle `_rj_portfolio_ctx` compute is gated `if getattr(settings, ...)` (autonomous_loop.py:782) so it is skipped when OFF. New kwargs default to `None`/`""` (`portfolio_context: str | None = None` on `_run_single_analysis`; `= ""` on both analyzers) so existing callers are unaffected. No live flip: settings loader returns False.

**C3 — Prompt-context correctness with the flag ON (F-8) — PASS.**
`test_prompt_content_flag_on_real_cap_and_sector_line`: system prompt contains `"exceed 30% of portfolio NAV in one sector"` and NOT `"10% of portfolio in one sector"` (lines 203-204); `_build_portfolio_sector_context(fake_positions)` yields `"Technology 100.0%"` (line 211); the rendered template contains the injected `"Current portfolio context: invested-book sector weights: Technology 100.0%"` line (line 217). The cap is read from `paper_max_per_sector_nav_pct` (autonomous_loop.py:1593), the real configured value — fixing the phantom-cap defect.

**C4 — Per-cycle single-compute (concurrency-correct) — PASS.**
`test_analyzers_receive_precomputed_context_not_positions_fetch`: all three analyzers (`_run_single_analysis`, `_run_claude_analysis`, `_run_gemini_analysis`) carry a `portfolio_context` parameter AND `"get_positions" not in inspect.getsource(fn)` (lines 236-240); the single compute site `_build_portfolio_sector_context(positions)` is asserted to live in `run_daily_cycle` source (line 243). I confirmed in the diff that the compute happens ONCE at autonomous_loop.py:782 (after the positions read at :774, BEFORE the concurrent fan-out at the `_run_and_persist_one` dispatch) and is threaded as `portfolio_context=_rj_portfolio_ctx` (autonomous_loop.py:862-864) — neither analyzer calls `get_positions`. Mutation-resistant: a per-ticker `get_positions()` would trip the source-scan.

**C5 — Event-study artifact, honestly framed — PASS.**
`live_check_57.1.md` contains: the BQ query + 3-row table (HPE/DELL/066570.KS), the realized P&L (−$0.81 / +$0.54 / −$23.18; net **−$23.45**) — all **independently reproduced against live BQ above** — AND an explicit selection-bias caveat (n=3 descriptive of these specific decisions, NOT the gate's EV; conditioned on REJECT∧swap-executed∧closed-in-window; cites Ahern 2006 + arXiv:2511.15123) with **no annualized/Sharpe extrapolation**. Matches the research brief's mandatory-honesty framing.

**C6 — Verification command green + flag default-OFF unflipped — PASS.**
Immutable command exits 0 (7 passed; live_check present). Flag ships `Field(False, ...)` and is verified `False` at runtime; no flip inside phase-57.

---

## Anti-rubber-stamp: actively sought a hole

- **Does the swap-path test actually exercise `_compute_swap_candidates`?** YES — the flag-OFF assertion requires a `reason == "swap_buy"` order for TECH_NEW1 (lines 147-148). A `swap_buy` reason is emitted ONLY at pm:591 inside `_compute_swap_candidates`. If the swap path never ran, that assertion would fail. The test is not a no-op.
- **Is there another BUY-emission path that bypasses the candidate-build loop?** NO — grep confirms only pm:372 (main) and pm:589 (swap) in the live path, both downstream of the gate; backtest_trader.py and paper_trader.py:225 are not live-cycle TradeOrder BUY emissions.
- **Format-safety (would a `{`/`}` in a sector name break `.format`)?** Handled — `_build_risk_judge_template` escapes the injected literal via `context_line.replace("{", "{{").replace("}", "}}")` (autonomous_loop.py:1610) BEFORE the downstream `.format()`. The escape is applied AFTER the f-string substitution, so any braces in the live data are doubled correctly. The replacement TARGET string exists verbatim in `_LITE_RISK_JUDGE_TEMPLATE` (line 1567), so the `.replace` is not a silent no-op. Correct.
- **Could the gate accidentally block APPROVE_REDUCED/HEDGED?** NO — condition is strict equality `_rj_decision == "REJECT"` (pm:196). Verified.

---

## Notes (non-blocking)

- Blocked-BUY observability is log + `summary["risk_judge_blocked"]` only (no BQ table) — explicitly scoped as a possible DoD-7 follow-on in experiment_results; acceptable for 57.1.
- Full-pipeline (non-lite) RiskJudge path is out of 57.1 scope (the lite path is what trades autonomously today) — honestly disclosed.
- Test file uses box-drawing chars in section-header comments; NOTE-level only (not a logger string), no verdict impact.

**checks_run:** syntax, verification_command, full_suite, no_live_flip, bq_event_study_reproduction, other_buy_path_grep, code_review_heuristics, contract_verbatim_compare, mutation_resistance, evaluator_critique
