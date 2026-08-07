# Experiment results — Step 72.0.2: standard-tier fail-forward on rail-dead (FLAG-GATED DARK)

Date: 2026-08-07 (autonomous drain, cycle 177). Contract: `contract_72.0.2.md`.

## What was built

1. **Two settings fields** (`backend/config/settings.py`, beside the rail knobs): `paper_rail_failforward_enabled` (default **False** — DARK; description states both states, the metered-cost consequence, the operator-promotion requirement, the away-window and 4000.x interlocks, and the strict-reader guarantee) and `paper_failforward_model` (default `GEMINI_WORKHORSE` imported from `model_tiers` — never a literal; 2.5-family EOL tripwire preserved).
2. **Seam A** (`backend/agents/llm_client.py`, inside the CC-rail branch of `make_client`, before the `ClaudeCodeClient` return): flag ON + rail dead (`rail_skipped` OR `breaker_tripped` read from the public `rail_guard_status()`) + substitute is `gemini-*` + ADC genai client available → returns a REAL Vertex `GeminiClient` over an in-seam bundle built by new helper `_build_vertex_bundle` (decode pinned `temperature=0.0, top_k=1` — determinism is the finance-relevant substitution axis, arXiv:2511.07585). EVERY miss falls through to today's rail path (fail-open). Covers the orchestrator standard-tier clients, all six 78.1 C-block services, `quant_optimizer`, and `autonomous_loop`'s `make_client` callers.
3. **Seam B** (`backend/services/autonomous_loop.py`): `_select_lite_analyzer` gains optional `settings` (both call sites pass it; the legacy 1-arg signature routes exactly as before); flag ON + `claude-*` standard model + rail dead → new `_run_failforward_analysis`, which serves the lite analysis via `_run_gemini_analysis(..., model_override=paper_failforward_model)` under the **two-stage deterministic quality floor** `_failforward_floor_ok` (structural gate + degenerate-signature rejection incl. the `confidence==0` fabrication tell and `_parse_failed`), stamping provenance `_failforward/_failforward_provider/_failforward_reason`. Floor-fail → `_degraded: True` (the honest 61.2 path — `_fold_degraded_for_trading` drops it from decide_trades) — **never fabricates**. New `_rail_dead_reason()` is a strict fail-open READER of `rail_guard_status()`.
4. **`_run_gemini_analysis`** gains `model_override=None`; model resolution + the Gemini-only hard-raise extracted to pure `_resolve_lite_gemini_model` (runs before any I/O; raise message preserved verbatim); `_model_for_block` now stamps the model that actually serves.
5. **`backend/tests/test_phase_72_0_2_rail_failforward.py`** (new, 32 tests, $0/network-free): rail state via the PUBLIC `rail_guard_disable/reset` seam (66.1 precedent); genai stubbed at `_genai_client.get_genai_client`; flag-OFF byte-identity on client TYPE **and** the `timeout_s` kwarg (78.16 kwargs-drift guard); breaker-isolation deep-equal; None-trap; misconfig fail-open both seams; floor truth table (13 degenerate shapes); provenance; the 61.2 2×2 diagonal cells (floor-pass survives `_fold_degraded_for_trading`, floor-fail is dropped).

## Verification (cycle-1 capture, 32 tests — SUPERSEDED; the current capture is regenerated in the cycle-2 Follow-up below)

```
$ bash -c 'python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" backend/agents/llm_client.py'
immutable command exit: 0
$ .venv/bin/python -m pytest backend/tests/test_phase_72_0_2_rail_failforward.py -q
32 passed, 1 warning in 1.92s
$ .venv/bin/python -m pytest backend/tests/test_phase_72_0_2_rail_failforward.py backend/tests/test_phase_61_2_decision_integrity.py backend/tests/test_phase_75_5_2_model_pins.py backend/tests/test_phase_66_1_rail_guard.py backend/tests/test_phase_78_1_c_block_rail.py backend/tests/test_phase_78_16_prompt_caching_intent.py -q
428 passed, 1 warning in 3.63s
```

Lint gate (git-derived scope, tracked ∪ untracked, 6 files, non-empty, run BEFORE the Q/A): **"All checks passed!"**

## Mutation matrix — cycle-1 run, 11/11 KILLED (EXTENDED to 14/14 by the cycle-2 M10-M12 transport mutants, below; anchors count==1; real-file mutations with try/finally restores, every restore md5-verified; runner `scratchpad/mutation_matrix_72_0_2.py`)

| id | mutation | killed by |
|---|---|---|
| M1a/M1b | fail-forward fires with the flag OFF (each seam) | the two flag-OFF identity tests |
| M2a/M2b | fires on a HEALTHY rail (each seam) | the two rail-healthy identity tests |
| M3 | the seam mutates the rail guard (`_rail_guard_blocked` call) | test_seam_reads_but_never_mutates_rail_state |
| M4 | floor accepts `confidence==0` | floor truth table + the honest-degraded test |
| M5 | the None-trap (`GeminiClient(model=None)`) | test_flag_on_rail_dead_returns_vertex_gemini |
| M6 | STUB mutation: floor-fail fixture feeds a GOOD payload | test_failforward_floor_fail_is_honest_degraded (the test discriminates via its input, not vacuously) |
| M7 | provenance stamp dropped | test_failforward_stamps_provenance_on_floor_pass |
| M8 | override plumb dropped | the two resolve tests |
| M9 | misroute to GEMINI_DEEP_THINK | the re-pointed 75.5.2 guard (2 failed) — proves the moved guard kept its discriminating power |

## C3 cost derivation (measured + cited, per the criterion; full detail in ask #13)

- **Measured volume** (`pyfinagent_data.llm_call_log`, 14-day window ending 2026-08-07, cycles with ≥1 `claude-*` call): **12 cycles; avg 82.5 / max 173 claude-tier calls per cycle; avg ~196K / max ~522K output tokens per cycle** (roles: `cc_rail` 978, `lite_trader` 6, `lite_risk_judge` 6). Caveat stated: the rail's `input_tok` is known-undercounted (avg 240/cycle is implausible — phase-78 memory), so input is bounded by prompt-size estimate, not the log.
- **Rate** (Vertex `gemini-2.5-flash`, https://cloud.google.com/vertex-ai/generative-ai/pricing, accessed 2026-08-07): **$0.30/M input, $2.50/M output**.
- **Expected per-cycle cost with the flag ON during a rail-dead cycle:** typical ≈ **$0.3** (82.5 calls × ~2K in/~1K out ≈ $0.05 + $0.21); worst-case bound ≈ **$1.4** (173 calls, plus the rail-volume output proxy 522K × $2.50/M = $1.30 — an over-bound because Gemini outputs are capped far below the rail's thinking-heavy volume). A fully rail-dead month at one cycle/day ≈ **$9–40**.

## Deferral (recorded per contract): why the status does NOT flip this cycle

The step's `live_check` requires a real cycle with the flag ON and the rail dead — an INDUCED capture that makes **metered Vertex calls**, which the standing `$0 metered` away-ops constraint reserves for the operator. The flip is therefore HELD (the live_check gate holds auto-commit anyway); **ask #13** carries the induced-capture recipe (single cycle, `rail_guard_disable()` + flag ON, capture the routing line + BQ provider row + a non-degraded `_failforward=true` score) and the cost line above. All code/tests/verdict evidence is complete this cycle — the 61.2 pattern.

## Disclosures

- **Two prior regression suites were updated because this change touched their premises**, both preserving intent: (a) `test_phase_61_2_decision_integrity.py:113` — its `_select_lite_analyzer` stub took 1 positional arg; updated to the new `(model, settings=None)` signature. (b) `test_phase_75_5_2_model_pins.py` misroute guard — the workhorse-default property it guarded moved into `_resolve_lite_gemini_model`; the test now checks the helper (and that `_model_for_block` derives from the resolved name), and M9 proves by execution it still turns red on a deep-think misroute.
- **Two pre-existing red-test families were found mid-cycle and handled OUTSIDE this step's scope** (freeze-the-tree respected — both fixed/queued and committed under their own attributions BEFORE this step's Q/A): (a) the phase-23.2.14 lock roster (18→20; the two 83.0.x news-module counter locks; commit b14d741d, `fix(83.0)`); (b) three `test_phase_57_1_reject_binding.py` tests red because **the operator has promoted `paper_risk_judge_reject_binding` to TRUE in backend/.env** (measured: `get_settings()` resolves True; source default False) — queued as **36.30** (env-flag test-leakage class, commit d5583a72) and the stale "gate dark" memory corrected.
- The Seam-A fail-forward serves the ORCHESTRATOR/C-block clients a Vertex GeminiClient whose callers' own parse/validation applies; the payload-level quality floor applies where the fabrication defect lives (the lite path, Seam B). C1's "real (non-degraded) scores" is made auditable by the provenance stamps + the floor.
- `_run_claude_analysis` (the direct-SDK lite path) is intentionally untouched: rail-dead routing now diverts BEFORE it via the selector; its internal behavior remains the 61.2/70.4 honest-degraded chain.
- No change to `claude_code_client.py`, the probe block (`autonomous_loop.py:393-427`), or any alerting — verified by the diff scope (6 files: the three production files above + three test files).

## Files changed

`backend/config/settings.py`, `backend/agents/llm_client.py`, `backend/services/autonomous_loop.py`, `backend/tests/test_phase_72_0_2_rail_failforward.py` (new), `backend/tests/test_phase_61_2_decision_integrity.py` (stub signature), `backend/tests/test_phase_75_5_2_model_pins.py` (guard re-pointed). Handoff: contract, this file. Out-of-scope commits (own attributions): b14d741d (lock roster, 83.0), d5583a72 (36.30 queue). No masterplan status changes this cycle.


## Follow-up — cycle 2 (2026-08-07, after Q/A CONDITIONAL wf_c269017f-611; streak 1)

All three findings closed with executed proofs. (The first spawn wf_5f25b528-bbb died without emitting StructuredOutput — NO VERDICT; the recovery re-spawn is verdict #1, transcribed in `evaluator_critique_72.0.2.md`.)

**F1 (Seam B not Vertex; undisclosed GEMINI_API_KEY precondition)** — Seam B no longer routes through `make_client`'s priority order at all: new `_build_failforward_client(ff_model)` constructs the SAME in-seam ADC Vertex bundle Seam A uses (via `_build_vertex_bundle`), and `_run_gemini_analysis` gained `client_override` (default None = byte-identical legacy) so `_run_failforward_analysis` injects that client. ADC-unavailable → fail-open to the legacy lite path (logged), never `GeminiClient(model=None)`. The transport is now caller- AND key-independent on both seams; `GEMINI_API_KEY` is not load-bearing.

**F2 (Seam-B transport vacuity — no guard could fail)** — four new tests: `test_failforward_client_is_vertex_adc` (bundle non-None + `bundle.client is` the ADC stub — the AI-Studio leg builds its OWN client, so this discriminates the branch), `test_failforward_client_none_without_adc`, `test_failforward_no_adc_falls_to_legacy`, and `test_seam_b_full_path_uses_injected_client` — the REAL `_run_gemini_analysis` runs end-to-end (fake `_fetch_yf_market_data`, recording client returning valid JSON; 2 generate calls asserted). The Q/A's named mutations EXECUTED and KILLED: M10 (None-trap regression → 3 red), M11 (foreign AI-Studio-style client in the bundle → 3 red), M12 (`client_override` dropped, reverting to `make_client` → exactly the full-path test red). Matrix total: **14/14 KILLED**, restores md5-verified.

**F3 (billing-surface misattribution in ask #13)** — after F1 both seams bill **Vertex on the GCP project via ADC**; ask #13 rewritten to name the corrected surface explicitly, state that `GEMINI_API_KEY` is not load-bearing, and keep the unchanged magnitude (~$0.3/≤$1.4 — the Q/A confirmed the rates match across surfaces, so only attribution moved).

### Verification (verbatim, regenerated cycle 2 after the final edit)

```
$ bash -c 'python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" backend/agents/llm_client.py'
immutable exit: 0
$ .venv/bin/python -m pytest backend/tests/test_phase_72_0_2_rail_failforward.py -q
37 passed, 1 warning in 1.94s
$ (scoped regression: 72.0.2 + 61.2 + 75.5.2 + 66.1 + 78.1 + 78.16)
433 passed, 1 warning in 3.84s
```

Lint gate re-run over the derived scope (6 files): **"All checks passed!"** The cycle-1 capture block above is superseded by this one (32→37 tests, 428→433 scoped; append-only, body unedited).
