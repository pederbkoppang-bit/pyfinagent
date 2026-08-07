# Contract — Step 83.0.3: PBO false-pass on the MCP surface (PROOF step)

- **Step id:** 83.0.3 (P0, phase-83; depends_on: none)
- **Tier (named field):** T3 — executor Main (Opus 5, effort max); Q/A via qa-verdict Workflow (opus/max).
- **Date:** 2026-08-07, autonomous drain, cycle 169

## Research-gate summary

`handoff/current/research_brief_83.0.3.md` — gate_passed: **true** (7 external sources read in full / 27 URLs / recency scan / 9 internal files; envelope returned on the rail AND written to disk). Decisive findings:

1. **This is a PROOF step, not a fix step.** Commit `e5bb9f25` (82.27) already routes `pbo_check` through `compute_pbo_checked` with an explicit refusal payload. The step name's anchors (`risk_server.py:142-143`) are STALE — they now point at comment prose about the old defect; the live checked call is at `:149-150`. Criteria remain satisfiable as written (they demand tests + records, not a code change).
2. **MCP tools are closures** — the in-repo idiom is `asyncio.run(create_risk_server().get_tool("pbo_check")).fn` (`test_phase_82_27_pbo_sweep_producer.py:210`). No server/transport needed.
3. **Criterion-3 mutation prototyped and proven**: the exact 2-line target occurs once; the ast-validated mutant reproduces the original false PASS (`{ok: true, pbo: 0.0, vetoed: false}` on T=10/S=16) — demonstrably non-equivalent.
4. **Criterion-4 fixture boundary is tight**: eps=0.0005 → corr_mean 0.99742 (columns_diverse False); eps=0.001 → 0.98960 (True). Precondition-assert the fixture correlation FIRST so seed drift fails as a fixture, not a false verdict. Bailey: near-identical columns give a HIGH PBO — do not write the backwards assertion. `gate_grade` is False for BOTH 8-column fixtures (N=8<10) — do not assert it discriminates; the independent-columns mirror guard is mandatory.
5. **Criterion-5 census measured**: raw `compute_pbo` production call sites outside the wrapper = **0**; four true call sites total (wrapper delegation `analytics.py:240` KEEP; guarded `run_82_3_candidate_backtests.py:206`; two tests that deliberately pin the hazard). Record before=0/after=0 with sites enumerated. `grep -P` required (BSD grep `-E` cannot do `(?!)`).
6. **live_check BEFORE cannot be captured honestly** — that tree state no longer exists. Use the ast-validated mutant output labelled "pre-fix behaviour reproduced from an ast-validated mutant, NOT a historical capture", disclosed explicitly.
7. **Discovered defect queued as 83.0.6** (payload `isError` is not the MCP protocol `is_error`; FastMCP sets the protocol flag only on raise). Out of 83.0.3 scope.

## Hypothesis

A test file that (a) proves the refusal behaviourally on the real tool, (b) proves the near-identical-columns diagnostic with a precondition-asserted fixture and its mirror, (c) kills the raw-compute_pbo revert mutant using the SAME assertion set as the refusal test, and (d) records the census + gate.py disclosure, closes 83.0.3 without touching production code.

## Immutable success criteria (verbatim from `.claude/masterplan.json` 83.0.3)

1. "backend/agents/mcp_servers/risk_server.py::pbo_check routes through backend/backtest/analytics.py::compute_pbo_checked rather than compute_pbo, asserted by a test"
2. "a pnl_matrix with T below S*2 returns an explicit REFUSAL payload rather than a numeric PBO and rather than a veto-clear result, asserted by a test supplying T=10 with S=16 and inspecting the returned refusal field"
3. "the refusal is mutation-tested: reverting pbo_check to call raw compute_pbo makes the criterion-2 test FAIL, and the test must assert on the refusal payload rather than merely that the call returned"
4. "the returned payload carries the gate_grade and columns_diverse fields, and a test asserts that a matrix of 8 near-identical columns with pairwise correlation above 0.99 is reported as columns_diverse false"
5. "the number of remaining call sites reaching raw compute_pbo is recorded as a measured DELTA -- before-count and after-count from committed grep output -- rather than asserted to be zero"
6. "the backend/autoresearch/gate.py minimum-trials bypass is recorded as a disclosure with the verbatim source lines showing that a trial omitting pbo_n_trials skips the refusal, so 83.5 can be written to always emit that key"

**Verification command (immutable):** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_0_3_pbo_false_pass.py -q`

**live_check (immutable):** "verbatim mcp__pyfinagent-risk__pbo_check response for an undersized matrix (T=10, S=16) captured BEFORE the fix showing pbo 0.0 with no veto, and AFTER the fix showing the explicit refusal; plus the verbatim response for an 8-column near-identical matrix showing columns_diverse false" → artifact `handoff/current/live_check_83.0.3.md`; the BEFORE half is satisfied by the labelled mutant reproduction (research finding 6), disclosed as such.

## Explicit decisions

- **D1 — no production code change.** The fix shipped under e5bb9f25/82.27; this step ships the test file + records. Criterion-1 is proven BEHAVIOURALLY (refusal on undersized input, which raw compute_pbo cannot produce) — the 82.23 Q/A already rejected source-token scans.
- **D2 — criterion-4 payload fields are asserted on the NON-refused path** (T=600 matrices): the refusal path deliberately omits gate_grade/columns_diverse (risk_server.py:157-167). The criterion's wording ("the returned payload carries...") is read against the computed payload, which is the only payload that carries them.
- **D3 — the mutation test asserts via a shared `_assert_refusal(r)`** used by BOTH the criterion-2 test and the mutant test (`pytest.raises(AssertionError)`), so criterion 3's "makes the criterion-2 test FAIL" is literal.
- **D4 — one cheap Client-layer round-trip test** asserting `structured_content` carries the refusal (protocol-layer consensus), keeping the `is_error` mismatch OUT (owned by 83.0.6).
- **D5 — census delta before=0/after=0** with the 4 sites enumerated and classified; AST census is authoritative, grep -P committed alongside.

## Correction (cycle 2, 2026-08-07) — D1's coverage attribution was wrong

D1's sentence "Criterion-1 is proven BEHAVIOURALLY (refusal on undersized input, which raw compute_pbo cannot produce)" is FALSE as an attribution: the cycle-1 Q/A built an impostor `pbo_check` that refuses INLINE over raw `compute_pbo` (never touching `compute_pbo_checked`) and all four behavioural tests passed — only the `test_c3` source anchor caught it. So the behavioural refusal is necessary but NOT sufficient for routing. Cycle 2 implements the Q/A's named fixes: (a) this correction, with the original D1 text above left unedited as the record; (b) a do-not-delete comment on the `test_c3` anchor; (c) a direct routing SPY (`test_c1_routing_spy_compute_pbo_checked_is_invoked`, monkeypatching the analytics attribute the function-scoped import resolves at call time) so routing is behaviourally observable. Measured: the impostor shape drives the spy to 0 calls — the spy discriminates it.

## Plan

1. Write `backend/tests/test_phase_83_0_3_pbo_false_pass.py` per the brief's skeleton (fixtures, precondition asserts, mirror guard, shared refusal assertion set, in-file mutant via ast+exec — never writes to disk).
2. Run the suite + the 82.x PBO suites (regression guard).
3. Capture live_check: real MCP tool responses (undersized + near-identical) via mcp__pyfinagent-risk__pbo_check; mutant-reproduced BEFORE, labelled.
4. Record census (AST + grep -P outputs) and the gate.py:43-45 verbatim disclosure in experiment_results_83.0.3.md.
5. qa-verdict Workflow → transcribe → harness_log → flip.

## References

`handoff/current/research_brief_83.0.3.md` (source table therein: FastMCP testing docs, MCP spec, Bailey/Borwein/Lopez de Prado/Zhu SSRN 2326253, arXiv:2408.01760 on equivalent mutants, in-repo test idioms).
