# phase-51.1 EVALUATE -- 2026-06-01

**Step:** 51.1 -- SecretStr unwrap, resurrect 4 dead LLM alpha overlays
**Evaluator:** Q/A (Layer-3, merged qa-evaluator + harness-verifier), single fresh instance
**Verdict: PASS**
**ok:** true | **certified_fallback:** false

This OVERWRITES the stale phase-50.5 critique (archived under
`handoff/archive/phase-50.5/`). A prior 51.1 Q/A instance truncated before
writing; this is the FIRST real 51.1 verdict on UNCHANGED, complete evidence
(not a re-spawn to overturn anything). 0 prior 51.1 verdicts, 0 prior 51.1
CONDITIONALs in `handoff/harness_log.md`.

---

## 1. Harness-compliance audit (5-item, FIRST)

| Item | State | Evidence |
|------|-------|----------|
| researcher BEFORE contract | PASS | `research_brief.md` header `# research_brief -- phase-51.1`; JSON envelope `gate_passed:true`, tier moderate, 5 external sources read in full, 12 URLs, recency scan performed, 8 internal files. Regression commit `d3f34caf` (2026-05-13) pinned with git-blame evidence. |
| contract before GENERATE | PASS | `contract.md` is the 51.1 contract; its 4 success_criteria are VERBATIM-identical to masterplan `51.1.verification.success_criteria` (`.claude/masterplan.json:13737-13742`). Diffed word-for-word -- exact match, none amended. |
| experiment_results + live_check present/complete | PASS | `experiment_results.md` lists file changes + verbatim command output + artifact shape. `live_check_51.1.md` present (`test -f` exit 0) with the $0 proof + criterion-by-criterion table. |
| log-last | PASS | `handoff/harness_log.md` has ZERO 51.1 entries; masterplan 51.1 `status=pending`. Main logs + flips AFTER this PASS, in the correct order. |
| no verdict-shopping | PASS | No prior 51.1 verdict exists (earlier instance never wrote one). This is the first verdict, not a second opinion on a CONDITIONAL. 0 prior consecutive CONDITIONALs -> 3rd-CONDITIONAL auto-FAIL rule N/A. |

## 2. Deterministic checks (reproduced)

```
$ python -m pytest backend/tests/test_phase_51_1_secretstr.py -q
.......                                                                  [100%]
7 passed in 0.16s

$ python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read())"
llm_client.py SYNTAX OK

$ test -f handoff/current/live_check_51.1.md
live_check present
```

Independent QA proof (I constructed the boundary myself, did NOT trust Main):
```
ClaudeClient(SecretStr('sk-ant-z'))._api_key -> 'sk-ant-z', isinstance str, no get_secret_value, != '**********'
ClaudeClient(api_key='plain')._api_key       -> 'plain'  (no double-unwrap)
make_client('claude-haiku-4-5', None, settings-with-SecretStr)._api_key -> str
=> INDEPENDENT QA PROOF OK
```

Regression sweep:
```
$ pytest test_anthropic_fallback.py test_claude_code_client.py test_phase_31_1_fixes.py -q
22 passed
```
(Main also ran test_phase_37_3_budget_tokens -> 32 passed, 1 xfailed; xfail is pre-existing, not introduced here.)

## 3. The 4 IMMUTABLE success criteria

**Criterion 1 -- the 4 overlays pass a plain str (not SecretStr): PASS.**
All 4 sites now read `anthropic_key = unwrap_secret(getattr(settings,"anthropic_api_key",""))`:
`news_screen.py:261`, `macro_regime.py:429`, `pead_signal.py:250`, `meta_scorer.py:168`
(grep-verified; the old `... or ""` truthiness footgun is gone at all four). `test_overlay_services_pass_str_to_claudeclient` (test:64-78) mirrors the exact idiom on a `SecretStr("sk-ant-live")` setting and asserts the resolved key is `str`, equals the real value, and `!= '**********'`. The SDK header can no longer receive a SecretStr.

**Criterion 2 -- ClaudeClient.__init__ self-unwraps; no-op for plain str: PASS.**
`llm_client.py:1243` `self._api_key = unwrap_secret(api_key)`. `test_claude_client_self_unwraps_secretstr` (test:43-48) proves SecretStr->str; `test_claude_client_plain_str_no_double_unwrap` (test:51-54) proves a plain str passes through unchanged. Mechanism is double-unwrap-safe by construction: `unwrap_secret` (`:32-45`) only calls `.get_secret_value()` when `hasattr(v,"get_secret_value")` -- a plain str lacks that attr, so it returns unchanged; and `get_secret_value()` returns a str which also lacks the attr, so it cannot be unwrapped twice.

**Criterion 3 -- $0 unit test proves the boundary str; US pure-quant path unchanged: PASS.**
7 `$0`, no-network tests (ctors defer SDK instantiation to `_get_client`, so construction is free). US byte-identity: `make_client` (`:1913`) still unwraps BEFORE constructing ClaudeClient -- so every existing make_client caller (the live US pure-quant Claude path, when used) is unchanged; my independent proof confirmed `make_client(...,settings-with-SecretStr)._api_key` is a str. Scope diff (`git diff --name-only`) touches ONLY `llm_client.py` + the 4 overlay services + the new test -- NO `paper_trader`, `kill_switch`, `risk_engine`, `screener`, `decide_trades`, `perf_metrics`, or `backtest_engine`. The overlays are additive Signal-Stack flags; the live US screener (momentum/RSI/vol, $0 LLM) imports none of them.

**Criterion 4 -- live_check records the $0 proof; live cycle only if LLM-spend approved: PASS ($0 leg).**
`live_check_51.1.md` records the full $0 proof (the truthiness bug, the `str()` mask footgun avoided, the stored `_api_key` type assertion, make_client unchanged, sibling self-unwrap, no-double-unwrap) plus a criterion-by-criterion table. The live paid-cycle confirmation is correctly FLAGGED operator-LLM-spend-gated and NOT run -- consistent with CLAUDE.md ("LLM API costs require Peder's explicit approval") and the contract's `$0 LLM` GENERATE constraint. The $0 type-assertion is sufficient gate evidence for criteria 1-3 (see Adversarial 4 below).

## 4. Adversarial LLM judgment

- **str() footgun (pydantic #4217) -- AVOIDED.** `unwrap_secret` (`:45`) uses `v.get_secret_value() if hasattr(v,"get_secret_value") else str(v)`. The `str(v)` branch is reached ONLY for non-SecretStr values (where `str()` is safe). A SecretStr always takes the `get_secret_value()` branch, so it NEVER renders as `'**********'`. `test_unwrap_secret_on_secretstr_uses_real_value_not_mask` (test:25-27) explicitly asserts `!= _MASK`. My independent proof re-confirmed `_api_key != '**********'`. The masking footgun cannot silently inject a 401.

- **Double-unwrap / US regression -- PROVEN NONE.** Self-unwrap is guarded by `hasattr`; a plain str passes through (test:51-54 + my proof). The old local `_unwrap` closure in `make_client` was REMOVED (git diff shows the `-def _unwrap` block deleted) and replaced by the module-level helper -- no divergent duplicate. The 2 already-guarded sites (`call_transcript_gpr.py:90-93`, `analyst_narrative_scorer.py:107-113`) were NOT touched (`git diff --stat` empty for both files) and still hold their own `hasattr...get_secret_value` unwrap -- their residual `... or ""` is benign because the per-site unwrap runs on the very next lines. No site double-unwraps.

- **Over-claim on the live cycle -- NOT over-claimed.** Main did NOT run a paid cycle and says so plainly (experiment_results "Operator decision flagged"; live_check "NOT run here"). A type assertion (`isinstance(_api_key, str)` + value equality + `!= mask`) CANNOT be hidden by masking (logs mask the wrapper AND the plaintext identically -- only the type distinguishes them), so the $0 proof substantiates criteria 1-3 rigorously. This is the correct, honest $0-first discipline, not hand-waving.

- **Scope honesty / US engine untouched -- CONFIRMED.** Diff is minimal and surgical (3 ctor self-unwraps in llm_client + 4 one-line edge unwraps + 1 new test file). No risk guard, kill-switch, position-sizing, or perf-metrics path altered. No secret literal in the diff (test uses a fake `sk-ant-test`/`sk-ant-live`). The defense-in-depth extension to OpenAIClient (`:1107`) and BatchClient (`:1804`) is a no-op for the str those receive (test:57-61), a cheap consistent hardening, not a behavior change.

## 5. Code-review heuristics (5 dimensions evaluated)

No BLOCK/WARN/NOTE fired. secret-in-diff: clean (fake test key only). No execution-path / kill-switch / stop-loss / perf-metrics / risk-guard code touched (Dimension 2 N/A by scope). No financial-math file in diff -> financial-logic-without-behavioral-test N/A. Tests are behavioral, not tautological (assert real unwrap values + type + `!= mask`, no `assert x==x`, no over-mocking of the module under test). No sycophancy/verdict-shopping (first verdict, evidence unchanged).

## Verdict

**PASS.** The root cause is genuinely fixed at the ClaudeClient boundary (covers all 4 overlays + the latent SkillFileIdCache/sibling paths), the `str()` mask-injection footgun is explicitly avoided via `get_secret_value()`, there is no double-unwrap and no US pure-quant regression (proven by test + my independent construction), and the $0 type-assertion proof substantiates criteria 1-3 (a type assertion cannot be hidden by masking). The live paid-cycle leg of criterion 4 is correctly operator-gated and out of scope for the $0 GENERATE. All 5 harness-compliance items pass; log-last order is intact.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria met. (1) 4 overlays use unwrap_secret(getattr(...)) -- news_screen.py:261, macro_regime.py:429, pead_signal.py:250, meta_scorer.py:168, the `or \"\"` footgun gone. (2) ClaudeClient.__init__ self-unwraps via unwrap_secret (llm_client.py:1243), no-op for str, no double-unwrap. (3) 7/7 $0 tests + independent QA proof confirm SecretStr->str at the boundary; scope diff touches only llm_client+4 overlays+test, US pure-quant path untouched. (4) live_check_51.1.md records the $0 proof, live cycle correctly operator-gated. str() mask footgun avoided (get_secret_value, test asserts != mask). old _unwrap removed, 2 guarded sites untouched. Deterministic: pytest 7 passed, ast.parse OK, regression 22 passed, INDEPENDENT QA PROOF OK.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "independent_qa_proof", "regression_sweep", "code_review_heuristics", "research_brief", "contract_verbatim_match", "experiment_results", "live_check", "scope_diff"]
}
```
