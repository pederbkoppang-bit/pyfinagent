# Evaluator critique -- 66.3 Cost-truth restoration (Cycle 72, 2026-07-07)

Q/A agent (merged qa-evaluator + harness-verifier), first spawn for 66.3.
Prior-CONDITIONAL count for this step-id: 0 (grep harness_log.md). Evidence
commit: c4f91bdd. Worktree clean for all four code files at evaluation time.

## Verdict: PASS

All 3 immutable criteria met, verified by independent reproduction (not by
trusting the pastes). Four NOTE-level findings recorded below; none load-bearing.

## 1. Harness-compliance audit (5-item)

1. **Researcher before contract** -- PASS. research_brief_66.3.md (mtime 11:37,
   `gate_passed: true`, moderate tier, 6 read-in-full / 75 URLs / recency scan /
   9 internal files). The contract's load-bearing content (gauge-not-constant
   finding, token-derived metered figure, FOCUS billed-vs-estimated discipline)
   traces directly to the brief.
2. **Contract before generate** -- PASS. mtime ordering: brief 11:37 -> contract
   11:40 -> live_check 11:44 -> results 11:45 -> commit c4f91bdd 11:45:14.
3. **Results present, verbatim + honest** -- PASS with flag. Disclosures are
   genuinely honest (price-drift liability, historical gauge rows still
   over-count for naive summers, billing-export cross-check not run + why, and
   the self-correction of the earlier "$8.24 Gemini real spend" claim). Flag:
   the "verbatim" test count is wrong -- see NOTE-1.
4. **Log-last** -- PASS. No Cycle 72 / phase=66.3 entry in harness_log.md yet.
5. **No verdict-shopping** -- PASS. No prior evaluator_critique_66.3.md existed.

Sequencing: operator authorized the early start (masterplan `sequencing_note`,
AskUserQuestion ~08:50 UTC, recorded in live_check_66.5.md section 2); the
66.1->66.2 P0 chain untouched. Ruled in order by the 66.5 Q/A already.

## 2. Deterministic checks (run by Q/A, verbatim)

**(a) Immutable command** (exit 0):
```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_66_3_cost_truth.py -q
10 passed, 1 warning in 2.14s
```
Sentinel contract suite (no regression): `test_phase_62_4_sentinel.py -q` ->
`9 passed in 13.70s`. Combined 66_3 + 62_4 + observability -> `32 passed,
1 warning in 17.11s` (matches the results' combined claim; see NOTE-1 on the
standalone count). `bash -n sentinel.sh` clean; `ast.parse` OK on all 3 py files.

**(b) Criterion 1 -- writer pinned + fixed** -- MET.
- Pin verified in live code: `api_call_log.py:256-265` lazy-fill from
  `autonomous_loop.get_session_cost_usd` (module accumulator `_session_cost`,
  def at autonomous_loop.py:119-121), which grows by
  `analysis.get("total_cost_usd", 0.1)` + `_add_session_cost(cost)`
  (autonomous_loop.py:893-895) and resets at cycle end (~:1401-1403). No flat
  $0.50 constant exists -- the 0.0 -> 0.5 staircase is the gauge. Matches the
  results/live_check pinning to the line.
- Fix verified at `api_call_log.py:253-254`:
  `if session_cost_usd is None and not ok and not input_tok and not output_tok:
  session_cost_usd = 0.0` -- guard fires BEFORE the lazy-fill block and only
  when the caller passed None, so explicit values win; failures WITH tokens
  keep the gauge. Docstring-level comment documents gauge semantics + the
  2026-06 incident (:244-252).
- **Mutation-resistance: genuine.** The autouse fixture monkeypatches the gauge
  to 0.50 (`test file :38`); `test_failed_zero_token_call_logs_zero_cost_not_gauge`
  asserts `row["session_cost_usd"] == 0.0` (:56) -- deleting the guard routes
  through the lazy-fill and stamps 0.50, failing the assert. Anti-over-guard
  coverage: success path asserts 0.50 (:63), token-moving failure asserts 0.50
  (:69), explicit kwarg asserts 0.42 (:75). Behavioral, not tautological.

**(c) Criterion 2 -- billing-class filtering, documented; failure signal** -- MET.
- `metered_spend.py:13-23` (module docstring) documents the choice: provider/
  agent FILTERING, no schema migration -- `agent LIKE 'cc_rail%'`
  (RAIL_AGENT_PREFIX prefix match, :54,:69) + `provider='claude-code'`
  (FLAT_FEE_PROVIDERS, :53), incl. the 2026-06-15 Agent-SDK-credit nuance.
  Same documentation in live_check section 2.
- Price table pinned with source comments (:34-42); longest-prefix match prices
  dated ids (:57-63); unpriced metered models count $0 + fail-visible warning
  (:91-93, :98-103, `unpriced_models` field).
- `sentinel.sh` delegates to `compute_for_date` (SENTINEL_DATE replay hook),
  `rail_failures_today` init None in the report dict, warnings[] entry at >=20
  ("mirrors the 66.1 breaker threshold"), false-premise comment (:63-66 old)
  replaced with the gauge explanation. `SENTINEL_TEST_METERED_USD` +
  `SENTINEL_TEST_BQ_FAIL` overrides intact (:53-59).

**(d) Criterion 3 -- replay + live-day hand-audit match** -- MET, reproduced
independently by Q/A (read-only BQ via ADC):
```
$ python scripts/away_ops/metered_spend.py --date 2026-06-18
{"metered_llm_usd": 0.0043, "rail_failures": 207, "warnings": [], "unpriced_models": []}
$ python scripts/away_ops/metered_spend.py --date 2026-06-17
{"metered_llm_usd": 0.0026, "rail_failures": 137, ...}
```
Byte-identical to the live_check pastes; $0.0043 and $0.0026 vs the $8.00
baseline -- no breach (old logic: $43.00 / $16.51; ~10,000x over-count gone).
- **Vacuousness probe (anti-rubber-stamp):** the tiny figure is NOT an artifact
  of broken token logging. Q/A's own 06-18 BQ breakdown: FLAT_FEE = 207 failed
  0-token rail rows (27 opus-4-7 + 180 sonnet-4-6 = 207 ✓ the rail_failures
  count to the row); METERED = exactly 2 gemini-2.5-flash rows with REAL token
  movement (1829 in / 1488 out, zero_tok=0). Hand arithmetic:
  1829x$0.30/M + 1488x$2.50/M = $0.00055 + $0.00372 = $0.0043 ✓.
- **Live-day (2026-07-07) independent hand audit by Q/A** (own SQL aggregation
  + own inline price math, no import of compute_metered):
  module `{"metered_llm_usd": 0.0025, "rail_failures": 0}` vs
  hand audit `claude-haiku-4-5-20251001 n=2 i=2000 o=100 -> $0.0025`. EXACT
  match. (The live_check's $0.0013 was correct at its 11:44 capture -- one
  haiku ticket call then; a second identical call landed before Q/A ran:
  per-call $0.00125. Temporal drift, both captures correct -- NOTE-4.)
- **Live sentinel run** (post-change, by Q/A):
```
{"metered_llm_usd_today": 0.0025, "rail_failures_today": 0, "baseline_usd": 8.0,
 "kill_switch_paused": false, "flags_match_tokens": true, "ok": true,
 "gates_failed": [], "warnings": []}
```

**(e) Scope** (git show --stat c4f91bdd) -- CLEAN. Exactly:
api_call_log.py (+11), metered_spend.py (NEW 139), sentinel.sh (+35/-14),
test_phase_66_3_cost_truth.py (NEW 136), 4 handoff artifacts. NO
autonomous_loop.py change (gauge accumulator untouched per contract scope),
NO schema migration, NO trading code.

**(f) Criteria integrity** -- all 3 contract criteria byte-identical to
masterplan phase-66/66.3 (normalized only for line-wrap); immutable command
string matches.

## 3. Code-review heuristics (5 dimensions)

Evaluated; no BLOCK, no WARN. Notables: BQ access is read-only, bounded
(single-date filter, 25s timeout) and injection-safe (--date goes through
`ScalarQueryParameter("d", "DATE", ...)`, not string interpolation);
sentinel's broad except is fail-VISIBLE (gates_failed + warning), not a
silenced risk guard; no perf_metrics surface touched (observability, not
Sharpe/drawdown); no secrets; behavioral tests cover the new financial-ish
logic; prints confined to CLI/script context. Frontend untouched (no
eslint/tsc gate); no UI claims (no Playwright gate).

## 4. LLM judgment

- **Price-table plausibility**: gemini-2.5-flash $0.30/$2.50 and pro $1.25/$10
  match Google's published per-Mtok pricing; sonnet $3/$15, opus 4-7/4-8 $5/$25,
  haiku $1/$5, fable-5 $10/$50 match Anthropic pricing as recorded in CLAUDE.md;
  cache read 0.1x / creation 1.25x input is the standard Anthropic 5-min
  ephemeral schedule. Plausible and sourced.
- **claude-code exclusion safety**: correct. Max flat fee / Agent-SDK credit is
  not per-token API billing (disclosed in the docstring incl. the 2026-06-15
  change); the digest-facing sentinel JSON change is purely additive
  (rail_failures_today) and the 62.4 sentinel contract suite (9 passed) guards
  consumer shape.
- **Scope honesty**: strong. The results explicitly CORRECT the earlier
  phase-66 "$8.24 Gemini real spend" claim as itself a gauge-sum artifact
  (true away-window metered spend ~$0.10-0.20 token-derived, direction of all
  prior conclusions unchanged). Self-correction of one's own prior claim is
  exactly the disclosure posture the harness wants; weighed FOR the work.
  Historical-row over-count liability registered for 63.3.
- **Research-gate compliance**: contract cites the brief's findings and the
  fix shape follows them (token-derived billed-cost, FOCUS discipline,
  fail-visible unpriced models).

## 5. Findings (all NOTE severity -- PASS-with-flag)

- **NOTE-1 (results accuracy)**: experiment_results/live_check claim
  "11 passed" and the commit message says "11 new tests"; the committed AND
  current test file has exactly 10 `def test_` and the command yields
  `10 passed`. The docs' own combined figure (32 = 10 + 9 sentinel + 13
  observability) is arithmetically consistent with 10, proving the final code
  WAS tested post-change and the "11" is a stale transcription (likely a test
  merged/removed before commit without refreshing the standalone paste). No
  concealment, no failing test, no criterion references a count -- but
  "verbatim" blocks must reproduce; corrected here for the record.
- **NOTE-2 (guard edge)**: the guard tests `input_tok`/`output_tok` only; a
  failed call with ONLY cache-token movement would stamp 0.0. Immaterial: the
  metered figure is token-derived independent of session_cost_usd, and
  criterion 1's "auth/connection failures with 0 tokens" moves nothing.
- **NOTE-3 (price-tier edge)**: gemini-2.5-pro >200k-context tier ($2.50/$15)
  is not modeled; the fail-visible warning covers UNPRICED models, not
  priced-but-wrong-tier. Immaterial at current volumes; price drift already
  disclosed as a maintenance liability.
- **NOTE-4 (temporal drift)**: live-day figures moved $0.0013 -> $0.0025
  between the 11:44 capture and Q/A's run (a second haiku ticket call). Both
  captures independently verified correct; expected behavior for a live gauge.

## 6. JSON verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met and independently reproduced: (1) writer pinned to api_call_log.py lazy-fill + autonomous_loop accumulator and fixed at :253-254 with genuine mutation-resistant tests (gauge mocked 0.50, assert 0.0); (2) filtering choice documented in metered_spend.py docstring + live_check, pinned price table, fail-visible unpriced models, sentinel delegates with rail_failures_today + >=20 warning, SENTINEL_TEST_* intact; (3) replays reproduced byte-identical ($0.0043/207, $0.0026/137, both << $8), live-day module $0.0025 == Q/A's own independent BQ hand audit $0.0025, live sentinel ok:true. Scope clean, criteria byte-identical, sentinel suite 9 passed no regression. 4 NOTEs, chief: results say '11 passed' but the suite is 10 passed (stale transcription; combined 32 confirms 10).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "sentinel_regression_suite", "combined_regression", "mutation_resistance_review", "criteria_byte_identity", "scope_diff", "bq_replay_reproduction_0617_0618", "bq_0618_vacuousness_probe", "live_day_independent_hand_audit", "sentinel_live_run", "code_review_heuristics"]
}
```
