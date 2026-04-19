# phase-12.3 Q/A evaluator critique (qa_123_v1)

**Step:** phase-12.3 -- canary split + SLO diff tooling (rainbow_canary.py + canary-split.yaml).
**Cycle:** 1
**Date:** 2026-04-19
**qa_id:** qa_123_v1

---

## 5-item harness-compliance audit

1. **Research gate:** PASS. `handoff/current/phase-12.3-research-brief.md` present (mtime 15:56:37), gate_passed=true verified in brief, 3-query discipline.
2. **Contract PRE-commit:** PASS. `phase-12.3-contract.md` mtime 15:57:26 < `rainbow_canary.py` 15:57:58 < test file 15:58:41 < yaml 15:58:18 < results 15:59:34. Contract precedes all generated artifacts.
3. **Experiment results match diff:** PASS. `git ls-files --others` shows exactly the 3 new files + 3 phase-12.3 handoffs; results doc lists same.
4. **harness_log.md last entry:** PASS. Last cycle block is `Cycle N+56 ... phase=12.2 result=PASS` — NOT 12.3 (log is last, still to be written).
5. **No verdict-shopping:** PASS. Cycle-1, no prior Q/A verdict on 12.3.

Audit result: 5/5.

---

## Deterministic checks

| ID | Check | Result |
|----|-------|--------|
| A  | Syntax parse (rainbow_canary.py + test) | `SYNTAX_OK` (yaml not parsed via ast; see F) |
| B  | Immutable: `pytest backend/tests/test_rainbow_canary.py -q` | **13 passed in 0.01s** |
| C  | Regression: `pytest backend/tests/ -q --ignore=test_paper_trading_v2.py` | **103 passed, 1 skipped** |
| D  | Scope: `git ls-files --others --exclude-standard` filtered to phase-12.3 | 3 code + 3 handoff (contract/research/results) only; no stray file |
| E  | Import smoke (all 4 exports) | `IMPORT_OK` |
| F  | `yaml.safe_load_all(canary-split.yaml)` | `YAML_OK docs=1` |
| G  | Math spot-checks | see below |
| H  | `dataclasses.asdict(SLODiff(...))` | `type=dict keys=['blue_p95','blue_samples','green_p95','green_samples','ratio','reason','regression','threshold']` — confirms `@dataclass` |

### G. Math spot-check outputs

- `percentile(list(range(1,101)), 50) == 50.5` -- linear interpolation confirmed (NumPy-style, not nearest-rank).
- `compute_slo_diff(blue=[100]*20, green=[125]*20)` → `ratio=1.25, regression=True, reason='ok'` -- 125/100 > 1.2 default threshold, flagged correctly.
- `compute_slo_diff(blue=[100]*5, green=[300]*5)` → `regression=False, reason='insufficient_samples'` -- min_samples fail-open confirmed (5 < 10 default).
- `compute_slo_diff(blue=[100]*20, green=[80]*20)` → `ratio=0.8, regression=False, reason='ok'` -- one-sided threshold confirmed (green faster is NOT a regression).
- `compute_slo_diff(blue=[0]*20, green=[1]*20)` → `ratio=inf, regression=True, reason='ok'` -- zero-blue edge: Python float div-by-zero yields `inf`, which satisfies `> threshold` so regression is flagged. **Note:** no dedicated `reason='degenerate_blue'` branch; the `inf > 1.2` path coincidentally produces the safe outcome (flag as regression). Acceptable for MVP but a caveat worth listing.

All deterministic checks: PASS.

---

## LLM judgment

- **Threshold semantics (one-sided):** confirmed by code-read of `compute_slo_diff` docstring at `rainbow_canary.py`: "The threshold is one-sided (we only alarm when GREEN is slower)." Test `test_compute_slo_diff_green_faster_not_regression` pins this. Sensible design — rolling back because the canary is faster would be nonsense.
- **Fail-open completeness:** empty → `reason='empty'`; below min_samples → `reason='insufficient_samples'`; both verified by G.3 + inspection of signature at `rainbow_canary.py:n_<cutoff>`. `blue_p95=0` edge is NOT a dedicated branch but yields a safe outcome via `inf > threshold`. Recommend a future tightening (phase-12.x) to emit `reason='degenerate_blue'` for operator clarity.
- **`canary_snapshot_from_buffer` coupling:** reads `api_call_log._buffer` in-process. Results doc §"Known caveats #3" explicitly flags this as in-process only; documents BQ `pyfinagent_data.api_call_log` as the multi-pod path. Caveat is disclosed — acceptable for MVP.
- **Replica-weighted canary:** `canary-split.yaml` line 1 + 21-22 document the `1/20 = 5%` arithmetic AND the Gateway API HTTPRoute path for 1%-granularity futures. Scope bounds honestly disclosed.
- **Pre-Q/A self-check:** results §"Known caveats #5" claims Main ran immutable + regression + yaml + import. Reproduced all 4 here; all green.
- **Contract alignment:** all 7 success criteria in `phase-12.3-contract.md` have matching evidence in results §"verification evidence" table.
- **Scope honesty:** docstrings + results disclose three explicit caveats (no `color` column, in-process buffer, 5% replica granularity). No overclaim.

---

## Violated criteria

None.

## Violation details

None.

## checks_run

`["protocol_audit_5", "syntax", "verification_command", "regression", "scope_diff", "import_smoke", "yaml_parse", "math_spot_checks", "dataclass_serialization", "threshold_semantics", "fail_open_completeness", "contract_alignment", "scope_honesty"]`

## Verdict

**PASS**

- 13/13 immutable tests pass (<0.01s).
- 103p/1s regression preserved.
- All 7 contract criteria satisfied with reproducible evidence.
- Math semantics (one-sided threshold, linear-interp p95, fail-open) all verified empirically.
- 5/5 protocol audit clean.
- Scope caveats disclosed, not hidden.

**Non-blocking follow-up (not a violation):** consider adding a dedicated `reason='degenerate_blue'` branch when `blue_p95 == 0` in a future phase-12.x iteration; current `inf` path is safe but less self-documenting.

certified_fallback: false
