# Evaluator Critique -- phase-28.16 -- M&A pre-announcement aggregator (FINAL phase-28 item)

**Step ID:** phase-28.16
**Date:** 2026-05-18
**Cycle:** 1
**Verdict:** **PASS** (with NOTE: research-brief artifact size flag — see Audit Item 1)

---

## STEP 1 — 5-item harness-compliance audit

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Researcher gate | **NOTE / PASS-with-flag** | `handoff/current/phase-28.16-research-brief.md` exists (5 lines, supplement-Gap-3 reference; cites Augustin-Brenner-Subrahmanyam + Duong-Pi-Sapp 2025). **NOTE:** The brief lacks the standard JSON envelope (`fallback_authoring: Main`, `external_sources_read_in_full`, `gate_passed`). Per user directive cited in prompt: "If Researcher crashes again, log failure signature and fall back to direct WebFetch; do not block the run." Main's fallback authoring is documented in `contract.md:5` and `contract.md:10` (5 read in full + 6 failed). Cited research is in `experiment_results.md:71` and 5 legality surfaces. Treating as PASS-with-NOTE per user directive; recommend backfilling the JSON envelope to the brief file for harness-log auditability. |
| 2 | Contract pre-commit | PASS | `contract.md` written before generate; immutable verification command verbatim at line 25; success criteria copied (5 items). |
| 3 | Results present | PASS | `experiment_results.md` (113 lines) with verbatim verification output, synthetic 6-ticker test results, 13D stub honesty section, 5-place legality coverage table. |
| 4 | Log-last discipline | PASS | `harness_log.md` shows cycle 30 (phase-28.17) as last entry. No phase-28.16 log entry yet — correct: log appends AFTER Q/A PASS and BEFORE status flip. |
| 5 | No verdict-shopping | PASS | First Q/A pass for phase-28.16 (no prior critique). Zero prior CONDITIONAL/FAIL for this step-id in `harness_log.md`. Counter clean. |

---

## STEP 2 — Deterministic checks (verbatim)

### Immutable verification command
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/ma_preannounce_screen.py').read()); print('syntax OK')" && grep -q 'ma_preannounce_enabled' backend/config/settings.py && grep -qE '13[dg]|SCHEDULE.13' backend/services/ma_preannounce_screen.py
syntax OK
EXIT 0
```
**PASS.**

### 4-file syntax
- `backend/services/ma_preannounce_screen.py`: syntax OK
- `backend/config/settings.py`: syntax OK
- `backend/tools/screener.py`: syntax OK
- `backend/services/autonomous_loop.py`: syntax OK

### Settings defaults (runtime via `Settings()`)
```
enabled=False strong=0.1 moderate=0.05
SETTINGS DEFAULTS: OK
```
Confirms default-OFF, +10% strong, +5% moderate per contract spec.

### Public API importable
```
from backend.services.ma_preannounce_screen import (
    compute_ma_preannounce_signals, apply_ma_preannounce_to_score,
    _classify_boost, _fetch_13d_filings_for, MAPreannounceSignal
)
Public API: importable
```

### `_classify_boost` boundary table
```
legs=0 -> (1.00, 'none')
legs=1 -> (1.05, 'moderate')
legs=2 -> (1.10, 'strong')
legs=3 -> (1.10, 'strong')   # capped at strong (no triple boost)
```

### `rank_candidates(ma_preannounce_signals=None)` kwarg
```
KWARG PRESENT (default=None)
```

### Synthetic 6-ticker / 3-leg unit test (per contract)
Inputs: tickers=[AAPL,NVDA,TSLA,COIN,GME,UNUSED]; options_signals={AAPL,NVDA,COIN}; insider_signals={AAPL,TSLA,COIN}; schedule_13d={COIN}.

```
Returned signals: 4
  AAPL: legs=['options','insider']         count=2 boost=1.10 tier=strong
  NVDA: legs=['options']                   count=1 boost=1.05 tier=moderate
  TSLA: legs=['insider']                   count=1 boost=1.05 tier=moderate
  COIN: legs=['options','insider','13d']   count=3 boost=1.10 tier=strong (capped)
```
- Exactly 4 signals returned (GME/UNUSED correctly excluded with 0 legs).
- AAPL strong (2 legs) — PASS.
- NVDA moderate (1 leg) — PASS.
- COIN strong with 3 legs CAPPED at 1.10 (no triple-boost) — PASS.
- TSLA moderate (1 leg) — PASS bonus.

### Apply identity paths
```
  AAPL: 10.000 -> 11.000 (+10%)
  NVDA: 10.000 -> 10.500 (+5%)
  GME:  10.000 -> 10.000 (identity, 0 legs)
  None signals: 10.0 -> 10.0 (identity)
  No ticker:    10.0 -> 10.0 (identity)
```
All 5 identity / boost paths behave as designed.

### 13D stub honesty
```
asyncio.run(_fetch_13d_filings_for('COIN'))  ->  []   # expected []
```
Stub returns empty list. Docstring at lines 56-72 honestly documents the HTTP 403 from `efts.sec.gov` and lists two future implementation paths (`httpx.AsyncClient` with browser UA, or `sec-edgar` PyPI). TODO marker present at line 73: `# TODO phase-28.16-followup-13d-edgar`. Module docstring (lines 14-19) flags Leg 3 as STUBBED. **HONEST.**

---

## STEP 3 — LLM judgment & code-review heuristics

### Contract alignment
All 5 immutable success criteria evidenced:

| Criterion | Evidence | Result |
|---|---|---|
| `ma_preannounce_screen_module_created` | `backend/services/ma_preannounce_screen.py` (151 lines on disk; contract said 130, slightly longer due to docstrings) — importable | PASS |
| `three_legs_present_OTM_options_and_Form_4_cluster_and_13D_polling` | Module docstring lines 4-19 names all 3 legs; `compute_ma_preannounce_signals` lines 116-121 evaluates all 3; grep matches `13[dg]|SCHEDULE.13` (immutable verification clause) | PASS |
| `uses_only_public_data_per_legality_boundary_note` | LEGALITY visible in: (a) module docstring lines 26-29, (b) `settings.py:300` field description, (c) `contract.md:44-49`, (d) `experiment_results.md:75-90` (5-place table), (e) `live_check_28.16.md:10-12` and `:54` — **5 surfaces confirmed** | PASS |
| `feature_flag_ma_preannounce_enabled_default_false` | `Settings().ma_preannounce_enabled == False` at runtime | PASS |
| `live_check_lists_M_A_signal_tickers_for_one_cycle` | `live_check_28.16.md` documents 4 signals across 6 tickers with per-ticker leg checklist, aggregation rule, score impact, leg-by-leg provenance | PASS |

### Default-OFF discipline
- `ma_preannounce_enabled: bool = Field(False, ...)` — `settings.py:300`
- `autonomous_loop.py:305` guarded by `getattr(settings, "ma_preannounce_enabled", False)`
- `screener.py:366` guarded by truthy `ma_preannounce_signals` (None default at line 233)
- Three layers of OFF-by-default protection. Production unchanged when flag stays False. **PASS.**

### Pure-function design (no I/O in compute)
`compute_ma_preannounce_signals` (lines 85-136): no `await`, no network, no file I/O. Reads dicts, returns dict. Single `logger.info` summary line at end. **PASS.** The only async I/O is `_fetch_13d_filings_for` which is currently a stub returning `[]`.

### Reuses phase-28.9 + 28.10 signals (no duplicate fetch)
`autonomous_loop.py:311-312` passes already-fetched `options_surge_signals` (28.9) and `insider_signals` (28.10) into the aggregator. **Zero additional network cost** — verified by reading lines 301-319 inline comment. `schedule_13d_signals={}` is passed empty (line 313). **PASS.**

### 13D stub honesty
The stub does NOT pretend to work:
- Module docstring lines 14-19: "STUBBED for Phase-2... returns [] until the EDGAR client is wired."
- Function docstring lines 56-72: explains HTTP 403, lists endpoint + 2 future paths.
- `live_check_28.16.md:54-65`: "currently returns []" + follow-up section "phase-28.16-followup-13d-edgar".
- `experiment_results.md:69-73`: "STUB documented" with explicit HTTP 403 attribution.
- Aggregator behavior: COIN with `schedule_13d={'COIN':{}}` correctly produces a 3-leg signal in the synthetic test, demonstrating the wiring would work once `_fetch_13d_filings_for` returns real data. **PASS.**

### Researcher fallback handled per user directive
- Brief authored by Main after Researcher stopped mid-step (explicitly cited in `contract.md:5,10` and `experiment_results.md:6`).
- 5 sources read in full + 6 failed documented in contract (lines 10-13).
- **NOTE (downgraded from WARN):** Brief file itself is only 5 lines — JSON envelope (`fallback_authoring: Main`, `external_sources_read_in_full`, `gate_passed`) per `.claude/rules/research-gate.md` is absent from the file even though the substantive evidence is captured in the contract. Per user's standing directive "do not block the run", this is recorded as PASS-with-flag, not a blocker. Recommend backfilling the envelope to the brief file post-cycle for audit parity.

### Code-review heuristics — clean across 5 dimensions

| Dimension | Result | Note |
|---|---|---|
| 1. Security | PASS | No secrets, no `subprocess`/`eval`/`exec`, no `yaml.load`, no `pickle.load`. New module is pure-aggregator with one optional `httpx`-future stub. |
| 2. Trading-domain | PASS | No `kill_switch`/`stop_loss`/`perf_metrics` touched. No risk-engine math changed. Default-OFF guards in place. |
| 3. Code quality | PASS | No `except Exception: pass`, no `print()`, no global mutable state, no `eval`. Type hints on all public functions. Pydantic model with `extra="forbid"`. Em-dashes (U+2014) appear in docstrings only — NOT in logger format strings (line 132 logger call uses ASCII-only `%d` / `%g` format). |
| 4. Anti-rubber-stamp | PASS | Behavioral 6-ticker / 3-leg test validates 4 PASS assertions + cap behavior + identity paths. 13D stub honesty verified by returning `[]` exactly as documented. Aggregator logic tested at boundaries (legs=0,1,2,3). |
| 5. LLM-evaluator | PASS | First cycle — no rebuttal context. Verdict reflects evidence directly, with file:line citations throughout. |

### Scope honesty
`experiment_results.md` does NOT overclaim Leg 3. The 13D leg is explicitly STUBBED, with the HTTP 403 root cause cited and two recovery paths named. The aggregator's behavior with a non-empty `schedule_13d_signals` dict is verified by synthetic test (COIN 3-leg case) — showing Leg 3 wiring is correct, only the fetch is deferred. **PASS.**

---

## STEP 4 — Final verdict

**PASS.**

All 5 immutable success criteria met with file:line evidence. Deterministic verification command exits 0. Synthetic 3-leg test produces exactly 4 signals with correct tier assignments (AAPL strong 2-leg, NVDA moderate, TSLA moderate, COIN strong capped at 3 legs). 13D stub honestly documented in 5+ surfaces. Legality boundary visible in 5 places. Zero broad-except, zero print, zero perf_metrics bypass. Researcher fallback explicitly handled per user directive (NOTE on brief-file envelope shape, downgraded from WARN per directive "do not block").

**Recommendation:** flip phase-28.16 to `done`. Append cycle 31 to `harness_log.md` BEFORE the status flip per log-last discipline. **Phase-28 = 18/18 complete.**

**Follow-up tracked:** `phase-28.16-followup-13d-edgar` (wire authenticated SEC EDGAR client — `httpx.AsyncClient` with browser UA or `sec-edgar` PyPI dep; endpoint `https://efts.sec.gov/LATEST/search-index`).

---

## JSON verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate": "PASS-with-NOTE (brief file lacks JSON envelope; fallback authoring documented in contract+results per user directive 'do not block')",
    "contract_pre_commit": "PASS",
    "results_present": "PASS",
    "log_last": "PASS (no phase-28.16 entry yet; correct order log-after-Q/A)",
    "no_verdict_shopping": "PASS (first cycle, zero prior CONDITIONAL/FAIL)"
  },
  "deterministic_checks": [
    "immutable_verification_command_exit_0",
    "4_file_syntax_ok",
    "settings_defaults_runtime_off_strong_0.10_moderate_0.05",
    "public_api_importable",
    "classify_boost_boundary_table_0_1_2_3_legs",
    "rank_candidates_kwarg_ma_preannounce_signals_None_default",
    "synthetic_6_ticker_3_leg_unit_test_4_signals_AAPL_strong_NVDA_moderate_TSLA_moderate_COIN_3leg_capped",
    "apply_identity_paths_5_cases",
    "13d_stub_returns_empty_list_honest_docstring"
  ],
  "violated_criteria": [],
  "violation_details": "",
  "certified_fallback": false,
  "checks_run": 9
}
```
