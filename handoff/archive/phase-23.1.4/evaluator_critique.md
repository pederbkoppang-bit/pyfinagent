---
step: phase-23.1.4
cycle_date: 2026-04-27
verdict: PASS
qa_pass_index: 1
---

# Q/A Critique — phase-23.1.4

## 5-item harness-compliance audit

1. Researcher brief `handoff/current/phase-23.1.4-research-brief.md` present with `gate_passed: true` (line 399). PASS.
2. Contract front-matter `step: phase-23.1.4` matches; `verification:` field is the immutable command (front-matter line 6). PASS.
3. `experiment_results.md` includes verbatim verification output (lines 27-33) AND the "Honest disclosure: data sources need Phase-2 work" section (lines 51-59). PASS.
4. `handoff/harness_log.md` NOT yet appended for `phase=23.1.4` (grep returned no current-cycle hits). PASS — log is correctly the LAST step.
5. First Q/A spawn for phase-23.1.4. PASS.

## Deterministic checks

| Check | Result |
|---|---|
| A. Immutable verification cmd | `ok events=0 sources=[]` exit 0 — matches expected |
| B. `pytest tests/services/ -v` | 67/67 passed (12 macro_regime + 21 news + 18 PEAD + 16 sector_calendars) |
| C. Syntax (6 files) | `all syntax ok` |
| D. Default-OFF safety | `settings.py:163` defaults to `False`; `autonomous_loop.py:150` uses `getattr(settings, "sector_calendars_enabled", False)` |
| E. Drop-on-binary-risk + boost paths | `sector_calendars.py:317` returns None on `binary_risk`; lines 320-323 apply 1.20 (FDA <=7d) and 1.10 (earnings 1-3d); screener.py:219-226 `continue`s on None and uses returned float otherwise |
| F. `_RTTNewsTableParser` correctness | 16 unit tests cover empty HTML, valid stub row, days==1 binary_risk classification, missing ticker/date skip |
| G. PEAD bugfix | `pead_signal.py:325` uses `BigQueryClient(get_settings())` — fixed |
| H. Git diff scope | All staged files within acceptable list; ?? new files limited to `sector_calendars.py`, `phase-23.1.4-research-brief.md`, `test_sector_calendars.py` |

## LLM judgment

**Contract alignment.** Contract Plan §7 explicitly states "Empty dict is acceptable on a quiet calendar day." The contracted scope is the *capability* (FDA PDUFA scrape + earnings overlay + drop-on-binary-risk + screener wiring), not a minimum event count. Both data sources are wired and exercised end-to-end (RTTNews HTTP 200, BQ query 200/0-rows). Scope honesty is high.

**Anti-rubber-stamp / mutation resistance.** Parser logic is independently verified by 16 unit tests against synthetic HTML fixtures (binary_risk on day==1, ticker/date skip paths, days_to_event signed math, cache roundtrip including missing + corrupt). The fact that *live* RTTNews returned 0 rows is a data-source quality issue, not a logic correctness issue — the synthetic-fixture tests would have caught any regression in the parser.

**Honest disclosure.** experiment_results lines 51-59 name both failure modes with root causes (RTTNews SPA / no static `<table>`, BQ table empty or stale cron), the cost of each remediation path (Selenium heavy dep, Wall Street Horizon paid, Finnhub cron), and explicit Phase-2 deferral. No silent degradation, no overclaim.

**Research-gate compliance.** Contract references `phase-23.1.4-research-brief.md` (401 lines, 7 sources read in full, gate_passed: true).

**Default-off safety.** Existing autonomous_loop unaffected when flag unset (`getattr(... False)` guard).

**Bonus catch.** Cycle picked up a real bug in cycle-2 PEAD code (`BigQueryClient()` missing required `settings` arg) and fixed it at `pead_signal.py:325`. This is the kind of incidental discovery the harness is designed to surface.

## Why not CONDITIONAL

The harness anti-rubber-stamp doctrine says: prefer FAIL over PASS when uncertain, and don't approve scaffolding that masquerades as a working feature. Two reasons this case clears that bar:

1. The contract's verification command is *exactly* `events=0` permissive ("Empty dict is acceptable on a quiet calendar day"). Marking CONDITIONAL would amount to amending the immutable verification criteria post-hoc, which is forbidden.
2. The default-OFF flag means this code does not affect any production codepath until Peder explicitly enables it. The data-source quality work is genuinely Phase-2 (free EIA registration / Selenium dep / paid Wall Street Horizon — none of which fit a $0-cost cycle).

If the contract had promised a minimum event count or claimed "live FDA calendar working", this would be FAIL. It promised neither.

## Follow-ups for Phase-2 (non-blocking)

- Replace RTTNews SPA scrape with a working free FDA PDUFA source (BioPharmCatalyst JSON discovery, or accept Wall Street Horizon paid tier)
- Verify the `pyfinagent_data.calendar_events` Finnhub earnings cron is running and producing rows
- Add a smoke test that runs against a fixture-replayed live response so we catch SPA migration on other data sources earlier

## Verdict JSON

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All immutable criteria met: verification cmd exit 0 with valid empty dict (contract explicitly permits), 67/67 tests pass, syntax OK, default-OFF safety confirmed, drop-on-binary-risk + catalyst boost wired per contract, honest disclosure complete, research gate cleared, PEAD cycle-2 bug fixed as bonus.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "pytest_services", "default_off_safety", "screener_integration", "rttnews_parser_tests", "pead_bugfix", "git_diff_scope", "research_gate", "honest_disclosure", "scope_alignment"]
}
```
