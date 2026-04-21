# Q/A Critique -- phase-7 / step 7.4 (ETF flows ingestion scaffold)

**Cycle:** 1 **Date:** 2026-04-19 **Q/A id:** `qa_74_v1`
**Verdict:** **PASS**

---

## 5-item protocol audit

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Researcher spawn ran with `gate_passed: true`, >=5 sources in full, three-variant queries, recency scan | PASS | `handoff/current/phase-7.4-research-brief.md` JSON envelope lines 187-193: `external_sources_read_in_full=5`, `urls_collected=13`, `recency_scan_performed=true`, `gate_passed=true`. Brief head shows three-variant query list. |
| 2 | Contract written BEFORE production code (mtime ordering) | PASS | `stat -f "%m %N"` -> `contract.md=1776633077` < `etf_flows.py=1776633114`. Delta 37s: contract then code. Main's "researcher overreached, deleted, re-owned" disclosure is consistent with the timestamps. |
| 3 | Experiment results present with verbatim output | PASS | `phase-7.4-experiment-results.md` contains the `ast.parse` output, `--dry-run` JSON line, regression `152 passed, 1 skipped`, and a contract-criterion table. |
| 4 | Log-last discipline: `harness_log.md` last block is 7.3, not 7.4 | PASS | `tail -40 harness_log.md` last cycle-block header is `## Cycle -- 2026-04-19 23:05 UTC -- phase=7.3 result=PASS`. No 7.4 entry exists (`grep 7.4 harness_log.md` returns only historical 3.7.4 / 4.7.4 / 4.4.4.2 hits). |
| 5 | First Q/A on 7.4 (no verdict-shopping) | PASS | No `phase-7.4-evaluator-critique*.md` exists prior to this one. |

All 5 PASS.

---

## Deterministic checks A-F

| Check | Command | Expected | Actual | Result |
|-------|---------|----------|--------|--------|
| A | `python -c "import ast; ast.parse(open('backend/alt_data/etf_flows.py').read()); print('SYNTAX OK')"` | exit 0, `SYNTAX OK` | exit 0, `SYNTAX OK` | PASS |
| B | `python -m backend.alt_data.etf_flows --dry-run` | JSON with `scaffold_only: true, ingested: 0` | `{"ts": "2026-04-19T21:14:42.654018+00:00", "dry_run": true, "ingested": 0, "scaffold_only": true}` | PASS |
| C | `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` | 152 passed / 1 skipped | `152 passed, 1 skipped, 1 warning in 12.97s` | PASS |
| D | `file etf_flows.py` reports ASCII | ASCII text | `Python script text executable, ASCII text` | PASS |
| E | Scope: only new `backend/alt_data/etf_flows.py` + handoff trio | Tight | git status shows no new tracked files beyond the scaffold; `backend/alt_data/` is untracked (consistent with cycles 7.1-7.3) | PASS |
| F | Mtime ordering contract < code | contract older | 1776633077 < 1776633114 (37s earlier) | PASS |

6 / 6 PASS.

---

## LLM judgment

- **Scaffold completeness.** `etf_flows.py` implements all 6 promised functions: `fetch_issuer_page` (lines 69-78), `derive_flow` (81-92), `ensure_table` (127-139), `upsert` (142-164), `ingest_tickers` (167-212), `_cli` (215-231). Helpers `_resolve_target` and `_get_bq_client` added (sensible house-style reuse from `finra_short.py`/`f13.py`). Stubs return empties (`{}`, `None`, `0`, `False`), not `NotImplementedError`. `ingest_tickers` actually walks the tickers through the stub and will emit rows the moment `fetch_issuer_page` is filled in -- this is a *real* scaffold, not a shell. PASS.
- **DDL shape matches house pattern.** `_CREATE_TABLE_SQL` (lines 44-62): partition by `as_of_date`, cluster by `ticker, issuer`. 10 columns in the exact order the contract promised (`flow_id, as_of_date, ticker, issuer, nav, shares_out, shares_out_prev, flow_usd, source, raw_payload`). `raw_payload JSON` is consistent with house style (and `f13.py`). PASS.
- **`_STARTER_TICKERS` baked in.** Lines 38-42: 24 tickers as a `tuple[str, ...]` -- `SPY QQQ IWM DIA VTI VOO VEA VWO EFA EEM AGG TLT IEF HYG LQD JNK GLD SLV USO XLK XLF XLE XLV XLP`. Exactly the 24 the research brief's objective section specifies. PASS.
- **Rate-limit = 1 req / 2 s.** Line 36: `_RATE_INTERVAL_S = 2.0  # 1 req per 2s per compliance row 7.4`. `_rate_limit()` at line 65-66 uses `time.sleep(_RATE_INTERVAL_S)` and is called inside `fetch_issuer_page` (line 76). Minor contract drift: contract text wrote `_RATE_INTERVAL_S = 1.0 / 0.5` (algebraically 2.0); code uses the literal `2.0`. Same value; non-issue. PASS.
- **Protocol-discipline artifact credible.** The experiment_results disclosure says Main deleted the researcher-authored scaffold and re-owned GENERATE. Mtime ordering corroborates: contract at 22:51:17, code at 22:51:54 (37-second gap, same cycle, contract first). If Main had kept the researcher's file, the code mtime would be earlier than the contract. The disclosure matches the physical evidence. PASS.
- **Scope honesty.** Git status for the cycle shows the usual long-standing modifications from prior cycles; the only 7.4-scope addition is `backend/alt_data/etf_flows.py` (untracked, same pattern as 7.1-7.3) plus the handoff trio. No accidental edits outside scope. PASS.
- **Compliance alignment.** Module docstring (lines 13-14) cites `docs/compliance/alt-data.md` row 7.4. `_USER_AGENT` (line 34) is identifiable with contact info, per SEC-EDGAR-style etiquette called out in `security.md`. PASS.

---

## Violated criteria

None.

## Advisories (non-blocking, for phase-7.12 tracker)

- `adv_74_scaffold_only`: `fetch_issuer_page` returns `{}`; `ingest_tickers` returns 0. Live implementation (issuer-page HTML/JSON scrapers for iShares, Vanguard, SSGA) lands in phase-7.12. Contract + experiment_results both flag this.
- `adv_74_no_bq_table`: `ensure_table()` is fail-open but never called in this cycle; BQ table `alt_etf_flows` not yet created. Consistent with immutable criterion (ast-parse only). Create in phase-7.12.
- `adv_74_backend_alt_data_untracked`: `backend/alt_data/` remains untracked (same as 7.1-7.3). Single `git add backend/alt_data/` after owner approval will pick up all four modules at once.
- `adv_74_contract_ratelimit_algebra`: Contract writes `_RATE_INTERVAL_S = 1.0 / 0.5`; code uses `2.0`. Equivalent; note for future readability.
- `adv_74_researcher_overreach`: Researcher prematurely wrote production code during research-gate. Main remediated by delete + re-own. Future: researcher prompt should be reinforced to stop at brief + design proposal (already flagged in experiment_results).

## Immutable criteria

- `python -c "import ast; ast.parse(open('backend/alt_data/etf_flows.py').read())"` -> PASS (exit 0, `SYNTAX OK`).

## Decision

**PASS.** Status flip 7.4 pending -> done is approved. Append `harness_log.md` cycle block BEFORE flipping masterplan.json.

---

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_74_v1",
  "reason": "Sole immutable criterion (ast.parse) PASS. 5/5 protocol audit PASS. 6/6 deterministic checks PASS. Scaffold completeness, DDL shape, starter tickers (24), rate-limit (2.0s), and Main's 'researcher overreach, re-owned GENERATE' disclosure all corroborated. Regression 152/1. Mtime ordering confirms contract-before-code.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "protocol_audit_5_items",
    "syntax_ast_parse",
    "dry_run_cli",
    "pytest_regression",
    "ascii_decode",
    "scope_git_status",
    "mtime_ordering",
    "llm_scaffold_completeness",
    "llm_ddl_house_pattern",
    "llm_starter_tickers",
    "llm_rate_limit",
    "llm_protocol_disclosure_credibility",
    "llm_scope_honesty",
    "llm_compliance_alignment"
  ]
}
```
