# Q/A Evaluator Critique — phase-7 / 7.10 (Hiring scaffold)

**Evaluator:** qa_710_v1  **Date:** 2026-04-20

## 5-item protocol audit

1. **Researcher:** `phase-7.10-research-brief.md` present, `gate_passed: true`, 6 sources in full, 11 URLs, three-variant queries, recency scan. PASS.
2. **Contract pre-commit:** contract mtime 1776637814 < results mtime 1776637888. PASS.
3. **Experiment results:** verbatim outputs for ast.parse, dry-run CLI, ASCII decode, regression. PASS.
4. **Log-last:** last cycle block is phase-7.7; no 7.10 entry yet. PASS.
5. **No verdict-shopping:** first Q/A on 7.10. PASS.

## Deterministic A–E

- A. `ast.parse(hiring.py)` → SYNTAX OK (re-run). PASS.
- B. `python -m backend.alt_data.hiring --dry-run` → `{scaffold_only: true, ingested: 0}`. PASS.
- C. Regression 152 passed / 1 skipped (attested from results). PASS.
- D. ASCII decode of `hiring.py`. PASS.
- E. Scope: only new module + handoff trio; no top-level `import requests` or `os.environ[...]`. PASS.

## LLM judgment

- Scaffold discipline honored: `LINKUP_API_KEY` referenced only inside `fetch_postings` docstring (line 72 comment: "read INSIDE this function, never at import"). No top-level auth imports.
- DDL has exactly 12 columns matching the brief: `posting_id, as_of_date, ticker, company_name, title, department, location, posted_at, last_seen_at, is_active, source, raw_payload`. Partition `as_of_date`, cluster `ticker, department`.
- `is_active` derived at `normalize` lines 95–102 via 7-day recency window on `last_seen_at` (matches brief pitfall note).
- `_posting_id = sha256(f"{ticker}|{title}|{posted_at}")[:24]` — deterministic 24-char surrogate key.
- `_STARTER_COMPANIES` has exactly 5 tickers (AAPL, MSFT, NVDA, AMZN, GOOGL).
- Compliance docstring references `alt-data.md` row 7.10 + LinkUp MSA + "not LinkedIn scrape".
- Mutation resistance N/A at scaffold tier — `fetch_postings` returns `[]`, no data path to mutate.

## violated_criteria / violation_details

[]

## checks_run

`["harness_audit_5items", "syntax", "dry_run_cli", "ascii_decode", "scope_grep_no_auth_imports", "contract_vs_results_mtime", "research_gate_envelope", "log_last_ordering", "first_qa_dedup", "ddl_column_count", "is_active_derivation", "posting_id_deterministic", "starter_companies_count", "compliance_docstring"]`

## Decision

**PASS — qa_710_v1**
