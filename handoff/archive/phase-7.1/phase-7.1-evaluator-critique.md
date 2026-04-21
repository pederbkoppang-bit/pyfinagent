# Q/A Critique -- phase-7 / 7.1 Congressional trades ingestion

**Cycle:** 1 **Reviewer:** qa (merged qa-evaluator + harness-verifier) **Verdict tag:** qa_71_v1
**Date:** 2026-04-19 UTC

---

## 5-item protocol audit

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Researcher spawn proof -- brief present, gate_passed, three-variant, recency, 5+ sources in full | PASS | `handoff/current/phase-7.1-research-brief.md` present; `"external_sources_read_in_full": 6`, `"gate_passed": true`. |
| 2 | Contract PRE-commit (contract mtime < experiment_results mtime) | PASS | contract `Apr 19 22:28:31`, results `Apr 19 22:32:42` (4m11s gap, correct order). Research brief `22:27:26` precedes contract (correct). |
| 3 | Experiment results present -- verbatim immutable output + regression + source-URL disclosure | PASS | Both immutable commands quoted verbatim with output; regression `152 passed, 1 skipped` quoted; dedicated "Source-URL discovery finding" section documents S3 -> GitHub-raw pivot. |
| 4 | Log-last -- last harness_log cycle is 7.0 (21:55 UTC), not 7.1 | PASS | `tail -40 harness_log.md` final cycle block = `phase=7.0 result=PASS` at 21:55 UTC. No 7.1 entry yet. |
| 5 | No verdict-shopping -- this is cycle 1, no prior Q/A to overturn | PASS | First Q/A of 7.1. |

All 5 audit items PASS.

---

## Deterministic checks (A-I)

### A. Immutable (A) -- ast.parse

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/alt_data/congress.py').read()); print('SYNTAX OK')"
SYNTAX OK
EXIT=0
```
PASS.

### B. Immutable (B) -- bq COUNT > 100

```
$ bq query --use_legacy_sql=false --project_id=sunny-might-477607-p8 --format=csv \
    'SELECT COUNT(*) FROM pyfinagent_data.alt_congress_trades WHERE as_of_date >= CURRENT_DATE() - 30'
f0_
7262

$ ... | tail -n 1 | awk '{ exit ($1 > 100 ? 0 : 1) }'
AWK_EXIT=0
```
PASS. COUNT=7262 (matches experiment_results), awk exit 0 (>100 satisfied ~72x).

### C. Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped, 1 warning in 13.69s
```
PASS. Unchanged green baseline (matches 7.0 reference).

### D. File existence

- `backend/alt_data/__init__.py` -- present (87 bytes).
- `backend/alt_data/congress.py` -- present (12,311 bytes, ~310 lines).
- `handoff/current/phase-7.1-{research-brief,contract,experiment-results}.md` -- all three present.

PASS.

### E. Scope (`git status --short`)

New (A/??) files within scope:
- `backend/alt_data/__init__.py` (new, untracked)
- `backend/alt_data/congress.py` (new, untracked)
- `handoff/current/phase-7.1-*.md` (3 files, untracked)

No production module modifications introduced by this cycle. The large existing ` M` set (multi_agent_orchestrator, MCP servers, etc.) is pre-existing uncommitted work from earlier phases, not from 7.1. PASS (scope honest).

### F. BQ table shape matches contract

`bq show --schema` returns 13 columns in the contract-specified order:

| # | name | type | mode |
|---|---|---|---|
| 1 | disclosure_id | STRING | REQUIRED |
| 2 | as_of_date | DATE | REQUIRED |
| 3 | senator_or_rep | STRING | NULLABLE |
| 4 | party | STRING | NULLABLE |
| 5 | chamber | STRING | NULLABLE |
| 6 | transaction_type | STRING | NULLABLE |
| 7 | ticker | STRING | NULLABLE |
| 8 | amount_min | FLOAT | NULLABLE |
| 9 | amount_max | FLOAT | NULLABLE |
| 10 | transaction_date | DATE | NULLABLE |
| 11 | disclosure_date | DATE | NULLABLE |
| 12 | source | STRING | NULLABLE |
| 13 | raw_payload | JSON | NULLABLE |

Matches contract's 13-column shape and types exactly. PASS.

### G. Idempotent table creation

Code at `congress.py:240` uses `_CREATE_TABLE_SQL.format(...)` which (per grep + read) is a `CREATE TABLE IF NOT EXISTS` DDL wrapped in try/except fail-open. `bq show` confirms the table exists with matching shape. Re-invocation of `ensure_table()` would be a no-op on the existing table (IF NOT EXISTS semantics). PASS.

### H. ASCII discipline

```
$ python -c "open('backend/alt_data/congress.py','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK
```
PASS.

### I. Source-URL disclosure honest (re-curl)

```
S3_HOUSE  = 403  (matches disclosure: S3 bucket ACL-dead)
GH_SENATE = 200  (matches disclosure: GitHub raw mirror live)
```
PASS. The mid-cycle S3 -> GitHub-raw pivot claimed in experiment_results is independently reproducible. This is a real anti-rubber-stamp artifact (Main caught and documented a broken upstream mid-cycle).

---

## LLM judgment

### Data-source compliance (GitHub raw URL, STOCK Act public data)

ACCEPT. `raw.githubusercontent.com` does not present an OAuth click-through or ToS modal; it serves static content under GitHub's public-repo Terms which permit unauthenticated raw content access. The underlying data is STOCK-Act-mandated (2 U.S.C. Sec. 13101-13106) public disclosure. Compliance doc row 7.1 lists "Government-public + HTTP scraper" as the policy profile; pulling a community mirror of STOCK-Act public data satisfies both the policy's spirit (no contract formation, no personal-data handling beyond what filers voluntarily disclose) and the carry-forward advisory from qa_70_v1 (OAuth-ToS column N/A here). Experiment_results caveat #5 correctly draws this line. No pushback.

### House-missing partial-delivery verdict

ACCEPT as NON-blocking, with explicit follow-up recorded.

Reasoning:
- The immutable criterion is `COUNT(*) > 100`. It is satisfied 72x over (7,262 vs 100). The step as contractually defined is met.
- Main disclosed the gap IN BOLD ("House data is NOT yet ingested") rather than burying it -- scope honesty discipline honored.
- Compliance doc row 7.1 does list both chambers, but treating "Senate-only row count met, House deferred" as CONDITIONAL would be moving the goalposts (the immutable criterion was intentionally chamber-agnostic). The research gate's own recommendation acknowledged the two-URL pull and House's S3 source may drift -- R1 in the risk register predicted exactly this outcome.
- The deferral is technically defensible: `disclosures-clerk.house.gov` PDF/XML parsing is non-trivial and belongs in phase-7.11 shared-infra per the contract's explicit out-of-scope list.

However: the masterplan step description should carry an explicit House-ingest follow-up. Recommend adding a phase-7.1.1 (or tagging phase-7.11 shared-infra) with the PDF/ZIP parser as a hard dependency before phase-7.12 signal extraction. This is a NON-blocking recommendation for qa_71_v1.

### Same-day re-run duplication verdict

ACCEPT as NON-blocking, with a MINOR docstring-accuracy concern.

Reasoning:
- Confirmed by code read: `upsert_trades` at `congress.py:249-277` calls `client.insert_rows_json(table_ref, rows)` -- a streaming insert, not a MERGE DML. Same-day re-runs will produce duplicate `(disclosure_id, as_of_date)` pairs.
- The in-code comment at lines 262-267 transparently acknowledges this ("BQ streaming has no native UPSERT; we rely on read-side dedup"). Experiment_results caveat #2 restates it publicly.
- Read-side dedup via `SELECT ... QUALIFY ROW_NUMBER() OVER (PARTITION BY disclosure_id ORDER BY as_of_date DESC) = 1` or equivalent is trivial and consistent with phase-6.5.7 append-only design.
- Criterion B cares about rows existing, not about uniqueness, so duplication does not threaten the contract.

MINOR issue (not blocking): the function docstring at line 255 says `"""MERGE rows into alt_congress_trades on disclosure_id. Fail-open, returns count upserted."""` -- this is a lie-by-docstring. The implementation does not MERGE. Recommend changing the docstring to match the comment block below it ("Stream-insert rows; read-side dedup by (disclosure_id, MAX(as_of_date)). No MERGE yet."). Recording as advisory `adv_71_docstring_merge`, NOT a blocker, because the truthful disclosure exists in the code comment two lines later and in experiment_results caveat #2.

### Anti-rubber-stamp test

PASS. The S3 403 -> GitHub-raw pivot is a real upstream fault caught mid-cycle, independently verified by my re-curl (S3_HOUSE=403, GH_SENATE=200). This is not a rehearsed compliance artifact; it's a genuine "the contract's recommended URL was dead and the work adapted" event. Research-gate brief recommended S3; Main found it broken; Main pivoted and disclosed. Good harness hygiene.

### Scope honesty

PASS. `git status --short` new files are confined to `backend/alt_data/*` + handoff. The pre-existing ` M` set is phase-6 carry and not attributable to 7.1.

### Research-gate compliance

PASS. Contract references research-brief at line 9 with summary (6 in full, 16 URLs, three-variant, recency). Brief JSON envelope confirms `gate_passed: true`.

---

## violated_criteria

```json
[]
```

## violation_details

```json
[]
```

## Advisories (non-blocking carry-forward)

- `adv_71_docstring_merge` -- `upsert_trades` docstring claims MERGE; implementation is stream-insert. Fix the docstring in a future cycle (phase-7.11 shared-infra or a 7.1 cleanup).
- `adv_71_house_ingest_followup` -- House chamber data not ingested. Add explicit follow-up step (phase-7.1.1 or phase-7.11 scope expansion) to parse `disclosures-clerk.house.gov` PDF/XML ZIP archive before phase-7.12 signal extraction closes.

## checks_run

```json
["syntax", "verification_command", "bq_count_threshold", "regression", "file_existence", "git_scope", "bq_schema_shape", "ascii_discipline", "source_url_recurl", "researcher_gate_passed", "contract_mtime_ordering", "log_last_ordering", "code_read_upsert_trades", "compliance_doc_alignment"]
```

## Certified fallback

`false` (cycle 1; retry_count = 0; no fallback needed).

---

## Final Decision: **PASS** (qa_71_v1)

All immutable criteria numerically met. All 5 protocol audit items pass. All 9 deterministic checks pass. Two non-blocking advisories recorded for future cleanup. Mid-cycle S3 outage handled honestly. No rubber-stamp.

Main may proceed to:
1. Append `harness_log.md` cycle block for phase-7.1 (log-last discipline).
2. Flip masterplan 7.1 `pending` -> `done`.
3. Record advisories `adv_71_docstring_merge` and `adv_71_house_ingest_followup` in the log cycle block so they don't get lost.
