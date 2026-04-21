# Q/A Critique — phase-6.5 / step 6.5.1 (BigQuery intel schema migration)

**Evaluator id:** qa_651_v1
**Date:** 2026-04-19
**Cycle:** 1
**Step:** phase-6.5.1 — first executable step under Path D

---

## 5-item protocol audit

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Researcher spawn proof | PASS | `handoff/current/phase-6.5.1-research-brief.md` present; JSON envelope shows `external_sources_read_in_full: 6` (≥5), `recency_scan_performed: true`, `urls_collected: 16` (≥10), three-variant queries listed (2026 frontier, 2025 window, year-less canonical, plus a 4th dedup-canonical query), `gate_passed: true`. Rolling `research_brief.md` ignored per instructions. |
| 2 | Contract PRE-commit | PASS | `stat` mtimes: contract=1776624841 < experiment-results=1776624969. Contract written before generate. Research brief mtime=1776623541 (earlier still), confirming research→contract→generate ordering. |
| 3 | Experiment results present | PASS | `phase-6.5.1-experiment-results.md` contains verbatim chain output (all 5 DDL banners + "dry-run: no BigQuery writes executed." + "8 passed in 0.02s" + `CHAIN_EXIT=0`), regression output (111 passed, 1 skipped), file list (2 created, 0 modified), criterion table (3/3 PASS). |
| 4 | Log-last discipline | PASS | Tail of `handoff/harness_log.md` is the `phase=6.5-decision result=PASS` block (20:52 UTC, 2026-04-19). No 6.5.1 block yet, so status has not been flipped ahead of the log. |
| 5 | No verdict-shopping | PASS | First Q/A on 6.5.1 (qa_651_v1); no prior critique file exists for this step. |

All 5 audit items PASS.

---

## Deterministic checks (A–H)

### A. Immutable command exit code

```
$ source .venv/bin/activate && python scripts/migrations/phase_6_5_intel_schema.py --dry-run && pytest backend/tests/test_intel_schema.py -q
== intel_sources (dry-run) ==
CREATE TABLE IF NOT EXISTS `sunny-might-477607-p8.pyfinagent_data.intel_sources` (
  source_id STRING NOT NULL,
  source_name STRING NOT NULL,
  source_type STRING NOT NULL,
  kill_switch BOOL NOT NULL,
  rate_limit_per_day INT64,
  last_scanned_at TIMESTAMP,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP,
  metadata JSON
)
PARTITION BY DATE(created_at)
CLUSTER BY source_type, source_name
OPTIONS (
  description = "phase-6.5.1 intel source registry + kill-switch"
)
... (4 more DDL banners: intel_documents, intel_chunks, intel_novelty_scores, intel_prompt_patches — all with CREATE TABLE IF NOT EXISTS, PARTITION BY DATE(...), CLUSTER BY, OPTIONS (description)) ...
dry-run: no BigQuery writes executed.
........                                                                 [100%]
8 passed in 0.02s
CHAIN_EXIT=0
```

**Verdict:** PASS. Exit 0. 5 DDL banners printed. All 8 tests green. Chain matches masterplan verification.command verbatim.

### B. Full regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
........................................................................ [ 64%]
....................s...................                                 [100%]
111 passed, 1 skipped, 1 warning in 6.59s
```

**Verdict:** PASS. Baseline 103 + 8 new = 111. Zero regressions.

### C. File existence

`scripts/migrations/phase_6_5_intel_schema.py` (192 lines) and `backend/tests/test_intel_schema.py` (149 lines) both present. All handoff artifacts phase-scoped (`phase-6.5.1-*.md`); rolling files were not clobbered.

**Verdict:** PASS.

### D. Scope

`git status --short` shows only two new files that belong to 6.5.1:
- `?? scripts/migrations/phase_6_5_intel_schema.py`
- `?? backend/tests/test_intel_schema.py`

Plus the 6.5.1 handoff trio. The bulk of `M` entries in `git status` (researcher.md, mas-architecture.md, agent_definitions.py, etc.) are pre-existing long-lived modifications not introduced by this step — they were already modified before 6.5.1 began and are out of scope for this step's scope check.

**Verdict:** PASS. Scope matches contract ("Created: 2 files. Modified: none. No production backend module was touched.").

### E. Criterion alignment

| Criterion | Interpretation | Verdict |
|---|---|---|
| `migration_dry_run_exit_0` | Dry-run `rc==0` and stdout contains "dry-run: no BigQuery writes executed." | PASS (stdout confirmed above) |
| `all_intel_tables_defined_in_script` | All 5 tables {intel_sources, intel_documents, intel_chunks, intel_novelty_scores, intel_prompt_patches} have `CREATE TABLE IF NOT EXISTS` blocks AND every column documented in the contract's "Tables" section is present in the DDL | PASS — I diffed contract § "Tables" against the 5 DDL constants in the migration file: intel_sources 9/9 columns, intel_documents 14/14, intel_chunks 8/8, intel_novelty_scores 9/9, intel_prompt_patches 11/11. All 51 columns match. |
| `schema_test_green` | `pytest backend/tests/test_intel_schema.py -q` exits 0 | PASS (8 passed / 0 failed) |

**Verdict:** PASS on all three immutable criteria.

### F. Idempotency discipline

```
$ python scripts/migrations/phase_6_5_intel_schema.py --dry-run > /dev/null && python scripts/migrations/phase_6_5_intel_schema.py --dry-run > /dev/null
SECOND_RUN_EXIT=0
```

Every DDL contains `CREATE TABLE IF NOT EXISTS` (verified by `test_each_ddl_is_idempotent`). Second sequential run still exits 0 — the script itself is re-runnable, not just the DDLs.

**Verdict:** PASS.

### G. Lazy-import discipline

Read `scripts/migrations/phase_6_5_intel_schema.py:154-184`:
- `main(dry_run)` at line 154
- Dry-run early-exit at line 173-175: `if dry_run: print(...); return 0`
- `from google.cloud import bigquery` at line 177 — AFTER the early-exit

Test `test_dry_run_returns_zero_without_bq_import` (line 120-131) pops `google.cloud.bigquery` from `sys.modules`, calls `main(dry_run=True)`, then asserts `"google.cloud.bigquery" not in sys.modules`. This test passes in suite run A.

**Verdict:** PASS. Dry-run path has zero BQ dependency; asserted both structurally (code inspection) and dynamically (test).

### H. Security-rule compliance

Migration file ASCII check:
```
$ python -c "open('scripts/migrations/phase_6_5_intel_schema.py','rb').read().decode('ascii')"
MIG_ASCII_OK len= 5467
```
Migration file is pure ASCII. PASS.

Test file ASCII check:
```
UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 2375
line 63: '    # by PARTITION / CLUSTER / OPTIONS / end-of-DDL — none of which are inside'
```
One em-dash (U+2014) in a Python COMMENT on line 63 of `test_intel_schema.py`. `.claude/rules/security.md` scopes the ASCII rule to **`logger.*()` calls** (runtime messages that hit Windows cp1252 uvicorn handlers). A comment in a test file never reaches a logger, and the test file parses + imports cleanly (pytest green). So this is NOT a security-rule violation.

However, `experiment_results.md` §"Known caveats" item 4 asserts "No stray em-dashes / arrows in the migration file" — that claim is accurate for the migration file. It does NOT claim the test file is em-dash-free, so no misrepresentation.

**Verdict:** PASS on the scoped security rule. NON-BLOCKING observation: em-dash in test-file comment is cosmetic inconsistency with the spirit of the ASCII-only discipline. Easy to polish by replacing with `--` in a future cleanup, but not blocking this cycle.

---

## LLM judgment

**Schema sanity.** The five tables are the right five for the reduced-scope Path D intel pipeline: a source registry (governance/kill-switch), an append-only document fact table, a chunk table with inline embeddings for VECTOR_SEARCH, a re-scorable novelty enrichment, and a prompt-patch queue. Contract § "Tables" lists 51 columns total; DDL diff confirms all 51 are declared. Two specific checks per the spawn prompt: (a) `intel_prompt_patches` DOES include `reviewed_by STRING` (line 133 of migration) — PASS, the reviewer-audit column is present. (b) `updated_at TIMESTAMP` is on `intel_sources` (line 48) but NOT on `intel_documents`, `intel_chunks`, `intel_novelty_scores`, or `intel_prompt_patches` — this is intentional given the append-only / re-scorable design (documents and chunks don't mutate; novelty scores are versioned by `scorer_model`+`scorer_version`; prompt-patches have `reviewed_at` + `applied_at` in lieu of a generic `updated_at`). Acceptable. No `ingested_by` column anywhere — minor miss if you want row-level provenance of the cron worker identity, but the contract never promised it and `raw_payload JSON` can carry that metadata. Non-blocking.

**Partition+cluster sanity.** `intel_chunks` PARTITION BY DATE(ingested_at) CLUSTER BY (doc_id, chunk_index) — dominant query pattern is "fetch all chunks for doc X in order"; clustering on doc_id prunes to the right micro-partitions and chunk_index gives ordered scan. Correct. `intel_novelty_scores` PARTITION BY DATE(scored_at) CLUSTER BY (chunk_id, scorer_model) — re-scoring query pattern is "latest score for chunk X by model Y" or "all scores for chunk X" — both benefit from chunk_id as the first cluster key. Correct. `intel_prompt_patches` clustered on (status, patch_type) is right for the "show me all pending strategy_hint patches" queue-query workload. All 5 tables use a time-column DATE() partition (idiomatic for BigQuery's time-unit partitioning per the GCP doc in the research brief). No anti-pattern spotted.

**Anti-rubber-stamp.** Experiment_results.md § "Known caveats" item 3 transparently discloses a first-draft extractor bug: initial `_extract_columns` used `ddl.rfind(")")` which wrapped the `OPTIONS (description=...)` block, leaking `description` into the column set and failing `test_each_ddl_has_expected_columns` on `intel_sources`. The fix — walk paren-depth, exit on first balanced close — is visible in `test_intel_schema.py:65-76`. I verified independently: the extractor increments depth on `(`, decrements on `)`, and breaks when `depth==0`. This is the documented fix and it correctly handles nested parens (e.g. `ARRAY<STRING>` isn't a paren, but `OPTIONS (...)` and the main column list are). The disclosure of a caught-mid-cycle bug is the OPPOSITE of rubber-stamping — it's the rigor signal this evaluator looks for. Strong positive.

**Scope honesty.** Contract committed: "Created: 2 files. Modified: none. No production backend module was touched." Verified: `git status --short` shows the two new files (migration + test) plus handoff trio; no `M` entry under `backend/` introduced by this step (the pre-existing modifications listed in `git status` all predate 6.5.1). Scope statement matches reality.

**Weakest-link (non-blocking, for future cycles).** The biggest risk on future live BQ migration is NOT the schema design — it's the config chain: `settings.gcp_project_id` + `settings.bq_dataset_observability` fallback to `"pyfinagent_data"` (line 158-160 of the migration). If `bq_dataset_observability` is unset on the live runner AND the runner's `gcp_project_id` doesn't match `sunny-might-477607-p8`, the migration would silently create tables in the wrong project/dataset. Second-tier risk: JSON column compatibility across BQ client versions — the `JSON` type requires a reasonably recent `google-cloud-bigquery` (≥3.4). Third-tier: `ARRAY<FLOAT64>` embedding column size limits (BQ row size cap at 100MB; 1536-dim embeddings are fine, but 32k-dim ones would blow up — not a concern for current models). Flag to 6.5.9 smoketest: verify project+dataset before executing. Non-blocking for 6.5.1 because live execution is explicitly out-of-scope per the contract.

---

## Summary

```json
{
  "violated_criteria": [],
  "violation_details": [],
  "checks_run": [
    "audit_1_researcher_spawn_proof",
    "audit_2_contract_pre_commit_mtime",
    "audit_3_experiment_results_verbatim",
    "audit_4_log_last_discipline",
    "audit_5_no_verdict_shopping",
    "A_immutable_command_exit_0",
    "B_full_regression_no_deltas",
    "C_file_existence",
    "D_scope_matches_contract",
    "E_criterion_alignment",
    "F_idempotency_double_run",
    "G_lazy_import_structural_and_dynamic",
    "H_ascii_security_rule",
    "llm_schema_sanity",
    "llm_partition_cluster_sanity",
    "llm_anti_rubber_stamp_extractor_disclosure",
    "llm_scope_honesty_git_status",
    "llm_weakest_link_nonblocking"
  ]
}
```

**Non-blocking observations** (documented for future cleanup, NOT cause for CONDITIONAL):
1. Em-dash in `test_intel_schema.py:63` comment. Security rule scopes ASCII to logger calls, so not a violation; polish to `--` in a later cycle.
2. No `ingested_by` row-level-provenance column anywhere. Can be carried in `raw_payload JSON`; contract never promised it.
3. Live-run config-chain risk for `bq_dataset_observability` fallback — flag to the 6.5.9 smoketest contract to verify project+dataset before executing.

---

## Final Decision

**qa_651_v1: PASS**

All 3 immutable criteria met. All 5 protocol audit items PASS. All 8 deterministic checks A–H PASS. LLM judgment positive on schema sanity, partition+cluster sanity, anti-rubber-stamp rigor, and scope honesty. Ready to append the `harness_log.md` cycle block and flip `phase-6.5.1.status: pending → done`.
